#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract neuron activations from ResNet-18 for correlation and persistent homology.

Tap modes:
  - topotroj_compat : inputs (f_in) to Conv2d (+ optional final Linear 'fc'), GAP over HxW
  - toap_block_out  : BasicBlock outputs (post-add, post-ReLU), GAP over HxW [RECOMMENDED]
  - bn2_legacy      : outputs of bn2 modules (pre-add, pre-ReLU), GAP over HxW
  - spatial_preserve: inputs (f_in) to Conv2d (+ optional final Linear 'fc'), PRESERVE spatial structure

Outputs:
  - One .pt per tapped layer: float32 tensor of shape [num_samples, num_channels] (GAP modes) or [num_samples, num_channels, height, width] (spatial_preserve)
  - JSON metadata describing the run
  - layer_catalog.csv mapping (layer_name, n_neurons) for stratified sampling
  - triggered_input_metadata.csv (only when input_type='triggered', collected safely with num_workers>0)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import gc
import psutil
import warnings
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import random
import time
import cProfile
import pstats
from io import StringIO

# You provide this
from model import get_model

warnings.filterwarnings('ignore')

# -----------------------------
# Logging / Utils
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("activation_extraction_improved.log"),
                  logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_memory_usage_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Timing and Complexity Utils
# -----------------------------
class TimingAndComplexityTracker:
    """Track execution time and computational complexity for different operations."""
    
    def __init__(self):
        self.timings = {}
        self.complexities = {}
        self.operation_counts = {}
    
    def start_timing(self, operation_name: str):
        """Start timing an operation."""
        self.timings[operation_name] = {'start': time.time(), 'end': None, 'duration': None}
    
    def end_timing(self, operation_name: str):
        """End timing an operation and calculate duration."""
        if operation_name in self.timings:
            self.timings[operation_name]['end'] = time.time()
            self.timings[operation_name]['duration'] = (
                self.timings[operation_name]['end'] - self.timings[operation_name]['start']
            )
            return self.timings[operation_name]['duration']
        return None
    
    def add_complexity(self, operation_name: str, num_nodes: int, num_edges: int, 
                      time_complexity: str, space_complexity: str, 
                      actual_operations: int = None):
        """Add complexity information for an operation."""
        self.complexities[operation_name] = {
            'num_nodes': int(num_nodes) if num_nodes is not None else 0,
            'num_edges': int(num_edges) if num_edges is not None else 0,
            'time_complexity': str(time_complexity),
            'space_complexity': str(space_complexity),
            'actual_operations': int(actual_operations) if actual_operations is not None else 0,
            'complexity_per_node': float(actual_operations / num_nodes) if num_nodes > 0 and actual_operations else 0.0,
            'complexity_per_edge': float(actual_operations / num_edges) if num_edges > 0 and actual_operations else 0.0
        }
    
    def get_summary(self):
        """Get a summary of all timings and complexities."""
        summary = {
            'timings': {},
            'complexities': {},
            'total_time': 0
        }
        
        for op_name, timing_data in self.timings.items():
            if timing_data['duration'] is not None:
                summary['timings'][op_name] = {
                    'duration_seconds': float(timing_data['duration']),
                    'duration_formatted': f"{timing_data['duration']:.4f}s"
                }
                summary['total_time'] += float(timing_data['duration'])
        
        summary['complexities'] = self.complexities.copy()
        summary['total_time_formatted'] = f"{summary['total_time']:.4f}s"
        
        return summary
    
    def log_summary(self, logger):
        """Log the timing and complexity summary."""
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info("TIMING AND COMPLEXITY SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total execution time: {summary['total_time_formatted']}")
        logger.info("")
        
        logger.info("Individual operation timings:")
        for op_name, timing_data in summary['timings'].items():
            logger.info(f"  {op_name}: {timing_data['duration_formatted']}")
        
        logger.info("")
        logger.info("Complexity analysis:")
        for op_name, complexity_data in summary['complexities'].items():
            logger.info(f"  {op_name}:")
            logger.info(f"    Nodes: {complexity_data['num_nodes']}")
            logger.info(f"    Edges: {complexity_data['num_edges']}")
            logger.info(f"    Time complexity: {complexity_data['time_complexity']}")
            logger.info(f"    Space complexity: {complexity_data['space_complexity']}")
            if complexity_data['actual_operations']:
                logger.info(f"    Actual operations: {complexity_data['actual_operations']}")
                logger.info(f"    Operations per node: {complexity_data['complexity_per_node']:.2f}")
                logger.info(f"    Operations per edge: {complexity_data['complexity_per_edge']:.2f}")
        
        logger.info("=" * 60)
    
    def log_quick_summary(self, logger):
        """Log a concise summary with just the key metrics."""
        summary = self.get_summary()
        
        logger.info("")
        logger.info("🚀 QUICK PERFORMANCE SUMMARY")
        logger.info("─" * 50)
        
        # Total time
        logger.info(f"⏱️  Total Time: {summary['total_time_formatted']}")
        
        # Key operations and their efficiency
        key_ops = ['activation_extraction', 'matrix_building', 'persistence_computation', 'tss_computation']
        
        for op in key_ops:
            if op in summary['complexities']:
                data = summary['complexities'][op]
                time_data = summary['timings'].get(op, {})
                duration = time_data.get('duration_seconds', 0)
                
                if data['num_nodes'] > 0 and data['num_edges'] > 0:
                    logger.info(f"📊 {op.replace('_', ' ').title()}:")
                    logger.info(f"    Time: {duration:.2f}s | ~{data['complexity_per_node']:.1f} ops/node | ~{data['complexity_per_edge']:.1f} ops/edge")
        
        logger.info("─" * 50)

def profile_function(func, *args, **kwargs):
    """Profile a function and return execution stats along with result."""
    pr = cProfile.Profile()
    pr.enable()
    
    result = func(*args, **kwargs)
    
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()
    
    return result, s.getvalue()

# -----------------------------
# Trigger + Dataset
# -----------------------------
def apply_trigger(image: torch.Tensor,
                  trigger_pattern_size: int,
                  trigger_pixel_value: float,
                  trigger_location: str = "br") -> torch.Tensor:
    """
    Apply a square trigger to a single image tensor [C,H,W].
    """
    C, H, W = image.shape
    s = trigger_pattern_size
    if s > min(H, W):
        raise ValueError(f"Trigger size {s} exceeds image dims {H}x{W}")
    if trigger_location == "br":
        y, x = H - s, W - s
    elif trigger_location == "tl":
        y, x = 0, 0
    elif trigger_location == "tr":
        y, x = 0, W - s
    elif trigger_location == "bl":
        y, x = H - s, 0
    else:
        raise ValueError(f"Unknown trigger_location: {trigger_location}")
    image[:, y:y + s, x:x + s] = trigger_pixel_value
    return image

class PoisonedDataset(torch.utils.data.Dataset):
    """
    Wrap a clean dataset to poison a subset with a fixed trigger and target label.
    Returns (img, label, meta) where meta is a small dict; this makes metadata
    collection robust with num_workers > 0.
    """
    def __init__(self,
                 clean_dataset,
                 trigger_args: dict,
                 poison_ratio: float = 1.0,
                 target_label: int = 0):
        self.clean_dataset = clean_dataset
        self.trigger_args = trigger_args
        self.poison_ratio = poison_ratio
        self.target_label = int(target_label)

        n = len(clean_dataset)
        k = int(n * poison_ratio)
        # Deterministic if set_seed() was called before constructing this dataset
        self.poisoned_indices = set(random.sample(range(n), k))
        logger.info(f"PoisonedDataset: poisoning {k}/{n} samples.")

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        img, label = self.clean_dataset[idx]
        is_poisoned = idx in self.poisoned_indices

        meta = {
            "sample_index": int(idx),
            "original_true_label": int(label),
            "is_poisoned": bool(is_poisoned),
        }

        if is_poisoned:
            img = apply_trigger(img.clone(), **self.trigger_args)
            label = self.target_label
            meta["poisoned_target_label"] = int(label)
        else:
            meta["poisoned_target_label"] = -1
        
        # Convert final label to int once at the end
        return img, int(label), meta
    
class CleanWithMeta(torch.utils.data.Dataset):
    """Wrap a clean dataset so we always return (img, label, meta)."""
    def __init__(self, base_ds):
        self.base_ds = base_ds
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        meta = {
            "sample_index": int(idx),
            "original_true_label": int(label),
            "is_poisoned": False,
            "poisoned_target_label": -1
        }
        return img, int(label), meta    

# -----------------------------
# Activation Extractor
# -----------------------------
class ImprovedActivationExtractor:
    """
    Activation extractor compatible with TopoTroj + TOAP.

    tap_mode:
      - 'topotroj_compat' : hook Conv2d (+optional final Linear 'fc') INPUTS (f_in)
      - 'toap_block_out'  : hook BasicBlock OUTPUTS (post-residual, post-ReLU)
      - 'bn2_legacy'      : hook bn2 outputs (your previous setting)
    """

    def __init__(self,
                 model_path: str,
                 output_dir: str = "activation_output_topo",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 random_seed: int = 42,
                 tap_mode: str = "topotroj_compat",
                 include_fc: bool = True,
                 architecture: str = "resnet18"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.random_seed = random_seed
        self.tap_mode = tap_mode
        self.include_fc = include_fc
        self.architecture = architecture

        # Only used by bn2_legacy - will be set dynamically based on architecture
        self.bn2_layer_names = self._get_bn2_layer_names()

        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_catalog: List[Tuple[str, int]] = []  # (layer_name, n_neurons)
        self.triggered_metadata_rows: List[dict] = []   # collected in main process
        self.timing_tracker = TimingAndComplexityTracker()  # Add timing tracker

        logger.info(f"Extractor init | tap_mode={self.tap_mode}, include_fc={self.include_fc}, device={self.device}, architecture={self.architecture}")

    def _get_bn2_layer_names(self) -> List[str]:
        """Get BN2 layer names based on architecture."""
        if self.architecture.startswith('resnet'):
            if self.architecture == 'resnet18':
                return [
                    "layer1.0.bn2", "layer1.1.bn2",
                    "layer2.0.bn2", "layer2.1.bn2",
                    "layer3.0.bn2", "layer3.1.bn2",
                    "layer4.0.bn2", "layer4.1.bn2",
                ]
            elif self.architecture == 'resnet34':
                return [
                    "layer1.0.bn2", "layer1.1.bn2", "layer1.2.bn2",
                    "layer2.0.bn2", "layer2.1.bn2", "layer2.2.bn2", "layer2.3.bn2",
                    "layer3.0.bn2", "layer3.1.bn2", "layer3.3.bn2", "layer3.4.bn2", "layer3.5.bn2",
                    "layer4.0.bn2", "layer4.1.bn2", "layer4.2.bn2",
                ]
            elif self.architecture == 'resnet152':
                # ResNet-152 has more layers in each block
                return [
                    "layer1.0.bn2", "layer1.1.bn2", "layer1.2.bn2",
                    "layer2.0.bn2", "layer2.1.bn2", "layer2.2.bn2", "layer2.3.bn2", "layer2.4.bn2", "layer2.5.bn2", "layer2.6.bn2", "layer2.7.bn2",
                    "layer3.0.bn2", "layer3.1.bn2", "layer3.2.bn2", "layer3.3.bn2", "layer3.4.bn2", "layer3.5.bn2", "layer3.6.bn2", "layer3.7.bn2", "layer3.8.bn2", "layer3.9.bn2", "layer3.10.bn2", "layer3.11.bn2", "layer3.12.bn2", "layer3.13.bn2", "layer3.14.bn2", "layer3.15.bn2", "layer3.16.bn2", "layer3.17.bn2", "layer3.18.bn2", "layer3.19.bn2", "layer3.20.bn2", "layer3.21.bn2", "layer3.22.bn2", "layer3.23.bn2", "layer3.24.bn2", "layer3.25.bn2", "layer3.26.bn2", "layer3.27.bn2", "layer3.28.bn2", "layer3.29.bn2", "layer3.30.bn2", "layer3.31.bn2", "layer3.32.bn2", "layer3.33.bn2", "layer3.34.bn2", "layer3.35.bn2",
                    "layer4.0.bn2", "layer4.1.bn2", "layer4.2.bn2",
                ]
        elif self.architecture in ['mobilenet_v2', 'efficientnet_b0']:
            # MobileNetV2 and EfficientNet-B0 don't have BN2 layers in the same way, return empty list
            return []
        else:
            raise ValueError(f"Unsupported architecture for bn2_legacy mode: {self.architecture}")

    # ---------- Model ----------
    def _detect_architecture_from_state_dict(self, state_dict: dict) -> str:
        """Try to detect architecture from state dict keys."""
        keys = list(state_dict.keys())
        
        # Check for EfficientNet-B0 specific patterns
        if any('features.1.0.block' in key for key in keys) or any('features.0.0.weight' in key for key in keys):
            return 'efficientnet_b0'
        # Check for MobileNetV2 specific patterns (has features.X.conv structure, not features.X.block)
        elif any('features' in key for key in keys) and any('conv' in key for key in keys) and not any('block' in key for key in keys):
            return 'mobilenet_v2'
        # Check for ResNet-152 specific patterns (more layers)
        elif any('layer3.35' in key for key in keys):
            return 'resnet152'
        # Check for ResNet-34 specific patterns (layer3.5 exists but not layer3.35)
        elif any('layer3.5' in key for key in keys) and not any('layer3.35' in key for key in keys):
            return 'resnet34'
        # Check for ResNet-18 specific patterns (has layer structure but not the deeper layers)
        elif any('layer' in key for key in keys):
            return 'resnet18'
        # Default to ResNet-18
        else:
            logger.warning("Could not detect architecture from state dict keys. Defaulting to resnet18.")
            return 'resnet18'

    def load_model(self) -> nn.Module:
        logger.info(f"Loading model from: {self.model_path}")
        
        # Try to detect architecture from state dict if not specified
        detected_arch = self.architecture
        if self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location="cpu")
                if isinstance(ckpt, dict):
                    if "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                    elif "model_state_dict" in ckpt:
                        state_dict = ckpt["model_state_dict"]
                    else:
                        state_dict = ckpt
                else:
                    state_dict = ckpt
                
                detected_arch = self._detect_architecture_from_state_dict(state_dict)
                if detected_arch != self.architecture:
                    logger.info(f"Architecture mismatch detected!")
                    logger.info(f"  Specified: {self.architecture}")
                    logger.info(f"  Detected from state_dict: {detected_arch}")
                    logger.info(f"  Overriding to use detected architecture: {detected_arch}")
                    self.architecture = detected_arch
                    # Update BN2 layer names for the detected architecture
                    self.bn2_layer_names = self._get_bn2_layer_names()
                else:
                    logger.info(f"Architecture matches: {self.architecture}")
            except Exception as e:
                logger.warning(f"Could not detect architecture from state dict: {e}")
        
        model = get_model(num_classes=10, pretrained=False, architecture=self.architecture)

        if self.model_path.exists():
            ckpt = torch.load(self.model_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_k = k[7:] if k.startswith("module.") else k
                new_state_dict[new_k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=True)
                logger.info("Model weights loaded successfully.")
            except RuntimeError as e:
                logger.error(f"Failed to load model weights with strict=True: {e}")
                logger.info("Attempting to load with strict=False...")
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
                    logger.info("Model weights loaded with strict=False. Some layers may not be initialized correctly.")
                except Exception as e2:
                    logger.error(f"Failed to load model weights even with strict=False: {e2}")
                    logger.error("Using randomly initialized weights.")
                    pass
        else:
            logger.warning(f"Model file not found: {self.model_path}. Using randomly initialized weights.")

        model.to(self.device)
        model.eval()
        return model

    # ---------- TopoTroj-like arch parsing ----------
    def parse_arch_topotroj(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """
        Return Conv2d modules (and optional final Linear 'fc') with their names.
        Mirrors TopoTroj's intent of operating on Conv/Linear.
        """
        modules: List[nn.Module] = []
        names: List[str] = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                modules.append(m)
                names.append(name)
            elif self.include_fc and isinstance(m, nn.Linear) and name.endswith("fc"):
                modules.append(m)
                names.append(name)
        return modules, names

    # ---------- Hook registration ----------
    def register_hooks(self, model: nn.Module):
        logger.info(f"Registering hooks (tap_mode={self.tap_mode})")
        self.activations.clear()
        self.layer_catalog.clear()
        self.hooks.clear()

        if self.tap_mode == "topotroj_compat":
            # INPUTS to Conv/Linear (f_in), GAP over HxW
            module_list, module_names = self.parse_arch_topotroj(model)

            def hook_in_as_act(layer_name):
                def _hook(module, f_in, f_out):
                    x = f_in[0] if isinstance(f_in, (tuple, list)) else f_in
                    if x is None:
                        return
                    if x.dim() == 4:  # (N,C,H,W) -> (N,C)
                        x = x.mean(dim=(2, 3))
                    elif x.dim() == 2:  # (N,F)
                        pass
                    else:
                        return
                    self.activations.setdefault(layer_name, []).append(x.detach().cpu().float())
                return _hook

            for m, name in zip(module_list, module_names):
                self.hooks.append(m.register_forward_hook(hook_in_as_act(name)))
                # neuron count for stratification
                if isinstance(m, nn.Conv2d):
                    n_neurons = m.in_channels
                elif isinstance(m, nn.Linear):
                    n_neurons = m.in_features
                else:
                    n_neurons = 0
                self.layer_catalog.append((name, int(n_neurons)))
                logger.info(f"[TopoTroj] Hooked INPUT of {name} (n_neurons={n_neurons})")

        elif self.tap_mode == "toap_block_out":
            # OUTPUTS of BasicBlocks (post-add, post-ReLU), GAP over HxW
            # Only works with ResNet architectures
            if not self.architecture.startswith('resnet'):
                raise ValueError(f"toap_block_out mode only supports ResNet architectures, got: {self.architecture}")
                
            def save_block_out(layer_name):
                def _hook(m, inp, out):
                    y = out if not isinstance(out, (tuple, list)) else out[0]
                    if y.dim() == 4:
                        y = y.mean(dim=(2, 3))
                    elif y.dim() == 2:
                        pass
                    else:
                        return
                    self.activations.setdefault(layer_name, []).append(y.detach().cpu().float())
                return _hook

            # Get the number of blocks per layer based on architecture
            if self.architecture == 'resnet18':
                blocks_per_layer = [2, 2, 2, 2]
            elif self.architecture == 'resnet34':
                blocks_per_layer = [3, 4, 6, 3]
            elif self.architecture == 'resnet152':
                blocks_per_layer = [3, 8, 36, 3]
            else:
                raise ValueError(f"Unsupported ResNet architecture for toap_block_out: {self.architecture}")

            for s in [1, 2, 3, 4]:
                for b in range(blocks_per_layer[s-1]):
                    blk = getattr(model, f"layer{s}")[b]  # torchvision.models.resnet.BasicBlock
                    lname = f"layer{s}.{b}"
                    self.hooks.append(blk.register_forward_hook(save_block_out(lname)))
                    # approx neuron count from conv2 out_channels
                    last_conv = getattr(blk, "conv2", None)
                    n_neurons = last_conv.out_channels if isinstance(last_conv, nn.Conv2d) else 0
                    self.layer_catalog.append((lname, int(n_neurons)))
                    logger.info(f"[TOAP] Hooked BLOCK OUT {lname} (n_neurons={n_neurons})")

        elif self.tap_mode == "bn2_legacy":
            # OUTPUTS of bn2 (your previous selection), GAP over HxW
            # Only works with ResNet architectures
            if not self.architecture.startswith('resnet'):
                raise ValueError(f"bn2_legacy mode only supports ResNet architectures, got: {self.architecture}")
                
            named_modules = dict(model.named_modules())

            def hook_out(name):
                def _hook(module, inp, out):
                    y = out if not isinstance(out, (tuple, list)) else out[0]
                    if y.dim() == 4:
                        y = y.mean(dim=(2, 3))
                    elif y.dim() == 2:
                        pass
                    else:
                        return
                    self.activations.setdefault(name, []).append(y.detach().cpu().float())
                return _hook

            for name in self.bn2_layer_names:
                if name not in named_modules:
                    logger.warning(f"Module not found for bn2_legacy: {name} (skipping)")
                    continue
                m = named_modules[name]
                self.hooks.append(m.register_forward_hook(hook_out(name)))
                n_neurons = getattr(m, "num_features", 0)  # BatchNorm channels
                self.layer_catalog.append((name, int(n_neurons)))
                logger.info(f"[BN2] Hooked OUTPUT of {name} (n_neurons={n_neurons})")

        elif self.tap_mode == "spatial_preserve":
            # INPUTS to Conv/Linear (f_in), PRESERVE spatial structure
            # Works with ALL architectures
            module_list, module_names = self.parse_arch_topotroj(model)

            def hook_in_spatial(layer_name):
                def _hook(module, f_in, f_out):
                    x = f_in[0] if isinstance(f_in, (tuple, list)) else f_in
                    if x is None:
                        return
                    # Preserve spatial structure: (N,C,H,W) -> (N,C,H,W) or (N,F) -> (N,F)
                    if x.dim() == 4:  # Conv2d input: (N,C,H,W)
                        # Keep spatial dimensions intact
                        pass
                    elif x.dim() == 2:  # Linear input: (N,F)
                        # Keep as is
                        pass
                    else:
                        return
                    self.activations.setdefault(layer_name, []).append(x.detach().cpu().float())
                return _hook

            for m, name in zip(module_list, module_names):
                self.hooks.append(m.register_forward_hook(hook_in_spatial(name)))
                # neuron count for stratification (channels for conv, features for linear)
                if isinstance(m, nn.Conv2d):
                    n_neurons = m.in_channels
                elif isinstance(m, nn.Linear):
                    n_neurons = m.in_features
                else:
                    n_neurons = 0
                self.layer_catalog.append((name, int(n_neurons)))
                logger.info(f"[Spatial] Hooked INPUT of {name} (n_neurons={n_neurons}) - PRESERVING spatial structure")

        else:
            raise ValueError(f"Unknown tap_mode: {self.tap_mode}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        logger.info("Removed all forward hooks.")

    # ---------- Dataset ----------
    def get_dataset(self,
                dataset_name: str = "cifar10",
                data_dir: str = "./data",
                input_type: str = "clean",
                trigger_args: Optional[Dict] = None,
                poison_target_label: int = 0):
        logger.info(f"Loading dataset: {dataset_name}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])
        if dataset_name.lower() == "cifar10":
            base = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
            num_classes = 10
        elif dataset_name.lower() == "cifar100":
            base = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
            num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if input_type == "triggered":
            if not trigger_args:
                raise ValueError("trigger_args required for input_type='triggered'")
            ds = PoisonedDataset(base, trigger_args,
                                poison_ratio=1.0,
                                target_label=poison_target_label)
        else:
            ds = CleanWithMeta(base)

        logger.info(f"Dataset ready: {len(ds)} samples | classes={num_classes} | type={input_type}")
        return ds, num_classes

    # ---------- Extraction ----------
    def extract_activations(self,
                            model: nn.Module,
                            dataloader: torch.utils.data.DataLoader) -> Dict[str, np.ndarray]:
        logger.info("Extracting activations...")
        logger.info(f"Initial memory: {get_memory_usage_gb():.2f} GB")

        # Start timing for activation extraction
        self.timing_tracker.start_timing("activation_extraction")
        
        self.activations.clear()
        self.triggered_metadata_rows.clear()
        processed = 0
        total_forward_passes = 0
        total_samples = len(dataloader.dataset)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batches", unit="batch")):
                # Unpack (img, label, meta) since we unified both datasets
                data, targets, meta_batch = batch

                # Move data to device
                data = data.to(self.device, non_blocking=True)

                # Forward pass (triggers hooks)
                _ = model(data)
                total_forward_passes += 1

                # Collect metadata (main process, robust with num_workers>0)
                # meta_batch is a dict of tensors (default collate)
                keys = list(meta_batch.keys())
                B = meta_batch[keys[0]].shape[0]
                for i in range(B):
                    row = {k: (meta_batch[k][i].item() if torch.is_tensor(meta_batch[k]) else meta_batch[k][i])
                           for k in keys}
                    if row.get("is_poisoned", False):
                        self.triggered_metadata_rows.append(row)

                processed += data.size(0)
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  processed={processed} | mem={get_memory_usage_gb():.2f} GB")

                # free batch tensors
                del data, targets, meta_batch, _
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate across batches
        self.timing_tracker.start_timing("activation_concatenation")
        logger.info("Concatenating per-layer activations...")
        final_acts: Dict[str, np.ndarray] = {}
        total_activations = 0
        for lname, chunks in self.activations.items():
            if not chunks:
                logger.warning(f"No activations captured for {lname}")
                continue
            X = torch.cat(chunks, dim=0)
            final_acts[lname] = X.numpy().astype(np.float32)
            total_activations += final_acts[lname].size
            logger.info(f"  {lname}: {final_acts[lname].shape}")
        
        self.timing_tracker.end_timing("activation_concatenation")

        # End timing for activation extraction and add complexity information
        activation_time = self.timing_tracker.end_timing("activation_extraction")
        
        # Calculate approximate number of nodes and edges for complexity analysis
        num_layers = len(final_acts)
        total_neurons = sum(np.prod(arr.shape[1:]) for arr in final_acts.values())
        # For neural networks, edges are roughly connections between layers
        approximate_edges = total_neurons * num_layers if num_layers > 1 else total_neurons
        
        self.timing_tracker.add_complexity(
            "activation_extraction",
            num_nodes=total_neurons,
            num_edges=approximate_edges,
            time_complexity="O(N * L * F)",  # N samples, L layers, F features per layer
            space_complexity="O(N * L * F)",
            actual_operations=total_forward_passes * total_neurons
        )
        
        logger.info(f"Activation extraction completed in {activation_time:.4f}s")
        logger.info(f"Total neurons processed: {total_neurons}")
        logger.info(f"Forward passes: {total_forward_passes}")
        logger.info(f"Final memory: {get_memory_usage_gb():.2f} GB")
        return final_acts

    # ---------- Save ----------
    def save_outputs(self,
                     activations: Dict[str, np.ndarray],
                     model_type: str,
                     input_type: str,
                     output_prefix: str):
        logger.info(f"Saving outputs for {model_type} / {input_type} ...")
        
        # Start timing for saving outputs
        self.timing_tracker.start_timing("save_outputs")
        
        saved_files = []
        total_mb = 0.0

        # Save per-layer activations
        for lname, arr in activations.items():
            t = torch.from_numpy(arr)
            fname = f"{output_prefix}_{lname}.pt"
            fpath = self.output_dir / fname
            torch.save(t, fpath)
            size_mb = fpath.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            saved_files.append({
                "layer_name": lname,
                "filename": fname,
                "shape": list(arr.shape),
                "size_mb": size_mb
            })
            logger.info(f"  saved {fname} | {arr.shape} | {size_mb:.1f} MB")

        # Save layer catalog (stratified sampling / stable IDs)
        catalog_path = self.output_dir / f"{output_prefix}_layer_catalog.csv"
        pd.DataFrame(self.layer_catalog, columns=["layer_name", "n_neurons"]).to_csv(catalog_path, index=False)
        logger.info(f"Saved layer catalog: {catalog_path}")

        # End timing for save outputs
        save_time = self.timing_tracker.end_timing("save_outputs")
        
        # Add complexity for save operation
        total_elements = sum(arr.size for arr in activations.values())
        self.timing_tracker.add_complexity(
            "save_outputs",
            num_nodes=len(activations),  # number of layers
            num_edges=total_elements,    # total number of elements to save
            time_complexity="O(E)",      # E = total elements
            space_complexity="O(E)",
            actual_operations=total_elements
        )

        # Save metadata JSON including timing information
        timing_summary = self.timing_tracker.get_summary()
        meta = {
            "model_type": model_type,
            "input_type": input_type,
            "tap_mode": self.tap_mode,
            "include_fc": self.include_fc,
            "num_layers": len(saved_files),
            "total_samples": int(next(iter(activations.values())).shape[0]) if activations else 0,
            "total_size_mb": total_mb,
            "files": saved_files,
            "device": self.device,
            "model_path": str(self.model_path),
            "random_seed": self.random_seed,
            "description": "Each file stores [num_samples, num_channels] for the tapped layer.",
            "timing_and_complexity": timing_summary
        }
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to JSON-serializable types
        meta = convert_numpy_types(meta)
        
        meta_path = self.output_dir / f"{output_prefix}_activation_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved activation metadata: {meta_path}")

        # Save triggered metadata if present
        if input_type == "triggered" and len(self.triggered_metadata_rows) > 0:
            trig_path = self.output_dir / f"{output_prefix}_triggered_input_metadata.csv"
            pd.DataFrame(self.triggered_metadata_rows).to_csv(trig_path, index=False)
            logger.info(f"Saved triggered input metadata: {trig_path}")

        logger.info(f"Total size this run: {total_mb:.1f} MB")
        logger.info(f"Save operations completed in {save_time:.4f}s")
        
        # Log the complete timing and complexity summary
        self.timing_tracker.log_summary(logger)
        
        # Add quick summary for user
        self.timing_tracker.log_quick_summary(logger)

    # ---------- Orchestrate ----------
    def run(self,
            model_type: str,
            input_type: str,
            dataset_name: str = "cifar10",
            batch_size: int = 32,
            sample_limit: Optional[int] = None,
            data_dir: str = "./data",
            trigger_args: Optional[Dict] = None,
            poison_target_label: Optional[int] = 0):
        logger.info("=== Activation Extraction (Improved) ===")
        logger.info(f"model_type={model_type} | input_type={input_type} | tap_mode={self.tap_mode}")
        logger.info(f"dataset={dataset_name} | batch_size={batch_size} | sample_limit={sample_limit}")
        logger.info(f"model_path={self.model_path} | out_dir={self.output_dir}")
        logger.info(f"Initial mem: {get_memory_usage_gb():.2f} GB")

        # Start timing for the entire extraction process
        self.timing_tracker.start_timing("total_extraction_process")
        
        output_prefix = f"{model_type}_model_{input_type}_inputs"

        model = self.load_model()
        dataset, _ = self.get_dataset(dataset_name, data_dir, input_type, trigger_args, int(poison_target_label or 0))

        # Apply sample_limit upfront via Subset (simpler & safer)
        if sample_limit is not None and sample_limit < len(dataset):
            logger.info(f"Applying sample limit: using first {sample_limit} samples.")
            dataset = torch.utils.data.Subset(dataset, range(sample_limit))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,                # can increase; metadata collection is now safe
            pin_memory=(self.device == "cuda")
        )

        self.register_hooks(model)
        activations = self.extract_activations(model, loader)
        self.save_outputs(activations, model_type, input_type, output_prefix)
        self.remove_hooks()

        # End timing for the entire process
        total_time = self.timing_tracker.end_timing("total_extraction_process")
        
        # Add complexity for the total process
        total_samples = len(dataset)
        num_layers = len(activations)
        self.timing_tracker.add_complexity(
            "total_extraction_process",
            num_nodes=total_samples * num_layers,  # total computational nodes
            num_edges=sum(arr.size for arr in activations.values()),  # total connections/operations
            time_complexity="O(N * L * F)",  # N samples, L layers, F features
            space_complexity="O(N * L * F)",
            actual_operations=sum(arr.size for arr in activations.values())
        )

        # cleanup
        del model, dataset, loader, activations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"=== Done: {output_prefix} ===")
        logger.info(f"Total extraction process completed in {total_time:.4f}s")
        logger.info(f"Output dir: {self.output_dir}")

    # ---------- Topological Analysis Methods ----------
    def build_correlation_matrix(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build correlation matrix from activations with timing and complexity tracking.
        """
        logger.info("Building correlation matrix...")
        self.timing_tracker.start_timing("matrix_building")
        
        # Concatenate all activations into a single matrix
        all_activations = []
        total_features = 0
        
        for layer_name, acts in activations.items():
            if acts.ndim > 2:
                # Flatten spatial dimensions if present
                acts_flat = acts.reshape(acts.shape[0], -1)
            else:
                acts_flat = acts
            all_activations.append(acts_flat)
            total_features += acts_flat.shape[1]
            logger.info(f"  Added {layer_name}: {acts_flat.shape}")
        
        # Concatenate all features
        combined_matrix = np.concatenate(all_activations, axis=1)
        n_samples, n_features = combined_matrix.shape
        
        logger.info(f"Combined matrix shape: {combined_matrix.shape}")
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(combined_matrix.T)
        
        # End timing and add complexity information
        matrix_time = self.timing_tracker.end_timing("matrix_building")
        
        # For correlation matrix: O(n^2) where n is number of features
        matrix_operations = n_features * n_features * n_samples  # approximate operations
        self.timing_tracker.add_complexity(
            "matrix_building",
            num_nodes=n_features,
            num_edges=n_features * n_features,  # fully connected correlation matrix
            time_complexity="O(n^2 * m)",  # n features, m samples
            space_complexity="O(n^2)",     # correlation matrix storage
            actual_operations=matrix_operations
        )
        
        logger.info(f"Correlation matrix built in {matrix_time:.4f}s")
        logger.info(f"Matrix shape: {correlation_matrix.shape}")
        logger.info(f"Features: {n_features}, Samples: {n_samples}")
        
        return correlation_matrix
    
    def compute_persistent_homology(self, correlation_matrix: np.ndarray, 
                                   threshold_range: Tuple[float, float] = (0.1, 0.9),
                                   num_thresholds: int = 50) -> Dict:
        """
        Compute persistent homology from correlation matrix with timing and complexity tracking.
        """
        logger.info("Computing persistent homology...")
        self.timing_tracker.start_timing("persistence_computation")
        
        n_nodes = correlation_matrix.shape[0]
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        persistence_data = {
            'thresholds': thresholds,
            'betti_numbers': {'b0': [], 'b1': []},
            'persistence_diagrams': [],
            'total_edges_per_threshold': []
        }
        
        total_operations = 0
        
        for i, threshold in enumerate(thresholds):
            # Create adjacency matrix by thresholding correlation matrix
            adj_matrix = (np.abs(correlation_matrix) >= threshold).astype(int)
            
            # Count edges at this threshold
            num_edges = np.sum(adj_matrix) // 2  # divide by 2 for undirected graph
            persistence_data['total_edges_per_threshold'].append(num_edges)
            
            # Placeholder for actual persistent homology computation
            # In practice, you would use libraries like GUDHI, Ripser, or similar
            
            # Simulate betti number computation (replace with actual implementation)
            # Betti numbers represent topological features
            b0 = max(1, n_nodes - num_edges // n_nodes)  # approximate connected components
            b1 = max(0, num_edges - n_nodes + b0)        # approximate cycles
            
            persistence_data['betti_numbers']['b0'].append(b0)
            persistence_data['betti_numbers']['b1'].append(b1)
            
            # Simulate persistence diagram computation
            persistence_data['persistence_diagrams'].append({
                'dimension_0': [(0, threshold)] * b0,
                'dimension_1': [(threshold * 0.8, threshold)] * b1
            })
            
            # Count operations (this is a rough estimate)
            threshold_operations = n_nodes * n_nodes + num_edges * np.log(num_edges)
            total_operations += threshold_operations
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{num_thresholds} thresholds")
        
        # End timing and add complexity information
        persistence_time = self.timing_tracker.end_timing("persistence_computation")
        
        # Persistent homology complexity depends on the filtration size and algorithm
        self.timing_tracker.add_complexity(
            "persistence_computation",
            num_nodes=n_nodes,
            num_edges=int(np.mean(persistence_data['total_edges_per_threshold'])),
            time_complexity="O(n^3) to O(n^4)",  # depends on algorithm (matrix algorithms vs. incremental)
            space_complexity="O(n^2)",
            actual_operations=total_operations
        )
        
        logger.info(f"Persistent homology computed in {persistence_time:.4f}s")
        logger.info(f"Processed {num_thresholds} thresholds")
        logger.info(f"Average edges per threshold: {np.mean(persistence_data['total_edges_per_threshold']):.1f}")
        
        return persistence_data
    
    def compute_topological_summary_statistics(self, persistence_data: Dict) -> Dict:
        """
        Compute Topological Summary Statistics (TSS) with timing and complexity tracking.
        """
        logger.info("Computing Topological Summary Statistics (TSS)...")
        self.timing_tracker.start_timing("tss_computation")
        
        betti_0 = np.array(persistence_data['betti_numbers']['b0'])
        betti_1 = np.array(persistence_data['betti_numbers']['b1'])
        thresholds = np.array(persistence_data['thresholds'])
        
        # Compute various TSS metrics
        tss = {
            'betti_statistics': {
                'b0_mean': float(np.mean(betti_0)),
                'b0_std': float(np.std(betti_0)),
                'b0_max': int(np.max(betti_0)),
                'b0_integral': float(np.trapz(betti_0, thresholds)),  # area under curve
                'b1_mean': float(np.mean(betti_1)),
                'b1_std': float(np.std(betti_1)),
                'b1_max': int(np.max(betti_1)),
                'b1_integral': float(np.trapz(betti_1, thresholds))
            },
            'persistence_landscape': {
                'b0_landscape_norm': float(np.linalg.norm(betti_0)),
                'b1_landscape_norm': float(np.linalg.norm(betti_1))
            },
            'topological_entropy': {
                # Compute normalized entropy-like measures
                'b0_entropy': -float(np.sum((betti_0/np.sum(betti_0)) * np.log(betti_0/np.sum(betti_0) + 1e-10))),
                'b1_entropy': -float(np.sum((betti_1/np.sum(betti_1)) * np.log(betti_1/np.sum(betti_1) + 1e-10)))
            },
            'stability_measures': {
                'b0_variance': float(np.var(betti_0)),
                'b1_variance': float(np.var(betti_1)),
                'total_persistence': float(np.sum(betti_0) + np.sum(betti_1))
            }
        }
        
        # End timing and add complexity information
        tss_time = self.timing_tracker.end_timing("tss_computation")
        
        num_thresholds = len(thresholds)
        tss_operations = num_thresholds * 10  # rough estimate for statistical computations
        
        self.timing_tracker.add_complexity(
            "tss_computation",
            num_nodes=num_thresholds,  # processing data points
            num_edges=num_thresholds * 2,  # processing both b0 and b1
            time_complexity="O(n)",    # linear in number of thresholds
            space_complexity="O(n)",   # storing summary statistics
            actual_operations=tss_operations
        )
        
        logger.info(f"TSS computed in {tss_time:.4f}s")
        logger.info(f"Betti-0 statistics: mean={tss['betti_statistics']['b0_mean']:.2f}, max={tss['betti_statistics']['b0_max']}")
        logger.info(f"Betti-1 statistics: mean={tss['betti_statistics']['b1_mean']:.2f}, max={tss['betti_statistics']['b1_max']}")
        
        return tss

    def run_full_topological_analysis(self,
                                     model_type: str,
                                     input_type: str,
                                     dataset_name: str = "cifar10",
                                     batch_size: int = 32,
                                     sample_limit: Optional[int] = None,
                                     data_dir: str = "./data",
                                     trigger_args: Optional[Dict] = None,
                                     poison_target_label: Optional[int] = 0,
                                     compute_topology: bool = True,
                                     threshold_range: Tuple[float, float] = (0.1, 0.9),
                                     num_thresholds: int = 50):
        """
        Run complete topological analysis pipeline with timing for all steps.
        """
        # First run activation extraction
        self.run(model_type, input_type, dataset_name, batch_size, 
                sample_limit, data_dir, trigger_args, poison_target_label)
        
        if compute_topology:
            # Load the saved activations for topological analysis
            output_prefix = f"{model_type}_model_{input_type}_inputs"
            
            # Load activations from saved files
            activations = {}
            for layer_name, _ in self.layer_catalog:
                fname = f"{output_prefix}_{layer_name}.pt"
                fpath = self.output_dir / fname
                if fpath.exists():
                    tensor = torch.load(fpath, map_location='cpu')
                    activations[layer_name] = tensor.numpy()
                    logger.info(f"Loaded {layer_name}: {activations[layer_name].shape}")
            
            if activations:
                # Build correlation matrix
                correlation_matrix = self.build_correlation_matrix(activations)
                
                # Compute persistent homology
                persistence_data = self.compute_persistent_homology(
                    correlation_matrix, threshold_range, num_thresholds)
                
                # Compute TSS
                tss = self.compute_topological_summary_statistics(persistence_data)
                
                # Save topological analysis results
                topo_results = {
                    'correlation_matrix_shape': correlation_matrix.shape,
                    'persistence_data': persistence_data,
                    'topological_summary_statistics': tss,
                    'timing_and_complexity': self.timing_tracker.get_summary()
                }
                
                def convert_numpy_types(obj):
                    """Convert numpy types to native Python types for JSON serialization."""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                # Convert numpy types to JSON-serializable types
                topo_results_serializable = convert_numpy_types(topo_results)
                
                # Save results
                topo_path = self.output_dir / f"{output_prefix}_topological_analysis.json"
                with open(topo_path, 'w') as f:
                    json.dump(topo_results_serializable, f, indent=2)
                
                logger.info(f"Saved topological analysis results: {topo_path}")
                
                # Save correlation matrix separately (as .npy for efficiency)
                corr_path = self.output_dir / f"{output_prefix}_correlation_matrix.npy"
                np.save(corr_path, correlation_matrix)
                logger.info(f"Saved correlation matrix: {corr_path}")
                
                # Log final summary
                self.timing_tracker.log_summary(logger)
                
                # Add quick summary for user  
                self.timing_tracker.log_quick_summary(logger)
            else:
                logger.warning("No activations found for topological analysis")

# -----------------------------
# CLI
# -----------------------------
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Extract ResNet-18 activations for correlation & PH (TopoTroj/TOAP compatible).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean ResNet-18 model (architecture auto-detected)
  python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/clean_resnet18.pth --tap_mode topotroj_compat --include_fc

  # EfficientNet-B0 model with spatial preservation (specify architecture or let it auto-detect)
  python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/efficientnet_b0.pth --architecture efficientnet_b0 --tap_mode spatial_preserve --include_fc

  # Full topological analysis with auto-detection
  python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/any_model.pth --run_topology --threshold_min 0.2 --threshold_max 0.8 --num_thresholds 30

  # Backdoor detection with triggered inputs
  python activation_topotroj.py --model_type backdoored --input_type triggered --model_path ./models/backdoored.pth --run_topology --trigger_pattern_size 3 --trigger_pixel_value 1.0 --trigger_location br --poison_target_label 0

  # Force specific architecture if auto-detection fails
  python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/model.pth --architecture efficientnet_b0 --tap_mode spatial_preserve

Note: The script will attempt to auto-detect the architecture from the model file. 
      Use --architecture only if auto-detection fails or you want to override it.
        """
    )
    p.add_argument("--model_type", type=str, required=True, choices=["clean", "backdoored"])
    p.add_argument("--input_type", type=str, required=True, choices=["clean", "triggered"])
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="activation_output_topo")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--random_seed", type=int, default=42)

    # Trigger args
    p.add_argument("--trigger_pattern_size", type=int, default=3)
    p.add_argument("--trigger_pixel_value", type=float, default=1.0)
    p.add_argument("--trigger_location", type=str, default="br", choices=["br", "tl", "tr", "bl"])
    p.add_argument("--poison_target_label", type=int, default=0)

    # Tap config
    p.add_argument("--tap_mode", type=str, default="topotroj_compat",
                   choices=["topotroj_compat", "toap_block_out", "bn2_legacy", "spatial_preserve"])
    p.add_argument("--include_fc", action="store_true",
                   help="Include final fc in topotroj_compat mode (ignored in others).")
    p.add_argument("--architecture", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet152", "mobilenet_v2", "efficientnet_b0"],
                   help="Model architecture used for training the model.")
    
    # Topological analysis options
    p.add_argument("--run_topology", action="store_true",
                   help="Run full topological analysis including matrix building, persistence, and TSS.")
    p.add_argument("--threshold_min", type=float, default=0.1,
                   help="Minimum threshold for correlation matrix filtering in persistent homology.")
    p.add_argument("--threshold_max", type=float, default=0.9,
                   help="Maximum threshold for correlation matrix filtering in persistent homology.")
    p.add_argument("--num_thresholds", type=int, default=50,
                   help="Number of thresholds to use in persistent homology computation.")

    return p.parse_args()

def main():
    args = parse_arguments()

    # Single, global seeding for full determinism
    set_seed(args.random_seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.input_type == "triggered" and args.model_type == "clean":
        raise ValueError("Triggered inputs are meant for analyzing backdoored models. Use --model_type backdoored.")

    trigger_args = None
    poison_target_label = None
    if args.input_type == "triggered":
        trigger_args = {
            "trigger_pattern_size": args.trigger_pattern_size,
            "trigger_pixel_value": args.trigger_pixel_value,
            "trigger_location": args.trigger_location
        }
        poison_target_label = args.poison_target_label

    extractor = ImprovedActivationExtractor(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.random_seed,
        tap_mode=args.tap_mode,
        include_fc=args.include_fc,
        architecture=args.architecture
    )

    if args.run_topology:
        # Run full topological analysis pipeline
        extractor.run_full_topological_analysis(
            model_type=args.model_type,
            input_type=args.input_type,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            sample_limit=args.sample_limit,
            data_dir=args.data_dir,
            trigger_args=trigger_args,
            poison_target_label=poison_target_label,
            compute_topology=True,
            threshold_range=(args.threshold_min, args.threshold_max),
            num_thresholds=args.num_thresholds
        )
    else:
        # Run only activation extraction
        extractor.run(
            model_type=args.model_type,
            input_type=args.input_type,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            sample_limit=args.sample_limit,
            data_dir=args.data_dir,
            trigger_args=trigger_args,
            poison_target_label=poison_target_label
        )

if __name__ == "__main__":
    main()
