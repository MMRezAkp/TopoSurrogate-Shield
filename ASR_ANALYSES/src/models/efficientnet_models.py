"""
EfficientNet feature extraction models.
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.layer_encoder import LayerEncoder


class OptimizedEfficientNetFeatureExtractor:
    """Memory-efficient feature extraction from EfficientNet model."""
    
    def __init__(self, model_path, device='cuda', num_classes=10):
        self.device = device
        self.num_classes = num_classes
        self.model = self._load_model(model_path)
        self.layer_encoder = LayerEncoder()
        self.activation_hooks = {}
        self.activations = {}

    def _load_model(self, model_path):
        """Load EfficientNet model from checkpoint."""
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, self.num_classes))
        
        if isinstance(state_dict, dict):
            state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        model = model.to(self.device).eval()
        print(f"Successfully loaded EfficientNet model with {self.num_classes} classes on {self.device}")
        return model

    def get_important_layers(self, all_layers, max_layers=20):
        """Get the most important layers for feature extraction."""
        priority_layers = [
            'features.0.0','features.0.1',
            'features.1.0','features.1.1','features.1.2',
            'features.2.0','features.2.1','features.2.2','features.2.3',
            'features.3.0','features.3.1','features.3.2','features.3.3','features.3.4','features.3.5',
            'features.4.0','features.4.1','features.4.2','features.4.3','features.4.4','features.4.5',
            'features.5.0','features.5.1','features.5.2','features.5.3','features.5.4','features.5.5',
            'features.6.0','features.6.1.block.1.0','features.6.1.block.2.0',
            'features.7.0','features.7.0.block.1.0','features.7.0.block.2.0',
            'features.8.0','features.8.0.block.1.0','features.8.0.block.2.0',
            'classifier.0','classifier.1'
        ]
        selected = []
        for l in priority_layers:
            if l in all_layers:
                selected.append(l)
                if len(selected) >= max_layers:
                    break
        if len(selected) < max_layers:
            rest = [l for l in all_layers if l not in selected]
            selected.extend(rest[:max_layers-len(selected)])
        return selected

    def _register_hooks(self, layer_names):
        """Register forward hooks for activation extraction."""
        self.activations = {}
        def create_hook(name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    self.activations[name] = out.detach().cpu()
            return hook
        named = dict(self.model.named_modules())
        for name in layer_names:
            if name in named:
                self.activation_hooks[name] = named[name].register_forward_hook(create_hook(name))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.activation_hooks.values():
            h.remove()
        self.activation_hooks.clear()

    @torch.no_grad()
    def extract_activations(self, dataloader, layer_names):
        """Extract activation features from specified layers."""
        self._register_hooks(layer_names)
        stats = {}
        batches = 0
        for b, (x, _) in enumerate(dataloader):
            x = x.to(self.device, non_blocking=True)
            _ = self.model(x)
            batches += 1
            for lname in layer_names:
                if lname in self.activations:
                    a = self.activations[lname]
                    if a.dim() == 4:
                        a = torch.mean(a, dim=(2,3))
                    elif a.dim() != 2:
                        continue
                    for ch in range(a.shape[1]):
                        v = a[:, ch]
                        m = float(v.mean().item())
                        var = float(v.var().item())
                        stats.setdefault((lname, ch), []).append((m, var))
            if batches >= 3:
                break
        final = {}
        for k, arr in stats.items():
            ms = [u for (u, _) in arr]
            vs = [v for (_, v) in arr]
            final[k] = (float(np.mean(ms)), float(np.mean(vs)))
        self._remove_hooks()
        return final

    def extract_weight_features(self, layer_names):
        """Extract weight features from specified layers."""
        feats = {}
        named = dict(self.model.named_modules())
        for lname in layer_names:
            if lname not in named: 
                continue
            m = named[lname]
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                W = m.weight.detach()
                for ch in range(W.shape[0]):
                    if isinstance(m, nn.Conv2d):
                        w_norm = float(W[ch].norm().item())
                        k = m.kernel_size[0]*m.kernel_size[1]
                        fan_in = int(m.in_channels * k)
                        fan_out = int(k)
                    else:
                        w_norm = float(W[ch].norm().item())
                        fan_in = m.in_features
                        fan_out = 1
                    grad_norm = 0.0
                    if m.weight.grad is not None:
                        grad_norm = float(m.weight.grad[ch].norm().item())
                    feats[(lname, ch)] = {
                        'w_out_norm': w_norm, 'grad_out_norm': grad_norm,
                        'fan_in': fan_in, 'fan_out': fan_out
                    }
        return feats

    def get_all_layer_names(self):
        """Get all layer names that have weights."""
        return [n for n, mod in self.model.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]

    def extract_all_features(self, dataloader, max_layers=20):
        """Extract all features (weights and activations) from the model."""
        all_layers = self.get_all_layer_names()
        selected = self.get_important_layers(all_layers, max_layers)
        w = self.extract_weight_features(selected)
        a = self.extract_activations(dataloader, selected)
        return w, a, selected



