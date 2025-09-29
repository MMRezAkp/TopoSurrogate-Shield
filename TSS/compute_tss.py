#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Topological Susceptibility Score (TSS) - Edge-only on dense VR, with robust perturbations.

Adds:
- Edge-only scoring (skip per-neuron).
- Dense-only perturbations (two modes):
  - remove: set the tested edge distance to a large value; optionally inflate adjacent loop edges by epsilon.
  - multiplicative: scale the tested edge distance by a factor; optionally scale adjacent loop edges by a smaller factor.
- Perturb multiple adjacent edges (optional).
- Random Gaussian noise per test (optional; breaks symmetries).
- Per-edge features: sim/dist, weight/grad norms, activation mean/var, layer depth, fan-in/out.
- Total edges and total unique edges counted and printed.
- Global progress bar + periodic checkpoints every N edges to resume mid-loop.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
from ripser import Rips
import torch.nn as nn
from tqdm import tqdm
import inspect

# -----------------------------
# Small utils
# -----------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ascii_print(s):
    try:
        print(s)
    except Exception:
        print(s.encode("ascii", "ignore").decode("ascii"))

# -----------------------------
# Distance loading & PH recompute
# -----------------------------
def load_distance_or_build(bundle_dir, method="pearson"):
    dist_path = os.path.join(bundle_dir, "distance_sqeuclid.npy")
    sim_path  = os.path.join(bundle_dir, f"{method}_corr.npy")
    if os.path.exists(dist_path):
        D = np.load(dist_path).astype(np.float32)
        return D, f"loaded:{os.path.basename(dist_path)}"
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Neither distance_sqeuclid.npy nor {method}_corr.npy found in {bundle_dir}")
    S = np.load(sim_path).astype(np.float32)
    S = np.clip(S, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - S))).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D, f"built_from:{os.path.basename(sim_path)}"

def run_ripser_on_perturbed_dm(D_dense: np.ndarray, maxdim: int, coeff: int):
    rips = Rips(verbose=False, maxdim=maxdim, do_cocycles=False, coeff=coeff)
    return rips.fit_transform(D_dense, distance_matrix=True)

def robust_loop_persistence(h1_diagram: np.ndarray, baseline_birth: float, baseline_death: float) -> float:
    if h1_diagram is None or h1_diagram.size == 0:
        return 0.0
    bd = h1_diagram[:, :2]
    target = np.array([baseline_birth, baseline_death], dtype=float)
    i = int(np.argmin(np.linalg.norm(bd - target, axis=1)))
    b, d = bd[i]
    if not (np.isfinite(b) and np.isfinite(d)):
        return 0.0
    return max(0.0, float(d - b))

def nearest_persistence(diagram: np.ndarray, ref_birth: float, ref_death: float) -> float:
    if diagram is None or diagram.size == 0:
        return 0.0
    bd = diagram[:, :2]
    i = int(np.argmin(np.linalg.norm(bd - np.array([ref_birth, ref_death]), axis=1)))
    b, d = bd[i]
    return max(0.0, float(d - b)) if np.isfinite(b) and np.isfinite(d) else 0.0

# -----------------------------
# Cocycle parsing
# -----------------------------
def read_cocycles(bundle_ph_dir: Path, node_table_df: pd.DataFrame):
    mapped_path = bundle_ph_dir / "h1_cocycles_mapped.json"
    raw_path    = bundle_ph_dir / "h1_generators.npy"

    loops = []
    if mapped_path.exists():
        with open(mapped_path, "r") as f:
            mapped = json.load(f)
        for cyc in mapped:
            edges = []
            nodes = set()
            for e in cyc:
                i = int(e["i"]); j = int(e["j"]); coef = float(e.get("coef", 1.0))
                edges.append((i, j, coef))
                nodes.add(i); nodes.add(j)
            loops.append({"edges": edges, "nodes": sorted(nodes)})
        return loops

    if raw_path.exists():
        raw = np.load(raw_path, allow_pickle=True)
        for cyc in raw:
            cyc = np.asarray(cyc)
            edges = []
            nodes = set()
            for row in cyc:
                if len(row) < 2:
                    continue
                i, j = int(row[0]), int(row[1])
                coef = float(row[2]) if len(row) > 2 else 1.0
                edges.append((i, j, coef))
                nodes.add(i); nodes.add(j)
            loops.append({"edges": edges, "nodes": sorted(nodes)})
        return loops

    raise FileNotFoundError("Neither h1_cocycles_mapped.json nor h1_generators.npy found.")

# -----------------------------
# Masking for ASR (compat; unused in edge-only)
# -----------------------------
def pick_mask_target(layer_name: str, model: torch.nn.Module) -> str:
    named = dict(model.named_modules())
    if layer_name in named and not layer_name.endswith(".bn2"):
        return layer_name
    base = None
    for suf, cut in ((".conv1", 6), (".bn1", 4), (".relu", 5), (".conv2", 6), (".bn2", 4)):
        if layer_name.endswith(suf):
            base = layer_name[:-cut]
            break
    if base:
        for suffix in (".", ".relu", ".bn2", ".conv2"):
            cand = base if suffix == "." else base + suffix
            if cand in named:
                return cand
    return layer_name if layer_name in named else None

def compute_asr_with_mask(model: torch.nn.Module,
                          data_loader,
                          node_table_df: pd.DataFrame,
                          target_label: int,
                          neuron_mask=None,
                          print_hook_fires: bool = False) -> float:
    device = next(model.parameters()).device
    model.eval()
    hooks = []
    fires = defaultdict(int)
    named = dict(model.named_modules())

    if neuron_mask:
        layer_to_idxs = defaultdict(list)
        for lname, idx in neuron_mask:
            if lname in named:
                layer_to_idxs[lname].append(int(idx))
            else:
                ascii_print(f"[WARN] mask target not found: {lname}")

        for lname, indices in layer_to_idxs.items():
            m = named[lname]
            if isinstance(m, nn.Conv2d):
                def pre_hook(_m, inputs, indices=indices, lname=lname):
                    fires[lname] += 1
                    x = inputs[0]
                    Cin = x.shape[1]
                    valid = [i for i in indices if 0 <= i < Cin]
                    if valid:
                        x[:, valid, :, :] = 0
                    return (x,)
                hooks.append(m.register_forward_pre_hook(pre_hook))
            elif isinstance(m, nn.Linear):
                def pre_hook(_m, inputs, indices=indices, lname=lname):
                    fires[lname] += 1
                    x = inputs[0]
                    Fin = x.shape[1]
                    valid = [i for i in indices if 0 <= i < Fin]
                    if valid:
                        x[:, valid] = 0
                    return (x,)
                hooks.append(m.register_forward_pre_hook(pre_hook))
            else:
                def fwd_hook(_m, _inp, out, indices=indices, lname=lname):
                    fires[lname] += 1
                    if torch.is_tensor(out):
                        if out.dim() == 4:
                            C = out.shape[1]
                            valid = [i for i in indices if 0 <= i < C]
                            if valid:
                                out[:, valid, :, :] = 0
                        elif out.dim() == 2:
                            C = out.shape[1]
                            valid = [i for i in indices if 0 <= i < C]
                            if valid:
                                out[:, valid] = 0
                    return out
                hooks.append(m.register_forward_hook(fwd_hook))

    total = 0
    hits = 0
    with torch.no_grad():
        for images, _labels in data_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            hits += int((preds.cpu().numpy() == target_label).sum())
            total += preds.shape[0]

    for h in hooks:
        h.remove()

    if print_hook_fires and neuron_mask:
        for lname, cnt in fires.items():
            ascii_print(f"[INFO] hook fired: {lname} -> {cnt}")

    return (hits / total) if total > 0 else 0.0

# -----------------------------
# Flexible loader wrapper
# -----------------------------
def flexible_get_data_loader(fn, data_root_positional, **kwargs):
    sig = inspect.signature(fn)
    if "trigger_pixel_value" in sig.parameters and "trigger_pixel_value" not in kwargs and "trigger_value" in kwargs:
        kwargs["trigger_pixel_value"] = kwargs["trigger_value"]
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(data_root_positional, **accepted)

# -----------------------------
# Feature extraction for edges
# -----------------------------
def _parse_layer_depth(layer_name: str) -> int:
    """
    Parse layer depth from layer name for different architectures.
    Returns a depth score based on the layer position in the network.
    """
    try:
        parts = layer_name.split(".")
        
        # Handle different architecture patterns
        if layer_name.startswith("features."):
            # EfficientNet-B0: features.0.0, features.1.0, features.2.0, etc.
            if len(parts) >= 2:
                stage = int(parts[1])  # features.X.Y -> stage X
                # For EfficientNet-B0, the main stages are 0, 1, 2, 3, 4, 5, 6, 7, 8
                # Each stage has multiple blocks, so we use stage * 10 + block
                if len(parts) >= 3:
                    block = int(parts[2])  # features.X.Y.Z -> block Z
                    return stage * 10 + block
                else:
                    return stage * 10  # features.X.Y -> stage X, block 0
            else:
                return 0
        elif layer_name.startswith("layer"):
            # ResNet: layer1.0, layer2.0, etc.
            s = int(parts[0].replace("layer", ""))
            b = int(parts[1]) if len(parts) > 1 else 0
            return s * 10 + b
        elif layer_name.startswith("conv"):
            # MobileNetV2: conv1, conv2, etc.
            conv_num = int(parts[0].replace("conv", ""))
            return conv_num
        else:
            # For other patterns, try to extract numbers
            numbers = [int(p) for p in parts if p.isdigit()]
            if numbers:
                return sum(numbers)
            return 0
    except Exception:
        return 0

def _load_activation_stats_for_layers(activations_dir: Path, prefix: str, layer_names):
    stats = {}
    print(f"DEBUG: Loading activation stats from {activations_dir} with prefix {prefix}")
    print(f"DEBUG: Looking for {len(set(layer_names))} unique layers")
    
    for ln in set(layer_names):
        f = activations_dir / f"{prefix}_{ln}.pt"
        if not f.exists():
            print(f"DEBUG: Activation file not found: {f}")
            continue
        X = torch.load(f, map_location="cpu")
        if not isinstance(X, torch.Tensor):
            print(f"DEBUG: Activation file {f} is not a tensor")
            continue
        
        print(f"DEBUG: Loading activation stats for {ln}, shape: {X.shape}")
        
        # Handle both 2D (GAP modes) and 4D (spatial_preserve mode) tensors
        if X.dim() == 2:
            # 2D tensor [N, C] - from GAP modes
            mu = X.mean(dim=0).float().numpy()
            va = X.var(dim=0, unbiased=False).float().numpy()
            for ch in range(X.shape[1]):
                stats[(ln, int(ch))] = (float(mu[ch]), float(va[ch]))
            print(f"DEBUG: Added {X.shape[1]} channels for {ln}")
        elif X.dim() == 4:
            # 4D tensor [N, C, H, W] - from spatial_preserve mode
            # Flatten spatial dimensions: [N, C, H, W] -> [N, C*H*W]
            N, C, H, W = X.shape
            X_flat = X.view(N, C * H * W)  # Flatten spatial dimensions
            mu = X_flat.mean(dim=0).float().numpy()
            va = X_flat.var(dim=0, unbiased=False).float().numpy()
            for ch in range(C * H * W):
                stats[(ln, int(ch))] = (float(mu[ch]), float(va[ch]))
            print(f"DEBUG: Added {C * H * W} spatial locations for {ln}")
        else:
            print(f"DEBUG: Skipping {ln} with unexpected dimension {X.dim()}")
    
    print(f"DEBUG: Loaded activation stats for {len(stats)} spatial locations")
    if len(stats) > 0:
        sample_stats = list(stats.items())[:3]
        print(f"DEBUG: Sample activation stats: {sample_stats}")
    return stats

def _precompute_weight_and_grad_features(model: torch.nn.Module,
                                         node_df: pd.DataFrame,
                                         poison_loader,
                                         num_batches: int,
                                         device: str):
    model.zero_grad(set_to_none=True)
    if num_batches > 0:
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        done = 0
        for images, labels in poison_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            done += 1
            if done >= num_batches:
                break
        model.eval()

    feats = {}
    named = dict(model.named_modules())
    print(f"DEBUG: Processing {len(node_df)} nodes for weight/gradient features")
    print(f"DEBUG: Available model layers: {list(named.keys())[:10]}...")  # Show first 10 layers
    
    # Show sample layer names from node table
    sample_layers = node_df['layer_name'].head(5).tolist()
    print(f"DEBUG: Sample node table layer names: {sample_layers}")
    
    # Test depth parsing on sample layers
    print("DEBUG: Testing depth parsing on sample layers:")
    for layer in sample_layers:
        depth = _parse_layer_depth(layer)
        print(f"  {layer} -> depth: {depth}")
    
    for _, r in node_df.iterrows():
        ln = str(r["layer_name"])
        # Handle both spatial and non-spatial node tables
        if "spatial_h" in node_df.columns and "spatial_w" in node_df.columns and "channel" in node_df.columns:
            # Spatial mode: use channel column
            ch = int(r["channel"])
        else:
            # Non-spatial mode: use local_index column
            ch = int(r["local_index"])
        
        # Try to find the corresponding layer for weight/gradient extraction
        # For spatial mode, ln is the actual layer name from model.named_modules()
        conv_layer = None
        
        # First try the direct layer name
        if ln in named:
            conv_layer = named[ln]
        else:
            # Try to find the layer by searching for exact matches first
            for layer_name, module in named.items():
                if layer_name == ln and isinstance(module, nn.Conv2d):
                    conv_layer = module
                    break
            # If still not found, try partial matches
            if conv_layer is None:
                for layer_name, module in named.items():
                    if ln in layer_name and isinstance(module, nn.Conv2d):
                        conv_layer = module
                        break
        
        if isinstance(conv_layer, nn.Conv2d) and ch < conv_layer.out_channels:
            W = conv_layer.weight.detach()
            wnorm = float(W[ch].norm().item())
            gnorm = 0.0
            # Note: Gradients will be zero unless the model is in training mode and has been backpropagated
            # For TSS computation, we typically don't need gradients, so this is expected to be 0
            if conv_layer.weight.grad is not None:
                g = conv_layer.weight.grad[ch]
                gnorm = float(g.norm().item())
            k = conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
            fan_in = int(conv_layer.in_channels * k)
            fan_out = int(k)
            feats[(ln, ch)] = {"w_out_norm": wnorm, "grad_out_norm": gnorm, "fan_in": fan_in, "fan_out": fan_out}
            if len(feats) <= 10:  # Print more for debugging
                print(f"DEBUG: Found layer {ln} -> {type(conv_layer).__name__}, ch={ch}, wnorm={wnorm:.4f}")
        else:
            # If we can't find the layer or it's not Conv2d, set default values
            feats[(ln, ch)] = {"w_out_norm": 0.0, "grad_out_norm": 0.0, "fan_in": 0, "fan_out": 0}
            if len(feats) <= 20:  # Print more for debugging
                if conv_layer is None:
                    print(f"DEBUG: No layer found for {ln}, ch={ch}")
                elif not isinstance(conv_layer, nn.Conv2d):
                    print(f"DEBUG: Layer {ln} is not Conv2d, ch={ch}, type={type(conv_layer).__name__}")
                else:
                    print(f"DEBUG: Channel {ch} >= out_channels {conv_layer.out_channels} for layer {ln}")
    
    print(f"DEBUG: Found {len([f for f in feats.values() if f['w_out_norm'] > 0])} layers with valid weight features out of {len(feats)} total")
    
    # Debug: Check depth parsing on all unique layer names
    unique_layers = node_df['layer_name'].unique()
    print(f"DEBUG: Checking depth parsing on {len(unique_layers)} unique layers:")
    zero_depth_count = 0
    for layer in unique_layers[:10]:  # Check first 10
        depth = _parse_layer_depth(layer)
        if depth == 0:
            zero_depth_count += 1
            print(f"  ZERO DEPTH: {layer} -> {depth}")
        else:
            print(f"  OK: {layer} -> {depth}")
    
    print(f"DEBUG: {zero_depth_count} out of {len(unique_layers[:10])} sample layers have zero depth")
    return feats

# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)

    prefix = f"{args.model_type}_model_{args.input_type}_inputs"
    corr_dir = Path(args.correlation_output_base_dir) / prefix
    ph_dir   = Path(args.ph_output_base_dir) / prefix
    out_dir  = ph_dir / "tss"
    out_dir.mkdir(parents=True, exist_ok=True)

    ascii_print(f"=== TSS (edge-only, dense) for: {prefix} ===")
    ascii_print(f"Read corr from: {corr_dir}")
    ascii_print(f"Read PH   from: {ph_dir}")
    ascii_print(f"Write TSS to : {out_dir}")

    # Load node map & distances
    node_table_path = corr_dir / "node_table.csv"
    if not node_table_path.exists():
        raise FileNotFoundError(f"node_table.csv not found in {corr_dir}")
    node_df = pd.read_csv(node_table_path)
    N = int(len(node_df))

    D_base, D_note = load_distance_or_build(str(corr_dir), method=args.correlation_method)
    if D_base.shape[0] != N:
        raise ValueError(f"Distance size {D_base.shape} != nodes {N}")
    ascii_print(f"Distance (dense): {D_note}  shape={D_base.shape}")

    # Load PH artifacts
    h1_path = ph_dir / "h1_persistence.npy"
    if not h1_path.exists():
        raise FileNotFoundError(f"h1_persistence.npy not found in {ph_dir}")
    H1 = np.load(h1_path)

    # finite-only mask
    finite_idx_path = ph_dir / "h1_finite_indices.json"
    finite_mask = None
    if finite_idx_path.exists():
        with open(finite_idx_path, "r") as f:
            finite_indices = set(json.load(f))
        finite_mask = np.array([i in finite_indices for i in range(H1.shape[0])], dtype=bool)

    is_finite_bar = np.isfinite(H1).all(axis=1) if H1.size > 0 else np.array([], dtype=bool)
    if finite_mask is not None:
        is_finite_bar = is_finite_bar & finite_mask
    persistences = (H1[:, 1] - H1[:, 0]) if H1.size > 0 else np.array([])
    persistences = np.where(is_finite_bar, persistences, -np.inf)

    # Select loops (top by persistence, finite only)
    order = np.argsort(persistences)[::-1]
    order = [int(i) for i in order if np.isfinite(persistences[i]) and persistences[i] > 0]
    num_loops = min(args.num_top_loops, len(order))
    ascii_print(f"Analyzing {num_loops} finite loop(s): alpha={args.alpha}, beta={args.beta}")

    # Optional CLEAN contrast (dense)
    D_clean = None
    H1_clean = None
    if args.contrast_with_clean:
        clean_prefix = f"{args.clean_model_type}_model_{args.clean_input_type}_inputs"
        corr_dir_clean = Path(args.correlation_output_base_dir) / clean_prefix
        ph_dir_clean   = Path(args.ph_output_base_dir) / clean_prefix

        D_clean, _ = load_distance_or_build(str(corr_dir_clean), method=args.correlation_method)
        if D_clean.shape != D_base.shape:
            raise ValueError(f"[contrast] clean distance shape {D_clean.shape} != backdoor distance shape {D_base.shape}")
        h1_clean_path = ph_dir_clean / "h1_persistence.npy"
        if not h1_clean_path.exists():
            raise FileNotFoundError(f"[contrast] missing {h1_clean_path}")
        H1_clean = np.load(h1_clean_path)

    # Cocycle edges/nodes
    loops = read_cocycles(ph_dir, node_df)

    # Model and eval loader for grads/features
    from model import get_model
    from data import get_data_loader

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    model = get_model(num_classes=args.num_classes, pretrained=False, architecture=args.architecture)
    state = torch.load(args.model_ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    dl_kwargs = dict(
        batch_size=args.batch_size,
        poison_ratio=args.poison_ratio_eval,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        trigger_color=[args.trigger_value] * 3,
        trigger_location=args.trigger_location,
        trigger_value=args.trigger_value,
        train=False,
    )
    _, _, poison_loader = flexible_get_data_loader(get_data_loader, args.data_root, **dl_kwargs)

    # Features
    act_dir = Path(args.activations_dir)
    
    # Debug: Check what activation files actually exist
    print(f"DEBUG: Checking activation directory: {act_dir}")
    if act_dir.exists():
        activation_files = list(act_dir.glob("*.pt"))
        print(f"DEBUG: Found {len(activation_files)} activation files")
        if activation_files:
            print(f"DEBUG: Sample activation files:")
            for f in activation_files[:5]:
                print(f"  {f.name}")
    else:
        print(f"DEBUG: Activation directory {act_dir} does not exist!")
    
    # For activation stats, we need to use the correct prefix format
    # Try different prefix formats to find the right one
    possible_prefixes = [
        f"{args.model_type}_model_{args.input_type}_inputs",  # clean_model_clean_inputs
        f"{args.model_type}_{args.input_type}",  # clean_clean
        f"{args.model_type}",  # clean
        ""  # No prefix
    ]
    
    act_stats = {}
    for act_prefix in possible_prefixes:
        print(f"DEBUG: Trying activation prefix: '{act_prefix}'")
        act_stats = _load_activation_stats_for_layers(act_dir, act_prefix, node_df["layer_name"].tolist())
        if len(act_stats) > 0:
            print(f"DEBUG: Successfully loaded activation stats with prefix '{act_prefix}'")
            break
        else:
            print(f"DEBUG: No activation stats found with prefix '{act_prefix}'")
    
    if len(act_stats) == 0:
        print("WARNING: No activation stats loaded! All activation features will be zero.")
    wg_feats  = _precompute_weight_and_grad_features(model, node_df, poison_loader, args.num_grad_batches, device)

    # Progress (resume)
    per_edge_rows = []
    all_results = []
    progress_path = out_dir / "progress.json"
    edges_progress_csv = out_dir / "tss_edges_progress.csv"
    loops_progress_json = out_dir / "all_loops_edges_results.json"
    edges_done = 0
    if args.resume_from_progress and progress_path.exists() and edges_progress_csv.exists():
        try:
            with open(progress_path, "r") as f:
                prog = json.load(f)
            edges_done = int(prog.get("edges_done", 0))
            per_edge_rows = pd.read_csv(edges_progress_csv).to_dict("records")
            if loops_progress_json.exists():
                with open(loops_progress_json, "r") as f:
                    all_results = json.load(f)
            ascii_print(f"Resuming: edges_done={edges_done}")
        except Exception:
            edges_done = len(per_edge_rows)
            all_results = []

    # Pre-compute total and unique edges across selected loops
    total_edges_all = 0
    loop_edges_cache = []
    for lr in range(0, num_loops):
        idx = int(order[lr])
        if idx >= len(loops):
            loop_edges_cache.append([])
            continue
        loop = loops[idx]
        edges_list = [(a, b_) for (a, b_, _coef) in loop["edges"]]
        loop_edges_cache.append(edges_list)
        total_edges_all += len(edges_list)

    unique_edges = set()
    for edges_list in loop_edges_cache:
        for (a, b_) in edges_list:
            u, v = (a, b_) if a < b_ else (b_, a)
            unique_edges.add((u, v))
    total_unique_edges_all = len(unique_edges)

    ascii_print(f"Total edges across selected loops: {total_edges_all}")
    ascii_print(f"Total unique edges across selected loops: {total_unique_edges_all}")

    # Progress bar
    pbar = tqdm(total=total_edges_all, initial=min(edges_done, total_edges_all), desc="Edges", dynamic_ncols=True)

    # Main loops (edge-only)
    processed_edges_global = edges_done
    for loop_rank in range(0, num_loops):
        idx = int(order[loop_rank])
        b, d = float(H1[idx, 0]), float(H1[idx, 1])
        if not (np.isfinite(b) and np.isfinite(d)):
            continue
        base_pers = max(0.0, d - b)
        if base_pers <= 0:
            continue

        loop = loops[idx] if idx < len(loops) else None
        if loop is None:
            continue
        loop_nodes = loop["nodes"]
        loop_edges = loop_edges_cache[loop_rank]

        # Skip edges already processed globally
        prev_edges = sum(len(loop_edges_cache[i]) for i in range(loop_rank))
        start_edge_idx = min(max(0, edges_done - prev_edges), len(loop_edges))

        # Precompute adjacency-in-loop for fast inflation
        adj_in_loop = {}
        if args.adjacent_eps > 0.0 or args.adjacent_factor != 1.0:
            neigh = defaultdict(set)
            for (u, v) in loop_edges:
                neigh[u].add(v)
                neigh[v].add(u)
            adj_in_loop = {k: list(v) for k, v in neigh.items()}

        for eidx in range(start_edge_idx, len(loop_edges)):
            a, b_ = loop_edges[eidx]

            Dp = D_base.copy()

            if args.perturb_mode == "remove":
                # Set tested edge to large value; optional additive inflation on adjacent loop edges
                rem = float(args.edge_remove_dist)
                Dp[a, b_] = rem
                Dp[b_, a] = rem
                if args.adjacent_eps > 0.0:
                    eps = float(args.adjacent_eps)
                    for x in adj_in_loop.get(a, []):
                        if x == b_:
                            continue
                        val = min(rem, Dp[a, x] + eps); Dp[a, x] = val; Dp[x, a] = val
                    for x in adj_in_loop.get(b_, []):
                        if x == a:
                            continue
                        val = min(rem, Dp[b_, x] + eps); Dp[b_, x] = val; Dp[x, b_] = val

            elif args.perturb_mode == "multiplicative":
                # Scale tested edge; optional multiplicative inflation on adjacent loop edges
                ef = float(args.edge_factor)
                af = float(args.adjacent_factor)
                Dp[a, b_] = Dp[a, b_] * ef
                Dp[b_, a] = Dp[b_, a] * ef
                if af != 1.0:
                    for x in adj_in_loop.get(a, []):
                        if x == b_:
                            continue
                        Dp[a, x] = Dp[a, x] * af; Dp[x, a] = Dp[x, a] * af
                    for x in adj_in_loop.get(b_, []):
                        if x == a:
                            continue
                        Dp[b_, x] = Dp[b_, x] * af; Dp[x, b_] = Dp[x, b_] * af
            else:
                raise ValueError(f"Unknown perturb_mode: {args.perturb_mode}")

            # Optional random noise to break symmetries
            if args.random_noise_std > 0.0:
                noise = np.random.normal(0, args.random_noise_std, Dp.shape).astype(np.float32)
                noise = (noise + noise.T) / 2.0  # symmetrize
                np.fill_diagonal(noise, 0.0)
                Dp += noise
                Dp = np.maximum(0.0, Dp)  # non-negative

            PHp = run_ripser_on_perturbed_dm(Dp, maxdim=int(args.max_homology_dimension), coeff=int(args.coeff))
            H1p = PHp[1] if len(PHp) > 1 else np.empty((0, 2))
            new_p = robust_loop_persistence(H1p, b, d)
            p_drop_bd_edge = float(max(0.0, base_pers - new_p))

            # Optional clean contrast (dense)
            p_drop_clean_edge = 0.0
            if args.contrast_with_clean and D_clean is not None and H1_clean is not None:
                Dc = D_clean.copy()
                if args.perturb_mode == "remove":
                    rem = float(args.edge_remove_dist)
                    Dc[a, b_] = rem; Dc[b_, a] = rem
                    if args.adjacent_eps > 0.0:
                        eps = float(args.adjacent_eps)
                        for x in adj_in_loop.get(a, []):
                            if x == b_: continue
                            val = min(rem, Dc[a, x] + eps); Dc[a, x] = val; Dc[x, a] = val
                        for x in adj_in_loop.get(b_, []):
                            if x == a: continue
                            val = min(rem, Dc[b_, x] + eps); Dc[b_, x] = val; Dc[x, b_] = val
                else:
                    ef = float(args.edge_factor); af = float(args.adjacent_factor)
                    Dc[a, b_] = Dc[a, b_] * ef; Dc[b_, a] = Dc[b_, a] * ef
                    if af != 1.0:
                        for x in adj_in_loop.get(a, []):
                            if x == b_: continue
                            Dc[a, x] = Dc[a, x] * af; Dc[x, a] = Dc[x, a] * af
                        for x in adj_in_loop.get(b_, []):
                            if x == a: continue
                            Dc[b_, x] = Dc[b_, x] * af; Dc[x, b_] = Dc[x, b_] * af

                if args.random_noise_std > 0.0:
                    noise = np.random.normal(0, args.random_noise_std, Dc.shape).astype(np.float32)
                    noise = (noise + noise.T) / 2.0
                    np.fill_diagonal(noise, 0.0)
                    Dc += noise
                    Dc = np.maximum(0.0, Dc)

                PHc = run_ripser_on_perturbed_dm(Dc, maxdim=int(args.max_homology_dimension), coeff=int(args.coeff))
                H1c = PHc[1] if len(PHc) > 1 else np.empty((0, 2))
                base_clean = nearest_persistence(H1_clean, b, d)
                new_clean  = robust_loop_persistence(H1c, b, d)
                p_drop_clean_edge = float(max(0.0, base_clean - new_clean))

            tss_edge = args.alpha * (p_drop_bd_edge - p_drop_clean_edge)

            # features (from base D)
            row_a = node_df.iloc[a]; row_b = node_df.iloc[b_]
            ln_a = str(row_a["layer_name"])
            ln_b = str(row_b["layer_name"])
            # Handle both spatial and non-spatial node tables
            if "spatial_h" in node_df.columns and "spatial_w" in node_df.columns and "channel" in node_df.columns:
                # Spatial mode: use channel column
                ch_a = int(row_a["channel"])
                ch_b = int(row_b["channel"])
            else:
                # Non-spatial mode: use local_index column
                ch_a = int(row_a["local_index"])
                ch_b = int(row_b["local_index"])
            dist = float(D_base[a, b_])
            sim  = float(1.0 - 0.5 * (dist ** 2)); sim = max(-1.0, min(1.0, sim))
            a_wg = wg_feats.get((ln_a, ch_a), {"w_out_norm": 0.0, "grad_out_norm": 0.0, "fan_in": 0, "fan_out": 0})
            b_wg = wg_feats.get((ln_b, ch_b), {"w_out_norm": 0.0, "grad_out_norm": 0.0, "fan_in": 0, "fan_out": 0})
            a_mu, a_va = act_stats.get((ln_a, ch_a), (0.0, 0.0))
            b_mu, b_va = act_stats.get((ln_b, ch_b), (0.0, 0.0))
            depth_a = _parse_layer_depth(ln_a); depth_b = _parse_layer_depth(ln_b)
            
            # Debug: Show depth calculation for first few edges
            if len(per_edge_rows) < 10:
                print(f"DEBUG: Layer depths - {ln_a}: {depth_a}, {ln_b}: {depth_b}")
                print(f"DEBUG: Layer name parts - {ln_a}: {ln_a.split('.')}, {ln_b}: {ln_b.split('.')}")
            
            # Check if depths are still zero and warn
            if depth_a == 0 or depth_b == 0:
                print(f"WARNING: Zero depth detected! {ln_a}: {depth_a}, {ln_b}: {depth_b}")
                print(f"WARNING: Layer name analysis - {ln_a}: starts_with_features={ln_a.startswith('features.')}, parts={ln_a.split('.')}")
                print(f"WARNING: Layer name analysis - {ln_b}: starts_with_features={ln_b.startswith('features.')}, parts={ln_b.split('.')}")
            
            # Enhanced features for better TSS prediction
            # 1. Layer position features
            depth_diff = abs(depth_a - depth_b)
            is_same_layer = 1.0 if ln_a == ln_b else 0.0
            
            # 2. Activation distribution features
            a_std = float(np.sqrt(a_va)) if a_va > 0 else 0.0
            b_std = float(np.sqrt(b_va)) if b_va > 0 else 0.0
            act_mean_diff = abs(a_mu - b_mu)
            act_std_ratio = a_std / (b_std + 1e-8)
            
            # 3. Weight features
            w_norm_ratio = a_wg["w_out_norm"] / (b_wg["w_out_norm"] + 1e-8)
            fan_in_ratio = a_wg["fan_in"] / (b_wg["fan_in"] + 1e-8)
            fan_out_ratio = a_wg["fan_out"] / (b_wg["fan_out"] + 1e-8)
            
            # 4. Gradient features (if available)
            grad_norm_ratio = a_wg["grad_out_norm"] / (b_wg["grad_out_norm"] + 1e-8)
            
            # 5. Topological features (from persistence diagram)
            birth_a, death_a = float(b), float(d)
            birth_b, death_b = float(b), float(d)  # Same for both nodes in the edge
            persistence = death_a - birth_a if death_a != float('inf') else 0.0
            birth_death_ratio = birth_a / (death_a + 1e-8) if death_a != float('inf') else 0.0

            per_edge_rows.append({
                "loop_rank": loop_rank,
                "loop_original_index": idx,
                "src_index": int(a),
                "src_layer": ln_a,
                "src_ch": int(ch_a),
                "dst_index": int(b_),
                "dst_layer": ln_b,
                "dst_ch": int(ch_b),
                "is_backdoored": int(1 if args.model_type == "backdoored" else 0),
                "old_persistence": float(base_pers),
                "new_persistence": float(new_p),
                "persistence_drop": p_drop_bd_edge,
                "tss": tss_edge,
                "sim": sim,
                "dist": dist,
                "w_out_norm_a": a_wg["w_out_norm"],
                "w_out_norm_b": b_wg["w_out_norm"],
                "grad_out_norm_a": a_wg["grad_out_norm"],
                "grad_out_norm_b": b_wg["grad_out_norm"],
                "act_mean_a": a_mu,
                "act_var_a": a_va,
                "act_mean_b": b_mu,
                "act_var_b": b_va,
                "fan_in_a": a_wg["fan_in"],
                "fan_out_a": a_wg["fan_out"],
                "fan_in_b": b_wg["fan_in"],
                "fan_out_b": b_wg["fan_out"],
                "depth_a": depth_a,
                "depth_b": depth_b,
                
                # Enhanced features for better TSS prediction
                # Layer position features
                "depth_diff": depth_diff,
                "is_same_layer": is_same_layer,
                
                # Activation distribution features
                "act_std_a": a_std,
                "act_std_b": b_std,
                "act_mean_diff": act_mean_diff,
                "act_std_ratio": act_std_ratio,
                
                # Weight features
                "w_norm_ratio": w_norm_ratio,
                "fan_in_ratio": fan_in_ratio,
                "fan_out_ratio": fan_out_ratio,
                
                # Gradient features
                "grad_norm_ratio": grad_norm_ratio,
                
                # Topological features
                "persistence": persistence,
                "birth_death_ratio": birth_death_ratio,
                "birth_a": birth_a,
                "death_a": death_a,
            })

            processed_edges_global += 1
            pbar.update(1)

            # periodic checkpoint
            if args.save_every_edges > 0 and (processed_edges_global % args.save_every_edges == 0):
                df_e = pd.DataFrame(per_edge_rows)
                df_e.sort_values(["loop_rank", "tss"], ascending=[True, False], inplace=True)
                df_e.to_csv(edges_progress_csv, index=False)
                with open(progress_path, "w") as f:
                    json.dump({"edges_done": processed_edges_global}, f, indent=2)
                ascii_print(f"[Checkpoint] Saved edges_done={processed_edges_global}")

        # save loop metadata
        all_results.append({
            "loop_rank": int(loop_rank),
            "loop_original_index": int(idx),
            "loop_birth": float(b),
            "loop_death": float(d),
            "baseline_persistence": float(base_pers),
            "loop_nodes": [int(x) for x in loop_nodes],
            "loop_edges": [f"({u},{v})" for (u, v) in loop_edges],
        })
        with open(loops_progress_json, "w") as f:
            json.dump(all_results, f, indent=2)

    pbar.close()

    # outputs
    if per_edge_rows:
        df_e = pd.DataFrame(per_edge_rows)
        df_e.sort_values(["loop_rank", "tss"], ascending=[True, False], inplace=True)
        df_e.to_csv(out_dir / "tss_per_edge.csv", index=False)

    cfg = {
        "alpha": args.alpha,
        "beta": args.beta,
        "num_top_loops": args.num_top_loops,
        "top_k": args.top_k,
        "seed_persistence": args.seed_persistence,
        "seed_asr": args.seed_asr,
        "model_type": args.model_type,
        "input_type": args.input_type,
        "correlation_method": args.correlation_method,
        "num_nodes": int(N),
        "seed": args.seed,
        "max_homology_dimension": int(args.max_homology_dimension),
        "coeff": int(args.coeff),
        "edges_only": True,
        "num_grad_batches": int(args.num_grad_batches),
        "total_edges_selected": int(total_edges_all),
        "total_unique_edges_selected": int(total_unique_edges_all),
        "save_every_edges": int(args.save_every_edges),
        "perturb_mode": str(args.perturb_mode),
        "edge_remove_dist": float(args.edge_remove_dist),
        "adjacent_eps": float(args.adjacent_eps),
        "edge_factor": float(args.edge_factor),
        "adjacent_factor": float(args.adjacent_factor),
        "random_noise_std": float(args.random_noise_std),
    }
    with open(out_dir / "tss_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    with open(progress_path, "w") as f:
        json.dump({"edges_done": processed_edges_global}, f, indent=2)
    # Final depth statistics
    if per_edge_rows:
        df_results = pd.DataFrame(per_edge_rows)
        zero_depth_a = (df_results['depth_a'] == 0).sum()
        zero_depth_b = (df_results['depth_b'] == 0).sum()
        total_edges = len(df_results)
        print(f"DEBUG: Final depth statistics:")
        print(f"  Total edges: {total_edges}")
        print(f"  Edges with zero depth_a: {zero_depth_a} ({zero_depth_a/total_edges*100:.1f}%)")
        print(f"  Edges with zero depth_b: {zero_depth_b} ({zero_depth_b/total_edges*100:.1f}%)")
        print(f"  Non-zero depth_a range: {df_results['depth_a'].min()} - {df_results['depth_a'].max()}")
        print(f"  Non-zero depth_b range: {df_results['depth_b'].min()} - {df_results['depth_b'].max()}")
    
    ascii_print("TSS (edge-only, dense) done.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Topological Susceptibility Scores (Edge-only, dense VR).")
    parser.add_argument("--ph_output_base_dir", type=str, default="./ph_output")
    parser.add_argument("--correlation_output_base_dir", type=str, default="./correlation_output")
    parser.add_argument("--model_type", type=str, required=True, choices=["clean", "backdoored"])
    parser.add_argument("--input_type", type=str, required=True, choices=["clean", "triggered"])
    parser.add_argument("--correlation_method", type=str, default="pearson", choices=["pearson", "cos"])

    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--architecture", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet152", "mobilenet_v2", "efficientnet_b0"],
                        help="Model architecture used for training the model.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--poison_ratio_eval", type=float, default=1.0)
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_location", type=str, default="br")
    parser.add_argument("--trigger_value", type=float, default=1.0)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")

    parser.add_argument("--max_homology_dimension", type=int, default=1)
    parser.add_argument("--coeff", type=int, default=2)

    parser.add_argument("--num_top_loops", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta",  type=float, default=1.0)

    parser.add_argument("--seed_persistence", type=int, default=6)
    parser.add_argument("--seed_asr",         type=int, default=6)

    parser.add_argument("--run_sanity_checks", action="store_true")
    parser.add_argument("--print_hook_fires", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--contrast_with_clean", action="store_true")
    parser.add_argument("--clean_model_type", type=str, default="clean")
    parser.add_argument("--clean_input_type", type=str, default="clean")

    parser.add_argument("--activations_dir", type=str, default="./activation_output_topo")
    parser.add_argument("--num_grad_batches", type=int, default=0)

    # Progress
    parser.add_argument("--save_every_edges", type=int, default=10)
    parser.add_argument("--resume_from_progress", action="store_true", default=True)

    # Dense perturbation controls
    parser.add_argument("--perturb_mode", type=str, default="remove", choices=["remove", "multiplicative"],
                        help="remove: set tested edge to large distance (+adjacent eps); multiplicative: scale tested and adjacent edges.")
    parser.add_argument("--edge_remove_dist", type=float, default=3.0,
                        help="Distance when removing an edge (must exceed chord metric max=2.0).")
    parser.add_argument("--adjacent_eps", type=float, default=0.0,
                        help="Add epsilon to edges adjacent (in loop) to the tested edge (remove mode).")
    parser.add_argument("--edge_factor", type=float, default=2.0,
                        help="Scale factor for the tested edge (multiplicative mode).")
    parser.add_argument("--adjacent_factor", type=float, default=1.1,
                        help="Scale factor for loop-adjacent edges to the tested edge (multiplicative mode).")
    parser.add_argument("--random_noise_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to Dp per test (breaks symmetries; try 0.01).")

    args = parser.parse_args()
    main(args)