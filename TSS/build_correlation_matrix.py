#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build channel-level correlation matrices from activations extracted by extract_activation_improved.py

- Detects layers from activation metadata / layer_catalog (works across tap modes)
- Optionally drops fc and downsample convs
- Concatenates channels across layers -> global matrix
- Computes similarity (pearson/cos) and PH-friendly distance
- Builds a k-NN graph (symmetrized) and saves as edge list
- Saves full node mapping + sampling indices for reproducibility

Outputs (under output_base_dir/<prefix>/):
  - <method>_corr.npy                (float32, NxN similarity)
  - distance_sqeuclid.npy            (float32, NxN distance: sqrt(2*(1-S)))
  - knn_edges.csv                    (src,dst,sim,dist) undirected, symmetrized
  - node_table.csv                   (new_index,layer_name,local_index,orig_global_index)
  - layer_offsets.json               ({layer_name: {"start":i,"end":j,"count":c}})
  - sampled_indices.npy              (int64, original global indices kept, length=N)
  - graph_config.json                (params + provenance)
  - neuron_labels.json               ({new_index: "layer_n<local>"})
  - neuron_index_map.json            ({new_index: {...}})
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict, defaultdict

# Your project utils
from topo_utils import mat_pearson_adjacency, mat_cos_adjacency

# --------------------------
# Helpers
# --------------------------
def ascii_print(s):  # Windows-safe logging to console
    try:
        print(s)
    except Exception:
        print(s.encode("ascii", "ignore").decode("ascii"))

def load_metadata(prefix_dir, file_prefix):
    meta_path = os.path.join(prefix_dir, f"{file_prefix}_activation_metadata.json")
    catalog_path = os.path.join(prefix_dir, f"{file_prefix}_layer_catalog.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing activation metadata JSON: {meta_path}")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Missing layer catalog CSV: {catalog_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    catalog = pd.read_csv(catalog_path)
    return meta, catalog

def filter_layers(layer_names, drop_fc=False, drop_downsample=False):
    filtered = []
    for name in layer_names:
        if drop_fc and name == "fc":
            continue
        if drop_downsample and ".downsample.0" in name:
            continue
        filtered.append(name)
    return filtered

def build_layer_offsets(layer_names, layer_counts):
    """Return dict layer->(start,end,count) and total count."""
    offsets = {}
    cursor = 0
    for name, cnt in zip(layer_names, layer_counts):
        cnt = int(cnt)
        offsets[name] = {"start": cursor, "end": cursor + cnt, "count": cnt}
        cursor += cnt
    return offsets, cursor

def stratified_sample_indices(layer_counts, total_count, n_keep, rng):
    """
    Stratified sampling: proportional by layer channel counts,
    returns sorted global indices (0..total_count-1)
    """
    if n_keep is None or n_keep >= total_count:
        return np.arange(total_count, dtype=np.int64)

    # proportional allocation
    alloc = [int(n_keep * c / sum(layer_counts)) for c in layer_counts]
    # fix rounding to hit exactly n_keep
    deficit = n_keep - sum(alloc)
    # greedily add remaining to largest layers
    order = np.argsort(layer_counts)[::-1]
    i = 0
    while deficit > 0:
        alloc[order[i % len(order)]] += 1
        i += 1
        deficit -= 1

    # sample within each layer range
    starts = np.cumsum([0] + layer_counts[:-1]).astype(int)
    chosen = []
    for start, cnt, k in zip(starts, layer_counts, alloc):
        if k <= 0:
            continue
        chosen.extend(start + rng.choice(cnt, size=k, replace=False))
    chosen = np.array(sorted(chosen), dtype=np.int64)
    return chosen

def build_knn_edges(sim_mat: np.ndarray, k: int):
    """
    Build undirected k-NN graph from similarity matrix.
    Returns list of (src,dst,sim,dist) with src<dst unique edges.
    dist = sqrt(2*(1 - sim_clipped))
    """
    n = sim_mat.shape[0]
    sim = sim_mat.copy()
    np.fill_diagonal(sim, -np.inf)  # exclude self

    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.clip(sim_mat, -1.0, 1.0)))).astype(np.float32)

    edges = {}
    for i in range(n):
        # top-k neighbors by similarity
        neigh = np.argpartition(sim[i], -k)[-k:]
        # sort them (optional)
        neigh = neigh[np.argsort(-sim[i, neigh])]
        for j in neigh:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            w = float(sim_mat[i, j])
            d = float(D[i, j])
            # keep the stronger similarity if duplicate appears
            if key not in edges or w > edges[key][0]:
                edges[key] = (w, d)

    # convert dict -> list
    out = [(a, b, w, d) for (a, b), (w, d) in edges.items()]
    return out, D

def compute_spatial_aware_correlation(loaded_activations, layer_names, method="pearson"):
    """
    Compute spatial-aware correlations for 4D tensors.
    For each spatial location (h,w), compute correlations across channels and samples.
    Returns correlation matrix and spatial location mapping.
    """
    spatial_locations = []  # (layer_name, h, w, channel)
    spatial_activations = []  # [N, spatial_locations]
    
    # Use actual loaded layer names instead of catalog layer names
    actual_layer_names = list(loaded_activations.keys())
    print(f"Computing spatial locations for {len(actual_layer_names)} layers...")
    
    for ln in actual_layer_names:
        X = loaded_activations[ln]
        if X.dim() == 4:
            N, C, H, W = X.shape
            print(f"  {ln}: {X.shape} -> {C * H * W} spatial locations")
            # For each spatial location, collect activations across channels
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        spatial_locations.append((ln, h, w, c))
                        spatial_activations.append(X[:, c, h, w].numpy())  # [N]
        elif X.dim() == 2:
            # For 2D tensors (like fc), treat as single spatial location
            N, F = X.shape
            print(f"  {ln}: {X.shape} -> {F} spatial locations")
            for f in range(F):
                spatial_locations.append((ln, 0, 0, f))
                spatial_activations.append(X[:, f].numpy())  # [N]
    
    print(f"Total spatial locations: {len(spatial_locations)}")
    
    # Convert to matrix: [spatial_locations, N]
    print("Converting to spatial matrix...")
    spatial_matrix = np.array(spatial_activations).T  # [N, spatial_locations]
    spatial_matrix = spatial_matrix.T  # [spatial_locations, N]
    
    print(f"Spatial matrix shape: {spatial_matrix.shape}")
    
    # Check if matrix is too large and suggest sampling
    if spatial_matrix.shape[0] > 20000:
        print(f"WARNING: Large correlation matrix ({spatial_matrix.shape[0]}x{spatial_matrix.shape[0]}) will be slow to compute!")
        print("Consider using --max_spatial_nodes to limit the number of spatial locations.")
    
    # Compute correlations
    print(f"Computing {method} correlations...")
    if method == "pearson":
        S = mat_pearson_adjacency(torch.from_numpy(spatial_matrix))
    elif method == "cos":
        S = mat_cos_adjacency(torch.from_numpy(spatial_matrix))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("Correlation computation completed!")
    return S.detach().cpu().numpy().astype(np.float32), spatial_locations

def build_spatial_layer_offsets(spatial_locations, layer_names=None):
    """Build offsets for spatial locations grouped by layer."""
    offsets = {}
    cursor = 0
    
    # Get unique layer names from spatial locations
    if layer_names is None:
        layer_names = list(set([layer_name for layer_name, h, w, c in spatial_locations]))
    
    for ln in layer_names:
        # Count spatial locations for this layer
        layer_locations = [i for i, (layer_name, h, w, c) in enumerate(spatial_locations) if layer_name == ln]
        count = len(layer_locations)
        offsets[ln] = {"start": cursor, "end": cursor + count, "count": count}
        cursor += count
    
    return offsets, cursor

# --------------------------
# Main routine
# --------------------------
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prefix_dir = args.activations_dir
    file_prefix = f"{args.model_type}_model_{args.input_type}_inputs"
    ascii_print(f"Reading activation bundle: {os.path.join(prefix_dir, file_prefix)}_*")

    meta, catalog = load_metadata(prefix_dir, file_prefix)
    tap_mode = meta.get("tap_mode", "unknown")
    ascii_print(f"tap_mode={tap_mode} | include_fc={meta.get('include_fc', False)}")

    # determine layer order from catalog (authoritative)
    layer_names = catalog["layer_name"].tolist()
    layer_counts = catalog["n_neurons"].astype(int).tolist()

    # allow dropping fc / downsample by flag
    layer_names = filter_layers(layer_names,
                                drop_fc=args.drop_fc,
                                drop_downsample=args.drop_downsample)
    # keep counts in the same filtered order
    name2cnt = dict(zip(catalog["layer_name"], catalog["n_neurons"].astype(int)))
    layer_counts = [int(name2cnt[n]) for n in layer_names]

    # Check if we should use spatial-aware correlation
    use_spatial_aware = (tap_mode == "spatial_preserve")
    
    if use_spatial_aware:
        ascii_print("Using spatial-aware correlation mode (preserving spatial structure)")
        
        # Load activations without flattening
        loaded = OrderedDict()
        for ln in layer_names:
            fpath = os.path.join(prefix_dir, f"{file_prefix}_{ln}.pt")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing activation file: {fpath}")
            X = torch.load(fpath, map_location="cpu")
            if not isinstance(X, torch.Tensor):
                raise ValueError(f"{ln}: expected torch.Tensor, got {type(X)}")
            loaded[ln] = X
            ascii_print(f"  loaded {ln}: {tuple(X.shape)} (spatial-aware mode)")
        
        # For spatial mode, limit nodes BEFORE correlation computation if requested
        if args.max_spatial_nodes is not None:
            ascii_print(f"Pre-sampling to {args.max_spatial_nodes} spatial locations before correlation computation...")
            # First, compute spatial locations to get the total count
            temp_spatial_locations = []
            for ln in loaded.keys():
                X = loaded[ln]
                if X.dim() == 4:
                    N, C, H, W = X.shape
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                temp_spatial_locations.append((ln, h, w, c))
                elif X.dim() == 2:
                    N, F = X.shape
                    for f in range(F):
                        temp_spatial_locations.append((ln, 0, 0, f))
            
            total_spatial_locations = len(temp_spatial_locations)
            ascii_print(f"Total available spatial locations: {total_spatial_locations}")
            
            if args.max_spatial_nodes < total_spatial_locations:
                # Sample spatial locations before correlation computation
                rng = np.random.default_rng(args.seed)
                sample_indices = rng.choice(total_spatial_locations, size=args.max_spatial_nodes, replace=False)
                sampled_spatial_locations = [temp_spatial_locations[i] for i in sample_indices]
                ascii_print(f"Sampled {len(sampled_spatial_locations)} spatial locations")
                
                # Create sampled activations
                sampled_activations = []
                for ln, h, w, c in sampled_spatial_locations:
                    X = loaded[ln]
                    if X.dim() == 4:
                        sampled_activations.append(X[:, c, h, w].numpy())
                    elif X.dim() == 2:
                        sampled_activations.append(X[:, c].numpy())
                
                # Convert to matrix
                spatial_matrix = np.array(sampled_activations).T  # [N, sampled_locations]
                spatial_matrix = spatial_matrix.T  # [sampled_locations, N]
                
                ascii_print(f"Sampled spatial matrix shape: {spatial_matrix.shape}")
                
                # Compute correlations on sampled data
                ascii_print(f"Computing {args.method} correlations on sampled data...")
                if args.method == "pearson":
                    S = mat_pearson_adjacency(torch.from_numpy(spatial_matrix))
                elif args.method == "cos":
                    S = mat_cos_adjacency(torch.from_numpy(spatial_matrix))
                else:
                    raise ValueError(f"Unknown method: {args.method}")
                S = S.detach().cpu().numpy().astype(np.float32)
                
                spatial_locations = sampled_spatial_locations
                sample_indices_global = np.arange(len(sampled_spatial_locations), dtype=np.int64)
            else:
                # Use all spatial locations
                ascii_print("Using all spatial locations...")
                S, spatial_locations = compute_spatial_aware_correlation(loaded, layer_names, args.method)
                sample_indices_global = np.arange(len(spatial_locations), dtype=np.int64)
        else:
            # No pre-sampling, compute on all data
            ascii_print("Computing spatial-aware correlations on all data...")
            S, spatial_locations = compute_spatial_aware_correlation(loaded, layer_names, args.method)
            sample_indices_global = np.arange(len(spatial_locations), dtype=np.int64)
        
        ascii_print(f"Computed spatial-aware correlation matrix: {S.shape}")
        
        # Build spatial layer offsets
        offsets, total_C = build_spatial_layer_offsets(spatial_locations)
        ascii_print(f"Total spatial locations across all layers: {total_C}")
        
        # Build node table for spatial locations
        node_rows = []
        neuron_labels = {}
        neuron_index_map = {}
        for new_idx, (ln, h, w, c) in enumerate(spatial_locations):
            node_rows.append({
                "new_index": new_idx,
                "layer_name": ln,
                "spatial_h": int(h),
                "spatial_w": int(w),
                "channel": int(c),
                "orig_global_index": int(new_idx)
            })
            neuron_labels[str(new_idx)] = f"{ln}_h{h}_w{w}_c{c}"
            neuron_index_map[str(new_idx)] = {
                "layer_name": ln,
                "spatial_h": int(h),
                "spatial_w": int(w),
                "channel": int(c),
                "type": "spatial_location",
                "original_global_index": int(new_idx)
            }
    
    else:
        # Original flattening approach for non-spatial modes
        loaded = OrderedDict()
        actual_layer_counts = []  # Track actual flattened dimensions
        
        for ln in layer_names:
            fpath = os.path.join(prefix_dir, f"{file_prefix}_{ln}.pt")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing activation file: {fpath}")
            X = torch.load(fpath, map_location="cpu")
            if not isinstance(X, torch.Tensor):
                raise ValueError(f"{ln}: expected torch.Tensor, got {type(X)}")
            
            # Handle both 2D (GAP modes) and 4D (spatial_preserve mode) tensors
            if X.dim() == 2:
                # 2D tensor [N, C] - from GAP modes (topotroj_compat, toap_block_out, bn2_legacy)
                loaded[ln] = X  # [N, C]
                actual_layer_counts.append(X.shape[1])  # C channels
                ascii_print(f"  loaded {ln}: {tuple(X.shape)} (2D - GAP mode)")
            elif X.dim() == 4:
                # 4D tensor [N, C, H, W] - from spatial_preserve mode
                # Flatten spatial dimensions: [N, C, H, W] -> [N, C*H*W]
                N, C, H, W = X.shape
                X_flat = X.view(N, C * H * W)  # Flatten spatial dimensions
                loaded[ln] = X_flat  # [N, C*H*W]
                actual_layer_counts.append(C * H * W)  # C*H*W flattened features
                ascii_print(f"  loaded {ln}: {tuple(X.shape)} -> flattened to {tuple(X_flat.shape)} (4D - flattened mode)")
            else:
                raise ValueError(f"{ln}: expected 2D [N,C] or 4D [N,C,H,W] torch.Tensor, got shape {X.shape}")

        # build layer offsets BEFORE sampling (for provenance) using actual counts
        offsets, total_C = build_layer_offsets(layer_names, actual_layer_counts)

        # concatenate channels across layers -> [N, sumC] then transpose -> [sumC, N]
        X_cols = []
        for ln, X in loaded.items():
            X_cols.append(X)  # [N, C]
        X_all = torch.cat(X_cols, dim=1)      # [N, sumC]
        neural_act = X_all.T.contiguous().float()  # [sumC, N]
        del X_all

        # stratified sampling across layers if requested
        rng = np.random.default_rng(args.seed)
        sample_indices_global = stratified_sample_indices(actual_layer_counts, total_C, args.n_sample_neurons, rng)
        neural_act = neural_act[sample_indices_global, :]  # [K, N], rows are sampled neurons

        # Build node table / mappings for the sampled set
        # Map from global index -> (layer, local)
        global_to_pair = {}
        for ln in layer_names:
            start, end = offsets[ln]["start"], offsets[ln]["end"]
            for local in range(end - start):
                global_to_pair[start + local] = (ln, local)

        node_rows = []
        neuron_labels = {}
        neuron_index_map = {}
        for new_idx, orig_idx in enumerate(sample_indices_global.tolist()):
            ln, local = global_to_pair[int(orig_idx)]
            node_rows.append({
                "new_index": new_idx,
                "layer_name": ln,
                "local_index": int(local),
                "orig_global_index": int(orig_idx)
            })
            neuron_labels[str(new_idx)] = f"{ln}_n{local}"
            neuron_index_map[str(new_idx)] = {
                "layer_name": ln,
                "local_index": int(local),
                "type": "channel",
                "original_global_index": int(orig_idx)
            }

        # Compute similarity
        if args.method == "pearson":
            S = mat_pearson_adjacency(neural_act)  # expects [neurons, samples]
        elif args.method == "cos":
            S = mat_cos_adjacency(neural_act)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        S = S.detach().cpu().numpy().astype(np.float32)  # [K, K]

    # Build kNN edges + PH distance
    edges, D = build_knn_edges(S, k=args.knn_k)  # edges list + full distance

    # Prepare output dir
    out_dir = os.path.join(args.output_base_dir, file_prefix)
    os.makedirs(out_dir, exist_ok=True)
    ascii_print(f"Output dir: {out_dir}")

    # Save matrices
    if args.save_similarity:
        np.save(os.path.join(out_dir, f"{args.method}_corr.npy"), S)
        ascii_print(f"Saved similarity matrix: {args.method}_corr.npy  shape={S.shape}")
    if args.save_dense_distance:
        np.save(os.path.join(out_dir, "distance_sqeuclid.npy"), D)
        ascii_print(f"Saved PH distance matrix: distance_sqeuclid.npy  shape={D.shape}")

    # Save kNN edges (undirected, symmetrized)
    edges_df = pd.DataFrame(edges, columns=["src", "dst", "sim", "dist"])
    # Rectify similarity to [0,1] for PH, then threshold
    edges_df["w"] = np.clip(edges_df["sim"], 0.0, 1.0)
    edges_df = edges_df[edges_df["w"] >= args.min_w].copy()
    edges_df["tau"] = 1.0 - edges_df["w"]
    edges_df.to_csv(os.path.join(out_dir, "knn_edges.csv"), index=False)
    ascii_print(f"Saved k-NN edges: knn_edges.csv  rows={len(edges_df)} (k={args.knn_k})")

    # Save node table / labels / maps
    if use_spatial_aware:
        # For spatial mode, include spatial coordinates
        nodes_df = pd.DataFrame(node_rows, columns=["new_index", "layer_name", "spatial_h", "spatial_w", "channel", "orig_global_index"])
        nodes_df.to_csv(os.path.join(out_dir, "node_table.csv"), index=False)
        ascii_print(f"Saved spatial node table: node_table.csv  rows={len(nodes_df)}")

        # Map src/dst indices -> (layer_name, spatial_h, spatial_w, channel)
        left  = nodes_df.rename(columns={"new_index":"src", "layer_name":"src_layer", "spatial_h":"src_h", "spatial_w":"src_w", "channel":"src_ch"})
        right = nodes_df.rename(columns={"new_index":"dst", "layer_name":"dst_layer", "spatial_h":"dst_h", "spatial_w":"dst_w", "channel":"dst_ch"})
        g_sem = edges_df.merge(left[["src","src_layer","src_h","src_w","src_ch"]], on="src") \
                        .merge(right[["dst","dst_layer","dst_h","dst_w","dst_ch"]], on="dst")

        graph_cols = ["src_layer","src_h","src_w","src_ch","dst_layer","dst_h","dst_w","dst_ch","w","tau"]
        graph_df = g_sem[graph_cols].copy()
    else:
        # For non-spatial mode, use original format
        nodes_df = pd.DataFrame(node_rows, columns=["new_index", "layer_name", "local_index", "orig_global_index"])
        nodes_df.to_csv(os.path.join(out_dir, "node_table.csv"), index=False)
        ascii_print(f"Saved node table: node_table.csv  rows={len(nodes_df)}")

        # Map src/dst indices -> (layer_name, local_index)
        left  = nodes_df.rename(columns={"new_index":"src", "layer_name":"src_layer", "local_index":"src_ch"})
        right = nodes_df.rename(columns={"new_index":"dst", "layer_name":"dst_layer", "local_index":"dst_ch"})
        g_sem = edges_df.merge(left[["src","src_layer","src_ch"]], on="src") \
                        .merge(right[["dst","dst_layer","dst_ch"]], on="dst")

        graph_cols = ["src_layer","src_ch","dst_layer","dst_ch","w","tau"]
        graph_df = g_sem[graph_cols].copy()

    # Decide filename based on (model_type, input_type)
    if args.model_type=="clean" and args.input_type=="clean":
        graph_name = "graph_clean.csv"
    elif args.model_type=="backdoored" and args.input_type=="triggered":
        graph_name = "graph_backdoor.csv"
    else:
        graph_name = f"graph_{args.model_type}_{args.input_type}.csv"

    graph_path = os.path.join(out_dir, graph_name)
    graph_df.to_csv(graph_path, index=False)
    ascii_print(f"Saved PH graph: {graph_name}  rows={len(graph_df)}")

    with open(os.path.join(out_dir, "neuron_labels.json"), "w") as f:
        json.dump(neuron_labels, f, indent=2)
    with open(os.path.join(out_dir, "neuron_index_map.json"), "w") as f:
        json.dump(neuron_index_map, f, indent=2)

    # Save layer offsets (for reproducibility / later masking)
    with open(os.path.join(out_dir, "layer_offsets.json"), "w") as f:
        json.dump(offsets, f, indent=2)

    # Save which original global indices we kept (sampling)
    np.save(os.path.join(out_dir, "sampled_indices.npy"), sample_indices_global)

    # Save config / provenance
    if use_spatial_aware:
        # For spatial mode, use the actual number of spatial locations
        n_sample_neurons = total_C
        requested_n_sample_neurons = args.max_spatial_nodes
    else:
        # For non-spatial mode, use neural_act shape
        n_sample_neurons = int(neural_act.shape[0])
        requested_n_sample_neurons = args.n_sample_neurons
    
    config = {
        "method": args.method,
        "knn_k": args.knn_k,
        "n_sample_neurons": n_sample_neurons,
        "requested_n_sample_neurons": requested_n_sample_neurons,
        "drop_fc": args.drop_fc,
        "drop_downsample": args.drop_downsample,
        "tap_mode": tap_mode,
        "activations_dir": prefix_dir,
        "prefix": file_prefix,
        "seed": args.seed,
        "save_similarity": args.save_similarity,
        "save_dense_distance": args.save_dense_distance
    }
    with open(os.path.join(out_dir, "graph_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    ascii_print("Saved graph_config.json")

    ascii_print("All done ✅")

# --------------------------
# CLI
'''
python build_correlation_matrix.py --activations_dir "activation_output_topo" --model_type clean --input_type clean --method pearson --knn_k 8 --min_w 0.05 --seed 42 --save_similarity --save_dense_distance
python build_correlation_matrix.py --activations_dir activation_output_topo --model_type backdoored --input_type triggered --method pearson --knn_k 8 --min_w 0.05 --seed 42 --save_similarity --save_dense_distance
'''


# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build global channel correlation + kNN graph for TOAP/TopoTroj.")
    parser.add_argument("--activations_dir", type=str, required=True,
                        help="Directory where extractor saved outputs (the *_inputs_*.pt files and metadata).")
    parser.add_argument("--model_type", type=str, required=True, choices=["clean", "backdoored"])
    parser.add_argument("--input_type", type=str, required=True, choices=["clean", "triggered"])
    parser.add_argument("--output_base_dir", type=str, default="./correlation_output",
                        help="Base directory to save outputs.")
    parser.add_argument("--method", type=str, default="pearson", choices=["pearson", "cos"])
    parser.add_argument("--n_sample_neurons", type=int, default=None,
                        help="If set, stratified sample to this many neurons; default keeps all.")
    parser.add_argument("--max_spatial_nodes", type=int, default=None,
                        help="If set, limit total number of spatial locations (nodes) in correlation matrix. Useful for spatial_preserve mode to control matrix size.")
    parser.add_argument("--knn_k", type=int, default=8, help="k for k-NN graph (ensure cycles for PH).")
    parser.add_argument("--drop_fc", action="store_true", help="Exclude 'fc' layer from graph.")
    parser.add_argument("--drop_downsample", action="store_true", help="Exclude '.downsample.0' convs.")
    parser.add_argument("--save_similarity", action="store_true", help="Save full similarity matrix (NxN).")
    parser.add_argument("--save_dense_distance", action="store_true", help="Save dense distance matrix (NxN).")
    parser.add_argument("--seed", type=int, default=42)
    # in the CLI section
    parser.add_argument("--min_w", type=float, default=0.05, help="Minimum edge weight after rectification for PH.")


    args = parser.parse_args()
    main(args)
