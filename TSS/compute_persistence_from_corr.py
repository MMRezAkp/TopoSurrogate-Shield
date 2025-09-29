#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute persistent homology for TOAP from correlation outputs (robust version).

Defaults:
- Runs PH on a **dense, finite metric** (distance_sqeuclid.npy if present).
- Never uses k-NN unless --use_knn_sparse is explicitly set.

Safety / Robustness:
- If using k-NN, you can convert to a **proper metric** via APSP
  (--knn_apsp_metric) before PH to avoid spurious essential H1 bars.
- Writes barcodes_summary.json with finite/inf counts and quick stats.
- Exports h1_finite_indices.json (original indices of finite H1 bars) so TSS
  can skip essential bars automatically.

Inputs (under: <correlation_base_dir>/<model_type>_model_<input_type>_inputs/):
  - distance_sqeuclid.npy (preferred)
  - pearson_corr.npy / cos_corr.npy (fallback to build distance)
  - knn_edges.csv (only if --use_knn_sparse)
  - node_table.csv (for cocycle -> (layer,channel) mapping)
  - graph_config.json (optional provenance)

Outputs (under: <output_base_dir>/<model_type>_model_<input_type>_inputs/):
  - persistence_diagram.npy, h0_persistence.npy, h1_persistence.npy
  - barcodes.json, barcodes_summary.json, h1_topK.json, h1_finite_indices.json
  - h1_cocycles_mapped.json (if --do_cocycles)
  - H0/H1 plots (if --save_plots)
  - tda_config.json (provenance)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from ripser import Rips
import persim

# Optional helpers
try:
    from topological_feature_extractor import getGreedyPerm, getApproxSparseDM, calc_topo_feature
    HAS_TFE = True
except Exception:
    HAS_TFE = False


# --------------------------
# Helpers
# --------------------------
def ascii_print(s):
    try:
        print(s)
    except Exception:
        print(s.encode("ascii", "ignore").decode("ascii"))


def load_distance_or_build(bundle_dir, method="pearson"):
    """Prefer precomputed PH distance; fallback to building from similarity."""
    dist_path = os.path.join(bundle_dir, "distance_sqeuclid.npy")
    sim_path  = os.path.join(bundle_dir, f"{method}_corr.npy")
    if os.path.exists(dist_path):
        D = np.load(dist_path).astype(np.float32)
        return D, f"loaded:{os.path.basename(dist_path)}"
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Neither distance_sqeuclid.npy nor {method}_corr.npy found in {bundle_dir}")
    S = np.load(sim_path).astype(np.float32)
    S = np.clip(S, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - S))).astype(np.float32)  # chord metric
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D, f"built_from:{os.path.basename(sim_path)}"


def build_sparse_distance_from_knn(bundle_dir, n_nodes=None):
    """
    Build symmetric CSR distance from knn_edges.csv with columns: src,dst,sim,dist.
    """
    edges_path = os.path.join(bundle_dir, "knn_edges.csv")
    nodes_path = os.path.join(bundle_dir, "node_table.csv")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"knn_edges.csv not found in {bundle_dir}")
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"node_table.csv not found in {bundle_dir}")

    nodes_df = pd.read_csv(nodes_path)
    N = int(len(nodes_df)) if n_nodes is None else int(n_nodes)

    df = pd.read_csv(edges_path)
    src  = df["src"].astype(int).to_numpy()
    dst  = df["dst"].astype(int).to_numpy()
    dist = df["dist"].astype(np.float32).to_numpy()

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([dist, dist])

    # zero diag
    diag = np.arange(N, dtype=int)
    rows = np.concatenate([rows, diag])
    cols = np.concatenate([cols, diag])
    data = np.concatenate([data, np.zeros(N, dtype=np.float32)])

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def compute_barcodes(diagrams):
    out = {}
    for dim, diag in enumerate(diagrams):
        if diag is None or len(diag) == 0:
            out[f"H{dim}"] = []
            continue
        bars = []
        for b, d in diag:
            # np.inf possible if metric/sparsity causes essential features
            persistence = float(d - b) if np.isfinite(d) else float("inf")
            bars.append({"birth": float(b), "death": float(d), "persistence": persistence})
        out[f"H{dim}"] = bars
    return out


def summarize_barcodes(bars, distance_note):
    """
    Return a compact summary including finite/inf counts and simple stats.
    """
    def stats(arr):
        if len(arr) == 0:
            return {"count": 0, "min": None, "max": None, "mean": None}
        a = np.array(arr, dtype=float)
        return {"count": int(a.size), "min": float(np.min(a)), "max": float(np.max(a)), "mean": float(np.mean(a))}
    H0 = bars.get("H0", [])
    H1 = bars.get("H1", [])
    H1_pers = [x["persistence"] for x in H1 if np.isfinite(x["persistence"])]
    H1_inf  = [x for x in H1 if not np.isfinite(x["persistence"])]
    return {
        "distance_source": distance_note,
        "H0": {"num": len(H0)},
        "H1": {
            "num": len(H1),
            "num_finite": len(H1_pers),
            "num_infinite": len(H1_inf),
            "persistence_stats_finite": stats(H1_pers),
        },
        "has_infinite_bars": len(H1_inf) > 0
    }


def map_cocycles(cocycles_dim1, node_table_df):
    """
    Map H1 cocycles to layer/channel metadata.
    Handles both spatial and non-spatial node tables.
    """
    # Check if this is a spatial node table (has spatial_h, spatial_w, channel)
    # or a regular node table (has local_index)
    if "spatial_h" in node_table_df.columns and "spatial_w" in node_table_df.columns and "channel" in node_table_df.columns:
        # Spatial mode: use spatial coordinates
        idx_to_meta = {
            int(r["new_index"]): {
                "layer_name": r["layer_name"],
                "spatial_h": int(r["spatial_h"]),
                "spatial_w": int(r["spatial_w"]),
                "channel": int(r["channel"]),
                "orig_global_index": int(r["orig_global_index"]),
            }
            for _, r in node_table_df.iterrows()
        }
    else:
        # Non-spatial mode: use local_index
        idx_to_meta = {
            int(r["new_index"]): {
                "layer_name": r["layer_name"],
                "local_index": int(r["local_index"]),
                "orig_global_index": int(r["orig_global_index"]),
            }
            for _, r in node_table_df.iterrows()
        }

    mapped = []
    for cyc in cocycles_dim1:
        cyc = np.asarray(cyc)
        edges = []
        for entry in cyc:
            if len(entry) < 3:
                i, j = int(entry[0]), int(entry[1])
                coef = 1.0
            else:
                i, j, coef = int(entry[0]), int(entry[1]), float(entry[2])
            edges.append({
                "i": i, "j": j, "coef": coef,
                "i_meta": idx_to_meta.get(i, {}),
                "j_meta": idx_to_meta.get(j, {}),
            })
        mapped.append(edges)
    return mapped


# --------------------------
# Main
# --------------------------
def main(args):
    bundle_dir = os.path.join(args.correlation_base_dir, f"{args.model_type}_model_{args.input_type}_inputs")
    out_dir    = os.path.join(args.output_base_dir,       f"{args.model_type}_model_{args.input_type}_inputs")
    os.makedirs(out_dir, exist_ok=True)

    ascii_print(f"=== PH for: model={args.model_type} | input={args.input_type} ===")
    ascii_print(f"Reading bundle: {bundle_dir}")
    ascii_print(f"Writing to:     {out_dir}")

    # node table is required for mapping cocycles
    node_table_path = os.path.join(bundle_dir, "node_table.csv")
    if not os.path.exists(node_table_path):
        raise FileNotFoundError(f"node_table.csv not found in {bundle_dir}")
    node_df = pd.read_csv(node_table_path)
    N_nodes = int(len(node_df))
    ascii_print(f"Nodes: {N_nodes}")

    # ---- Distance selection (dense by default) ----
    D_used_note = ""
    D_dense = None
    D_sparse = None

    if args.use_knn_sparse:
        try:
            D_sparse = build_sparse_distance_from_knn(bundle_dir, n_nodes=N_nodes)
            D_used_note = "knn_edges.csv -> CSR"
            ascii_print("Using sparse CSR distance built from k-NN edges.")
            if args.knn_apsp_metric:
                ascii_print("Converting k-NN CSR to dense metric via APSP ...")
                D_dense = shortest_path(D_sparse, directed=False, unweighted=False)
                D_dense = np.asarray(D_dense, dtype=np.float32)
                # Clean up tiny asymmetries / diag
                np.fill_diagonal(D_dense, 0.0)
                D_dense = (D_dense + D_dense.T) / 2.0
                D_sparse = None
                D_used_note += " + APSP(metric)"
        except Exception as e:
            ascii_print(f"Warning: failed to build/convert k-NN ({e}). Falling back to dense.")
            D_sparse = None

    if D_sparse is None and D_dense is None:
        D_dense, D_used_note = load_distance_or_build(bundle_dir, method=args.correlation_method)
        ascii_print(f"Using dense distance ({D_used_note}). shape={D_dense.shape}")

    # Optional greedy-perm approximation (dense only)
    if args.use_greedy_perm and D_sparse is None:
        if not HAS_TFE:
            ascii_print("Warning: topological_feature_extractor not available; skipping greedy-perm path.")
        else:
            ascii_print(f"Applying greedy-perm approx with eps={args.rips_epsilon} ...")
            lambdas = getGreedyPerm(D_dense.copy())
            D_dense = getApproxSparseDM(lambdas, eps=args.rips_epsilon, D=D_dense.copy())
            ascii_print("Greedy-perm approximation applied.")

    # ---- PH ----
    ascii_print("Running Ripser ...")
    rips = Rips(verbose=True, maxdim=args.max_homology_dimension, do_cocycles=args.do_cocycles, coeff=args.coeff)
    diagrams = rips.fit_transform(D_sparse if D_sparse is not None else D_dense, distance_matrix=True)
    cocycles = getattr(rips, "cocycles_", None) or []
    ascii_print("Ripser complete.")

    # ---- Save diagrams ----
    np.save(os.path.join(out_dir, "persistence_diagram.npy"), np.array(diagrams, dtype=object))
    if len(diagrams) > 0:
        np.save(os.path.join(out_dir, "h0_persistence.npy"), diagrams[0])
    if len(diagrams) > 1:
        np.save(os.path.join(out_dir, "h1_persistence.npy"), diagrams[1])

    # ---- Barcodes + summary ----
    bars = compute_barcodes(diagrams)
    with open(os.path.join(out_dir, "barcodes.json"), "w") as f:
        json.dump(bars, f, indent=2)

    summary = summarize_barcodes(bars, distance_note=D_used_note)
    with open(os.path.join(out_dir, "barcodes_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Finite-only H1 indices (by original order in diagrams[1])
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
    finite_mask = np.isfinite(h1).all(axis=1)
    finite_indices = np.where(finite_mask)[0].astype(int).tolist()
    with open(os.path.join(out_dir, "h1_finite_indices.json"), "w") as f:
        json.dump(finite_indices, f, indent=2)

    # Top-K H1 by persistence (finite only)
    if h1.size > 0 and finite_indices:
        pers = (h1[:, 1] - h1[:, 0])
        pers[~finite_mask] = -np.inf
        order = np.argsort(pers)[::-1]
        order = [int(i) for i in order if np.isfinite(pers[i])]
        h1_top = []
        for i in order[: args.top_k_h1]:
            h1_top.append({"index": i, "birth": float(h1[i, 0]), "death": float(h1[i, 1]), "persistence": float(pers[i])})
        with open(os.path.join(out_dir, "h1_topK.json"), "w") as f:
            json.dump(h1_top, f, indent=2)

    # ---- Optional features ----
    if getattr(args, "save_features", False) and len(diagrams) > 1 and HAS_TFE:
        H1_features = calc_topo_feature(diagrams, dim=1)
        with open(os.path.join(out_dir, "H1_topo_features.json"), "w") as f:
            json.dump(H1_features.tolist() if isinstance(H1_features, np.ndarray) else H1_features, f, indent=2)

    # ---- Cocycle mapping ----
    if args.do_cocycles and len(cocycles) > 1:
        mapped = map_cocycles(cocycles[1], node_df)
        with open(os.path.join(out_dir, "h1_cocycles_mapped.json"), "w") as f:
            json.dump(mapped, f, indent=2)
    elif args.do_cocycles:
        ascii_print("Warning: cocycles requested but not available for H1.")

    # ---- Plots ----
    if args.save_plots or args.show_plot:
        if len(diagrams) > 0 and diagrams[0].size > 0:
            plt.figure(figsize=(7, 6))
            persim.plot_diagrams(diagrams, show=False, plot_only=[0])
            plt.title(f"H0 Persistence ({args.model_type}/{args.input_type})")
            plt.savefig(os.path.join(out_dir, "H0_persistence_diagram.png"), dpi=300, bbox_inches="tight")
            plt.close()
        if len(diagrams) > 1 and diagrams[1].size > 0:
            plt.figure(figsize=(7, 6))
            persim.plot_diagrams(diagrams, show=False, plot_only=[1])
            title = f"H1 Persistence ({args.model_type}/{args.input_type})"
            if summary.get("has_infinite_bars", False):
                title += "  [warning: ∞ bars detected]"
            plt.title(title)
            plt.savefig(os.path.join(out_dir, "H1_persistence_diagram.png"), dpi=300, bbox_inches="tight")
            plt.close()
        if args.show_plot and len(diagrams) > 1 and diagrams[1].size > 0:
            plt.figure(figsize=(7, 6))
            persim.plot_diagrams(diagrams, show=True, plot_only=[1])

    # ---- Provenance ----
    tda_cfg = {
        "model_type": args.model_type,
        "input_type": args.input_type,
        "correlation_method": args.correlation_method,
        "used_distance_source": D_used_note if D_sparse is None else "knn_sparse",
        "sparse_used": D_sparse is not None,
        "knn_apsp_metric": bool(args.knn_apsp_metric),
        "max_homology_dimension": args.max_homology_dimension,
        "do_cocycles": args.do_cocycles,
        "coeff": args.coeff,
        "top_k_h1": args.top_k_h1,
        "num_nodes": int(N_nodes),
        "num_h0": int(len(diagrams[0])) if len(diagrams) > 0 else 0,
        "num_h1": int(len(diagrams[1])) if len(diagrams) > 1 else 0,
    }
    with open(os.path.join(out_dir, "tda_config.json"), "w") as f:
        json.dump(tda_cfg, f, indent=2)

    ascii_print("Done.")


# --------------------------
# CLI
'''
recommended dense PH)

python compute_persistence_from_corr.py --correlation_base_dir ./correlation_output --model_type clean --input_type clean --correlation_method pearson --do_cocycles --save_plots

python compute_persistence_from_corr.py --correlation_base_dir ./correlation_output --model_type backdoored --input_type triggered --correlation_method pearson --do_cocycles --save_plots

k-NN PH (safer with APSP metric):

python compute_persistence_from_corr.py --correlation_base_dir ./correlation_output --model_type backdoored --input_type triggered --correlation_method pearson --use_knn_sparse --knn_apsp_metric --do_cocycles --save_plots

'''


# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Persistent Homology from correlation outputs (TOAP, robust).")
    parser.add_argument("--correlation_base_dir", type=str, default="./correlation_output")
    parser.add_argument("--model_type", type=str, required=True, choices=["clean", "backdoored"])
    parser.add_argument("--input_type", type=str, required=True, choices=["clean", "triggered"])
    parser.add_argument("--correlation_method", type=str, default="pearson", choices=["pearson", "cos"])

    # PH controls
    parser.add_argument("--max_homology_dimension", type=int, default=1)
    parser.add_argument("--do_cocycles", action="store_true")
    parser.add_argument("--coeff", type=int, default=2, help="Field coefficient")

    # Distance sourcing
    parser.add_argument("--use_knn_sparse", action="store_true",
                        help="Use k-NN CSR for PH (not recommended).")
    parser.add_argument("--knn_apsp_metric", action="store_true",
                        help="If using k-NN, convert to a dense metric with APSP before PH.")
    parser.add_argument("--use_greedy_perm", action="store_true",
                        help="Apply greedy-perm approximation to dense distance.")
    parser.add_argument("--rips_epsilon", type=float, default=0.1,
                        help="Epsilon for greedy-perm (only if --use_greedy_perm).")

    # Output
    parser.add_argument("--output_base_dir", type=str, default="./ph_output")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--top_k_h1", type=int, default=20)
    parser.add_argument("--save_features", action="store_true",
                        help="Compute and save H1_topo_features.json (requires topological_feature_extractor).")

    args = parser.parse_args()
    main(args)
