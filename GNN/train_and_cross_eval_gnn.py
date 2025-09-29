"""
Train-and-Cross-Eval GNN
========================

Trains the GNN via main.py on EfficientNet CSVs, then evaluates the trained
GNN on a MobileNet CSV by handling only the layer encoder during test preprocessing.

- Does NOT modify any existing functions or files
- Reuses saved preprocessors (scaler/encoder) and trained GNN from main.py output
- Computes MobileNet test graph using MobileNet layer encoder, keeping feature order identical

Usage examples:
  python train_and_cross_eval_gnn.py \
    --train_backdoored "Efficient Net TSS backdoored.csv" \
    --train_clean "Efficient Net TSS clean.csv" \
    --test_csv "Mobile Net TSS backdoored.csv" \
    --train_arch efficientnet --test_arch mobilenet \
    --output_dir efnet_training_output
"""

import os
import sys
import argparse
import subprocess
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import GNNSurrogate
from training_utils import evaluate_model, compute_metrics
from layer_encoders import get_layer_encoder


# Linear features used in training (must match training_utils.set_linear_features and DataProcessor)
BASE_LINEAR_FEATURES: List[str] = [
    # Weight features
    'w_out_norm_a', 'w_out_norm_b', 'w_norm_ratio',
    # Layer position features
    'depth_a', 'depth_b', 'depth_diff',
    # Connectivity features
    'fan_in_a', 'fan_in_b', 'fan_out_a', 'fan_out_b', 
    'fan_in_ratio', 'fan_out_ratio',
    # Gradient features
    'grad_out_norm_a', 'grad_out_norm_b', 'grad_norm_ratio',
    # Activation features
    'act_mean_a', 'act_mean_b', 'act_var_a', 'act_var_b',
    'act_std_a', 'act_std_b', 'act_mean_diff', 'act_std_ratio',
]
DISTANCE_FEATURES: List[str] = ['layer_distance', 'is_cross_stage', 'is_skip_connection']
ALL_LINEAR_FEATURES: List[str] = BASE_LINEAR_FEATURES + DISTANCE_FEATURES

TARGET_COL = 'tss'


def run_training_via_main(train_backdoored: str, train_clean: str, output_dir: str, train_arch: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, 'main.py',
        '--architecture', train_arch,
        '--model_type', 'gnn',
        '--backdoored_path', train_backdoored,
        '--clean_path', train_clean,
        '--output_dir', output_dir,
        '--skip_plots'
    ]
    print(f"Running training: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode


def infer_dims_from_checkpoint(ckpt: dict, fallback_edge_dim: int) -> Tuple[int, int, int]:
    hidden_dim = 64
    node_dim = None
    edge_dim = None
    if isinstance(ckpt, dict) and 'gat1.W.weight' in ckpt:
        w = ckpt['gat1.W.weight']
        hidden_dim = int(w.shape[0])
        node_dim = int(w.shape[1])
    if isinstance(ckpt, dict) and 'head.0.weight' in ckpt:
        hw = ckpt['head.0.weight']
        edge_dim = int(hw.shape[1] - (hidden_dim * 2))
    if edge_dim is None:
        edge_dim = fallback_edge_dim
    if node_dim is None:
        node_dim = fallback_edge_dim  # reasonable fallback
    return node_dim, edge_dim, hidden_dim


def compute_distance_features(df: pd.DataFrame, test_arch: str) -> pd.DataFrame:
    encoder = get_layer_encoder(test_arch)
    df = df.copy()
    # Ensure channels are ints
    if 'src_ch' in df.columns:
        df['src_ch'] = df['src_ch'].astype(float).astype(int)
    if 'dst_ch' in df.columns:
        df['dst_ch'] = df['dst_ch'].astype(float).astype(int)

    # Missing columns safety
    for col in BASE_LINEAR_FEATURES + ['src_layer', 'dst_layer', TARGET_COL, 'is_same_layer']:
        if col not in df.columns:
            # Create safe default
            if col == 'is_same_layer':
                df[col] = (df.get('src_layer', '') == df.get('dst_layer', '')).astype(int)
            elif col == TARGET_COL:
                raise ValueError(f"Required target column missing: {TARGET_COL}")
            else:
                df[col] = 0.0

    # Compute distances
    df['layer_distance'] = df.apply(lambda r: encoder.calculate_layer_distance(r['src_layer'], r['dst_layer']), axis=1)
    df['is_cross_stage'] = df.apply(lambda r: int(encoder.is_cross_stage(r['src_layer'], r['dst_layer'])), axis=1)
    df['is_skip_connection'] = df.apply(lambda r: int(encoder.is_skip_connection(r['src_layer'], r['dst_layer'])), axis=1)

    return df


def build_graph_from_df(df: pd.DataFrame, feature_order: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Build edge_index and features similar to DataProcessor.build_graph
    all_channels = pd.concat([df['src_ch'], df['dst_ch']]).dropna().unique()
    node_to_idx = {int(ch): i for i, ch in enumerate(all_channels)}

    edge_indices = []
    edge_feats = []

    # Collect node features as mean of edge features touching the node
    node_accum = {int(ch): [] for ch in all_channels}

    for _, row in df.iterrows():
        if pd.isna(row['src_ch']) or pd.isna(row['dst_ch']):
            continue
        try:
            src_ch, dst_ch = int(row['src_ch']), int(row['dst_ch'])
            src_idx, dst_idx = node_to_idx[src_ch], node_to_idx[dst_ch]
            edge_indices.append([src_idx, dst_idx])
            feats = row[feature_order].values.astype(np.float32)
            edge_feats.append(feats)
            node_accum[src_ch].append(feats)
            node_accum[dst_ch].append(feats)
        except Exception:
            continue

    # Node features: mean of incident edge features
    node_feats = []
    feat_len = len(feature_order)
    for ch in all_channels:
        arrs = node_accum.get(int(ch), [])
        if not arrs:
            node_feats.append(np.zeros(feat_len, dtype=np.float32))
        else:
            node_feats.append(np.mean(np.stack(arrs, axis=0), axis=0))

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32)
    node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
    y_tensor = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
    return edge_index, edge_attr, node_feats, y_tensor


def main():
    parser = argparse.ArgumentParser(description='Train via main.py on chosen architecture and cross-evaluate GNN on another architecture using only layer-encoder differences')
    parser.add_argument('--train_backdoored', type=str, required=True, help='Training backdoored CSV file')
    parser.add_argument('--train_clean', type=str, required=True, help='Training clean CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Test CSV file for cross-architecture evaluation')
    parser.add_argument('--train_arch', type=str, default='efficientnet', choices=['efficientnet', 'mobilenet', 'mobilenetv2'], help='Training architecture')
    parser.add_argument('--test_arch', type=str, default='mobilenet', choices=['efficientnet', 'mobilenet', 'mobilenetv2'], help='Test architecture')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output dir where main.py saves model and preprocessors')
    parser.add_argument('--enable_edge_removal', action='store_true', help='Enable edge removal with threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001, help='TSS threshold used if edge removal enabled')
    parser.add_argument('--force_train', action='store_true', help='Force retraining via main.py even if checkpoints exist')

    args = parser.parse_args()

    print('=' * 70)
    print('Train-and-Cross-Eval GNN')
    print('=' * 70)
    print(f"Train ({args.train_arch}): backdoored={args.train_backdoored} | clean={args.train_clean}")
    print(f"Test ({args.test_arch}): {args.test_csv}")
    print(f"Output dir: {args.output_dir}")

    # Step 1: Train if needed via main.py
    gnn_ckpt_path = os.path.join(args.output_dir, 'gnn_shortcut_aware_model.pth')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    encoder_path = os.path.join(args.output_dir, 'encoder.pkl')

    if args.force_train or (not os.path.exists(gnn_ckpt_path) or not os.path.exists(scaler_path)):
        print('\nLaunching training via main.py...')
        rc = run_training_via_main(args.train_backdoored, args.train_clean, args.output_dir, args.train_arch)
        if rc != 0:
            return rc
    else:
        print('\nFound existing trained GNN and preprocessors; skipping training.')

    # Step 2: Load preprocessors
    print('\nLoading saved preprocessors...')
    import joblib
    scaler = joblib.load(scaler_path)
    # encoder is not required for graph features (edge_attr uses only linear features)

    # Step 3: Load test CSV and compute distance features using TEST architecture layer encoder
    print(f"\nLoading and preprocessing {args.test_arch} test CSV...")
    test_df = pd.read_csv(args.test_csv, on_bad_lines='skip')

    # Optional edge removal
    if args.enable_edge_removal and args.tss_threshold > 0:
        before = len(test_df)
        test_df = test_df[test_df[TARGET_COL] > args.tss_threshold]
        print(f"Edge removal: kept {len(test_df)}/{before} rows (tss > {args.tss_threshold})")

    test_df = compute_distance_features(test_df, args.test_arch)

    # Ensure all required linear features exist; fill missing with zeros
    for col in ALL_LINEAR_FEATURES:
        if col not in test_df.columns:
            test_df[col] = 0.0

    # Step 4: Apply training scaler on the same feature order
    feature_order = ALL_LINEAR_FEATURES
    try:
        test_df[feature_order] = scaler.transform(test_df[feature_order])
    except Exception as e:
        print(f"Warning: scaler transform failed ({e}); attempting safe casting and retry...")
        test_df[feature_order] = test_df[feature_order].astype(float)
        test_df[feature_order] = scaler.transform(test_df[feature_order])

    # Step 5: Build graph tensors (edge_index, edge_attr, node_feats, y)
    edge_index, edge_attr, node_feats, y_tensor = build_graph_from_df(test_df, feature_order)

    test_dataset = TensorDataset(edge_index.t(), edge_attr, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 6: Load trained GNN
    print('\nLoading trained GNN...')
    ckpt = torch.load(gnn_ckpt_path, map_location='cpu')
    node_dim, edge_dim, hidden_dim = infer_dims_from_checkpoint(ckpt, fallback_edge_dim=len(feature_order))

    model = GNNSurrogate(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, dropout=0.2)
    model.load_state_dict(ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # Step 7: Evaluate
    print(f"\nEvaluating on {args.test_arch} test CSV using GNN...")
    criterion = torch.nn.MSELoss()
    test_loss, preds, targets = evaluate_model(model, test_loader, device, node_feats.to(device), criterion)
    metrics = compute_metrics(preds, targets)

    print(f"\nGNN Cross-Architecture Results ({args.train_arch} train -> {args.test_arch} test):")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Spearman's ρ: {metrics['spearman']:.6f}")
    print(f"Kendall's τ: {metrics['kendall']:.6f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f'gnn_cross_{args.train_arch}_to_{args.test_arch}_results.csv')
    pd.DataFrame({
        'true_tss': targets,
        'gnn_pred': preds,
        'residual': np.array(preds) - np.array(targets),
        'abs_residual': np.abs(np.array(preds) - np.array(targets))
    }).to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
