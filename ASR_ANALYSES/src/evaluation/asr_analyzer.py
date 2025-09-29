"""
ASR Analysis using GNN-based TSS prediction.
"""

import os
import gc
import copy
import time
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.gnn_models import FixedGNNSurrogate
from models.efficientnet_models import OptimizedEfficientNetFeatureExtractor
from data.loaders import DataLoaderFactory
from data.feature_extractors import FeatureExtractor
from utils.layer_encoder import LayerEncoder
from utils.model_utils import ModelUtils
from config.settings import TSS_CONFIG


class ASRAnalyzerWithRealFeatures:
    """ASR Analysis using real features from the working TSS prediction framework."""
    
    def __init__(self, model_path, gnn_path, scaler_path, encoder_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path  # Store the model path for later use
        self.layer_encoder = LayerEncoder()
        self.feature_extractor = FeatureExtractor(self.layer_encoder)

        # Force load GNN (no fallback)
        self.gnn_model = self._load_gnn_model(gnn_path)
        assert self.gnn_model is not None, "GNN is required; failed to load."

        # Preprocessors
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        # EfficientNet model
        self.model = ModelUtils.load_efficientnet_model(model_path, self.device)
        print(f"✓ ASR Analyzer initialized on {self.device}")

    def _load_gnn_model(self, gnn_path):
        """Load GNN model from checkpoint."""
        if not os.path.exists(gnn_path):
            raise FileNotFoundError(f"GNN checkpoint not found: {gnn_path}")
        checkpoint = torch.load(gnn_path, map_location=self.device)

        # Infer dims from checkpoint
        gat1_w_weight = checkpoint['gat1.W.weight']
        head_0_weight = checkpoint['head.0.weight']
        node_dim = gat1_w_weight.shape[1]
        hidden_dim = gat1_w_weight.shape[0]
        edge_dim = head_0_weight.shape[1] - (hidden_dim * 2)

        gnn_model = FixedGNNSurrogate(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        gnn_model.load_state_dict(checkpoint)
        gnn_model.to(self.device).eval()
        print(f"✓ Loaded GNN model from {gnn_path}")
        return gnn_model

    def create_evaluation_loaders(self, batch_size=64, target_label=0, trigger_size=3, trigger_color=(1.0,1.0,1.0)):
        """Create evaluation loaders (clean + fully-poisoned)."""
        return DataLoaderFactory.create_evaluation_loaders(
            batch_size=batch_size, target_label=target_label, 
            trigger_size=trigger_size, trigger_color=trigger_color
        )

    def extract_features_using_working_framework(self, dataloader, max_layers=20):
        """Extract features using working framework."""
        print("Extracting features using working framework...")
        # Use the same model that was already loaded for the analyzer
        extractor = OptimizedEfficientNetFeatureExtractor(
            model_path=self.model_path, device=self.device, num_classes=10
        )
        wfeat, afeat, layer_names = extractor.extract_all_features(dataloader, max_layers)
        return wfeat, afeat, layer_names

    def predict_tss_batch(self, edge_data, batch_size=1000):
        """Predict TSS scores using GNN in batches."""
        if self.gnn_model is None:
            raise RuntimeError("GNN model is required but not loaded.")

        print(f"Predicting TSS scores in batches of {batch_size}...")
        df = pd.DataFrame(edge_data)
        linear_features = [
            'w_out_norm_a','w_out_norm_b','w_norm_ratio',
            'depth_a','depth_b','depth_diff',
            'fan_in_a','fan_in_b','fan_out_a','fan_out_b',
            'fan_in_ratio','fan_out_ratio',
            'grad_out_norm_a','grad_out_norm_b','grad_norm_ratio',
            'act_mean_a','act_mean_b','act_var_a','act_var_b',
            'act_std_a','act_std_b','act_mean_diff','act_std_ratio',
            'layer_distance','is_cross_stage','is_skip_connection'
        ]
        
        src_layers = df['src_layer'].values
        dst_layers = df['dst_layer'].values
        src_chs = df['src_ch'].values
        dst_chs = df['dst_ch'].values

        # OneHot encoding with error handling for unknown categories
        try:
            enc = self.encoder.transform(df[['src_layer','dst_layer']].values)
            if hasattr(enc, "toarray"):
                enc = enc.toarray()
            enc_cols = list(self.encoder.get_feature_names_out())
            enc_df = pd.DataFrame(enc, columns=enc_cols, index=df.index)
        except ValueError as e:
            if "unknown categories" in str(e):
                print(f"Warning: Found unknown layer names. Creating compatible encoder...")
                # Get unique layer names from current data
                unique_layers = list(set(df['src_layer'].unique().tolist() + df['dst_layer'].unique().tolist()))
                print(f"Found {len(unique_layers)} unique layer names in current data")
                
                # Create a simple encoding based on layer names
                layer_to_idx = {layer: idx for idx, layer in enumerate(unique_layers)}
                
                # Create encoded features: one-hot for src and dst layers
                src_encoded = np.zeros((len(df), len(unique_layers)))
                dst_encoded = np.zeros((len(df), len(unique_layers)))
                
                for i, (src, dst) in enumerate(zip(df['src_layer'], df['dst_layer'])):
                    if src in layer_to_idx:
                        src_encoded[i, layer_to_idx[src]] = 1
                    if dst in layer_to_idx:
                        dst_encoded[i, layer_to_idx[dst]] = 1
                
                # Combine src and dst encodings
                combined_encoding = np.concatenate([src_encoded, dst_encoded], axis=1)
                enc_df = pd.DataFrame(combined_encoding, index=df.index)
                enc_df.columns = [f'src_layer_{layer}' for layer in unique_layers] + [f'dst_layer_{layer}' for layer in unique_layers]
            else:
                raise e

        df_enc = pd.concat([df.drop(['src_layer','dst_layer'], axis=1), enc_df], axis=1)
        df_enc[linear_features] = self.scaler.transform(df_enc[linear_features])

        preds = []
        for i in range(0, len(df_enc), batch_size):
            bdf = df_enc.iloc[i:i+batch_size]
            edge_attr = torch.tensor(bdf[linear_features].values, dtype=torch.float, device=self.device)

            b_src_layers = src_layers[i:i+batch_size]
            b_dst_layers = dst_layers[i:i+batch_size]
            b_src_chs = src_chs[i:i+batch_size]
            b_dst_chs = dst_chs[i:i+batch_size]

            uniq = list(set([(b_src_layers[k], int(b_src_chs[k])) for k in range(len(b_src_chs))] +
                            [(b_dst_layers[k], int(b_dst_chs[k])) for k in range(len(b_dst_chs))]))
            node_index = {k: idx for idx, k in enumerate(uniq)}

            node_feats = []
            for k in uniq:
                mask = (b_src_layers == k[0]) & (b_src_chs == k[1])
                if np.any(mask):
                    node_feats.append(bdf.loc[mask, linear_features].mean().values)
                else:
                    node_feats.append(np.zeros(len(linear_features)))
            node_feats = torch.tensor(np.array(node_feats), dtype=torch.float, device=self.device)

            eidx = []
            for k in range(len(b_src_layers)):
                eidx.append([node_index[(b_src_layers[k], int(b_src_chs[k]))],
                             node_index[(b_dst_layers[k], int(b_dst_chs[k]))]])
            edge_index = torch.tensor(eidx, dtype=torch.long, device=self.device).t().contiguous()

            with torch.no_grad():
                raw = self.gnn_model(node_feats, edge_index, edge_attr)
                # Make all TSS positive
                raw = torch.abs(raw)
                raw = torch.clamp(raw, min=TSS_CONFIG['eps'])
                preds.extend(raw.detach().cpu().numpy().tolist())

            del bdf, edge_attr, node_feats, edge_index
            gc.collect()

        for i, edge in enumerate(edge_data):
            edge['tss'] = float(max(preds[i], TSS_CONFIG['eps']))
        return edge_data

    def create_modified_model(self, model, edges_to_remove):
        """Create modified model by zeroing out removed edge weights."""
        print("Creating modified model by zeroing out removed edge weights...")
        modified_model = copy.deepcopy(model).to(self.device).eval()
        for edge, tss in edges_to_remove:
            src_layer, dst_layer, sc, dc = edge['src_layer'], edge['dst_layer'], edge['src_ch'], edge['dst_ch']
            try:
                dst_module = dict(modified_model.named_modules())[dst_layer]
                if hasattr(dst_module, 'weight') and dst_module.weight is not None:
                    W = dst_module.weight
                    if W.ndim >= 2:
                        if dc < W.shape[0] and sc < W.shape[1]:
                            if W.ndim == 2:
                                W.data[dc, sc] = 0.0
                            elif W.ndim == 4:
                                W.data[dc, sc, :, :] = 0.0
                            if getattr(dst_module, 'bias', None) is not None and dc < dst_module.bias.shape[0]:
                                dst_module.bias.data[dc] = 0.0
            except Exception:
                continue
        print(f"Modified model by zeroing out {len(edges_to_remove)} edge weights")
        return modified_model

    def run_asr_analysis(self, removal_ratio=0.1, max_layers=20, max_edges=10000):
        """Run complete ASR analysis."""
        total_start_time = time.time()
        
        # Initialize timing variables
        loader_time = baseline_time = feature_time = edge_creation_time = 0.0
        edge_feature_time = tss_prediction_time = edge_removal_time = 0.0
        model_modification_time = final_metrics_time = total_time = 0.0
        
        print("=" * 60)
        print("ASR Analysis: Before and After TSS Edge Removal (GNN FORCED)")
        print("=" * 60)

        # Evaluation loaders
        print("Preparing evaluation loaders (clean & fully-poisoned)...")
        start_time = time.time()
        clean_loader, poisoned_full_loader = self.create_evaluation_loaders(batch_size=64, target_label=0)
        loader_time = time.time() - start_time
        print(f"✓ Data loaders prepared in {loader_time:.2f} seconds")

        # Baseline metrics
        print("\n1. Calculating baseline metrics...")
        start_time = time.time()
        initial_acc = ModelUtils.calculate_accuracy(self.model, clean_loader, self.device)
        initial_asr = ModelUtils.calculate_asr(self.model, poisoned_full_loader, device=self.device)
        baseline_time = time.time() - start_time
        print(f"Initial Clean Accuracy: {initial_acc:.2f}%")
        print(f"Initial ASR:           {initial_asr:.2f}%")
        print(f"✓ Baseline metrics calculated in {baseline_time:.2f} seconds")

        # Feature extraction
        print("\n2. Extracting features using working framework...")
        start_time = time.time()
        wfeat, afeat, layer_names = self.extract_features_using_working_framework(clean_loader, max_layers)
        feature_time = time.time() - start_time
        print(f"✓ Feature extraction completed in {feature_time:.2f} seconds")

        # Edges
        print("\n3. Creating edges...")
        start_time = time.time()
        edges = self.feature_extractor.create_edges_optimized(layer_names, wfeat, afeat, max_edges)
        edge_creation_time = time.time() - start_time
        print(f"✓ Edge creation completed in {edge_creation_time:.2f} seconds")

        # Edge features
        print("\n4. Extracting edge features...")
        start_time = time.time()
        edge_data = self.feature_extractor.extract_edge_features_batch(edges, wfeat, afeat)
        edge_feature_time = time.time() - start_time
        print(f"✓ Edge feature extraction completed in {edge_feature_time:.2f} seconds")

        # TSS prediction
        print("\n5. Predicting TSS scores (GNN)...")
        start_time = time.time()
        edge_data_with_tss = self.predict_tss_batch(edge_data)
        tss_prediction_time = time.time() - start_time
        print(f"✓ TSS prediction completed in {tss_prediction_time:.2f} seconds")

        # Remove top-TSS edges
        print(f"\n6. Removing top {removal_ratio*100:.1f}% TSS edges...")
        start_time = time.time()
        edge_tss_pairs = [(edge, edge['tss']) for edge in edge_data_with_tss]
        edge_tss_pairs.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = int(len(edge_data_with_tss) * removal_ratio)
        edges_to_remove = edge_tss_pairs[:num_to_remove]
        edge_removal_time = time.time() - start_time
        print(f"Removing {num_to_remove} edges with highest TSS scores")
        print(f"✓ Edge sorting and selection completed in {edge_removal_time:.2f} seconds")

        # Modified model
        print("\n7. Creating modified model...")
        start_time = time.time()
        modified_model = self.create_modified_model(self.model, edges_to_remove)
        model_modification_time = time.time() - start_time
        print(f"✓ Model modification completed in {model_modification_time:.2f} seconds")

        # Post-removal metrics
        print("\n8. Calculating metrics after edge removal...")
        start_time = time.time()
        final_acc = ModelUtils.calculate_accuracy(modified_model, clean_loader, self.device)
        final_asr = ModelUtils.calculate_asr(modified_model, poisoned_full_loader, device=self.device)
        final_metrics_time = time.time() - start_time
        print(f"Final Clean Accuracy:  {final_acc:.2f}%")
        print(f"Final ASR:             {final_asr:.2f}%")
        print(f"✓ Final metrics calculated in {final_metrics_time:.2f} seconds")

        # Summary
        print("\n" + "=" * 60)
        print("ASR ANALYSIS RESULTS (GNN FORCED)")
        print("=" * 60)
        print(f"Initial Clean Acc: {initial_acc:.2f}%")
        print(f"Final   Clean Acc: {final_acc:.2f}%")
        print(f"Acc Drop:          {initial_acc - final_acc:.2f}%")
        print("-" * 60)
        print(f"Initial ASR:       {initial_asr:.2f}%")
        print(f"Final   ASR:       {final_asr:.2f}%")
        print(f"ASR Reduction:     {initial_asr - final_asr:.2f}%")
        print(f"Edges Removed:     {len(edges_to_remove)}")
        print(f"Removal Ratio:     {removal_ratio*100:.1f}%")

        tss_scores = [edge['tss'] for edge in edge_data_with_tss]
        print(f"\nTSS Statistics:")
        print(f"Mean TSS: {np.mean(tss_scores):.6f}")
        print(f"Min TSS:  {np.min(tss_scores):.6f}")
        print(f"Max TSS:  {np.max(tss_scores):.6f}")

        print(f"\nTop 5 Removed Edges (by TSS score):")
        for i, (edge, tss) in enumerate(edges_to_remove[:5]):
            print(f"{i+1}. {edge['src_layer']}:{edge['src_ch']} -> {edge['dst_layer']}:{edge['dst_ch']} (TSS: {tss:.6f})")

        # Total time and timing breakdown
        total_time = time.time() - total_start_time
        print(f"\n" + "=" * 60)
        print("TIMING BREAKDOWN")
        print("=" * 60)
        print(f"Data Loader Preparation:    {loader_time:.2f} seconds")
        print(f"Baseline Metrics:           {baseline_time:.2f} seconds")
        print(f"Feature Extraction:         {feature_time:.2f} seconds")
        print(f"Edge Creation:              {edge_creation_time:.2f} seconds")
        print(f"Edge Feature Extraction:    {edge_feature_time:.2f} seconds")
        print(f"TSS Prediction (GNN):       {tss_prediction_time:.2f} seconds")
        print(f"Edge Sorting & Selection:   {edge_removal_time:.2f} seconds")
        print(f"Model Modification:         {model_modification_time:.2f} seconds")
        print(f"Final Metrics:              {final_metrics_time:.2f} seconds")
        print("-" * 60)
        print(f"TOTAL ANALYSIS TIME:        {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("=" * 60)
        
        # Debug: Show timing dictionary before adding to results
        timing_dict = {
            'loader_time': loader_time,
            'baseline_time': baseline_time,
            'feature_time': feature_time,
            'edge_creation_time': edge_creation_time,
            'edge_feature_time': edge_feature_time,
            'tss_prediction_time': tss_prediction_time,
            'edge_removal_time': edge_removal_time,
            'model_modification_time': model_modification_time,
            'final_metrics_time': final_metrics_time,
            'total_time': total_time
        }
        print(f"\nDEBUG: Timing dictionary to be returned: {timing_dict}")

        return {
            'initial_acc': initial_acc,
            'final_acc': final_acc,
            'acc_drop': initial_acc - final_acc,
            'initial_asr': initial_asr,
            'final_asr': final_asr,
            'asr_reduction': initial_asr - final_asr,
            'edges_removed': len(edges_to_remove),
            'removal_ratio': removal_ratio,
            'removed_edges': edges_to_remove,
            'tss_scores': tss_scores,
            'timing': timing_dict
        }


