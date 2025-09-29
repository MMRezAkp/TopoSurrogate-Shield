"""
Feature extraction utilities for TSS analysis.
"""

import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.layer_encoder import LayerEncoder


class FeatureExtractor:
    """Feature extraction utilities for TSS analysis."""
    
    def __init__(self, layer_encoder=None):
        self.layer_encoder = layer_encoder or LayerEncoder()

    def create_edges_optimized(self, layer_names, weight_features, activation_features, max_edges=10000):
        """Create edges between layers for graph construction."""
        print(f"Creating edges between {len(layer_names)} layers (max {max_edges} edges)...")
        layer_depths = [(name, self.layer_encoder.get_layer_depth(name)) for name in layer_names]
        layer_depths.sort(key=lambda x: x[1])
        sorted_layers = [name for name, _ in layer_depths]
        edges, edge_count = [], 0
        
        for i, src_layer in enumerate(sorted_layers):
            if edge_count >= max_edges: 
                break
            src_channels = [ch for (lname, ch) in weight_features.keys() if lname == src_layer]
            if len(src_channels) > 50: 
                src_channels = src_channels[:50]
            
            for j, dst_layer in enumerate(sorted_layers):
                if i == j or edge_count >= max_edges: 
                    continue
                if abs(i - j) > 3 and not self.layer_encoder.is_skip_connection(src_layer, dst_layer):
                    continue
                
                dst_channels = [ch for (lname, ch) in weight_features.keys() if lname == dst_layer]
                if len(dst_channels) > 50: 
                    dst_channels = dst_channels[:50]
                
                for sc in src_channels:
                    if edge_count >= max_edges: 
                        break
                    for dc in dst_channels:
                        if edge_count >= max_edges: 
                            break
                        edges.append({
                            'src_layer': src_layer, 
                            'dst_layer': dst_layer, 
                            'src_ch': sc, 
                            'dst_ch': dc
                        })
                        edge_count += 1
        
        print(f"Created {len(edges)} edges (limited to {max_edges})")
        return edges

    def extract_edge_features_batch(self, edges, weight_features, activation_features, batch_size=5000):
        """Extract features for edges in batches."""
        print(f"Extracting edge features in batches of {batch_size}...")
        all_edge_data = []
        
        for i in range(0, len(edges), batch_size):
            batch_edges = edges[i:i + batch_size]
            print(f"Processing edge batch {i//batch_size + 1}/{(len(edges) + batch_size - 1)//batch_size}")
            batch_edge_data = []
            
            for e in batch_edges:
                sa, da, sc, dc = e['src_layer'], e['dst_layer'], e['src_ch'], e['dst_ch']
                sw = weight_features.get((sa, sc), {
                    'w_out_norm': 0.0, 'grad_out_norm': 0.0, 'fan_in': 0, 'fan_out': 0
                })
                dw = weight_features.get((da, dc), {
                    'w_out_norm': 0.0, 'grad_out_norm': 0.0, 'fan_in': 0, 'fan_out': 0
                })
                sa_act = activation_features.get((sa, sc), (0.0, 0.0))
                da_act = activation_features.get((da, dc), (0.0, 0.0))
                
                layer_distance = self.layer_encoder.calculate_layer_distance(sa, da)
                depth_a = self.layer_encoder.get_layer_depth(sa)
                depth_b = self.layer_encoder.get_layer_depth(da)
                depth_diff = abs(depth_a - depth_b)
                is_cross = self.layer_encoder.is_cross_stage(sa, da)
                is_skip = self.layer_encoder.is_skip_connection(sa, da)
                
                a_mean, a_var = sa_act
                b_mean, b_var = da_act
                a_std = float(np.sqrt(a_var)) if a_var > 0 else 0.0
                b_std = float(np.sqrt(b_var)) if b_var > 0 else 0.0
                act_mean_diff = abs(a_mean - b_mean)
                act_std_ratio = a_std / (b_std + 1e-8)
                w_norm_ratio = sw['w_out_norm'] / (dw['w_out_norm'] + 1e-8)
                fan_in_ratio = sw['fan_in'] / (dw['fan_in'] + 1e-8)
                fan_out_ratio = sw['fan_out'] / (dw['fan_out'] + 1e-8)
                grad_norm_ratio = sw['grad_out_norm'] / (dw['grad_out_norm'] + 1e-8)
                
                batch_edge_data.append({
                    'src_layer': sa, 'dst_layer': da, 'src_ch': sc, 'dst_ch': dc,
                    'is_same_layer': float(sa == da),
                    'w_out_norm_a': sw['w_out_norm'], 'w_out_norm_b': dw['w_out_norm'],
                    'grad_out_norm_a': sw['grad_out_norm'], 'grad_out_norm_b': dw['grad_out_norm'],
                    'act_mean_a': a_mean, 'act_var_a': a_var, 'act_mean_b': b_mean, 'act_var_b': b_var,
                    'fan_in_a': sw['fan_in'], 'fan_out_a': sw['fan_out'],
                    'fan_in_b': dw['fan_in'], 'fan_out_b': dw['fan_out'],
                    'depth_a': depth_a, 'depth_b': depth_b, 'depth_diff': depth_diff,
                    'act_std_a': a_std, 'act_std_b': b_std,
                    'act_mean_diff': act_mean_diff, 'act_std_ratio': act_std_ratio,
                    'w_norm_ratio': w_norm_ratio, 'fan_in_ratio': fan_in_ratio, 
                    'fan_out_ratio': fan_out_ratio, 'grad_norm_ratio': grad_norm_ratio,
                    'layer_distance': layer_distance, 'is_cross_stage': int(is_cross), 
                    'is_skip_connection': int(is_skip)
                })
            
            all_edge_data.extend(batch_edge_data)
            del batch_edge_data
            gc.collect()
        
        print(f"Extracted features for {len(all_edge_data)} edges")
        return all_edge_data



