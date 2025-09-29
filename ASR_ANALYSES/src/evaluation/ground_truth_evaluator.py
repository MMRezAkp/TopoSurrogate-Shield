"""
ASR Evaluation using Ground Truth TSS Data.
"""

import copy
import time
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_utils import ModelUtils


class ASRGroundTruthEvaluator:
    """ASR Evaluation using Ground Truth TSS Data."""
    
    def __init__(self, model_path, tss_data_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.tss_data_path = tss_data_path
        
        # Load model
        print(f"Loading EfficientNet model from {model_path}...")
        self.model = ModelUtils.load_efficientnet_model(model_path, device)
        self.model.to(device)
        self.model.eval()
        
        # Load TSS data
        print("Loading ground truth TSS data...")
        self.tss_data = pd.read_csv(tss_data_path)
        print(f"Loaded {len(self.tss_data)} TSS records")
        print(f"TSS scores range: {self.tss_data['tss'].min():.6f} to {self.tss_data['tss'].max():.6f}")
        
        print(f"✓ ASR Ground Truth Evaluator initialized")
    
    def remove_top_tss_edges(self, removal_ratio=0.1):
        """Remove top TSS edges from the model."""
        print(f"Removing top {int(removal_ratio * 100)}% TSS edges...")
        
        # Sort by TSS score and get top edges
        top_edges = self.tss_data.nlargest(int(len(self.tss_data) * removal_ratio), 'tss')
        print(f"Top {len(top_edges)} edges TSS range: {top_edges['tss'].min():.6f} to {top_edges['tss'].max():.6f}")
        
        # Create a copy of the model for modification
        modified_model = copy.deepcopy(self.model).to(self.device)
        modified_model.eval()
        
        print("Removing edges...")
        removed_count = 0
        
        for _, edge in tqdm(top_edges.iterrows(), total=len(top_edges), desc="Pruning"):
            try:
                # Parse edge information
                src_layer = edge['src_layer']
                src_ch = int(edge['src_ch'])
                dst_layer = edge['dst_layer']
                dst_ch = int(edge['dst_ch'])
                
                # Get the actual layer objects
                src_layer_obj = ModelUtils.get_layer_by_name(modified_model, src_layer)
                dst_layer_obj = ModelUtils.get_layer_by_name(modified_model, dst_layer)
                
                if src_layer_obj is not None and dst_layer_obj is not None:
                    # Zero out the specific weight connection
                    if hasattr(src_layer_obj, 'weight') and src_layer_obj.weight is not None:
                        if len(src_layer_obj.weight.shape) >= 2:
                            # For linear layers: weight[out_features, in_features]
                            if dst_ch < src_layer_obj.weight.shape[0] and src_ch < src_layer_obj.weight.shape[1]:
                                src_layer_obj.weight.data[dst_ch, src_ch] = 0.0
                                removed_count += 1
                    
                    # Also zero out bias if it exists
                    if hasattr(src_layer_obj, 'bias') and src_layer_obj.bias is not None:
                        if dst_ch < src_layer_obj.bias.shape[0]:
                            src_layer_obj.bias.data[dst_ch] = 0.0
                
            except Exception as e:
                print(f"Error removing edge {src_layer}:{src_ch} -> {dst_layer}:{dst_ch}: {e}")
                continue
        
        print(f"Successfully removed {removed_count} edge weights")
        return modified_model
    
    def run_evaluation(self, dataloader, removal_ratio=0.1, poison_ratio=0.1):
        """Run complete ASR evaluation."""
        total_start_time = time.time()
        
        print("=" * 60)
        print("ASR Drop Evaluation using Ground Truth TSS Data")
        print("=" * 60)
        
        # 0. Baseline Clean Accuracy (on provided clean dataloader)
        print("\n0. Calculating initial Clean Accuracy (clean dataloader)...")
        start_time = time.time()
        initial_acc = ModelUtils.calculate_accuracy(self.model, dataloader, self.device)
        initial_acc_time = time.time() - start_time
        print(f"✓ Initial clean accuracy calculated in {initial_acc_time:.2f} seconds")
        
        # 1. Initial ASR (using current simplified metric & poison_ratio)
        print(f"\n1. Calculating initial ASR...")
        start_time = time.time()
        initial_asr = ModelUtils.calculate_asr(self.model, dataloader, poison_ratio, self.device)
        initial_asr_time = time.time() - start_time
        print(f"✓ Initial ASR calculated in {initial_asr_time:.2f} seconds")
        
        # 2. Remove top TSS edges (prune)
        print(f"\n2. Removing top {int(removal_ratio * 100)}% TSS edges...")
        start_time = time.time()
        modified_model = self.remove_top_tss_edges(removal_ratio)
        edge_removal_time = time.time() - start_time
        print(f"✓ Edge removal completed in {edge_removal_time:.2f} seconds")
        
        # 3. Post-removal metrics
        print(f"\n3. Calculating post-removal metrics...")
        start_time = time.time()
        original_model = self.model
        try:
            # swap in modified model
            self.model = modified_model
            final_acc = ModelUtils.calculate_accuracy(self.model, dataloader, self.device)
            final_asr = ModelUtils.calculate_asr(self.model, dataloader, poison_ratio, self.device)
        finally:
            # restore original model
            self.model = original_model
        final_metrics_time = time.time() - start_time
        print(f"✓ Final metrics calculated in {final_metrics_time:.2f} seconds")
        
        # 4. Summaries
        acc_drop = initial_acc - final_acc
        asr_reduction = initial_asr - final_asr
        reduction_percentage = (asr_reduction / initial_asr) * 100 if initial_asr > 0 else 0
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Initial Clean Acc: {initial_acc:.2f}%")
        print(f"Final   Clean Acc: {final_acc:.2f}%")
        print(f"Acc Drop:          {acc_drop:.2f}%")
        print("-" * 60)
        print(f"Initial ASR:       {initial_asr:.2f}%")
        print(f"Final   ASR:       {final_asr:.2f}%")
        print(f"ASR Reduction:     {asr_reduction:.2f}%")
        print(f"Reduction %:       {reduction_percentage:.2f}%")
        print(f"Edges Removed:     {int(len(self.tss_data) * removal_ratio)}")
        print(f"Removal Ratio:     {removal_ratio * 100:.1f}%")
        
        print(f"\n" + "=" * 60)
        print("TIMING BREAKDOWN")
        print("=" * 60)
        print(f"Initial Clean Accuracy:     {initial_acc_time:.2f} seconds")
        print(f"Initial ASR Calculation:    {initial_asr_time:.2f} seconds")
        print(f"Edge Removal:               {edge_removal_time:.2f} seconds")
        print(f"Final Metrics:              {final_metrics_time:.2f} seconds")
        print("-" * 60)
        print(f"TOTAL EVALUATION TIME:      {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("=" * 60)
        
        return {
            'initial_acc': initial_acc,
            'final_acc': final_acc,
            'acc_drop': acc_drop,
            'initial_asr': initial_asr,
            'final_asr': final_asr,
            'asr_reduction': asr_reduction,
            'reduction_percentage': reduction_percentage,
            'edges_removed': int(len(self.tss_data) * removal_ratio),
            'timing': {
                'initial_acc_time': initial_acc_time,
                'initial_asr_time': initial_asr_time,
                'edge_removal_time': edge_removal_time,
                'final_metrics_time': final_metrics_time,
                'total_time': total_time
            }
        }


