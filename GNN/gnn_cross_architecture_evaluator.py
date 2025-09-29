"""
GNN Cross-Architecture Evaluator
================================

This script uses your actual trained GNN model for cross-architecture evaluation,
showing epoch-by-epoch training metrics and using the same GNN architecture you built.

Usage:
    python gnn_cross_architecture_evaluator.py --train_csv efficientnet_train.csv --test_csv mobilenet_test.csv --gnn_model_path gnn_shortcut_aware_model.pth
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path

# Import your existing modules (without modification)
from data_processing import DataProcessor
from models import GNNSurrogate, create_model
from training_utils import compute_metrics, evaluate_model, TrainingLogger
from config import MODEL_CONFIG
import torch.nn as nn


class GNNCrossArchitectureEvaluator:
    """
    Cross-architecture evaluator using your actual trained GNN model.
    """
    
    def __init__(self, train_architecture: str = 'efficientnet', test_architecture: str = 'mobilenet'):
        """Initialize the GNN cross-architecture evaluator."""
        self.train_architecture = train_architecture
        self.test_architecture = test_architecture
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
    def load_trained_gnn(self, model_path: str, node_dim: int = None, edge_dim: int = None, 
                        hidden_dim: int = 64) -> torch.nn.Module:
        """Load a trained GNN model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GNN model file not found: {model_path}")
        
        print(f"Loading GNN model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to infer dimensions from checkpoint if not provided
        if node_dim is None or edge_dim is None:
            try:
                # Look for GAT layer weights to infer dimensions
                if 'gat1.W.weight' in checkpoint:
                    gat1_w_weight = checkpoint['gat1.W.weight']
                    node_dim = gat1_w_weight.shape[1]
                    hidden_dim = gat1_w_weight.shape[0]
                    
                if 'head.0.weight' in checkpoint:
                    head_0_weight = checkpoint['head.0.weight']
                    edge_dim = head_0_weight.shape[1] - (hidden_dim * 2)
                    
                print(f"Inferred dimensions: node_dim={node_dim}, edge_dim={edge_dim}, hidden_dim={hidden_dim}")
            except Exception as e:
                print(f"Could not infer dimensions from checkpoint: {e}")
                # Use defaults
                node_dim = node_dim or 26  # Common default
                edge_dim = edge_dim or 26
                print(f"Using default dimensions: node_dim={node_dim}, edge_dim={edge_dim}")
        
        # Create model
        model = GNNSurrogate(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=MODEL_CONFIG['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded GNN model successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        self.gnn_model = model
        return model
    
    def create_cross_arch_data_processors(self) -> Tuple[DataProcessor, DataProcessor]:
        """Create data processors for train and test architectures."""
        train_processor = DataProcessor(self.train_architecture)
        test_processor = DataProcessor(self.test_architecture)
        return train_processor, test_processor
    
    def load_and_process_data(self, train_csv_path: str, test_csv_path: str,
                            enable_edge_removal: bool = False, tss_threshold: float = 0.0001):
        """Load and process data using architecture-specific processors."""
        print("=" * 60)
        print("DATA LOADING AND PROCESSING")
        print("=" * 60)
        
        # Create processors
        train_processor, test_processor = self.create_cross_arch_data_processors()
        
        # Process training data
        print(f"Processing TRAINING data ({self.train_architecture})...")
        train_loader, val_loader, _, train_edge_index, train_edge_attr, train_node_feats = train_processor.process_pipeline(
            train_csv_path, train_csv_path,  # Use same file for both (we'll split differently)
            enable_edge_removal=enable_edge_removal,
            tss_threshold=tss_threshold
        )
        
        # Process test data using the TRAINING architecture processor for consistency
        print(f"\nProcessing TEST data ({self.test_architecture}) using {self.train_architecture} processor...")
        try:
            # Try to use the same processor (may fail due to different layer names)
            test_df = pd.read_csv(test_csv_path, on_bad_lines='skip')
            test_processed = train_processor.preprocess_data(
                test_df, 
                enable_edge_removal=enable_edge_removal, 
                tss_threshold=tss_threshold
            )
            
            # Build graph using train processor
            test_edge_index, test_edge_attr, test_node_feats, test_y_tensor = train_processor.build_graph(test_processed)
            
            # Create test loader
            from torch.utils.data import TensorDataset, DataLoader
            test_dataset = TensorDataset(
                test_edge_index.t(),
                test_edge_attr,
                test_y_tensor
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            print(f"✓ Successfully processed test data using training architecture processor")
            
        except Exception as e:
            print(f"Cross-architecture processing failed: {e}")
            print("This is expected when layer structures are very different.")
            return None, None, None, None, None, None
        
        return train_loader, val_loader, test_loader, train_node_feats, test_node_feats, train_processor
    
    def evaluate_gnn_cross_architecture(self, test_loader, node_feats) -> Dict:
        """Evaluate the GNN on cross-architecture test data."""
        if self.gnn_model is None:
            raise ValueError("GNN model not loaded. Call load_trained_gnn() first.")
        
        print("\n" + "=" * 60)
        print("GNN CROSS-ARCHITECTURE EVALUATION")
        print("=" * 60)
        
        # Evaluate using your existing evaluation function
        criterion = nn.MSELoss()
        test_loss, predictions, targets = evaluate_model(
            self.gnn_model, test_loader, self.device, node_feats, criterion
        )
        
        # Compute metrics using your existing function
        metrics = compute_metrics(predictions, targets)
        
        print(f"\nGNN Cross-Architecture Results:")
        print(f"Training Architecture: {self.train_architecture}")
        print(f"Test Architecture: {self.test_architecture}")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"Spearman's ρ: {metrics['spearman']:.6f}")
        print(f"Kendall's τ: {metrics['kendall']:.6f}")
        
        return {
            'test_loss': test_loss,
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_tss': results['targets'],
            'gnn_pred': results['predictions'],
            'residuals': np.array(results['predictions']) - np.array(results['targets']),
            'abs_residuals': np.abs(np.array(results['predictions']) - np.array(results['targets']))
        })
        
        results_path = os.path.join(output_dir, f'gnn_{self.train_architecture}_to_{self.test_architecture}_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'metric': list(results['metrics'].keys()) + ['test_loss'],
            'value': list(results['metrics'].values()) + [results['test_loss']]
        })
        
        metrics_path = os.path.join(output_dir, f'gnn_{self.train_architecture}_to_{self.test_architecture}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Metrics saved to: {metrics_path}")


def main():
    """Main function for GNN cross-architecture evaluation."""
    parser = argparse.ArgumentParser(description='GNN Cross-architecture TSS evaluation')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--gnn_model_path', type=str, required=True,
                       help='Path to trained GNN model (.pth file)')
    parser.add_argument('--train_arch', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Training data architecture')
    parser.add_argument('--test_arch', type=str, default='mobilenet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Test data architecture')
    parser.add_argument('--output_dir', type=str, default='gnn_cross_arch_results',
                       help='Output directory for results')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001,
                       help='TSS threshold for edge removal')
    parser.add_argument('--node_dim', type=int, default=None,
                       help='Node feature dimension (auto-inferred if not provided)')
    parser.add_argument('--edge_dim', type=int, default=None,
                       help='Edge feature dimension (auto-inferred if not provided)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for GNN')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GNN Cross-Architecture Evaluation for TSS Prediction")
    print("=" * 70)
    print(f"Training Architecture: {args.train_arch}")
    print(f"Test Architecture: {args.test_arch}")
    print(f"GNN Model: {args.gnn_model_path}")
    print(f"Training data: {args.train_csv}")
    print(f"Test data: {args.test_csv}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    try:
        # Initialize evaluator
        evaluator = GNNCrossArchitectureEvaluator(args.train_arch, args.test_arch)
        
        # Load trained GNN
        evaluator.load_trained_gnn(
            args.gnn_model_path,
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim
        )
        
        # Load and process data
        train_loader, val_loader, test_loader, train_node_feats, test_node_feats, processor = evaluator.load_and_process_data(
            args.train_csv, args.test_csv,
            enable_edge_removal=args.enable_edge_removal,
            tss_threshold=args.tss_threshold
        )
        
        if test_loader is None:
            print("\n❌ Cross-architecture data processing failed!")
            print("This typically happens when architectures have very different layer structures.")
            print("Consider using the XGBoost cross-architecture evaluator instead:")
            print(f"python cross_architecture_evaluator.py --train_csv \"{args.train_csv}\" --test_csv \"{args.test_csv}\" --train_arch {args.train_arch} --test_arch {args.test_arch}")
            return 1
        
        # Evaluate GNN
        results = evaluator.evaluate_gnn_cross_architecture(test_loader, train_node_feats)
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        print(f"\n{'='*20} GNN CROSS-ARCHITECTURE EVALUATION COMPLETE {'='*20}")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

