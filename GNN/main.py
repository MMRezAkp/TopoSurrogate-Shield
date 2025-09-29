"""
Main training script for GNN-based TSS prediction
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from config import DATA_PATHS, OUTPUT_PATHS, MODEL_CONFIG, LOSS_WEIGHTS, PREPROCESSING
from data_processing import DataProcessor
from models import create_model, get_model_info
from training_utils import train_model, train_mlp, compute_metrics
from evaluation import evaluate_and_save_results, create_visualizations

def setup_device() -> torch.device:
    """Setup and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def check_data_files(architecture: str, backdoored_path: str = None, clean_path: str = None) -> tuple:
    """Check if required data files exist."""
    # Use provided paths or fall back to config defaults
    if backdoored_path is None:
        if architecture.lower() == 'efficientnet':
            backdoored_path = DATA_PATHS['backdoored']
        elif architecture.lower() in ['mobilenet', 'mobilenetv2']:
            backdoored_path = DATA_PATHS['mobilenet_backdoored']
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    if clean_path is None:
        if architecture.lower() == 'efficientnet':
            clean_path = DATA_PATHS['clean']
        elif architecture.lower() in ['mobilenet', 'mobilenetv2']:
            clean_path = DATA_PATHS['mobilenet_clean']
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    if not os.path.exists(backdoored_path):
        raise FileNotFoundError(f"Backdoored data file not found: {backdoored_path}")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Clean data file not found: {clean_path}")
    
    print(f"✓ Found backdoored data: {backdoored_path}")
    print(f"✓ Found clean data: {clean_path}")
    
    return backdoored_path, clean_path

def print_system_info():
    """Print system and environment information."""
    print("=" * 60)
    print("GNN TSS Prediction Training")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("=" * 60)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GNN for TSS prediction')
    parser.add_argument('--architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Neural network architecture')
    parser.add_argument('--model_type', type=str, default='gnn',
                       choices=['gnn', 'mlp', 'residual_gnn'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['max_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'],
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=MODEL_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=MODEL_CONFIG['hidden_dim'],
                       help='Hidden dimension for models')
    parser.add_argument('--patience', type=int, default=MODEL_CONFIG['patience'],
                       help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--skip_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=PREPROCESSING['min_tss_threshold'],
                       help='TSS threshold for edge removal (default: 0.0001)')
    
    # Data paths
    parser.add_argument('--backdoored_path', type=str, default=None,
                       help='Path to backdoored data CSV file')
    parser.add_argument('--clean_path', type=str, default=None,
                       help='Path to clean data CSV file')
    
    # Physical loss weights
    parser.add_argument('--lambda_mono', type=float, default=LOSS_WEIGHTS['lambda_mono'],
                       help='Weight for monotonicity loss (default: 0.15)')
    parser.add_argument('--lambda_shortcut', type=float, default=LOSS_WEIGHTS['lambda_shortcut'],
                       help='Weight for shortcut awareness loss (default: 0.001)')
    parser.add_argument('--lambda_consistency', type=float, default=LOSS_WEIGHTS['lambda_consistency'],
                       help='Weight for consistency loss (default: 0.0)')
    parser.add_argument('--lambda_ranking', type=float, default=LOSS_WEIGHTS['lambda_ranking'],
                       help='Weight for ranking loss (default: 0.0)')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Setup device
    device = setup_device()
    
    # Check data files
    try:
        backdoored_path, clean_path = check_data_files(args.architecture, args.backdoored_path, args.clean_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nTraining Configuration:")
    print(f"Architecture: {args.architecture}")
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Backdoored Data: {backdoored_path}")
    print(f"Clean Data: {clean_path}")
    print(f"Edge Removal: {'Enabled' if args.enable_edge_removal else 'Disabled'}")
    if args.enable_edge_removal:
        print(f"TSS Threshold: {args.tss_threshold}")
    print(f"\nPhysical Loss Weights:")
    print(f"  Monotonicity (λ_mono): {args.lambda_mono}")
    print(f"  Shortcut (λ_shortcut): {args.lambda_shortcut}")
    print(f"  Consistency (λ_consistency): {args.lambda_consistency}")
    print(f"  Ranking (λ_ranking): {args.lambda_ranking}")
    
    try:
        # Step 1: Data Processing
        print(f"\n{'='*20} DATA PROCESSING {'='*20}")
        processor = DataProcessor(architecture=args.architecture)
        train_loader, val_loader, test_loader, edge_index, edge_attr, node_feats = processor.process_pipeline(
            backdoored_path, clean_path, enable_edge_removal=args.enable_edge_removal, tss_threshold=args.tss_threshold
        )
        
        # Get feature information
        feature_info = processor.get_feature_info()
        print(f"\nFeature Information:")
        print(f"Edge dimension: {feature_info['edge_dim']}")
        print(f"Node dimension: {feature_info['node_dim']}")
        print(f"Total features: {feature_info['total_features']}")
        
        # Step 2: Model Creation
        print(f"\n{'='*20} MODEL CREATION {'='*20}")
        model = create_model(
            args.model_type, 
            feature_info['node_dim'], 
            feature_info['edge_dim'],
            hidden_dim=args.hidden_dim,
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        model_info = get_model_info(model)
        print(f"Model: {model_info['model_type']}")
        print(f"Parameters: {model_info['total_params']:,}")
        print(f"Device: {model_info['device']}")
        
        # Step 3: Training
        print(f"\n{'='*20} TRAINING {'='*20}")
        
        # Create custom loss weights from command line arguments
        custom_loss_weights = {
            'lambda_mono': args.lambda_mono,
            'lambda_shortcut': args.lambda_shortcut,
            'lambda_consistency': args.lambda_consistency,
            'lambda_ranking': args.lambda_ranking
        }
        
        trained_model, gnn_logger = train_model(
            model, train_loader, val_loader, test_loader, node_feats, device,
            max_epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=MODEL_CONFIG['weight_decay'], patience=args.patience,
            loss_weights=custom_loss_weights
        )
        
        # Step 4: Evaluation and Results
        print(f"\n{'='*20} EVALUATION {'='*20}")
        results = evaluate_and_save_results(
            trained_model, test_loader, node_feats, device,
            args.output_dir, gnn_logger
        )
        
        # Step 5: Visualizations (if not skipped)
        if not args.skip_plots:
            print(f"\n{'='*20} VISUALIZATIONS {'='*20}")
            create_visualizations(results, gnn_logger, args.output_dir)
        
        # Step 6: Save preprocessors
        processor.save_preprocessors(args.output_dir)
        
        # Step 7: Final Summary
        print(f"\n{'='*20} TRAINING COMPLETE {'='*20}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Best GNN validation loss: {gnn_logger.get_best_metrics()['val_loss']:.6f}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
