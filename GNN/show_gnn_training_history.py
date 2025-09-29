"""
Show GNN Training History
========================

This script displays the training history and epoch metrics from your GNN training sessions.
It looks for training logs and model checkpoints to show you the epoch-by-epoch progress.

Usage:
    python show_gnn_training_history.py --output_dir ./results
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
import pickle
from typing import Dict, List, Optional


def find_training_logs(directory: str) -> List[str]:
    """Find training log files in the directory."""
    log_files = []
    
    # Look for various log file patterns
    patterns = [
        "*.log",
        "*training*.txt",
        "*results*.csv",
        "*metrics*.json",
        "*logger*.pkl"
    ]
    
    for pattern in patterns:
        log_files.extend(glob.glob(os.path.join(directory, pattern)))
        log_files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
    
    return log_files


def extract_metrics_from_results_csv(csv_path: str) -> Optional[Dict]:
    """Extract metrics from results CSV if it contains training history."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if this is a results file with predictions
        if 'gnn_pred' in df.columns and 'true_tss' in df.columns:
            return {
                'type': 'final_results',
                'num_predictions': len(df),
                'rmse': np.sqrt(np.mean((df['gnn_pred'] - df['true_tss'])**2)),
                'mae': np.mean(np.abs(df['gnn_pred'] - df['true_tss'])),
                'correlation': np.corrcoef(df['gnn_pred'], df['true_tss'])[0, 1]
            }
        
        return None
        
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        return None


def analyze_model_checkpoint(model_path: str) -> Optional[Dict]:
    """Analyze a model checkpoint file."""
    try:
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'type': 'model_checkpoint',
            'file_size': os.path.getsize(model_path) / (1024 * 1024),  # MB
        }
        
        # Try to extract model information
        if isinstance(checkpoint, dict):
            # Count parameters
            param_count = 0
            for key, value in checkpoint.items():
                if torch.is_tensor(value):
                    param_count += value.numel()
            
            info['parameters'] = param_count
            info['layers'] = list(checkpoint.keys())[:10]  # First 10 layer names
            
        return info
        
    except Exception as e:
        print(f"Could not analyze model {model_path}: {e}")
        return None


def create_training_summary_plot(results_dir: str):
    """Create a summary plot of training results."""
    print(f"Analyzing training results in: {results_dir}")
    
    # Find all relevant files
    log_files = find_training_logs(results_dir)
    
    if not log_files:
        print("No training log files found!")
        return
    
    print(f"Found {len(log_files)} files to analyze:")
    
    analysis_results = []
    
    for file_path in log_files:
        print(f"  - {file_path}")
        
        file_info = {
            'file': os.path.basename(file_path),
            'path': file_path,
            'size': os.path.getsize(file_path),
            'modified': os.path.getmtime(file_path)
        }
        
        # Analyze based on file type
        if file_path.endswith('.csv'):
            metrics = extract_metrics_from_results_csv(file_path)
            if metrics:
                file_info.update(metrics)
                
        elif file_path.endswith('.pth'):
            model_info = analyze_model_checkpoint(file_path)
            if model_info:
                file_info.update(model_info)
        
        analysis_results.append(file_info)
    
    # Create summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    model_files = [r for r in analysis_results if r.get('type') == 'model_checkpoint']
    result_files = [r for r in analysis_results if r.get('type') == 'final_results']
    
    if model_files:
        print(f"\nModel Checkpoints Found: {len(model_files)}")
        for model in model_files:
            print(f"  • {model['file']}: {model['parameters']:,} parameters ({model['file_size']:.1f} MB)")
    
    if result_files:
        print(f"\nFinal Results Found: {len(result_files)}")
        for result in result_files:
            print(f"  • {result['file']}: {result['num_predictions']} predictions")
            print(f"    - RMSE: {result['rmse']:.6f}")
            print(f"    - MAE: {result['mae']:.6f}")
            print(f"    - Correlation: {result['correlation']:.6f}")
    
    # Create visualization if we have results
    if result_files:
        create_results_visualization(result_files, results_dir)


def create_results_visualization(result_files: List[Dict], output_dir: str):
    """Create visualization of results."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract metrics
        files = [r['file'] for r in result_files]
        rmse_values = [r['rmse'] for r in result_files]
        mae_values = [r['mae'] for r in result_files]
        corr_values = [r['correlation'] for r in result_files]
        
        # RMSE plot
        axes[0].bar(range(len(files)), rmse_values, alpha=0.7, color='skyblue')
        axes[0].set_title('RMSE by Model')
        axes[0].set_ylabel('RMSE')
        axes[0].set_xticks(range(len(files)))
        axes[0].set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], rotation=45)
        
        # MAE plot
        axes[1].bar(range(len(files)), mae_values, alpha=0.7, color='lightcoral')
        axes[1].set_title('MAE by Model')
        axes[1].set_ylabel('MAE')
        axes[1].set_xticks(range(len(files)))
        axes[1].set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], rotation=45)
        
        # Correlation plot
        axes[2].bar(range(len(files)), corr_values, alpha=0.7, color='lightgreen')
        axes[2].set_title('Correlation by Model')
        axes[2].set_ylabel('Correlation')
        axes[2].set_xticks(range(len(files)))
        axes[2].set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in files], rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'training_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSummary plot saved to: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Show GNN training history and results')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to search for training results (default: current directory)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search recursively in subdirectories')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Directory not found: {args.output_dir}")
        return 1
    
    print("=" * 60)
    print("GNN Training History Analyzer")
    print("=" * 60)
    print(f"Searching in: {args.output_dir}")
    print(f"Recursive search: {'Yes' if args.recursive else 'No'}")
    
    try:
        create_training_summary_plot(args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nNote: This tool analyzes saved results and model checkpoints.")
        print("For real-time epoch-by-epoch training, run the GNN training with:")
        print("python main.py --architecture efficientnet --epochs 40")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

