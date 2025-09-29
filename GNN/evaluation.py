"""
Evaluation and visualization module
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

from training_utils import evaluate_model, evaluate_mlp, compute_metrics
from config import OUTPUT_PATHS

def evaluate_and_save_results(gnn_model: torch.nn.Module, test_loader: DataLoader, 
                             node_feats: torch.Tensor, device: torch.device,
                             output_dir: str, gnn_logger) -> Dict:
    """Evaluate models and save results."""
    print("Evaluating models...")
    
    # Evaluate GNN
    gnn_test_loss, gnn_preds, gnn_targets = evaluate_model(
        gnn_model, test_loader, device, node_feats, torch.nn.MSELoss()
    )
    gnn_metrics = compute_metrics(gnn_preds, gnn_targets)
    
    print(f"\nGNN Results:")
    print(f"Test Loss: {gnn_test_loss:.6f}")
    print(f"RMSE: {gnn_metrics['rmse']:.6f}")
    print(f"MAE: {gnn_metrics['mae']:.6f}")
    print(f"Spearman's ρ: {gnn_metrics['spearman']:.6f}")
    print(f"Kendall's τ: {gnn_metrics['kendall']:.6f}")
    
    # Create results DataFrame
    results_data = {
        'true_tss': gnn_targets,
        'gnn_pred': gnn_preds
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Save results
    results_path = os.path.join(output_dir, OUTPUT_PATHS['results'])
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save models
    torch.save(gnn_model.state_dict(), os.path.join(output_dir, OUTPUT_PATHS['gnn_model']))
    
    print(f"Models saved to: {output_dir}")
    
    return {
        'gnn_metrics': gnn_metrics,
        'gnn_preds': gnn_preds,
        'targets': gnn_targets,
        'results_df': results_df
    }

def create_visualizations(results: Dict, gnn_logger, output_dir: str):
    """Create and save visualization plots."""
    print("Creating visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, OUTPUT_PATHS['plots'])
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training curves
    create_training_curves(gnn_logger, plots_dir)
    
    # 2. Prediction scatter plots
    create_prediction_plots(results, plots_dir)
    
    # 3. Residual plots
    create_residual_plots(results, plots_dir)
    
    # 4. Loss component analysis
    create_loss_analysis(gnn_logger, plots_dir)
    
    print(f"Visualizations saved to: {plots_dir}")

def create_training_curves(gnn_logger, plots_dir: str):
    """Create training curve plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(gnn_logger.metrics['train_loss'], label='GNN Train', alpha=0.8)
    axes[0, 0].plot(gnn_logger.metrics['val_loss'], label='GNN Val', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test RMSE
    axes[0, 1].plot(gnn_logger.metrics['test_rmse'], label='GNN', alpha=0.8)
    axes[0, 1].set_title('Test RMSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss components (GNN only)
    if hasattr(gnn_logger, 'loss_components'):
        axes[1, 0].plot(gnn_logger.loss_components['mse'], label='MSE', alpha=0.8)
        axes[1, 0].plot(gnn_logger.loss_components['mono'], label='Monotonicity', alpha=0.8)
        axes[1, 0].plot(gnn_logger.loss_components['shortcut'], label='Shortcut', alpha=0.8)
        axes[1, 0].set_title('GNN Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Epoch times
    axes[1, 1].plot(gnn_logger.metrics['epoch_times'], label='GNN', alpha=0.8)
    axes[1, 1].set_title('Epoch Times')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_plots(results: Dict, plots_dir: str):
    """Create prediction scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    targets = np.array(results['targets'])
    gnn_preds = np.array(results['gnn_preds'])
    
    # GNN predictions
    axes[0].scatter(targets, gnn_preds, alpha=0.6, s=20)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', alpha=0.8)
    axes[0].set_xlabel('True TSS')
    axes[0].set_ylabel('Predicted TSS')
    axes[0].set_title(f'GNN Predictions (RMSE: {results["gnn_metrics"]["rmse"]:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(targets, gnn_preds)[0, 1]
    axes[0].text(0.05, 0.95, f'R = {corr:.3f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residual analysis
    axes[1].scatter(targets, gnn_preds - targets, alpha=0.6, s=20, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('True TSS')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('GNN Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plots(results: Dict, plots_dir: str):
    """Create residual analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    targets = np.array(results['targets'])
    gnn_preds = np.array(results['gnn_preds'])
    
    # GNN residuals
    gnn_residuals = gnn_preds - targets
    
    axes[0, 0].scatter(targets, gnn_residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('True TSS')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('GNN Residuals vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # GNN residual histogram
    axes[0, 1].hist(gnn_residuals, bins=50, alpha=0.7, density=True)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('GNN Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    from scipy import stats
    stats.probplot(gnn_residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs predicted values
    axes[1, 1].scatter(gnn_preds, gnn_residuals, alpha=0.6, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Predicted TSS')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Predicted Values')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_loss_analysis(gnn_logger, plots_dir: str):
    """Create loss component analysis plots."""
    if not hasattr(gnn_logger, 'loss_components'):
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss components over time
    axes[0, 0].plot(gnn_logger.loss_components['mse'], label='MSE', alpha=0.8)
    axes[0, 0].plot(gnn_logger.loss_components['mono'], label='Monotonicity', alpha=0.8)
    axes[0, 0].plot(gnn_logger.loss_components['shortcut'], label='Shortcut', alpha=0.8)
    axes[0, 0].plot(gnn_logger.loss_components['consistency'], label='Consistency', alpha=0.8)
    axes[0, 0].plot(gnn_logger.loss_components['ranking'], label='Ranking', alpha=0.8)
    axes[0, 0].set_title('Loss Components Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss component ratios
    total_loss = np.array(gnn_logger.loss_components['mse']) + \
                 np.array(gnn_logger.loss_components['mono']) + \
                 np.array(gnn_logger.loss_components['shortcut']) + \
                 np.array(gnn_logger.loss_components['consistency']) + \
                 np.array(gnn_logger.loss_components['ranking'])
    
    mse_ratio = np.array(gnn_logger.loss_components['mse']) / (total_loss + 1e-8)
    mono_ratio = np.array(gnn_logger.loss_components['mono']) / (total_loss + 1e-8)
    shortcut_ratio = np.array(gnn_logger.loss_components['shortcut']) / (total_loss + 1e-8)
    
    axes[0, 1].plot(mse_ratio, label='MSE Ratio', alpha=0.8)
    axes[0, 1].plot(mono_ratio, label='Monotonicity Ratio', alpha=0.8)
    axes[0, 1].plot(shortcut_ratio, label='Shortcut Ratio', alpha=0.8)
    axes[0, 1].set_title('Loss Component Ratios')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final loss component distribution
    final_losses = [gnn_logger.loss_components[key][-1] for key in gnn_logger.loss_components.keys()]
    final_labels = list(gnn_logger.loss_components.keys())
    
    axes[1, 0].bar(final_labels, final_losses, alpha=0.7)
    axes[1, 0].set_title('Final Loss Components')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss component correlation
    loss_matrix = np.array([gnn_logger.loss_components[key] for key in gnn_logger.loss_components.keys()])
    corr_matrix = np.corrcoef(loss_matrix)
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(final_labels)))
    axes[1, 1].set_yticks(range(len(final_labels)))
    axes[1, 1].set_xticklabels(final_labels, rotation=45)
    axes[1, 1].set_yticklabels(final_labels)
    axes[1, 1].set_title('Loss Component Correlation')
    
    # Add correlation values to the plot
    for i in range(len(final_labels)):
        for j in range(len(final_labels)):
            text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results: Dict, gnn_logger, mlp_logger, output_dir: str):
    """Create a summary report of the training results."""
    report_path = os.path.join(output_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("GNN TSS Prediction Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        # GNN Results
        f.write("GNN Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"RMSE: {results['gnn_metrics']['rmse']:.6f}\n")
        f.write(f"MAE: {results['gnn_metrics']['mae']:.6f}\n")
        f.write(f"Spearman's ρ: {results['gnn_metrics']['spearman']:.6f}\n")
        f.write(f"Kendall's τ: {results['gnn_metrics']['kendall']:.6f}\n")
        f.write(f"Best Epoch: {gnn_logger.get_best_metrics()['epoch']}\n")
        f.write(f"Best Validation Loss: {gnn_logger.get_best_metrics()['val_loss']:.6f}\n\n")
        
        # MLP Results (if available)
        if mlp_logger and results['mlp_metrics']:
            f.write("MLP Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"RMSE: {results['mlp_metrics']['rmse']:.6f}\n")
            f.write(f"MAE: {results['mlp_metrics']['mae']:.6f}\n")
            f.write(f"Spearman's ρ: {results['mlp_metrics']['spearman']:.6f}\n")
            f.write(f"Kendall's τ: {results['mlp_metrics']['kendall']:.6f}\n")
            f.write(f"Best Epoch: {mlp_logger.get_best_metrics()['epoch']}\n")
            f.write(f"Best Validation Loss: {mlp_logger.get_best_metrics()['val_loss']:.6f}\n\n")
        
        # Model Comparison
        if mlp_logger and results['mlp_metrics']:
            f.write("Model Comparison:\n")
            f.write("-" * 20 + "\n")
            rmse_improvement = (results['mlp_metrics']['rmse'] - results['gnn_metrics']['rmse']) / results['mlp_metrics']['rmse'] * 100
            f.write(f"RMSE Improvement: {rmse_improvement:.2f}%\n")
            f.write(f"Best GNN RMSE: {results['gnn_metrics']['rmse']:.6f}\n")
            f.write(f"Best MLP RMSE: {results['mlp_metrics']['rmse']:.6f}\n")
    
    print(f"Summary report saved to: {report_path}")
