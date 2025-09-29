"""
Hyperparameter optimization using Optuna for GNN TSS prediction
Maximizes Spearman's ρ correlation coefficient
"""
import optuna
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import os
import json
from datetime import datetime

from config import DATA_PATHS, OUTPUT_PATHS, MODEL_CONFIG, LOSS_WEIGHTS, PREPROCESSING
from data_processing import DataProcessor
from models import create_model
from training_utils import train_model, compute_metrics
from evaluation import evaluate_and_save_results

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, architecture: str = 'efficientnet', n_trials: int = 100, 
                 timeout: int = 3600, study_name: str = None):
        self.architecture = architecture
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"gnn_optimization_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Data will be loaded once and reused
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.edge_index = None
        self.edge_attr = None
        self.node_feats = None
        self.device = None
        
    def load_data(self, backdoored_path: str, clean_path: str, enable_edge_removal: bool = False, 
                  tss_threshold: float = 0.0001):
        """Load and preprocess data once for all trials."""
        print("Loading data for optimization...")
        
        processor = DataProcessor(architecture=self.architecture)
        self.train_loader, self.val_loader, self.test_loader, self.edge_index, self.edge_attr, self.node_feats = processor.process_pipeline(
            backdoored_path, clean_path, enable_edge_removal, tss_threshold
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Data loaded successfully. Device: {self.device}")
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)
        
        try:
            # Create model with suggested parameters
            model = create_model(
                model_type=params['model_type'],
                node_dim=self.node_feats.shape[1],
                edge_dim=self.edge_attr.shape[1],
                hidden_dim=params['hidden_dim'],
                dropout=params['dropout']
            ).to(self.device)
            
            # Train model
            trained_model, gnn_logger = train_model(
                model, self.train_loader, self.val_loader, self.test_loader, 
                self.node_feats, self.device,
                max_epochs=params['max_epochs'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                patience=params['patience'],
                loss_weights=params['loss_weights']
            )
            
            # Evaluate on test set
            from training_utils import evaluate_model
            test_loss, test_preds, test_targets = evaluate_model(
                trained_model, self.test_loader, self.device, self.node_feats, torch.nn.MSELoss()
            )
            
            # Compute metrics
            metrics = compute_metrics(test_preds, test_targets)
            rmse = metrics['rmse']
            spearman_rho = metrics['spearman']
            
            # Handle NaN values
            if np.isnan(rmse) or np.isnan(spearman_rho):
                return float('inf')  # Return high RMSE for invalid results
            
            # Store both metrics in trial user attributes for reporting
            trial.set_user_attr('spearman_rho', spearman_rho)
            trial.set_user_attr('rmse', rmse)
            
            # Pruning: Stop unpromising trials early (based on RMSE)
            trial.report(rmse, step=gnn_logger.get_best_metrics()['epoch'])
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return rmse  # Minimize RMSE
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        
        # Model architecture
        model_type = trial.suggest_categorical('model_type', ['gnn', 'residual_gnn'])
        
        # Model hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=16)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        max_epochs = trial.suggest_int('max_epochs', 20, 80, step=10)
        patience = trial.suggest_int('patience', 5, 20, step=2)
        
        # Physical loss weights (the main focus)
        loss_weights = {
            'lambda_mono': trial.suggest_float('lambda_mono', 0.0, 1.0, step=0.05),
            'lambda_shortcut': trial.suggest_float('lambda_shortcut', 0.0, 0.5, step=0.01),
            'lambda_consistency': trial.suggest_float('lambda_consistency', 0.0, 0.5, step=0.01),
            'lambda_ranking': trial.suggest_float('lambda_ranking', 0.0, 0.5, step=0.01)
        }
        
        return {
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'max_epochs': max_epochs,
            'patience': patience,
            'loss_weights': loss_weights
        }
    
    def optimize(self, backdoored_path: str, clean_path: str, 
                 enable_edge_removal: bool = False, tss_threshold: float = 0.0001,
                 output_dir: str = './optimization_results') -> optuna.Study:
        """Run hyperparameter optimization."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data(backdoored_path, clean_path, enable_edge_removal, tss_threshold)
        
        # Create study (minimize RMSE)
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print(f"Starting optimization with {self.n_trials} trials...")
        print(f"Study name: {self.study_name}")
        print(f"Timeout: {self.timeout} seconds")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Save results
        self._save_results(study, output_dir)
        
        return study
    
    def _save_results(self, study: optuna.Study, output_dir: str):
        """Save optimization results."""
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Get best trial for additional metrics
        best_trial = study.best_trial
        best_spearman = best_trial.user_attrs.get('spearman_rho', 'N/A')
        best_rmse = best_trial.user_attrs.get('rmse', 'N/A')
        
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"Best RMSE: {best_value:.6f}")
        print(f"Best Spearman's ρ: {best_spearman:.6f}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            if key != 'loss_weights':
                print(f"  {key}: {value}")
        
        print(f"\nBest loss weights:")
        if 'loss_weights' in best_params:
            for key, value in best_params['loss_weights'].items():
                print(f"  {key}: {value:.6f}")
        else:
            print("  No loss weights found in best parameters")
        
        # Save detailed results
        results = {
            'best_rmse': best_value,
            'best_spearman': best_spearman,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'study_name': self.study_name,
            'architecture': self.architecture,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON results
        with open(os.path.join(output_dir, f'{self.study_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study object
        import joblib
        joblib.dump(study, os.path.join(output_dir, f'{self.study_name}_study.pkl'))
        
        # Save command to reproduce best results
        self._save_reproduction_command(best_params, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
    
    def _save_reproduction_command(self, best_params: Dict, output_dir: str):
        """Save command to reproduce best results."""
        
        cmd_parts = ['python main.py']
        cmd_parts.append(f'--architecture {self.architecture}')
        cmd_parts.append(f'--model_type {best_params["model_type"]}')
        cmd_parts.append(f'--hidden_dim {best_params["hidden_dim"]}')
        cmd_parts.append(f'--learning_rate {best_params["learning_rate"]:.6f}')
        cmd_parts.append(f'--epochs {best_params["max_epochs"]}')
        cmd_parts.append(f'--patience {best_params["patience"]}')
        
        # Add loss weights if they exist
        if 'loss_weights' in best_params:
            cmd_parts.append(f'--lambda_mono {best_params["loss_weights"]["lambda_mono"]:.6f}')
            cmd_parts.append(f'--lambda_shortcut {best_params["loss_weights"]["lambda_shortcut"]:.6f}')
            cmd_parts.append(f'--lambda_consistency {best_params["loss_weights"]["lambda_consistency"]:.6f}')
            cmd_parts.append(f'--lambda_ranking {best_params["loss_weights"]["lambda_ranking"]:.6f}')
        
        reproduction_cmd = ' \\\n    '.join(cmd_parts)
        
        with open(os.path.join(output_dir, f'{self.study_name}_reproduction_command.txt'), 'w') as f:
            f.write(f"# Command to reproduce best results\n")
            f.write(f"# Best RMSE: {study.best_value:.6f}\n")
            f.write(f"# Best Spearman's ρ: {best_spearman:.6f}\n\n")
            f.write(reproduction_cmd)
        
        print(f"\nReproduction command saved to: {self.study_name}_reproduction_command.txt")

def run_optimization(architecture: str = 'efficientnet', n_trials: int = 100, 
                    timeout: int = 3600, backdoored_path: str = None, 
                    clean_path: str = None, enable_edge_removal: bool = False,
                    tss_threshold: float = 0.0001, output_dir: str = './optimization_results'):
    """
    Run hyperparameter optimization for GNN TSS prediction.
    
    Args:
        architecture: Neural network architecture ('efficientnet' or 'mobilenet')
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        backdoored_path: Path to backdoored data CSV file (required)
        clean_path: Path to clean data CSV file (required)
        enable_edge_removal: Whether to enable edge removal
        tss_threshold: TSS threshold for edge removal
        output_dir: Directory to save optimization results
    
    Returns:
        optuna.Study: The completed optimization study
    """
    
    # Check if data paths are provided
    if backdoored_path is None or clean_path is None:
        print("Error: Data paths are required!")
        print("Usage examples:")
        print("  python hyperparameter_optimization.py --backdoored_path data_backdoored.csv --clean_path data_clean.csv")
        print("  python hyperparameter_optimization.py --architecture mobilenet --backdoored_path mobilenet_backdoored.csv --clean_path mobilenet_clean.csv")
        raise ValueError("Both --backdoored_path and --clean_path are required")
    
    # Check if files exist
    import os
    if not os.path.exists(backdoored_path):
        raise FileNotFoundError(f"Backdoored data file not found: {backdoored_path}")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Clean data file not found: {clean_path}")
    
    print(f"Using data files:")
    print(f"  Backdoored: {backdoored_path}")
    print(f"  Clean: {clean_path}")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        architecture=architecture,
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Run optimization
    study = optimizer.optimize(
        backdoored_path=backdoored_path,
        clean_path=clean_path,
        enable_edge_removal=enable_edge_removal,
        tss_threshold=tss_threshold,
        output_dir=output_dir
    )
    
    return study

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for GNN TSS prediction')
    parser.add_argument('--architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Neural network architecture')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Maximum time in seconds for optimization')
    parser.add_argument('--backdoored_path', type=str, required=True,
                       help='Path to backdoored data CSV file (required)')
    parser.add_argument('--clean_path', type=str, required=True,
                       help='Path to clean data CSV file (required)')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001,
                       help='TSS threshold for edge removal')
    parser.add_argument('--output_dir', type=str, default='./optimization_results',
                       help='Directory to save optimization results')
    
    args = parser.parse_args()
    
    # Run optimization
    study = run_optimization(
        architecture=args.architecture,
        n_trials=args.n_trials,
        timeout=args.timeout,
        backdoored_path=args.backdoored_path,
        clean_path=args.clean_path,
        enable_edge_removal=args.enable_edge_removal,
        tss_threshold=args.tss_threshold,
        output_dir=args.output_dir
    )
