# Hyperparameter Optimization Guide

This guide explains how to use Optuna for hyperparameter optimization to minimize RMSE while reporting both RMSE and Spearman's ρ correlation coefficient.

## Overview

The optimization system searches for the best combination of:
- **Model parameters**: hidden dimensions, dropout, learning rate, etc.
- **Physical loss weights**: λ_mono, λ_shortcut, λ_consistency, λ_ranking
- **Training parameters**: epochs, patience, weight decay

## Quick Start

### 1. Basic Optimization
```bash
# EfficientNet optimization (50 trials, 30 minutes)
python hyperparameter_optimization.py \
    --architecture efficientnet \
    --backdoored_path /path/to/backdoored.csv \
    --clean_path /path/to/clean.csv \
    --n_trials 50 \
    --timeout 1800

# MobileNet v2 optimization
python hyperparameter_optimization.py \
    --architecture mobilenet \
    --backdoored_path /path/to/mobilenet_backdoored.csv \
    --clean_path /path/to/mobilenet_clean.csv \
    --n_trials 50 \
    --timeout 1800
```

### 2. Interactive Optimization
```bash
# Run interactive script
python run_optimization.py
```

### 3. Custom Data Paths
```bash
python hyperparameter_optimization.py \
    --architecture efficientnet \
    --backdoored_path /path/to/backdoored.csv \
    --clean_path /path/to/clean.csv \
    --n_trials 100 \
    --timeout 3600
```

## Optimization Parameters

### Model Parameters
- **hidden_dim**: 32-256 (step 16)
- **dropout**: 0.1-0.5 (step 0.05)
- **learning_rate**: 1e-4 to 1e-2 (log scale)
- **weight_decay**: 1e-6 to 1e-3 (log scale)
- **max_epochs**: 20-80 (step 10)
- **patience**: 5-20 (step 2)

### Physical Loss Weights (Main Focus)
- **λ_mono** (Monotonicity): 0.0-1.0 (step 0.05)
- **λ_shortcut** (Shortcut Awareness): 0.0-0.5 (step 0.01)
- **λ_consistency** (Consistency): 0.0-0.5 (step 0.01)
- **λ_ranking** (Ranking): 0.0-0.5 (step 0.01)

### Model Types
- **gnn**: Standard Graph Attention Network
- **residual_gnn**: GNN with residual connections

## Usage Examples

### 1. Quick Test (10 trials)
```bash
python hyperparameter_optimization.py \
    --architecture efficientnet \
    --backdoored_path /path/to/backdoored.csv \
    --clean_path /path/to/clean.csv \
    --n_trials 10 \
    --timeout 600
```

### 2. Full Optimization (100 trials, 1 hour)
```bash
python hyperparameter_optimization.py \
    --architecture mobilenet \
    --backdoored_path /path/to/mobilenet_backdoored.csv \
    --clean_path /path/to/mobilenet_clean.csv \
    --n_trials 100 \
    --timeout 3600 \
    --enable_edge_removal \
    --tss_threshold 0.001
```

### 3. Custom Configuration
```bash
python hyperparameter_optimization.py \
    --architecture efficientnet \
    --backdoored_path my_backdoored.csv \
    --clean_path my_clean.csv \
    --n_trials 200 \
    --timeout 7200 \
    --output_dir ./my_optimization_results
```

## Output Files

After optimization, you'll get:

### 1. Results JSON
```json
{
  "best_value": 0.854321,
  "best_params": {
    "model_type": "gnn",
    "hidden_dim": 128,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "patience": 10,
    "loss_weights": {
      "lambda_mono": 0.25,
      "lambda_shortcut": 0.05,
      "lambda_consistency": 0.1,
      "lambda_ranking": 0.02
    }
  }
}
```

### 2. Reproduction Command
```bash
python main.py \
    --architecture efficientnet \
    --model_type gnn \
    --hidden_dim 128 \
    --learning_rate 0.001000 \
    --epochs 50 \
    --patience 10 \
    --lambda_mono 0.250000 \
    --lambda_shortcut 0.050000 \
    --lambda_consistency 0.100000 \
    --lambda_ranking 0.020000
```

### 3. Study Object
- Pickle file containing the complete Optuna study
- Can be loaded for further analysis

## Optimization Strategies

### 1. Focus on Physical Weights
If you want to focus on physical constraint tuning:
```python
# Modify the suggest_hyperparameters function
def _suggest_hyperparameters(self, trial):
    # ... other parameters ...
    
    # More focused search on physical weights
    loss_weights = {
        'lambda_mono': trial.suggest_float('lambda_mono', 0.0, 0.8, step=0.05),
        'lambda_shortcut': trial.suggest_float('lambda_shortcut', 0.0, 0.3, step=0.01),
        'lambda_consistency': trial.suggest_float('lambda_consistency', 0.0, 0.3, step=0.01),
        'lambda_ranking': trial.suggest_float('lambda_ranking', 0.0, 0.2, step=0.01)
    }
```

### 2. Multi-Objective Optimization
To optimize both Spearman's ρ and RMSE:
```python
# Modify objective function
def objective(self, trial):
    # ... training code ...
    
    metrics = compute_metrics(test_preds, test_targets)
    spearman_rho = metrics['spearman']
    rmse = metrics['rmse']
    
    # Multi-objective: maximize spearman, minimize rmse
    return spearman_rho - 0.1 * rmse  # Weighted combination
```

### 3. Architecture-Specific Optimization
```bash
# Optimize EfficientNet
python hyperparameter_optimization.py --architecture efficientnet --n_trials 100

# Optimize MobileNet v2
python hyperparameter_optimization.py --architecture mobilenet --n_trials 100

# Compare results
```

## Monitoring Progress

### 1. Real-time Progress
The optimization shows progress bars and trial results in real-time.

### 2. Early Stopping
- Uses MedianPruner to stop unpromising trials early
- Saves time on trials that won't improve

### 3. Best Trial Tracking
- Continuously tracks the best trial
- Shows current best Spearman's ρ

## Tips for Better Results

### 1. Start Small
```bash
# Quick test first
python hyperparameter_optimization.py --n_trials 20 --timeout 600
```

### 2. Increase Trials Gradually
```bash
# Medium optimization
python hyperparameter_optimization.py --n_trials 50 --timeout 1800

# Full optimization
python hyperparameter_optimization.py --n_trials 100 --timeout 3600
```

### 3. Use Edge Removal
```bash
# Often improves results
python hyperparameter_optimization.py --enable_edge_removal --tss_threshold 0.001
```

### 4. Focus on Physical Weights
The physical loss weights (λ_mono, λ_shortcut, etc.) are often the most important parameters for TSS prediction.

## Troubleshooting

### 1. Out of Memory
- Reduce `hidden_dim` range
- Use smaller batch sizes
- Enable edge removal to reduce data size

### 2. Slow Convergence
- Increase `patience` range
- Use more trials
- Focus on learning rate tuning

### 3. Poor Results
- Check data quality
- Try different architectures
- Adjust physical weight ranges

## Advanced Usage

### 1. Custom Search Spaces
Modify `_suggest_hyperparameters` to customize search ranges.

### 2. Multi-Study Comparison
Run multiple studies and compare results:
```python
# Study 1: Basic
study1 = run_optimization(architecture='efficientnet', n_trials=50)

# Study 2: With edge removal
study2 = run_optimization(architecture='efficientnet', n_trials=50, enable_edge_removal=True)

# Compare
print(f"Basic: {study1.best_value:.6f}")
print(f"With edge removal: {study2.best_value:.6f}")
```

### 3. Resume Optimization
```python
import optuna

# Load existing study
study = optuna.load_study(study_name="my_study", storage="sqlite:///my_study.db")

# Continue optimization
study.optimize(objective, n_trials=50)
```

## Expected Results

### RMSE Values (Lower is Better):
- **Good**: 0.01-0.05
- **Very Good**: 0.005-0.01
- **Excellent**: <0.005

### Spearman's ρ Values (Higher is Better):
- **Good**: 0.7-0.8
- **Very Good**: 0.8-0.9
- **Excellent**: 0.9+

The optimization minimizes RMSE while tracking Spearman's ρ to find the best combination of physical constraint weights and model parameters for accurate TSS prediction.
