"""
Configuration file for GNN training module
"""
import os
from pathlib import Path

# Data paths
DATA_PATHS = {
    'backdoored': 'Efficient Net TSS backdoored.csv',
    'clean': 'Efficient Net TSS clean.csv',
    'mobilenet_backdoored': 'MobileNet TSS backdoored.csv',  # Add your MobileNet data paths
    'mobilenet_clean': 'MobileNet TSS clean.csv',
}

# Model architecture
MODEL_CONFIG = {
    'hidden_dim': 64,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'weight_decay': 1e-5,
    'max_epochs': 40,
    'batch_size': 32,
    'patience': 10,
}

# Loss weights
LOSS_WEIGHTS = {
    'lambda_mono': 0.15,
    'lambda_shortcut': 0.001,
    'lambda_consistency': 0.0,
    'lambda_ranking': 0.0,
}

# Data preprocessing
PREPROCESSING = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'min_tss_threshold': 0.0001,
}

# Output paths
OUTPUT_PATHS = {
    'results': 'tss_predictions_with_shortcuts.csv',
    'gnn_model': 'gnn_shortcut_aware_model.pth',
    'mlp_model': 'mlp_model.pth',
    'scaler': 'scaler.pkl',
    'encoder': 'encoder.pkl',
    'plots': 'plots/',
}

# Create output directories
for path in OUTPUT_PATHS.values():
    if path.endswith('/'):
        os.makedirs(path, exist_ok=True)




