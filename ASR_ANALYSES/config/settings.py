"""
Configuration settings for TSS comparison tools.
"""

import os
from pathlib import Path

# Base paths - these should point to the main project directory
BASE_DIR = Path(__file__).parent.parent.parent
MAIN_PROJECT_DIR = BASE_DIR

# Model paths (relative to main project directory)
MODEL_PATHS = {
    'efficientnet_model': os.path.join(MAIN_PROJECT_DIR, 'tss_comparison', 'backdoored.pth'),
    'gnn_model': os.path.join(MAIN_PROJECT_DIR, 'tss_comparison', 'gnn_shortcut_aware_model.pth'),
    'scaler': os.path.join(MAIN_PROJECT_DIR, 'tss_comparison', 'scripts', 'scaler.pkl'),
    'encoder': os.path.join(MAIN_PROJECT_DIR, 'tss_comparison', 'scripts', 'encoder.pkl'),
}

# Data paths (relative to main project directory)
DATA_PATHS = {
    'tss_backdoored': os.path.join(MAIN_PROJECT_DIR, 'Efficient Net TSS backdoored.csv'),
    'tss_clean': os.path.join(MAIN_PROJECT_DIR, 'Efficient Net TSS clean.csv'),
    'cifar10_data': os.path.join(MAIN_PROJECT_DIR, 'data'),
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'device': 'cuda',
    'batch_size': 64,
    'max_layers': 20,
    'max_edges': 10000,
    'removal_ratio': 0.15,
    'poison_ratio': 0.1,
    'target_label': 0,
    'trigger_size': 3,
    'trigger_color': (1.0, 1.0, 1.0),
    'trigger_position': 'bottom_right',
}

# CIFAR-10 normalization parameters
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# TSS prediction parameters
TSS_CONFIG = {
    'eps': 1e-8,  # Small epsilon for positive TSS values
    'hidden_dim': 64,
    'dropout': 0.2,
}

# Output settings
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'plots_dir': 'plots',
    'verbose': True,
}



