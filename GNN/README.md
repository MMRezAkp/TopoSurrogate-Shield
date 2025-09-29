# GNN TSS Prediction Training Module

A modular, automated system for training Graph Neural Networks (GNNs) to predict TSS (Training Signal Strength) with support for both EfficientNet and MobileNet v2 architectures.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data processing, models, training, and evaluation
- **Multi-Architecture Support**: Supports both EfficientNet and MobileNet v2 layer encodings
- **Automated Training**: Command-line interface with configurable parameters
- **Physical Loss Functions**: Incorporates domain-specific loss functions for better TSS prediction
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and comparison with MLP baselines
- **Easy Configuration**: Centralized configuration management

## File Structure

```
├── config.py              # Configuration settings
├── layer_encoders.py      # Layer encoders for different architectures
├── data_processing.py     # Data loading and preprocessing
├── models.py              # Model definitions (GNN, MLP, Residual GNN)
├── loss_functions.py      # Physical loss functions
├── training_utils.py      # Training utilities and logging
├── evaluation.py          # Evaluation and visualization
├── main.py               # Main training script
└── README.md             # This file
```

## Installation

1. Install required dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn scipy
```

2. Ensure your data files are in the correct locations (see Configuration section).

## Quick Start

### 1. Prepare Your Data
Place your CSV files in a directory (e.g., `/data/`):
- `Efficient Net TSS backdoored.csv` - Backdoored EfficientNet data
- `Efficient Net TSS clean.csv` - Clean EfficientNet data
- `MobileNet TSS backdoored.csv` - Backdoored MobileNet v2 data  
- `MobileNet TSS clean.csv` - Clean MobileNet v2 data

### 2. Run Training
```bash
# EfficientNet training
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv

# MobileNet v2 training
python main.py \
    --architecture mobilenet \
    --backdoored_path /data/MobileNet TSS backdoored.csv \
    --clean_path /data/MobileNet TSS clean.csv
```

### 3. Check Results
After training, you'll find:
- `tss_predictions_with_shortcuts.csv` - Predictions and true values
- `gnn_shortcut_aware_model.pth` - Trained model
- `plots/` - Training curves and analysis plots
- `training_report.txt` - Summary metrics

## Usage

### Basic Usage

Train with EfficientNet data:
```bash
python main.py \
    --architecture efficientnet \
    --backdoored_path /path/to/Efficient Net TSS backdoored.csv \
    --clean_path /path/to/Efficient Net TSS clean.csv
```

Train with MobileNet v2 data:
```bash
python main.py \
    --architecture mobilenet \
    --backdoored_path /path/to/MobileNet TSS backdoored.csv \
    --clean_path /path/to/MobileNet TSS clean.csv
```

**Note**: Data paths are required! Replace `/path/to/` with your actual file paths.

### Architecture Selection

| Architecture | Command | Data Files Expected |
|-------------|---------|-------------------|
| **EfficientNet** | `--architecture efficientnet` | `Efficient Net TSS backdoored.csv`, `Efficient Net TSS clean.csv` |
| **MobileNet v2** | `--architecture mobilenet` | `MobileNet TSS backdoored.csv`, `MobileNet TSS clean.csv` |

### Quick Start Examples

```bash
# EfficientNet with custom data paths
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv

# MobileNet v2 with custom data paths
python main.py \
    --architecture mobilenet \
    --backdoored_path /data/MobileNet TSS backdoored.csv \
    --clean_path /data/MobileNet TSS clean.csv

# EfficientNet with custom loss weights
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --lambda_mono 0.3 \
    --lambda_shortcut 0.01
```

### Advanced Usage

```bash
python main.py \
    --architecture mobilenet \
    --backdoored_path /data/MobileNet TSS backdoored.csv \
    --clean_path /data/MobileNet TSS clean.csv \
    --model_type gnn \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_dim 128 \
    --patience 15 \
    --output_dir ./results
```

### Custom Data Paths

```bash
# Specify custom data files
python main.py \
    --architecture efficientnet \
    --backdoored_path /path/to/backdoored.csv \
    --clean_path /path/to/clean.csv

# Use custom data with MobileNet v2
python main.py \
    --architecture mobilenet \
    --backdoored_path /data/mobilenet_backdoored.csv \
    --clean_path /data/mobilenet_clean.csv
```

### Custom Physical Loss Weights

```bash
# High monotonicity constraint
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --lambda_mono 0.5 \
    --lambda_shortcut 0.01

# Strong shortcut awareness
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --lambda_shortcut 0.1 \
    --lambda_consistency 0.2

# All constraints enabled
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --lambda_mono 0.3 \
    --lambda_shortcut 0.05 \
    --lambda_consistency 0.1 \
    --lambda_ranking 0.05
```

### Edge Removal Options

```bash
# Enable edge removal with custom threshold
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --enable_edge_removal \
    --tss_threshold 0.001

# Disable edge removal (use all data)
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv

# Custom data with edge removal
python main.py \
    --architecture efficientnet \
    --backdoored_path /data/Efficient Net TSS backdoored.csv \
    --clean_path /data/Efficient Net TSS clean.csv \
    --enable_edge_removal \
    --tss_threshold 0.0005
```

### Command Line Arguments

- `--architecture`: Neural network architecture (`efficientnet`, `mobilenet`, `mobilenetv2`)
- `--model_type`: Model type (`gnn`, `mlp`, `residual_gnn`)
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.005)
- `--hidden_dim`: Hidden dimension for models (default: 64)
- `--patience`: Early stopping patience (default: 10)
- `--output_dir`: Output directory for results (default: current directory)
- `--skip_plots`: Skip generating plots
- `--enable_edge_removal`: Enable edge removal based on TSS threshold
- `--tss_threshold`: TSS threshold for edge removal (default: 0.0001)

#### Data Paths
- `--backdoored_path`: Path to backdoored data CSV file (optional, uses config defaults)
- `--clean_path`: Path to clean data CSV file (optional, uses config defaults)

#### Physical Loss Weights
- `--lambda_mono`: Weight for monotonicity loss (default: 0.15)
- `--lambda_shortcut`: Weight for shortcut awareness loss (default: 0.001)
- `--lambda_consistency`: Weight for consistency loss (default: 0.0)
- `--lambda_ranking`: Weight for ranking loss (default: 0.0)

## Configuration

Edit `config.py` to modify:

### Data Paths
```python
DATA_PATHS = {
    'backdoored': 'Efficient Net TSS backdoored.csv',
    'clean': 'Efficient Net TSS clean.csv',
    'mobilenet_backdoored': 'MobileNet TSS backdoored.csv',
    'mobilenet_clean': 'MobileNet TSS clean.csv',
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    'hidden_dim': 64,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'weight_decay': 1e-5,
    'max_epochs': 40,
    'batch_size': 32,
    'patience': 10,
}
```

### Loss Weights
```python
LOSS_WEIGHTS = {
    'lambda_mono': 0.15,
    'lambda_shortcut': 0.001,
    'lambda_consistency': 0.0,
    'lambda_ranking': 0.0,
}
```

## Data Format

Your CSV files should contain the following columns:

### Required Columns
- `src_layer`: Source layer name
- `dst_layer`: Destination layer name
- `src_ch`: Source channel index
- `dst_ch`: Destination channel index
- `tss`: Target TSS values

### Feature Columns
- Weight features: `w_out_norm_a`, `w_out_norm_b`, `w_norm_ratio`
- Layer position: `depth_a`, `depth_b`, `depth_diff`
- Connectivity: `fan_in_a`, `fan_in_b`, `fan_out_a`, `fan_out_b`, `fan_in_ratio`, `fan_out_ratio`
- Gradients: `grad_out_norm_a`, `grad_out_norm_b`, `grad_norm_ratio`
- Activations: `act_mean_a`, `act_mean_b`, `act_var_a`, `act_var_b`, `act_std_a`, `act_std_b`, `act_mean_diff`, `act_std_ratio`

## Output Files

The training process generates:

### Results
- `tss_predictions_with_shortcuts.csv`: Predictions and true values
- `training_report.txt`: Summary report with metrics

### Models
- `gnn_shortcut_aware_model.pth`: Trained GNN model
- `mlp_model.pth`: Trained MLP model (if trained)

### Preprocessors
- `scaler.pkl`: Feature scaler
- `encoder.pkl`: Categorical encoder

### Visualizations (in `plots/` directory)
- `training_curves.png`: Training and validation curves
- `prediction_scatter.png`: Prediction vs true value scatter plots
- `residual_analysis.png`: Residual analysis plots
- `loss_analysis.png`: Loss component analysis

## Model Types

### GNN (Graph Neural Network)
- Uses Graph Attention Network (GAT) layers
- Incorporates both node and edge features
- Includes physical loss functions for better TSS prediction

### MLP (Multi-Layer Perceptron)
- Baseline model for comparison
- Uses only edge features
- Simpler architecture for comparison

### Residual GNN
- GNN with residual connections
- Better gradient flow for deeper networks

## Physical Loss Functions

The system includes several domain-specific loss functions:

1. **Monotonicity Loss**: Ensures activation variance relationships are maintained
2. **Shortcut Awareness Loss**: Highlights the effect of shortcuts in TSS prediction
3. **Consistency Loss**: Ensures TSS predictions are consistent with shortcut characteristics
4. **Ranking Loss**: Maintains proper ranking based on shortcut characteristics

## Architecture Support

### EfficientNet
- Supports EfficientNet-B0 layer structure
- Detailed layer mapping for all stages and blocks

### MobileNet v2
- Supports MobileNet v2 layer structure
- Includes inverted residual blocks and depthwise convolutions
- Comprehensive layer mapping for all 18 blocks

## Example Workflow

1. **Prepare Data**: Ensure your CSV files are in the correct format and location
2. **Configure**: Edit `config.py` if needed
3. **Train**: Run `python main.py --architecture mobilenet --epochs 50`
4. **Analyze**: Check the generated plots and results
5. **Compare**: Review the training report for model comparison

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure data files are in the correct paths specified in `config.py`
2. **CUDA Out of Memory**: Reduce batch size or hidden dimension
3. **Poor Performance**: Try adjusting loss weights or model architecture
4. **Missing Dependencies**: Install all required packages

### Performance Tips

1. Use GPU if available (automatically detected)
2. Adjust batch size based on available memory
3. Tune learning rate and loss weights for your specific data
4. Use early stopping to prevent overfitting

## Contributing

To add support for new architectures:

1. Create a new encoder class in `layer_encoders.py` inheriting from `BaseLayerEncoder`
2. Implement the `_build_encoding()` method with your layer mapping
3. Add the architecture to the `get_layer_encoder()` factory function
4. Update the data paths in `config.py` if needed

## License

This project is provided as-is for research and educational purposes.
