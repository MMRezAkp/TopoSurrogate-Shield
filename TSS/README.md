# TSS (Topological Summary Statistics) Module

This module contains the complete pipeline for extracting Topological Summary Statistics (TSS) from neural networks.

## Pipeline Overview

The TSS extraction involves 5 main steps:

### Step 1: Model Training
- **Script**: `main.py` / `train.py`
- **Purpose**: Train neural network models using different architectures
- **Output**: Trained model checkpoints (`.pth` files)

### Step 2: Activation Extraction
- **Script**: `activation_topotroj.py`
- **Purpose**: Extract neural network activations from trained models with multiple tap modes
- **Output**: Activation tensors (`.pt` files)

### Step 3: Correlation Computation  
- **Script**: `build_correlation_matrix.py`
- **Purpose**: Build correlation matrices and k-NN graphs from activations
- **Output**: Correlation matrices (`.npy`), k-NN edges (`.csv`), node mappings

### Step 4: Persistent Homology
- **Script**: `compute_persistence_from_corr.py` 
- **Purpose**: Compute persistence diagrams from correlation matrices
- **Output**: Persistence diagrams (`.npy`), cocycles (`.json`)

### Step 5: TSS Computation
- **Script**: `compute_tss.py`
- **Purpose**: Calculate edge-level TSS scores with robust perturbations
- **Output**: TSS scores (`.csv`) with comprehensive edge features

## Key Files

### Core Pipeline Scripts
- `main.py` / `train.py` - Model training with different architectures
- `activation_topotroj.py` - Main activation extraction with topological analysis
- `build_correlation_matrix.py` - Correlation matrix and graph construction
- `compute_persistence_from_corr.py` - Persistent homology computation
- `compute_tss.py` - TSS scoring with perturbation analysis

### Utilities and Support
- `topo_utils.py` - Topological analysis utilities
- `vis_corr_matrix.py` - Correlation matrix visualization
- `model.py` - Neural network model definitions
- `data.py` - Data loading and preprocessing utilities
- `utils.py` - General utility functions

### Additional Scripts
- `main.py` - Main TSS pipeline orchestrator
- `train.py` - Model training utilities
- `extract_cifar10_batch.py` - Batch processing for CIFAR-10
- `load_model_and_data.py` - Model and data loading helpers
- `test.py` - Testing utilities
- `check.py` - Verification scripts

## Usage

### Complete TSS Pipeline
```bash
# Step 1: Train models
python main.py --architecture resnet18 --epochs 100 --dataset cifar10  # Train clean model
python main.py --architecture resnet18 --epochs 100 --dataset cifar10 --inject_backdoor --trigger_size 3  # Train backdoored model

# Step 2: Extract activations
python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/clean_model.pth
python activation_topotroj.py --model_type backdoored --input_type clean --model_path ./models/backdoored_model.pth

# Step 3: Build correlation matrices  
python build_correlation_matrix.py --model_type clean --input_type clean --save_similarity --save_dense_distance
python build_correlation_matrix.py --model_type backdoored --input_type clean --save_similarity --save_dense_distance

# Step 4: Compute persistent homology
python compute_persistence_from_corr.py --model_type clean --input_type clean --do_cocycles --save_plots
python compute_persistence_from_corr.py --model_type backdoored --input_type clean --do_cocycles --save_plots

# Step 5: Compute TSS scores
python compute_tss.py --model_type clean --input_type clean --model_ckpt ./models/clean_model.pth
python compute_tss.py --model_type backdoored --input_type clean --model_ckpt ./models/backdoored_model.pth --contrast_with_clean
```

### Supported Architectures
- ResNet (18, 34, 152)
- EfficientNet-B0  
- MobileNetV2

### Tap Modes
- `topotroj_compat` - TopoTroj compatible extraction
- `toap_block_out` - TOAP block output extraction  
- `bn2_legacy` - BatchNorm2d output extraction
- `spatial_preserve` - Spatial structure preservation

## Output Structure

```
TSS_output/
├── activation_output_topo/     # Step 1 outputs
│   ├── *.pt                   # Activation tensors
│   ├── *_layer_catalog.csv    # Layer information
│   └── *_metadata.json        # Extraction metadata
├── correlation_output/         # Step 2 outputs  
│   ├── *_corr.npy             # Correlation matrices
│   ├── distance_*.npy         # Distance matrices
│   ├── knn_edges.csv          # k-NN graph edges
│   └── node_table.csv         # Node mappings
├── ph_output/                  # Step 3 outputs
│   ├── h1_persistence.npy     # Persistence diagrams
│   ├── *_cocycles_mapped.json # Mapped cocycles
│   └── *.png                  # Visualization plots
└── tss/                        # Step 4 outputs
    ├── tss_per_edge.csv       # Edge-level TSS scores
    └── tss_config.json        # Configuration
```

## Integration

The TSS module feeds into:
- **GNN Module**: TSS data trains Graph Neural Networks for prediction
- **ASR Analysis**: TSS scores used for attack success rate evaluation
