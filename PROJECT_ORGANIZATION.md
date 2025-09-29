# Neural Network Activation Extractor - Project Organization

This project follows a 3-stage workflow for neural network analysis using topological signal strength (TSS) and graph neural networks (GNN) for backdoor detection.

## Project Structure

```
Other Architectures/
├── TSS/                              # Stage 1: TSS Extraction
│   ├── activation_topotroj.py       # Main TSS extraction tool
│   ├── compute_tss.py               # TSS computation utilities
│   ├── build_correlation_matrix.py  # Correlation matrix building
│   ├── compute_persistence_from_corr.py # Persistent homology
│   ├── topo_utils.py                # Topological utilities
│   ├── vis_corr_matrix.py           # Visualization tools
│   ├── model.py                     # Model definitions
│   ├── data.py                      # Data utilities
│   ├── utils.py                     # General utilities
│   ├── load_model_and_data.py       # Model/data loading
│   ├── extract_cifar10_batch.py     # CIFAR-10 extraction
│   ├── test.py                      # Testing utilities
│   ├── check.py                     # Verification scripts
│   ├── main.py                      # Main TSS pipeline
│   └── ph_output/                   # TSS output directory
│       └── clean_model_clean_inputs/
│           └── tss/
├── GNN/                             # Stage 2: GNN Training & Prediction
│   ├── README.md                    # GNN module documentation
│   ├── config.py                    # GNN configuration
│   ├── models.py                    # GNN model definitions (GATLayer, GNNSurrogate, etc.)
│   ├── data_processing.py           # GNN data preprocessing
│   ├── layer_encoders.py            # Architecture-specific encoders
│   ├── training_utils.py            # Training utilities
│   ├── loss_functions.py            # Physical loss functions
│   ├── main.py                      # Main GNN training script
│   ├── evaluation.py                # GNN evaluation tools
│   ├── hyperparameter_optimization.py # Hyperparameter tuning
│   ├── cross_architecture_evaluator.py # Cross-architecture evaluation
│   ├── train_and_cross_eval_gnn.py  # Training and cross-evaluation
│   ├── run_evaluation.py            # Evaluation runner
│   ├── run_optimization.py          # Optimization runner
│   ├── show_gnn_training_history.py # Training history visualization
│   └── requirements.txt             # GNN-specific requirements
└── ASR_ANALYSES/                    # Stage 3: ASR Analysis & TSS Comparison
    └── tss_comparison/              # TSS comparison and ASR evaluation
        ├── README.md                # TSS comparison documentation
        ├── PROJECT_STRUCTURE.md     # Detailed structure documentation
        ├── setup.py                 # Package setup
        ├── requirements.txt         # TSS comparison requirements
        ├── config/
        │   └── settings.py          # Analysis configuration
        ├── src/
        │   ├── models/
        │   │   ├── gnn_models.py    # GNN model wrappers
        │   │   └── efficientnet_models.py # EfficientNet extractors
        │   ├── data/
        │   │   ├── loaders.py       # Data loader factory
        │   │   └── feature_extractors.py # Feature extraction
        │   ├── evaluation/
        │   │   ├── asr_analyzer.py  # ASR analysis with GNN predictions
        │   │   └── ground_truth_evaluator.py # Ground truth TSS evaluation
        │   └── utils/
        │       ├── layer_encoder.py # Layer encoding utilities
        │       └── model_utils.py   # Model utilities
        ├── scripts/
        │   ├── gnn_prediction_analysis.py # GNN-based ASR analysis
        │   └── ground_truth_analysis.py   # Ground truth ASR analysis
        └── tests/
            └── test_evaluation.py   # Unit tests
```

## Workflow Overview

### Stage 1: TSS Extraction (`TSS/`)
**Purpose**: Extract neural network activations and compute Topological Summary Statistics (TSS)

**Key Components**:
- `activation_topotroj.py`: Neural network activation extraction with topological analysis
- `build_correlation_matrix.py`: Channel-level correlation matrices with k-NN graphs
- `compute_persistence_from_corr.py`: Persistent homology computation with cocycle analysis
- `compute_tss.py`: Edge-level TSS scoring with robust perturbations
- `topo_utils.py`: Topological utilities and helper functions
- `vis_corr_matrix.py`: Correlation matrix visualization

**Input**: Trained neural network models (.pth files) for both clean and backdoored models
**Output**: Complete TSS data including correlation matrices, persistence diagrams, and edge-level TSS scores

**Usage Example**:
```bash
cd TSS/
python activation_topotroj.py --model_type clean --input_type clean --model_path ./models/model.pth --run_topology
```

### Stage 2: GNN Training & Prediction (`GNN/`)
**Purpose**: Train Graph Neural Networks to predict TSS values from model architectures

**Key Components**:
- `models.py`: GNN architectures (GATLayer, GNNSurrogate, ResidualGNN)
- `data_processing.py`: Convert model architectures to graph representations
- `training_utils.py`: Training loops and utilities
- `main.py`: Main training pipeline

**Input**: TSS data from Stage 1
**Output**: Trained GNN models that can predict TSS

**Usage Example**:
```bash
cd GNN/
python main.py --train_backdoored ../TSS/tss_data_backdoored.csv --train_clean ../TSS/tss_data_clean.csv
```

### Stage 3: ASR Analysis (`ASR_ANALYSES/tss_comparison/`)
**Purpose**: Use trained GNNs to analyze Attack Success Rate (ASR) and compare TSS prediction methods

**Key Components**:
- `evaluation/asr_analyzer.py`: Real-time ASR analysis using GNN predictions
- `evaluation/ground_truth_evaluator.py`: ASR analysis using ground truth TSS
- `scripts/gnn_prediction_analysis.py`: GNN-based analysis workflow
- `scripts/ground_truth_analysis.py`: Ground truth analysis workflow

**Input**: Trained GNN models from Stage 2, backdoored models for analysis
**Output**: ASR metrics, pruning results, performance comparisons

**Usage Examples**:
```bash
cd ASR_ANALYSES/tss_comparison/

# GNN-based analysis
python scripts/gnn_prediction_analysis.py --model_path ../backdoored.pth --gnn_path ../gnn_model.pth

# Ground truth analysis  
python scripts/ground_truth_analysis.py --model_path ../backdoored.pth --tss_data ../tss_data.csv
```

## Data Flow

```
Neural Network Models (.pth)
    ↓
[Step 1] Activation Extraction → Activation Tensors (.pt)
    ↓
[Step 2] Correlation Computation → Correlation Matrices (.npy) + k-NN Graphs (.csv)
    ↓  
[Step 3] Persistent Homology → Persistence Diagrams (.npy) + Cocycles (.json)
    ↓
[Step 4] TSS Computation → Edge-level TSS Scores (.csv)
    ↓
[GNN Training] → Trained GNN Model (.pth)
    ↓
[ASR Analysis] → Attack Success Rate Metrics (.json)
```

## Key Features by Stage

### TSS Stage Features:
- Multi-architecture support (ResNet, EfficientNet, MobileNetV2)
- Multiple tap modes (topotroj_compat, toap_block_out, spatial_preserve)
- Topological analysis (persistent homology, correlation matrices)
- Performance tracking and complexity analysis

### GNN Stage Features:
- Graph Attention Networks (GAT) for TSS prediction
- Multi-architecture layer encoders
- Physical loss functions for domain-specific training
- Cross-architecture evaluation capabilities

### ASR Analysis Stage Features:
- Real-time GNN-based TSS prediction
- Ground truth TSS comparison
- Model pruning based on TSS scores
- Comprehensive ASR evaluation metrics

## Integration Points

1. **TSS → GNN**: TSS data feeds into GNN training
2. **GNN → ASR**: Trained GNN models predict TSS for ASR analysis
3. **Cross-validation**: Results can be compared between prediction and ground truth methods

## Benefits of This Organization

1. **Clear Separation**: Each stage has distinct responsibilities
2. **Modular Design**: Stages can be run independently or as a pipeline
3. **Scalable**: Easy to add new architectures or analysis methods
4. **Maintainable**: Each component is self-contained with clear interfaces
5. **Research-Friendly**: Easy to experiment with different approaches at each stage
