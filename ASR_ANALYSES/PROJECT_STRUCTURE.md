# TSS Comparison Project Structure

## Overview
This project reorganizes the "Comparing TSS" folder into a proper git project structure with modular components that can access the main project files.

## Project Structure
```
tss_comparison/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── PROJECT_STRUCTURE.md              # This file
├── config/
│   └── settings.py                   # Configuration settings
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py            # GNN model definitions
│   │   └── efficientnet_models.py   # EfficientNet feature extractors
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py               # Data loader factory
│   │   └── feature_extractors.py   # Feature extraction utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── asr_analyzer.py         # GNN-based ASR analysis
│   │   └── ground_truth_evaluator.py # Ground truth TSS evaluation
│   └── utils/
│       ├── __init__.py
│       ├── layer_encoder.py        # Layer encoding utilities
│       └── model_utils.py          # Model utility functions
├── scripts/
│   ├── gnn_prediction_analysis.py   # Main script for GNN analysis
│   └── ground_truth_analysis.py     # Main script for ground truth analysis
└── tests/
    ├── __init__.py
    └── test_evaluation.py          # Basic unit tests
```

## Key Features

### 1. Modular Design
- **Models**: Separate modules for GNN and EfficientNet models
- **Data**: Data loading and feature extraction utilities
- **Evaluation**: Two different analysis approaches
- **Utils**: Common utility functions

### 2. Configuration Management
- Centralized settings in `config/settings.py`
- Paths point to main project directory
- Easy to modify parameters

### 3. Two Analysis Methods
- **GNN-based**: Uses trained GNN to predict TSS scores in real-time
- **Ground Truth**: Uses pre-computed TSS scores from CSV files

### 4. Main Project Integration
- Uses modules from the main project (data_processing.py, models.py, etc.)
- Accesses files from the main project directory
- Maintains compatibility with existing workflow

## Usage

### Installation
```bash
cd tss_comparison
pip install -e .
```

### Running Analysis

#### GNN-based Analysis
```bash
python scripts/gnn_prediction_analysis.py --model_path ../backdoored.pth --gnn_path ../gnn_shortcut_aware_model.pth
```

#### Ground Truth Analysis
```bash
python scripts/ground_truth_analysis.py --model_path ../backdoored.pth --tss_data "../Efficient Net TSS backdoored.csv"
```

### Running Tests
```bash
python -m pytest tests/
```

## Dependencies

The project requires the following files from the main project directory:
- `backdoored.pth` - Pre-trained EfficientNet model
- `gnn_shortcut_aware_model.pth` - Trained GNN model (for GNN analysis)
- `scaler.pkl` - Feature scaler (for GNN analysis)
- `encoder_fixed.pkl` - Categorical encoder (for GNN analysis)
- `Efficient Net TSS backdoored.csv` - Ground truth TSS data (for ground truth analysis)

## Benefits

1. **Clean Separation**: Each component has a single responsibility
2. **Reusability**: Modules can be imported and used independently
3. **Maintainability**: Easy to modify and extend
4. **Testability**: Unit tests for individual components
5. **Documentation**: Clear structure and documentation
6. **Git Ready**: Proper project structure for version control

## Migration from Original Files

The original files have been refactored as follows:
- `GNN based preds.py` → `src/evaluation/asr_analyzer.py` + `scripts/gnn_prediction_analysis.py`
- `ground truth usage.py` → `src/evaluation/ground_truth_evaluator.py` + `scripts/ground_truth_analysis.py`

Common functionality has been extracted into utility modules to avoid code duplication.



