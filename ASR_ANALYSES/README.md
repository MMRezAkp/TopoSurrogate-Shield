# TSS Comparison Tools

A modular toolkit for comparing TSS (Training Signal Strength) prediction methods and evaluating ASR (Attack Success Rate) performance.

## Project Structure

```
tss_comparison/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py
│   │   └── efficientnet_models.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── feature_extractors.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── asr_analyzer.py
│   │   └── ground_truth_evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── layer_encoder.py
│       └── model_utils.py
├── scripts/
│   ├── gnn_prediction_analysis.py
│   └── ground_truth_analysis.py
└── tests/
    ├── __init__.py
    └── test_evaluation.py
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Two Analysis Methods**:
  - GNN-based TSS prediction with real-time feature extraction
  - Ground truth TSS evaluation using pre-computed data
- **ASR Evaluation**: Comprehensive attack success rate analysis
- **Model Pruning**: Edge removal based on TSS scores
- **Performance Metrics**: Clean accuracy and ASR before/after pruning

## Installation

```bash
pip install -e .
```

## Usage

### GNN-based Analysis
```bash
python scripts/gnn_prediction_analysis.py --model_path ../backdoored.pth --gnn_path ../gnn_shortcut_aware_model.pth
```

### Ground Truth Analysis
```bash
python scripts/ground_truth_analysis.py --model_path ../backdoored.pth --tss_data ../Efficient\ Net\ TSS\ backdoored.csv
```

## Dependencies

- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- joblib
- tqdm

## Configuration

Edit `config/settings.py` to modify default paths and parameters.




