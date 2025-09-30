# Neural Network Analysis Pipeline for Backdoor Detection

A comprehensive 3-stage pipeline for neural network analysis using topological signal strength (TSS) and graph neural networks (GNN) for backdoor detection. This toolkit provides end-to-end capabilities from activation extraction to attack success rate analysis.

## Pipeline Overview

```
Stage 1: TSS Extraction → Stage 2: GNN Training → Stage 3: ASR Analysis
     (TSS folder)           (GNN folder)         (ASR_ANALYSES folder)
```

**Stage 1 (TSS/)**: Extract neural network activations and compute Topological Summary Statistics  
**Stage 2 (GNN/)**: Train Graph Neural Networks to predict TSS from model architectures  
**Stage 3 (ASR_ANALYSES/)**: Analyze Attack Success Rate using GNN predictions and TSS comparison

## Pipeline Features

### Stage 1: TSS Extraction (TSS/)
- **Multi-Architecture Support**: ResNet (18, 34, 152), MobileNetV2, EfficientNet-B0
- **Flexible Activation Extraction**: Multiple tap modes for different analysis needs
- **Topological Analysis**: Persistent homology computation and topological summary statistics
- **Performance Tracking**: Built-in timing and complexity analysis
- **Memory Efficient**: Optimized for large-scale activation extraction

### Stage 2: GNN Training (GNN/)
- **Graph Neural Networks**: GAT-based models for TSS prediction
- **Multi-Architecture Encoders**: Support for different neural network architectures
- **Physical Loss Functions**: Domain-specific loss functions for better training
- **Cross-Architecture Evaluation**: Train on one architecture, test on another

### Stage 3: ASR Analysis (ASR_ANALYSES/)
- **Real-time GNN Predictions**: Use trained GNNs for TSS prediction and model analysis
- **Ground Truth Comparison**: Compare GNN predictions with actual TSS values
- **Model Pruning**: Remove edges based on TSS scores for backdoor mitigation
- **Attack Success Rate Metrics**: Comprehensive ASR evaluation before/after pruning

## Quick Start

### Complete Pipeline
```bash
# Run the full 3-stage pipeline
./run_full_pipeline.sh models/backdoored_model.pth models/clean_model.pth resnet18 results/my_analysis

# Or on Windows:
run_full_pipeline.bat models/backdoored_model.pth models/clean_model.pth resnet18 results/my_analysis
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/MMRezAkp/TopoSurrogate-Shield.git
```

2. Install dependencies for each stage:
```bash
# Base requirements
pip install -r requirements.txt

# GNN-specific requirements
cd GNN/
pip install -r requirements.txt

# ASR analysis tools
cd ../ASR_ANALYSES/tss_comparison/
pip install -e .
cd ../..
```

## Requirements

The tool requires the following Python packages:
- PyTorch
- torchvision
- numpy
- pandas
- tqdm
- psutil

See `requirements.txt` for specific versions.

## Usage by Stage

### Stage 1: TSS Extraction (Complete Pipeline)
```bash
cd TSS/

# Step 1: Train models with different architectures
python main.py --architecture resnet18 --epochs 100 --output-dir ./experiment_output
# This creates both clean and backdoored models with poison-ratio, trigger settings

# Step 2: Extract activations for both models  
python activation_topotroj.py --model_type clean --input_type clean --model_path ./experiment_output/models/clean.pth
python activation_topotroj.py --model_type backdoored --input_type clean --model_path ./experiment_output/models/backdoored.pth

# Step 3: Build correlation matrices
python build_correlation_matrix.py --model_type clean --input_type clean --save_similarity --save_dense_distance
python build_correlation_matrix.py --model_type backdoored --input_type clean --save_similarity --save_dense_distance

# Step 4: Compute persistent homology
python compute_persistence_from_corr.py --model_type clean --input_type clean --do_cocycles --save_plots
python compute_persistence_from_corr.py --model_type backdoored --input_type clean --do_cocycles --save_plots

# Step 5: Compute TSS scores
python compute_tss.py --model_type clean --input_type clean --model_ckpt ./experiment_output/models/clean.pth
python compute_tss.py --model_type backdoored --input_type clean --model_ckpt ./experiment_output/models/backdoored.pth --contrast_with_clean
```

### Stage 2: GNN Training
```bash
cd GNN/

# Train GNN to predict TSS
python main.py --train_backdoored "../data/backdoored_tss.csv" --train_clean "../data/clean_tss.csv" --architecture resnet18

# Cross-architecture evaluation
python train_and_cross_eval_gnn.py --train_arch resnet18 --eval_arch efficientnet
```

### Stage 3: ASR Analysis
```bash
cd ASR_ANALYSES/tss_comparison/

# GNN-based ASR analysis
python scripts/gnn_prediction_analysis.py --model_path ../../models/backdoored.pth --gnn_path ../../models/gnn_model.pth

# Ground truth ASR analysis
python scripts/ground_truth_analysis.py --model_path ../../models/backdoored.pth --tss_data ../../data/tss_data.csv
```

### Tap Modes

The tool supports different tap modes for activation extraction:

- `topotroj_compat`: Extract inputs to Conv2d and optional Linear layers (TopoTroj compatible)
- `toap_block_out`: Extract BasicBlock outputs post-residual and post-ReLU (TOAP compatible)
- `bn2_legacy`: Extract BatchNorm2d outputs (legacy mode)
- `spatial_preserve`: Preserve spatial structure in activations

## Architecture Support

The tool automatically detects the model architecture from the state dict, but you can also specify it manually:

- ResNet-18, ResNet-34, ResNet-152
- MobileNetV2
- EfficientNet-B0

## Project Structure

```
Other Architectures/
├── TSS/                              # Stage 1: TSS Extraction
│   ├── activation_topotroj.py       # Main TSS extraction tool
│   ├── build_correlation_matrix.py  # Correlation matrix computation
│   ├── compute_persistence_from_corr.py # Persistent homology
│   ├── compute_tss.py               # TSS computation
│   ├── topo_utils.py                # Topological utilities
│   ├── model.py                     # Model definitions
│   └── README.md                    # TSS documentation
├── GNN/                             # Stage 2: GNN Training
│   ├── models.py                    # GNN architectures (GAT, GNNSurrogate)
│   ├── main.py                      # Training pipeline
│   ├── data_processing.py           # Graph data processing
│   ├── training_utils.py            # Training utilities
│   └── README.md                    # GNN documentation
└── ASR_ANALYSES/                    # Stage 3: ASR Analysis
    ├── scripts/                     # Analysis scripts
    │   ├── gnn_prediction_analysis.py # GNN-based ASR analysis
    │   └── ground_truth_analysis.py   # Ground truth analysis
    ├── src/evaluation/              # ASR evaluation engines
    ├── config/settings.py           # Analysis configuration
    └── README.md                    # ASR analysis documentation
```

## Output Files by Stage

### Stage 1 (TSS/):
- `*.pt`: Per-layer activation tensors
- `*_correlation_matrix.npy`: Correlation matrices
- `*_topological_analysis.json`: Persistent homology results
- `*_layer_catalog.csv`: Layer information

### Stage 2 (GNN/):
- `gnn_model.pth`: Trained GNN model
- `scaler.pkl` & `encoder.pkl`: Data preprocessing objects
- `training_history.json`: Training metrics

### Stage 3 (ASR_ANALYSES/):
- `*_asr_analysis.json`: ASR evaluation results
- `results/`: Analysis reports and metrics
- Performance comparison data

## Command Line Options

### Required Arguments
- `--model_type`: Type of model (`clean` or `backdoored`)
- `--input_type`: Type of inputs (`clean` or `triggered`)
- `--model_path`: Path to the model file

### Optional Arguments
- `--output_dir`: Output directory (default: `activation_output_topo`)
- `--dataset`: Dataset to use (`cifar10` or `cifar100`)
- `--batch_size`: Batch size for processing (default: 32)
- `--sample_limit`: Limit number of samples processed
- `--tap_mode`: Activation extraction mode
- `--architecture`: Force specific architecture
- `--run_topology`: Enable topological analysis

### Trigger Arguments (for backdoor detection)
- `--trigger_pattern_size`: Size of trigger pattern
- `--trigger_pixel_value`: Pixel value for trigger
- `--trigger_location`: Trigger location (`br`, `tl`, `tr`, `bl`)
- `--poison_target_label`: Target label for poisoned samples

### Topological Analysis Arguments
- `--threshold_min`: Minimum correlation threshold
- `--threshold_max`: Maximum correlation threshold
- `--num_thresholds`: Number of thresholds for persistence

## Model File Requirements

You need to provide a `model.py` file that implements a `get_model()` function:

```python
def get_model(num_classes=10, pretrained=False, architecture="resnet18"):
    # Your model implementation here
    return model
```

## Examples

### Example 1: Basic Usage
```bash
python activation_topotroj.py \
    --model_type clean \
    --input_type clean \
    --model_path ./models/resnet18_clean.pth \
    --tap_mode topotroj_compat \
    --include_fc
```

### Example 2: Topological Analysis
```bash
python activation_topotroj.py \
    --model_type clean \
    --input_type clean \
    --model_path ./models/model.pth \
    --run_topology \
    --threshold_min 0.2 \
    --threshold_max 0.8 \
    --num_thresholds 30
```

### Example 3: Backdoor Detection
```bash
python activation_topotroj.py \
    --model_type backdoored \
    --input_type triggered \
    --model_path ./models/backdoored.pth \
    --run_topology \
    --trigger_pattern_size 3 \
    --trigger_pixel_value 1.0 \
    --trigger_location br \
    --poison_target_label 0
```

## Detailed Documentation

- **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)**: Complete project structure and workflow explanation
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)**: Step-by-step usage guide for all stages
- **[TSS/README.md](TSS/README.md)**: TSS extraction documentation
- **[GNN/README.md](GNN/README.md)**: GNN training documentation  
- **[ASR_ANALYSES/tss_comparison/README.md](ASR_ANALYSES/tss_comparison/README.md)**: ASR analysis documentation

## Performance Features

- **Timing Tracking**: Detailed execution time analysis across all stages
- **Complexity Analysis**: Computational complexity metrics for each component
- **Memory Monitoring**: Real-time memory usage tracking
- **Parallel Processing**: Multi-architecture and cross-evaluation support
- **Automated Pipeline**: End-to-end automation with progress tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please consider citing:

```bibtex
@software{neural_activation_extractor,
  title={Neural Network Activation Extractor for Topological Analysis},
  author={[Mohamadreza Akbari Pour] and [Samaneh Shamshiri] and [Ali Dehghantanha]},
  year={2025},
  url={https://github.com/MMRezAkp/TopoSurrogate-Shield}
}
```

## Acknowledgments

This tool is designed to be compatible with TopoTroj and TOAP methodologies for topological analysis of neural networks.


