# Neural Network Analysis Workflow Guide

This guide explains how to use the three-stage pipeline for neural network backdoor detection using topological analysis and graph neural networks.

## Quick Start

### Prerequisites
```bash
# Install base requirements
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn

# Install GNN-specific requirements
cd GNN/
pip install -r requirements.txt

# Install ASR analysis tools
cd ../ASR_ANALYSES/tss_comparison/
pip install -e .
```

### Complete Pipeline Example

```bash
# Stage 1: Extract TSS from a backdoored model
cd TSS/
python activation_topotroj.py \
    --model_type backdoored \
    --input_type clean \
    --model_path ../models/backdoored_resnet18.pth \
    --run_topology \
    --output_dir ../data/tss_output

# Stage 2: Train GNN to predict TSS
cd ../GNN/
python main.py \
    --train_backdoored ../data/backdoored_tss.csv \
    --train_clean ../data/clean_tss.csv \
    --architecture efficientnet \
    --output_dir ../models/gnn_output

# Stage 3: Analyze ASR using trained GNN
cd ../ASR_ANALYSES/tss_comparison/
python scripts/gnn_prediction_analysis.py \
    --model_path ../../models/backdoored_resnet18.pth \
    --gnn_path ../../models/gnn_output/gnn_model.pth \
    --save_results results/asr_analysis.json
```

## Detailed Stage Instructions

### Stage 1: TSS Extraction

The TSS stage involves a complete 5-step pipeline for both clean and backdoored models:

#### Step 1: Model Training
```bash
cd TSS/

# Train clean model
python main.py --architecture resnet18 --epochs 100 --batch-size 128 --learning-rate 0.001 --output-dir ./experiment_output

# Train backdoored model  
python main.py --architecture resnet18 --epochs 100 --batch-size 128 --learning-rate 0.001 --poison-ratio 0.1 --trigger-size 3 --target-label 0 --output-dir ./experiment_output
```

#### Step 2: Activation Extraction
```bash
cd TSS/

# Extract activations from clean model
python activation_topotroj.py \
    --model_type clean \
    --input_type clean \
    --model_path ../models/clean_model.pth \
    --output_dir ./activation_output_topo \
    --tap_mode topotroj_compat \
    --architecture resnet18

# Extract activations from backdoored model  
python activation_topotroj.py \
    --model_type backdoored \
    --input_type clean \
    --model_path ../models/backdoored_model.pth \
    --output_dir ./activation_output_topo \
    --tap_mode topotroj_compat \
    --architecture resnet18

# Extract activations from backdoored model with triggered inputs
python activation_topotroj.py \
    --model_type backdoored \
    --input_type triggered \
    --model_path ../models/backdoored_model.pth \
    --output_dir ./activation_output_topo \
    --tap_mode topotroj_compat \
    --trigger_pattern_size 3 \
    --trigger_pixel_value 1.0 \
    --architecture resnet18
```

#### Step 3: Correlation Matrix Building
```bash
# Build correlation matrices from clean model activations
python build_correlation_matrix.py \
    --activations_dir ./activation_output_topo \
    --model_type clean \
    --input_type clean \
    --method pearson \
    --knn_k 8 \
    --output_base_dir ./correlation_output \
    --save_similarity \
    --save_dense_distance

# Build correlation matrices from backdoored model activations
python build_correlation_matrix.py \
    --activations_dir ./activation_output_topo \
    --model_type backdoored \
    --input_type clean \
    --method pearson \
    --knn_k 8 \
    --output_base_dir ./correlation_output \
    --save_similarity \
    --save_dense_distance

# Build correlation matrices from triggered inputs
python build_correlation_matrix.py \
    --activations_dir ./activation_output_topo \
    --model_type backdoored \
    --input_type triggered \
    --method pearson \
    --knn_k 8 \
    --output_base_dir ./correlation_output \
    --save_similarity \
    --save_dense_distance
```

#### Step 4: Persistent Homology Computation
```bash
# Compute persistent homology for clean model
python compute_persistence_from_corr.py \
    --correlation_base_dir ./correlation_output \
    --model_type clean \
    --input_type clean \
    --correlation_method pearson \
    --output_base_dir ./ph_output \
    --do_cocycles \
    --save_plots

# Compute persistent homology for backdoored model
python compute_persistence_from_corr.py \
    --correlation_base_dir ./correlation_output \
    --model_type backdoored \
    --input_type clean \
    --correlation_method pearson \
    --output_base_dir ./ph_output \
    --do_cocycles \
    --save_plots

# Compute persistent homology for triggered inputs
python compute_persistence_from_corr.py \
    --correlation_base_dir ./correlation_output \
    --model_type backdoored \
    --input_type triggered \
    --correlation_method pearson \
    --output_base_dir ./ph_output \
    --do_cocycles \
    --save_plots
```

#### Step 5: TSS Computation
```bash
# Compute TSS for clean model
python compute_tss.py \
    --model_type clean \
    --input_type clean \
    --model_ckpt ../models/clean_model.pth \
    --correlation_output_base_dir ./correlation_output \
    --ph_output_base_dir ./ph_output \
    --activations_dir ./activation_output_topo \
    --architecture resnet18

# Compute TSS for backdoored model (with clean contrast)
python compute_tss.py \
    --model_type backdoored \
    --input_type clean \
    --model_ckpt ../models/backdoored_model.pth \
    --correlation_output_base_dir ./correlation_output \
    --ph_output_base_dir ./ph_output \
    --activations_dir ./activation_output_topo \
    --contrast_with_clean \
    --architecture resnet18

# Compute TSS for triggered inputs (with clean contrast)
python compute_tss.py \
    --model_type backdoored \
    --input_type triggered \
    --model_ckpt ../models/backdoored_model.pth \
    --correlation_output_base_dir ./correlation_output \
    --ph_output_base_dir ./ph_output \
    --activations_dir ./activation_output_topo \
    --contrast_with_clean \
    --architecture resnet18
```

#### Complete TSS Pipeline Output Files

**Model Files** (`experiment_output/models/`):
- `clean.pth`: Trained clean model state dict
- `backdoored.pth`: Trained backdoored model state dict
- `*_model_config.json`: Training configuration files
- `*_best_checkpoint.pth`: Full training checkpoints

**Activation Files** (`activation_output_topo/`):
- `*.pt`: Per-layer activation tensors
- `*_layer_catalog.csv`: Layer information for sampling  
- `*_activation_metadata.json`: Extraction metadata and timing

**Correlation Files** (`correlation_output/`):
- `pearson_corr.npy`: Correlation matrices
- `distance_sqeuclid.npy`: Distance matrices for persistent homology
- `knn_edges.csv`: k-NN graph edges
- `node_table.csv`: Node mapping (layer/channel information)

**Persistent Homology Files** (`ph_output/`):
- `h1_persistence.npy`: H1 persistence diagrams
- `h1_cocycles_mapped.json`: Mapped cocycles with layer information
- `barcodes_summary.json`: Summary of persistence features
- `H1_persistence_diagram.png`: Visualization plots

**TSS Files** (`ph_output/*/tss/`):
- `tss_per_edge.csv`: Edge-level TSS scores with features
- `tss_config.json`: TSS computation configuration
- `all_loops_edges_results.json`: Loop analysis results

### Stage 2: GNN Training

#### Train GNN for TSS Prediction
```bash
cd GNN/
python main.py \
    --train_backdoored "../data/Efficient Net TSS backdoored.csv" \
    --train_clean "../data/Efficient Net TSS clean.csv" \
    --architecture efficientnet \
    --model_type gnn \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 32 \
    --hidden_dim 64
```

#### Cross-Architecture Evaluation
```bash
python train_and_cross_eval_gnn.py \
    --train_arch efficientnet \
    --eval_arch mobilenet \
    --output_dir ../results/cross_eval
```

#### Hyperparameter Optimization
```bash
python run_optimization.py \
    --data_dir ../data/ \
    --architecture efficientnet \
    --n_trials 50
```

#### Output Files
- `gnn_model.pth`: Trained GNN model
- `scaler.pkl`: Feature scaler
- `encoder.pkl`: Categorical encoder
- `training_history.json`: Training metrics
- `evaluation_results.json`: Performance metrics

### Stage 3: ASR Analysis

#### Setup TSS Comparison Environment
```bash
cd ASR_ANALYSES/tss_comparison/
pip install -e .
```

#### GNN-Based ASR Analysis
```bash
python scripts/gnn_prediction_analysis.py \
    --model_path ../../models/backdoored.pth \
    --gnn_path ../../models/gnn_model.pth \
    --device cuda \
    --removal_ratio 0.15 \
    --save_results results/gnn_asr_analysis.json
```

#### Ground Truth ASR Analysis
```bash
python scripts/ground_truth_analysis.py \
    --model_path ../../models/backdoored.pth \
    --tss_data "../../data/Efficient Net TSS backdoored.csv" \
    --removal_ratio 0.15 \
    --save_results results/ground_truth_asr_analysis.json
```

#### Output Files
- `results/gnn_asr_analysis.json`: GNN-based ASR metrics
- `results/ground_truth_asr_analysis.json`: Ground truth ASR metrics
- `plots/`: Visualization plots
- Performance comparison reports

## Advanced Usage

### Custom Tap Modes
The TSS extraction supports different tap modes for various analysis needs:

```bash
# TopoTroj compatible mode (inputs to Conv2d + Linear)
python activation_topotroj.py --tap_mode topotroj_compat --include_fc

# TOAP block output mode (post-residual, post-ReLU)
python activation_topotroj.py --tap_mode toap_block_out

# Spatial preservation mode (maintain spatial dimensions)
python activation_topotroj.py --tap_mode spatial_preserve

# Legacy BatchNorm2d mode
python activation_topotroj.py --tap_mode bn2_legacy
```

### Multiple Architecture Analysis
```bash
# Analyze different architectures
for arch in resnet18 resnet34 efficientnet_b0 mobilenet_v2; do
    python activation_topotroj.py \
        --model_path ../models/${arch}_backdoored.pth \
        --architecture $arch \
        --output_dir ../data/tss_${arch}
done
```

### Batch Processing
```bash
# Process multiple models in batch
python extract_cifar10_batch.py \
    --model_dir ../models/ \
    --output_dir ../data/batch_tss/ \
    --architectures resnet18 efficientnet_b0
```

## Performance Optimization

### Memory Management
```bash
# For large models, use smaller batch sizes
python activation_topotroj.py --batch_size 16 --sample_limit 1000

# Monitor memory usage
python activation_topotroj.py --device cuda  # Will show memory stats
```

### Parallel Processing
```bash
# Use multiple workers for data loading
python activation_topotroj.py --num_workers 4

# Process multiple architectures in parallel
python GNN/cross_architecture_evaluator.py --parallel
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or use CPU
   python activation_topotroj.py --batch_size 8 --device cpu
   ```

2. **Architecture Auto-Detection Fails**:
   ```bash
   # Manually specify architecture
   python activation_topotroj.py --architecture resnet18
   ```

3. **Missing Dependencies**:
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   pip install -r GNN/requirements.txt
   cd ASR_ANALYSES/tss_comparison && pip install -e .
   ```

### Verification
```bash
# Test TSS extraction
cd TSS/
python test.py

# Test GNN training
cd GNN/
python -m pytest tests/ -v

# Test ASR analysis
cd ASR_ANALYSES/tss_comparison/
python -m pytest tests/ -v
```

## Integration with Research Workflow

### Paper Results Reproduction
```bash
# Run complete pipeline for paper results
./run_full_pipeline.sh  # (create this script based on your specific needs)
```

### Custom Analysis
```bash
# Create custom analysis scripts in each stage
# TSS/: custom_tss_analysis.py
# GNN/: custom_gnn_training.py  
# ASR_ANALYSES/: custom_asr_evaluation.py
```

This workflow provides a complete pipeline from raw neural network models to backdoor detection results, with clear separation of concerns and modular components that can be used independently or as an integrated system.
