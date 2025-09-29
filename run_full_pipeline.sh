#!/bin/bash

# Neural Network Analysis - Full Pipeline Runner
# This script runs the complete 3-stage analysis pipeline

set -e  # Exit on any error

# Configuration
MODEL_PATH="${1:-models/backdoored_resnet18.pth}"
CLEAN_MODEL_PATH="${2:-models/clean_resnet18.pth}"
ARCHITECTURE="${3:-resnet18}"
OUTPUT_DIR="${4:-results/pipeline_$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "Neural Network Analysis - Full Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Clean Model: $CLEAN_MODEL_PATH"
echo "Architecture: $ARCHITECTURE"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Stage 1: TSS Extraction (Complete Pipeline)
echo ""
echo "Stage 1: TSS Extraction (Complete Pipeline)"
echo "--------------------------------------------"
cd TSS/

# Create activation output directory
mkdir -p "../$OUTPUT_DIR/activation_output_topo"
mkdir -p "../$OUTPUT_DIR/correlation_output"  
mkdir -p "../$OUTPUT_DIR/ph_output"

echo "Step 1: Training neural network models..."
echo "Training clean model..."
python main.py \
    --architecture "$ARCHITECTURE" \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --output-dir "../$OUTPUT_DIR"

echo "Training backdoored model..."  
python main.py \
    --architecture "$ARCHITECTURE" \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --poison-ratio 0.1 \
    --trigger-size 3 \
    --target-label 0 \
    --output-dir "../$OUTPUT_DIR"

echo "Step 2: Extracting activations from trained models..."
echo "Extracting activations from backdoored model (clean inputs)..."
python activation_topotroj.py \
    --model_type backdoored \
    --input_type clean \
    --model_path "../$MODEL_PATH" \
    --architecture "$ARCHITECTURE" \
    --output_dir "../$OUTPUT_DIR/activation_output_topo" \
    --batch_size 32 \
    --sample_limit 2000

echo "Extracting activations from backdoored model (triggered inputs)..."
python activation_topotroj.py \
    --model_type backdoored \
    --input_type triggered \
    --model_path "../$MODEL_PATH" \
    --architecture "$ARCHITECTURE" \
    --trigger_pattern_size 3 \
    --trigger_pixel_value 1.0 \
    --trigger_location br \
    --poison_target_label 0 \
    --output_dir "../$OUTPUT_DIR/activation_output_topo" \
    --batch_size 32 \
    --sample_limit 2000

echo "Extracting activations from clean model..."
python activation_topotroj.py \
    --model_type clean \
    --input_type clean \
    --model_path "../$OUTPUT_DIR/models/clean.pth" \
    --architecture "$ARCHITECTURE" \
    --output_dir "../$OUTPUT_DIR/activation_output_topo" \
    --batch_size 32 \
    --sample_limit 2000

echo "Step 3: Building correlation matrices..."
python build_correlation_matrix.py \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --model_type backdoored \
    --input_type clean \
    --output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --method pearson \
    --knn_k 8 \
    --save_similarity \
    --save_dense_distance

python build_correlation_matrix.py \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --model_type backdoored \
    --input_type triggered \
    --output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --method pearson \
    --knn_k 8 \
    --save_similarity \
    --save_dense_distance

python build_correlation_matrix.py \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --model_type clean \
    --input_type clean \
    --output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --method pearson \
    --knn_k 8 \
    --save_similarity \
    --save_dense_distance

echo "Step 4: Computing persistent homology..."
python compute_persistence_from_corr.py \
    --correlation_base_dir "../$OUTPUT_DIR/correlation_output" \
    --model_type backdoored \
    --input_type clean \
    --output_base_dir "../$OUTPUT_DIR/ph_output" \
    --correlation_method pearson \
    --do_cocycles \
    --save_plots

python compute_persistence_from_corr.py \
    --correlation_base_dir "../$OUTPUT_DIR/correlation_output" \
    --model_type backdoored \
    --input_type triggered \
    --output_base_dir "../$OUTPUT_DIR/ph_output" \
    --correlation_method pearson \
    --do_cocycles \
    --save_plots

python compute_persistence_from_corr.py \
    --correlation_base_dir "../$OUTPUT_DIR/correlation_output" \
    --model_type clean \
    --input_type clean \
    --output_base_dir "../$OUTPUT_DIR/ph_output" \
    --correlation_method pearson \
    --do_cocycles \
    --save_plots

echo "Step 5: Computing TSS scores..."
python compute_tss.py \
    --model_type backdoored \
    --input_type clean \
    --model_ckpt "../$MODEL_PATH" \
    --correlation_output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --ph_output_base_dir "../$OUTPUT_DIR/ph_output" \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --architecture "$ARCHITECTURE" \
    --contrast_with_clean

python compute_tss.py \
    --model_type backdoored \
    --input_type triggered \
    --model_ckpt "../$MODEL_PATH" \
    --correlation_output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --ph_output_base_dir "../$OUTPUT_DIR/ph_output" \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --architecture "$ARCHITECTURE" \
    --contrast_with_clean

python compute_tss.py \
    --model_type clean \
    --input_type clean \
    --model_ckpt "../$OUTPUT_DIR/models/clean.pth" \
    --correlation_output_base_dir "../$OUTPUT_DIR/correlation_output" \
    --ph_output_base_dir "../$OUTPUT_DIR/ph_output" \
    --activations_dir "../$OUTPUT_DIR/activation_output_topo" \
    --architecture "$ARCHITECTURE"

cd ..

# Stage 2: GNN Training
echo ""
echo "Stage 2: GNN Training"
echo "---------------------"
cd GNN/

# Check if TSS data exists, if not create dummy data for demo
BACKDOORED_TSS="../$OUTPUT_DIR/tss_backdoored/backdoored_model_clean_inputs_topological_analysis.json"
CLEAN_TSS="../$OUTPUT_DIR/tss_clean/clean_model_clean_inputs_topological_analysis.json"

if [ ! -f "$BACKDOORED_TSS" ]; then
    echo "Warning: TSS data not found. Using existing TSS data for training..."
    BACKDOORED_TSS="../data/Efficient Net TSS backdoored.csv"
    CLEAN_TSS="../data/Efficient Net TSS clean.csv"
fi

echo "Training GNN with TSS data..."
python main.py \
    --train_backdoored "$BACKDOORED_TSS" \
    --train_clean "$CLEAN_TSS" \
    --architecture "$ARCHITECTURE" \
    --model_type gnn \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 32 \
    --hidden_dim 64 \
    --output_dir "../$OUTPUT_DIR/gnn_models"

echo "Running cross-architecture evaluation..."
python train_and_cross_eval_gnn.py \
    --train_arch "$ARCHITECTURE" \
    --output_dir "../$OUTPUT_DIR/gnn_cross_eval"

cd ..

# Stage 3: ASR Analysis
echo ""
echo "Stage 3: ASR Analysis"
echo "---------------------"
cd ASR_ANALYSES/tss_comparison/

# Setup environment if not already done
if [ ! -f "setup_done.flag" ]; then
    echo "Setting up TSS comparison environment..."
    pip install -e . > /dev/null 2>&1
    touch setup_done.flag
fi

echo "Running GNN-based ASR analysis..."
python scripts/gnn_prediction_analysis.py \
    --model_path "../../$MODEL_PATH" \
    --gnn_path "../../$OUTPUT_DIR/gnn_models/gnn_model.pth" \
    --device cuda \
    --removal_ratio 0.15 \
    --poison_ratio 0.1 \
    --save_results "../../$OUTPUT_DIR/gnn_asr_analysis.json"

echo "Running ground truth ASR analysis..."
python scripts/ground_truth_analysis.py \
    --model_path "../../$MODEL_PATH" \
    --tss_data "../../data/Efficient Net TSS backdoored.csv" \
    --removal_ratio 0.15 \
    --poison_ratio 0.1 \
    --save_results "../../$OUTPUT_DIR/ground_truth_asr_analysis.json"

cd ../..

# Generate Summary Report
echo ""
echo "Generating Summary Report"
echo "-------------------------"

cat > "$OUTPUT_DIR/pipeline_summary.md" << EOF
# Pipeline Analysis Summary

**Date**: $(date)
**Model**: $MODEL_PATH
**Clean Model**: $CLEAN_MODEL_PATH
**Architecture**: $ARCHITECTURE

## Stage 1: TSS Extraction
- Backdoored model (clean inputs): \`$OUTPUT_DIR/tss_backdoored/\`
- Backdoored model (triggered inputs): \`$OUTPUT_DIR/tss_backdoored_triggered/\`
- Clean model: \`$OUTPUT_DIR/tss_clean/\`

## Stage 2: GNN Training
- Trained models: \`$OUTPUT_DIR/gnn_models/\`
- Cross-evaluation: \`$OUTPUT_DIR/gnn_cross_eval/\`

## Stage 3: ASR Analysis
- GNN-based analysis: \`$OUTPUT_DIR/gnn_asr_analysis.json\`
- Ground truth analysis: \`$OUTPUT_DIR/ground_truth_asr_analysis.json\`

## Key Files
- \`gnn_asr_analysis.json\`: GNN-based ASR metrics
- \`ground_truth_asr_analysis.json\`: Ground truth ASR metrics
- \`gnn_models/\`: Trained GNN models and encoders
- \`tss_*/\`: TSS extraction results and topological analysis

## Next Steps
1. Review ASR analysis results
2. Compare GNN predictions vs ground truth
3. Analyze model pruning effectiveness
4. Generate visualization plots
EOF

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key Results:"
echo "- TSS extraction: $OUTPUT_DIR/tss_*/"
echo "- GNN models: $OUTPUT_DIR/gnn_models/"
echo "- ASR analysis: $OUTPUT_DIR/*_asr_analysis.json"
echo "- Summary: $OUTPUT_DIR/pipeline_summary.md"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/pipeline_summary.md"
echo "  python -m json.tool $OUTPUT_DIR/gnn_asr_analysis.json"
echo "  python -m json.tool $OUTPUT_DIR/ground_truth_asr_analysis.json"
echo "=========================================="
