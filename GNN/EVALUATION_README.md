# Architecture Evaluation Tools

This directory contains tools to evaluate different machine learning architectures on TSS (True Sensitivity Score) prediction using the same data processing pipeline as your GNN system, without modifying any of the existing GNN code.

## Files Created

### Core Scripts
- **`data_evaluator.py`** - Main evaluation script that can choose train/test CSV files
- **`cross_architecture_evaluator.py`** - Cross-architecture evaluation using XGBoost/ML models
- **`gnn_cross_architecture_evaluator.py`** - Cross-architecture evaluation using your actual trained GNN
- **`show_gnn_training_history.py`** - Displays training history and epoch metrics from GNN training
- **`run_evaluation.py`** - Simplified runner script for interactive use
- **`run_all_evaluations.bat`** - Batch script for Windows to run all evaluations
- **`run_all_evaluations.sh`** - Shell script for Unix/Linux/macOS to run all evaluations

## Quick Start

### Option 1: Interactive Single Evaluation
```bash
# On Windows
python run_evaluation.py

# On Unix/Linux/macOS
python3 run_evaluation.py
```
This will interactively prompt you to select train and test CSV files from available files.

### Option 2: Direct Single Evaluation
```bash
# Specify files directly
python data_evaluator.py --train_csv path/to/train.csv --test_csv path/to/test.csv --architecture efficientnet --predictor xgboost

# With edge removal enabled
python data_evaluator.py --train_csv train.csv --test_csv test.csv --enable_edge_removal --tss_threshold 0.001
```

### Option 3: Cross-Architecture Evaluation

#### 3a. Using ML Models (XGBoost/Random Forest)
```bash
python cross_architecture_evaluator.py --train_csv efficientnet_data.csv --test_csv mobilenet_data.csv --train_arch efficientnet --test_arch mobilenet
```

#### 3b. Using Your Actual Trained GNN
```bash
python gnn_cross_architecture_evaluator.py --train_csv efficientnet_data.csv --test_csv mobilenet_data.csv --gnn_model_path gnn_shortcut_aware_model.pth --train_arch efficientnet --test_arch mobilenet
```

#### 3c. View Training History
```bash
python show_gnn_training_history.py --output_dir ./results
```

### Option 4: Batch Evaluation (All Architectures & Predictors)
```bash
# On Windows
run_all_evaluations.bat

# On Unix/Linux/macOS
./run_all_evaluations.sh
```

## Supported Architectures
- `efficientnet` (default)
- `mobilenet`
- `mobilenetv2`

## Supported Predictors
- `baseline` - Simple linear regression or mean prediction
- `random_forest` - Random Forest regressor
- `xgboost` - XGBoost regressor (recommended)

## Output Files

Each evaluation creates the following files in the output directory:
- `*_results.csv` - Main results with true values, predictions, and residuals
- `*_metrics.csv` - Evaluation metrics (RMSE, MAE, Spearman's ρ, Kendall's τ)
- `*_scaler.pkl` - Saved StandardScaler for reproducibility
- `*_encoder.pkl` - Saved OneHotEncoder for reproducibility

## Detailed Usage Examples

### 1. Basic Evaluation
```bash
python data_evaluator.py --train_csv train_data.csv --test_csv test_data.csv
```

### 2. Different Architecture
```bash
python data_evaluator.py --train_csv train.csv --test_csv test.csv --architecture mobilenet
```

### 3. Different Predictor
```bash
python data_evaluator.py --train_csv train.csv --test_csv test.csv --predictor random_forest
```

### 4. Custom Output Directory
```bash
python data_evaluator.py --train_csv train.csv --test_csv test.csv --output_dir my_results
```

### 5. With Edge Removal
```bash
python data_evaluator.py --train_csv train.csv --test_csv test.csv --enable_edge_removal --tss_threshold 0.0005
```

## Cross-Architecture Evaluation

When training and test data come from different architectures (e.g., EfficientNet vs MobileNet), you have two options:

### Option A: ML-based Cross-Architecture Evaluation (Recommended for different architectures)
```bash
python cross_architecture_evaluator.py --train_csv efficientnet_train.csv --test_csv mobilenet_test.csv --train_arch efficientnet --test_arch mobilenet
```

**What it does:**
- ✅ Uses **XGBoost** (not GNN) trained on your CSV data
- ✅ **No epoch metrics** (XGBoost trains in one step)
- ✅ Handles different layer structures robustly
- ✅ Uses only numerical features (avoids layer name conflicts)
- ✅ Fast and reliable for cross-architecture scenarios

### Option B: GNN-based Cross-Architecture Evaluation (For architecture compatibility)
```bash
python gnn_cross_architecture_evaluator.py --train_csv efficientnet_train.csv --test_csv mobilenet_test.csv --gnn_model_path gnn_shortcut_aware_model.pth --train_arch efficientnet --test_arch mobilenet
```

**What it does:**
- ✅ Uses your **actual trained GNN model**
- ✅ Shows **training history** if available
- ⚠️ May fail if architectures are too different (layer structure mismatch)
- ✅ Same preprocessing pipeline as your GNN training
- ✅ Exact same model architecture and weights

### When to Use Which?
- **Use ML-based** (`cross_architecture_evaluator.py`) when:
  - Architectures are very different (EfficientNet ↔ MobileNet)
  - You want robust, guaranteed results
  - You want to compare different ML approaches
  
- **Use GNN-based** (`gnn_cross_architecture_evaluator.py`) when:
  - You want to use your exact trained GNN
  - Architectures are similar enough
  - You want to see how your specific GNN generalizes

### Cross-Architecture Examples
```bash
# EfficientNet to MobileNet
python cross_architecture_evaluator.py --train_csv efficientnet_backdoored.csv --test_csv mobilenet_clean.csv --train_arch efficientnet --test_arch mobilenet

# MobileNet to EfficientNet  
python cross_architecture_evaluator.py --train_csv mobilenet_data.csv --test_csv efficientnet_data.csv --train_arch mobilenet --test_arch efficientnet

# With edge removal
python cross_architecture_evaluator.py --train_csv train.csv --test_csv test.csv --train_arch efficientnet --test_arch mobilenet --enable_edge_removal --tss_threshold 0.001
```

## Interactive Runner Options

The `run_evaluation.py` script provides several convenience options:

```bash
# Force interactive mode even with specified files
python run_evaluation.py --train_csv train.csv --test_csv test.csv --interactive

# Specify architecture and predictor but select files interactively
python run_evaluation.py --architecture mobilenet --predictor xgboost
```

## Batch Evaluation Modes

Both batch scripts (`run_all_evaluations.bat` and `run_all_evaluations.sh`) offer three modes:

1. **Interactive Mode**: You select train/test files for each evaluation
2. **Auto Mode**: Automatically finds files matching `*train*.csv` and `*test*.csv` patterns
3. **Custom Mode**: You specify train/test files once, then all architectures and predictors are tested

## Dependencies

The evaluation tools require:
- All existing GNN dependencies (pandas, numpy, scikit-learn, etc.)
- Optional: `xgboost` for XGBoost predictor (will fall back to Random Forest if not available)

To install XGBoost:
```bash
pip install xgboost
```

## Data Format Requirements

Your CSV files should contain the same columns as used in the GNN system:
- Feature columns: `w_out_norm_a`, `w_out_norm_b`, `w_norm_ratio`, `depth_a`, `depth_b`, etc.
- Layer information: `src_layer`, `dst_layer`, `src_ch`, `dst_ch`, `is_same_layer`
- Target: `tss`

## Key Features

1. **Non-intrusive**: Doesn't modify any existing GNN code
2. **Same preprocessing**: Uses identical data processing pipeline as GNN
3. **Same metrics**: Computes RMSE, MAE, and correlation metrics
4. **Flexible**: Works with any CSV files that have the required columns
5. **Batch processing**: Can evaluate multiple architectures and predictors automatically
6. **Cross-platform**: Works on Windows, Linux, and macOS

## Error Handling

- Missing columns will trigger warnings but won't crash the evaluation
- Invalid file paths will be caught and reported
- Failed predictors will fall back to simpler alternatives
- Detailed error messages help with debugging

## Integration with Existing System

These tools are designed to:
- Use the same `DataProcessor` class without modification
- Use the same `compute_metrics` function from `training_utils`
- Use the same layer encoders from `layer_encoders`
- Maintain compatibility with existing config and preprocessing settings

You can continue using your GNN system exactly as before - these tools are completely separate and only add new functionality for evaluating other architectures.
