#!/bin/bash

# Shell script for running multiple architecture evaluations on Unix-like systems
# This script will run evaluations for different architectures and predictors

echo "============================================================"
echo "Architecture Evaluation Batch Runner for Unix/Linux/macOS"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python and make sure it's accessible"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python command: $PYTHON_CMD"

# Check if required files exist
if [ ! -f "data_evaluator.py" ]; then
    echo "Error: data_evaluator.py not found!"
    echo "Make sure data_evaluator.py is in the current directory."
    exit 1
fi

if [ ! -f "run_evaluation.py" ]; then
    echo "Error: run_evaluation.py not found!"
    echo "Make sure run_evaluation.py is in the current directory."
    exit 1
fi

# Create output directory
mkdir -p batch_evaluation_results

echo ""
echo "Available modes:"
echo "1. Interactive mode - select train/test CSV files for each evaluation"
echo "2. Auto mode - uses predefined CSV file patterns"
echo "3. Custom mode - specify train and test CSV files once, run all predictors"
echo ""

read -p "Select mode (1-3): " mode

case $mode in
    1)
        echo ""
        echo "========== INTERACTIVE MODE =========="
        echo "You will be prompted to select CSV files for each evaluation."
        echo ""
        
        # List of architectures to evaluate
        architectures=("efficientnet" "mobilenet" "mobilenetv2")
        
        # List of predictors to evaluate
        predictors=("baseline" "random_forest" "xgboost")
        
        for arch in "${architectures[@]}"; do
            echo ""
            echo "=========================================="
            echo "Evaluating architecture: $arch"
            echo "=========================================="
            
            for pred in "${predictors[@]}"; do
                echo ""
                echo "Running evaluation with $pred predictor..."
                echo "------------------------------------------"
                
                $PYTHON_CMD run_evaluation.py --architecture "$arch" --predictor "$pred" --output_dir "batch_evaluation_results/${arch}_${pred}" --interactive
                
                if [ $? -ne 0 ]; then
                    echo "Error occurred during $arch with $pred predictor"
                    echo "Continuing with next evaluation..."
                else
                    echo "$arch with $pred completed successfully"
                fi
                
                echo ""
                read -p "Press Enter to continue to next evaluation..."
            done
        done
        ;;
        
    2)
        echo ""
        echo "========== AUTO MODE =========="
        echo "Looking for CSV files with standard naming patterns..."
        echo "Expected patterns: *train*.csv, *test*.csv"
        
        # Find train and test files automatically
        train_files=($(find . -maxdepth 2 -name "*train*.csv" -type f))
        test_files=($(find . -maxdepth 2 -name "*test*.csv" -type f))
        
        if [ ${#train_files[@]} -eq 0 ]; then
            echo "No training CSV files found with pattern *train*.csv"
            echo "Please ensure your CSV files follow naming conventions or use custom mode"
            exit 1
        fi
        
        if [ ${#test_files[@]} -eq 0 ]; then
            echo "No test CSV files found with pattern *test*.csv"
            echo "Please ensure your CSV files follow naming conventions or use custom mode"
            exit 1
        fi
        
        # Use first matching files
        train_file="${train_files[0]}"
        test_file="${test_files[0]}"
        
        echo "Found training file: $train_file"
        echo "Found test file: $test_file"
        echo ""
        
        architectures=("efficientnet" "mobilenet" "mobilenetv2")
        predictors=("baseline" "random_forest" "xgboost")
        
        for arch in "${architectures[@]}"; do
            echo ""
            echo "=========================================="
            echo "Evaluating architecture: $arch"
            echo "=========================================="
            
            for pred in "${predictors[@]}"; do
                echo ""
                echo "Running $arch with $pred predictor..."
                echo "------------------------------------------"
                
                $PYTHON_CMD data_evaluator.py --train_csv "$train_file" --test_csv "$test_file" --architecture "$arch" --predictor "$pred" --output_dir "batch_evaluation_results/${arch}_${pred}"
                
                if [ $? -ne 0 ]; then
                    echo "Error occurred during $arch with $pred predictor"
                else
                    echo "$arch with $pred completed successfully"
                fi
            done
        done
        ;;
        
    3)
        echo ""
        echo "========== CUSTOM MODE =========="
        echo "Please specify the train and test CSV files to use for all evaluations."
        
        read -p "Enter path to training CSV file: " train_csv
        read -p "Enter path to test CSV file: " test_csv
        
        if [ ! -f "$train_csv" ]; then
            echo "Error: Training file not found: $train_csv"
            exit 1
        fi
        
        if [ ! -f "$test_csv" ]; then
            echo "Error: Test file not found: $test_csv"
            exit 1
        fi
        
        echo ""
        echo "Using training file: $train_csv"
        echo "Using test file: $test_csv"
        echo ""
        
        architectures=("efficientnet" "mobilenet" "mobilenetv2")
        predictors=("baseline" "random_forest" "xgboost")
        
        for arch in "${architectures[@]}"; do
            echo ""
            echo "=========================================="
            echo "Evaluating architecture: $arch"
            echo "=========================================="
            
            for pred in "${predictors[@]}"; do
                echo ""
                echo "Running $arch with $pred predictor..."
                echo "------------------------------------------"
                
                $PYTHON_CMD data_evaluator.py --train_csv "$train_csv" --test_csv "$test_csv" --architecture "$arch" --predictor "$pred" --output_dir "batch_evaluation_results/${arch}_${pred}"
                
                if [ $? -ne 0 ]; then
                    echo "Error occurred during $arch with $pred predictor"
                else
                    echo "$arch with $pred completed successfully"
                fi
            done
        done
        ;;
        
    *)
        echo "Invalid selection. Using interactive mode."
        # Recursively call the script with mode 1
        exec "$0"
        ;;
esac

echo ""
echo "============================================================"
echo "BATCH EVALUATION COMPLETED"
echo "============================================================"
echo "Results saved in: batch_evaluation_results/"
echo ""

# Show summary of results
if [ -d "batch_evaluation_results" ]; then
    echo "Generated result directories:"
    ls -1 batch_evaluation_results/
    echo ""
    echo "Each directory contains:"
    echo "- *_results.csv: Main results with predictions"
    echo "- *_metrics.csv: Evaluation metrics (RMSE, MAE, correlation)"
    echo "- *.pkl files: Saved preprocessors"
fi

echo ""
echo "Script completed successfully!"

