"""
Simple Runner for Architecture Evaluation
=========================================

This script provides a simplified interface to run evaluations on different
architectures with various train/test CSV file combinations.

Usage Examples:
    # Basic usage
    python run_evaluation.py

    # Specify custom files
    python run_evaluation.py --train_csv my_train.csv --test_csv my_test.csv

    # Run with different predictor
    python run_evaluation.py --predictor random_forest

    # Run with different architecture
    python run_evaluation.py --architecture mobilenet
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple
import subprocess

def find_csv_files(directory: str = ".") -> List[str]:
    """Find all CSV files in the given directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    csv_files.extend(glob.glob(os.path.join(directory, "**/*.csv"), recursive=True))
    return sorted(csv_files)

def detect_architecture_from_filename(filename: str) -> str:
    """Detect architecture from filename."""
    filename_lower = filename.lower()
    if 'efficient' in filename_lower or 'efficientnet' in filename_lower:
        return 'efficientnet'
    elif 'mobile' in filename_lower or 'mobilenet' in filename_lower:
        return 'mobilenet'
    elif 'mobilenetv2' in filename_lower:
        return 'mobilenetv2'
    else:
        return 'unknown'

def select_files_interactive(csv_files: List[str]) -> Tuple[str, str]:
    """Interactive file selection for train and test CSV files."""
    if not csv_files:
        print("No CSV files found in the current directory!")
        print("Please make sure your CSV files are in the current directory or subdirectories.")
        sys.exit(1)
    
    print("\nAvailable CSV files:")
    print("-" * 50)
    for i, file_path in enumerate(csv_files, 1):
        arch = detect_architecture_from_filename(file_path)
        arch_info = f" [{arch}]" if arch != 'unknown' else ""
        print(f"{i:2d}. {file_path}{arch_info}")
    
    # Select training file
    while True:
        try:
            train_choice = input(f"\nSelect training CSV file (1-{len(csv_files)}): ").strip()
            train_idx = int(train_choice) - 1
            if 0 <= train_idx < len(csv_files):
                train_csv = csv_files[train_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    
    # Select test file
    while True:
        try:
            test_choice = input(f"Select test CSV file (1-{len(csv_files)}): ").strip()
            test_idx = int(test_choice) - 1
            if 0 <= test_idx < len(csv_files):
                test_csv = csv_files[test_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    
    # Detect potential cross-architecture scenario
    train_arch = detect_architecture_from_filename(train_csv)
    test_arch = detect_architecture_from_filename(test_csv)
    
    if train_arch != 'unknown' and test_arch != 'unknown' and train_arch != test_arch:
        print(f"\n⚠️  CROSS-ARCHITECTURE DETECTED ⚠️")
        print(f"Training file appears to be: {train_arch}")
        print(f"Test file appears to be: {test_arch}")
        print()
        print("For cross-architecture evaluation, consider using:")
        print(f"python cross_architecture_evaluator.py --train_csv \"{train_csv}\" --test_csv \"{test_csv}\" --train_arch {train_arch} --test_arch {test_arch}")
        print()
        
        choice = input("Continue with standard evaluation (y) or exit to use cross-architecture evaluator (n)? [y/n]: ").strip().lower()
        if choice in ['n', 'no']:
            print("Please use the cross_architecture_evaluator.py script for better cross-architecture handling.")
            sys.exit(0)
    
    return train_csv, test_csv

def run_evaluation(train_csv: str, test_csv: str, architecture: str = "efficientnet", 
                  predictor: str = "xgboost", output_dir: str = "evaluation_results",
                  enable_edge_removal: bool = False, tss_threshold: float = 0.0001) -> int:
    """Run the evaluation using data_evaluator.py."""
    
    cmd = [
        sys.executable, "data_evaluator.py",
        "--train_csv", train_csv,
        "--test_csv", test_csv,
        "--architecture", architecture,
        "--predictor", predictor,
        "--output_dir", output_dir,
        "--tss_threshold", str(tss_threshold)
    ]
    
    if enable_edge_removal:
        cmd.append("--enable_edge_removal")
    
    print(f"\nRunning evaluation with:")
    print(f"  Training CSV: {train_csv}")
    print(f"  Test CSV: {test_csv}")
    print(f"  Architecture: {architecture}")
    print(f"  Predictor: {predictor}")
    print(f"  Output directory: {output_dir}")
    print(f"  Edge removal: {'Enabled' if enable_edge_removal else 'Disabled'}")
    if enable_edge_removal:
        print(f"  TSS threshold: {tss_threshold}")
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: data_evaluator.py not found!")
        print("Make sure data_evaluator.py is in the current directory.")
        return 1

def main():
    """Main function for the evaluation runner."""
    parser = argparse.ArgumentParser(description='Simple runner for architecture evaluation')
    parser.add_argument('--train_csv', type=str, default=None,
                       help='Path to training CSV file (interactive selection if not provided)')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV file (interactive selection if not provided)')
    parser.add_argument('--architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Neural network architecture (default: efficientnet)')
    parser.add_argument('--predictor', type=str, default='xgboost',
                       choices=['baseline', 'random_forest', 'xgboost'],
                       help='Predictor type to use (default: xgboost)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001,
                       help='TSS threshold for edge removal (default: 0.0001)')
    parser.add_argument('--interactive', action='store_true',
                       help='Force interactive file selection even if files are specified')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Architecture Evaluation Runner")
    print("=" * 60)
    
    # Check if data_evaluator.py exists
    if not os.path.exists("data_evaluator.py"):
        print("Error: data_evaluator.py not found!")
        print("Make sure data_evaluator.py is in the current directory.")
        return 1
    
    # Determine train and test CSV files
    if args.interactive or (args.train_csv is None or args.test_csv is None):
        print("Interactive file selection mode")
        csv_files = find_csv_files()
        train_csv, test_csv = select_files_interactive(csv_files)
    else:
        train_csv = args.train_csv
        test_csv = args.test_csv
        
        # Check if files exist
        if not os.path.exists(train_csv):
            print(f"Error: Training CSV file not found: {train_csv}")
            return 1
        if not os.path.exists(test_csv):
            print(f"Error: Test CSV file not found: {test_csv}")
            return 1
    
    # Run the evaluation
    exit_code = run_evaluation(
        train_csv=train_csv,
        test_csv=test_csv,
        architecture=args.architecture,
        predictor=args.predictor,
        output_dir=args.output_dir,
        enable_edge_removal=args.enable_edge_removal,
        tss_threshold=args.tss_threshold
    )
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        
        # List output files
        if os.path.exists(args.output_dir):
            output_files = os.listdir(args.output_dir)
            if output_files:
                print("\nGenerated files:")
                for file in sorted(output_files):
                    print(f"  - {file}")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION FAILED!")
        print("=" * 60)
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
