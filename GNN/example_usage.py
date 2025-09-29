"""
Example usage of the GNN TSS prediction training module
"""
import subprocess
import sys
import os

def run_training_example():
    """Run example training with different configurations."""
    
    print("GNN TSS Prediction Training Examples")
    print("=" * 50)
    
    # Example 1: Basic EfficientNet training
    print("\n1. Basic EfficientNet Training:")
    print("Command: python main.py --architecture efficientnet --epochs 20")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--architecture", "efficientnet",
            "--epochs", "20",
            "--output_dir", "./example_results_efficientnet"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ EfficientNet training completed successfully")
        else:
            print(f"✗ EfficientNet training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ EfficientNet training timed out")
    except Exception as e:
        print(f"✗ EfficientNet training error: {e}")
    
    # Example 2: MobileNet v2 training with custom parameters
    print("\n2. MobileNet v2 Training with Custom Parameters:")
    print("Command: python main.py --architecture mobilenet --epochs 30 --hidden_dim 128 --batch_size 64")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py",
            "--architecture", "mobilenet",
            "--epochs", "30",
            "--hidden_dim", "128",
            "--batch_size", "64",
            "--output_dir", "./example_results_mobilenet"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ MobileNet v2 training completed successfully")
        else:
            print(f"✗ MobileNet v2 training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ MobileNet v2 training timed out")
    except Exception as e:
        print(f"✗ MobileNet v2 training error: {e}")
    
    # Example 3: Residual GNN training
    print("\n3. Residual GNN Training:")
    print("Command: python main.py --model_type residual_gnn --epochs 25")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py",
            "--model_type", "residual_gnn",
            "--epochs", "25",
            "--output_dir", "./example_results_residual"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Residual GNN training completed successfully")
        else:
            print(f"✗ Residual GNN training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Residual GNN training timed out")
    except Exception as e:
        print(f"✗ Residual GNN training error: {e}")
    
    # Example 4: Custom loss weights training
    print("\n4. Custom Loss Weights Training:")
    print("Command: python main.py --architecture mobilenet --lambda_mono 0.3 --lambda_shortcut 0.01 --lambda_consistency 0.1")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py",
            "--architecture", "mobilenet",
            "--lambda_mono", "0.3",
            "--lambda_shortcut", "0.01", 
            "--lambda_consistency", "0.1",
            "--lambda_ranking", "0.05",
            "--epochs", "25",
            "--output_dir", "./example_results_custom_weights"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Custom loss weights training completed successfully")
        else:
            print(f"✗ Custom loss weights training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Custom loss weights training timed out")
    except Exception as e:
        print(f"✗ Custom loss weights training error: {e}")
    
    # Example 5: High physical constraint training
    print("\n5. High Physical Constraints Training:")
    print("Command: python main.py --lambda_mono 0.5 --lambda_shortcut 0.1 --lambda_consistency 0.2 --lambda_ranking 0.1")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py",
            "--lambda_mono", "0.5",
            "--lambda_shortcut", "0.1",
            "--lambda_consistency", "0.2", 
            "--lambda_ranking", "0.1",
            "--epochs", "30",
            "--output_dir", "./example_results_high_constraints"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ High physical constraints training completed successfully")
        else:
            print(f"✗ High physical constraints training failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ High physical constraints training timed out")
    except Exception as e:
        print(f"✗ High physical constraints training error: {e}")
    
    # Example 6: Custom data paths training
    print("\n6. Custom Data Paths Training:")
    print("Command: python main.py --backdoored_path /path/to/backdoored.csv --clean_path /path/to/clean.csv")
    print("Note: This example shows the command format but won't run without actual data files")
    
    print("\n" + "=" * 50)
    print("Example training completed!")
    print("Check the output directories for results and visualizations.")

def check_data_files():
    """Check if required data files exist."""
    print("Checking for required data files...")
    
    from config import DATA_PATHS
    
    missing_files = []
    for name, path in DATA_PATHS.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
        else:
            print(f"✓ Found {name}: {path}")
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"✗ {file}")
        print("\nPlease ensure all data files are present before running training.")
        return False
    else:
        print("\n✓ All required data files found!")
        return True

def show_help():
    """Show help information."""
    print("GNN TSS Prediction Training Module")
    print("=" * 40)
    print("\nAvailable commands:")
    print("python main.py --help                    # Show help")
    print("python main.py                           # Basic training")
    print("python main.py --architecture mobilenet  # MobileNet v2 training")
    print("python example_usage.py                  # Run examples")
    print("\nData Path Examples:")
    print("python main.py --backdoored_path /path/to/backdoored.csv --clean_path /path/to/clean.csv")
    print("python main.py --architecture mobilenet --backdoored_path /data/mobilenet_backdoored.csv")
    print("\nPhysical Loss Weight Examples:")
    print("python main.py --lambda_mono 0.3 --lambda_shortcut 0.01")
    print("python main.py --lambda_consistency 0.1 --lambda_ranking 0.05")
    print("python main.py --lambda_mono 0.5 --lambda_shortcut 0.1 --lambda_consistency 0.2")
    print("\nFor more options, run: python main.py --help")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        # Check data files first
        if check_data_files():
            # Run example training
            run_training_example()
        else:
            print("\nPlease add your data files and try again.")
