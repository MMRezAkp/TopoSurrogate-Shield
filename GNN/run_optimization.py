"""
Simple script to run hyperparameter optimization
"""
from hyperparameter_optimization import run_optimization

def main():
    """Run hyperparameter optimization with different configurations."""
    
    print("GNN Hyperparameter Optimization")
    print("=" * 50)
    
    # Get data paths from user
    print("Please provide the paths to your data files:")
    backdoored_path = input("Backdoored data CSV path: ").strip()
    clean_path = input("Clean data CSV path: ").strip()
    
    if not backdoored_path or not clean_path:
        print("Error: Both data paths are required!")
        return
    
    # Check if files exist
    import os
    if not os.path.exists(backdoored_path):
        print(f"Error: Backdoored data file not found: {backdoored_path}")
        return
    if not os.path.exists(clean_path):
        print(f"Error: Clean data file not found: {clean_path}")
        return
    
    # Configuration options
    configs = [
        {
            'name': 'EfficientNet Basic',
            'architecture': 'efficientnet',
            'n_trials': 50,
            'timeout': 1800,  # 30 minutes
            'enable_edge_removal': False
        },
        {
            'name': 'EfficientNet with Edge Removal',
            'architecture': 'efficientnet',
            'n_trials': 50,
            'timeout': 1800,
            'enable_edge_removal': True,
            'tss_threshold': 0.001
        },
        {
            'name': 'MobileNet v2 Basic',
            'architecture': 'mobilenet',
            'n_trials': 50,
            'timeout': 1800,
            'enable_edge_removal': False
        },
        {
            'name': 'MobileNet v2 with Edge Removal',
            'architecture': 'mobilenet',
            'n_trials': 50,
            'timeout': 1800,
            'enable_edge_removal': True,
            'tss_threshold': 0.001
        }
    ]
    
    print("Available optimization configurations:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
    
    # Let user choose
    try:
        choice = int(input("\nSelect configuration (1-4): ")) - 1
        if choice < 0 or choice >= len(configs):
            print("Invalid choice. Using default configuration.")
            choice = 0
    except ValueError:
        print("Invalid input. Using default configuration.")
        choice = 0
    
    selected_config = configs[choice]
    print(f"\nSelected: {selected_config['name']}")
    
    # Run optimization
    study = run_optimization(
        architecture=selected_config['architecture'],
        n_trials=selected_config['n_trials'],
        timeout=selected_config['timeout'],
        backdoored_path=backdoored_path,
        clean_path=clean_path,
        enable_edge_removal=selected_config.get('enable_edge_removal', False),
        tss_threshold=selected_config.get('tss_threshold', 0.0001),
        output_dir=f'./optimization_results_{selected_config["architecture"]}'
    )
    
    print(f"\nOptimization completed!")
    print(f"Best RMSE: {study.best_value:.6f}")
    
    # Get additional metrics from best trial
    best_trial = study.best_trial
    best_spearman = best_trial.user_attrs.get('spearman_rho', 'N/A')
    print(f"Best Spearman's ρ: {best_spearman:.6f}")
    print(f"Best parameters saved to: ./optimization_results_{selected_config['architecture']}")

if __name__ == "__main__":
    main()
