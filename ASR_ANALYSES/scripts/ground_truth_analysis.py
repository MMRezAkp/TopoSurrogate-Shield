#!/usr/bin/env python3
"""
Ground Truth TSS Analysis Script
"""

import argparse
import sys
import os
import json
import datetime

# Add the src directory and parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.ground_truth_evaluator import ASRGroundTruthEvaluator
from data.loaders import DataLoaderFactory
from config.settings import MODEL_PATHS, DATA_PATHS, ANALYSIS_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Ground Truth TSS Analysis')
    parser.add_argument('--model_path', type=str, default=MODEL_PATHS['efficientnet_model'],
                       help='Path to EfficientNet model')
    parser.add_argument('--tss_data', type=str, default=DATA_PATHS['tss_backdoored'],
                       help='Path to TSS data CSV file')
    parser.add_argument('--device', type=str, default=ANALYSIS_CONFIG['device'],
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--removal_ratio', type=float, default=ANALYSIS_CONFIG['removal_ratio'],
                       help='Ratio of edges to remove')
    parser.add_argument('--poison_ratio', type=float, default=ANALYSIS_CONFIG['poison_ratio'],
                       help='Ratio of poisoned samples for ASR calculation')
    parser.add_argument('--batch_size', type=int, default=ANALYSIS_CONFIG['batch_size'],
                       help='Batch size for data loading')
    parser.add_argument('--data_root', type=str, default=DATA_PATHS['cifar10_data'],
                       help='Root directory for CIFAR-10 data')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results and timing data as JSON file')
    
    args = parser.parse_args()
    
    print("Starting Ground Truth TSS Analysis...")
    print(f"Model path: {args.model_path}")
    print(f"TSS data: {args.tss_data}")
    print(f"Device: {args.device}")
    
    # Create data loader
    print("Creating data loader...")
    dataloader = DataLoaderFactory.create_clean_test_loader(
        batch_size=args.batch_size, data_root=args.data_root
    )
    print(f"✓ Created dataloader with {len(dataloader)} batches")
    
    # Initialize evaluator
    evaluator = ASRGroundTruthEvaluator(
        model_path=args.model_path,
        tss_data_path=args.tss_data,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        dataloader=dataloader,
        removal_ratio=args.removal_ratio,
        poison_ratio=args.poison_ratio
    )
    
    # Save results if requested
    if args.save_results:
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': {
                'model_path': args.model_path,
                'tss_data': args.tss_data,
                'device': args.device,
                'removal_ratio': args.removal_ratio,
                'poison_ratio': args.poison_ratio,
                'batch_size': args.batch_size,
                'data_root': args.data_root
            },
            'results': {
                'initial_acc': results.get('initial_acc', 0.0),
                'final_acc': results.get('final_acc', 0.0),
                'acc_drop': results.get('acc_drop', 0.0),
                'initial_asr': results.get('initial_asr', 0.0),
                'final_asr': results.get('final_asr', 0.0),
                'asr_reduction': results.get('asr_reduction', 0.0),
                'reduction_percentage': results.get('reduction_percentage', 0.0),
                'edges_removed': results.get('edges_removed', 0)
            },
            'timing': results.get('timing', {})
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Results and timing data saved to: {args.save_results}")
    
    print("\nGround Truth TSS Analysis completed successfully!")
    return results


if __name__ == "__main__":
    results = main()


