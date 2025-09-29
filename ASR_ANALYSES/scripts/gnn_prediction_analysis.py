#!/usr/bin/env python3
"""
GNN-based TSS Prediction Analysis Script
"""

import argparse
import sys
import os
import json
import datetime

# Add the src directory and parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.asr_analyzer import ASRAnalyzerWithRealFeatures
from config.settings import MODEL_PATHS, ANALYSIS_CONFIG


def main():
    parser = argparse.ArgumentParser(description='GNN-based TSS Prediction Analysis')
    parser.add_argument('--model_path', type=str, default=MODEL_PATHS['efficientnet_model'],
                       help='Path to EfficientNet model')
    parser.add_argument('--gnn_path', type=str, default=MODEL_PATHS['gnn_model'],
                       help='Path to GNN model')
    parser.add_argument('--scaler_path', type=str, default=MODEL_PATHS['scaler'],
                       help='Path to scaler')
    parser.add_argument('--encoder_path', type=str, default=MODEL_PATHS['encoder'],
                       help='Path to encoder')
    parser.add_argument('--device', type=str, default=ANALYSIS_CONFIG['device'],
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--removal_ratio', type=float, default=ANALYSIS_CONFIG['removal_ratio'],
                       help='Ratio of edges to remove')
    parser.add_argument('--max_layers', type=int, default=ANALYSIS_CONFIG['max_layers'],
                       help='Maximum number of layers to analyze')
    parser.add_argument('--max_edges', type=int, default=ANALYSIS_CONFIG['max_edges'],
                       help='Maximum number of edges to create')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results and timing data as JSON file')
    
    args = parser.parse_args()
    
    print("Starting GNN-based TSS Prediction Analysis...")
    print(f"Model path: {args.model_path}")
    print(f"GNN path: {args.gnn_path}")
    print(f"Device: {args.device}")
    
    # Initialize analyzer
    analyzer = ASRAnalyzerWithRealFeatures(
        model_path=args.model_path,
        gnn_path=args.gnn_path,
        scaler_path=args.scaler_path,
        encoder_path=args.encoder_path,
        device=args.device
    )
    
    # Run analysis
    results = analyzer.run_asr_analysis(
        removal_ratio=args.removal_ratio,
        max_layers=args.max_layers,
        max_edges=args.max_edges
    )
    
    # Save results if requested
    if args.save_results:
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameters': {
                'model_path': args.model_path,
                'gnn_path': args.gnn_path,
                'scaler_path': args.scaler_path,
                'encoder_path': args.encoder_path,
                'device': args.device,
                'removal_ratio': args.removal_ratio,
                'max_layers': args.max_layers,
                'max_edges': args.max_edges
            },
            'results': {
                'initial_acc': results.get('initial_acc', 0.0),
                'final_acc': results.get('final_acc', 0.0),
                'acc_drop': results.get('acc_drop', 0.0),
                'initial_asr': results.get('initial_asr', 0.0),
                'final_asr': results.get('final_asr', 0.0),
                'asr_reduction': results.get('asr_reduction', 0.0),
                'edges_removed': results.get('edges_removed', 0),
                'removal_ratio': results.get('removal_ratio', 0.0)
            },
            'timing': results.get('timing', {}),
            'tss_statistics': {}
        }
        
        # Add TSS statistics if available
        if 'tss_scores' in results and results['tss_scores']:
            json_results['tss_statistics'] = {
                'mean_tss': float(sum(results['tss_scores']) / len(results['tss_scores'])),
                'min_tss': float(min(results['tss_scores'])),
                'max_tss': float(max(results['tss_scores'])),
                'num_edges': len(results['tss_scores'])
            }
        
        with open(args.save_results, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Results and timing data saved to: {args.save_results}")
    
    print("\nGNN-based TSS Prediction Analysis completed successfully!")
    return results


if __name__ == "__main__":
    results = main()


