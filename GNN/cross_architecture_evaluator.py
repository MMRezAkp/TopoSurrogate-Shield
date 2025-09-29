"""
Cross-Architecture Evaluator
============================

This script is specifically designed for evaluating across different architectures
(e.g., training on EfficientNet data, testing on MobileNet data).

It handles:
- Different layer structures between architectures
- Feature alignment
- Architecture-specific preprocessing
- Robust categorical encoding

Usage:
    python cross_architecture_evaluator.py --train_csv efficientnet_train.csv --test_csv mobilenet_test.csv --train_arch efficientnet --test_arch mobilenet
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau

# Import existing modules
from training_utils import compute_metrics


class CrossArchitectureEvaluator:
    """
    Evaluator specifically designed for cross-architecture evaluation.
    Handles different layer structures and feature mismatches gracefully.
    """
    
    def __init__(self, train_architecture: str = 'efficientnet', test_architecture: str = 'mobilenet'):
        """Initialize the cross-architecture evaluator."""
        self.train_architecture = train_architecture
        self.test_architecture = test_architecture
        self.scaler = None
        
        # Core numerical features that should be consistent across architectures
        self.core_features = [
            # Weight features
            'w_out_norm_a', 'w_out_norm_b', 'w_norm_ratio',
            # Layer position features  
            'depth_a', 'depth_b', 'depth_diff',
            # Connectivity features
            'fan_in_a', 'fan_in_b', 'fan_out_a', 'fan_out_b', 
            'fan_in_ratio', 'fan_out_ratio',
            # Gradient features
            'grad_out_norm_a', 'grad_out_norm_b', 'grad_norm_ratio',
            # Activation features
            'act_mean_a', 'act_mean_b', 'act_var_a', 'act_var_b',
            'act_std_a', 'act_std_b', 'act_mean_diff', 'act_std_ratio',
            # Channel features
            'src_ch', 'dst_ch', 'is_same_layer'
        ]
        
        self.target_feature = 'tss'
        
    def load_data(self, train_csv_path: str, test_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test CSV files."""
        try:
            print(f"Loading training data ({self.train_architecture}) from: {train_csv_path}")
            train_df = pd.read_csv(train_csv_path, on_bad_lines='skip')
            print(f"Loaded {len(train_df)} rows from training dataset.")
            
            print(f"Loading test data ({self.test_architecture}) from: {test_csv_path}")
            test_df = pd.read_csv(test_csv_path, on_bad_lines='skip')
            print(f"Loaded {len(test_df)} rows from test dataset.")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
    
    def preprocess_for_cross_architecture(self, df: pd.DataFrame, is_training: bool = True,
                                        enable_edge_removal: bool = False, tss_threshold: float = 0.0001) -> pd.DataFrame:
        """Preprocess data for cross-architecture evaluation using only core numerical features."""
        print(f"Preprocessing {'training' if is_training else 'test'} data for cross-architecture evaluation...")
        
        # Remove unnecessary columns
        df = df.drop('is_backdoored', axis=1, errors='ignore')
        
        # Use only core features + target
        keep_cols = self.core_features + [self.target_feature]
        
        # Filter columns that exist in the dataframe
        existing_cols = [col for col in keep_cols if col in df.columns]
        missing_cols = [col for col in keep_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in data: {missing_cols}")
        
        df = df[existing_cols]
        
        # Filter out rows with very low TSS values (only if edge removal is enabled)
        if enable_edge_removal and tss_threshold > 0:
            initial_len = len(df)
            df = df[df[self.target_feature] > tss_threshold]
            print(f"Edge removal enabled: Retained {len(df)} rows after removing tss <= {tss_threshold} "
                  f"(removed {initial_len - len(df)} rows).")
        else:
            print(f"Edge removal disabled: Using all {len(df)} rows.")
        
        # Convert channel indices to proper format
        if 'src_ch' in df.columns and 'dst_ch' in df.columns:
            df['src_ch'] = df['src_ch'].astype(float).astype(int)
            df['dst_ch'] = df['dst_ch'].astype(float).astype(int)
        
        df = df.dropna().reset_index(drop=True)
        
        # Identify numerical features for scaling
        numerical_features = [col for col in df.columns if col != self.target_feature and col not in ['src_ch', 'dst_ch', 'is_same_layer']]
        
        # Scale numerical features
        if is_training:
            print("Fitting and transforming numerical features...")
            self.scaler = StandardScaler()
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        elif self.scaler is not None:
            print("Applying existing numerical scaling...")
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        else:
            print("Warning: No scaler available, skipping numerical scaling")
        
        print(f"Preprocessing completed. Dataset shape: {df.shape}")
        print(f"Features used: {[col for col in df.columns if col != self.target_feature]}")
        
        return df
    
    def extract_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target vector from processed dataframe."""
        # Get all feature columns (everything except target)
        feature_cols = [col for col in df.columns if col != self.target_feature]
        
        X = df[feature_cols].values
        y = df[self.target_feature].values
        
        print(f"Extracted features shape: {X.shape}")
        print(f"Extracted targets shape: {y.shape}")
        print(f"Feature columns: {feature_cols}")
        
        return X, y
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "Model") -> Dict[str, float]:
        """Evaluate predictions using the same metrics as GNN system."""
        # Use the existing compute_metrics function from training_utils
        metrics = compute_metrics(y_pred.tolist(), y_true.tolist())
        
        print(f"\n{model_name} Cross-Architecture Results:")
        print(f"Training Architecture: {self.train_architecture}")
        print(f"Test Architecture: {self.test_architecture}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"Spearman's ρ: {metrics['spearman']:.6f}")
        print(f"Kendall's τ: {metrics['kendall']:.6f}")
        
        return metrics
    
    def save_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    metrics: Dict[str, float], output_path: str, model_name: str = "CrossArch"):
        """Save results to CSV file."""
        results_df = pd.DataFrame({
            'true_tss': y_true,
            f'{model_name.lower()}_pred': y_pred,
            'residuals': y_pred - y_true,
            'abs_residuals': np.abs(y_pred - y_true)
        })
        
        # Add metadata
        results_df.attrs['train_architecture'] = self.train_architecture
        results_df.attrs['test_architecture'] = self.test_architecture
        
        # Add metrics as metadata in the first few rows
        metrics_df = pd.DataFrame({
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        })
        
        # Save results
        results_df.to_csv(output_path, index=False)
        metrics_path = output_path.replace('.csv', '_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Results saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")
    
    def save_preprocessors(self, output_dir: str = '.'):
        """Save preprocessors for later use."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, f'cross_arch_{self.train_architecture}_to_{self.test_architecture}_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")


def xgboost_predictor(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray) -> np.ndarray:
    """XGBoost predictor for cross-architecture evaluation."""
    try:
        import xgboost as xgb
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using XGBoost predictor")
    except ImportError:
        print("XGBoost not available, falling back to Random Forest")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"XGBoost failed: {e}")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    return y_pred


def main():
    """Main function for cross-architecture evaluation."""
    parser = argparse.ArgumentParser(description='Cross-architecture TSS evaluation')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--train_arch', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Training data architecture')
    parser.add_argument('--test_arch', type=str, default='mobilenet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Test data architecture')
    parser.add_argument('--predictor', type=str, default='xgboost',
                       choices=['xgboost'],
                       help='Predictor type to use')
    parser.add_argument('--output_dir', type=str, default='cross_arch_results',
                       help='Output directory for results')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001,
                       help='TSS threshold for edge removal')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Cross-Architecture Evaluation for TSS Prediction")
    print("=" * 70)
    print(f"Training Architecture: {args.train_arch}")
    print(f"Test Architecture: {args.test_arch}")
    print(f"Predictor: {args.predictor}")
    print(f"Training data: {args.train_csv}")
    print(f"Test data: {args.test_csv}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    try:
        # Initialize evaluator
        evaluator = CrossArchitectureEvaluator(args.train_arch, args.test_arch)
        
        # Load data
        train_df, test_df = evaluator.load_data(args.train_csv, args.test_csv)
        
        # Preprocess data (using only core numerical features)
        print(f"\n{'='*20} PREPROCESSING {'='*20}")
        train_processed = evaluator.preprocess_for_cross_architecture(
            train_df, 
            is_training=True,
            enable_edge_removal=args.enable_edge_removal,
            tss_threshold=args.tss_threshold
        )
        
        test_processed = evaluator.preprocess_for_cross_architecture(
            test_df,
            is_training=False,
            enable_edge_removal=args.enable_edge_removal,
            tss_threshold=args.tss_threshold
        )
        
        # Extract features and targets
        X_train, y_train = evaluator.extract_features_and_targets(train_processed)
        X_test, y_test = evaluator.extract_features_and_targets(test_processed)
        
        # Make predictions
        print(f"\n{'='*20} TRAINING AND PREDICTION {'='*20}")
        y_pred = xgboost_predictor(X_train, y_train, X_test)
        
        # Evaluate results
        print(f"\n{'='*20} EVALUATION {'='*20}")
        model_name = f"{args.predictor}_{args.train_arch}_to_{args.test_arch}"
        metrics = evaluator.evaluate_predictions(y_test, y_pred, model_name)
        
        # Save results
        output_path = os.path.join(args.output_dir, f'{model_name}_results.csv')
        evaluator.save_results(y_test, y_pred, metrics, output_path, model_name)
        
        # Save preprocessors
        evaluator.save_preprocessors(args.output_dir)
        
        print(f"\n{'='*20} CROSS-ARCHITECTURE EVALUATION COMPLETE {'='*20}")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

