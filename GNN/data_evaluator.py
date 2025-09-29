"""
Data Evaluator for Other Architectures
======================================

This module provides functionality to evaluate other machine learning architectures
using the same data processing and evaluation metrics as the GNN system, but without
changing any GNN-specific code.

Key Features:
- Choose custom train and test CSV files
- Use the same data preprocessing pipeline
- Compute RMSE, MAE, and correlation metrics
- Support for multiple architectures (EfficientNet, MobileNet, etc.)
- Batch processing capabilities

Usage:
    python data_evaluator.py --train_csv path/to/train.csv --test_csv path/to/test.csv --architecture efficientnet
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau

# Import existing modules (without modification)
from data_processing import DataProcessor
from training_utils import compute_metrics
from layer_encoders import get_layer_encoder


class ArchitectureEvaluator:
    """
    Evaluator for different architectures using GNN preprocessing pipeline.
    Keeps all existing GNN functions intact while providing new functionality.
    """
    
    def __init__(self, architecture: str = 'efficientnet'):
        """Initialize the evaluator for a specific architecture."""
        self.architecture = architecture
        self.layer_encoder = get_layer_encoder(architecture)
        self.scaler = None
        self.encoder = None
        
        # Use the same features as the GNN data processor
        self.classification_features = [
            'src_layer', 'dst_layer', 'src_ch', 'dst_ch', 'is_same_layer'
        ]
        
        self.linear_features = [
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
        ]
        
        self.target_feature = 'tss'
        
    def load_data(self, train_csv_path: str, test_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test CSV files."""
        try:
            print(f"Loading training data from: {train_csv_path}")
            train_df = pd.read_csv(train_csv_path, on_bad_lines='skip')
            print(f"Loaded {len(train_df)} rows from training dataset.")
            
            print(f"Loading test data from: {test_csv_path}")
            test_df = pd.read_csv(test_csv_path, on_bad_lines='skip')
            print(f"Loaded {len(test_df)} rows from test dataset.")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, fit_transformers: bool = False, 
                       enable_edge_removal: bool = False, tss_threshold: float = 0.0001) -> pd.DataFrame:
        """Preprocess the dataset using the same pipeline as GNN."""
        print("Starting data preprocessing...")
        
        # Remove unnecessary columns
        df = df.drop('is_backdoored', axis=1, errors='ignore')
        
        # Define columns to keep
        keep_cols = self.classification_features + self.linear_features + [self.target_feature]
        
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
        df['src_ch'] = df['src_ch'].astype(float).astype(int)
        df['dst_ch'] = df['dst_ch'].astype(float).astype(int)
        df = df.dropna().reset_index(drop=True)
        
        # Add layer distance features (if src_layer and dst_layer exist)
        if 'src_layer' in df.columns and 'dst_layer' in df.columns:
            print("Adding layer distance features...")
            df['layer_distance'] = df.apply(
                lambda row: self.layer_encoder.calculate_layer_distance(
                    row['src_layer'], row['dst_layer']
                ), axis=1
            )
            df['is_cross_stage'] = df.apply(
                lambda row: self.layer_encoder.is_cross_stage(
                    row['src_layer'], row['dst_layer']
                ), axis=1
            ).astype(int)
            df['is_skip_connection'] = df.apply(
                lambda row: self.layer_encoder.is_skip_connection(
                    row['src_layer'], row['dst_layer']
                ), axis=1
            ).astype(int)
            
            # Update linear features to include new distance features
            extended_linear_features = self.linear_features + ['layer_distance', 'is_cross_stage', 'is_skip_connection']
        else:
            extended_linear_features = self.linear_features
            print("Warning: src_layer and dst_layer not found, skipping layer distance features")
        
        # Handle categorical features differently for cross-architecture evaluation
        if 'src_layer' in df.columns and 'dst_layer' in df.columns:
            if fit_transformers:
                print("Encoding categorical features...")
                self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                encoded_categorical = self.encoder.fit_transform(df[['src_layer', 'dst_layer']])
                encoded_df = pd.DataFrame(
                    encoded_categorical, 
                    columns=self.encoder.get_feature_names_out(['src_layer', 'dst_layer'])
                )
                
                # Combine features
                df_part = df.drop(['src_layer', 'dst_layer'], axis=1)
                df = pd.concat([df_part, encoded_df], axis=1)
                
            elif self.encoder is not None:
                print("Applying existing categorical encoding with unknown handling...")
                try:
                    encoded_categorical = self.encoder.transform(df[['src_layer', 'dst_layer']])
                    encoded_df = pd.DataFrame(
                        encoded_categorical, 
                        columns=self.encoder.get_feature_names_out(['src_layer', 'dst_layer'])
                    )
                    
                    # Combine features
                    df_part = df.drop(['src_layer', 'dst_layer'], axis=1)
                    df = pd.concat([df_part, encoded_df], axis=1)
                    
                except Exception as e:
                    print(f"Cross-architecture encoding failed: {e}")
                    print("Skipping categorical encoding for test data (using numerical features only)")
                    # Just drop the categorical columns and continue with numerical features
                    df = df.drop(['src_layer', 'dst_layer'], axis=1, errors='ignore')
            else:
                print("Warning: No encoder available, skipping categorical encoding")
                df = df.drop(['src_layer', 'dst_layer'], axis=1, errors='ignore')
        
        # Scale numerical features
        available_linear_features = [f for f in extended_linear_features if f in df.columns]
        
        if fit_transformers:
            print("Fitting and transforming numerical features...")
            self.scaler = StandardScaler()
            df[available_linear_features] = self.scaler.fit_transform(df[available_linear_features])
        elif self.scaler is not None:
            print("Applying existing numerical scaling...")
            df[available_linear_features] = self.scaler.transform(df[available_linear_features])
        else:
            print("Warning: No scaler available, skipping numerical scaling")
        
        print(f"Preprocessing completed. Dataset shape: {df.shape}")
        return df
    
    def extract_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and target vector from processed dataframe."""
        # Get all feature columns (everything except target)
        feature_cols = [col for col in df.columns if col != self.target_feature]
        
        X = df[feature_cols].values
        y = df[self.target_feature].values
        
        print(f"Extracted features shape: {X.shape}")
        print(f"Extracted targets shape: {y.shape}")
        
        return X, y
    
    def align_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align features between train and test datasets for cross-architecture evaluation."""
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # Find common columns (excluding target)
        common_cols = train_cols.intersection(test_cols)
        if self.target_feature in common_cols:
            common_cols.remove(self.target_feature)
        
        # Always include target column
        final_cols = list(common_cols) + [self.target_feature]
        
        print(f"Feature alignment:")
        print(f"  Train features: {len(train_cols) - 1}")  # -1 for target
        print(f"  Test features: {len(test_cols) - 1}")    # -1 for target
        print(f"  Common features: {len(common_cols)}")
        
        # Filter both datasets to common columns
        train_aligned = train_df[final_cols].copy()
        test_aligned = test_df[final_cols].copy()
        
        print(f"  Aligned train shape: {train_aligned.shape}")
        print(f"  Aligned test shape: {test_aligned.shape}")
        
        return train_aligned, test_aligned
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "Model") -> Dict[str, float]:
        """Evaluate predictions using the same metrics as GNN system."""
        # Use the existing compute_metrics function from training_utils
        metrics = compute_metrics(y_pred.tolist(), y_true.tolist())
        
        print(f"\n{model_name} Results:")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"Spearman's ρ: {metrics['spearman']:.6f}")
        print(f"Kendall's τ: {metrics['kendall']:.6f}")
        
        return metrics
    
    def save_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    metrics: Dict[str, float], output_path: str, model_name: str = "Model"):
        """Save results to CSV file."""
        results_df = pd.DataFrame({
            'true_tss': y_true,
            f'{model_name.lower()}_pred': y_pred,
            'residuals': y_pred - y_true,
            'abs_residuals': np.abs(y_pred - y_true)
        })
        
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
            joblib.dump(self.scaler, os.path.join(output_dir, f'{self.architecture}_scaler.pkl'))
            print(f"Scaler saved for {self.architecture}")
        
        if self.encoder is not None:
            joblib.dump(self.encoder, os.path.join(output_dir, f'{self.architecture}_encoder.pkl'))
            print(f"Encoder saved for {self.architecture}")
    
    def load_preprocessors(self, output_dir: str = '.'):
        """Load preprocessors from files."""
        scaler_path = os.path.join(output_dir, f'{self.architecture}_scaler.pkl')
        encoder_path = os.path.join(output_dir, f'{self.architecture}_encoder.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded for {self.architecture}")
        
        if os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
            print(f"Encoder loaded for {self.architecture}")


def simple_baseline_predictor(X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray) -> np.ndarray:
    """Simple baseline predictor using mean or linear regression."""
    from sklearn.linear_model import LinearRegression
    
    # Try linear regression as baseline
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using Linear Regression baseline")
    except Exception as e:
        print(f"Linear regression failed: {e}")
        # Fallback to mean prediction
        y_pred = np.full(X_test.shape[0], np.mean(y_train))
        print("Using mean prediction baseline")
    
    return y_pred


def random_forest_predictor(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray) -> np.ndarray:
    """Random Forest predictor."""
    from sklearn.ensemble import RandomForestRegressor
    
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Using Random Forest predictor")
    except Exception as e:
        print(f"Random Forest failed: {e}")
        # Fallback to simple baseline
        y_pred = simple_baseline_predictor(X_train, y_train, X_test)
    
    return y_pred


def xgboost_predictor(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray) -> np.ndarray:
    """XGBoost predictor."""
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
        y_pred = random_forest_predictor(X_train, y_train, X_test)
    except Exception as e:
        print(f"XGBoost failed: {e}")
        y_pred = random_forest_predictor(X_train, y_train, X_test)
    
    return y_pred


def main():
    """Main function for evaluating other architectures."""
    parser = argparse.ArgumentParser(description='Evaluate other architectures on TSS prediction')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'mobilenetv2'],
                       help='Neural network architecture')
    parser.add_argument('--predictor', type=str, default='xgboost',
                       choices=['baseline', 'random_forest', 'xgboost'],
                       help='Predictor type to use')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--enable_edge_removal', action='store_true',
                       help='Enable edge removal based on TSS threshold')
    parser.add_argument('--tss_threshold', type=float, default=0.0001,
                       help='TSS threshold for edge removal')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Architecture Evaluation for TSS Prediction")
    print("=" * 60)
    print(f"Architecture: {args.architecture}")
    print(f"Predictor: {args.predictor}")
    print(f"Training data: {args.train_csv}")
    print(f"Test data: {args.test_csv}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = ArchitectureEvaluator(args.architecture)
        
        # Load data
        train_df, test_df = evaluator.load_data(args.train_csv, args.test_csv)
        
        # Preprocess training data (fit transformers)
        train_processed = evaluator.preprocess_data(
            train_df, 
            fit_transformers=True,
            enable_edge_removal=args.enable_edge_removal,
            tss_threshold=args.tss_threshold
        )
        
        # Preprocess test data (apply fitted transformers)
        test_processed = evaluator.preprocess_data(
            test_df,
            fit_transformers=False,
            enable_edge_removal=args.enable_edge_removal,
            tss_threshold=args.tss_threshold
        )
        
        # Align features for cross-architecture evaluation
        print(f"\n{'='*20} FEATURE ALIGNMENT {'='*20}")
        train_aligned, test_aligned = evaluator.align_features(train_processed, test_processed)
        
        # Extract features and targets
        X_train, y_train = evaluator.extract_features_and_targets(train_aligned)
        X_test, y_test = evaluator.extract_features_and_targets(test_aligned)
        
        # Choose predictor
        predictor_map = {
            'baseline': simple_baseline_predictor,
            'random_forest': random_forest_predictor,
            'xgboost': xgboost_predictor
        }
        
        predictor_func = predictor_map[args.predictor]
        
        print(f"\n{'='*20} TRAINING AND PREDICTION {'='*20}")
        
        # Make predictions
        y_pred = predictor_func(X_train, y_train, X_test)
        
        # Evaluate results
        print(f"\n{'='*20} EVALUATION {'='*20}")
        metrics = evaluator.evaluate_predictions(y_test, y_pred, args.predictor.title())
        
        # Save results
        output_path = os.path.join(args.output_dir, f'{args.architecture}_{args.predictor}_results.csv')
        evaluator.save_results(y_test, y_pred, metrics, output_path, args.predictor)
        
        # Save preprocessors
        evaluator.save_preprocessors(args.output_dir)
        
        print(f"\n{'='*20} EVALUATION COMPLETE {'='*20}")
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
