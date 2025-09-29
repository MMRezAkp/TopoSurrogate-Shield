"""
Data loading and preprocessing module for GNN training
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional
import os

from config import DATA_PATHS, PREPROCESSING
from layer_encoders import get_layer_encoder

class DataProcessor:
    """Handles data loading, preprocessing, and graph construction."""
    
    def __init__(self, architecture: str = 'efficientnet'):
        self.architecture = architecture
        self.layer_encoder = get_layer_encoder(architecture)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Define features
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
        
    def load_data(self, backdoored_path: str, clean_path: str) -> pd.DataFrame:
        """Load and combine backdoored and clean datasets."""
        try:
            # Load backdoored dataset
            df_backdoored = pd.read_csv(backdoored_path, on_bad_lines='skip')
            print(f"Loaded {len(df_backdoored)} rows from backdoored dataset.")
            
            # Load clean dataset
            df_clean = pd.read_csv(clean_path, on_bad_lines='skip')
            print(f"Loaded {len(df_clean)} rows from clean dataset.")
            
            # Combine datasets
            df = pd.concat([df_backdoored, df_clean], ignore_index=True)
            print(f"Combined dataset has {len(df)} total rows.")
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, enable_edge_removal: bool = False, tss_threshold: float = 0.0001) -> pd.DataFrame:
        """Preprocess the dataset."""
        print("Starting data preprocessing...")
        
        # Remove unnecessary columns
        df = df.drop('is_backdoored', axis=1, errors='ignore')
        
        # Define columns to keep
        keep_cols = self.classification_features + self.linear_features + [self.target_feature]
        df = df[keep_cols]
        
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
        
        # Add layer distance features
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
        self.linear_features.extend(['layer_distance', 'is_cross_stage', 'is_skip_connection'])
        
        print(f"Preprocessing completed. Dataset shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features."""
        print("Encoding categorical features...")
        
        # One-hot encode categorical features
        encoded_categorical = self.encoder.fit_transform(df[['src_layer', 'dst_layer']])
        encoded_df = pd.DataFrame(
            encoded_categorical, 
            columns=self.encoder.get_feature_names_out(['src_layer', 'dst_layer'])
        )
        
        # Combine features
        df_part = df.drop(['src_layer', 'dst_layer'], axis=1)
        df_encoded = pd.concat([df_part, encoded_df], axis=1)
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        print("Scaling numerical features...")
        
        df_scaled = df.copy()
        df_scaled[self.linear_features] = self.scaler.fit_transform(df[self.linear_features])
        
        return df_scaled
    
    def build_graph(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph from preprocessed data."""
        print("Building graph...")
        
        # Get all unique channels
        all_channels = pd.concat([df['src_ch'], df['dst_ch']]).unique()
        node_to_idx = {ch: i for i, ch in enumerate(all_channels)}
        num_nodes = len(all_channels)
        
        print(f"Graph has {num_nodes} nodes and {len(df)} edges")
        
        # Build edge indices and features
        edge_indices = []
        edge_feats = []
        node_features = defaultdict(list)
        
        for idx, row in df.iterrows():
            if pd.isna(row['src_ch']) or pd.isna(row['dst_ch']):
                continue
            try:
                src_ch, dst_ch = int(row['src_ch']), int(row['dst_ch'])
                src_idx, dst_idx = node_to_idx[src_ch], node_to_idx[dst_ch]
                edge_indices.append([src_idx, dst_idx])
                edge_feats.append(row[self.linear_features].values)
                node_features[src_ch].append(row[self.linear_features].values)
                node_features[dst_ch].append(row[self.linear_features].values)
            except (ValueError, TypeError):
                continue
        
        # Compute mean node features
        node_feats_list = []
        for ch in all_channels:
            feats = np.mean(node_features.get(ch, [np.zeros(len(self.linear_features))]), axis=0)
            node_feats_list.append(feats)
        node_feats = torch.tensor(np.array(node_feats_list), dtype=torch.float)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)
        y_tensor = torch.tensor(df[self.target_feature].values, dtype=torch.float)
        
        return edge_index, edge_attr, node_feats, y_tensor
    
    def create_data_loaders(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                          y_tensor: torch.Tensor, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        print("Creating data loaders...")
        
        # Train-test-validation split
        train_idx, test_idx = train_test_split(
            range(edge_index.shape[1]), 
            test_size=PREPROCESSING['test_size'], 
            random_state=PREPROCESSING['random_state']
        )
        train_idx, val_idx = train_test_split(
            train_idx, 
            test_size=PREPROCESSING['val_size'], 
            random_state=PREPROCESSING['random_state']
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            edge_index[:, train_idx].t(),
            edge_attr[train_idx],
            y_tensor[train_idx]
        )
        val_dataset = TensorDataset(
            edge_index[:, val_idx].t(),
            edge_attr[val_idx],
            y_tensor[val_idx]
        )
        test_dataset = TensorDataset(
            edge_index[:, test_idx].t(),
            edge_attr[test_idx],
            y_tensor[test_idx]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_loader, val_loader, test_loader
    
    def process_pipeline(self, backdoored_path: str, clean_path: str, enable_edge_removal: bool = False, tss_threshold: float = 0.0001) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Complete data processing pipeline."""
        # Load data
        df = self.load_data(backdoored_path, clean_path)
        
        # Preprocess
        df = self.preprocess_data(df, enable_edge_removal, tss_threshold)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale features
        df = self.scale_features(df)
        
        # Build graph
        edge_index, edge_attr, node_feats, y_tensor = self.build_graph(df)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            edge_index, edge_attr, y_tensor, batch_size=32
        )
        
        return train_loader, val_loader, test_loader, edge_index, edge_attr, node_feats
    
    def save_preprocessors(self, output_dir: str = '.'):
        """Save preprocessors for later use."""
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.encoder, os.path.join(output_dir, 'encoder.pkl'))
        print(f"Preprocessors saved to {output_dir}")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features."""
        return {
            'classification_features': self.classification_features,
            'linear_features': self.linear_features,
            'target_feature': self.target_feature,
            'total_features': len(self.classification_features) + len(self.linear_features),
            'edge_dim': len(self.linear_features),
            'node_dim': len(self.linear_features)
        }
