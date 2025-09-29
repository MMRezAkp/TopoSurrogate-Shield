"""
Training utilities and functions
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau

from config import MODEL_CONFIG, LOSS_WEIGHTS
from loss_functions import compute_combined_loss, set_linear_features

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should stop early."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save the current best model weights."""
        self.best_weights = model.state_dict().copy()

class TrainingLogger:
    """Logger for training metrics and progress."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_rmse': [],
            'epoch_times': []
        }
        self.loss_components = {
            'mse': [],
            'mono': [],
            'shortcut': [],
            'consistency': [],
            'ranking': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  test_rmse: float, epoch_time: float, loss_components: Dict[str, float]):
        """Log metrics for a single epoch."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['test_rmse'].append(test_rmse)
        self.metrics['epoch_times'].append(epoch_time)
        
        for key, value in loss_components.items():
            self.loss_components[key].append(value)
        
        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | Test RMSE: {test_rmse:.6f} | '
              f'Time: {epoch_time:.2f}s')
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best validation loss."""
        return np.argmin(self.metrics['val_loss'])
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics achieved during training."""
        best_epoch = self.get_best_epoch()
        return {
            'epoch': best_epoch,
            'train_loss': self.metrics['train_loss'][best_epoch],
            'val_loss': self.metrics['val_loss'][best_epoch],
            'test_rmse': self.metrics['test_rmse'][best_epoch]
        }

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, node_feats: torch.Tensor, loss_weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_components = {'mse': 0.0, 'mono': 0.0, 'shortcut': 0.0, 'consistency': 0.0, 'ranking': 0.0}
    num_batches = 0
    
    for batch_edge_idx, batch_edge_attr, batch_y in train_loader:
        batch_edge_idx = batch_edge_idx.to(device)
        batch_edge_attr = batch_edge_attr.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(node_feats.to(device), batch_edge_idx, batch_edge_attr)
        
        # Compute loss
        loss, loss_components = compute_combined_loss(
            out, batch_y, batch_edge_attr, batch_edge_idx, node_feats, loss_weights
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for key, value in loss_components.items():
            total_loss_components[key] += value
        num_batches += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_loss_components = {key: value / num_batches for key, value in total_loss_components.items()}
    
    return avg_loss, avg_loss_components

def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, device: torch.device,
                   node_feats: torch.Tensor, criterion: torch.nn.Module) -> Tuple[float, List[float], List[float]]:
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_edge_idx, batch_edge_attr, batch_y in data_loader:
            batch_edge_idx = batch_edge_idx.to(device)
            batch_edge_attr = batch_edge_attr.to(device)
            batch_y = batch_y.to(device)
            
            out = model(node_feats.to(device), batch_edge_idx, batch_edge_attr)
            loss = criterion(out, batch_y)
            
            total_loss += loss.item()
            predictions.extend(out.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, targets

def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                test_loader: DataLoader, node_feats: torch.Tensor, device: torch.device,
                max_epochs: int = 40, learning_rate: float = 0.005, 
                weight_decay: float = 1e-5, patience: int = 10,
                loss_weights: dict = None) -> Tuple[torch.nn.Module, TrainingLogger]:
    """Train the model with early stopping and logging."""
    
    # Set up training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)
    logger = TrainingLogger()
    
    # Use custom loss weights if provided, otherwise use defaults
    if loss_weights is None:
        loss_weights = LOSS_WEIGHTS
    
    # Set linear features for loss functions
    set_linear_features(['w_out_norm_a', 'w_out_norm_b', 'w_norm_ratio',
                        'depth_a', 'depth_b', 'depth_diff',
                        'fan_in_a', 'fan_in_b', 'fan_out_a', 'fan_out_b',
                        'fan_in_ratio', 'fan_out_ratio',
                        'grad_out_norm_a', 'grad_out_norm_b', 'grad_norm_ratio',
                        'act_mean_a', 'act_mean_b', 'act_var_a', 'act_var_b',
                        'act_std_a', 'act_std_b', 'act_mean_diff', 'act_std_ratio',
                        'layer_distance', 'is_cross_stage', 'is_skip_connection'])
    
    print(f"Starting training for {max_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(max_epochs):
        start_time = time.time()
        
        # Training
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, node_feats, loss_weights
        )
        
        # Validation
        val_loss, _, _ = evaluate_model(model, val_loader, device, node_feats, criterion)
        
        # Test evaluation
        test_loss, test_preds, test_targets = evaluate_model(model, test_loader, device, node_feats, criterion)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.log_epoch(epoch, train_loss, val_loss, test_rmse, epoch_time, loss_components)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Get best metrics
    best_metrics = logger.get_best_metrics()
    print(f"\nTraining completed!")
    print(f"Best epoch: {best_metrics['epoch']}")
    print(f"Best validation loss: {best_metrics['val_loss']:.6f}")
    print(f"Best test RMSE: {best_metrics['test_rmse']:.6f}")
    
    return model, logger

def train_mlp(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              test_loader: DataLoader, device: torch.device, max_epochs: int = 45,
              learning_rate: float = 0.0005, weight_decay: float = 1e-5, 
              patience: int = 10) -> Tuple[torch.nn.Module, TrainingLogger]:
    """Train MLP model for comparison."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)
    logger = TrainingLogger()
    
    print(f"Starting MLP training for {max_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(max_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0.0
        for _, batch_edge_attr, batch_y in train_loader:
            batch_edge_attr = batch_edge_attr.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_edge_attr)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, _, _ = evaluate_mlp(model, val_loader, device, criterion)
        
        # Test evaluation
        test_loss, test_preds, test_targets = evaluate_mlp(model, test_loader, device, criterion)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.log_epoch(epoch, train_loss, val_loss, test_rmse, epoch_time, {'mse': train_loss})
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Get best metrics
    best_metrics = logger.get_best_metrics()
    print(f"\nMLP training completed!")
    print(f"Best epoch: {best_metrics['epoch']}")
    print(f"Best validation loss: {best_metrics['val_loss']:.6f}")
    print(f"Best test RMSE: {best_metrics['test_rmse']:.6f}")
    
    return model, logger

def evaluate_mlp(model: torch.nn.Module, data_loader: DataLoader, device: torch.device,
                 criterion: torch.nn.Module) -> Tuple[float, List[float], List[float]]:
    """Evaluate MLP model on a dataset."""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for _, batch_edge_attr, batch_y in data_loader:
            batch_edge_attr = batch_edge_attr.to(device)
            batch_y = batch_y.to(device)
            
            out = model(batch_edge_attr)
            loss = criterion(out, batch_y)
            
            total_loss += loss.item()
            predictions.extend(out.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, targets

def compute_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = np.mean(np.abs(targets - predictions))
    
    # Correlation metrics
    spearman = np.nan
    kendall = np.nan
    
    if np.std(targets) > 1e-10 and np.std(predictions) > 1e-10:
        try:
            spearman, _ = spearmanr(targets, predictions)
        except:
            pass
        try:
            kendall, _ = kendalltau(targets, predictions)
        except:
            pass
    
    return {
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'kendall': kendall
    }
