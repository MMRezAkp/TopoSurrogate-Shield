"""
Physical loss functions for GNN training
"""
import torch
import torch.nn as nn
from typing import List

# Global variable to store linear features for loss functions
linear_features = []

def set_linear_features(features: List[str]):
    """Set the linear features list for loss functions."""
    global linear_features
    linear_features = features

def compute_monotonicity_loss(preds: torch.Tensor, edge_attrs: torch.Tensor, var_indices: List[int] = [5, 7]) -> torch.Tensor:
    """Compute monotonicity penalty for activation variances."""
    mono_penalty = 0.0
    n = preds.size(0)
    if n < 2:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    for idx in var_indices:
        if idx < edge_attrs.shape[1]:
            var = edge_attrs[:, idx]
            sorted_idx = torch.argsort(var)
            sorted_preds = preds[sorted_idx]
            diffs = sorted_preds[:-1] - sorted_preds[1:]
            penalty = torch.clamp(diffs, min=0).sum()
            mono_penalty += penalty / (n - 1)

    return mono_penalty

def compute_shortcut_awareness_loss(preds: torch.Tensor, edge_attrs: torch.Tensor, edge_indices: torch.Tensor, 
                                   node_features: torch.Tensor, lambda_shortcut: float = 0.2, 
                                   lambda_consistency: float = 0.1) -> torch.Tensor:
    """Compute shortcut-aware loss that highlights the effect of shortcuts in predicting TSS."""
    device = preds.device
    batch_size = preds.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Get feature indices safely
    try:
        depth_a_idx = linear_features.index('depth_a')
        depth_b_idx = linear_features.index('depth_b')
        layer_distance_idx = linear_features.index('layer_distance')
        w_norm_a_idx = linear_features.index('w_out_norm_a')
        w_norm_b_idx = linear_features.index('w_out_norm_b')
        fan_in_a_idx = linear_features.index('fan_in_a')
        fan_in_b_idx = linear_features.index('fan_in_b')
    except ValueError as e:
        print(f"Warning: Feature not found in linear_features: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Extract relevant features for shortcut detection
    depth_a = edge_attrs[:, depth_a_idx]
    depth_b = edge_attrs[:, depth_b_idx]
    depth_diff = torch.abs(depth_a - depth_b)
    layer_distance = edge_attrs[:, layer_distance_idx]
    
    w_norm_a = edge_attrs[:, w_norm_a_idx]
    w_norm_b = edge_attrs[:, w_norm_b_idx]
    fan_in_a = edge_attrs[:, fan_in_a_idx]
    fan_in_b = edge_attrs[:, fan_in_b_idx]
    
    # 1. Shortcut Strength Penalty
    shortcut_strength = (w_norm_a * w_norm_b) / (fan_in_a * fan_in_b + 1e-8)
    shortcut_strength = torch.clamp(shortcut_strength, 0, 1)
    
    # Penalty: TSS predictions should be higher for stronger shortcuts
    shortcut_penalty = torch.mean(torch.clamp(shortcut_strength - preds, min=0))
    
    # 2. Layer Distance Awareness
    # Higher TSS for larger layer distances (skip connections)
    distance_penalty = torch.mean(torch.clamp(preds - layer_distance * 0.1, min=0))
    
    # 3. Cross-Stage Connection Bonus
    cross_stage_mask = depth_diff > 1.0
    cross_stage_bonus = torch.mean(
        torch.clamp(preds[cross_stage_mask] - 0.1, min=0)
    ) if cross_stage_mask.any() else torch.tensor(0.0, device=device)
    
    # 4. Weight Ratio Consistency
    w_ratio = w_norm_a / (w_norm_b + 1e-8)
    w_ratio_consistency = torch.mean(torch.abs(preds - torch.tanh(w_ratio * 0.5)))
    
    # 5. Skip Connection Bonus
    skip_mask = layer_distance > 3.0
    skip_bonus = torch.mean(
        torch.clamp(preds[skip_mask] - 0.15, min=0)
    ) if skip_mask.any() else torch.tensor(0.0, device=device)
    
    # Combine all losses
    total_shortcut_loss = (
        lambda_shortcut * shortcut_penalty +
        lambda_consistency * w_ratio_consistency +
        0.1 * distance_penalty +
        0.1 * cross_stage_bonus +
        0.1 * skip_bonus
    )
    
    return total_shortcut_loss

def compute_shortcut_consistency_loss(preds: torch.Tensor, edge_attrs: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
    """Ensure TSS predictions are consistent with shortcut characteristics."""
    device = preds.device
    batch_size = preds.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    try:
        # Extract features
        depth_a_idx = linear_features.index('depth_a')
        depth_b_idx = linear_features.index('depth_b')
        layer_distance_idx = linear_features.index('layer_distance')
        w_norm_a_idx = linear_features.index('w_out_norm_a')
        w_norm_b_idx = linear_features.index('w_out_norm_b')
    except ValueError as e:
        print(f"Warning: Feature not found in linear_features: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    depth_a = edge_attrs[:, depth_a_idx]
    depth_b = edge_attrs[:, depth_b_idx]
    depth_diff = torch.abs(depth_a - depth_b)
    layer_distance = edge_attrs[:, layer_distance_idx]
    
    w_norm_a = edge_attrs[:, w_norm_a_idx]
    w_norm_b = edge_attrs[:, w_norm_b_idx]
    
    # Consistency 1: High TSS for cross-layer connections
    cross_layer_mask = depth_diff > 2.0
    cross_layer_consistency = torch.mean(
        torch.clamp(preds[cross_layer_mask] - 0.1, min=0)
    ) if cross_layer_mask.any() else torch.tensor(0.0, device=device)
    
    # Consistency 2: High TSS for high-weight connections
    high_weight_mask = (w_norm_a > 0.5) & (w_norm_b > 0.5)
    high_weight_consistency = torch.mean(
        torch.clamp(preds[high_weight_mask] - 0.15, min=0)
    ) if high_weight_mask.any() else torch.tensor(0.0, device=device)
    
    # Consistency 3: High TSS for skip connections
    skip_mask = layer_distance > 4.0
    skip_consistency = torch.mean(
        torch.clamp(preds[skip_mask] - 0.2, min=0)
    ) if skip_mask.any() else torch.tensor(0.0, device=device)
    
    # Consistency 4: Low TSS for same-layer connections
    same_layer_mask = depth_diff < 0.5
    same_layer_consistency = torch.mean(
        torch.clamp(0.05 - preds[same_layer_mask], min=0)
    ) if same_layer_mask.any() else torch.tensor(0.0, device=device)
    
    total_consistency = (
        cross_layer_consistency +
        high_weight_consistency +
        skip_consistency +
        same_layer_consistency
    )
    
    return total_consistency

def compute_shortcut_ranking_loss(preds: torch.Tensor, edge_attrs: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
    """Ensure TSS predictions maintain proper ranking based on shortcut characteristics."""
    device = preds.device
    batch_size = preds.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    try:
        # Calculate shortcut scores for ranking
        depth_a_idx = linear_features.index('depth_a')
        depth_b_idx = linear_features.index('depth_b')
        layer_distance_idx = linear_features.index('layer_distance')
        w_norm_a_idx = linear_features.index('w_out_norm_a')
        w_norm_b_idx = linear_features.index('w_out_norm_b')
        fan_in_a_idx = linear_features.index('fan_in_a')
        fan_in_b_idx = linear_features.index('fan_in_b')
    except ValueError as e:
        print(f"Warning: Feature not found in linear_features: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    depth_a = edge_attrs[:, depth_a_idx]
    depth_b = edge_attrs[:, depth_b_idx]
    depth_diff = torch.abs(depth_a - depth_b)
    layer_distance = edge_attrs[:, layer_distance_idx]
    
    w_norm_a = edge_attrs[:, w_norm_a_idx]
    w_norm_b = edge_attrs[:, w_norm_b_idx]
    fan_in_a = edge_attrs[:, fan_in_a_idx]
    fan_in_b = edge_attrs[:, fan_in_b_idx]
    
    # Calculate shortcut score (higher = more likely to be shortcut)
    shortcut_score = (
        layer_distance * 0.4 +  # Layer distance is most important
        depth_diff * 0.3 +      # Cross-layer bonus
        (w_norm_a * w_norm_b) / (fan_in_a * fan_in_b + 1e-8) * 0.3  # Weight connectivity
    )
    
    # Sort by shortcut score
    sorted_indices = torch.argsort(shortcut_score, descending=True)
    sorted_preds = preds[sorted_indices]
    
    # Ranking loss: TSS should decrease as shortcut score decreases
    ranking_penalty = 0.0
    for i in range(batch_size - 1):
        if sorted_preds[i] < sorted_preds[i + 1]:
            ranking_penalty += (sorted_preds[i + 1] - sorted_preds[i])
    
    return ranking_penalty / (batch_size - 1)

def compute_combined_loss(preds: torch.Tensor, targets: torch.Tensor, edge_attrs: torch.Tensor, 
                         edge_indices: torch.Tensor, node_features: torch.Tensor,
                         loss_weights: dict) -> torch.Tensor:
    """Compute combined loss with all physical constraints."""
    # Base MSE loss
    mse_loss = nn.MSELoss()(preds, targets)
    
    # Physical losses
    mono_loss = compute_monotonicity_loss(preds, edge_attrs)
    shortcut_loss = compute_shortcut_awareness_loss(
        preds, edge_attrs, edge_indices, node_features,
        loss_weights.get('lambda_shortcut', 0.2),
        loss_weights.get('lambda_consistency', 0.1)
    )
    consistency_loss = compute_shortcut_consistency_loss(preds, edge_attrs, edge_indices)
    ranking_loss = compute_shortcut_ranking_loss(preds, edge_attrs, edge_indices)
    
    # Combined loss
    total_loss = (
        mse_loss +
        loss_weights.get('lambda_mono', 0.15) * mono_loss +
        loss_weights.get('lambda_shortcut', 0.001) * shortcut_loss +
        loss_weights.get('lambda_consistency', 0.0) * consistency_loss +
        loss_weights.get('lambda_ranking', 0.0) * ranking_loss
    )
    
    # Helper function to safely get item value
    def safe_item(value):
        return value.item() if hasattr(value, 'item') else float(value)
    
    return total_loss, {
        'mse': safe_item(mse_loss),
        'mono': safe_item(mono_loss),
        'shortcut': safe_item(shortcut_loss),
        'consistency': safe_item(consistency_loss),
        'ranking': safe_item(ranking_loss)
    }
