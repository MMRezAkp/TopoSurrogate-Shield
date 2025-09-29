"""
Model definitions for GNN training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GATLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * out_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of GAT layer."""
        edge_index = edge_index.t()
        Wh = self.W(h)
        src, tgt = edge_index[0], edge_index[1]
        e_src = Wh[src]
        e_tgt = Wh[tgt]
        e_concat = torch.cat([e_src, e_tgt], dim=1)
        attn = self.leaky_relu(self.a(e_concat)).squeeze()
        attn = torch.softmax(attn, dim=0)
        attn = self.dropout(attn)
        
        messages = torch.zeros_like(Wh)
        messages.index_add_(0, tgt, attn.unsqueeze(1) * e_src)
        return self.leaky_relu(messages)

class GNNSurrogate(nn.Module):
    """GNN-based surrogate model for TSS prediction."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(GNNSurrogate, self).__init__()
        self.gat1 = GATLayer(node_dim, hidden_dim, dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, dropout)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass of GNN surrogate."""
        # Apply GAT layers
        h1 = self.gat1(node_feats, edge_index)
        h2 = self.gat2(h1, edge_index)
        
        # Get edge representations
        edge_index = edge_index.t()
        src, tgt = edge_index[0], edge_index[1]
        src_emb = h2[src]
        tgt_emb = h2[tgt]
        
        # Combine edge features
        edge_input = torch.cat([src_emb, tgt_emb, edge_attr], dim=1)
        return self.head(edge_input).squeeze()

class MLP(nn.Module):
    """Multi-layer perceptron for comparison."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        return self.layers(x).squeeze()

class ResidualGNN(nn.Module):
    """GNN with residual connections for better gradient flow."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(ResidualGNN, self).__init__()
        self.gat1 = GATLayer(node_dim, hidden_dim, dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, dropout)
        self.gat3 = GATLayer(hidden_dim, hidden_dim, dropout)
        
        # Residual projection
        self.residual_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # First GAT layer
        h1 = self.gat1(node_feats, edge_index)
        
        # Second GAT layer with residual connection
        h2 = self.gat2(h1, edge_index)
        h2 = h2 + h1  # Residual connection
        
        # Third GAT layer
        h3 = self.gat3(h2, edge_index)
        
        # Get edge representations
        edge_index = edge_index.t()
        src, tgt = edge_index[0], edge_index[1]
        src_emb = h3[src]
        tgt_emb = h3[tgt]
        
        # Combine edge features
        edge_input = torch.cat([src_emb, tgt_emb, edge_attr], dim=1)
        return self.head(edge_input).squeeze()

def create_model(model_type: str, node_dim: int, edge_dim: int, **kwargs) -> nn.Module:
    """Factory function to create models."""
    if model_type.lower() == 'gnn':
        return GNNSurrogate(node_dim, edge_dim, **kwargs)
    elif model_type.lower() == 'mlp':
        return MLP(edge_dim, **kwargs)
    elif model_type.lower() == 'residual_gnn':
        return ResidualGNN(node_dim, edge_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model: nn.Module) -> dict:
    """Get information about the model."""
    return {
        'total_params': count_parameters(model),
        'model_type': type(model).__name__,
        'device': next(model.parameters()).device
    }




