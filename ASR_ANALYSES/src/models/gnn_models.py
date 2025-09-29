"""
Graph Neural Network models for TSS prediction.
"""

import torch
import torch.nn as nn


class FixedGATLayer(nn.Module):
    """Fixed Graph Attention Network layer."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * out_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h, edge_index):
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        Wh = self.W(h)
        src, tgt = edge_index[0], edge_index[1]
        e_src, e_tgt = Wh[src], Wh[tgt]
        e_concat = torch.cat([e_src, e_tgt], dim=1)
        attn = self.leaky_relu(self.a(e_concat)).squeeze()
        if attn.dim() == 0:
            attn = attn.unsqueeze(0)
        attn = torch.softmax(attn, dim=0)
        messages = torch.zeros_like(Wh)
        for i, (si, ti) in enumerate(zip(src, tgt)):
            messages[ti] += attn[i] * e_src[i]
        return self.leaky_relu(messages)


class FixedGNNSurrogate(nn.Module):
    """Fixed GNN surrogate model for TSS prediction."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.gat1 = FixedGATLayer(node_dim, hidden_dim)
        self.gat2 = FixedGATLayer(hidden_dim, hidden_dim)
        
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

    def forward(self, node_feats, edge_index, edge_attr):
        h1 = self.gat1(node_feats, edge_index)
        h2 = self.gat2(h1, edge_index)
        
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        src, tgt = edge_index[0], edge_index[1]
        src_emb, tgt_emb = h2[src], h2[tgt]
        edge_input = torch.cat([src_emb, tgt_emb, edge_attr], dim=1)
        return self.head(edge_input).squeeze()



