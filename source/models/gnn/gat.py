from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, RGCNConv, GINConv
from torch_geometric.data import Data
from source.utils.models_utils import BaseGNN

class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_layers=2, dropout_rate=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate # GATConv has its own dropout
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout_rate))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        # GATConv can take edge_attr for weighted attention, if edge_dim matches.
        # For simplicity, if edge_weight is 1D, it might be used. If multi-dim, GATConv might error or ignore.
        # PyG GATConv v2 handles edge_attr more explicitly.
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_weight if conv.edge_dim is not None else None)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                # Dropout is typically part of GATConv itself
        self.embedding_output = x
        return x