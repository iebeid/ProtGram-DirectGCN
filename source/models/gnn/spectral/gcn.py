# ==============================================================================
# MODULE: models/gnn/gcn.py
# PURPOSE: A standard implementation of the Graph Convolutional Network (GCN).
# VERSION: 6.0 (Disabled redundant self-loops in GCNConv)
# AUTHOR: Islam Ebeid
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    A standard implementation of the Graph Convolutional Network (GCN) model.
    This architecture performs spectral-based convolutions on graphs.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        # --- DEFINITIVE FIX: Disable automatic self-loops in the GCNConv layers ---
        # The graph data is already pre-processed with self-loops during normalization,
        # so we set add_self_loops=False to prevent them from being added twice.
        kwargs['add_self_loops'] = False

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, **kwargs))
                self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels, **kwargs))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        if len(self.convs) == 1:
            logits = self.convs[0](x, edge_index, edge_weight=edge_weight)
            self.embedding_output = logits
            return logits, self.embedding_output

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        self.embedding_output = x
        logits = self.convs[-1](self.embedding_output, edge_index, edge_weight)

        return logits, self.embedding_output