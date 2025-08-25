# ==============================================================================
# MODULE: models/gnn/graphsage.py
# PURPOSE: A standard implementation of the GraphSAGE model.
# VERSION: 5.0 (Refactored to be a standalone nn.Module for architectural correctness)
# AUTHOR: Islam Ebeid
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    A standard implementation of the GraphSAGE model.
    This architecture learns to aggregate feature information from a node's
    local neighborhood.
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

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, **kwargs))
            self.norms.append(nn.LayerNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, **kwargs))
                self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels, **kwargs))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # --- DEFINITIVE FIX: Apply LayerNorm BEFORE activation for stability ---
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        logits = self.convs[-1](self.embedding_output, edge_index)
        return logits, self.embedding_output
