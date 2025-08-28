# ==============================================================================
# MODULE: models/gnn/spectral/dirgnn.py
# PURPOSE: Implements the Dir-GNN model from "Edge Directionality Improves
#          Learning on Heterophilic Graphs" using the official PyG wrapper.
# VERSION: 7.5 (Removed unexpected edge_weight argument from forward pass)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class DirGNN(nn.Module):
    """
    An implementation of the Dir-GNN model from the paper "Edge Directionality
    Improves Learning on Heterophilic Graphs".
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.alpha = 0.5  # As specified in the original paper
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the DirGNN model.
        """
        x, edge_index, edge_index_backward = data.x, data.edge_index, getattr(data, 'edge_index_backward', None)

        for i in range(len(self.convs) - 1):
            # --- DEFINITIVE FIX: Manually implement the Dir-GNN logic ---
            # This bypasses the problematic DirGNNConv wrapper.
            h_forward = self.convs[i](x, edge_index)
            h_backward = self.convs[i](x, edge_index_backward)
            x = self.alpha * h_forward + (1 - self.alpha) * h_backward
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        self.embedding_output = x
        h_forward = self.convs[-1](self.embedding_output, edge_index)
        h_backward = self.convs[-1](self.embedding_output, edge_index_backward)
        logits = self.alpha * h_forward + (1 - self.alpha) * h_backward

        return logits, self.embedding_output
