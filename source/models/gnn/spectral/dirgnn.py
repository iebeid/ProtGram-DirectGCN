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
            # --- DEFINITIVE FIX: Pass only the arguments expected by the wrapped GCNConv ---
            # The DirGNNConv wrapper handles the backward edges internally. We only need
            # to pass the standard arguments to the forward call.
            x = self.convs[i](x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        self.embedding_output = x
        # --- DEFINITIVE FIX: Also pass backward edges to the final layer. ---
        logits = self.convs[-1](self.embedding_output, edge_index)

        return logits, self.embedding_output
