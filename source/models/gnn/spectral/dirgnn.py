# ==============================================================================
# MODULE: models/gnn/spectral/dirgnn.py
# PURPOSE: Implements the Dir-GNN model by manually handling the forward and
#          backward convolutions for maximum compatibility.
# VERSION: 8.0 (Definitively fixed TypeError by removing DirGNNConv wrapper)
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
    Improves Learning on Heterophilic Graphs". This version manually implements
    the core logic to avoid library version issues with the DirGNNConv wrapper.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5, alpha: float = 0.5, **kwargs):
        super().__init__()
        # We will now use the base GCNConv layers directly
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.alpha = alpha  # Store alpha for the forward pass
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        # Define the stack of GCN layers
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
        The forward pass for the DirGNN model with manual forward/backward logic.
        """
        x, edge_index = data.x, data.edge_index
        # DirGNN requires the backward edges. We can get these from 'edge_index_in'
        # if pre-calculated, or simply by flipping the forward edges.
        edge_index_backward = getattr(data, 'edge_index_in', None)
        if edge_index_backward is None:
            edge_index_backward = getattr(data, 'edge_index_backward', edge_index.flip(0))

        # Loop through intermediate layers
        for i in range(len(self.convs) - 1):
            conv = self.convs[i]
            # --- Manually perform the forward and backward convolutions ---
            x_forward = conv(x, edge_index)
            x_backward = conv(x, edge_index_backward)
            # --- Combine them using the alpha parameter ---
            x = (1 - self.alpha) * x_forward + self.alpha * x_backward

            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The embedding output is the feature representation before the final layer
        self.embedding_output = x

        # Apply the final layer to get the output logits
        final_conv = self.convs[-1]
        x_forward_final = final_conv(self.embedding_output, edge_index)
        x_backward_final = final_conv(self.embedding_output, edge_index_backward)
        logits = (1 - self.alpha) * x_forward_final + self.alpha * x_backward_final

        return logits, self.embedding_output