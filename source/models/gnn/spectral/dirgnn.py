# ==============================================================================
# MODULE: models/gnn/spectral/dirgnn.py
# PURPOSE: Implements the Dir-GNN model from "Edge Directionality Improves
#          Learning on Heterophilic Graphs" using the official PyG wrapper.
# VERSION: 9.0 (Corrected to align with the official PyG DirGNNConv implementation)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import DirGNNConv, GCNConv


class DirGNN(nn.Module):
    """
    An implementation of the Dir-GNN model from the paper "Edge Directionality
    Improves Learning on Heterophilic Graphs", corrected to align with the
    official torch_geometric implementation.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        # --- DEFINITIVE FIX: Use the DirGNNConv wrapper exactly as designed ---
        # The wrapper takes a standard GNN layer (like GCNConv) and handles the
        # forward and backward message passing internally.
        if num_layers == 1:
            # For a single layer, we don't concatenate and use a single head for the output.
            base_conv = GCNConv(in_channels, out_channels, add_self_loops=False)
            self.convs.append(DirGNNConv(conv=base_conv, alpha=alpha, root_weight=True))
        else:
            # Input layer
            base_conv_in = GCNConv(in_channels, hidden_channels, add_self_loops=False)
            self.convs.append(DirGNNConv(conv=base_conv_in, alpha=alpha, root_weight=True))
            self.norms.append(nn.LayerNorm(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 2):
                base_conv_hidden = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
                self.convs.append(DirGNNConv(conv=base_conv_hidden, alpha=alpha, root_weight=True))
                self.norms.append(nn.LayerNorm(hidden_channels))

            # Output layer
            base_conv_out = GCNConv(hidden_channels, out_channels, add_self_loops=False)
            self.convs.append(DirGNNConv(conv=base_conv_out, alpha=alpha, root_weight=True))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the DirGNN model. It now only requires x and edge_index.
        """
        x, edge_index = data.x, data.edge_index

        # Loop through intermediate layers
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The embedding output is the feature representation before the final layer
        self.embedding_output = x

        # Apply the final layer to get the output logits
        logits = self.convs[-1](self.embedding_output, edge_index)

        return logits, self.embedding_output