# ==============================================================================
# MODULE: models/gnn/spectral/dirgnn.py
# PURPOSE: Implements the Dir-GNN model from "Edge Directionality Improves
#          Learning on Heterophilic Graphs" using the official PyG wrapper.
# VERSION: 7.2 (Definitively fixed forward pass logic)
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
    Improves Learning on Heterophilic Graphs".
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
            base_conv = GCNConv(in_channels, out_channels)
            self.convs.append(DirGNNConv(conv=base_conv, alpha=0.5))
        else:
            base_conv_in = GCNConv(in_channels, hidden_channels)
            self.convs.append(DirGNNConv(conv=base_conv_in, alpha=0.5))
            self.norms.append(nn.LayerNorm(hidden_channels))

            for _ in range(num_layers - 2):
                base_conv_hidden = GCNConv(hidden_channels, hidden_channels)
                self.convs.append(DirGNNConv(conv=base_conv_hidden, alpha=0.5))
                self.norms.append(nn.LayerNorm(hidden_channels))

            base_conv_out = GCNConv(hidden_channels, out_channels)
            self.convs.append(DirGNNConv(conv=base_conv_out, alpha=0.5))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the DirGNN model.
        """
        x, edge_index = data.x, data.edge_index
        edge_index_backward = getattr(data, 'edge_index_backward', None)
        if edge_index_backward is None:
            raise ValueError("DirGNN requires 'edge_index_backward' in the Data object.")

        # --- DEFINITIVE FIX: Correctly loop through layers and apply activations ---
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_index_backward)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The embedding output is the feature representation before the final layer
        self.embedding_output = x

        # Apply the final layer to get logits
        x = self.convs[-1](x, edge_index, edge_index_backward)
        logits = x

        return logits, self.embedding_output