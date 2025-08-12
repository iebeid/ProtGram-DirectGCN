# ==============================================================================
# MODULE: models/gnn/spectral/dirgnn.py
# PURPOSE: Implements the Dir-GNN model from "Edge Directionality Improves
#          Learning on Heterophilic Graphs" using the official PyG wrapper.
# VERSION: 7.0 (Corrected implementation to align with PyG's DirGNNConv wrapper pattern)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
# --- FIX: Import both the wrapper and the base convolution layer ---
from torch_geometric.nn import DirGNNConv, GCNConv


class DirGNN(nn.Module):
    """
    An implementation of the Dir-GNN model from the paper "Edge Directionality
    Improves Learning on Heterophilic Graphs".

    This model correctly uses the official PyG `DirGNNConv` layer, which acts
    as a wrapper around a standard convolution (like GCNConv) to handle
    separate forward and backward message passing.
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

        # --- DEFINITIVE FIX: Correctly use DirGNNConv as a wrapper ---
        if num_layers == 1:
            # 1. Create the base convolution layer.
            base_conv = GCNConv(in_channels, out_channels)
            # 2. Wrap it with DirGNNConv.
            self.convs.append(DirGNNConv(conv=base_conv, alpha=0.5))
        else:
            # Input layer
            base_conv_in = GCNConv(in_channels, hidden_channels)
            self.convs.append(DirGNNConv(conv=base_conv_in, alpha=0.5))

            # Hidden layers
            for _ in range(num_layers - 2):
                base_conv_hidden = GCNConv(hidden_channels, hidden_channels)
                self.convs.append(DirGNNConv(conv=base_conv_hidden, alpha=0.5))
                self.norms.append(nn.LayerNorm(hidden_channels))

            # Output layer
            base_conv_out = GCNConv(hidden_channels, out_channels)
            self.convs.append(DirGNNConv(conv=base_conv_out, alpha=0.5))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the DirGNN model. The DirGNNConv wrapper handles
        the backward pass internally, so only `edge_index` is needed.
        """
        x, edge_index = data.x, data.edge_index
        edge_index_backward = getattr(data, 'edge_index_backward', None)
        if edge_index_backward is None:
            raise ValueError("DirGNN requires 'edge_index_backward' in the Data object.")

        # --- DEFINITIVE FIX: Correctly handle single-layer case and apply final layer ---
        if len(self.convs) == 1:
            logits = self.convs0
            self.embedding_output = logits
            return logits, self.embedding_output

        # Multi-layer case
        for i, conv in enumerate(self.convs[:-1]):
            # Pass both forward and backward edges to the wrapper.
            x = conv(x, edge_index, edge_index_backward)
            if i < len(self.norms):
                x = self.normsi # Apply norm before activation
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        self.embedding_output = x
        logits = self.convs-1

        return logits, self.embedding_output