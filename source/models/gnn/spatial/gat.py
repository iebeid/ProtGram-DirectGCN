# ==============================================================================
# MODULE: models/gnn/gat.py
# PURPOSE: A standard implementation of the Graph Attention Network (GAT).
# VERSION: 10.0 (Refactored to be a standalone nn.Module for architectural correctness)
# AUTHOR: Islam Ebeid
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    A standard implementation of the Graph Attention Network (GAT) model.
    This architecture uses self-attention to weigh the importance of neighboring
    nodes during message passing.
    This is a standalone module because its layer-to-layer dimensions depend on
    the number of heads, which is incompatible with the generic BaseGNN class.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 8, num_layers: int = 2, dropout_rate: float = 0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        # --- DEFINITIVE FIX: Implement GAT as a standalone module for correctness ---
        if num_layers == 1:
            # For a single layer, we don't concatenate and use a single head for the output.
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False))
        else:
            # Input layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

            # Hidden layers (if any)
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))

            # Output layer
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index

        # For a single-layer model, the logits are also the embeddings.
        if len(self.convs) == 1:
            logits = self.convs[0](x, edge_index)
            self.embedding_output = logits
            return logits, self.embedding_output

        # Multi-layer case
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        self.embedding_output = x
        logits = self.convs[-1](self.embedding_output, edge_index)

        return logits, self.embedding_output