
# ==============================================================================
# MODULE: models/gnn/gat.py
# PURPOSE: Contains the PyTorch class definitions for the custom GAT model.
# VERSION: 8.3 (Stable & Corrected - Cleaned up hierarchical gating and dual-path logic)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from source.utils.models import BaseGNN


class GAT(BaseGNN):
    """
    A standard implementation of the Graph Attention Network (GAT) model.
    This architecture uses self-attention to weigh the importance of
    neighboring nodes during message passing.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 8, num_layers: int = 2, dropout_rate: float = 0.6):
        """
        Initializes the GAT model layers.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            heads (int): Number of attention heads. Defaults to 8.
            num_layers (int): The number of GAT layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply. Defaults to 0.6.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate

        if num_layers == 1:
            # If only one layer, it goes directly from in to out.
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout_rate))
        else:
            # Input layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
            # Output layer
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GAT model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index = data.x, data.edge_index

        # If there's only one layer, the logic is simpler.
        if len(self.convs) == 1:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            logits = self.convs[0](x, edge_index)
            # For a single-layer model, the output logits are also the embeddings.
            self.embedding_output = logits
            return logits, self.embedding_output

        # Process hidden layers
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        x_for_final_layer = F.dropout(self.embedding_output, p=self.dropout_rate, training=self.training)
        logits = self.convs[-1](x_for_final_layer, edge_index)

        return logits, self.embedding_output