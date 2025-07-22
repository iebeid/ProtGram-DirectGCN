# ==============================================================================
# MODULE: models/gnn/gcn.py
# PURPOSE: A standard implementation of the Graph Convolutional Network (GCN).
# VERSION: 2.0 (Refactored to return both logits and embeddings)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from source.utils.models import BaseGNN


class GCN(BaseGNN):
    """
    A standard implementation of the Graph Convolutional Network (GCN) model.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the GCN model layers.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            num_layers (int): The number of GCN layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        current_dim = in_channels
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(current_dim, hidden_channels))
            current_dim = hidden_channels
        # Output layer
        self.convs.append(GCNConv(current_dim, out_channels))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GCN model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings (which are the same as the logits in this architecture).
        """
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # Process all layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            # Apply activation and dropout to all but the last layer
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the final layer serves as both logits and embeddings
        self.embedding_output = x
        logits = x

        return logits, self.embedding_output