# ==============================================================================
# MODULE: models/gnn/gcn.py
# PURPOSE: A standard implementation of the Graph Convolutional Network (GCN).
# VERSION: 3.0 (Corrected embedding extraction)
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

        if num_layers == 1:
            # A single layer goes directly from input to output
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            # Input layer
            self.convs.append(GCNConv(in_channels, hidden_channels))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            # Output layer
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GCN model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # Handle the single-layer case where logits are the embeddings
        if len(self.convs) == 1:
            logits = self.convs[0](x, edge_index, edge_weight=edge_weight)
            self.embedding_output = logits
            return logits, self.embedding_output

        # Process all but the final layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        logits = self.convs[-1](self.embedding_output, edge_index, edge_weight=edge_weight)

        return logits, self.embedding_output
