# ==============================================================================
# MODULE: models/gnn/chebnet.py
# PURPOSE: A standard implementation of the Chebyshev Spectral CNN (ChebNet).
# VERSION: 2.0 (Refactored to return both logits and embeddings)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv

from source.utils.models import BaseGNN


class ChebNet(BaseGNN):
    """
    A standard implementation of the Chebyshev Spectral CNN (ChebNet) model.
    This architecture uses Chebyshev polynomials to define convolutions in the
    Fourier domain.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 K: int = 3, num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the ChebNet model layers.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            K (int): The filter size (number of hops). Defaults to 3.
            num_layers (int): The number of ChebNet layers. Defaults to 2.
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
            self.convs.append(ChebConv(current_dim, hidden_channels, K=K))
            current_dim = hidden_channels
        # Output layer
        self.convs.append(ChebConv(current_dim, out_channels, K=K))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the ChebNet model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # Process hidden layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        logits = self.convs[-1](self.embedding_output, edge_index, edge_weight=edge_weight)

        return logits, self.embedding_output