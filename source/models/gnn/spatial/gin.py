# ==============================================================================
# MODULE: models/gnn/gin.py
# PURPOSE: A standard implementation of the Graph Isomorphism Network (GIN).
# VERSION: 3.3 (Corrected forward pass indexing)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv


class GIN(nn.Module):
    """
    A standard implementation of the Graph Isomorphism Network (GIN) model.
    This architecture uses a multi-layer perceptron to update node features,
    providing high expressive power.

    Note: This model is standalone and does not inherit from the BaseGNN class.
    This is necessary because the GINConv layer requires a full `nn.Sequential`
    module as its primary argument, which differs from other standard PyG layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the GIN model layers.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            num_layers (int): The number of GIN layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        super().__init__()
        self.embedding_output = None
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_layers == 1:
            # A single layer goes directly from input to output
            mlp = nn.Sequential(nn.Linear(in_channels, out_channels))
            self.convs.append(GINConv(mlp))
        else:
            # Input layer
            mlp_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp_in))

            # Hidden layers
            for _ in range(num_layers - 2):
                mlp_hidden = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(mlp_hidden))

            # Output layer
            mlp_out = nn.Sequential(nn.Linear(hidden_channels, out_channels))
            self.convs.append(GINConv(mlp_out))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GIN model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index = data.x, data.edge_index

        # Handle the single-layer case
        if len(self.convs) == 1:
            logits = self.convs0
            # For a single-layer model, the logits are also the embeddings
            self.embedding_output = logits
            return logits, self.embedding_output.clone()

        # Process all but the final layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)  # Activation is applied *after* the GINConv
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        logits = self.convs-1

        return logits, self.embedding_output.clone()