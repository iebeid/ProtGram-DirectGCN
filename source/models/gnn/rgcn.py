# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: A standard implementation of the Relational Graph Convolutional
#          Network (RGCN) for use in benchmarking.
# VERSION: 2.0 (Refactored to return both logits and embeddings)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

from source.utils.models import BaseGNN


class RGCN(BaseGNN):
    """
    A standard Relational Graph Convolutional Network (RGCN) model.
    This architecture is designed for node classification on graphs with multiple
    edge types (relations).
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the RGCN model layers.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            num_relations (int): The number of unique edge types in the graph.
            num_layers (int): The number of RGCN layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate

        if num_layers == 1:
            # A single layer goes directly from input to output
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            # Input layer
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            # Output layer
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the RGCN model, adapted for standard PyG Data objects.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index = data.x, data.edge_index

        # For standard benchmark datasets, there's only one relation type (0).
        # We create the edge_type tensor if it doesn't exist.
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        # Handle the single-layer case
        if len(self.convs) == 1:
            logits = self.convs[0](x, edge_index, edge_type)
            # For a single-layer model, the logits are also the embeddings
            self.embedding_output = logits
            return logits, self.embedding_output

        # Process all but the final layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        logits = self.convs[-1](self.embedding_output, edge_index, edge_type)

        return logits, self.embedding_output