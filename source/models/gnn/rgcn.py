# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: A standard implementation of the Relational Graph Convolutional
#          Network (RGCN) for use in benchmarking.
# VERSION: 1.0
# AUTHOR: Your Name (Assembled by Coding Partner)
# ==============================================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
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
        self.convs = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate

        # Input layer
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        # Output layer
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x