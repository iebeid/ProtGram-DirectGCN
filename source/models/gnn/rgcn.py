# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: A standard implementation of the Relational Graph Convolutional
#          Network (RGCN) for use in benchmarking.
# VERSION: 1.1 (Corrected forward pass for benchmarker compatibility)
# AUTHOR: Your Name (Assembled by Coding Partner)
# ==============================================================================

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
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

    def forward(self, data: Data) -> torch.Tensor:
        """
        The forward pass for the RGCN model, adapted for standard PyG Data objects.
        """
        x, edge_index = data.x, data.edge_index

        # For standard benchmark datasets, there's only one relation type (0).
        # We create the edge_type tensor if it doesn't exist.
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type=edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x