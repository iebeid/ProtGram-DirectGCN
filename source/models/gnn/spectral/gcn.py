# ==============================================================================
# MODULE: models/gnn/gcn.py
# PURPOSE: A standard implementation of the Graph Convolutional Network (GCN).
# VERSION: 4.0 (Refactored to use generic BaseGNN)
# AUTHOR: Islam Ebeid
# ==============================================================================
from torch_geometric.nn import GCNConv

from source.models.gnn.base import GNN


class GCN(GNN):
    """
    A standard implementation of the Graph Convolutional Network (GCN) model.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the GCN model layers by deferring to the BaseGNN.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            num_layers (int): The number of GCN layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        super().__init__(
            conv_layer_class=GCNConv,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
