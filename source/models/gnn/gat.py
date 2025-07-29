# ==============================================================================
# MODULE: models/gnn/gat.py
# PURPOSE: A standard implementation of the Graph Attention Network (GAT).
# VERSION: 9.0 (Refactored to use generic BaseGNN)
# AUTHOR: Islam Ebeid
# ==============================================================================
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
        Initializes the GAT model layers by deferring to the BaseGNN.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            heads (int): Number of attention heads. Defaults to 8.
            num_layers (int): The number of GAT layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.6.
        """
        # Note: For simplicity in this benchmark, we use a single head.
        # A more complex BaseGNN would be needed for multi-head intermediate layers.
        super().__init__(
            conv_layer_class=GATConv,
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
            num_layers=num_layers, dropout_rate=dropout_rate,
            # Pass GAT-specific arguments
            heads=1, concat=False
        )