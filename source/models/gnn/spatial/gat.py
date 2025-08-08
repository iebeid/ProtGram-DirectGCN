# ==============================================================================
# MODULE: models/gnn/gat.py
# PURPOSE: A standard implementation of the Graph Attention Network (GAT).
# VERSION: 9.3 (Removed incorrect forward override to inherit from base class)
# AUTHOR: Islam Ebeid
# ==============================================================================
from torch_geometric.nn import GATConv

from source.models.gnn.base import GNN


class GAT(GNN):
    """
    A standard implementation of the Graph Attention Network (GAT) model.
    This architecture uses self-attention to weigh the importance of neighboring
    nodes during message passing.
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
            dropout_rate (float): The dropout rate. Defaults to 0.6.
        """
        # Defer to the BaseGNN constructor to build the layers.
        # Pass GAT-specific arguments (heads, concat) via **conv_kwargs.
        super().__init__(
            conv_layer_class=GATConv,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            # GAT-specific kwargs for the constructor
            heads=heads,
            concat=True  # Concatenate heads for all but the last layer
        )
        # The last layer should average heads, so we rebuild it.
        if num_layers > 1:
            self.convs[-1] = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)