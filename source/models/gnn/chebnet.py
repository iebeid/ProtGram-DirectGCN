# ==============================================================================
# MODULE: models/gnn/chebnet.py
# PURPOSE: A standard implementation of the Chebyshev Spectral CNN (ChebNet).
# VERSION: 3.0 (Refactored to use generic BaseGNN)
# AUTHOR: Islam Ebeid
# ==============================================================================
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
        Initializes the ChebNet model layers by deferring to the BaseGNN.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            K (int): The filter size (number of hops). Defaults to 3.
            num_layers (int): The number of ChebNet layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        super().__init__(
            conv_layer_class=ChebConv,
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
            num_layers=num_layers, dropout_rate=dropout_rate, K=K
        )