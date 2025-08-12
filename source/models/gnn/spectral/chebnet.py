# ==============================================================================
# MODULE: models/gnn/chebnet.py
# PURPOSE: A standard implementation of the Chebyshev Spectral CNN (ChebNet).
# VERSION: 4.0 (Refactored to be a standalone nn.Module for architectural correctness)
# AUTHOR: Islam Ebeid
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv


class ChebNet(nn.Module):
    """
    A standard implementation of the Chebyshev Spectral CNN (ChebNet) model.
    This architecture uses Chebyshev polynomials to define convolutions in the
    Fourier domain.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 K: int = 3, num_layers: int = 2, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_layers == 1:
            self.convs.append(ChebConv(in_channels, out_channels, K=K, **kwargs))
        else:
            self.convs.append(ChebConv(in_channels, hidden_channels, K=K, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K, **kwargs))
            self.convs.append(ChebConv(hidden_channels, out_channels, K=K, **kwargs))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        logits = self.convs[-1](self.embedding_output, edge_index, edge_weight)
        return logits, self.embedding_output