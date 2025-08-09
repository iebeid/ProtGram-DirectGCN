# ==============================================================================
# MODULE: models/gnn/spectral/tongidigcn.py
# PURPOSE: Implements the DiGCN variant from Tong et al.
# VERSION: 3.2 (Corrected inheritance to be standalone from nn.Module)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from source.models.gnn.spectral.gcn import GCN


class TongDiGCN(nn.Module):
    """
    Implements the DiGCN model variant from "Harnessing the Power of Choices:
    A Survey on Selection Bias in Graph-based Recommender Systems" by Tong et al.
    This model runs two separate GCNs on the forward and backward adjacency matrices
    and concatenates their outputs.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the TongDiGCN model.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers for the internal GCNs.
            out_channels (int): Dimensionality of the final output (number of classes).
            num_layers (int): The number of layers for each internal GCN. Defaults to 2.
            dropout_rate (float): The dropout rate for the internal GCNs. Defaults to 0.5.
        """
        # This call is now correct because it initializes the base torch.nn.Module
        super().__init__()
        self.embedding_output = None
        # The internal GCNs produce embeddings of size hidden_channels.
        self.gcn_forward = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout_rate)
        self.gcn_backward = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout_rate)
        # The final linear layer maps the concatenated embeddings to the output channels.
        self.final_linear = nn.Linear(hidden_channels * 2, out_channels)
        self.out_channels = out_channels

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the TongDiGCN model.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The concatenated node embeddings from the forward and backward GCNs.
        """
        # Create a new Data object for the forward pass
        data_forward = Data(x=data.x, edge_index=data.edge_index, edge_attr=getattr(data, 'edge_attr', None))
        # The GCN model returns (logits, embeddings). We use the embeddings.
        _, x_forward = self.gcn_forward(data_forward)

        # Create a new Data object for the backward pass
        if not hasattr(data, 'edge_index_backward'):
            raise ValueError("TongDiGCN requires 'edge_index_backward' in the Data object.")
        data_backward = Data(x=data.x, edge_index=data.edge_index_backward, edge_attr=getattr(data, 'edge_attr', None))
        _, x_backward = self.gcn_backward(data_backward)

        # Concatenate the outputs from both GCNs to form the final embeddings
        x_combined = torch.cat([x_forward, x_backward], dim=1)
        self.embedding_output = x_combined

        # Apply the final linear layer to get the classification logits
        logits = self.final_linear(self.embedding_output)

        return logits, self.embedding_output