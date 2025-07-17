import torch
import torch.nn as nn
from torch_geometric.data import Data

from source.models.gnn.gcn import GCN
from source.utils.models import BaseGNN


class TongDiGCN(BaseGNN):
    """
    Implements the DiGCN model variant from "Harnessing the Power of Choices:
    A Survey on Selection Bias in Graph-based Recommender Systems" by Tong et al.
    This model runs two separate GCNs on the forward and backward adjacency matrices
    and concatenates their outputs.
    """

    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int):
        super().__init__()
        self.gcn_forward = GCN(in_channels, hidden_dim, hidden_dim)  # Output of GCN is hidden_dim
        self.gcn_backward = GCN(in_channels, hidden_dim, hidden_dim)
        # The final linear layer maps the concatenated embeddings to the output channels
        self.final_linear = nn.Linear(hidden_dim * 2, out_channels)
        self.out_channels = out_channels

    def forward(self, data: Data) -> torch.Tensor:
        # Create a new Data object for the forward pass
        data_forward = Data(x=data.x, edge_index=data.edge_index)
        x_forward = self.gcn_forward(data_forward)

        # Create a new Data object for the backward pass
        if not hasattr(data, 'edge_index_backward'):
            raise ValueError("TongDiGCN requires 'edge_index_backward' in the Data object.")
        data_backward = Data(x=data.x, edge_index=data.edge_index_backward)
        x_backward = self.gcn_backward(data_backward)

        # Concatenate the outputs from both GCNs
        x_combined = torch.cat([x_forward, x_backward], dim=1)

        # Apply the final linear layer
        x_out = self.final_linear(x_combined)
        self.embedding_output = x_combined  # Store the concatenated embeddings before the final projection
        return x_out
