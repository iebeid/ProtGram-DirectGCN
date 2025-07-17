import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from source.utils.models import BaseGNN

class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(current_dim, hidden_channels))
            current_dim = hidden_channels
        self.convs.append(GCNConv(current_dim, out_channels))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x