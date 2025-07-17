import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from source.utils.models import BaseGNN

class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_layers=2, dropout_rate=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
        current_dim = hidden_channels * heads

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(current_dim, hidden_channels, heads=heads, dropout=dropout_rate))
            current_dim = hidden_channels * heads

        # Output layer
        self.convs.append(GATConv(current_dim, out_channels, heads=1, concat=False, dropout=dropout_rate))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        self.embedding_output = x
        return x