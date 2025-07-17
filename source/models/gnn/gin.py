import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from source.utils.models import BaseGNN

class GIN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers):
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            nn_module = nn.Sequential(
                nn.Linear(current_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.convs.append(GINConv(nn_module))
            current_dim = out_dim

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x