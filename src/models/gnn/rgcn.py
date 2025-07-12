import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from src.utils.models_utils import BaseGNN


class RGCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=1, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.num_relations = num_relations
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers - 1):
            self.convs.append(RGCNConv(current_dim, hidden_channels, num_relations=num_relations))
            current_dim = hidden_channels
        self.convs.append(RGCNConv(current_dim, out_channels, num_relations=num_relations))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            if self.num_relations > 1:
                print(f"Warning: RGCN using default edge_type (all zeros) but num_relations is {self.num_relations}")

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type=edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x