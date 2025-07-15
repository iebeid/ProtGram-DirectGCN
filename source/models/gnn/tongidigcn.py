import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from src.utils.models_utils import BaseGNN
from src.models.gnn.gcn import GCN

class TongDiGCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.gcn_forward = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout_rate)
        self.gcn_backward = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout_rate)
        self.final_linear = nn.Linear(hidden_channels * 2, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # Forward pass
        x_fwd = self.gcn_forward(data)

        # Backward pass (reverse edges)
        edge_index_bwd = edge_index[[1, 0], :]
        # Create a new Data object for the backward pass, copying relevant attributes
        data_bwd = Data(x=x, edge_index=edge_index_bwd)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data_bwd.edge_attr = data.edge_attr # Assuming edge_attr is symmetric or handled by GCN
        # Copy other necessary attributes if your GCN model uses them
        # for attr_name in ['batch', 'ptr', 'num_nodes']: # Example attributes
        #     if hasattr(data, attr_name):
        #         setattr(data_bwd, attr_name, getattr(data, attr_name))

        x_bwd = self.gcn_backward(data_bwd)

        x_combined = torch.cat([x_fwd, x_bwd], dim=-1)
        x_combined = F.dropout(x_combined, p=self.dropout_rate, training=self.training)
        out = self.final_linear(x_combined)
        self.embedding_output = x_combined
        return out