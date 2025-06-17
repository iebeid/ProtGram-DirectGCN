# ==============================================================================
# MODULE: models/gnn_zoo.py
# PURPOSE: Contains PyTorch Geometric implementations of standard GNN
#          architectures for benchmarking purposes.
# VERSION: 2.2 (Minor adjustments for consistency)
# AUTHOR: Islam Ebeid
# ==============================================================================
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, SignedConv, RGCNConv, GINConv
from torch_geometric.data import Data


class BaseGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_output = None

    def get_embeddings(self, data: Data) -> Optional[torch.Tensor]:
        """
        A standardized way to get embeddings after a forward pass.
        Assumes the forward pass stores embeddings in self.embedding_output.
        """
        # Ensure forward pass has occurred and stored embeddings
        if self.embedding_output is None:
            print(f"Warning: embedding_output is None for {self.__class__.__name__}. Call forward pass first.")
            # Optionally, could run a forward pass here if data is available and it's safe
            # self.forward(data) # This might have side effects or require specific mode (eval)
        return self.embedding_output


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


class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_layers=2, dropout_rate=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate # GATConv has its own dropout
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout_rate))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        # GATConv can take edge_attr for weighted attention, if edge_dim matches.
        # For simplicity, if edge_weight is 1D, it might be used. If multi-dim, GATConv might error or ignore.
        # PyG GATConv v2 handles edge_attr more explicitly.
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_weight if conv.edge_dim is not None else None)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                # Dropout is typically part of GATConv itself
        self.embedding_output = x
        return x


class GraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(current_dim, hidden_channels))
            current_dim = hidden_channels
        self.convs.append(SAGEConv(current_dim, out_channels))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index # SAGEConv doesn't typically use edge_weight in its basic form
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x


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


class ChebNet(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers - 1):
            self.convs.append(ChebConv(current_dim, hidden_channels, K=K))
            current_dim = hidden_channels
        self.convs.append(ChebConv(current_dim, out_channels, K=K))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight) # ChebConv can use edge_weight
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x


# In src/models/gnn_zoo.py
class SignedNet(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch norms can help
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers):
            is_first_layer = (i == 0)
            # The output dimension of this specific SignedConv layer
            layer_out_channels = hidden_channels if i < num_layers - 1 else out_channels

            self.convs.append(SignedConv(current_dim, layer_out_channels, first_aggr=is_first_layer))

            # Determine the input dimension for the *next* layer
            if is_first_layer:
                current_dim = layer_out_channels * 2  # Output of first_aggr=True is 2 * its out_channels
            else:
                current_dim = layer_out_channels  # Output of first_aggr=False is its out_channels

            if i < num_layers - 1:  # Add batch norm for intermediate layers
                self.bns.append(nn.BatchNorm1d(current_dim))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # For unsigned graphs, create an empty neg_edge_index
        neg_edge_index = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, pos_edge_index=edge_index, neg_edge_index=neg_edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)  # Apply batch norm
                x = F.relu(x)  # Or F.tanh(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x


class GIN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        if num_layers <= 0: raise ValueError("num_layers must be positive")

        current_dim = in_channels
        for i in range(num_layers):
            final_out_dim_gin = hidden_channels if i < num_layers - 1 else out_channels
            mlp = Sequential(
                Linear(current_dim, hidden_channels),
                ReLU(),
                Linear(hidden_channels, final_out_dim_gin)
            )
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            current_dim = final_out_dim_gin

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index # GINConv doesn't typically use edge_weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        self.embedding_output = x
        return x


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