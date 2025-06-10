# ==============================================================================
# MODULE: models/gnn_zoo.py
# PURPOSE: Contains PyTorch Geometric implementations of standard GNN
#          architectures for benchmarking purposes.
# VERSION: 2.0 (Adds MoNet)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, WLConv, ChebConv, SignedConv, RGCNConv, GINConv
from torch.nn import Sequential, Linear, ReLU  # Needed for GINConv's MLP



class BaseGNN(nn.Module):
    """A base class for all GNNs to ensure they have an embedding_output attribute."""

    def __init__(self):
        super().__init__()
        self.embedding_output = None


# --- Basic PyG Model Implementations for the Zoo ---
class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


class GraphSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x


# --- New Model Implementations ---
class WLGCN(BaseGNN):
    """WLConv as a feature transformation layer followed by GCN layers."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.wl = WLConv()  # Parameter-free Weisfeiler-Lehman feature updater
        self.conv1 = GCNConv(in_channels, hidden_channels)  # Assumes WLConv preserves feature dim
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.wl(x, edge_index)  # Update features using WL algorithm
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class ChebNet(BaseGNN):
    """GNN with Chebyshev Spectral Graph Convolutions."""

    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SignedNet(BaseGNN):
    """GNN with SignedConv layers, adapted for unsigned graphs."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Using first_aggr=False for a more general SignedConv behavior
        self.sconv1 = SignedConv(in_channels, hidden_channels, first_aggr=False)
        self.sconv2 = SignedConv(hidden_channels, out_channels, first_aggr=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Provide an empty tensor for negative edges as PPI is unsigned
        neg_edge_index = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)

        x = self.sconv1(x, pos_edge_index=edge_index, neg_edge_index=neg_edge_index)
        x = F.relu(x)
        x = self.sconv2(x, pos_edge_index=edge_index, neg_edge_index=neg_edge_index)
        return x


class GIN(BaseGNN):
    """Graph Isomorphism Network (GIN) model."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # MLP for the first GIN layer
        mlp1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.gin1 = GINConv(nn=mlp1, train_eps=True)  # train_eps makes epsilon a learnable parameter

        # MLP for the second GIN layer
        mlp2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.gin2 = GINConv(nn=mlp2, train_eps=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin1(x, edge_index)
        x = F.relu(x)  # Apply activation after GINConv
        x = self.gin2(x, edge_index)
        return x


class RGCN(BaseGNN):
    """Relational GCN assuming a single relation type for PPI data."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations=1):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Assume all edges belong to a single relation type (type 0)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        x = self.conv1(x, edge_index, edge_type=edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type=edge_type)
        return x


# --- Model Factory ---
def get_gnn_model_from_zoo(model_name: str, num_features: int, num_classes: int) -> BaseGNN:
    """
    A simple factory to get standard GNN models from PyG.
    This replaces the need for a separate gnn_zoo.py file for this script.
    """
    model_name_upper = model_name.upper()
    if model_name_upper == 'GCN':
        return GCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    elif model_name_upper == 'GAT':
        return GAT(in_channels=num_features, hidden_channels=256, out_channels=num_classes, heads=4)
    elif model_name_upper == 'GRAPHSAGE':
        return GraphSAGE(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    elif model_name_upper == 'WLGCN':
        return WLGCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    elif model_name_upper == 'CHEBNET':
        return ChebNet(in_channels=num_features, hidden_channels=256, out_channels=num_classes, K=3)
    elif model_name_upper == 'SIGNEDNET':
        return SignedNet(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    elif model_name_upper == 'RGCN_SR':  # SR for Single Relation
        return RGCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes, num_relations=1)
    elif model_name_upper == 'GIN':
        return GIN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported in the local GNN Zoo.")
