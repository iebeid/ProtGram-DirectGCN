# ==============================================================================
# MODULE: models/gnn_zoo.py
# PURPOSE: Contains PyTorch Geometric implementations of standard GNN
#          architectures for benchmarks purposes.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, RGCNConv
from typing import Dict, Any, Type

class BaseGNN(nn.Module):
    """A base class for all GNNs to ensure they have an embedding_output attribute."""
    def __init__(self):
        super().__init__()
        self.embedding_output = None

class GCN(BaseGNN):
    """A standard Graph Convolutional Network (GCN)."""
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = F.dropout(x, p=self.dropout, training=self.training)
        self.embedding_output = self.conv1(x, edge_index, edge_weight)
        x = F.relu(self.embedding_output)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), self.embedding_output

class GAT(BaseGNN):
    """A Graph Attention Network (GAT)."""
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 8, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, **kwargs):
        x = F.dropout(x, p=self.dropout, training=self.training)
        self.embedding_output = self.conv1(x, edge_index)
        x = F.elu(self.embedding_output)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), self.embedding_output

class GraphSAGE(BaseGNN):
    """A GraphSAGE Network."""
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, **kwargs):
        self.embedding_output = self.conv1(x, edge_index)
        x = F.relu(self.embedding_output)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), self.embedding_output

class GIN(BaseGNN):
    """A Graph Isomorphism Network (GIN)."""
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        mlp1 = nn.Sequential(nn.Linear(num_features, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(mlp1, train_eps=True)
        mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, num_classes))
        self.conv2 = GINConv(mlp2, train_eps=True)

    def forward(self, x, edge_index, **kwargs):
        self.embedding_output = self.conv1(x, edge_index)
        x = F.relu(self.embedding_output)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), self.embedding_output

class RGCN(BaseGNN):
    """A Relational Graph Convolutional Network (RGCN)."""
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int = 64, num_relations: int = 1, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = RGCNConv(num_features, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, num_classes, num_relations)

    def forward(self, x, edge_index, edge_type=None, **kwargs):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        self.embedding_output = self.conv1(x, edge_index, edge_type)
        x = F.relu(self.embedding_output)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1), self.embedding_output


# --- Model Factory ---
def get_gnn_model_from_zoo(model_name: str, num_features: int, num_classes: int) -> nn.Module:
    """
    A factory function to create a GNN model instance from the zoo.
    """
    MODEL_ZOO: Dict[str, Type[BaseGNN]] = {
        "GCN": GCN,
        "GAT": GAT,
        "GraphSAGE": GraphSAGE,
        "GIN": GIN,
        "RGCN": RGCN
    }

    if model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model in GNN Zoo: {model_name}")

    model_class = MODEL_ZOO[model_name]
    return model_class(num_features=num_features, num_classes=num_classes)