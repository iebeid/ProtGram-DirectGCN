# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: Defines a simple RGCN model wrapper for use in the ProtGram pipeline.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

from source.utils.models import EmbeddingProcessor

class ProtGramRGCN(nn.Module):
    """
    An RGCN model for node classification, expecting 'x', 'edge_index',
    and 'edge_type' in its Data object.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, data: Data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # In this setup, the output of the first layer is the embedding
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(x)

        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1), final_normalized_embeddings