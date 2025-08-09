# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: A standard implementation of the Relational Graph Convolutional
#          Network (RGCN).
# VERSION: 6.0 (Corrected to be a standalone nn.Module and handle edge_type)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    """
    A standard Relational Graph Convolutional Network (RGCN) model.
    This architecture is designed for node classification on graphs with multiple
    edge types (relations).
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int = 2, dropout_rate: float = 0.5, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None

        if num_layers == 1:
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations, **kwargs))
        else:
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, **kwargs))
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, **kwargs))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom forward pass for RGCN that correctly handles the `edge_type` attribute.
        """
        x, edge_index = data.x, data.edge_index
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        # Handle single-layer case
        if len(self.convs) == 1:
            # --- FIX: Corrected a typo that would crash single-layer models. ---
            # The original code had `self.convs0` which is invalid syntax.
            logits = self.convs[0](x, edge_index, edge_type)
            self.embedding_output = logits
            return logits, self.embedding_output

        # Multi-layer case
        # --- FIX: Correctly separate embedding generation from final logit calculation ---
        # Process all but the final layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x
        # Apply the final layer to get logits
        logits = self.convs[-1](x, edge_index, edge_type)

        return logits, self.embedding_output