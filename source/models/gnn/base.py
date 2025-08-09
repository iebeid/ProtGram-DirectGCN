import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple, Optional


class GNN(nn.Module):
    """
    A generic base class for GNN models like GCN and GraphSAGE.
    It handles the layer creation and forward pass logic to reduce code duplication.
    """

    def __init__(self,
                 conv_layer_class: type, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout_rate: float = 0.5, **conv_kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.embedding_output = None
        self.conv_kwargs = conv_kwargs

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_layers == 1:
            self.convs.append(conv_layer_class(in_channels, out_channels, **self.conv_kwargs))
        else:
            self.convs.append(conv_layer_class(in_channels, hidden_channels, **self.conv_kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(conv_layer_class(hidden_channels, hidden_channels, **self.conv_kwargs))
            self.convs.append(conv_layer_class(hidden_channels, out_channels, **self.conv_kwargs))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        # --- FIX: Correctly get edge weights from data.edge_attr ---
        # This was the root cause of the poor performance for standard GNNs.
        edge_weight = getattr(data, 'edge_attr', None)

        # Handle the single-layer case where logits are the embeddings
        if len(self.convs) == 1:
            # --- FIX: Use explicit check for edge_weight for clarity and robustness ---
            if edge_weight is not None:
                logits = self.convs[0](x, edge_index, edge_weight=edge_weight)
            else:
                logits = self.convs[0](x, edge_index)
            self.embedding_output = logits
            return logits, self.embedding_output

        # Process all but the final layer
        for conv in self.convs[:-1]:
            if edge_weight is not None:
                # Pass edge_weight to the convolution
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        if edge_weight is not None:
            # Pass edge_weight to the final convolution
            logits = self.convs[-1](self.embedding_output, edge_index, edge_weight=edge_weight)
        else:
            logits = self.convs[-1](self.embedding_output, edge_index)

        return logits, self.embedding_output

    def get_embeddings(self, data: Data) -> Optional[torch.Tensor]:
        if self.embedding_output is None:
            print(f"Warning: embedding_output is None for {self.__class__.__name__}. Call forward pass first.")
        return self.embedding_output