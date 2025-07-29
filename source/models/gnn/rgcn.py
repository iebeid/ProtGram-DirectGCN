# ==============================================================================
# MODULE: models/gnn/rgcn.py
# PURPOSE: A standard implementation of the Relational Graph Convolutional
#          Network (RGCN).
# VERSION: 3.1 (Corrected forward pass implementation)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

from source.utils.models import BaseGNN


class RGCN(BaseGNN):
    """
    A standard Relational Graph Convolutional Network (RGCN) model.
    This architecture is designed for node classification on graphs with multiple
    edge types (relations). It overrides the forward pass from BaseGNN to handle
    the `edge_type` attribute required by RGCNConv.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the RGCN model layers by deferring to the BaseGNN.

        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden layers.
            out_channels (int): Dimensionality of output (number of classes).
            num_relations (int): The number of unique edge types in the graph.
            num_layers (int): The number of RGCN layers. Defaults to 2.
            dropout_rate (float): The dropout rate to apply between layers. Defaults to 0.5.
        """
        # Use the powerful BaseGNN __init__ to create the self.convs list.
        super().__init__(
            conv_layer_class=RGCNConv,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            # Pass RGCN-specific arguments to the constructor
            num_relations=num_relations
        )
        # We still need dropout_rate for the custom forward pass.
        self.dropout_rate = dropout_rate

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the RGCN model. This is overridden from BaseGNN
        to handle the `edge_type` tensor required by RGCNConv.

        Returns:
            A tuple containing:
            - The final logits for classification.
            - The node embeddings from the last hidden layer.
        """
        x, edge_index = data.x, data.edge_index

        # For standard benchmark datasets, there's often only one relation type (0).
        # We create the edge_type tensor if it doesn't exist.
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        # Handle the single-layer case
        if len(self.convs) == 1:
            logits = self.convs0
            # For a single-layer model, the logits are also the embeddings
            self.embedding_output = logits
            return logits, self.embedding_output

        # Process all but the final layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # The output of the last hidden layer is the embedding
        self.embedding_output = x

        # Apply the final layer to get logits
        logits = self.convs-1

        return logits, self.embedding_output