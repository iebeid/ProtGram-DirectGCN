# src/models/protgram_directgcn.py
# ==============================================================================
# MODULE: models/protgram_directgcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 7.5 (Ensured _apply_pe is out-of-place to fix autograd error)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.utils.models_utils import EmbeddingProcessor


class DirectGCNLayer(MessagePassing):
    """
    The custom directed GCN layer.
    Shared component is now a direct addition, not propagated.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')  # 'add' aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes  # Needed if C_in_vec/C_out_vec are used
        self.use_vector_coeffs = use_vector_coeffs

        # --- Main Path Weights ---
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))

        # --- Shared Path (Skip/W_all) Weights ---
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)  # W_S
        self.bias_shared_in = nn.Parameter(torch.Tensor(out_channels))  # b_S_I
        self.bias_shared_out = nn.Parameter(torch.Tensor(out_channels))  # b_S_O

        # --- Adaptive Coefficients ---
        if self.use_vector_coeffs:
            if self.num_nodes <= 0:  # Fallback if num_nodes is not valid for vector coeffs
                # print(f"Warning: num_nodes is {self.num_nodes} for DirectGCNLayer, but use_vector_coeffs is True. Defaulting to scalar C_in/C_out.")
                self.use_vector_coeffs = False  # Override
                self.C_in = nn.Parameter(torch.Tensor(1))
                self.C_out = nn.Parameter(torch.Tensor(1))
            else:
                self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
                self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
        else:
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all learnable parameters."""
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_shared]:
            nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_shared_in, self.bias_shared_out]:
            nn.init.zeros_(bias)

        if self.use_vector_coeffs:  # This check is now safe due to __init__ fallback
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)

    def forward(self, x: torch.Tensor,
                edge_index_in: torch.Tensor, edge_weight_in: Optional[torch.Tensor],
                edge_index_out: torch.Tensor, edge_weight_out: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass implementing the ProtGram-DirectGCN layer logic."""

        # Incoming Message Aggregation
        # Main incoming path (propagated)
        h_main_in_transformed = self.lin_main_in(x)
        h_main_in_propagated = self.propagate(edge_index_in, x=h_main_in_transformed, edge_weight=edge_weight_in)

        # Shared component for incoming (NOT propagated, direct linear transformation)
        h_shared_for_in = self.lin_shared(x)

        ic_combined = (h_main_in_propagated + self.bias_main_in) + (h_shared_for_in + self.bias_shared_in)

        # Outgoing Message Aggregation
        # Main outgoing path (propagated)
        h_main_out_transformed = self.lin_main_out(x)
        h_main_out_propagated = self.propagate(edge_index_out, x=h_main_out_transformed, edge_weight=edge_weight_out)

        # Shared component for outgoing (NOT propagated, direct linear transformation)
        h_shared_for_out = self.lin_shared(x)  # Re-use lin_shared(x)

        oc_combined = (h_main_out_propagated + self.bias_main_out) + (h_shared_for_out + self.bias_shared_out)

        # Adaptive Coefficients
        c_in = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out = self.C_out_vec if self.use_vector_coeffs else self.C_out

        return c_in * ic_combined + c_out * oc_combined

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """The message function for PyG's message passing."""
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class ProtGramDirectGCN(nn.Module):
    """
    The main GCN architecture, with a DYNAMIC number of GCN and residual
    layers.
    """

    def __init__(self, layer_dims: List[int], num_graph_nodes: Optional[int],  # num_graph_nodes can be None for PPI
                 task_num_output_classes: int, n_gram_len: int,
                 one_gram_dim: int, max_pe_len: int, dropout: float,
                 use_vector_coeffs: bool, l2_eps: float = 1e-12):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.one_gram_dim = one_gram_dim  # Used by _apply_pe
        self.dropout = dropout
        self.l2_eps = l2_eps

        self.pe_layer = None
        if one_gram_dim > 0 and max_pe_len > 0:
            self.pe_layer = nn.Embedding(max_pe_len, one_gram_dim)

        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        if not layer_dims or len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least input and output dimensions (length >= 2).")

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            current_num_nodes_for_layer = num_graph_nodes if num_graph_nodes is not None else 0

            effective_use_vector_coeffs = use_vector_coeffs
            if use_vector_coeffs and current_num_nodes_for_layer <= 0:
                effective_use_vector_coeffs = False

            self.convs.append(DirectGCNLayer(in_dim, out_dim, current_num_nodes_for_layer, effective_use_vector_coeffs))
            self.res_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())

        final_embedding_dim = layer_dims[-1]
        self.decoder_fc = nn.Linear(final_embedding_dim, task_num_output_classes)

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Applies positional embeddings if pe_layer is configured. Ensures out-of-place modification."""
        if self.pe_layer is None:
            return x  # Return original x if no PE

        if self.n_gram_len > 0 and self.one_gram_dim > 0:
            expected_dim = self.n_gram_len * self.one_gram_dim
            if x.shape[1] == expected_dim:
                # Clone x to ensure any modifications are on a new tensor,
                # leaving the original input x untouched.
                x_with_pe = x.clone()

                # Reshape the clone for PE application
                x_reshaped = x_with_pe.view(-1, self.n_gram_len, self.one_gram_dim)

                pos_to_enc = min(self.n_gram_len, self.pe_layer.num_embeddings)
                if pos_to_enc > 0:
                    pos_indices = torch.arange(0, pos_to_enc, device=x.device, dtype=torch.long)
                    pe_values = self.pe_layer(pos_indices)  # Shape: (pos_to_enc, one_gram_dim)

                    # Perform the addition on the slice of the reshaped clone.
                    # This modifies x_reshaped (and thus x_with_pe) but not the original x.
                    x_reshaped[:, :pos_to_enc, :] = x_reshaped[:, :pos_to_enc, :] + pe_values.unsqueeze(0)

                # Return the view of the modified clone
                return x_reshaped.view(-1, expected_dim)
                # else:
            # If dimensions don't match, return original x (no PE applied)
            return x

            # If PE conditions (n_gram_len, one_gram_dim) not met, return original x
        return x

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        ei_in, ew_in = data.edge_index_in, data.edge_weight_in
        ei_out, ew_out = data.edge_index_out, data.edge_weight_out

        h = self._apply_pe(x)

        for i in range(len(self.convs)):
            h_res = h

            gcn_layer = self.convs[i]
            res_layer = self.res_projs[i]

            gcn_output = gcn_layer(h_res, ei_in, ew_in, ei_out, ew_out)
            residual_output = res_layer(h_res)

            h = gcn_output + residual_output
            h = F.tanh(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_embed_for_task = h
        task_logits = self.decoder_fc(final_embed_for_task)
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(final_embed_for_task, eps=self.l2_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embeddings
