# src/models/protgram_directgcn.py
# ==============================================================================
# MODULE: models/protgram_directgcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 8.0 (Implemented undirected structural path and learnable constant)
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
    The custom directed GCN layer, now with an additional path for
    undirected structural information and a learnable node-specific constant.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.use_vector_coeffs = use_vector_coeffs

        # --- Components for Directed Paths (Incoming and Outgoing) ---
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))

        # --- Components for Shared Directed Paths ---
        self.lin_directed_shared = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_directed_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_directed_shared_out = nn.Parameter(torch.Tensor(out_channels))

        # --- Components for Undirected Paths ---
        self.lin_undirected = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_undirected = nn.Parameter(torch.Tensor(out_channels))  # Bias for the undirected path

        # --- Learnable Coefficients (c_in, c_out, c_all) ---
        if self.use_vector_coeffs and self.num_nodes > 0:
            self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_undirected_vec = nn.Parameter(torch.Tensor(num_nodes, 1))  # NEW
        else:
            self.use_vector_coeffs = False  # Fallback to scalar if num_nodes is invalid
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
            self.C_undirected = nn.Parameter(torch.Tensor(1))  # NEW

        # --- Learnable Node-Specific Constant ---
        if self.num_nodes > 0:
            self.constant = nn.Parameter(torch.Tensor(num_nodes, out_channels))  # NEW
        else:
            self.constant = None  # No constant if num_nodes is unknown

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_directed_shared, self.lin_undirected]:
            nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_directed_shared_in, self.bias_directed_shared_out, self.bias_undirected]:
            nn.init.zeros_(bias)

        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
            nn.init.ones_(self.C_undirected_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)
            nn.init.ones_(self.C_undirected)

        if self.constant is not None:
            nn.init.xavier_uniform_(self.constant)

    def forward(self, x: torch.Tensor,
                edge_index_in: torch.Tensor, edge_weight_in: Optional[torch.Tensor],
                edge_index_out: torch.Tensor, edge_weight_out: Optional[torch.Tensor],
                edge_index_undirected: torch.Tensor, edge_weight_undirected: Optional[torch.Tensor],  # NEW
                original_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass implementing the new combined layer logic."""

        # --- 1. Directed Incoming Path ---
        h_main_in_transformed = self.lin_main_in(x)
        h_main_in_propagated = self.propagate(edge_index_in, x=h_main_in_transformed, edge_weight=edge_weight_in)
        h_shared_for_in = self.lin_shared(x)
        h_shared_in_propagated = self.propagate(edge_index_in, x=h_shared_for_in, edge_weight=edge_weight_in)
        ic_combined = (h_main_in_propagated + self.bias_main_in) + (h_shared_in_propagated + self.bias_shared_in)

        # --- 2. Directed Outgoing Path ---
        h_main_out_transformed = self.lin_main_out(x)
        h_main_out_propagated = self.propagate(edge_index_out, x=h_main_out_transformed, edge_weight=edge_weight_out)
        h_shared_for_out = self.lin_shared(x)
        h_shared_out_propagated = self.propagate(edge_index_out, x=h_shared_for_out, edge_weight=edge_weight_out)
        oc_combined = (h_main_out_propagated + self.bias_main_out) + (h_shared_out_propagated + self.bias_shared_out)

        # --- 3. Undirected Structural Path (NEW) ---
        # We reuse h_shared_for_out as per the request's spirit of a shared structural understanding
        h_undirected_transformed = self.lin_undirected(x)
        h_undirected_propagated = self.propagate(edge_index_undirected, x=h_undirected_transformed, edge_weight=edge_weight_undirected)
        all_c_combined = h_undirected_propagated + self.bias_undirected

        # --- 4. Get Coefficients and Constant ---
        if self.use_vector_coeffs and original_indices is not None:
            # Subgraph mode: select the right coefficients and constants
            c_in = self.C_in_vec[original_indices]
            c_out = self.C_out_vec[original_indices]
            c_all = self.C_all_vec[original_indices]
            constant_term = self.constant[original_indices] if self.constant is not None else 0
        elif self.use_vector_coeffs:
            # Full graph mode
            c_in = self.C_in_vec
            c_out = self.C_out_vec
            c_all = self.C_all_vec
            constant_term = self.constant if self.constant is not None else 0
        else:
            # Scalar mode
            c_in = self.C_in
            c_out = self.C_out
            c_all = self.C_all
            constant_term = 0  # Constant is only per-node, so not applicable in scalar mode

        # --- 5. Final Combination ---
        final_combination = (c_all * all_c_combined) + (c_in * ic_combined) + (c_out * oc_combined) + constant_term

        return final_combination

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class ProtGramDirectGCN(nn.Module):
    """The main GCN architecture, adapted for the new layer."""

    def __init__(self, layer_dims: List[int], num_graph_nodes: Optional[int],
                 task_num_output_classes: int, n_gram_len: int,
                 one_gram_dim: int, max_pe_len: int, dropout: float,
                 use_vector_coeffs: bool, l2_eps: float = 1e-12):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.one_gram_dim = one_gram_dim
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
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            current_num_nodes = num_graph_nodes if num_graph_nodes is not None else 0
            effective_use_vector_coeffs = use_vector_coeffs and current_num_nodes > 0
            self.convs.append(DirectGCNLayer(in_dim, out_dim, current_num_nodes, effective_use_vector_coeffs))
            self.res_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())

        final_embedding_dim = layer_dims[-1]
        decoder_hidden_dim = final_embedding_dim // 2 if final_embedding_dim > 1 else 1
        self.decoder_fc = nn.Sequential(
            nn.Linear(final_embedding_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(decoder_hidden_dim, task_num_output_classes)
        )

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        # This method remains unchanged
        if self.pe_layer is None: return x
        if self.n_gram_len > 0 and self.one_gram_dim > 0 and x.shape[1] == self.n_gram_len * self.one_gram_dim:
            x_with_pe = x.clone()
            x_reshaped = x_with_pe.view(-1, self.n_gram_len, self.one_gram_dim)
            pos_to_enc = min(self.n_gram_len, self.pe_layer.num_embeddings)
            if pos_to_enc > 0:
                pos_indices = torch.arange(0, pos_to_enc, device=x.device, dtype=torch.long)
                pe_values = self.pe_layer(pos_indices)
                x_reshaped[:, :pos_to_enc, :] += pe_values.unsqueeze(0)
            return x_reshaped.view(-1, self.n_gram_len * self.one_gram_dim)
        return x

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Safely get all attributes, defaulting to None if they don't exist
        x = data.x
        ei_in = getattr(data, 'edge_index_in', None)
        ew_in = getattr(data, 'edge_weight_in', None)
        ei_out = getattr(data, 'edge_index_out', None)
        ew_out = getattr(data, 'edge_weight_out', None)
        ei_undir = getattr(data, 'edge_index_undirected_norm', None)  # NEW
        ew_undir = getattr(data, 'edge_weight_undirected_norm', None)  # NEW
        original_indices = getattr(data, 'original_indices', None)

        # Ensure required attributes are present
        if ei_in is None or ei_out is None or ei_undir is None:
            raise ValueError("ProtGramDirectGCN requires 'edge_index_in', 'edge_index_out', and 'edge_index_undirected_norm' in the Data object.")

        h = self._apply_pe(x)

        for i in range(len(self.convs)):
            h_res = h
            gcn_layer, res_layer = self.convs[i], self.res_projs[i]
            # Pass all edge information to the layer
            gcn_output = gcn_layer(h_res, ei_in, ew_in, ei_out, ew_out, ei_undir, ew_undir, original_indices)
            residual_output = res_layer(h_res)
            h = F.leaky_relu(gcn_output + residual_output)
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_embed_for_task = h
        task_logits = self.decoder_fc(final_embed_for_task)
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(final_embed_for_task, eps=self.l2_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embeddings
