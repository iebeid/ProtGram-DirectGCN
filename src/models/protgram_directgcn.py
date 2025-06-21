# src/models/protgram_directgcn.py
# ==============================================================================
# MODULE: models/protgram_directgcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 8.2 (Implemented hierarchical gating and dual-path transformations)
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
    A highly expressive GCN layer with separate and shared transformations for
    directed and undirected paths, combined via a hierarchical gating mechanism.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.use_vector_coeffs = use_vector_coeffs

        # --- Path-Specific Components ---
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_undirected = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_undirected = nn.Parameter(torch.Tensor(out_channels))

        # --- Shared Components (used by all paths) ---
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_directed_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_directed_shared_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_undirected_shared = nn.Parameter(torch.Tensor(out_channels))

        # --- Hierarchical Learnable Coefficients ---
        if self.use_vector_coeffs and self.num_nodes > 0:
            self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_directed_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_undirected_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_all_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
        else:
            self.use_vector_coeffs = False
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
            self.C_directed = nn.Parameter(torch.Tensor(1))
            self.C_undirected = nn.Parameter(torch.Tensor(1))
            self.C_all = nn.Parameter(torch.Tensor(1))

        # --- Learnable Node-Specific Constant ---
        if self.num_nodes > 0:
            self.constant = nn.Parameter(torch.Tensor(num_nodes, out_channels))
        else:
            self.constant = None

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_shared, self.lin_undirected]:
            nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_directed_shared_in,
                     self.bias_directed_shared_out, self.bias_undirected, self.bias_undirected_shared]:
            nn.init.zeros_(bias)

        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
            nn.init.ones_(self.C_directed_vec)
            nn.init.ones_(self.C_undirected_vec)
            nn.init.ones_(self.C_all_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)
            nn.init.ones_(self.C_directed)
            nn.init.ones_(self.C_undirected)
            nn.init.ones_(self.C_all)

        if self.constant is not None:
            nn.init.xavier_uniform_(self.constant)

    def forward(self, x: torch.Tensor,
                edge_index_in: torch.Tensor, edge_weight_in: Optional[torch.Tensor],
                edge_index_out: torch.Tensor, edge_weight_out: Optional[torch.Tensor],
                edge_index_undirected: torch.Tensor, edge_weight_undirected: Optional[torch.Tensor],
                original_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass implementing the hierarchical, dual-path logic."""

        # --- 1. Directed Incoming Path ---
        h_main_in = self.propagate(edge_index_in, x=self.lin_main_in(x), edge_weight=edge_weight_in)
        h_shared_in = self.propagate(edge_index_in, x=self.lin_shared(x), edge_weight=edge_weight_in)
        ic_combined = (h_main_in + self.bias_main_in) + (h_shared_in + self.bias_directed_shared_in)

        # --- 2. Directed Outgoing Path ---
        h_main_out = self.propagate(edge_index_out, x=self.lin_main_out(x), edge_weight=edge_weight_out)
        h_shared_out = self.propagate(edge_index_out, x=self.lin_shared(x), edge_weight=edge_weight_out)
        oc_combined = (h_main_out + self.bias_main_out) + (h_shared_out + self.bias_directed_shared_out)

        # --- 3. Undirected Structural Path ---
        h_main_undir = self.propagate(edge_index_undirected, x=self.lin_undirected(x), edge_weight=edge_weight_undirected)
        h_shared_undir = self.propagate(edge_index_undirected, x=self.lin_shared(x), edge_weight=edge_weight_undirected)
        uc_combined = (h_main_undir + self.bias_undirected) + (h_shared_undir + self.bias_undirected_shared)

        # --- 4. Get Coefficients and Constant ---
        if self.use_vector_coeffs and original_indices is not None:
            c_in, c_out = self.C_in_vec[original_indices], self.C_out_vec[original_indices]
            c_directed, c_undirected = self.C_directed_vec[original_indices], self.C_undirected_vec[original_indices]
            c_all = self.C_all_vec[original_indices]
            constant_term = self.constant[original_indices] if self.constant is not None else 0
        elif self.use_vector_coeffs:
            c_in, c_out = self.C_in_vec, self.C_out_vec
            c_directed, c_undirected = self.C_directed_vec, self.C_undirected_vec
            c_all = self.C_all_vec
            constant_term = self.constant if self.constant is not None else 0
        else:
            c_in, c_out, c_directed, c_undirected, c_all = self.C_in, self.C_out, self.C_directed, self.C_undirected, self.C_all
            constant_term = 0

        # --- 5. Final Hierarchical Combination ---
        directed_signal = c_directed * ((c_in * ic_combined) + (c_out * oc_combined))
        undirected_signal = c_undirected * uc_combined
        final_combination = (c_all * (undirected_signal + directed_signal)) + constant_term

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
        x = getattr(data, 'x', None)
        ei_in = getattr(data, 'edge_index_in', None)
        ew_in = getattr(data, 'edge_weight_in', None)
        ei_out = getattr(data, 'edge_index_out', None)
        ew_out = getattr(data, 'edge_weight_out', None)
        ei_undir = getattr(data, 'edge_index_undirected_norm', None)
        ew_undir = getattr(data, 'edge_weight_undirected_norm', None)
        original_indices = getattr(data, 'original_indices', None)

        if x is None or ei_in is None or ei_out is None or ei_undir is None:
            raise ValueError("ProtGramDirectGCN requires 'x', 'edge_index_in', 'edge_index_out', and 'edge_index_undirected_norm' in the Data object.")

        h = self._apply_pe(x)

        for i in range(len(self.convs)):
            h_res = h
            gcn_layer, res_layer = self.convs[i], self.res_projs[i]
            gcn_output = gcn_layer(h_res, ei_in, ew_in, ei_out, ew_out, ei_undir, ew_undir, original_indices)
            residual_output = res_layer(h_res)
            h = F.leaky_relu(gcn_output + residual_output)
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_embed_for_task = h
        task_logits = self.decoder_fc(final_embed_for_task)
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(final_embed_for_task, eps=self.l2_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embeddings