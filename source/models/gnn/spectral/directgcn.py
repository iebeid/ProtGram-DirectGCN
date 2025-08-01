# ==============================================================================
# MODULE: models/gnn/directgcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 8.3 (Stable & Corrected - Cleaned up hierarchical gating and dual-path logic)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from source.utils.models import EmbeddingProcessor


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

        # --- NEW: Projection layers for concatenated path features ---
        self.proj_in = nn.Linear(out_channels * 2, out_channels)
        self.proj_out = nn.Linear(out_channels * 2, out_channels)
        self.proj_undir = nn.Linear(out_channels * 2, out_channels)

        # --- Shared Components (used by all paths) ---
        # A single shared linear layer for all paths to learn a general transformation
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)
        # Separate biases for the shared transformation on each path
        self.bias_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_undir = nn.Parameter(torch.Tensor(out_channels))

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
        # Initialize all linear layers
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_undirected, self.lin_shared,
                    self.proj_in, self.proj_out, self.proj_undir]:
            nn.init.xavier_uniform_(lin.weight)
            if hasattr(lin, 'bias') and lin.bias is not None:
                nn.init.zeros_(lin.bias)

        # Initialize all bias terms
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_undirected,
                     self.bias_shared_in, self.bias_shared_out, self.bias_shared_undir]:
            nn.init.zeros_(bias)

        # Initialize all gating coefficients to 1
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
        path_specific_in = h_main_in + self.bias_main_in
        shared_in = self.lin_shared(x) + self.bias_shared_in
        concatenated_in = torch.cat([path_specific_in, shared_in], dim=-1)
        ic_combined = self.proj_in(concatenated_in)

        # --- 2. Directed Outgoing Path ---
        h_main_out = self.propagate(edge_index_out, x=self.lin_main_out(x), edge_weight=edge_weight_out)
        path_specific_out = h_main_out + self.bias_main_out
        shared_out = self.lin_shared(x) + self.bias_shared_out
        concatenated_out = torch.cat([path_specific_out, shared_out], dim=-1)
        oc_combined = self.proj_out(concatenated_out)

        # --- 3. Undirected Structural Path ---
        h_main_undir = self.propagate(edge_index_undirected, x=self.lin_undirected(x), edge_weight=edge_weight_undirected)
        path_specific_undir = h_main_undir + self.bias_undirected
        shared_undir = self.lin_shared(x) + self.bias_shared_undir
        concatenated_undir = torch.cat([path_specific_undir, shared_undir], dim=-1)
        uc_combined = self.proj_undir(concatenated_undir)

        # --- 4. Get Coefficients and Constant ---
        if self.use_vector_coeffs and original_indices is not None:
            # When using Cluster-GCN, we need to select the coefficients for the nodes in the current subgraph
            c_in, c_out = self.C_in_vec[original_indices], self.C_out_vec[original_indices]
            c_undirected = self.C_undirected_vec[original_indices]
            constant_term = self.constant[original_indices] if self.constant is not None else 0
        elif self.use_vector_coeffs:
            # Full-batch training
            c_in, c_out = self.C_in_vec, self.C_out_vec
            c_undirected = self.C_undirected_vec
            constant_term = self.constant if self.constant is not None else 0
        else:
            # Using scalar coefficients
            c_in, c_out, c_undirected = self.C_in, self.C_out, self.C_undirected
            constant_term = 0

        # --- 5. Final Hierarchical Combination ---
        # Combine the three paths using the learned coefficients
        final_combination = (c_undirected * uc_combined) + (c_in * ic_combined) + (c_out * oc_combined) + constant_term

        return final_combination

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class DirectGCN(nn.Module):
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
            # Add a projection layer for the residual connection if dimensions don't match
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
        """Applies positional embeddings to the input features if applicable."""
        if self.pe_layer is None: return x
        # Check if the input feature dimension matches the expected format for PE
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
            raise ValueError("DirectGCN requires 'x', 'edge_index_in', 'edge_index_out', and 'edge_index_undirected_norm' in the Data object.")

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
        # The final embeddings for downstream tasks are L2 normalized
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(final_embed_for_task, eps=self.l2_eps)

        # Return raw logits for compatibility with F.cross_entropy loss
        return task_logits, final_normalized_embeddings