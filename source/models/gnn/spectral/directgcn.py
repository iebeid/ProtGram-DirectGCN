# ==============================================================================
# MODULE: models/gnn/spectral/directgcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 10.0 (Correctly implements conditional homophily/heterophily paths)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
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

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, gating_mode: str = 'vector',
                 use_homo_hetero_paths: bool = False):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.gating_mode = gating_mode
        self.use_homo_hetero_paths = use_homo_hetero_paths

        # --- Path-Specific Components ---
        if self.use_homo_hetero_paths:
            # Create separate layers for homophilous and heterophilous paths
            self.lin_main_in_homo = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_main_in_hetero = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_main_out_homo = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_main_out_hetero = nn.Linear(in_channels, out_channels, bias=False)
            self.bias_main_in_homo = nn.Parameter(torch.Tensor(out_channels))
            self.bias_main_in_hetero = nn.Parameter(torch.Tensor(out_channels))
            self.bias_main_out_homo = nn.Parameter(torch.Tensor(out_channels))
            self.bias_main_out_hetero = nn.Parameter(torch.Tensor(out_channels))
        else:
            # Standard single path layers
            self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
            self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
            self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))

        # Undirected path is always present
        self.lin_undirected = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_undirected = nn.Parameter(torch.Tensor(out_channels))

        # --- Projection layers for concatenated path features ---
        self.proj_in = nn.Linear(out_channels * 2, out_channels)
        self.proj_out = nn.Linear(out_channels * 2, out_channels)
        self.proj_undir = nn.Linear(out_channels * 2, out_channels)

        # --- Shared Components (used by all paths) ---
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_undir = nn.Parameter(torch.Tensor(out_channels))

        # --- Hierarchical Learnable Gating Coefficients ---
        if self.gating_mode in ['vector', 'node_gate_vector'] and self.num_nodes > 0:
            gate_dim = out_channels if self.gating_mode == 'node_gate_vector' else 1
            self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
            self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
            self.C_undirected_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
        elif self.gating_mode == 'scalar':
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
            self.C_undirected = nn.Parameter(torch.Tensor(1))

        # --- Learnable Node-Specific Constant ---
        if self.num_nodes > 0:
            self.constant = nn.Parameter(torch.Tensor(num_nodes, out_channels))
        else:
            self.constant = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all learnable parameters of the layer."""
        # --- FIX: Conditionally initialize the correct set of layers ---
        if self.use_homo_hetero_paths:
            layers_to_init = [self.lin_main_in_homo, self.lin_main_in_hetero, self.lin_main_out_homo,
                              self.lin_main_out_hetero]
            biases_to_init = [self.bias_main_in_homo, self.bias_main_in_hetero, self.bias_main_out_homo,
                              self.bias_main_out_hetero]
        else:
            layers_to_init = [self.lin_main_in, self.lin_main_out]
            biases_to_init = [self.bias_main_in, self.bias_main_out]

        # Always initialize these
        layers_to_init.extend([self.lin_undirected, self.lin_shared, self.proj_in, self.proj_out, self.proj_undir])
        biases_to_init.extend([self.bias_undirected, self.bias_shared_in, self.bias_shared_out, self.bias_shared_undir])

        for lin in layers_to_init:
            nn.init.xavier_uniform_(lin.weight)
            if hasattr(lin, 'bias') and lin.bias is not None:
                nn.init.zeros_(lin.bias)

        for bias in biases_to_init:
            nn.init.zeros_(bias)

        if self.gating_mode in ['vector', 'node_gate_vector'] and hasattr(self, 'C_in_vec'):
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
            nn.init.ones_(self.C_undirected_vec)
        elif self.gating_mode == 'scalar':
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)
            nn.init.ones_(self.C_undirected)

        if self.constant is not None:
            nn.init.xavier_uniform_(self.constant)

    def forward(self, x: torch.Tensor, data: Data) -> torch.Tensor:
        """Forward pass implementing the hierarchical, dual-path logic."""
        original_indices = getattr(data, 'original_indices', None)

        # --- 1. Directed Path Propagation ---
        if self.use_homo_hetero_paths:
            # --- FIX: Use the specialized homophily/heterophily paths ---
            h_in_homo = self.propagate(data.edge_index_in_homo, x=self.lin_main_in_homo(x),
                                       edge_weight=data.edge_weight_in_homo)
            h_in_hetero = self.propagate(data.edge_index_in_hetero, x=self.lin_main_in_hetero(x),
                                         edge_weight=data.edge_weight_in_hetero)
            h_out_homo = self.propagate(data.edge_index_out_homo, x=self.lin_main_out_homo(x),
                                        edge_weight=data.edge_weight_out_homo)
            h_out_hetero = self.propagate(data.edge_index_out_hetero, x=self.lin_main_out_hetero(x),
                                          edge_weight=data.edge_weight_out_hetero)

            # Combine features from homo/hetero paths before projection
            h_main_in = h_in_homo + self.bias_main_in_homo + h_in_hetero + self.bias_main_in_hetero
            h_main_out = h_out_homo + self.bias_main_out_homo + h_out_hetero + self.bias_main_out_hetero
        else:
            # Standard path propagation
            h_main_in_prop = self.propagate(data.edge_index_in, x=self.lin_main_in(x), edge_weight=data.edge_weight_in)
            h_main_out_prop = self.propagate(data.edge_index_out, x=self.lin_main_out(x),
                                             edge_weight=data.edge_weight_out)
            h_main_in = h_main_in_prop + self.bias_main_in
            h_main_out = h_main_out_prop + self.bias_main_out

        # --- 2. Undirected Path Propagation (always runs) ---
        h_main_undir_prop = self.propagate(data.edge_index_undirected_norm, x=self.lin_undirected(x),
                                           edge_weight=data.edge_weight_undirected_norm)
        h_main_undir = h_main_undir_prop + self.bias_undirected

        # --- 3. Shared transformation ---
        h_shared = self.lin_shared(x)

        # --- 4. Combine path-specific and shared features ---
        ic_combined = self.proj_in(torch.cat([h_main_in, h_shared + self.bias_shared_in], dim=-1))
        oc_combined = self.proj_out(torch.cat([h_main_out, h_shared + self.bias_shared_out], dim=-1))
        uc_combined = self.proj_undir(torch.cat([h_main_undir, h_shared + self.bias_shared_undir], dim=-1))

        # --- 5. Get Gating Coefficients and Constant based on the configured mode ---
        if self.gating_mode in ['vector', 'node_gate_vector']:
            if original_indices is not None:
                c_in, c_out = self.C_in_vec[original_indices], self.C_out_vec[original_indices]
                c_undirected = self.C_undirected_vec[original_indices]
                constant_term = self.constant[original_indices] if self.constant is not None else 0
            else:
                c_in, c_out = self.C_in_vec, self.C_out_vec
                c_undirected = self.C_undirected_vec
                constant_term = self.constant if self.constant is not None else 0
        elif self.gating_mode == 'scalar':
            c_in, c_out, c_undirected = self.C_in, self.C_out, self.C_undirected
            constant_term = 0
        elif self.gating_mode == 'none':
            c_in, c_out, c_undirected, constant_term = 1.0, 1.0, 1.0, 0.0
        else:
            raise ValueError(f"Unknown gating mode: '{self.gating_mode}'")

        # --- 6. Final Hierarchical Combination ---
        final_combination = (c_undirected * uc_combined) + (c_in * ic_combined) + (c_out * oc_combined) + constant_term
        return final_combination

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class DirectGCN(nn.Module):
    """The main GCN architecture, adapted for the new layer."""

    def __init__(self, layer_dims: List[int], num_graph_nodes: Optional[int], task_num_output_classes: int,
                 n_gram_len: int, use_homo_hetero_paths: bool,
                 one_gram_dim: int, max_pe_len: int, dropout: float, gating_mode: str,
                 l2_eps: float = 1e-12):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.one_gram_dim = one_gram_dim
        self.dropout = dropout
        self.l2_eps = l2_eps
        self.use_homo_hetero_paths = use_homo_hetero_paths

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
            self.convs.append(
                DirectGCNLayer(in_dim, out_dim, current_num_nodes, gating_mode, use_homo_hetero_paths))
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
        if x is None:
            raise ValueError("DirectGCN requires 'x' in the Data object.")

        # --- FIX: Check for required edge indices based on the mode ---
        if self.use_homo_hetero_paths:
            required_keys = ['edge_index_in_homo', 'edge_index_in_hetero', 'edge_index_out_homo',
                             'edge_index_out_hetero', 'edge_index_undirected_norm']
        else:
            required_keys = ['edge_index_in', 'edge_index_out', 'edge_index_undirected_norm']

        for key in required_keys:
            if not hasattr(data, key):
                raise ValueError(f"DirectGCN requires '{key}' in the Data object when use_homo_hetero_paths is {self.use_homo_hetero_paths}.")

        h = self._apply_pe(x)

        for i in range(len(self.convs)):
            h_res = h
            gcn_layer, res_layer = self.convs[i], self.res_projs[i]
            # Pass the entire data object to the layer
            gcn_output = gcn_layer(h_res, data)
            residual_output = res_layer(h_res)
            h = F.leaky_relu(gcn_output + residual_output)
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_embed_for_task = h
        logits = self.decoder_fc(final_embed_for_task)
        final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(final_embed_for_task, eps=self.l2_eps)

        return logits, final_normalized_embeddings