# ==============================================================================
# MODULE: models/prot_ngram_gcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# VERSION: 7.0 (Corrected Layer-Internal Architecture)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from typing import Optional, List



class DirectGCNLayer(MessagePassing):
    """
    The custom directed GCN layer, with the corrected internal architecture
    that explicitly separates main and shared pathways for each direction,
    as described in the user's PDF and reference code.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')
        # --- Main Path Weights ---
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))

        # --- Shared Path (Skip/W_all) Weights ---
        # Renamed lin_skip to lin_shared for clarity
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)
        # Separate biases for the shared path, as per the reference code
        self.bias_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_out = nn.Parameter(torch.Tensor(out_channels))

        self.use_vector_coeffs = use_vector_coeffs

        # --- Adaptive Coefficients ---
        if self.use_vector_coeffs:
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

        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)

    def forward(self, x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out):
        """Forward pass implementing the full ProtDiGCN layer logic."""

        # --- Incoming Path ---
        # 1. Main incoming component (A_in * (H * W_in))
        h_main_in = self.propagate(edge_index_in, x=self.lin_main_in(x), edge_weight=edge_weight_in)
        # 2. Shared incoming component (A_in * (H * W_all))
        h_shared_in = self.propagate(edge_index_in, x=self.lin_shared(x), edge_weight=edge_weight_in)
        # 3. Combine with biases: (A_in*H*W_in + b_in) + (A_in*H*W_all + b_all_in)
        ic_combined = (h_main_in + self.bias_main_in) + (h_shared_in + self.bias_shared_in)

        # --- Outgoing Path ---
        # 1. Main outgoing component (A_out * (H * W_out))
        h_main_out = self.propagate(edge_index_out, x=self.lin_main_out(x), edge_weight=edge_weight_out)
        # 2. Shared outgoing component (A_out * (H * W_all))
        h_shared_out = self.propagate(edge_index_out, x=self.lin_shared(x), edge_weight=edge_weight_out)
        # 3. Combine with biases: (A_out*H*W_out + b_out) + (A_out*H*W_all + b_all_out)
        oc_combined = (h_main_out + self.bias_main_out) + (h_shared_out + self.bias_shared_out)

        c_in = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out = self.C_out_vec if self.use_vector_coeffs else self.C_out

        # Final adaptive combination of the two directional paths
        return c_in * ic_combined + c_out * oc_combined

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """The message function for PyG's message passing."""
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j


class ProtNgramGCN(nn.Module):
    """
    The main GCN architecture, with a DYNAMIC number of GCN and residual
    layers. This class structure is UNCHANGED from the previous version.
    """

    def __init__(self, layer_dims: List[int], num_graph_nodes: int, task_num_output_classes: int, n_gram_len: int, one_gram_dim: int, max_pe_len: int, dropout: float, use_vector_coeffs: bool):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.one_gram_dim = one_gram_dim
        self.dropout = dropout
        self.l2_eps = 1e-12

        self.pe_layer = nn.Embedding(max_pe_len, one_gram_dim) if one_gram_dim > 0 and max_pe_len > 0 else None

        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            self.convs.append(DirectGCNLayer(in_dim, out_dim, num_graph_nodes, use_vector_coeffs))
            self.res_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())

        final_embedding_dim = layer_dims[-1]
        self.decoder_fc = nn.Linear(final_embedding_dim, task_num_output_classes)

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe_layer is None or self.n_gram_len == 0 or self.one_gram_dim == 0: return x
        x_pe = x.clone()
        x_reshaped = x_pe.view(-1, self.n_gram_len, self.one_gram_dim)
        pos_to_enc = min(self.n_gram_len, self.pe_layer.num_embeddings)
        if pos_to_enc > 0:
            pos_indices = torch.arange(0, pos_to_enc, device=x.device, dtype=torch.long)
            pe = self.pe_layer(pos_indices)
            x_reshaped[:, :pos_to_enc, :] += pe.unsqueeze(0)
        return x_reshaped.view(-1, self.n_gram_len * self.one_gram_dim)

    def forward(self, data=None, x=None, edge_index=None, **kwargs):
        if data is not None:
            x, ei_in, ew_in, ei_out, ew_out = data.x, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out
        elif x is not None and edge_index is not None:
            ei_in, ew_in = edge_index, kwargs.get('edge_weight', None)
            ei_out, ew_out = edge_index, kwargs.get('edge_weight', None)
        else:
            raise ValueError("ProtNgramGCN forward pass requires either a 'data' object or 'x' and 'edge_index' arguments.")

        h = self._apply_pe(x)

        for i in range(len(self.convs)):
            h_res = h
            h = F.tanh(self.convs[i](h, ei_in, ew_in, ei_out, ew_out) + self.res_projs[i](h_res))
            h = F.dropout(h, p=self.dropout, training=self.training)

        final_embed_for_task = h
        task_logits = self.decoder_fc(final_embed_for_task)

        norm = torch.norm(final_embed_for_task, p=2, dim=1, keepdim=True)
        final_normalized_embeddings = final_embed_for_task / (norm + self.l2_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embeddings
