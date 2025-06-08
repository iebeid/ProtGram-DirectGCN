# ==============================================================================
# MODULE: models/prot_ngram_gcn.py
# PURPOSE: Contains the PyTorch class definitions for the custom GCN model.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from typing import Optional


class DirectGCNLayer(MessagePassing):
    """
    The custom directed GCN layer, renamed from CustomDiGCNLayerPyG_ngram.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels))
        self.use_vector_coeffs = use_vector_coeffs

        if self.use_vector_coeffs:
            self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, 1))
        else:
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_skip]:
            nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_skip_in, self.bias_skip_out]:
            nn.init.zeros_(bias)
        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)

    def forward(self, x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out):
        aggr_main_in = self.propagate(edge_index_in, x=self.lin_main_in(x), edge_weight=edge_weight_in)
        aggr_skip_in = self.propagate(edge_index_in, x=self.lin_skip(x), edge_weight=edge_weight_in)
        ic_combined = aggr_main_in + self.bias_main_in + aggr_skip_in + self.bias_skip_in

        aggr_main_out = self.propagate(edge_index_out, x=self.lin_main_out(x), edge_weight=edge_weight_out)
        aggr_skip_out = self.propagate(edge_index_out, x=self.lin_skip(x), edge_weight=edge_weight_out)
        oc_combined = aggr_main_out + self.bias_main_out + aggr_skip_out + self.bias_skip_out

        c_in = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out = self.C_out_vec if self.use_vector_coeffs else self.C_out

        return c_in * ic_combined + c_out * oc_combined

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j


class ProtNgramGCN(nn.Module):
    """
    The main GCN architecture, renamed from ProtDiGCNEncoderDecoder_ngram.
    """

    def __init__(self, num_initial_features, hidden_dim1, hidden_dim2, num_graph_nodes, task_num_output_classes, n_gram_len, one_gram_dim, max_pe_len, dropout, use_vector_coeffs):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.one_gram_dim = one_gram_dim
        self.dropout = dropout
        self.l2_eps = 1e-12

        self.pe_layer = nn.Embedding(max_pe_len, one_gram_dim) if one_gram_dim > 0 and max_pe_len > 0 else None
        self.conv1 = DirectGCNLayer(num_initial_features, hidden_dim1, num_graph_nodes, use_vector_coeffs)
        self.conv2 = DirectGCNLayer(hidden_dim1, hidden_dim1, num_graph_nodes, use_vector_coeffs)
        self.conv3 = DirectGCNLayer(hidden_dim1, hidden_dim2, num_graph_nodes, use_vector_coeffs)
        self.res_proj1 = nn.Linear(num_initial_features, hidden_dim1) if num_initial_features != hidden_dim1 else nn.Identity()
        self.res_proj3 = nn.Linear(hidden_dim1, hidden_dim2) if hidden_dim1 != hidden_dim2 else nn.Identity()
        self.decoder_fc = nn.Linear(hidden_dim2, task_num_output_classes)

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe_layer is None: return x
        x_pe = x.clone()
        x_reshaped = x_pe.view(-1, self.n_gram_len, self.one_gram_dim)
        pos_to_enc = min(self.n_gram_len, self.pe_layer.num_embeddings)
        if pos_to_enc > 0:
            pos_indices = torch.arange(0, pos_to_enc, device=x.device, dtype=torch.long)
            pe = self.pe_layer(pos_indices)
            x_reshaped[:, :pos_to_enc, :] += pe.unsqueeze(0)
        return x_reshaped.view(-1, self.n_gram_len * self.one_gram_dim)

    def forward(self, data=None, x=None, edge_index=None, **kwargs):
        """
        A flexible forward pass that can accept either a PyG Data object
        or individual tensors as keyword arguments.
        """
        if data is not None:
            # Called from the main GCN training pipeline
            x, ei_in, ew_in, ei_out, ew_out = data.x, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out
        elif x is not None and edge_index is not None:
            # Called from the GNN benchmarking pipeline
            # The benchmark datasets have undirected edges, so we'll use the same edge_index for in and out.
            ei_in, ew_in = edge_index, kwargs.get('edge_weight', None)
            ei_out, ew_out = edge_index, kwargs.get('edge_weight', None)
        else:
            raise ValueError("ProtNgramGCN forward pass requires either a 'data' object or 'x' and 'edge_index' arguments.")

        x_pe = self._apply_pe(x)

        h1 = F.tanh(self.conv1(x_pe, ei_in, ew_in, ei_out, ew_out) + self.res_proj1(x_pe))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.tanh(self.conv2(h1, ei_in, ew_in, ei_out, ew_out) + h1)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        h3 = F.tanh(self.conv3(h2, ei_in, ew_in, ei_out, ew_out) + self.res_proj3(h2))

        final_embed_for_task = F.dropout(h3, p=self.dropout, training=self.training)
        task_logits = self.decoder_fc(final_embed_for_task)

        norm = torch.norm(h3, p=2, dim=1, keepdim=True)
        final_normalized_embeddings = h3 / (norm + self.l2_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embeddings
