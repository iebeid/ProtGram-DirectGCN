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

from source.utils.post.embedding_loader import EmbeddingLoader
from source.utils.post.embedding_processor import EmbeddingProcessor


class DirectGCNLayer(MessagePassing):
    """
    A highly expressive GCN layer with separate and shared transformations for
    directed and undirected paths, combined via a hierarchical gating mechanism. This layer
    is the core of the DirectGCN model, designed to capture complex relational
    information in directed graphs by processing multiple graph "views" in parallel.

    The key architectural ideas are:
    1.  **Multi-Path Message Passing:** Instead of a single convolution, it performs
        separate message passing operations over different graph representations
        (e.g., incoming edges, outgoing edges, undirected edges, homophilic edges).
    2.  **Shared & Specific Transformations:** Each path has its own learnable weight
        matrix, but also benefits from a shared transformation, allowing the model
        to learn both path-specific and general features.
    3.  **Hierarchical Gating:** A learnable gating mechanism weights the contribution
        of each path to the final node representation, allowing the model to
        dynamically decide which graph views are most important for each node.
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
            # --- NEW: Add layers for the new top-level homo/hetero paths ---
            self.lin_homo = nn.Linear(in_channels, out_channels, bias=False)
            self.lin_hetero = nn.Linear(in_channels, out_channels, bias=False)
            self.bias_homo = nn.Parameter(torch.Tensor(out_channels))
            self.bias_hetero = nn.Parameter(torch.Tensor(out_channels))
        # Standard in/out paths are always present
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
        # --- NEW: Add projection layers for the new paths ---
        if self.use_homo_hetero_paths:
            # Note: These do not use the shared bias, as they represent distinct views
            self.proj_homo = nn.Linear(out_channels * 2, out_channels)
            self.proj_hetero = nn.Linear(out_channels * 2, out_channels)

        # --- Shared Components (used by all paths) ---
        self.lin_shared = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_shared_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_shared_undir = nn.Parameter(torch.Tensor(out_channels))
        if self.use_homo_hetero_paths:
            self.bias_shared_homo = nn.Parameter(torch.Tensor(out_channels))
            self.bias_shared_hetero = nn.Parameter(torch.Tensor(out_channels))

        # --- Hierarchical Learnable Gating Coefficients ---
        if self.gating_mode in ['vector', 'node_gate_vector'] and self.num_nodes > 0:
            gate_dim = out_channels if self.gating_mode == 'node_gate_vector' else 1
            self.C_in_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
            self.C_out_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
            self.C_undirected_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
            # --- NEW: Add gating parameters for the new paths ---
            if self.use_homo_hetero_paths:
                self.C_homo_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
                self.C_hetero_vec = nn.Parameter(torch.Tensor(num_nodes, gate_dim))
        elif self.gating_mode == 'scalar':
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
            self.C_undirected = nn.Parameter(torch.Tensor(1))
            # --- NEW: Add gating parameters for the new paths ---
            if self.use_homo_hetero_paths:
                self.C_homo = nn.Parameter(torch.Tensor(1))
                self.C_hetero = nn.Parameter(torch.Tensor(1))

        # --- Learnable Node-Specific Constant ---
        if self.num_nodes > 0:
            self.constant = nn.Parameter(torch.Tensor(num_nodes, out_channels))
        else:
            self.constant = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all learnable parameters of the layer."""
        layers_to_init = [self.lin_main_in, self.lin_main_out]
        biases_to_init = [self.bias_main_in, self.bias_main_out]

        # Always initialize these
        layers_to_init.extend([self.lin_undirected, self.lin_shared, self.proj_in, self.proj_out, self.proj_undir])
        # --- FIX: Move conditional bias initializations into the conditional block ---
        biases_to_init.extend([self.bias_undirected, self.bias_shared_in, self.bias_shared_out, self.bias_shared_undir])
        # --- NEW: Add new layers to the initialization lists ---
        if self.use_homo_hetero_paths:
            layers_to_init.extend([self.lin_homo, self.lin_hetero, self.proj_homo, self.proj_hetero])
            # --- FIX: Add the corresponding biases here ---
            biases_to_init.extend([self.bias_homo, self.bias_hetero, self.bias_shared_homo, self.bias_shared_hetero])

        for lin in layers_to_init:
            nn.init.xavier_uniform_(lin.weight)
            if hasattr(lin, 'bias') and lin.bias is not None:
                nn.init.zeros_(lin.bias)

        for bias in biases_to_init:
            nn.init.zeros_(bias)

        if self.gating_mode in ['vector', 'node_gate_vector'] and hasattr(self, 'C_in_vec'):
            nn.init.xavier_uniform_(self.C_in_vec)
            nn.init.xavier_uniform_(self.C_out_vec)
            nn.init.xavier_uniform_(self.C_undirected_vec)
            # --- NEW: Initialize new gating vectors ---
            if self.use_homo_hetero_paths:
                nn.init.xavier_uniform_(self.C_homo_vec)
                nn.init.xavier_uniform_(self.C_hetero_vec)
        elif self.gating_mode == 'scalar':
            nn.init.xavier_uniform_(self.C_in)
            nn.init.xavier_uniform_(self.C_out)
            nn.init.xavier_uniform_(self.C_undirected)
            # --- NEW: Initialize new gating scalars ---
            if self.use_homo_hetero_paths:
                nn.init.xavier_uniform_(self.C_homo)
                nn.init.xavier_uniform_(self.C_hetero)

        if self.constant is not None:
            # --- FIX: Initialize bias-like constant to zeros for better stability ---
            nn.init.zeros_(self.constant)

    def forward(self, x: torch.Tensor, data: Data) -> torch.Tensor:
        """Forward pass implementing the hierarchical, dual-path logic."""
        # --- 0. Setup ---
        # Get original node indices if this is a subgraph from clustered training
        original_indices = getattr(data, 'original_indices', None)

        # --- 1. Shared Transformation ---
        # This computes a common feature representation that will be combined
        # with the output of each specialized message-passing path. This is
        # done once per forward pass for efficiency.
        h_shared = self.lin_shared(x)

        # --- 2. Path-Specific Propagation and Combination ---
        # Each path (in-degree, out-degree, undirected, etc.) performs its own
        # message passing on its specific graph view (e.g., mathcal_A_in).
        # The result is then combined with the shared features via concatenation
        # and a final projection layer, creating a rich, path-aware representation.
        path_combinations = []

        h_main_in = self.propagate(data.edge_index_mathcal_in, x=self.lin_main_in(x), edge_weight=data.edge_weight_mathcal_in) + self.bias_main_in
        h_main_out = self.propagate(data.edge_index_mathcal_out, x=self.lin_main_out(x), edge_weight=data.edge_weight_mathcal_out) + self.bias_main_out
        h_main_undir = self.propagate(data.edge_index_undirected_norm, x=self.lin_undirected(x), edge_weight=data.edge_weight_undirected_norm) + self.bias_undirected
        path_combinations.append(self.proj_in(torch.cat([h_main_in, h_shared + self.bias_shared_in], dim=-1)))
        path_combinations.append(self.proj_out(torch.cat([h_main_out, h_shared + self.bias_shared_out], dim=-1)))
        path_combinations.append(self.proj_undir(torch.cat([h_main_undir, h_shared + self.bias_shared_undir], dim=-1)))

        # Conditional paths for homophily/heterophily
        if self.use_homo_hetero_paths:
            h_homo = self.propagate(data.edge_index_homo_norm, x=self.lin_homo(x), edge_weight=data.edge_weight_homo_norm) + self.bias_homo
            h_hetero = self.propagate(data.edge_index_hetero_norm, x=self.lin_hetero(x), edge_weight=data.edge_weight_hetero_norm) + self.bias_hetero
            # Note: using bias_shared_undir for both as they are undirected views
            path_combinations.append(self.proj_homo(torch.cat([h_homo, h_shared + self.bias_shared_homo], dim=-1)))
            path_combinations.append(self.proj_hetero(torch.cat([h_hetero, h_shared + self.bias_shared_hetero], dim=-1)))

        # --- 3. Hierarchical Gating and Path Combination ---
        # A learnable gating mechanism (scalar, vector, or node-specific) calculates
        # weights for each path. These weights determine the contribution of each
        # path's features to the final node representation.
        if self.gating_mode == 'none':
            # In 'none' mode, all paths contribute equally (summation).
            final_combination = torch.stack(path_combinations, dim=0).sum(dim=0)
        else:
            # Handle 'scalar', 'vector', and 'node_gate_vector' modes
            gating_logits_list = []
            if self.gating_mode in ['vector', 'node_gate_vector']:
                gating_logits_list.extend([self.C_in_vec, self.C_out_vec, self.C_undirected_vec])
                if self.use_homo_hetero_paths:
                    gating_logits_list.extend([self.C_homo_vec, self.C_hetero_vec])
                gating_logits_full = torch.stack(gating_logits_list, dim=-1)
                gating_logits = gating_logits_full[original_indices] if original_indices is not None else gating_logits_full
            else:  # scalar mode
                gating_logits_list.extend([self.C_in, self.C_out, self.C_undirected])
                if self.use_homo_hetero_paths:
                    gating_logits_list.extend([self.C_homo, self.C_hetero])
                gating_logits = torch.cat(gating_logits_list, dim=0)

            gating_weights = torch.sigmoid(gating_logits)
            final_combination = torch.zeros_like(path_combinations[0])
            if self.gating_mode == 'scalar':
                for i, path_emb in enumerate(path_combinations):
                    final_combination += gating_weights[i] * path_emb
            else:  # vector and node_gate_vector modes
                for i, path_emb in enumerate(path_combinations):
                    final_combination += gating_weights[:, :, i] * path_emb

        # --- 4. Final Node-Specific Bias ---
        # A learnable, node-specific constant is added to the final combined
        # representation. This acts as a final bias term for each node, allowing
        # for fine-grained adjustments.
        if self.gating_mode != 'scalar' and self.constant is not None:
            constant_term = self.constant[original_indices] if original_indices is not None else self.constant
            final_combination += constant_term

        return final_combination

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class DirectGCN(nn.Module):
    """The main GCN architecture, adapted for the new layer."""

    def __init__(self, layer_dims: List[int], num_graph_nodes: Optional[int], task_num_output_classes: int,
                 n_gram_len: int, use_homo_hetero_paths: bool,
                 dropout_rate: float, gating_mode: str,
                 l2_eps: float = 1e-12):
        super().__init__()
        self.n_gram_len = n_gram_len
        self.dropout_rate = dropout_rate
        self.l2_eps = l2_eps
        self.embedding_output = None
        self.use_homo_hetero_paths = use_homo_hetero_paths

        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # --- DEFINITIVE FIX for Model Alignment ---
        # This standardizes the benchmark architecture to be a simple 2-layer stack,
        # consistent with the other benchmark models like GCN and GAT.
        # The complex multi-path logic is now only used for the main ProtGram pipeline.
        is_benchmark_or_singleton = num_graph_nodes is not None

        if is_benchmark_or_singleton:
            hidden_dim = layer_dims[1] # The first element after input_dim
            self.convs.append(DirectGCNLayer(layer_dims[0], hidden_dim, num_graph_nodes, gating_mode, use_homo_hetero_paths))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.convs.append(DirectGCNLayer(hidden_dim, task_num_output_classes, num_graph_nodes, gating_mode, use_homo_hetero_paths))
        else: # Original logic for the main ProtGram pipeline
            for i in range(len(layer_dims) - 1):
                in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
                self.convs.append(DirectGCNLayer(in_dim, out_dim, num_graph_nodes, gating_mode, use_homo_hetero_paths))
                self.res_projs.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())
                self.layer_norms.append(nn.LayerNorm(out_dim))

        if num_graph_nodes is not None:  # This check implies a benchmark or singleton context
            final_embedding_dim = layer_dims[-1]
            self.decoder_fc = nn.Linear(final_embedding_dim, task_num_output_classes)
        else:  # Main pipeline context
            final_embedding_dim = layer_dims[-1]
            decoder_hidden_dim = final_embedding_dim // 2 if final_embedding_dim > 1 else 1
            self.decoder_fc = nn.Sequential(
                nn.Linear(final_embedding_dim, decoder_hidden_dim), nn.ReLU(),
                nn.Dropout(p=0.5), nn.Linear(decoder_hidden_dim, task_num_output_classes)
            )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = getattr(data, 'x', None)
        if x is None:
            raise ValueError("DirectGCN requires 'x' in the Data object.")

        # Check for required edge indices based on the mode
        if self.use_homo_hetero_paths:
            required_keys = ['edge_index_mathcal_in', 'edge_index_mathcal_out', 'edge_index_undirected_norm',
                             'edge_index_homo_norm', 'edge_index_hetero_norm']
        else:
            required_keys = ['edge_index_mathcal_in', 'edge_index_mathcal_out', 'edge_index_undirected_norm']

        for key in required_keys:
            if not hasattr(data, key):
                raise ValueError(f"DirectGCN requires '{key}' in the Data object when use_homo_hetero_paths is {self.use_homo_hetero_paths}.")

        h = x

        # --- DEFINITIVE FIX for Model Alignment ---
        # Use a simple forward pass for the standardized benchmark architecture.
        is_benchmark_or_singleton = len(self.res_projs) == 0

        if is_benchmark_or_singleton:
            # Standard 2-layer GCN-like forward pass
            h = self.convs[0](h, data)
            h = self.layer_norms[0](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            self.embedding_output = h # Embedding is the output of the hidden layer
            # --- DEFINITIVE FIX: Use the dedicated decoder_fc for logits ---
            # The previous implementation used the final conv layer, which is inconsistent
            # with the model's __init__ method where a separate decoder is created.
            logits = self.decoder_fc(self.embedding_output)
            return logits, self.embedding_output.detach() # Return detached embeddings
        else: # Original logic for the main ProtGram pipeline
            for i in range(len(self.convs)):
                h_pre_act = self.convs[i](h, data) + self.res_projs[i](h)
                h_norm = self.layer_norms[i](h_pre_act)
            h = F.leaky_relu(h_norm)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)

            self.embedding_output = h
            logits = self.decoder_fc(self.embedding_output)
            final_normalized_embeddings = EmbeddingProcessor.l2_normalize_torch(h, eps=self.l2_eps)
            return logits, final_normalized_embeddings.detach()