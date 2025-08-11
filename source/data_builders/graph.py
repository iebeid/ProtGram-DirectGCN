# ==============================================================================
# MODULE: data_builders/graph.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 10.0 (Removed dead code and fixed device inconsistencies)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import os
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import subgraph


class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple]):  # nodes keys are int IDs
        self.idx_to_node_map_from_constructor = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []

        self.node_to_idx: Dict[Any, int] = {}
        self.idx_to_node: Dict[int, Any] = {}
        self.number_of_nodes: int = 0
        self.node_names: List[Any] = []
        self.edges: List[Tuple] = []
        self.number_of_edges: int = 0

        self._process_constructor_inputs()

    def _process_constructor_inputs(self):
        """
        Processes the nodes and edges passed to the constructor.
        Assumes `nodes` is a map from integer index to node name (e.g., n-gram string).
        Assumes `edges` contains tuples where the first two elements are integer indices.
        """
        if not self.idx_to_node_map_from_constructor and not self.original_edges:
            self.number_of_nodes = 0
            self.edges = []
            self.number_of_edges = 0
            return

        all_integer_indices = set()
        if self.idx_to_node_map_from_constructor:
            all_integer_indices.update(self.idx_to_node_map_from_constructor.keys())

        for edge_tuple in self.original_edges:
            if len(edge_tuple) >= 2:
                if not isinstance(edge_tuple[0], (int, np.integer)) or \
                        not isinstance(edge_tuple[1], (int, np.integer)):
                    continue
                all_integer_indices.add(int(edge_tuple[0]))
                all_integer_indices.add(int(edge_tuple[1]))

        if not all_integer_indices and not self.idx_to_node_map_from_constructor:
            self.number_of_nodes = 0
            return

        max_node_map_idx = -1
        if self.idx_to_node_map_from_constructor:
            valid_node_indices = {idx for idx in self.idx_to_node_map_from_constructor.keys() if
                                  isinstance(idx, (int, np.integer)) and idx >= 0}
            # --- FIX: Add a warning for potential data inconsistency ---
            if all_integer_indices and valid_node_indices and max(all_integer_indices) > max(valid_node_indices):
                # --- ENHANCEMENT: Make the warning more specific and explain the consequence ---
                print(f"  - WARNING: Data inconsistency detected. Max edge index ({max(all_integer_indices)}) "
                      f"exceeds max node map index ({max(valid_node_indices)}). This can lead to silent data loss.")
            if valid_node_indices:
                max_node_map_idx = max(valid_node_indices)

        max_edge_idx = -1
        if all_integer_indices:
            max_edge_idx = max(all_integer_indices)

        self.number_of_nodes = max(max_node_map_idx, max_edge_idx) + 1

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                node_name = f"__NODE_{i}__"
            temp_idx_to_node_name[i] = str(node_name)

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}
        self.node_names = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]

        self.edges = self.original_edges
        self.number_of_edges = len(self.edges)

    def get_node_to_idx_map(self) -> Dict[str, int]:
        """Returns a copy of the node name to index mapping."""
        return self.node_to_idx.copy()


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any],
                 edge_file_path: Optional[str] = None,
                 epsilon_propagation: float = 1e-9, n_value: Optional[int] = None):

        super().__init__(nodes=nodes, edges=[])

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = n_value

        self.A_out_w: torch.Tensor
        self.A_in_w: torch.Tensor
        self.A_undirected_norm_sparse: torch.Tensor
        self.mathcal_A_out: torch.Tensor
        self.mathcal_A_in: torch.Tensor
        # Top-level adjacency matrices for pure homophily/heterophily views
        self.A_homo_w: Optional[torch.Tensor] = None
        self.A_hetero_w: Optional[torch.Tensor] = None
        # --- NEW: Normalized versions of the homophily/heterophily matrices ---
        self.A_homo_norm: Optional[torch.Tensor] = None
        self.A_hetero_norm: Optional[torch.Tensor] = None

        if self.number_of_nodes > 0 and edge_file_path and os.path.exists(edge_file_path):
            print(f"    Loading edges from {os.path.basename(edge_file_path)}...")
            try:
                edge_df = pd.read_parquet(edge_file_path)
                source_indices = edge_df['source'].to_numpy(dtype=np.int64)
                target_indices = edge_df['target'].to_numpy(dtype=np.int64)
                weights = edge_df['weight'].to_numpy(dtype=np.float32)
                del edge_df
                gc.collect()

                self.number_of_edges = len(source_indices)
                self._create_raw_weighted_adj_matrices_torch(source_indices, target_indices, weights)
                self._create_undirected_normalized_adj_matrix()
                self._create_propagation_matrices()

            except Exception as e:
                print(f"    ❌ Error reading edge file {edge_file_path}: {e}. Initializing empty graph.")
                self._initialize_empty_matrices()
        else:
            self._initialize_empty_matrices()

    def _initialize_empty_matrices(self):
        """Helper to set all matrices to empty sparse tensors."""
        self.number_of_edges = 0
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        empty_values = torch.empty(0, dtype=torch.float32)
        size_empty = (self.number_of_nodes, self.number_of_nodes)

        self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.mathcal_A_out = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.mathcal_A_in = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        # Initialize top-level views
        self.A_homo_w = None
        self.A_hetero_w = None
        self.A_homo_norm = None
        self.A_hetero_norm = None

    @staticmethod
    def _sparse_identity(size: int, device: torch.device) -> torch.Tensor:
        """Creates a sparse identity matrix of given size."""
        if size <= 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_values = torch.empty(0, dtype=torch.float32, device=device)
            valid_size = max(0, size)
            return torch.sparse_coo_tensor(empty_indices, empty_values, (valid_size, valid_size)).coalesce()

        indices = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        values = torch.ones(size, device=device, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()

    def _create_raw_weighted_adj_matrices_torch(self, source_indices: np.ndarray, target_indices: np.ndarray,
                                                weights: np.ndarray):
        """Creates sparse adjacency matrices directly from numpy arrays with memory optimization."""
        size = (self.number_of_nodes, self.number_of_nodes)

        source_tensor = torch.from_numpy(source_indices)
        target_tensor = torch.from_numpy(target_indices)
        edge_indices_tensor = torch.stack([source_tensor, target_tensor]).long()
        del source_tensor, target_tensor
        gc.collect()

        edge_weights_tensor = torch.from_numpy(weights).float()
        del weights
        gc.collect()

        self.A_out_w = torch.sparse_coo_tensor(edge_indices_tensor, edge_weights_tensor, size).coalesce()
        del edge_indices_tensor, edge_weights_tensor
        gc.collect()

        self.A_in_w = self.A_out_w.t().coalesce()

    def _create_undirected_normalized_adj_matrix(self):
        """
        Creates a symmetric, degree-normalized adjacency matrix.

        This method follows a standard and robust procedure for creating a normalized
        undirected graph representation from the raw, directed transition counts:

        1.  **Symmetrize Raw Counts**: It first creates a symmetric matrix `A_undir_w`
            by adding the outgoing matrix `A_out_w` and the incoming matrix `A_in_w`.
            This correctly sums the raw counts for any reciprocal edges.
        2.  **Add Self-Loops**: It adds self-loops (with a weight of 1.0) to this
            symmetric, raw-count-weighted matrix for stability.
        3.  **Normalize**: Finally, it performs symmetric degree normalization
            (D^-0.5 * A * D^-0.5) on the result to produce the final matrix.
        """
        print(f"  Creating undirected normalized adjacency matrix for n={self.n_value}...")
        if self.number_of_nodes == 0:
            return

        # 1. Create a symmetric weighted matrix by adding A_out and its transpose (A_in)
        # This correctly sums weights for reciprocal edges.
        A_undir_w = (self.A_out_w + self.A_in_w).coalesce()

        # 2. Add self-loops to the weighted undirected graph.
        # This is crucial for stability and to ensure nodes are not isolated.
        edge_index, edge_weight = add_self_loops(
            A_undir_w.indices(), A_undir_w.values(),
            fill_value=1.0,  # Self-loops have a weight of 1
            num_nodes=self.number_of_nodes
        )

        # 3. Calculate symmetric normalization for the weighted graph: D^(-0.5) * A * D^(-0.5)
        row, col = edge_index
        # The degree is the sum of weights of incident edges.
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle nodes with degree 0

        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(
            edge_index, norm_values, (self.number_of_nodes, self.number_of_nodes)
        ).coalesce()
        print(f"    Undirected normalized matrix created with {self.A_undirected_norm_sparse._nnz()} non-zero elements.")

    def _normalize_symmetric_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Helper function to apply standard GCN normalization (D^-0.5 * A * D^-0.5)
        to any given symmetric matrix.
        """
        if matrix.numel() == 0 or matrix._nnz() == 0:
            return matrix

        # Add self-loops for stability
        edge_index, edge_weight = add_self_loops(
            matrix.indices(), matrix.values(),
            fill_value=1.0, num_nodes=self.number_of_nodes
        )

        # Calculate symmetric normalization
        row, col = edge_index
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return torch.sparse_coo_tensor(edge_index, norm_values, matrix.shape).coalesce()

    def _calculate_single_propagation_matrix(self, A_w_torch_sparse: torch.Tensor) -> torch.Tensor:
        """
        Calculates the propagation matrix mathcal{A} = sqrt(S^2 + K^2 + epsilon) + I
        using sparse tensor operations and an optimized formula.
        The formula sqrt(S^2 + K^2) is equivalent to sqrt(0.5 * (A_n^2 + (A_n^T)^2)).
        """

        def print_sparse_info(tensor: torch.Tensor, name: str, n_val_debug: Optional[int] = None):
            """Helper for debugging sparse tensor properties."""
            prefix = f"    DEBUG_SPARSE (n={n_val_debug if n_val_debug is not None else 'N/A'})"
            if tensor.is_sparse:
                nnz = tensor._nnz()
                shape = tensor.shape
                mem_bytes = (2 * nnz * 8) + (nnz * 4)  # 2x long for indices, 1x float for values
                mem_mb = mem_bytes / (1024 * 1024)
                print(
                    f"{prefix} [{name}]: shape={shape}, nnz={nnz}, device={tensor.device}, estimated_mem={mem_mb:.3f} MB")
            else:
                mem_bytes = tensor.numel() * tensor.element_size()
                mem_mb = mem_bytes / (1024 * 1024)
                print(
                    f"{prefix} [{name}]: shape={tensor.shape}, device={tensor.device} (Dense), estimated_mem={mem_mb:.3f} MB")

        current_n_val = self.n_value

        if self.number_of_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            size = (self.number_of_nodes, self.number_of_nodes)
            return torch.sparse_coo_tensor(empty_indices, empty_values, size).coalesce()

        dev = A_w_torch_sparse.device
        num_nodes = self.number_of_nodes

        # Step 1: Row-normalize A_w to get A_n
        row_sum = torch.sparse.sum(A_w_torch_sparse, dim=1).to_dense()
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=dev)
        non_zero_degrees_mask = row_sum != 0
        if torch.any(non_zero_degrees_mask):
            D_inv_diag_vals[non_zero_degrees_mask] = 1.0 / row_sum[non_zero_degrees_mask]
        del row_sum

        A_w_indices = A_w_torch_sparse.indices()
        A_w_values = A_w_torch_sparse.values()
        scaled_values = A_w_values * D_inv_diag_vals[A_w_indices[0]]
        A_n_sparse = torch.sparse_coo_tensor(A_w_indices, scaled_values, A_w_torch_sparse.size()).coalesce()
        del D_inv_diag_vals

        # Step 2: Calculate 0.5 * (A_n^2 + (A_n^T)^2)
        A_n_sq_values = A_n_sparse.values().pow(2)
        A_n_sq_sparse = torch.sparse_coo_tensor(A_n_sparse.indices(), A_n_sq_values, A_n_sparse.size()).coalesce()
        del A_n_sq_values

        A_n_sq_t_sparse = A_n_sq_sparse.t().coalesce()
        S_sq_plus_K_sq_sparse = (A_n_sq_sparse + A_n_sq_t_sparse).coalesce()
        S_sq_plus_K_sq_sparse = torch.sparse_coo_tensor(
            S_sq_plus_K_sq_sparse.indices(),
            S_sq_plus_K_sq_sparse.values() * 0.5,
            S_sq_plus_K_sq_sparse.size()
        ).coalesce()
        del A_n_sq_sparse, A_n_sq_t_sparse, A_n_sparse

        # Step 3: Final calculation: sqrt(...) + I
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()
        del S_sq_plus_K_sq_sparse, mathcal_A_base_values

        identity_sparse = self._sparse_identity(num_nodes, device=dev)
        mathcal_A_with_self_loops_sparse = (mathcal_A_base_sparse + identity_sparse).coalesce()
        del mathcal_A_base_sparse, identity_sparse

        return mathcal_A_with_self_loops_sparse

    def _create_propagation_matrices(self):
        """Computes the mathcal_A_out and mathcal_A_in propagation matrices sparsely."""
        print(f"  Creating mathcal_A_out for n={self.n_value}...")
        self.mathcal_A_out = self._calculate_single_propagation_matrix(self.A_out_w)
        gc.collect()
        print(f"  Creating mathcal_A_in for n={self.n_value}...")
        self.mathcal_A_in = self._calculate_single_propagation_matrix(self.A_in_w)
        gc.collect()

    def split_edges_by_homophily(self, labels: torch.Tensor):
        """
        Splits the raw weighted directed edge matrices (A_out_w, A_in_w) into
        homophilous and heterophilous components based on the provided node labels.
        This method should be called after the graph is initialized and labels are available.
        """
        if self.number_of_nodes == 0 or self.A_out_w._nnz() == 0:
            print("  Graph has no nodes or edges, skipping homophily split.")
            return

        print(f"  Splitting {self.A_out_w._nnz()} directed edges by homophily...")

        # --- Split Outgoing Edges ---
        out_indices = self.A_out_w.indices()
        out_weights = self.A_out_w.values()
        source_nodes, target_nodes = out_indices[0], out_indices[1]

        source_labels = labels[source_nodes]
        target_labels = labels[target_nodes]

        homo_mask_out = (source_labels == target_labels)
        hetero_mask_out = ~homo_mask_out

        A_out_w_homo = torch.sparse_coo_tensor(
            out_indices[:, homo_mask_out], out_weights[homo_mask_out], self.A_out_w.shape
        ).coalesce()
        A_out_w_hetero = torch.sparse_coo_tensor(
            out_indices[:, hetero_mask_out], out_weights[hetero_mask_out], self.A_out_w.shape
        ).coalesce()

        # --- Split Incoming Edges (by transposing the outgoing splits) ---
        A_in_w_homo = A_out_w_homo.t().coalesce()
        A_in_w_hetero = A_out_w_hetero.t().coalesce()

        # --- Create the undirected homophilic and heterophilic adjacency matrices ---
        # These represent the pure "sameness" and "differentness" connections.
        self.A_homo_w = (A_in_w_homo + A_out_w_homo).coalesce()
        self.A_hetero_w = (A_in_w_hetero + A_out_w_hetero).coalesce()

        # --- NEW: Normalize these symmetric matrices for use in the GNN ---
        print("    Normalizing homophilic and heterophilic matrices...")
        self.A_homo_norm = self._normalize_symmetric_matrix(self.A_homo_w)
        self.A_hetero_norm = self._normalize_symmetric_matrix(self.A_hetero_w)

        print(f"    - Undirected Homophilous Edges: {self.A_homo_w._nnz()}")
        print(f"    - Undirected Heterophilous Edges: {self.A_hetero_w._nnz()}")

    def create_subgraph_data_for_model(self, model_type: str,
                                       full_features: torch.Tensor, full_labels: torch.Tensor,
                                       node_subset: torch.Tensor) -> 'Data':
        """
        Creates a valid, self-contained PyG Data object for a subgraph of nodes.
        This is the critical fix for clustered training. It re-indexes edges.
        """
        sub_x = full_features[node_subset]
        sub_y = full_labels[node_subset]

        data_dict = {'x': sub_x, 'y': sub_y, 'original_indices': node_subset}

        # Use torch_geometric.utils.subgraph to get re-indexed edges for the subset of nodes
        if model_type == 'directgcn':
            # Check if the homophily-aware paths have been created
            if self.A_homo_w is not None and self.A_hetero_w is not None:
                # --- FIX: Use the correct, processed matrices for propagation ---
                path_matrices = {
                    'mathcal_in': self.mathcal_A_in, 'mathcal_out': self.mathcal_A_out,
                    'homo_norm': self.A_homo_norm, 'hetero_norm': self.A_hetero_norm,
                    'undirected_norm': self.A_undirected_norm_sparse
                }
            else:
                # Standard paths using the processed matrices
                path_matrices = {
                    'mathcal_in': self.mathcal_A_in, 'mathcal_out': self.mathcal_A_out,
                    'undirected_norm': self.A_undirected_norm_sparse
                }

            for name, matrix in path_matrices.items():
                if matrix is not None:
                    sub_edge_index, sub_edge_weight = subgraph(
                        subset=node_subset, edge_index=matrix.indices(), edge_attr=matrix.values(),
                        relabel_nodes=True, num_nodes=self.number_of_nodes
                    )
                    data_dict[f'edge_index_{name}'] = sub_edge_index
                    data_dict[f'edge_weight_{name}'] = sub_edge_weight
        elif model_type == 'rgcn':
            # --- FIX: Correctly populate data_dict for RGCN subgraphs ---
            sub_edge_index_out, _ = subgraph(node_subset, self.A_out_w.indices(), relabel_nodes=True,
                                             num_nodes=self.number_of_nodes)
            sub_edge_index_in, _ = subgraph(node_subset, self.A_in_w.indices(), relabel_nodes=True,
                                            num_nodes=self.number_of_nodes)
            device = sub_edge_index_out.device
            edge_type_out = torch.zeros(sub_edge_index_out.size(1), dtype=torch.long, device=device)
            edge_type_in = torch.ones(sub_edge_index_in.size(1), dtype=torch.long, device=device)
            data_dict['edge_index'] = torch.cat([sub_edge_index_out, sub_edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in])
        elif model_type == 'tongdigcn':
            # --- FIX: Correctly populate data_dict for TongDiGCN subgraphs ---
            sub_edge_index_out, sub_edge_weight_out = subgraph(node_subset, self.A_out_w.indices(),
                                                               self.A_out_w.values(), relabel_nodes=True,
                                                               num_nodes=self.number_of_nodes)
            sub_edge_index_in, _ = subgraph(node_subset, self.A_in_w.indices(), self.A_in_w.values(),
                                            relabel_nodes=True, num_nodes=self.number_of_nodes)
            data_dict['edge_index'] = sub_edge_index_out
            data_dict['edge_index_backward'] = sub_edge_index_in
            data_dict['edge_attr'] = sub_edge_weight_out  # For the forward pass GCN
        else:
            # --- FIX: Add a default case for all other standard GNNs (GCN, GAT, etc.) ---
            sub_edge_index, sub_edge_weight = subgraph(node_subset, self.A_undirected_norm_sparse.indices(),
                                                       self.A_undirected_norm_sparse.values(), relabel_nodes=True,
                                                       num_nodes=self.number_of_nodes)
            data_dict['edge_index'] = sub_edge_index
            data_dict['edge_attr'] = sub_edge_weight

        return Data.from_dict(data_dict)