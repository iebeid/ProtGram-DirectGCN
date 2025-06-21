# src/utils/graph_utils.py
# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 7.9 (Added creation of undirected, normalized adjacency matrix)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import add_self_loops, degree


class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple]):  # nodes keys are int IDs
        self.idx_to_node_map_from_constructor = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []

        self.node_to_idx: Dict[Any, int] = {}
        self.idx_to_node: Dict[int, Any] = {}
        self.number_of_nodes: int = 0
        self.node_sequences: List[Any] = []
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
            valid_node_indices = {idx for idx in self.idx_to_node_map_from_constructor.keys() if isinstance(idx, (int, np.integer)) and idx >= 0}
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
        self.node_sequences = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]

        self.edges = self.original_edges
        self.number_of_edges = len(self.edges)


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any],
                 edge_file_path: Optional[str] = None,
                 epsilon_propagation: float = 1e-9, n_value: Optional[int] = None):

        super().__init__(nodes=nodes, edges=[])

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = n_value

        self.A_out_w: torch.Tensor
        self.A_in_w: torch.Tensor
        self.A_undirected_norm_sparse: torch.Tensor  # NEW
        self.mathcal_A_out: torch.Tensor
        self.mathcal_A_in: torch.Tensor

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
                self._create_undirected_normalized_adj_matrix(source_indices, target_indices)  # NEW
                self._create_propagation_matrices_for_gcn()

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
        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)  # NEW
        self.mathcal_A_out = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.mathcal_A_in = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)

    def _create_raw_weighted_adj_matrices_torch(self, source_indices: np.ndarray, target_indices: np.ndarray, weights: np.ndarray):
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

    def _create_undirected_normalized_adj_matrix(self, source_indices: np.ndarray, target_indices: np.ndarray):
        """
        NEW: Creates a symmetric, degree-normalized adjacency matrix from raw unique edges.
        """
        print(f"  Creating undirected normalized adjacency matrix for n={self.n_value}...")
        if self.number_of_nodes == 0:
            return

        # 1. Create undirected edge index from unique source-target pairs
        edge_pairs = np.stack([source_indices, target_indices], axis=1)
        unique_edge_pairs = np.unique(edge_pairs, axis=0)

        # Create symmetric edges
        symmetric_edges = np.concatenate([unique_edge_pairs, unique_edge_pairs[:, [1, 0]]], axis=0)
        # Get unique symmetric edges
        unique_symmetric_edges = np.unique(symmetric_edges, axis=0)

        edge_index = torch.from_numpy(unique_symmetric_edges.T).long()

        # 2. Add self-loops to prevent nodes from disappearing during propagation
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.number_of_nodes)

        # 3. Create edge weights of 1 for all edges
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        # 4. Calculate symmetric normalization: D^(-0.5) * A * D^(-0.5)
        row, col = edge_index
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle nodes with degree 0

        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(
            edge_index, norm_values, (self.number_of_nodes, self.number_of_nodes)
        ).coalesce()
        print(f"    Undirected normalized matrix created with {self.A_undirected_norm_sparse._nnz()} non-zero elements.")

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch_sparse: torch.Tensor) -> torch.Tensor:
        """
        Calculates the propagation matrix mathcal{A} = sqrt(S^2 + K^2 + epsilon) + I
        using sparse tensor operations. This version uses a memory-optimized formula.
        """

        # This method remains unchanged from v7.8
        def print_sparse_info(tensor: torch.Tensor, name: str, n_val_debug: Optional[int] = None):
            prefix = f"    DEBUG_SPARSE (n={n_val_debug if n_val_debug is not None else 'N/A'})"
            if tensor.is_sparse:
                nnz = tensor._nnz()
                shape = tensor.shape
                mem_bytes = (2 * nnz * 8) + (nnz * 4)  # 2x long for indices, 1x float for values
                mem_mb = mem_bytes / (1024 * 1024)
                print(f"{prefix} [{name}]: shape={shape}, nnz={nnz}, device={tensor.device}, estimated_mem={mem_mb:.3f} MB")
            else:
                mem_bytes = tensor.numel() * tensor.element_size()
                mem_mb = mem_bytes / (1024 * 1024)
                print(f"{prefix} [{name}]: shape={tensor.shape}, device={tensor.device} (Dense), estimated_mem={mem_mb:.3f} MB")

        current_n_val = self.n_value

        if self.number_of_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            size = (self.number_of_nodes, self.number_of_nodes)
            return torch.sparse_coo_tensor(empty_indices, empty_values, size).coalesce()

        dev = A_w_torch_sparse.device
        num_nodes = self.number_of_nodes

        print_sparse_info(A_w_torch_sparse, "A_w_input", current_n_val)

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

        print_sparse_info(A_n_sparse, "A_n", current_n_val)
        del D_inv_diag_vals

        print("    Calculating S^2+K^2 using memory-optimized formula...")
        A_n_sq_values = A_n_sparse.values().pow(2)
        A_n_sq_sparse = torch.sparse_coo_tensor(A_n_sparse.indices(), A_n_sq_values, A_n_sparse.size()).coalesce()
        print_sparse_info(A_n_sq_sparse, "A_n_squared", current_n_val)
        del A_n_sq_values

        A_n_sq_t_sparse = A_n_sq_sparse.t().coalesce()
        S_sq_plus_K_sq_sparse = (A_n_sq_sparse + A_n_sq_t_sparse).coalesce()
        S_sq_plus_K_sq_sparse = torch.sparse_coo_tensor(
            S_sq_plus_K_sq_sparse.indices(),
            S_sq_plus_K_sq_sparse.values() * 0.5,
            S_sq_plus_K_sq_sparse.size()
        ).coalesce()
        print_sparse_info(S_sq_plus_K_sq_sparse, "S_sq_plus_K_sq", current_n_val)
        del A_n_sq_sparse, A_n_sq_t_sparse, A_n_sparse

        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values, S_sq_plus_K_sq_sparse.size()).coalesce()
        print_sparse_info(mathcal_A_base_sparse, "mathcal_A_base", current_n_val)
        del S_sq_plus_K_sq_sparse, mathcal_A_base_values

        identity_sparse = _sparse_identity(num_nodes, device=dev)
        mathcal_A_with_self_loops_sparse = (mathcal_A_base_sparse + identity_sparse).coalesce()
        print_sparse_info(mathcal_A_with_self_loops_sparse, "mathcal_A_final", current_n_val)
        del mathcal_A_base_sparse, identity_sparse

        return mathcal_A_with_self_loops_sparse

    def _create_propagation_matrices_for_gcn(self):
        """Computes the mathcal_A_out and mathcal_A_in propagation matrices sparsely."""
        if self.number_of_nodes == 0:
            self._initialize_empty_matrices()
            return

        print(f"  Creating mathcal_A_out for n={self.n_value}...")
        self.mathcal_A_out = self._calculate_single_propagation_matrix_for_gcn(self.A_out_w)
        gc.collect()

        print(f"  Creating mathcal_A_in for n={self.n_value}...")
        self.mathcal_A_in = self._calculate_single_propagation_matrix_for_gcn(self.A_in_w)
        gc.collect()


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