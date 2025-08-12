# ==============================================================================
# MODULE: data_builders/graph.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 11.0 (Corrected all identified errors and implemented review suggestions)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import os
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


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


class DirectedGraph:
    """
    A helper class containing matrix calculation logic, primarily for the
    benchmarking suite which uses standard PyG Data objects.
    """

    def __init__(self):
        pass

    def _calculate_single_propagation_matrix(self, A_w_torch_sparse: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Calculates the propagation matrix mathcal{A} = sqrt(S^2 + K^2 + epsilon) + I
        This logic is replicated from the main GraphBuilder for benchmark compatibility.
        """
        if num_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            return torch.sparse_coo_tensor(empty_indices, empty_values, (num_nodes, num_nodes)).coalesce()

        dev = A_w_torch_sparse.device
        row_sum = torch.sparse.sum(A_w_torch_sparse, dim=1).to_dense()
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=dev)
        non_zero_degrees_mask = row_sum != 0
        if torch.any(non_zero_degrees_mask):
            D_inv_diag_vals[non_zero_degrees_mask] = 1.0 / row_sum[non_zero_degrees_mask]

        A_w_indices = A_w_torch_sparse.indices()
        A_w_values = A_w_torch_sparse.values()
        scaled_values = A_w_values * D_inv_diag_vals[A_w_indices[0]]
        A_n_sparse = torch.sparse_coo_tensor(A_w_indices, scaled_values, A_w_torch_sparse.size()).coalesce()

        A_n_sq_values = A_n_sparse.values().pow(2)
        A_n_sq_sparse = torch.sparse_coo_tensor(A_n_sparse.indices(), A_n_sq_values, A_n_sparse.size()).coalesce()
        A_n_sq_t_sparse = A_n_sq_sparse.t().coalesce()
        S_sq_plus_K_sq_sparse = (A_n_sq_sparse + A_n_sq_t_sparse).coalesce()
        S_sq_plus_K_sq_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(),
                                                        S_sq_plus_K_sq_sparse.values() * 0.5,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()

        epsilon_tensor = torch.tensor(1e-9, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()

        identity_sparse = Graph._sparse_identity(num_nodes, device=dev)
        return (mathcal_A_base_sparse + identity_sparse).coalesce()

    def _normalize_symmetric_matrix(self, matrix: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Helper to apply GCN normalization to a symmetric matrix."""  # noqa
        if matrix.numel() == 0 or matrix._nnz() == 0: return matrix
        edge_index, edge_weight = add_self_loops(matrix.indices(), matrix.values(), fill_value=1.0,
                                                 num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, norm_values, matrix.shape).coalesce()

    def _preprocess_for_custom_models(self, data: Data, use_homo_hetero_paths: bool) -> Data:
        """Prepares a data object with all necessary edge indices for custom models."""
        base_edge_weight = data.edge_attr if hasattr(data,
                                                     'edge_attr') and data.edge_attr is not None else torch.ones(
            data.edge_index.shape[1], device=data.edge_index.device)

        if use_homo_hetero_paths:
            edge_index = data.edge_index
            source_nodes, target_nodes = edge_index[0], edge_index[1]
            source_labels = data.y[source_nodes]
            target_labels = data.y[target_nodes]
            homo_mask = (source_labels == target_labels)
            hetero_mask = ~homo_mask

            edge_index_out_homo = edge_index[:, homo_mask]
            edge_weight_out_homo = base_edge_weight[homo_mask]
            A_out_w_homo = torch.sparse_coo_tensor(edge_index_out_homo, edge_weight_out_homo,
                                                   (data.num_nodes, data.num_nodes)).coalesce()
            A_homo_w = (A_out_w_homo + A_out_w_homo.t()).coalesce()
            data.edge_index_homo_norm, data.edge_weight_homo_norm = self._normalize_symmetric_matrix(A_homo_w,
                                                                                                    data.num_nodes).coalesce().indices(), self._normalize_symmetric_matrix(
                A_homo_w, data.num_nodes).coalesce().values()

            edge_index_out_hetero = edge_index[:, hetero_mask]
            edge_weight_out_hetero = base_edge_weight[hetero_mask]
            A_out_w_hetero = torch.sparse_coo_tensor(edge_index_out_hetero, edge_weight_out_hetero,
                                                     (data.num_nodes, data.num_nodes)).coalesce()
            A_hetero_w = (A_out_w_hetero + A_out_w_hetero.t()).coalesce()
            data.edge_index_hetero_norm, data.edge_weight_hetero_norm = self._normalize_symmetric_matrix(A_hetero_w,
                                                                                                        data.num_nodes).coalesce().indices(), self._normalize_symmetric_matrix(
                A_hetero_w, data.num_nodes).coalesce().values()

        A_out_w_sparse = torch.sparse_coo_tensor(data.edge_index, base_edge_weight,
                                                 (data.num_nodes, data.num_nodes)).coalesce()
        A_in_w_sparse = A_out_w_sparse.t().coalesce()
        A_undir_w = (A_out_w_sparse + A_in_w_sparse).coalesce()
        A_undirected_norm = self._normalize_symmetric_matrix(A_undir_w, data.num_nodes)
        data.edge_index_out, data.edge_weight_out = A_out_w_sparse.indices(), A_out_w_sparse.values()
        data.edge_index_in, data.edge_weight_in = A_in_w_sparse.indices(), A_in_w_sparse.values()
        data.edge_index_undirected_norm, data.edge_weight_undirected_norm = A_undirected_norm.indices(), A_undirected_norm.values()
        data.edge_index_backward = data.edge_index_in

        data.edge_attr = data.edge_weight_undirected_norm
        data.edge_index = data.edge_index_undirected_norm

        mathcal_A_out = self._calculate_single_propagation_matrix(A_out_w_sparse, data.num_nodes)
        mathcal_A_in = self._calculate_single_propagation_matrix(A_in_w_sparse, data.num_nodes)
        data.edge_index_mathcal_out, data.edge_weight_mathcal_out = mathcal_A_out.indices(), mathcal_A_out.values()
        data.edge_index_mathcal_in, data.edge_weight_mathcal_in = mathcal_A_in.indices(), mathcal_A_in.values()
        return data


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any],
                 edge_file_path: Optional[str] = None,
                 epsilon_propagation: float = 1e-9, n_value: Optional[int] = None):

        super().__init__(nodes=nodes, edges=[])

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = n_value

        # --- FIX: Initialize all matrix attributes to prevent AttributeError ---
        self.A_out_w: Optional[torch.Tensor] = None
        self.A_in_w: Optional[torch.Tensor] = None
        self.A_undirected_norm_sparse: Optional[torch.Tensor] = None
        self.mathcal_A_out: Optional[torch.Tensor] = None
        self.mathcal_A_in: Optional[torch.Tensor] = None
        self.A_homo_w: Optional[torch.Tensor] = None
        self.A_hetero_w: Optional[torch.Tensor] = None
        self.A_homo_norm: Optional[torch.Tensor] = None
        self.A_hetero_norm: Optional[torch.Tensor] = None

        if self.number_of_nodes > 0 and edge_file_path and os.path.exists(edge_file_path):
            print(f"    Loading edges from {os.path.basename(edge_file_path)}...")
            try:
                # --- FIX: Avoid pd.read_parquet to prevent loading the entire edge file into memory. ---
                # Instead, iterate over the file in chunks using pyarrow for scalability.
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(edge_file_path)
                source_chunks, target_chunks, weight_chunks = [], [], []
                for batch in parquet_file.iter_batches(batch_size=10_000_000):
                    df_chunk = batch.to_pandas()
                    source_chunks.append(df_chunk['source'].to_numpy(dtype=np.int64))
                    target_chunks.append(df_chunk['target'].to_numpy(dtype=np.int64))
                    weight_chunks.append(df_chunk['weight'].to_numpy(dtype=np.float32))

                source_indices = np.concatenate(source_chunks)
                target_indices = np.concatenate(target_chunks)
                weights = np.concatenate(weight_chunks)
                del source_chunks, target_chunks, weight_chunks
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
        self.A_homo_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_hetero_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_homo_norm = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_hetero_norm = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)

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
        """
        print(f"  Creating undirected normalized adjacency matrix for n={self.n_value}...")
        if self.number_of_nodes == 0 or self.A_out_w is None or self.A_in_w is None:
            return

        A_undir_w = (self.A_out_w + self.A_in_w).coalesce()

        edge_index, edge_weight = add_self_loops(
            A_undir_w.indices(), A_undir_w.values(),
            fill_value=1.0,
            num_nodes=self.number_of_nodes
        )

        row, col = edge_index
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(
            edge_index, norm_values, (self.number_of_nodes, self.number_of_nodes)
        ).coalesce()
        print(
            f"    Undirected normalized matrix created with {self.A_undirected_norm_sparse._nnz()} non-zero elements.")

    def _normalize_symmetric_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Helper function to apply standard GCN normalization (D^-0.5 * A * D^-0.5)
        to any given symmetric matrix.
        """
        if matrix.numel() == 0 or matrix._nnz() == 0:
            return matrix

        edge_index, edge_weight = add_self_loops(
            matrix.indices(), matrix.values(),
            fill_value=1.0, num_nodes=self.number_of_nodes
        )

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
        """
        if self.number_of_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            size = (self.number_of_nodes, self.number_of_nodes)
            return torch.sparse_coo_tensor(empty_indices, empty_values, size).coalesce()

        dev = A_w_torch_sparse.device
        num_nodes = self.number_of_nodes

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
        gc.collect()

        # Step 3: Final calculation: sqrt(...) + I
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()
        del S_sq_plus_K_sq_sparse, mathcal_A_base_values
        gc.collect()

        identity_sparse = self._sparse_identity(num_nodes, device=dev)
        mathcal_A_with_self_loops_sparse = (mathcal_A_base_sparse + identity_sparse).coalesce()
        del mathcal_A_base_sparse, identity_sparse
        gc.collect()

        return mathcal_A_with_self_loops_sparse

    def _create_propagation_matrices(self):
        """Computes the mathcal_A_out and mathcal_A_in propagation matrices sparsely."""
        print(f"  Creating mathcal_A_out for n={self.n_value}...")
        if self.A_out_w is not None:
            self.mathcal_A_out = self._calculate_single_propagation_matrix(self.A_out_w)
        gc.collect()
        print(f"  Creating mathcal_A_in for n={self.n_value}...")
        if self.A_in_w is not None:
            self.mathcal_A_in = self._calculate_single_propagation_matrix(self.A_in_w)
        gc.collect()

    def split_edges_by_homophily(self, labels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits the raw weighted directed edge matrices (A_out_w, A_in_w) into
        homophilous and heterophilous components based on the provided node labels.
        This method is functional and returns the normalized matrices instead
        of modifying the object state, which is a safer design pattern.

        Returns:
            A tuple of (A_homo_norm, A_hetero_norm) or None if the graph is empty.
        """
        # --- ANTICIPATORY DEBUGGING: Check for invalid state before proceeding ---
        if self.number_of_nodes == 0 or self.A_out_w is None or self.A_out_w._nnz() == 0:
            print("  Graph has no nodes or edges, skipping homophily split.")
            return None

        print(f"  Splitting {self.A_out_w._nnz()} directed edges by homophily...")

        edge_index, edge_weights = self.A_out_w.indices(), self.A_out_w.values()
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        source_labels, target_labels = labels[source_nodes], labels[target_nodes]

        homo_mask = (source_labels == target_labels)
        hetero_mask = ~homo_mask

        A_out_w_homo = torch.sparse_coo_tensor(edge_index[:, homo_mask], edge_weights[homo_mask], self.A_out_w.shape).coalesce()
        A_out_w_hetero = torch.sparse_coo_tensor(edge_index[:, hetero_mask], edge_weights[hetero_mask], self.A_out_w.shape).coalesce()

        A_homo_w = (A_out_w_homo + A_out_w_homo.t()).coalesce()
        A_hetero_w = (A_out_w_hetero + A_out_w_hetero.t()).coalesce()

        print("    Normalizing homophilic and heterophilic matrices...")
        A_homo_norm = self._normalize_symmetric_matrix(A_homo_w)
        A_hetero_norm = self._normalize_symmetric_matrix(A_hetero_w)

        print(f"    - Undirected Homophilous Edges: {A_homo_w._nnz()}")
        print(f"    - Undirected Heterophilous Edges: {A_hetero_w._nnz()}")
        return A_homo_norm, A_hetero_norm

    def create_subgraph_data_for_model(self, model_type: str,
                                       full_features: torch.Tensor, full_labels: torch.Tensor,
                                       node_subset: torch.Tensor,
                                       A_homo_norm: Optional[torch.Tensor] = None,
                                       A_hetero_norm: Optional[torch.Tensor] = None) -> 'Data':
        """
        Creates a valid, self-contained PyG Data object for a subgraph of
        nodes. This is the critical fix for clustered training. It re-indexes edges.
        """
        subgraph_features = full_features[node_subset]
        subgraph_labels = full_labels[node_subset] if full_labels is not None else None

        if model_type.lower() == 'directgcn':
            path_matrices = {
                'mathcal_in': self.mathcal_A_in, 'mathcal_out': self.mathcal_A_out,
                'undirected_norm': self.A_undirected_norm_sparse
            }
            # --- ANTICIPATORY DEBUGGING: Conditionally add the new matrices if they were generated. ---
            # This makes the function flexible for both heterophilic and homophilic graphs.
            if A_homo_norm is not None and A_hetero_norm is not None:
                path_matrices.update({'homo_norm': A_homo_norm, 'hetero_norm': A_hetero_norm})

            subgraph_data_dict = {'x': subgraph_features, 'y': subgraph_labels, 'original_indices': node_subset}
            for name, matrix in path_matrices.items():
                if matrix is not None:
                    edge_index, edge_weight = subgraph(node_subset, matrix.indices(), matrix.values(),
                                                       relabel_nodes=True, num_nodes=self.number_of_nodes)
                    subgraph_data_dict[f'edge_index_{name}'] = edge_index
                    subgraph_data_dict[f'edge_weight_{name}'] = edge_weight
            return Data.from_dict(subgraph_data_dict)
        else:
            # Default for other GNNs (uses undirected graph)
            edge_index, edge_weight = subgraph(node_subset, self.A_undirected_norm_sparse.indices(),
                                               self.A_undirected_norm_sparse.values(),
                                               relabel_nodes=True, num_nodes=self.number_of_nodes)
            return Data(x=subgraph_features, y=subgraph_labels, edge_index=edge_index, edge_attr=edge_weight,
                        original_indices=node_subset)