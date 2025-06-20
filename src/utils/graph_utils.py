# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 7.0 (Propagation matrices mathcal_A_out/in computed sparsely, fai/fao removed)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Dict, Tuple, Any, Optional

import numpy as np  # Ensure numpy is imported
import torch
# dense_to_sparse is still used by PyG internally, but we won't use it to convert our main matrices
# from dense to sparse anymore, as they will be computed sparsely.
# from torch_geometric.utils import dense_to_sparse


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
                    # print( # Commented out verbose warning
                    #     f"    [Graph Constructor WARNING]: Non-integer index type in edge_tuple: {edge_tuple} (types: {type(edge_tuple[0])}, {type(edge_tuple[1])}). Skipping for index collection.")
                    continue
                all_integer_indices.add(int(edge_tuple[0]))
                all_integer_indices.add(int(edge_tuple[1]))

        if not all_integer_indices:
            self.number_of_nodes = 0
            self.edges = []
            self.number_of_edges = 0
            return

        max_idx = -1
        # Ensure indices are non-negative
        all_integer_indices = {idx for idx in all_integer_indices if isinstance(idx, (int, np.integer)) and idx >= 0}

        if not all_integer_indices: # Re-check after filtering negatives
             self.number_of_nodes = 0
             self.edges = []
             self.number_of_edges = 0
             return

        max_idx = max(all_integer_indices)

        self.number_of_nodes = max_idx + 1

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                # Assign a placeholder name if no name is provided for an index
                node_name = f"__NODE_{i}__"
            temp_idx_to_node_name[i] = str(node_name)

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}
        # Ensure node_sequences is populated for tasks like closest_aa
        self.node_sequences = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]


        self.edges = []
        for i, edge_tuple in enumerate(self.original_edges):
            if len(edge_tuple) < 2:
                # print( # Commented out verbose warning
                # f"    [Graph Constructor WARNING]: Edge {i}: Malformed edge_tuple (len < 2): {edge_tuple}. Skipping.")
                continue

            s_idx_orig, t_idx_orig = edge_tuple[0], edge_tuple[1]
            valid_s_type = isinstance(s_idx_orig, (int, np.integer))
            valid_t_type = isinstance(t_idx_orig, (int, np.integer))

            if not (valid_s_type and valid_t_type):
                 # print( # Commented out verbose warning
                 # f"    [Graph Constructor WARNING]: Edge {i}: Non-integer indices ({s_idx_orig}, {t_idx_orig}). Skipping.")
                 continue

            s_idx = int(s_idx_orig)
            t_idx = int(t_idx_orig)

            valid_s_bound = (0 <= s_idx < self.number_of_nodes)
            valid_t_bound = (0 <= t_idx < self.number_of_nodes)

            if not (valid_s_bound and valid_t_bound):
                # Consolidate warning print for invalid edges
                # print(f"    [Graph Constructor WARNING]: Edge {i} ({s_idx},{t_idx}) out of bounds for num_nodes={self.number_of_nodes}. Skipping.")
                continue

            weight = float(edge_tuple[2]) if len(edge_tuple) > 2 else 1.0
            processed_edge = (s_idx, t_idx, weight) # Only keep source, target, weight
            self.edges.append(processed_edge)

        self.number_of_edges = len(self.edges)


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        super().__init__(nodes=nodes, edges=edges)

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = None

        # A_out_w and A_in_w are sparse torch.Tensor
        self.A_out_w: torch.Tensor
        self.A_in_w: torch.Tensor

        # These will now also be sparse torch.Tensor
        self.mathcal_A_out: torch.Tensor
        self.mathcal_A_in: torch.Tensor

        # fai and fao are removed as they are not used by the current GCN model

        if self.number_of_nodes > 0:
            self._create_raw_weighted_adj_matrices_torch() # Creates sparse A_out_w, A_in_w
            self._create_propagation_matrices_for_gcn()    # Computes mathcal_A_out, mathcal_A_in sparsely
            # _create_symmetrized_magnitudes_fai_fao() is removed
        else:
            # Initialize as empty sparse tensors
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            size_empty = (0, 0) # Use (0,0) size for 0 nodes

            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            self.mathcal_A_out = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            self.mathcal_A_in = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            # fai and fao are removed

    def _create_raw_weighted_adj_matrices_torch(self):
        if self.number_of_nodes == 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            size_empty = (0, 0)
            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            return

        source_indices: List[int] = []
        target_indices: List[int] = []
        weights_list: List[float] = []

        for edge_tuple in self.edges:
            s_idx, t_idx, weight = edge_tuple[0], edge_tuple[1], edge_tuple[2]
            # Bounds check should have been done in Graph._process_constructor_inputs
            # but an additional check here is safe.
            if 0 <= s_idx < self.number_of_nodes and 0 <= t_idx < self.number_of_nodes:
                source_indices.append(s_idx)
                target_indices.append(t_idx)
                weights_list.append(weight)
            # else: # Commented out verbose warning
            # print(f"    [DirectedNgramGraph WARNING]: Edge ({s_idx},{t_idx}) out of bounds for num_nodes={self.number_of_nodes} during sparse adj matrix creation. Skipping.")


        size = (self.number_of_nodes, self.number_of_nodes)

        if not source_indices: # No valid edges
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size)
        else:
            edge_indices_tensor = torch.tensor([source_indices, target_indices], dtype=torch.long)
            edge_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
            self.A_out_w = torch.sparse_coo_tensor(edge_indices_tensor, edge_weights_tensor, size).coalesce()

        self.A_in_w = self.A_out_w.t().coalesce()
        # print(f"  [DirectedNgramGraph Info] Sparse A_out_w created. Non-zero elements: {self.A_out_w._nnz()}") # Optional

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch_sparse: torch.Tensor) -> torch.Tensor:
        """
        Calculates the propagation matrix mathcal{A} = sqrt(S^2 + K^2) + I
        using sparse tensor operations.
        Input A_w_torch_sparse is expected to be a sparse COO tensor.
        """
        if self.number_of_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0 and A_w_torch_sparse.shape[0] == 0) or (not A_w_torch_sparse.is_sparse and A_w_torch_sparse.shape[0] == 0) :
             # Return empty sparse tensor of correct shape (N, N) for 0 nodes or empty input
             empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device if A_w_torch_sparse.is_sparse else 'cpu')
             empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device if A_w_torch_sparse.is_sparse else 'cpu')
             size = (self.number_of_nodes, self.number_of_nodes)
             return torch.sparse_coo_tensor(empty_indices, empty_values, size).coalesce()


        dev = A_w_torch_sparse.device # Get device from the sparse tensor
        num_nodes = self.number_of_nodes

        # 1. Calculate row sums (degrees) - Result is dense
        # Use torch.sparse.sum for sparse tensor sum
        row_sum = torch.sparse.sum(A_w_torch_sparse, dim=1).to_dense() # .to_dense() is needed here as sparse.sum returns sparse

        # 2. Calculate D_inv_diag_vals (1 / row_sum) - Result is dense
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=dev)
        non_zero_degrees_mask = row_sum != 0
        if torch.any(non_zero_degrees_mask):
            D_inv_diag_vals[non_zero_degrees_mask] = 1.0 / row_sum[non_zero_degrees_mask]

        # 3. Create sparse diagonal matrix D_inv_sparse from D_inv_diag_vals
        # Indices for a diagonal matrix are [[0,1,2...], [0,1,2...]]
        diag_indices = torch.arange(num_nodes, device=dev).unsqueeze(0).repeat(2, 1)
        D_inv_sparse = torch.sparse_coo_tensor(diag_indices, D_inv_diag_vals, (num_nodes, num_nodes)).coalesce()

        # 4. Compute A_n = D_inv_sparse @ A_w_torch_sparse (Sparse matrix multiplication)
        A_n_sparse = torch.sparse.mm(D_inv_sparse, A_w_torch_sparse).coalesce()

        # 5. Compute S = (A_n + A_n.t()) / 2.0 (Sparse addition and transpose)
        S_sparse = torch.sparse.add(A_n_sparse, A_n_sparse.t().coalesce()).coalesce()
        # Element-wise division by 2.0 on sparse tensor values
        S_sparse = torch.sparse_coo_tensor(S_sparse.indices(), S_sparse.values() / 2.0, S_sparse.size()).coalesce()


        # 6. Compute K = (A_n - A_n.t()) / 2.0 (Sparse subtraction and transpose)
        # Sparse subtraction is sparse.add with negative values
        K_sparse = torch.sparse.add(A_n_sparse, A_n_sparse.t().coalesce(), alpha=-1.0).coalesce()
        # Element-wise division by 2.0 on sparse tensor values
        K_sparse = torch.sparse_coo_tensor(K_sparse.indices(), K_sparse.values() / 2.0, K_sparse.size()).coalesce()


        # 7. Compute mathcal_A_base = sqrt(S^2 + K^2 + epsilon) element-wise
        # This is the most complex part with sparse tensors.
        # S^2 (element-wise) and K^2 (element-wise) can be computed by squaring values.
        S_sq_sparse = torch.sparse_coo_tensor(S_sparse.indices(), S_sparse.values().pow(2), S_sparse.size()).coalesce()
        K_sq_sparse = torch.sparse_coo_tensor(K_sparse.indices(), K_sparse.values().pow(2), K_sparse.size()).coalesce()

        # Element-wise addition of S_sq_sparse and K_sq_sparse.
        # torch.sparse.add performs matrix addition, but for COO format,
        # if indices are the same, it sums values. If indices are different,
        # it includes both. This effectively computes the element-wise sum
        # of the non-zero elements and takes the union of sparsity patterns.
        S_sq_plus_K_sq_sparse = torch.sparse.add(S_sq_sparse, K_sq_sparse).coalesce()

        # Apply sqrt element-wise to the values of the resulting sparse tensor
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values, S_sq_plus_K_sq_sparse.size()).coalesce()

        # 8. Add sparse identity matrix: mathcal_A = mathcal_A_base + I
        identity_indices = torch.arange(num_nodes, device=dev).unsqueeze(0).repeat(2, 1)
        identity_values = torch.ones(num_nodes, device=dev, dtype=torch.float32)
        identity_sparse = torch.sparse_coo_tensor(identity_indices, identity_values, (num_nodes, num_nodes)).coalesce()

        # Sparse addition of mathcal_A_base_sparse and identity_sparse
        # This will add 1.0 to the diagonal elements that are already present in mathcal_A_base_sparse
        # and add the diagonal elements (1.0) where they are not present.
        mathcal_A_with_self_loops_sparse = torch.sparse.add(mathcal_A_base_sparse, identity_sparse).coalesce()

        return mathcal_A_with_self_loops_sparse

    def _create_propagation_matrices_for_gcn(self):
        """
        Computes the mathcal_A_out and mathcal_A_in propagation matrices sparsely.
        """
        if self.number_of_nodes == 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            size_empty = (0, 0)
            self.mathcal_A_out = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            self.mathcal_A_in = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
            return

        # _calculate_single_propagation_matrix_for_gcn now returns a sparse tensor
        self.mathcal_A_out = self._calculate_single_propagation_matrix_for_gcn(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix_for_gcn(self.A_in_w)

    # _create_symmetrized_magnitudes_fai_fao is removed

    # get_fai_sparse and get_fao_sparse are removed


# --- Helper function for creating sparse identity matrix (can be used internally) ---
def _sparse_identity(size: int, device: torch.device) -> torch.Tensor:
    """Creates a sparse identity matrix of given size."""
    if size <= 0:
        empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty(0, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(empty_indices, empty_values, (size, size)).coalesce()

    indices = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(size, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
