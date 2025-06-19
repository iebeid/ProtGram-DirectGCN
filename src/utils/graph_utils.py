# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 6.4 (A_out_w and A_in_w are now sparse tensors)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Dict, Tuple, Any, Optional

import numpy as np  # Ensure numpy is imported
import torch
from torch_geometric.utils import dense_to_sparse


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
                    print(
                        f"    [Graph Constructor WARNING]: Non-integer index type in edge_tuple: {edge_tuple} (types: {type(edge_tuple[0])}, {type(edge_tuple[1])}). Skipping for index collection.")
                    continue
                all_integer_indices.add(int(edge_tuple[0]))
                all_integer_indices.add(int(edge_tuple[1]))

        if not all_integer_indices:
            self.number_of_nodes = 0
            self.edges = []
            self.number_of_edges = 0
            return

        max_idx = -1
        for idx_val in all_integer_indices:
            if not isinstance(idx_val, (int, np.integer)):
                print(f"    [Graph Constructor CRITICAL WARNING]: Non-integer index '{idx_val}' (type: {type(idx_val)}) found in all_integer_indices despite prior checks.")
                continue
            if idx_val > max_idx:
                max_idx = idx_val

        self.number_of_nodes = max_idx + 1

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                node_name = f"__NODE_{i}__"
            temp_idx_to_node_name[i] = str(node_name)

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}
        self.node_sequences = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]

        self.edges = []
        for i, edge_tuple in enumerate(self.original_edges):
            if len(edge_tuple) < 2:
                print(f"    [Graph Constructor WARNING]: Edge {i}: Malformed edge_tuple (len < 2): {edge_tuple}. Skipping.")
                continue

            s_idx_orig, t_idx_orig = edge_tuple[0], edge_tuple[1]
            valid_s_type = isinstance(s_idx_orig, (int, np.integer))
            valid_t_type = isinstance(t_idx_orig, (int, np.integer))
            s_idx = int(s_idx_orig) if valid_s_type else s_idx_orig
            t_idx = int(t_idx_orig) if valid_t_type else t_idx_orig
            valid_s_bound = (0 <= s_idx < self.number_of_nodes) if valid_s_type else False
            valid_t_bound = (0 <= t_idx < self.number_of_nodes) if valid_t_type else False

            if not (valid_s_type and valid_s_bound and valid_t_type and valid_t_bound):
                # Consolidate warning print for invalid edges
                # print(f"    [Graph Constructor WARNING]: Edge {i} ({s_idx_orig},{t_idx_orig}) invalid or out of bounds. Skipping.")
                continue

            weight = float(edge_tuple[2]) if len(edge_tuple) > 2 else 1.0
            processed_edge = (s_idx, t_idx, weight) + tuple(edge_tuple[3:])
            self.edges.append(processed_edge)

        self.number_of_edges = len(self.edges)


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        super().__init__(nodes=nodes, edges=edges)

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = None

        # A_out_w and A_in_w will now be sparse torch.Tensor
        self.A_out_w: torch.Tensor
        self.A_in_w: torch.Tensor

        # These might still be dense depending on calculation complexity
        self.mathcal_A_out: np.ndarray
        self.mathcal_A_in: np.ndarray
        self.fai: torch.Tensor
        self.fao: torch.Tensor

        if self.number_of_nodes > 0:
            self._create_raw_weighted_adj_matrices_torch() # Now creates sparse A_out_w, A_in_w
            self._create_propagation_matrices_for_gcn()    # Consumes A_out_w, A_in_w
            self._create_symmetrized_magnitudes_fai_fao()  # Consumes A_out_w, A_in_w
        else:
            # Initialize as empty sparse tensors
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, (0, 0))
            self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, (0, 0))
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0, 0)
            self.fai = torch.empty((0, 0), dtype=torch.float32)
            self.fao = torch.empty((0, 0), dtype=torch.float32)

    def _create_raw_weighted_adj_matrices_torch(self):
        if self.number_of_nodes == 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, (0, 0))
            self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, (0, 0))
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
            else:
                print(f"    [DirectedNgramGraph WARNING]: Edge ({s_idx},{t_idx}) out of bounds for num_nodes={self.number_of_nodes} during sparse adj matrix creation. Skipping.")

        if not source_indices: # No valid edges
            empty_indices = torch.empty((2, 0), dtype=torch.long)
            empty_values = torch.empty(0, dtype=torch.float32)
            size = (self.number_of_nodes, self.number_of_nodes)
            self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size)
        else:
            edge_indices_tensor = torch.tensor([source_indices, target_indices], dtype=torch.long)
            edge_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
            size = (self.number_of_nodes, self.number_of_nodes)
            self.A_out_w = torch.sparse_coo_tensor(edge_indices_tensor, edge_weights_tensor, size).coalesce()

        self.A_in_w = self.A_out_w.t().coalesce()
        # print(f"  [DirectedNgramGraph Info] Sparse A_out_w created. Non-zero elements: {self.A_out_w._nnz()}") # Optional

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch: torch.Tensor) -> np.ndarray:
        if self.number_of_nodes == 0 or (A_w_torch.is_sparse and A_w_torch._nnz() == 0 and A_w_torch.shape[0] == 0) or (not A_w_torch.is_sparse and A_w_torch.shape[0] == 0) :
            return np.array([], dtype=np.float32).reshape(0, 0)

        # Convert to dense for existing calculations.
        # Future optimization: perform these ops sparsely if possible.
        A_w_dense = A_w_torch.to_dense() if A_w_torch.is_sparse else A_w_torch
        dev = A_w_dense.device # Get device from the (potentially densified) tensor

        row_sum = A_w_dense.sum(dim=1)
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=dev)
        non_zero_degrees = row_sum != 0
        if torch.any(non_zero_degrees):
            D_inv_diag_vals[non_zero_degrees] = 1.0 / row_sum[non_zero_degrees]
        # else: D_inv_diag_vals remains zeros, which is correct

        A_n = D_inv_diag_vals.unsqueeze(1) * A_w_dense # Element-wise multiplication
        S = (A_n + A_n.t()) / 2.0
        K = (A_n - A_n.t()) / 2.0
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base = torch.sqrt(torch.square(S) + torch.square(K) + epsilon_tensor)
        identity_matrix = torch.eye(self.number_of_nodes, dtype=torch.float32, device=dev)
        mathcal_A_with_self_loops = mathcal_A_base + identity_matrix
        return mathcal_A_with_self_loops.cpu().numpy()

    def _create_propagation_matrices_for_gcn(self):
        if self.number_of_nodes == 0:
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0,0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0,0)
            return
        self.mathcal_A_out = self._calculate_single_propagation_matrix_for_gcn(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix_for_gcn(self.A_in_w)

    def _create_symmetrized_magnitudes_fai_fao(self):
        if self.number_of_nodes == 0:
            self.fai = torch.empty((0, 0), dtype=torch.float32)
            self.fao = torch.empty((0, 0), dtype=torch.float32)
            return

        # Convert sparse to dense for these calculations for now
        A_out_w_dense = self.A_out_w.to_dense() if self.A_out_w.is_sparse else self.A_out_w
        A_in_w_dense = self.A_in_w.to_dense() if self.A_in_w.is_sparse else self.A_in_w
        dev = A_out_w_dense.device # Get device from the (potentially densified) tensor

        identity = torch.eye(self.number_of_nodes, device=dev, dtype=torch.float32)
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)

        A_out_w_sl = A_out_w_dense + identity
        D_out_sl_d = A_out_w_sl.sum(dim=1)
        D_out_sl_inv_d = torch.zeros_like(D_out_sl_d, device=dev)
        non_zero_D_out = D_out_sl_d != 0
        if torch.any(non_zero_D_out):
            D_out_sl_inv_d[non_zero_D_out] = 1.0 / D_out_sl_d[non_zero_D_out]

        A_out_n_sl = D_out_sl_inv_d.unsqueeze(1) * A_out_w_sl
        our = (A_out_n_sl + A_out_n_sl.t()) / 2.0
        oui = (A_out_n_sl - A_out_n_sl.t()) / 2.0
        self.fao = torch.sqrt(our.pow(2) + oui.pow(2) + epsilon_tensor)

        A_in_w_sl = A_in_w_dense + identity # Use A_in_w_dense here
        D_in_sl_d = A_in_w_sl.sum(dim=1)
        D_in_sl_inv_d = torch.zeros_like(D_in_sl_d, device=dev)
        non_zero_D_in = D_in_sl_d != 0
        if torch.any(non_zero_D_in):
            D_in_sl_inv_d[non_zero_D_in] = 1.0 / D_in_sl_d[non_zero_D_in]

        A_in_n_sl = D_in_sl_inv_d.unsqueeze(1) * A_in_w_sl
        ir = (A_in_n_sl + A_in_n_sl.t()) / 2.0
        ii = (A_in_n_sl - A_in_n_sl.t()) / 2.0
        self.fai = torch.sqrt(ir.pow(2) + ii.pow(2) + epsilon_tensor)

    def get_fai_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fai') or self.fai is None:
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else: # No nodes, return empty sparse representation
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        # Ensure self.fai is initialized even if _create_symmetrized_magnitudes_fai_fao was called
        # for a 0-node graph and then nodes were added (though this shouldn't happen with current flow)
        if self.fai is None: # Should ideally not be None if number_of_nodes > 0 after init
             if self.number_of_nodes > 0: # Attempt to create if somehow missed
                self._create_symmetrized_magnitudes_fai_fao()
             else: # Still 0 nodes
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        if self.fai.numel() == 0 and self.number_of_nodes > 0:
            print(f"Warning: FAI tensor is empty for a graph with {self.number_of_nodes} nodes. Returning empty sparse tensor.")
            # Determine device from A_out_w if possible, else default to CPU
            device_to_use = self.A_out_w.device if hasattr(self, 'A_out_w') and (self.A_out_w.is_sparse or self.A_out_w.numel() > 0) else 'cpu'
            return torch.empty((2,0), dtype=torch.long, device=device_to_use), \
                   torch.empty(0, dtype=torch.float32, device=device_to_use)

        if self.fai.numel() == 0 and self.number_of_nodes == 0: # Correct handling for 0-node graph after init
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        return dense_to_sparse(self.fai)

    def get_fao_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fao') or self.fao is None:
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else: # No nodes
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        if self.fao is None:
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        if self.fao.numel() == 0 and self.number_of_nodes > 0:
            print(f"Warning: FAO tensor is empty for a graph with {self.number_of_nodes} nodes. Returning empty sparse tensor.")
            device_to_use = self.A_out_w.device if hasattr(self, 'A_out_w') and (self.A_out_w.is_sparse or self.A_out_w.numel() > 0) else 'cpu'
            return torch.empty((2,0), dtype=torch.long, device=device_to_use), \
                   torch.empty(0, dtype=torch.float32, device=device_to_use)

        if self.fao.numel() == 0 and self.number_of_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        return dense_to_sparse(self.fao)