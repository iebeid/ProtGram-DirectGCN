# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 6.2 (Enhanced debugging in _process_constructor_inputs)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Dict, Tuple, Any, Optional

import numpy as np
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
        self.edges: List[Tuple] = []  # Initialize self.edges here
        self.number_of_edges: int = 0  # Initialize self.number_of_edges here

        self._process_constructor_inputs()

    def _process_constructor_inputs(self):
        """
        Processes the nodes and edges passed to the constructor.
        Assumes `nodes` is a map from integer index to node name (e.g., n-gram string).
        Assumes `edges` contains tuples where the first two elements are integer indices.
        """
        print("  [DEBUG Graph._process_constructor_inputs] Starting processing...")
        print(f"    Input idx_to_node_map_from_constructor (len): {len(self.idx_to_node_map_from_constructor)}")
        print(f"    Input original_edges (len): {len(self.original_edges)}")

        if not self.idx_to_node_map_from_constructor and not self.original_edges:
            self.number_of_nodes = 0
            self.edges = []  # Already initialized, but good to be explicit
            self.number_of_edges = 0
            print("  [DEBUG Graph._process_constructor_inputs] No nodes or edges from constructor. Exiting early.")
            return

        all_integer_indices = set()
        if self.idx_to_node_map_from_constructor:
            all_integer_indices.update(self.idx_to_node_map_from_constructor.keys())

        for edge_tuple in self.original_edges:
            if len(edge_tuple) >= 2:
                if not isinstance(edge_tuple[0], int) or not isinstance(edge_tuple[1], int):
                    print(f"    [DEBUG Graph._process_constructor_inputs] WARNING: Non-integer index in edge_tuple: {edge_tuple}. Skipping for index collection.")
                    continue
                all_integer_indices.add(edge_tuple[0])
                all_integer_indices.add(edge_tuple[1])

        print(f"    [DEBUG Graph._process_constructor_inputs] all_integer_indices (len): {len(all_integer_indices)}")
        if all_integer_indices:
            print(f"    [DEBUG Graph._process_constructor_inputs] all_integer_indices (sample): {list(all_integer_indices)[:10] if len(all_integer_indices) > 10 else list(all_integer_indices)}")

        if not all_integer_indices:
            self.number_of_nodes = 0
            self.edges = []  # Already initialized
            self.number_of_edges = 0
            print("  [DEBUG Graph._process_constructor_inputs] all_integer_indices is empty. Setting num_nodes to 0.")
            return

        max_idx = -1
        for idx_val in all_integer_indices:
            if not isinstance(idx_val, int):
                # This should ideally not happen if the above check for edges worked
                print(f"    [DEBUG Graph._process_constructor_inputs] CRITICAL WARNING: Non-integer index '{idx_val}' found in all_integer_indices despite prior checks.")
                continue  # Skip non-integer if it somehow got through
            if idx_val > max_idx:
                max_idx = idx_val

        self.number_of_nodes = max_idx + 1
        print(f"  [DEBUG Graph._process_constructor_inputs] max_idx from all_integer_indices: {max_idx}, self.number_of_nodes SET TO: {self.number_of_nodes}")

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                node_name = f"__NODE_{i}__"
            temp_idx_to_node_name[i] = str(node_name)

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}
        self.node_sequences = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]

        print(f"  [DEBUG Graph._process_constructor_inputs] Populated self.idx_to_node (len: {len(self.idx_to_node)}). First 5: {dict(list(self.idx_to_node.items())[:5])}")
        print(f"  [DEBUG Graph._process_constructor_inputs] Populated self.node_sequences (len: {len(self.node_sequences)}). First 5: {self.node_sequences[:5]}")

        # Re-initialize self.edges before populating
        self.edges = []
        print(f"  [DEBUG Graph._process_constructor_inputs] Iterating self.original_edges (len: {len(self.original_edges)}) to populate self.edges. self.number_of_nodes for check: {self.number_of_nodes}")

        edge_pass_count = 0
        edge_fail_s_idx_type_count = 0
        edge_fail_s_idx_bound_count = 0
        edge_fail_t_idx_type_count = 0
        edge_fail_t_idx_bound_count = 0

        for i, edge_tuple in enumerate(self.original_edges):
            if len(edge_tuple) < 2:
                print(f"    [DEBUG Graph._process_constructor_inputs] Edge {i}: Malformed edge_tuple (len < 2): {edge_tuple}. Skipping.")
                continue

            s_idx, t_idx = edge_tuple[0], edge_tuple[1]

            valid_s_type = isinstance(s_idx, int)
            valid_s_bound = (0 <= s_idx < self.number_of_nodes) if valid_s_type else False
            valid_t_type = isinstance(t_idx, int)
            valid_t_bound = (0 <= t_idx < self.number_of_nodes) if valid_t_type else False

            if i < 5 or not (valid_s_type and valid_s_bound and valid_t_type and valid_t_bound):  # Print first 5 and all failing
                print(
                    f"    [DEBUG Graph._process_constructor_inputs] Edge {i}: ({s_idx}, {t_idx}) | s_ok: {valid_s_type and valid_s_bound} (type:{valid_s_type}, bound:{valid_s_bound}) | t_ok: {valid_t_type and valid_t_bound} (type:{valid_t_type}, bound:{valid_t_bound})")

            if not valid_s_type:
                edge_fail_s_idx_type_count += 1
                continue
            if not valid_s_bound:
                edge_fail_s_idx_bound_count += 1
                continue
            if not valid_t_type:
                edge_fail_t_idx_type_count += 1
                continue
            if not valid_t_bound:
                edge_fail_t_idx_bound_count += 1
                continue

            self.edges.append(edge_tuple)
            edge_pass_count += 1

        self.number_of_edges = len(self.edges)
        print(
            f"  [DEBUG Graph._process_constructor_inputs] Finished processing edges. Passed: {edge_pass_count}. Failed s_type: {edge_fail_s_idx_type_count}, s_bound: {edge_fail_s_idx_bound_count}, t_type: {edge_fail_t_idx_type_count}, t_bound: {edge_fail_t_idx_bound_count}")
        print(f"  [DEBUG Graph._process_constructor_inputs] Final self.edges count: {len(self.edges)}, self.number_of_edges: {self.number_of_edges}")


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        super().__init__(nodes=nodes, edges=edges)
        # The [DEBUG DirectedNgramGraph] prints from the previous log will now be more insightful
        # as they reflect the state *after* the enhanced _process_constructor_inputs.

        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = None

        self.A_out_w: torch.Tensor
        self.A_in_w: torch.Tensor
        self.mathcal_A_out: np.ndarray
        self.mathcal_A_in: np.ndarray
        self.fai: torch.Tensor
        self.fao: torch.Tensor

        if self.number_of_nodes > 0:
            self._create_raw_weighted_adj_matrices_torch()
            self._create_propagation_matrices_for_gcn()
            self._create_symmetrized_magnitudes_fai_fao()
        else:
            self.A_out_w = torch.empty((0, 0), dtype=torch.float32)
            self.A_in_w = torch.empty((0, 0), dtype=torch.float32)
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0, 0)
            self.fai = torch.empty((0, 0), dtype=torch.float32)
            self.fao = torch.empty((0, 0), dtype=torch.float32)

    def _create_raw_weighted_adj_matrices_torch(self):
        if self.number_of_nodes == 0:
            self.A_out_w = torch.empty((0, 0), dtype=torch.float32)
            self.A_in_w = torch.empty((0, 0), dtype=torch.float32)
            print("  [DEBUG _create_raw_weighted_adj_matrices_torch] No nodes, adj matrices are empty.")
            return

        self.A_out_w = torch.zeros((self.number_of_nodes, self.number_of_nodes), dtype=torch.float32)
        print(f"  [DEBUG _create_raw_weighted_adj_matrices_torch] Initializing A_out_w for {self.number_of_nodes} nodes. Processing {len(self.edges)} edges.")

        for s_idx, t_idx, weight, *_ in self.edges:
            # This check should be redundant if _process_constructor_inputs worked, but good for safety
            if 0 <= s_idx < self.number_of_nodes and 0 <= t_idx < self.number_of_nodes:
                self.A_out_w[s_idx, t_idx] = float(weight)
            else:
                # This case should ideally not be reached if edges were properly filtered
                print(f"    [DEBUG _create_raw_weighted_adj_matrices_torch] WARNING: Edge ({s_idx},{t_idx}) out of bounds for num_nodes={self.number_of_nodes}. Skipping.")
        self.A_in_w = self.A_out_w.t().contiguous()
        print(f"  [DEBUG _create_raw_weighted_adj_matrices_torch] A_out_w non-zero elements: {torch.count_nonzero(self.A_out_w)}")

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch: torch.Tensor) -> np.ndarray:
        if A_w_torch.shape[0] == 0:
            return np.array([], dtype=np.float32).reshape(0, 0)

        row_sum = A_w_torch.sum(dim=1)
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=A_w_torch.device)
        non_zero_degrees = row_sum != 0
        D_inv_diag_vals[non_zero_degrees] = 1.0 / row_sum[non_zero_degrees]

        A_n = D_inv_diag_vals.unsqueeze(1) * A_w_torch
        S = (A_n + A_n.t()) / 2.0
        K = (A_n - A_n.t()) / 2.0
        mathcal_A_base = torch.sqrt(torch.square(S) + torch.square(K) + self.epsilon_propagation)
        identity_matrix = torch.eye(self.number_of_nodes, dtype=torch.float32, device=A_w_torch.device)
        mathcal_A_with_self_loops = mathcal_A_base + identity_matrix
        return mathcal_A_with_self_loops.cpu().numpy()

    def _create_propagation_matrices_for_gcn(self):
        self.mathcal_A_out = self._calculate_single_propagation_matrix_for_gcn(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix_for_gcn(self.A_in_w)

    def _create_symmetrized_magnitudes_fai_fao(self):
        if self.number_of_nodes == 0:
            self.fai = torch.empty((0, 0), dtype=torch.float32)
            self.fao = torch.empty((0, 0), dtype=torch.float32)
            return

        dev = self.A_out_w.device
        identity = torch.eye(self.number_of_nodes, device=dev, dtype=torch.float32)

        A_out_w_sl = self.A_out_w + identity
        D_out_sl_d = A_out_w_sl.sum(dim=1)
        D_out_sl_inv_d = torch.zeros_like(D_out_sl_d, device=dev)
        D_out_sl_inv_d[D_out_sl_d != 0] = 1.0 / D_out_sl_d[D_out_sl_d != 0]
        A_out_n_sl = D_out_sl_inv_d.unsqueeze(1) * A_out_w_sl
        our = (A_out_n_sl + A_out_n_sl.t()) / 2.0
        oui = (A_out_n_sl - A_out_n_sl.t()) / 2.0
        self.fao = torch.sqrt(our.pow(2) + oui.pow(2) + self.epsilon_propagation)

        A_in_w_for_fai_calc = self.A_out_w.t().contiguous()
        A_in_w_sl = A_in_w_for_fai_calc + identity
        D_in_sl_d = A_in_w_sl.sum(dim=1)
        D_in_sl_inv_d = torch.zeros_like(D_in_sl_d, device=dev)
        D_in_sl_inv_d[D_in_sl_d != 0] = 1.0 / D_in_sl_d[D_in_sl_d != 0]
        A_in_n_sl = D_in_sl_inv_d.unsqueeze(1) * A_in_w_sl
        ir = (A_in_n_sl + A_in_n_sl.t()) / 2.0
        ii = (A_in_n_sl - A_in_n_sl.t()) / 2.0
        self.fai = torch.sqrt(ir.pow(2) + ii.pow(2) + self.epsilon_propagation)

    def get_fai_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fai') or self.fai is None:
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
        if self.fai is None: raise ValueError("FAI could not be computed or is None.")
        return dense_to_sparse(self.fai)

    def get_fao_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fao') or self.fao is None:
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
        if self.fao is None: raise ValueError("FAO could not be computed or is None.")
        return dense_to_sparse(self.fao)
