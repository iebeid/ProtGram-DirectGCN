# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 6.2 (Enhanced debugging in _process_constructor_inputs)
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
                # MODIFICATION 1: Check for both Python int and numpy.integer
                if not isinstance(edge_tuple[0], (int, np.integer)) or \
                        not isinstance(edge_tuple[1], (int, np.integer)):
                    print(
                        f"    [DEBUG Graph._process_constructor_inputs] WARNING: Non-integer index type in edge_tuple: {edge_tuple} (types: {type(edge_tuple[0])}, {type(edge_tuple[1])}). Skipping for index collection.")
                    continue
                all_integer_indices.add(int(edge_tuple[0]))  # Cast to int for consistency in the set
                all_integer_indices.add(int(edge_tuple[1]))  # Cast to int for consistency in the set

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
            # MODIFICATION 2: Check for both Python int and numpy.integer (though idx_val should be int here due to casting above and from node_map)
            if not isinstance(idx_val, (int, np.integer)):
                # This should ideally not happen if the above check for edges worked
                print(f"    [DEBUG Graph._process_constructor_inputs] CRITICAL WARNING: Non-integer index '{idx_val}' (type: {type(idx_val)}) found in all_integer_indices despite prior checks.")
                continue  # Skip non-integer if it somehow got through
            if idx_val > max_idx:
                max_idx = idx_val

        self.number_of_nodes = max_idx + 1
        print(f"  [DEBUG Graph._process_constructor_inputs] max_idx from all_integer_indices: {max_idx}, self.number_of_nodes SET TO: {self.number_of_nodes}")

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                # If an index came from an edge and was not in the original nodes map,
                # it will get a default name.
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

            s_idx_orig, t_idx_orig = edge_tuple[0], edge_tuple[1]

            # MODIFICATION 3: Check for both Python int and numpy.integer
            valid_s_type = isinstance(s_idx_orig, (int, np.integer))
            valid_t_type = isinstance(t_idx_orig, (int, np.integer))

            s_idx = int(s_idx_orig) if valid_s_type else s_idx_orig
            t_idx = int(t_idx_orig) if valid_t_type else t_idx_orig

            valid_s_bound = (0 <= s_idx < self.number_of_nodes) if valid_s_type else False
            valid_t_bound = (0 <= t_idx < self.number_of_nodes) if valid_t_type else False

            # For debug printing, show original types if they failed the check
            s_print_val = s_idx_orig if not valid_s_type else s_idx
            t_print_val = t_idx_orig if not valid_t_type else t_idx

            if i < 5 or not (valid_s_type and valid_s_bound and valid_t_type and valid_t_bound):  # Print first 5 and all failing
                print(
                    f"    [DEBUG Graph._process_constructor_inputs] Edge {i}: ({s_print_val}, {t_print_val}) | s_ok: {valid_s_type and valid_s_bound} (type:{valid_s_type}, bound:{valid_s_bound}) | t_ok: {valid_t_type and valid_t_bound} (type:{valid_t_type}, bound:{valid_t_bound})")

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

            # MODIFICATION 4: Store edges with Python int indices and float weight
            # This ensures self.edges contains consistently typed tuples.
            weight = float(edge_tuple[2]) if len(edge_tuple) > 2 else 1.0  # Default weight if not present
            processed_edge = (s_idx, t_idx, weight) + tuple(edge_tuple[3:])
            self.edges.append(processed_edge)
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
            # Ensure correct empty tensor initialization
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

        for edge_tuple in self.edges:  # self.edges now contains consistently typed tuples
            s_idx, t_idx, weight = edge_tuple[0], edge_tuple[1], edge_tuple[2]
            # The bounds check here is still a good safeguard, though _process_constructor_inputs should handle it.
            if 0 <= s_idx < self.number_of_nodes and 0 <= t_idx < self.number_of_nodes:
                self.A_out_w[s_idx, t_idx] = weight  # weight is already float
            else:
                # This case should ideally not be reached if edges were properly filtered
                print(f"    [DEBUG _create_raw_weighted_adj_matrices_torch] WARNING: Edge ({s_idx},{t_idx}) out of bounds for num_nodes={self.number_of_nodes}. Skipping.")
        self.A_in_w = self.A_out_w.t().contiguous()
        print(f"  [DEBUG _create_raw_weighted_adj_matrices_torch] A_out_w non-zero elements: {torch.count_nonzero(self.A_out_w)}")

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch: torch.Tensor) -> np.ndarray:
        if self.number_of_nodes == 0 or A_w_torch.shape[0] == 0:  # Added self.number_of_nodes check
            return np.array([], dtype=np.float32).reshape(0, 0)

        row_sum = A_w_torch.sum(dim=1)
        D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=A_w_torch.device)
        non_zero_degrees = row_sum != 0
        if torch.any(non_zero_degrees):  # Ensure there are non-zero degrees before division
            D_inv_diag_vals[non_zero_degrees] = 1.0 / row_sum[non_zero_degrees]
        else:  # Handle case where all degrees are zero (e.g. graph with nodes but no edges)
            D_inv_diag_vals = torch.zeros_like(row_sum, dtype=torch.float32, device=A_w_torch.device)

        A_n = D_inv_diag_vals.unsqueeze(1) * A_w_torch
        S = (A_n + A_n.t()) / 2.0
        K = (A_n - A_n.t()) / 2.0
        # Ensure epsilon_propagation is on the same device
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=A_w_torch.device, dtype=torch.float32)
        mathcal_A_base = torch.sqrt(torch.square(S) + torch.square(K) + epsilon_tensor)
        identity_matrix = torch.eye(self.number_of_nodes, dtype=torch.float32, device=A_w_torch.device)
        mathcal_A_with_self_loops = mathcal_A_base + identity_matrix
        return mathcal_A_with_self_loops.cpu().numpy()

    def _create_propagation_matrices_for_gcn(self):
        if self.number_of_nodes == 0:
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0, 0)
            return
        self.mathcal_A_out = self._calculate_single_propagation_matrix_for_gcn(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix_for_gcn(self.A_in_w)

    def _create_symmetrized_magnitudes_fai_fao(self):
        if self.number_of_nodes == 0:
            self.fai = torch.empty((0, 0), dtype=torch.float32)
            self.fao = torch.empty((0, 0), dtype=torch.float32)
            return

        dev = self.A_out_w.device  # A_out_w should be on a device if number_of_nodes > 0
        identity = torch.eye(self.number_of_nodes, device=dev, dtype=torch.float32)
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)

        A_out_w_sl = self.A_out_w + identity
        D_out_sl_d = A_out_w_sl.sum(dim=1)
        D_out_sl_inv_d = torch.zeros_like(D_out_sl_d, device=dev)
        non_zero_D_out = D_out_sl_d != 0
        if torch.any(non_zero_D_out):
            D_out_sl_inv_d[non_zero_D_out] = 1.0 / D_out_sl_d[non_zero_D_out]

        A_out_n_sl = D_out_sl_inv_d.unsqueeze(1) * A_out_w_sl
        our = (A_out_n_sl + A_out_n_sl.t()) / 2.0
        oui = (A_out_n_sl - A_out_n_sl.t()) / 2.0
        self.fao = torch.sqrt(our.pow(2) + oui.pow(2) + epsilon_tensor)

        # A_in_w is already .t().contiguous() of A_out_w
        A_in_w_for_fai_calc = self.A_in_w
        A_in_w_sl = A_in_w_for_fai_calc + identity
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
        if not hasattr(self, 'fai') or self.fai is None:  # Check if fai exists
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:  # No nodes, return empty sparse representation
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        # After attempting creation, check again if self.fai is valid
        if self.fai is None or self.fai.numel() == 0 and self.number_of_nodes > 0:
            # If still None or empty (but nodes exist), it indicates an issue in creation or an edgeless graph
            print(f"Warning: FAI tensor is None or empty for a graph with {self.number_of_nodes} nodes. Returning empty sparse tensor.")
            return torch.empty((2, 0), dtype=torch.long, device=self.A_out_w.device if hasattr(self, 'A_out_w') else 'cpu'), \
                torch.empty(0, dtype=torch.float32, device=self.A_out_w.device if hasattr(self, 'A_out_w') else 'cpu')

        if self.fai.numel() == 0 and self.number_of_nodes == 0:  # Explicitly handle 0-node case
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        return dense_to_sparse(self.fai)

    def get_fao_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fao') or self.fao is None:  # Check if fao exists
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:  # No nodes, return empty sparse representation
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        # After attempting creation, check again if self.fao is valid
        if self.fao is None or self.fao.numel() == 0 and self.number_of_nodes > 0:
            print(f"Warning: FAO tensor is None or empty for a graph with {self.number_of_nodes} nodes. Returning empty sparse tensor.")
            return torch.empty((2, 0), dtype=torch.long, device=self.A_out_w.device if hasattr(self, 'A_out_w') else 'cpu'), \
                torch.empty(0, dtype=torch.float32, device=self.A_out_w.device if hasattr(self, 'A_out_w') else 'cpu')

        if self.fao.numel() == 0 and self.number_of_nodes == 0:  # Explicitly handle 0-node case
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

        return dense_to_sparse(self.fao)
