# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 6.1 (Implement FAI/FAO with self-loops in A_w, mathcal_A self-loops after sqrt, remove binary adj)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse


class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple]):  # nodes keys are int IDs
        self.idx_to_node_map_from_constructor = nodes if nodes is not None else {}  # Store the passed map
        self.original_edges = edges if edges is not None else []
        # self.edges will be populated by _index_edges, which expects original_edges to use original identifiers
        # However, in data_builder, weighted_edge_list_tuples already contains integer indices.

        self.node_to_idx: Dict[Any, int] = {}  # Will map original node identifiers (if they were strings) to int
        self.idx_to_node: Dict[int, Any] = {}  # Will map int to original node identifiers
        self.number_of_nodes: int = 0
        self.node_sequences: List[Any] = []

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
            return

        # Determine all unique integer node indices present
        all_integer_indices = set(self.idx_to_node_map_from_constructor.keys())
        for edge_tuple in self.original_edges:
            if len(edge_tuple) >= 2:
                all_integer_indices.add(edge_tuple[0])  # s_idx
                all_integer_indices.add(edge_tuple[1])  # t_idx

        if not all_integer_indices:
            self.number_of_nodes = 0
            self.edges = list(self.original_edges)  # Potentially empty
            return

        # Create a sorted list of unique integer indices to ensure consistent mapping
        # This also handles cases where some indices might only appear in edges
        # or only in the idx_to_node_map_from_constructor.
        # min_idx = 0 # Not strictly needed if we derive number_of_nodes from max_idx
        if all_integer_indices:
            max_idx = 0
            for idx in all_integer_indices:
                if not isinstance(idx, int):
                    # This check is crucial. If this error is raised, the input data types are wrong.
                    raise TypeError(f"Node and edge indices must be integers. Found: {idx} of type {type(idx)}")
                if idx > max_idx:
                    max_idx = idx
            self.number_of_nodes = max_idx + 1
        else:
            self.number_of_nodes = 0

        # Populate idx_to_node and node_to_idx
        # node_to_idx will map the node name (n-gram) to its canonical integer index
        # idx_to_node will map the canonical integer index to its node name (n-gram)
        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                # This case means an edge referred to an index not in the provided nodes map,
                # or the nodes map is not contiguous from 0 to max_idx.
                # For this pipeline, idx_to_node_map_from_constructor should be complete.
                # If it's truly missing, a placeholder or error is appropriate.
                node_name = f"__NODE_{i}__"  # Placeholder
                # Consider logging a warning if this happens unexpectedly.
                # print(f"Warning: Node name for index {i} not found in provided nodes map. Using placeholder.")
            temp_idx_to_node_name[i] = str(node_name)  # Ensure node names are strings

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}

        self.node_sequences = [self.idx_to_node[i] for i in range(self.number_of_nodes)]

        # Edges are already indexed (as integers), so filter and copy them
        self.edges = []
        for edge_tuple in self.original_edges:
            s_idx, t_idx = edge_tuple[0], edge_tuple[1]
            # Validate indices before adding the edge
            if not (isinstance(s_idx, int) and 0 <= s_idx < self.number_of_nodes):
                # print(f"Warning: Source index {s_idx} out of bounds or not int. Skipping edge: {edge_tuple}")
                continue
            if not (isinstance(t_idx, int) and 0 <= t_idx < self.number_of_nodes):
                # print(f"Warning: Target index {t_idx} out of bounds or not int. Skipping edge: {edge_tuple}")
                continue
            self.edges.append(edge_tuple)  # (s_idx, t_idx, weight, ...)

        self.number_of_edges = len(self.edges)

    # The old _create_node_indices and _index_edges methods are intentionally removed
    # as their logic is now integrated into _process_constructor_inputs.


class DirectedNgramGraph(Graph):
    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        # `nodes` is idx_to_node from data_builder, keys are int, values are str (n-grams)
        # `edges` is weighted_edge_list_tuples, (int, int, float)
        super().__init__(nodes=nodes, edges=edges)  # This will call the new _process_constructor_inputs
        self.epsilon_propagation = epsilon_propagation
        self.n_value: Optional[int] = None  # This will be set in data_builder.py

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
            # Initialize to empty tensors if no nodes
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
            return
        self.A_out_w = torch.zeros((self.number_of_nodes, self.number_of_nodes), dtype=torch.float32)
        for s_idx, t_idx, weight, *_ in self.edges:  # Unpack, assuming weight is the third element
            if 0 <= s_idx < self.number_of_nodes and 0 <= t_idx < self.number_of_nodes:
                self.A_out_w[s_idx, t_idx] = float(weight)
        self.A_in_w = self.A_out_w.t().contiguous()

    def _calculate_single_propagation_matrix_for_gcn(self, A_w_torch: torch.Tensor) -> np.ndarray:
        if A_w_torch.shape[0] == 0:  # Handle empty matrix
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

        dev = self.A_out_w.device  # Use device of A_out_w
        identity = torch.eye(self.number_of_nodes, device=dev, dtype=torch.float32)

        # FAO Calculation
        A_out_w_sl = self.A_out_w + identity
        D_out_sl_d = A_out_w_sl.sum(dim=1)
        D_out_sl_inv_d = torch.zeros_like(D_out_sl_d, device=dev)
        D_out_sl_inv_d[D_out_sl_d != 0] = 1.0 / D_out_sl_d[D_out_sl_d != 0]
        A_out_n_sl = D_out_sl_inv_d.unsqueeze(1) * A_out_w_sl
        our = (A_out_n_sl + A_out_n_sl.t()) / 2.0
        oui = (A_out_n_sl - A_out_n_sl.t()) / 2.0
        self.fao = torch.sqrt(our.pow(2) + oui.pow(2) + self.epsilon_propagation)

        # FAI Calculation
        A_in_w_for_fai_calc = self.A_out_w.t().contiguous()  # Use A_out_w.t() for consistency
        A_in_w_sl = A_in_w_for_fai_calc + identity
        D_in_sl_d = A_in_w_sl.sum(dim=1)
        D_in_sl_inv_d = torch.zeros_like(D_in_sl_d, device=dev)
        D_in_sl_inv_d[D_in_sl_d != 0] = 1.0 / D_in_sl_d[D_in_sl_d != 0]
        A_in_n_sl = D_in_sl_inv_d.unsqueeze(1) * A_in_w_sl
        ir = (A_in_n_sl + A_in_n_sl.t()) / 2.0
        ii = (A_in_n_sl - A_in_n_sl.t()) / 2.0
        self.fai = torch.sqrt(ir.pow(2) + ii.pow(2) + self.epsilon_propagation)

    def get_fai_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fai') or self.fai is None:  # Check if fai exists and is computed
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
        if self.fai is None: raise ValueError("FAI could not be computed or is None.")  # Should not happen if logic above is correct
        return dense_to_sparse(self.fai)

    def get_fao_sparse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'fao') or self.fao is None:  # Check if fao exists and is computed
            if self.number_of_nodes > 0:
                self._create_symmetrized_magnitudes_fai_fao()
            else:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
        if self.fao is None: raise ValueError("FAO could not be computed or is None.")  # Should not happen
        return dense_to_sparse(self.fao)
