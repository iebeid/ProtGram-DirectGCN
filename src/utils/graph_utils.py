# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/utils/graph_utils.py
# ==============================================================================
# MODULE: utils/graph_utils.py
# PURPOSE: Contains robust classes for n-gram graph representation.
# VERSION: 5.5 (Implemented specific propagation matrix preprocessing)
# ==============================================================================

from typing import List, Dict, Tuple, Any

import networkx as nx
import numpy as np


# Assuming Config is accessible for epsilon, or pass epsilon as an argument
# For simplicity here, we'll assume epsilon is a known small constant or passed if not from global config.
# In a full pipeline, it would come from the Config object.

class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        self.nodes_map = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []
        self.edges = list(self.original_edges)
        self.node_sequences: List[str] = []

        self._create_node_indices()
        self._index_edges()
        self.integrity_check()

    def _create_node_indices(self):
        self.node_to_idx = {}
        self.idx_to_node = {}
        node_keys = set(self.nodes_map.keys())
        if not node_keys and self.edges:
            for pair in self.original_edges:
                node_keys.add(str(pair[0]))
                node_keys.add(str(pair[1]))
        sorted_nodes = sorted(list(node_keys))
        for i, node_name in enumerate(sorted_nodes):
            self.node_to_idx[node_name] = i
            self.idx_to_node[i] = node_name
        self.number_of_nodes = len(self.node_to_idx)
        if self.number_of_nodes > 0:
            self.node_sequences = [self.idx_to_node[i] for i in range(self.number_of_nodes)]
        else:
            self.node_sequences = []

    def _index_edges(self):
        indexed_edge_list = []
        for es in self.edges:
            s_idx = self.node_to_idx.get(str(es[0]))
            t_idx = self.node_to_idx.get(str(es[1]))
            if s_idx is not None and t_idx is not None:
                indexed_edge_list.append((s_idx, t_idx) + es[2:])
        self.edges = indexed_edge_list
        self.number_of_edges = len(self.edges)

    def integrity_check(self):
        if not self.edges and self.number_of_nodes == 0: return
        if not self.edges and self.number_of_nodes > 0: return
        if not self.edges: return
        weightless_edges = [(e[0], e[1]) for e in self.edges]
        try:
            graph_nx = nx.Graph()
            if self.number_of_nodes > 0:
                graph_nx.add_nodes_from(range(self.number_of_nodes))
            graph_nx.add_edges_from(weightless_edges)
        except Exception as e:
            print(f"WARNING: Could not perform NetworkX integrity check during NgramGraph init. Error: {e}")


class DirectedNgramGraph(Graph):
    """
    A specialized graph object for the ProtDiGCN model. It stores:
    - Raw weighted adjacency matrices (A_out_w, A_in_w).
    - Preprocessed propagation matrices (mathcal_A_out, mathcal_A_in) using symmetric/skew-symmetric decomposition.
    - Binary adjacency matrices (out_adjacency_matrix, in_adjacency_matrix).
    """

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        super().__init__(nodes=nodes, edges=edges)
        self.epsilon_propagation = epsilon_propagation

        if self.number_of_nodes > 0:
            self._create_raw_weighted_adj_matrices()  # A_out_w, A_in_w
            self._create_binary_adj_matrices()  # out_adjacency_matrix, in_adjacency_matrix
            self._create_propagation_matrices()  # mathcal_A_out, mathcal_A_in
        else:
            self.A_out_w = np.array([], dtype=np.float32).reshape(0, 0)
            self.A_in_w = np.array([], dtype=np.float32).reshape(0, 0)
            self.out_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.in_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0, 0)

    def _create_raw_weighted_adj_matrices(self):
        self.A_out_w = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_idx, t_idx, w, *_ in self.edges:
            self.A_out_w[s_idx, t_idx] = w
        self.A_in_w = self.A_out_w.T

    def _create_binary_adj_matrices(self):
        self.out_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        self.in_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.edges:
            self.out_adjacency_matrix[s_n, t_n] = 1
            self.in_adjacency_matrix[t_n, s_n] = 1

    def _calculate_single_propagation_matrix(self, A_w: np.ndarray) -> np.ndarray:
        """
        Helper function to calculate the propagation matrix from a weighted adjacency matrix.
        Steps:
        1. Row-wise degree normalization: A_n = D^-1 * A_w
        2. Symmetric-like component: S = (A_n + A_n.T) / 2
        3. Skew-symmetric-like component: K = (A_n - A_n.T) / 2
        4. Propagation matrix: mathcal_A_uv = sqrt(S_uv^2 + K_uv^2 + epsilon)
        """
        if A_w.shape[0] == 0:  # Handle empty matrix
            return np.array([], dtype=np.float32).reshape(0, 0)

        # 1. Degree normalization (row-wise)
        # For A_out_w, D_out_ii = sum_j (A_out_w)_ij. This is row sum.
        # For A_in_w, D_in_ii = sum_j (A_in_w)_ij. This is also row sum of A_in_w.
        row_sum = A_w.sum(axis=1)
        D_inv_diag = np.zeros_like(row_sum, dtype=np.float32)
        non_zero_degrees = row_sum != 0
        D_inv_diag[non_zero_degrees] = 1.0 / row_sum[non_zero_degrees]

        # Efficient way to do D_inv @ A_w for diagonal D_inv
        A_n = np.diag(D_inv_diag) @ A_w
        # Or element-wise: A_n = D_inv_diag[:, np.newaxis] * A_w

        # 2. Symmetric-like component
        S = (A_n + A_n.T) / 2.0

        # 3. Skew-symmetric-like component
        K = (A_n - A_n.T) / 2.0

        # 4. Propagation matrix
        mathcal_A = np.sqrt(np.square(S) + np.square(K) + self.epsilon_propagation)

        return mathcal_A

    def _create_propagation_matrices(self):
        """
        Creates the preprocessed propagation matrices mathcal_A_out and mathcal_A_in.
        """
        print(f"Creating propagation matrices (mathcal_A) with epsilon={self.epsilon_propagation}...")
        self.mathcal_A_out = self._calculate_single_propagation_matrix(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix(self.A_in_w)
        print("Propagation matrices mathcal_A_out and mathcal_A_in created.")
