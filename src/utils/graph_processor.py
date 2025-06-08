# ==============================================================================
# MODULE: utils/graph_processor.py
# PURPOSE: Contains classes for n-gram graph representation and for creating
#          edge features for link prediction.
# ==============================================================================

import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Set, Tuple


# --- Utility Functions for Graph Classes ---

def _flip_list(list_of_tuples: List[Tuple]) -> List[Tuple]:
    """Flips the first two elements of each tuple in a list."""
    flipped_edges = []
    for es in list_of_tuples:
        # Assuming tuple structure (source, target, weight, ...)
        flipped_edges.append((es[1], es[0]) + es[2:])
    return flipped_edges


def _are_matrices_identical(A: np.ndarray, B: np.ndarray) -> bool:
    """Compares two numpy matrices and returns True if they are identical."""
    if A.shape != B.shape:
        return False
    return np.allclose(A, B)


# ==============================================================================
# --- N-GRAM GRAPH REPRESENTATION ---
# (Adapted from graph_utils.py)
# ==============================================================================

class NgramGraph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        self.nodes_map = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []
        self.edges = list(self.original_edges)

        self._create_node_indices()
        self._index_edges()
        self.integrity_check()

    def _create_node_indices(self):
        """Creates index and inverted_index mappings for nodes."""
        self.node_to_idx = {}
        self.idx_to_node = {}

        node_keys = set(self.nodes_map.keys())
        # Infer nodes from edges if the nodes_map is empty
        if not node_keys and self.edges:
            for pair in self.original_edges:
                node_keys.add(str(pair[0]))
                node_keys.add(str(pair[1]))

        sorted_nodes = sorted(list(node_keys))
        for i, node_name in enumerate(sorted_nodes):
            self.node_to_idx[node_name] = i
            self.idx_to_node[i] = node_name

        self.number_of_nodes = len(self.node_to_idx)
        print(f"Created node indices for {self.number_of_nodes} unique n-gram nodes.")

    def _index_edges(self):
        """Converts node names in edge list to their integer indices."""
        indexed_edge_list = []
        for es in self.edges:
            s_idx = self.node_to_idx.get(es[0])
            t_idx = self.node_to_idx.get(es[1])
            if s_idx is not None and t_idx is not None:
                indexed_edge_list.append((s_idx, t_idx) + es[2:])
        self.edges = indexed_edge_list
        self.number_of_edges = len(self.edges)

    def integrity_check(self):
        """Performs a basic integrity check on the graph structure."""
        print("Performing n-gram graph integrity check...")
        if not self.edges:
            print("Graph has no edges to check.")
            return

        weightless_edges = [(e[0], e[1]) for e in self.edges]
        graph_nx = nx.Graph(weightless_edges)
        num_components = nx.number_connected_components(graph_nx)
        if num_components > 1:
            print(f"WARNING: N-gram graph has {num_components} connected components.")
        else:
            print("N-gram graph has a single connected component.")


class DirectedNgramGraph(NgramGraph):
    """Represents a directed n-gram graph, computing adjacency and degree matrices."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        super().__init__(nodes=nodes, edges=edges)
        self.in_indexed_edges = _flip_list(self.edges)
        self.out_adjacency_matrix, self.in_adjacency_matrix = self._adjacency_matrices()
        self.out_weighted_adjacency, self.in_weighted_adjacency = self._weighted_adjacency_matrices()

    def _adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        out_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.edges: out_adj[s_n, t_n] = 1
        in_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.in_indexed_edges: in_adj[s_n, t_n] = 1
        return out_adj, in_adj

    def _weighted_adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        out_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, w, *_ in self.edges: out_w_adj[s_n, t_n] = w
        in_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, w, *_ in self.in_indexed_edges: in_w_adj[s_n, t_n] = w
        return out_w_adj, in_w_adj


# ==============================================================================
# --- EDGE FEATURE ENGINEERING FOR LINK PREDICTION ---
# (Adapted from evaluater.py)
# ==============================================================================

class EdgeFeatureProcessor:
    """A class to handle the creation of edge embeddings for the evaluation MLP."""

    @staticmethod
    def create_edge_embeddings(
            interaction_pairs: List[Tuple[str, str, int]],
            protein_embeddings: Dict[str, np.ndarray],
            method: str = 'concatenate'
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates edge features for link prediction from per-protein embeddings.
        This version pre-allocates the NumPy array for better memory efficiency.
        """
        print(f"Creating edge embeddings using method: '{method}'...")
        if not protein_embeddings:
            print("ERROR: Protein embeddings dictionary is empty.")
            return None

        embedding_dim = next(iter(protein_embeddings.values())).shape[0]
        feature_dim = embedding_dim * 2 if method == 'concatenate' else embedding_dim

        # Pre-filter pairs to find out the final size of the dataset
        valid_pairs = []
        for p1_id, p2_id, label in interaction_pairs:
            if p1_id in protein_embeddings and p2_id in protein_embeddings:
                valid_pairs.append((p1_id, p2_id, label))

        if not valid_pairs:
            print("ERROR: No valid pairs found with available embeddings.")
            return None

        # Pre-allocate NumPy arrays for memory efficiency
        num_valid_pairs = len(valid_pairs)
        edge_features = np.zeros((num_valid_pairs, feature_dim), dtype=np.float32)
        labels = np.zeros(num_valid_pairs, dtype=np.int32)

        idx = 0
        for p1_id, p2_id, label in tqdm(valid_pairs, desc="Creating Edge Features", leave=False):
            emb1 = protein_embeddings[p1_id]
            emb2 = protein_embeddings[p2_id]

            if method == 'concatenate':
                feature = np.concatenate((emb1, emb2))
            elif method == 'average':
                feature = (emb1 + emb2) / 2.0
            elif method == 'hadamard':
                feature = emb1 * emb2
            elif method == 'subtract':
                feature = np.abs(emb1 - emb2)
            else:
                feature = np.concatenate((emb1, emb2))

            edge_features[idx] = feature
            labels[idx] = label
            idx += 1

        print(f"Created {len(edge_features)} edge features with dimension {feature_dim}.")
        return edge_features, labels