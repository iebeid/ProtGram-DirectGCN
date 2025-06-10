# ==============================================================================
# MODULE: utils/graph_processor.py
# PURPOSE: Contains robust classes for n-gram graph representation and for
#          creating edge features for link prediction.
# ==============================================================================

import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Tuple, Any


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
        try:
            graph_nx = nx.Graph(weightless_edges)
            num_components = nx.number_connected_components(graph_nx)
            if num_components > 1:
                print(f"WARNING: N-gram graph has {num_components} connected components.")
            else:
                print("N-gram graph has a single connected component.")
        except Exception as e:
            print(f"WARNING: Could not perform NetworkX integrity check. Error: {e}")


class DirectedNgramGraph(NgramGraph):
    """Represents a directed n-gram graph, computing adjacency and degree matrices."""

    def _adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates binary adjacency matrices."""
        out_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        in_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.edges:
            out_adj[s_n, t_n] = 1
            in_adj[t_n, s_n] = 1
        return out_adj, in_adj

    def _weighted_adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates weighted adjacency matrices."""
        out_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        in_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, w, *_ in self.edges:
            out_w_adj[s_n, t_n] = w
            in_w_adj[t_n, s_n] = w
        return out_w_adj, in_w_adj

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        super().__init__(nodes=nodes, edges=edges)
        print("Constructing adjacency matrices for directed graph...")
        self.out_adjacency_matrix, self.in_adjacency_matrix = self._adjacency_matrices()
        self.out_weighted_adjacency, self.in_weighted_adjacency = self._weighted_adjacency_matrices()
        print("Adjacency matrices constructed.")


class EdgeFeatureProcessor:
    """A class to handle the creation of edge embeddings for the evaluation MLP."""

    @staticmethod
    def create_edge_embeddings(interaction_pairs: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray], method: str = 'concatenate') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates edge features for link prediction from per-protein embeddings.
        """
        print(f"Creating edge embeddings using method: '{method}'...")
        if not protein_embeddings:
            print("ERROR: Protein embeddings dictionary is empty.")
            return None

        # Determine the dimension from the first embedding vector
        embedding_dim = next(iter(protein_embeddings.values())).shape[0]
        feature_dim = embedding_dim * 2 if method == 'concatenate' else embedding_dim

        valid_pairs = [pair for pair in interaction_pairs if pair[0] in protein_embeddings and pair[1] in protein_embeddings]

        if not valid_pairs:
            print("ERROR: No valid pairs found with available embeddings.")
            return None

        num_valid_pairs = len(valid_pairs)
        edge_features = np.zeros((num_valid_pairs, feature_dim), dtype=np.float32)
        labels = np.zeros(num_valid_pairs, dtype=np.int32)

        for idx, (p1_id, p2_id, label) in enumerate(tqdm(valid_pairs, desc="Creating Edge Features")):
            emb1 = protein_embeddings[p1_id]
            emb2 = protein_embeddings[p2_id]

            if method == 'concatenate':
                feature = np.concatenate((emb1, emb2))
            elif method == 'average':
                feature = (emb1 + emb2) / 2.0
            elif method == 'hadamard':
                feature = emb1 * emb2
            elif method == 'l1':
                feature = np.abs(emb1 - emb2)
            elif method == 'l2':
                feature = (emb1 - emb2) ** 2
            else:
                # Default to concatenation if method is unknown
                feature = np.concatenate((emb1, emb2))

            edge_features[idx] = feature
            labels[idx] = label

        print(f"Created {len(edge_features)} edge features with dimension {feature_dim}.")
        return edge_features, labels
