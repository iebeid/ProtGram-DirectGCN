# ==============================================================================
# MODULE: utils/graph_processor.py
# PURPOSE: Contains robust classes for n-gram graph representation and for
#          creating edge features for link prediction.
# VERSION: 5.1 (Added direct node_sequences attribute to NgramGraph)
# ==============================================================================

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Tuple, Any


class NgramGraph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        self.nodes_map = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []
        self.edges = list(self.original_edges)
        self.node_sequences: List[str] = []  # Initialize attribute

        self._create_node_indices()
        self._index_edges()
        self.integrity_check()

    def _create_node_indices(self):
        """Creates index and inverted_index mappings for nodes, and node_sequences list."""
        self.node_to_idx = {}
        self.idx_to_node = {}

        node_keys = set(self.nodes_map.keys())
        if not node_keys and self.edges:
            # Infer nodes from edges if nodes_map is empty but edges are provided
            for pair in self.original_edges:
                # Ensure edge elements are treated as strings for node names
                node_keys.add(str(pair[0]))
                node_keys.add(str(pair[1]))

        sorted_nodes = sorted(list(node_keys))
        for i, node_name in enumerate(sorted_nodes):
            self.node_to_idx[node_name] = i
            self.idx_to_node[i] = node_name

        self.number_of_nodes = len(self.node_to_idx)

        # Populate node_sequences directly
        if self.number_of_nodes > 0:
            self.node_sequences = [self.idx_to_node[i] for i in range(self.number_of_nodes)]
        else:
            self.node_sequences = []

        print(f"Created node indices and sequences for {self.number_of_nodes} unique n-gram nodes.")

    def _index_edges(self):
        """Converts node names in edge list to their integer indices."""
        indexed_edge_list = []
        for es in self.edges:
            # Ensure edge elements used for lookup are strings
            s_idx = self.node_to_idx.get(str(es[0]))
            t_idx = self.node_to_idx.get(str(es[1]))
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
            # Ensure nodes are integers for NetworkX if they come from indexed_edge_list
            graph_nx = nx.Graph()
            if self.number_of_nodes > 0:  # Add nodes if they exist to handle isolated nodes
                graph_nx.add_nodes_from(range(self.number_of_nodes))
            graph_nx.add_edges_from(weightless_edges)

            if self.number_of_nodes == 0 and not weightless_edges:  # Truly empty graph
                print("N-gram graph is empty (no nodes, no edges).")
                return
            elif self.number_of_nodes > 0 and not weightless_edges:  # Nodes but no edges
                print(f"N-gram graph has {self.number_of_nodes} nodes but no edges. All nodes are isolated.")
                return

            num_components = nx.number_connected_components(graph_nx)
            if num_components > 1:
                print(f"WARNING: N-gram graph has {num_components} connected components.")
            else:
                print("N-gram graph has a single connected component.")
        except Exception as e:
            print(f"WARNING: Could not perform NetworkX integrity check. Error: {e}")


class DirectedNgramGraph(NgramGraph):
    """
    Represents a directed n-gram graph, computing adjacency and degree matrices.
    This class is preserved for any parts of the pipeline that might still use it.
    """

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        super().__init__(nodes=nodes, edges=edges)
        print("Constructing standard adjacency matrices for directed graph...")
        if self.number_of_nodes > 0:  # Avoid creating matrices for empty graphs
            self.out_adjacency_matrix, self.in_adjacency_matrix = self._adjacency_matrices()
            self.out_weighted_adjacency, self.in_weighted_adjacency = self._weighted_adjacency_matrices()
            print("Standard adjacency matrices constructed.")
        else:
            self.out_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.in_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.out_weighted_adjacency = np.array([], dtype=np.float32).reshape(0, 0)
            self.in_weighted_adjacency = np.array([], dtype=np.float32).reshape(0, 0)
            print("Skipped adjacency matrix construction for empty graph.")

    def _adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates binary adjacency matrices."""
        out_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        in_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.edges:  # self.edges are already indexed
            out_adj[s_n, t_n] = 1
            in_adj[t_n, s_n] = 1
        return out_adj, in_adj

    def _weighted_adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates weighted adjacency matrices."""
        out_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        in_w_adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, w, *_ in self.edges:  # self.edges are already indexed
            out_w_adj[s_n, t_n] = w
            in_w_adj[t_n, s_n] = w
        return out_w_adj, in_w_adj


class DirectedNgramGraphForGCN(NgramGraph):
    """
    A specialized graph object for the ProtDiGCN model. It stores the raw,
    unnormalized weighted adjacency matrices (A_out_w, A_in_w), as required
    by the GCN which performs preprocessing internally.
    """

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        # Initialize the base class to get node indexing and node_sequences
        super().__init__(nodes=nodes, edges=edges)
        print("Creating raw weighted adjacency matrices for ProtDiGCN...")
        self._create_raw_adjacency_matrices()
        print("Raw matrices (A_out_w, A_in_w) created successfully for GCN.")

    def _create_raw_adjacency_matrices(self):
        """Creates the raw weighted adjacency matrices from the edge list counts."""
        if self.number_of_nodes == 0:
            self.A_out_w = np.array([], dtype=np.float32).reshape(0, 0)
            self.A_in_w = np.array([], dtype=np.float32).reshape(0, 0)
            return

        # Using SciPy sparse matrices is generally more memory-efficient for large graphs
        # and can be faster for certain operations. PyG utilities often work well with them.
        # However, sticking to NumPy arrays as per existing structure for now.
        # from scipy.sparse import csr_matrix
        # row_indices = []
        # col_indices = []
        # data_weights = []
        # for s_idx, t_idx, w, *_ in self.edges: # self.edges are already indexed
        #     row_indices.append(s_idx)
        #     col_indices.append(t_idx)
        #     data_weights.append(w)
        # self.A_out_w = csr_matrix((data_weights, (row_indices, col_indices)),
        #                           shape=(self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        # self.A_in_w = self.A_out_w.transpose().tocsr()

        A_out_w = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        # self.edges is already indexed from the parent __init__ call
        for s_idx, t_idx, w, *_ in self.edges:
            A_out_w[s_idx, t_idx] = w

        self.A_out_w = A_out_w
        self.A_in_w = A_out_w.T  # For dense NumPy arrays, .T is efficient.


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
        try:
            embedding_dim = next(iter(protein_embeddings.values())).shape[0]
        except (StopIteration, AttributeError, IndexError):  # Added IndexError for empty ndarray
            print("ERROR: Could not determine embedding dimension from the first embedding vector.")
            return None

        if embedding_dim == 0:
            print("ERROR: Embedding dimension is 0.")
            return None

        feature_dim = embedding_dim * 2 if method == 'concatenate' else embedding_dim

        valid_pairs_data = []  # Store (emb1, emb2, label) to avoid repeated lookups
        for p1_id, p2_id, label in interaction_pairs:
            emb1 = protein_embeddings.get(p1_id)
            emb2 = protein_embeddings.get(p2_id)
            if emb1 is not None and emb2 is not None:
                if emb1.shape[0] == embedding_dim and emb2.shape[0] == embedding_dim:
                    valid_pairs_data.append((emb1, emb2, label))
                else:
                    print(f"Warning: Skipping pair ({p1_id}, {p2_id}) due to mismatched embedding dimension.")

        if not valid_pairs_data:
            print("ERROR: No valid pairs found with available and correctly dimensioned embeddings.")
            return None

        num_valid_pairs = len(valid_pairs_data)
        edge_features = np.zeros((num_valid_pairs, feature_dim), dtype=np.float32)
        labels = np.zeros(num_valid_pairs, dtype=np.int32)

        for idx, (emb1, emb2, label) in enumerate(tqdm(valid_pairs_data, desc="Creating Edge Features")):
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
            else:  # Default to concatenate
                print(f"Warning: Unknown method '{method}', defaulting to 'concatenate'.")
                feature = np.concatenate((emb1, emb2))

            edge_features[idx] = feature
            labels[idx] = label

        print(f"Created {len(edge_features)} edge features with dimension {feature_dim}.")
        return edge_features, labels
