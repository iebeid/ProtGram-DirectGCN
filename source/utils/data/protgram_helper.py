import math
import collections
from typing import Dict, Iterator, Optional, Tuple, Any, List

import torch
from torch_geometric.data import Data
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils
from source.utils.post.embedding_processor import EmbeddingProcessor
import networkx as nx
from torch_geometric.utils import homophily
import scipy
import scipy.sparse as sp


# ==============================================================================
# 4. Dask Helpers for ProtGram Graph Builder
# ==============================================================================
class ProtgramDaskHelpers:
    """
    Contains static helper methods used exclusively by the Dask pipeline
    in the GraphBuilder class. Isolating them here cleans up the namespace.
    """

    @staticmethod
    def log_graph_statistics(graph_object: 'DirectedNgramGraph', n_val: int):
        """Logs key statistics for a given graph object."""
        print(f"    --- Graph Statistics for n={n_val} ---")
        num_nodes = graph_object.number_of_nodes
        num_edges = graph_object.number_of_edges
        print(f"      Nodes: {num_nodes}")
        print(f"      Edges (unique weighted): {num_edges}")
        if num_nodes > 1:
            possible_edges_no_self_loops = num_nodes * (num_nodes - 1)
            density = num_edges / possible_edges_no_self_loops if possible_edges_no_self_loops > 0 else 0
            print(f"      Density (E / N(N-1)): {density:.4f}")

        # Connected Components and Communities
        labels, num_classes = DataUtils.generate_community_labels(graph_object)
        
        # Homophily Score
        if labels is not None:
            homophily_ratio = homophily(graph_object.A_undirected_w.indices(), labels, method='edge')
            print(f"      Homophily Ratio: {homophily_ratio:.4f}")

        # Centrality Score (Degree Centrality)
        sparse_csr_tensor = graph_object.A_undirected_w.to_sparse_csr()
        scipy_csr_matrix = sp.csr_matrix(
            (sparse_csr_tensor.values().cpu().numpy(), sparse_csr_tensor.col_indices().cpu().numpy(), sparse_csr_tensor.crow_indices().cpu().numpy()),
            shape=(graph_object.number_of_nodes, graph_object.number_of_nodes)
        )
        G = nx.from_scipy_sparse_array(scipy_csr_matrix, create_using=nx.Graph)
        centrality = nx.degree_centrality(G)
        avg_centrality = sum(centrality.values()) / len(centrality)
        print(f"      Average Degree Centrality: {avg_centrality:.4f}")

        # Average Clustering Coefficient
        avg_clustering = nx.average_clustering(G)
        print(f"      Average Clustering Coefficient: {avg_clustering:.4f}")

        # Assortativity
        assortativity = nx.degree_assortativity_coefficient(G)
        print(f"      Degree Assortativity: {assortativity:.4f}")

        # Closeness Centrality
        closeness = nx.closeness_centrality(G)
        avg_closeness = sum(closeness.values()) / len(closeness)
        print(f"      Average Closeness Centrality: {avg_closeness:.4f}")

        # Betweenness Centrality
        betweenness = nx.betweenness_centrality(G)
        avg_betweenness = sum(betweenness.values()) / len(betweenness)
        print(f"      Average Betweenness Centrality: {avg_betweenness:.4f}")

        print(f"    --- End of Graph Statistics for n={n_val} ---")



    @staticmethod
    def preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
        """Prepares a sequence tuple for Dask Bag processing."""
        pid, seq_text = seq_tuple
        # Add space padding for consistent n-gram extraction at sequence boundaries
        modified_seq_text = f" {seq_text}" if add_initial_space else str(seq_text)
        return pid, f"{modified_seq_text} "

    @staticmethod
    def extract_all_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_max: int) -> List[Tuple[int, str]]:
        """Extracts all n-grams from n=1 to n_max from a single sequence."""
        _, sequence = seq_tuple
        all_ngrams = []
        for n in range(1, n_max + 1):
            if len(sequence) >= n:
                for i in range(len(sequence) - n + 1):
                    all_ngrams.append((n, sequence[i:i + n]))
        return all_ngrams

    @staticmethod
    def extract_all_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_max: int, all_ngram_maps: Dict[int, Dict[str, int]]) -> List[Tuple[int, int, int]]:
        """Extracts all edges for all n-gram levels from a single sequence."""
        _, sequence = seq_tuple
        all_edges = []
        for n in range(1, n_max + 1):
            ngram_map = all_ngram_maps.get(n, {})
            if len(sequence) >= n + 1:
                for i in range(len(sequence) - n):
                    source_ngram = sequence[i:i + n]
                    target_ngram = sequence[i + 1:i + 1 + n]
                    source_id = ngram_map.get(source_ngram)
                    target_id = ngram_map.get(target_ngram)
                    if source_id is not None and target_id is not None:
                        all_edges.append((n, source_id, target_id))
        return all_edges

    @staticmethod
    def extract_all_string_edges(sequence_tuple: Tuple[str, str], n_max: int) -> Iterator[Dict[str, Any]]:
        """
        A Dask-friendly generator that extracts all string-based edges for all
        n-gram levels (from 1 to n_max) from a single sequence.
        """
        _, sequence = sequence_tuple
        for n in range(1, n_max + 1):
            if len(sequence) >= n + 1:
                for i in range(len(sequence) - n):
                    yield {'n': n, 'source_str': sequence[i:i + n], 'target_str': sequence[i + 1:i + 1 + n]}

    @staticmethod
    def extract_string_edges_for_n(sequence_tuple: Tuple[str, str], n_level: int) -> Iterator[Dict[str, str]]:
        _, sequence = sequence_tuple
        if len(sequence) >= n_level + 1:
            for i in range(len(sequence) - n_level):
                yield {'source_str': sequence[i:i + n_level], 'target_str': sequence[i + 1:i + 1 + n_level]}

    @staticmethod
    def prepare_pyg_data_from_protgram_graph(model_type: str, graph: 'DirectedNgramGraph', features: torch.Tensor,
                                             labels: Optional[torch.Tensor], A_homo_norm: Optional[torch.Tensor] = None, A_hetero_norm: Optional[torch.Tensor] = None,
                                             train_mask: Optional[torch.Tensor] = None,
                                             val_mask: Optional[torch.Tensor] = None,
                                             test_mask: Optional[torch.Tensor] = None) -> Data:
        """
        A centralized utility to prepare a PyG Data object from a DirectedNgramGraph,
        tailored to the specific model's needs. This eliminates duplicated logic
        and ensures each model receives the correct graph representation.
        """
        # --- DEFINITIVE FIX: Pass the masks through to the new Data object ---
        # This was the root cause of the AttributeError in the GNN benchmarker.
        data_dict: Dict[str, Any] = {
            'x': features, 'y': labels, 'graph_obj': graph,
            'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask
        }
        model_name_lower = model_type.lower()

        # --- REFACTOR: Use an explicit if/elif/else block for clarity ---
        # This avoids setting a baseline graph representation that is then immediately
        # overwritten or ignored by the specialized models.
        if model_name_lower == 'directgcn':
            # DirectGCN requires multiple, specific graph views. We add them here.
            print(f"  Preparing specialized graph views for '{model_type}'.")
            data_dict.update({
                'edge_index_undirected_norm': graph.A_undirected_norm_sparse.indices(),
                'edge_weight_undirected_norm': graph.A_undirected_norm_sparse.values(),
                'edge_index_mathcal_in': graph.mathcal_A_in.indices(),
                'edge_weight_mathcal_in': graph.mathcal_A_in.values(),
                'edge_index_mathcal_out': graph.mathcal_A_out.indices(),
                'edge_weight_mathcal_out': graph.mathcal_A_out.values()
            })
            if A_homo_norm is not None and A_hetero_norm is not None:
                print("    -> Adding normalized homophily/heterophily paths for DirectGCN.")
                data_dict.update({
                    'edge_index_homo_norm': A_homo_norm.indices(), 'edge_weight_homo_norm': A_homo_norm.values(),
                    'edge_index_hetero_norm': A_hetero_norm.indices(), 'edge_weight_hetero_norm': A_hetero_norm.values()
                })

        elif model_name_lower == 'rgcn':
            # RGCN needs raw directed edges combined into a single tensor with an edge_type attribute.
            print("    -> Overwriting baseline graph for RGCN with specific directed edge format.")
            edge_index_out = graph.A_out_w.indices()
            edge_index_in = graph.A_in_w.indices()
            device = edge_index_out.device
            edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long, device=device)
            edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long, device=device)
            data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in])
            # RGCN does not use edge_attr when edge_type is present.
            if 'edge_attr' in data_dict:
                del data_dict['edge_attr']

        elif model_name_lower == 'dirgnn':
            # DirGNN needs the raw, directed forward edges for its message passing.
            print("    -> Overwriting baseline graph for DirGNN with raw directed edges.")
            A_out_w = graph.A_out_w.coalesce()
            data_dict['edge_index'] = A_out_w.indices()
            data_dict['edge_attr'] = A_out_w.values()
            # DirGNN also needs the backward edges for its internal logic.
            data_dict['edge_index_backward'] = graph.A_in_w.indices()
        else:
            # Default case for standard GNNs (GCN, GAT, GraphSAGE, etc.)
            # Provide the RAW, un-normalized, undirected graph. The models themselves
            # are configured to perform normalization internally.
            print(f"  Preparing standard undirected graph for '{model_type}'.")
            A_undir_w = (graph.A_out_w + graph.A_in_w).coalesce()
            data_dict['edge_index'] = A_undir_w.indices()
            data_dict['edge_attr'] = A_undir_w.values()

        return Data.from_dict(data_dict)
