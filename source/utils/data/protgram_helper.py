import math
import collections
from typing import Dict, Iterator, Optional, Tuple, Any, List

import torch
from torch_geometric.data import Data
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils
from source.utils.post.embedding_processor import EmbeddingProcessor


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
        print(f"    --- End of Graph Statistics for n={n_val} ---\n")

    @staticmethod
    def preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
        """Prepares a sequence tuple for Dask Bag processing."""
        pid, seq_text = seq_tuple
        # Add space padding for consistent n-gram extraction at sequence boundaries
        modified_seq_text = f" {seq_text}" if add_initial_space else str(seq_text)
        return pid, f"{modified_seq_text} "

    @staticmethod
    def extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[str]:
        """Extracts n-grams from a single processed sequence."""
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val:
            for i in range(len(processed_seq_text) - n_val + 1):
                yield processed_seq_text[i:i + n_val]

    @staticmethod
    def extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int,
                                          ngram_to_id_map: Dict[str, int]) -> Iterator[str]:
        """Extracts n-gram transitions (edges) from a single processed sequence."""
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val + 1:
            for i in range(len(processed_seq_text) - n_val):
                source_id = ngram_to_id_map.get(processed_seq_text[i:i + n_val])
                target_id = ngram_to_id_map.get(processed_seq_text[i + 1:i + 1 + n_val])
                if source_id is not None and target_id is not None:
                    # Yield a string representation for easy writing to text files
                    yield f"{source_id} {target_id}"

    @staticmethod
    def extract_edge_ngram_pairs(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[Tuple[str, str]]:
        """Extracts n-gram transition pairs (source_ngram, target_ngram) from a single processed sequence."""
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val + 1:
            for i in range(len(processed_seq_text) - n_val):
                source_ngram = processed_seq_text[i:i + n_val]
                target_ngram = processed_seq_text[i + 1:i + 1 + n_val]
                yield source_ngram, target_ngram

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

        # --- Definitive Fix for "Zeroes" Issue: Prevent Double-Normalization ---
        # 1. Provide the RAW, un-normalized, undirected graph as the baseline. The models
        #    themselves (GCN, GAT, etc.) are configured to perform normalization internally.
        print(f"  Preparing data for '{model_type}' using RAW undirected graph as a baseline.")
        A_undir_w = (graph.A_out_w + graph.A_in_w).coalesce()
        data_dict['edge_index'] = A_undir_w.indices()
        data_dict['edge_attr'] = A_undir_w.values()

        # 2. Overwrite the default only for specialized models that require different graph structures.
        if model_name_lower == 'directgcn':
            # DirectGCN requires multiple, specific graph views. We add them here.
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
            data_dict['edge_index'] = graph.A_out_w.indices()
            data_dict['edge_attr'] = graph.A_out_w.values()
            # DirGNN also needs the backward edges for its internal logic.
            data_dict['edge_index_backward'] = graph.A_in_w.indices()

        return Data.from_dict(data_dict)
