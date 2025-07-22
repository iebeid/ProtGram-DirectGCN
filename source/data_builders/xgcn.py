# ==============================================================================
# MODULE: utils/xgcn.py
# PURPOSE: Generates labels for various self-supervised GNN training tasks.
# VERSION: 1.0 (Created by Gemini Code Assist)
# AUTHOR: Islam Ebeid
# ==============================================================================

import collections
import random
from typing import Tuple, Optional

import community as community_louvain
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.utils.data import FastaUtils


class XGCNDataset:
    """A class dedicated to generating labels for self-supervised tasks on graphs."""

    def __init__(self, config: Config):
        self.config = config

    def generate_task_labels(self, graph: DirectedNgramGraph, task_type: str) -> Tuple[Optional[torch.Tensor], int]:
        """Dispatcher for generating labels for different self-supervised tasks."""
        print(f"  Generating self-supervised labels for task: '{task_type}'")
        if task_type == 'community':
            return self._generate_community_labels(graph)
        elif task_type == 'next_node':
            return self._generate_next_node_labels(graph)
        elif task_type == 'closest_aa':
            return self._generate_closest_amino_acid_labels(graph, self.config.GCN_CLOSEST_AA_K_HOPS)
        else:
            raise ValueError(f"Unknown self-supervised task type: '{task_type}'")

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        """Generates community detection labels using the Louvain algorithm."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"    Generating community labels for all {num_nodes} nodes.")

        coo = graph.A_undirected_norm_sparse.cpu().coalesce()
        if coo._nnz() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1

        nx_graph = to_networkx(Data(edge_index=coo.indices(), edge_attr=coo.values(), num_nodes=num_nodes),
                               to_undirected=True, edge_attrs=['edge_attr'])
        if nx_graph.number_of_edges() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1

        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE, weight='edge_attr')
        labels_list = [partition.get(i, -1) for i in range(num_nodes)]

        unique_labels = sorted(list(set(labels_list)))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
        num_classes = len(unique_labels)
        print(f"      Found {num_classes} communities.")
        return labels, num_classes

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        """Generates labels by predicting the most likely next node based on transition weights."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"    Generating next_node labels for all {num_nodes} nodes.")
        adj_out_weighted_sparse = graph.A_out_w
        labels_list = [-1] * num_nodes
        for i in tqdm(range(num_nodes), desc="      Generating next_node labels", leave=False,
                      disable=not self.config.DEBUG_VERBOSE):
            row_mask = (adj_out_weighted_sparse.indices()[0] == i)
            if not torch.any(row_mask):
                labels_list[i] = i
            else:
                successors = adj_out_weighted_sparse.indices()[1][row_mask]
                weights = adj_out_weighted_sparse.values()[row_mask]
                max_weight_successors = successors[weights == weights.max()]
                labels_list[i] = random.choice(max_weight_successors.cpu().tolist())
        return torch.tensor(labels_list, dtype=torch.long), num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        """Generates labels by finding the shortest path distance to a randomly chosen amino acid."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        labels = torch.full((num_nodes,), k_hops, dtype=torch.long)
        print(f"    Generating closest_aa labels for all {num_nodes} nodes (k={k_hops}).")
        adj_out = graph.A_out_w.cpu()
        node_names = graph.node_names
        for start_node in tqdm(range(num_nodes), desc="      Generating closest_aa labels", leave=False,
                               disable=not self.config.DEBUG_VERBOSE):
            target_aa = random.choice(FastaUtils.AMINO_ACID_ALPHABET)
            if target_aa in str(node_names[start_node]):
                labels[start_node] = 0
                continue
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            found_at_hop = -1
            while q:
                curr, hop = q.popleft()
                if hop >= k_hops: break
                row_mask = (adj_out.indices()[0] == curr)
                for neighbor in adj_out.indices()[1][row_mask]:
                    n_idx = neighbor.item()
                    if n_idx not in visited:
                        visited.add(n_idx)
                        if target_aa in str(node_names[n_idx]):
                            found_at_hop = hop + 1
                            break
                        q.append((n_idx, hop + 1))
                if found_at_hop != -1: break
            if found_at_hop != -1:
                labels[start_node] = found_at_hop
        return labels, k_hops + 1