# ==============================================================================
# MODULE: data_builders/xgcn.py
# PURPOSE: Generates labels for various self-supervised GNN training tasks.
# VERSION: 1.0 (Created by Gemini Code Assist)
# AUTHOR: Islam Ebeid
# ==============================================================================
import collections
import random
from typing import Tuple, Optional, TYPE_CHECKING

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm
from source.utils.data.fasta_utils import FastaUtils

# --- FIX: Import classes for direct type hinting instead of forward references ---
# --- DEFINITIVE FIX: Import DataUtils to use the centralized community detection ---
from source.utils.data.data_utils import DataUtils
if TYPE_CHECKING:
    from configuration.config import Config
    from source.data_structures.graph import DirectedNgramGraph


class XGCNDataBuilder:
    """A class dedicated to generating labels for self-supervised tasks on graphs."""

    def __init__(self, config: 'Config'): # Keep forward ref here for constructor
        self.config = config

    def generate_task_labels(self, graph: 'DirectedNgramGraph', task_type: str) -> Tuple[
        Optional[torch.Tensor], int]:
        """Dispatcher for generating labels for different self-supervised tasks."""
        print(f"  Generating self-supervised labels for task: '{task_type}'")
        if task_type == 'community':
            # --- DEFINITIVE FIX: Use the centralized, robust community detection utility ---
            # This removes redundant code and ensures consistent logic.
            community_graph_data = Data(
                edge_index=graph.A_undirected_w.indices(),
                edge_attr=graph.A_undirected_w.values(), num_nodes=graph.number_of_nodes
            )
            return DataUtils.generate_community_labels(community_graph_data)
        elif task_type == 'next_node':
            return self._generate_next_node_labels(graph)
        elif task_type == 'closest_aa':
            return self._generate_closest_amino_acid_labels(graph, self.config.GCN_CLOSEST_AA_K_HOPS)
        elif task_type == 'masked_node':
            # This task is handled differently. The labels are generated dynamically
            # during training. We return None for the labels and the total number of
            # nodes as the number of "classes" for the model's output layer.
            print("    Task is 'masked_node'. Labels will be generated dynamically per epoch.")
            return None, graph.number_of_nodes
        else:
            raise ValueError(f"Unknown self-supervised task type: '{task_type}'")

    def _generate_next_node_labels(self, graph: 'DirectedNgramGraph') -> Tuple[torch.Tensor, int]:
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

    def _generate_closest_amino_acid_labels(self, graph: 'DirectedNgramGraph', k_hops: int) -> Tuple[
        torch.Tensor, int
    ]:
        """
        Generates labels by finding the shortest path distance to the nearest node
        containing a randomly chosen amino acid. This is done efficiently with a
        single multi-source Breadth-First Search (BFS).
        """
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        labels = torch.full((num_nodes,), k_hops, dtype=torch.long)
        print(f"    Generating closest_aa labels for all {num_nodes} nodes (k={k_hops}).")

        # --- DEFINITIVE FIX: Use a single multi-source BFS for massive performance improvement ---
        # The previous implementation ran a separate BFS for each node, which is O(V*(V+E)).
        # This new approach runs a single BFS, which is O(V+E).

        # 1. Pick one target amino acid for the entire graph.
        target_aa = random.choice(FastaUtils.AMINO_ACID_ALPHABET)
        print(f"      - Target Amino Acid for this graph: '{target_aa}'")

        # 2. Initialize distances and the BFS queue with all "goal" nodes.
        q = collections.deque()
        node_names = graph.node_names
        for i in range(num_nodes):
            if target_aa in str(node_names[i]):
                labels[i] = 0
                q.append((i, 0))  # (node_index, hop_distance)

        # 3. Perform the multi-source BFS.
        # We use the IN-DEGREE adjacency matrix (A_in_w) to traverse edges backwards,
        # from the target nodes outwards to all other nodes.
        adj_in = graph.A_in_w.cpu()
        visited = {node_idx for node_idx, _ in q}  # Keep track of visited nodes to avoid cycles

        with tqdm(total=num_nodes, desc="      - Running multi-source BFS", leave=False, disable=not self.config.DEBUG_VERBOSE) as pbar:
            pbar.update(len(q))
            while q:
                curr, hop = q.popleft()
                if hop >= k_hops: continue

                # Find all nodes that have an edge *to* the current node
                row_mask = (adj_in.indices()[0] == curr)
                for neighbor in adj_in.indices()[1][row_mask]:
                    n_idx = neighbor.item()
                    if n_idx not in visited:
                        visited.add(n_idx)
                        labels[n_idx] = hop + 1
                        q.append((n_idx, hop + 1))
                        pbar.update(1)

        return labels, k_hops + 1

    def generate_masked_node_task(self, graph_obj: 'DirectedNgramGraph', features: torch.Tensor, *,
                                  masking_fraction: float = 0.15,
                                  exclude_mask: Optional[torch.Tensor] = None
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates data for a Masked Node Prediction task.

        This task is analogous to Masked Language Modeling (MLM) in NLP. It masks a
        fraction of nodes in the graph and tasks the model with predicting the
        original identity of these masked nodes based on their neighborhood context.

        Args:
            graph_obj: The graph object.
            features: The original node feature matrix.
            masking_fraction: The fraction of nodes to mask.
            exclude_mask: An optional boolean tensor where `True` indicates nodes
                          that should NOT be masked (e.g., validation/test set).

        Returns:
            A tuple containing:
            - masked_features (torch.Tensor): A new feature matrix where some nodes are masked.
            - masked_indices (torch.Tensor): The indices of the nodes that were masked.
            - original_node_labels (torch.Tensor): The original node indices of the masked nodes,
                                                   which serve as the ground truth labels.
        """
        num_nodes = graph_obj.number_of_nodes

        # --- NEW: Respect the exclude_mask to prevent data leakage in other contexts ---
        if exclude_mask is not None:
            candidate_indices = torch.where(~exclude_mask)[0]
        else:
            candidate_indices = torch.arange(num_nodes)

        num_candidates = len(candidate_indices)
        num_to_mask = int(num_candidates * masking_fraction)

        # --- FIX: Ensure at least one node is masked if possible, to prevent empty test sets. ---
        if num_candidates > 0 and num_to_mask == 0:
            num_to_mask = 1

        if num_to_mask == 0:
            return features.clone(), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        all_node_indices = candidate_indices[torch.randperm(num_candidates)]
        masked_indices = all_node_indices[:num_to_mask]
        original_node_labels = masked_indices.clone()

        masked_features = features.clone()
        mask_token = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
        masked_features[masked_indices] = mask_token

        return masked_features, masked_indices, original_node_labels
