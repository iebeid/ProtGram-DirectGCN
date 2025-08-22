# ==============================================================================
# MODULE: data_structures/coarsener.py
# PURPOSE: Implements graph coarsening techniques to reduce graph size.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

from typing import Tuple, Optional

import numpy as np
import scipy.sparse.linalg as sp_linalg
import torch
from torch_geometric.nn.pool import graclus
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, get_laplacian, modularity
from tqdm.auto import tqdm

from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils


class GraphCoarsener:
    """
    A utility to reduce the size of a graph using coarsening algorithms.
    This is a form of lossy compression that aims to preserve key structural
    properties while reducing the number of nodes and edges.
    """

    @staticmethod
    def coarsen_graph(graph: DirectedNgramGraph, level: int = 1) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Coarsens a graph using the Graclus clustering algorithm.

        This method takes a graph and iteratively coarsens it `level` times.
        It returns the final coarsened adjacency matrix and the cluster mapping.

        Args:
            graph (DirectedNgramGraph): The input graph to coarsen.
            level (int): The number of coarsening levels to apply.

        Returns:
            A tuple containing:
            - adj (torch.Tensor): The adjacency matrix of the coarsened graph.
            - cluster_map (torch.Tensor): A tensor mapping each node in the original
              graph to its cluster ID in the final coarsened graph.
            - weights (torch.Tensor): The edge weights of the coarsened graph.
            Returns None if the input graph is invalid.
        """
        if graph.A_undirected_norm_sparse is None or graph.A_undirected_norm_sparse._nnz() == 0:
            print("  - Coarsening Error: Input graph has no undirected edges.")
            return None

        DataUtils.print_header(f"Graph Coarsening (Levels: {level})")

        # Start with the undirected, normalized adjacency matrix of the original graph
        current_adj = graph.A_undirected_norm_sparse.coalesce()
        edge_index = current_adj.indices()
        edge_weight = current_adj.values()

        # Initialize the cluster map where each node is its own cluster
        num_nodes = graph.number_of_nodes
        cluster_map = torch.arange(num_nodes, device=edge_index.device)

        print(f"  Initial graph: {num_nodes} nodes, {edge_index.size(1)} edges.")

        for i in tqdm(range(level), desc="  Coarsening Levels", leave=False):
            # Use Graclus to find clusters in the current graph
            clusters = graclus(edge_index, edge_weight)

            # Update the global cluster map: map original nodes to the new clusters
            cluster_map = clusters[cluster_map]

            # Create the new, smaller adjacency matrix based on the clusters
            # Convert to SciPy sparse matrix for efficient manipulation
            scipy_adj = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=current_adj.size(0))

            # Create a cluster matrix C where C[i, j] = 1 if node i is in cluster j
            C = torch.zeros((current_adj.size(0), clusters.max() + 1), dtype=torch.float32)
            C[torch.arange(current_adj.size(0)), clusters] = 1

            # The new coarsened adjacency matrix is C^T * A * C
            coarsened_scipy_adj = C.t().numpy() @ scipy_adj @ C.numpy()

            # Convert back to PyG format
            edge_index, edge_weight = from_scipy_sparse_matrix(coarsened_scipy_adj)
            current_adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(C.shape[1], C.shape[1]))

            print(f"  Level {i + 1}: Coarsened to {current_adj.size(0)} nodes, {edge_index.size(1)} edges.")

        print("--- Graph Coarsening Finished ---")
        return current_adj.coalesce().indices(), cluster_map, current_adj.coalesce().values()

    @staticmethod
    def validate_coarsening(
            original_graph: DirectedNgramGraph,
            coarsened_edge_index: torch.Tensor,
            coarsened_edge_weight: torch.Tensor,
            cluster_map: torch.Tensor,
            k_eigenvals: int = 10
    ) -> None:
        """
        Validates the quality of the coarsened graph by comparing its structural
        properties to the original graph.

        Args:
            original_graph: The original DirectedNgramGraph object.
            coarsened_edge_index: The edge index of the smaller, coarsened graph.
            coarsened_edge_weight: The edge weights of the smaller, coarsened graph.
            cluster_map: The mapping from original nodes to coarsened super-nodes.
            k_eigenvals: The number of top eigenvalues to compare.
        """
        DataUtils.print_header("Graph Coarsening Validation")

        # 1. Basic Statistics
        num_original_nodes = original_graph.number_of_nodes
        num_original_edges = original_graph.A_undirected_norm_sparse.coalesce()._nnz()
        num_coarsened_nodes = int(cluster_map.max().item()) + 1
        num_coarsened_edges = coarsened_edge_index.size(1)

        node_reduction = (1 - num_coarsened_nodes / num_original_nodes) * 100 if num_original_nodes > 0 else 0
        edge_reduction = (1 - num_coarsened_edges / num_original_edges) * 100 if num_original_edges > 0 else 0

        print(f"  - Node Reduction: {num_original_nodes} -> {num_coarsened_nodes} ({node_reduction:.2f}%)")
        print(f"  - Edge Reduction: {num_original_edges} -> {num_coarsened_edges} ({edge_reduction:.2f}%)")

        # 2. Modularity
        original_edge_index = original_graph.A_undirected_norm_sparse.coalesce().indices()
        original_edge_weight = original_graph.A_undirected_norm_sparse.coalesce().values()
        mod = modularity(original_edge_index, cluster_map, original_edge_weight, num_nodes=num_original_nodes)
        print(f"  - Partition Modularity: {mod:.4f} (Higher is better, >0.3 is often significant)")

        # 3. Spectral Comparison (Top-k Eigenvalues of Laplacian)
        print(f"  - Spectral Comparison (comparing top {k_eigenvals} smallest eigenvalues of the Laplacian):")
        try:
            L_orig_index, L_orig_weight = get_laplacian(original_edge_index, original_edge_weight, normalization='sym', num_nodes=num_original_nodes)
            L_orig_scipy = to_scipy_sparse_matrix(L_orig_index, L_orig_weight, num_nodes=num_original_nodes)
            L_coarsened_index, L_coarsened_weight = get_laplacian(coarsened_edge_index, coarsened_edge_weight, normalization='sym', num_nodes=num_coarsened_nodes)
            L_coarsened_scipy = to_scipy_sparse_matrix(L_coarsened_index, L_coarsened_weight, num_nodes=num_coarsened_nodes)

            # --- DEFINITIVE FIX for Numerical Stability ---
            # Use eigsh for symmetric matrices like the Laplacian. It's faster and more stable.
            eigvals_orig = np.sort(sp_linalg.eigsh(L_orig_scipy, k=k_eigenvals, which='SM', return_eigenvectors=False))
            eigvals_coarsened = np.sort(sp_linalg.eigsh(L_coarsened_scipy, k=k_eigenvals, which='SM', return_eigenvectors=False))
            spectral_distance = np.linalg.norm(eigvals_orig - eigvals_coarsened)
            print(f"    - Spectral Distance (L2):  {spectral_distance:.4f} (Lower is better)")
        except Exception as e:
            print(f"    - Spectral comparison failed: {e}")
        print("--- Coarsening Validation Finished ---")