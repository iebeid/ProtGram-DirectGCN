# ==============================================================================
# MODULE: data_structures/coarsener.py
# PURPOSE: Implements graph coarsening techniques to reduce graph size.
# VERSION: 1.1 (Refactored to use PyG's pool_edge)
# AUTHOR: Gemini Code Assist
# ==============================================================================

from typing import Tuple, Optional, TYPE_CHECKING

import numpy as np
import scipy.sparse.linalg as sp_linalg
import torch
# --- DEFINITIVE FIX: Add version-agnostic imports for PyG to handle API changes ---
try:
    # For PyG >= 2.0
    from torch_geometric.nn.pool import graclus, pool_edge
    from torch_geometric.nn import modularity
except ImportError:
    # Fallback for PyG < 2.0
    from torch_geometric.nn import functional as F
    from torch_geometric.utils import pool_edge
    from torch_geometric.utils import modularity
    graclus = F.graclus
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, get_laplacian
from tqdm.auto import tqdm

# --- FIX: Use a forward reference to prevent circular import errors ---
if TYPE_CHECKING:
    from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils


class GraphCoarsener:
    """
    A utility to reduce the size of a graph using coarsening algorithms.
    This is a form of lossy compression that aims to preserve key structural
    properties while reducing the number of nodes and edges.
    """

    @staticmethod
    def coarsen_graph(graph: 'DirectedNgramGraph', level: int = 1) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        edge_index = graph.A_undirected_norm_sparse.coalesce().indices()
        edge_weight = graph.A_undirected_norm_sparse.coalesce().values()

        # Initialize the cluster map where each node is its own cluster
        num_nodes = graph.number_of_nodes
        cluster_map = torch.arange(num_nodes, device=edge_index.device)

        print(f"  Initial graph: {num_nodes} nodes, {edge_index.size(1)} edges.")

        for i in tqdm(range(level), desc="  Coarsening Levels", leave=False):
            if edge_index.numel() == 0:
                print(f"  - Coarsening stopped at level {i+1}: No edges remain.")
                break

            clusters = graclus(edge_index, edge_weight)
            cluster_map = clusters[cluster_map]

            # --- REFACTOR: Use the idiomatic PyG function for coarsening edges ---
            # This is cleaner and more direct than converting to SciPy and back.
            edge_index, edge_weight = pool_edge(clusters, edge_index, edge_weight, reduce='add')
            num_nodes = int(clusters.max().item()) + 1

            print(f"  Level {i + 1}: Coarsened to {num_nodes} nodes, {edge_index.size(1)} edges.")

        print("--- Graph Coarsening Finished ---")
        return edge_index, cluster_map, edge_weight

    @staticmethod
    def validate_coarsening(
            original_graph: 'DirectedNgramGraph',
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