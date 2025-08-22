import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from .graph import Graph


class DirectedGraph:
    """
    A helper class containing matrix calculation logic, primarily for the
    benchmarking suite which uses standard PyG Data objects.
    """

    def __init__(self):
        pass

    def _calculate_single_propagation_matrix(self, A_w_torch_sparse: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Calculates the propagation matrix mathcal{A} = sqrt(S^2 + K^2 + epsilon) + I
        This logic is replicated from the main GraphBuilder for benchmark compatibility.
        """
        if num_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            return torch.sparse_coo_tensor(empty_indices, empty_values, (num_nodes, num_nodes)).coalesce()

        dev = A_w_torch_sparse.device
        row_sum = torch.sparse.sum(A_w_torch_sparse, dim=1).to_dense()
        # --- FIX: Add clamping for numerical stability, mirroring the main graph class ---
        # This prevents division by tiny numbers on sparse graphs, which can cause NaNs or performance drops.
        row_sum_clamped = torch.clamp(row_sum, min=1e-9)
        D_inv_diag_vals = torch.zeros_like(row_sum_clamped, dtype=torch.float32, device=dev)
        non_zero_degrees_mask = row_sum_clamped != 0
        if torch.any(non_zero_degrees_mask):
            D_inv_diag_vals[non_zero_degrees_mask] = 1.0 / row_sum_clamped[non_zero_degrees_mask]

        A_w_indices = A_w_torch_sparse.indices()
        A_w_values = A_w_torch_sparse.values()
        scaled_values = A_w_values * D_inv_diag_vals[A_w_indices[0]]
        A_n_sparse = torch.sparse_coo_tensor(A_w_indices, scaled_values, A_w_torch_sparse.size()).coalesce()

        A_n_sq_values = A_n_sparse.values().pow(2)
        A_n_sq_sparse = torch.sparse_coo_tensor(A_n_sparse.indices(), A_n_sq_values, A_n_sparse.size()).coalesce()
        A_n_sq_t_sparse = A_n_sq_sparse.t().coalesce()
        S_sq_plus_K_sq_sparse = (A_n_sq_sparse + A_n_sq_t_sparse).coalesce()
        S_sq_plus_K_sq_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(),
                                                        S_sq_plus_K_sq_sparse.values() * 0.5,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()

        epsilon_tensor = torch.tensor(1e-9, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()

        identity_sparse = Graph._sparse_identity(num_nodes, device=dev)
        return (mathcal_A_base_sparse + identity_sparse).coalesce()

    def _normalize_symmetric_matrix(self, matrix: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Helper to apply GCN normalization to a symmetric matrix."""  # noqa
        if matrix.numel() == 0 or matrix._nnz() == 0: return matrix
        edge_index, edge_weight = add_self_loops(matrix.indices(), matrix.values(), fill_value=1.0,
                                                 num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, norm_values, matrix.shape).coalesce()

    def _preprocess_for_custom_models(self, data: Data, use_homo_hetero_paths: bool) -> Data:
        """Prepares a data object with all necessary edge indices for custom models."""
        base_edge_weight = data.edge_attr if hasattr(data,
                                                     'edge_attr') and data.edge_attr is not None else torch.ones(
            data.edge_index.shape[1], device=data.edge_index.device)

        if use_homo_hetero_paths:
            edge_index = data.edge_index
            source_nodes, target_nodes = edge_index[0], edge_index[1]
            source_labels = data.y[source_nodes]
            target_labels = data.y[target_nodes]
            homo_mask = (source_labels == target_labels)
            hetero_mask = ~homo_mask

            edge_index_out_homo = edge_index[:, homo_mask]
            edge_weight_out_homo = base_edge_weight[homo_mask]
            A_out_w_homo = torch.sparse_coo_tensor(edge_index_out_homo, edge_weight_out_homo,
                                                   (data.num_nodes, data.num_nodes)).coalesce()
            A_homo_w = (A_out_w_homo + A_out_w_homo.t()).coalesce()
            data.edge_index_homo_norm, data.edge_weight_homo_norm = self._normalize_symmetric_matrix(A_homo_w,
                                                                                                     data.num_nodes).coalesce().indices(), self._normalize_symmetric_matrix(
                A_homo_w, data.num_nodes).coalesce().values()

            edge_index_out_hetero = edge_index[:, hetero_mask]
            edge_weight_out_hetero = base_edge_weight[hetero_mask]
            A_out_w_hetero = torch.sparse_coo_tensor(edge_index_out_hetero, edge_weight_out_hetero,
                                                     (data.num_nodes, data.num_nodes)).coalesce()
            A_hetero_w = (A_out_w_hetero + A_out_w_hetero.t()).coalesce()
            data.edge_index_hetero_norm, data.edge_weight_hetero_norm = self._normalize_symmetric_matrix(A_hetero_w,
                                                                                                         data.num_nodes).coalesce().indices(), self._normalize_symmetric_matrix(
                A_hetero_w, data.num_nodes).coalesce().values()

        A_out_w_sparse = torch.sparse_coo_tensor(data.edge_index, base_edge_weight,
                                                 (data.num_nodes, data.num_nodes)).coalesce()
        A_in_w_sparse = A_out_w_sparse.t().coalesce()
        # --- DEFINITIVE FIX: Attach the sparse tensors themselves to the data object ---
        # The downstream helper function (`prepare_pyg_data_from_protgram_graph`)
        # expects these attributes to exist on the object it receives.
        data.A_out_w = A_out_w_sparse
        data.A_in_w = A_in_w_sparse
        A_undir_w = (A_out_w_sparse + A_in_w_sparse).coalesce()
        # --- DEFINITIVE FIX: Attach the sparse tensor itself, not just its components ---
        A_undirected_norm = self._normalize_symmetric_matrix(A_undir_w, data.num_nodes)
        data.A_undirected_norm_sparse = A_undirected_norm
        data.edge_index_out, data.edge_weight_out = A_out_w_sparse.indices(), A_out_w_sparse.values()
        data.edge_index_in, data.edge_weight_in = A_in_w_sparse.indices(), A_in_w_sparse.values()
        data.edge_index_undirected_norm, data.edge_weight_undirected_norm = A_undirected_norm.indices(), A_undirected_norm.values()
        data.edge_index_backward = data.edge_index_in

        data.edge_attr = data.edge_weight_undirected_norm
        data.edge_index = data.edge_index_undirected_norm

        mathcal_A_out = self._calculate_single_propagation_matrix(A_out_w_sparse, data.num_nodes)
        mathcal_A_in = self._calculate_single_propagation_matrix(A_in_w_sparse, data.num_nodes)
        data.edge_index_mathcal_out, data.edge_weight_mathcal_out = mathcal_A_out.indices(), mathcal_A_out.values()
        data.edge_index_mathcal_in, data.edge_weight_mathcal_in = mathcal_A_in.indices(), mathcal_A_in.values()
        # --- DEFINITIVE FIX: Attach the full sparse tensors for downstream access ---
        data.mathcal_A_out = mathcal_A_out
        data.mathcal_A_in = mathcal_A_in
        return data
