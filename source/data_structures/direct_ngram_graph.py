import gc
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree, subgraph
from .graph import Graph


class DirectedNgramGraph(Graph):
    """
    A specialized graph structure for n-grams that stores multiple sparse
    adjacency matrices required for different GNN models, especially DirectGCN.

    This class is designed to be memory-efficient by loading edges from a file
    in chunks and representing all matrices as sparse PyTorch tensors.

    Key Attributes:
        A_out_w (torch.Tensor): Raw weighted adjacency matrix for outgoing edges.
        A_in_w (torch.Tensor): Raw weighted adjacency matrix for incoming edges (transpose of A_out_w).
        A_undirected_norm_sparse (torch.Tensor): Standard symmetrically normalized
            adjacency matrix (D^-0.5 * A * D^-0.5) for use by GCN, GAT, etc.
        mathcal_A_out (torch.Tensor): The outgoing propagation matrix for DirectGCN,
            calculated using a specific sparse formula.
        mathcal_A_in (torch.Tensor): The incoming propagation matrix for DirectGCN.
        A_homo_norm (torch.Tensor): A normalized matrix containing only homophilous edges.
        A_hetero_norm (torch.Tensor): A normalized matrix containing only heterophilous edges.
    """

    def __init__(self, nodes: Dict[int, Any], **kwargs):

        super().__init__(nodes=nodes, edges=[])

        # --- FIX: Extract params from kwargs to support both building and loading ---
        self.epsilon_propagation = kwargs.get('epsilon_propagation', 1e-9)
        self.n_value: Optional[int] = kwargs.get('n_value')
        edge_file_path = kwargs.get('edge_file_path')
        dir_path = kwargs.get('dir_path')

        # Initialize all matrix attributes
        self.A_out_w: Optional[torch.Tensor] = None
        self.A_in_w: Optional[torch.Tensor] = None
        self.A_undirected_norm_sparse: Optional[torch.Tensor] = None
        self.mathcal_A_out: Optional[torch.Tensor] = None
        self.mathcal_A_in: Optional[torch.Tensor] = None
        self.A_homo_w: Optional[torch.Tensor] = None
        self.A_hetero_w: Optional[torch.Tensor] = None
        self.A_homo_norm: Optional[torch.Tensor] = None
        self.A_hetero_norm: Optional[torch.Tensor] = None

        # --- FIX: Dispatch to either load from directory or build from scratch ---
        if dir_path:
            # We are in "load from directory" mode
            self._load_sparse_tensors_from_dir(Path(dir_path))
            # Verify that loading was successful, otherwise initialize empty
            if self.A_out_w is None:
                print("  - Warning: Loading from directory did not find any valid sparse tensors. Initializing an empty graph.")
                self._initialize_empty_matrices()

        elif self.number_of_nodes > 0 and edge_file_path and os.path.exists(edge_file_path):
            # We are in "build from scratch" mode
            print(f"    Loading edges from {os.path.basename(edge_file_path)}...")
            try:
                # --- FIX: Avoid pd.read_parquet to prevent loading the entire edge file into memory. ---
                # Instead, iterate over the file in chunks using pyarrow for scalability.
                import pyarrow.parquet as pq  # --- DEFINITIVE FIX: Use ParquetDataset to read a directory of files ---
                parquet_file = pq.ParquetDataset(edge_file_path)
                source_chunks, target_chunks, weight_chunks = [], [], []
                for batch in parquet_file.read().to_batches(max_chunksize=10_000_000):
                    # --- DEFINITIVE FIX: Handle cases where index names are lost during parquet read ---
                    # Instead of relying on column names like 'source', access by position.
                    # The structure is known: reset_index() creates [index_col_1, index_col_2, 'weight']
                    df_chunk = batch.to_pandas().reset_index()
                    source_chunks.append(df_chunk.iloc[:, 0].to_numpy(dtype=np.int64))
                    target_chunks.append(df_chunk.iloc[:, 1].to_numpy(dtype=np.int64))
                    weight_chunks.append(df_chunk['weight'].to_numpy(dtype=np.float32))

                source_indices = np.concatenate(source_chunks)
                target_indices = np.concatenate(target_chunks)
                weights = np.concatenate(weight_chunks)
                del source_chunks, target_chunks, weight_chunks
                gc.collect()

                self.number_of_edges = len(source_indices)
                self._create_raw_weighted_adj_matrices_torch(source_indices, target_indices, weights)
                self._create_undirected_normalized_adj_matrix()
                self._create_propagation_matrices()

            except Exception as e:
                import traceback
                print(f"    ❌ An unexpected and critical error occurred while reading the edge file '{edge_file_path}'.")
                print(f"    This may be due to a corrupted Parquet file or a memory issue.")
                print(f"    Error details: {e}")
                traceback.print_exc()
                print("    The pipeline will continue with an empty graph for this level, which may cause downstream failures.")
                self._initialize_empty_matrices()
        else:
            self._initialize_empty_matrices()

    def _load_sparse_tensors_from_dir(self, dir_path: Path):
        """
        Loads sparse tensor components from .npy files in a directory by reading
        the metadata file to discover which tensors to reconstruct.
        """
        print(f"  Reconstructing graph by loading sparse tensors from directory: {dir_path}")
        metadata_path = dir_path / "metadata.json"
        if not metadata_path.exists():
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        tensor_names_to_load = [k.replace('_shape', '') for k in metadata.keys() if k.endswith('_shape')]
        for attr_name in tensor_names_to_load:
            indices_path = dir_path / f"{attr_name}_indices.npy"
            values_path = dir_path / f"{attr_name}_values.npy"
            shape = metadata.get(f"{attr_name}_shape")

            if indices_path.exists() and values_path.exists() and shape:
                indices = torch.from_numpy(np.load(indices_path))
                values = torch.from_numpy(np.load(values_path))
                sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
                setattr(self, attr_name, sparse_tensor)

    def _initialize_empty_matrices(self):
        """Helper to set all matrices to empty sparse tensors."""
        self.number_of_edges = 0
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        empty_values = torch.empty(0, dtype=torch.float32)
        size_empty = (self.number_of_nodes, self.number_of_nodes)

        self.A_out_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_in_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.mathcal_A_out = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.mathcal_A_in = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_homo_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_hetero_w = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_homo_norm = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)
        self.A_hetero_norm = torch.sparse_coo_tensor(empty_indices, empty_values, size_empty)

    def _create_raw_weighted_adj_matrices_torch(self, source_indices: np.ndarray, target_indices: np.ndarray,
                                                weights: np.ndarray):
        """Creates sparse adjacency matrices directly from numpy arrays with memory optimization."""
        size = (self.number_of_nodes, self.number_of_nodes)

        source_tensor = torch.from_numpy(source_indices)
        target_tensor = torch.from_numpy(target_indices)
        edge_indices_tensor = torch.stack([source_tensor, target_tensor]).long()
        del source_tensor, target_tensor
        gc.collect()

        edge_weights_tensor = torch.from_numpy(weights).float()
        del weights
        gc.collect()

        self.A_out_w = torch.sparse_coo_tensor(edge_indices_tensor, edge_weights_tensor, size).coalesce()
        del edge_indices_tensor, edge_weights_tensor
        gc.collect()

        self.A_in_w = self.A_out_w.t().coalesce()

    def _create_undirected_normalized_adj_matrix(self):
        """
        Creates a symmetric, degree-normalized adjacency matrix (D^-0.5 * A * D^-0.5),
        which is the standard for models like GCN, GAT, and GraphSAGE. This matrix
        is created from the sum of the directed in- and out-degree matrices to form
        an undirected representation. Self-loops are added to ensure nodes are
        connected to themselves.
        """
        print(f"  Creating undirected normalized adjacency matrix for n={self.n_value}...")
        if self.number_of_nodes == 0 or self.A_out_w is None or self.A_in_w is None:
            return

        A_undir_w = (self.A_out_w + self.A_in_w).coalesce()

        edge_index, edge_weight = add_self_loops(
            A_undir_w.indices(), A_undir_w.values(),
            fill_value=1.0,
            num_nodes=self.number_of_nodes
        )

        row, col = edge_index
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        self.A_undirected_norm_sparse = torch.sparse_coo_tensor(
            edge_index, norm_values, (self.number_of_nodes, self.number_of_nodes)
        ).coalesce()
        print(
            f"    Undirected normalized matrix created with {self.A_undirected_norm_sparse._nnz()} non-zero elements.")

    def _normalize_symmetric_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Helper function to apply standard GCN normalization (D^-0.5 * A * D^-0.5)
        to any given symmetric matrix.
        """
        if matrix.numel() == 0 or matrix._nnz() == 0:
            return matrix

        edge_index, edge_weight = add_self_loops(
            matrix.indices(), matrix.values(),
            fill_value=1.0, num_nodes=self.number_of_nodes
        )

        row, col = edge_index
        deg = degree(col, self.number_of_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return torch.sparse_coo_tensor(edge_index, norm_values, matrix.shape).coalesce()

    def _calculate_single_propagation_matrix(self, A_w_torch_sparse: torch.Tensor) -> torch.Tensor:
        """
        Calculates the DirectGCN propagation matrix for a single direction (in or out).

        The formula is:
            mathcal{A} = sqrt( 0.5 * ( (D_inv * A)^2 + ((D_inv * A)^2)^T ) + epsilon ) + I

        Where A is the weighted adjacency matrix, D_inv is the inverse of the
        diagonal degree matrix, and I is the identity matrix.
        using sparse tensor operations and an optimized formula.
        """
        if self.number_of_nodes == 0 or (A_w_torch_sparse.is_sparse and A_w_torch_sparse._nnz() == 0):
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=A_w_torch_sparse.device)
            empty_values = torch.empty(0, dtype=torch.float32, device=A_w_torch_sparse.device)
            size = (self.number_of_nodes, self.number_of_nodes)
            return torch.sparse_coo_tensor(empty_indices, empty_values, size).coalesce()

        dev = A_w_torch_sparse.device
        # Step 1: Calculate D_inv * A (row-normalized matrix A_n)
        num_nodes = self.number_of_nodes

        row_sum = torch.sparse.sum(A_w_torch_sparse, dim=1).to_dense()
        # --- DEFINITIVE FIX: Clamp row_sum to prevent division by very small numbers, which causes NaNs. ---
        row_sum_clamped = torch.clamp(row_sum, min=self.epsilon_propagation)

        D_inv_diag_vals = torch.zeros_like(row_sum_clamped, dtype=torch.float32, device=dev)
        non_zero_degrees_mask = row_sum_clamped != 0
        if torch.any(non_zero_degrees_mask):
            D_inv_diag_vals[non_zero_degrees_mask] = 1.0 / row_sum_clamped[non_zero_degrees_mask]
        del row_sum, row_sum_clamped

        A_w_indices = A_w_torch_sparse.indices()
        A_w_values = A_w_torch_sparse.values()
        scaled_values = A_w_values * D_inv_diag_vals[A_w_indices[0]]
        A_n_sparse = torch.sparse_coo_tensor(A_w_indices, scaled_values, A_w_torch_sparse.size()).coalesce()
        del D_inv_diag_vals

        # Step 2: Calculate 0.5 * (A_n^2 + (A_n^2)^T)
        A_n_sq_values = A_n_sparse.values().pow(2)
        A_n_sq_sparse = torch.sparse_coo_tensor(A_n_sparse.indices(), A_n_sq_values, A_n_sparse.size()).coalesce()
        del A_n_sq_values

        A_n_sq_t_sparse = A_n_sq_sparse.t().coalesce()
        S_sq_plus_K_sq_sparse = (A_n_sq_sparse + A_n_sq_t_sparse).coalesce()
        S_sq_plus_K_sq_sparse = torch.sparse_coo_tensor(
            S_sq_plus_K_sq_sparse.indices(),
            S_sq_plus_K_sq_sparse.values() * 0.5,
            S_sq_plus_K_sq_sparse.size()
        ).coalesce()
        del A_n_sq_sparse, A_n_sq_t_sparse, A_n_sparse
        gc.collect()

        # Step 3: Final calculation: sqrt(...) + I
        epsilon_tensor = torch.tensor(self.epsilon_propagation, device=dev, dtype=torch.float32)
        mathcal_A_base_values = torch.sqrt(S_sq_plus_K_sq_sparse.values() + epsilon_tensor)
        mathcal_A_base_sparse = torch.sparse_coo_tensor(S_sq_plus_K_sq_sparse.indices(), mathcal_A_base_values,
                                                        S_sq_plus_K_sq_sparse.size()).coalesce()
        del S_sq_plus_K_sq_sparse, mathcal_A_base_values
        gc.collect()

        identity_sparse = self._sparse_identity(num_nodes, device=dev)
        mathcal_A_with_self_loops_sparse = (mathcal_A_base_sparse + identity_sparse).coalesce()
        del mathcal_A_base_sparse, identity_sparse
        gc.collect()

        return mathcal_A_with_self_loops_sparse

    def _create_propagation_matrices(self):
        """Computes the mathcal_A_out and mathcal_A_in propagation matrices sparsely."""
        print(f"  Creating mathcal_A_out for n={self.n_value}...")
        if self.A_out_w is not None:
            self.mathcal_A_out = self._calculate_single_propagation_matrix(self.A_out_w)
        gc.collect()
        print(f"  Creating mathcal_A_in for n={self.n_value}...")
        if self.A_in_w is not None:
            self.mathcal_A_in = self._calculate_single_propagation_matrix(self.A_in_w)
        gc.collect()

    def split_edges_by_homophily(self, labels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits the raw weighted directed edge matrices (A_out_w, A_in_w) into
        homophilous (nodes in an edge have the same label) and heterophilous
        (nodes have different labels) components. This method is functional and
        returns the normalized matrices instead
        of modifying the object state, which is a safer design pattern.

        Returns:
            A tuple of (A_homo_norm, A_hetero_norm) or None if the graph is empty.
        """
        # --- ANTICIPATORY DEBUGGING: Check for invalid state before proceeding ---
        if self.number_of_nodes == 0 or self.A_out_w is None or self.A_out_w._nnz() == 0:
            print("  Graph has no nodes or edges, skipping homophily split.")
            return None

        print(f"  Splitting {self.A_out_w._nnz()} directed edges by homophily...")

        edge_index, edge_weights = self.A_out_w.indices(), self.A_out_w.values()
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        source_labels, target_labels = labels[source_nodes], labels[target_nodes]

        homo_mask = (source_labels == target_labels)
        hetero_mask = ~homo_mask

        A_out_w_homo = torch.sparse_coo_tensor(edge_index[:, homo_mask], edge_weights[homo_mask], self.A_out_w.shape).coalesce()
        A_out_w_hetero = torch.sparse_coo_tensor(edge_index[:, hetero_mask], edge_weights[hetero_mask], self.A_out_w.shape).coalesce()

        print("    Normalizing homophilic and heterophilic matrices...")
        A_homo_norm = self._normalize_symmetric_matrix((A_out_w_homo + A_out_w_homo.t()).coalesce())
        A_hetero_norm = self._normalize_symmetric_matrix((A_out_w_hetero + A_out_w_hetero.t()).coalesce())

        print(f"    - Undirected Homophilous Edges: {(A_out_w_homo + A_out_w_homo.t())._nnz()}")
        print(f"    - Undirected Heterophilous Edges: {(A_out_w_hetero + A_out_w_hetero.t())._nnz()}")
        return A_homo_norm, A_hetero_norm

    def create_subgraph_data_for_model(self, model_type: str,
                                       full_features: torch.Tensor, full_labels: torch.Tensor,
                                       node_subset: torch.Tensor,
                                       A_homo_norm: Optional[torch.Tensor] = None,
                                       A_hetero_norm: Optional[torch.Tensor] = None) -> 'Data':
        """
        Creates a valid, self-contained PyG Data object for a subgraph of
        nodes. This is critical for clustered training, as it correctly re-indexes
        the edges of all necessary adjacency matrices to be local to the subgraph.
        """
        # Basic properties for any subgraph
        subgraph_features = full_features[node_subset]
        subgraph_labels = full_labels[node_subset] if full_labels is not None else None

        if model_type.lower() == 'directgcn':
            # --- DEFINITIVE FIX: Correctly initialize and populate the data dictionary for DirectGCN subgraphs ---
            subgraph_data_dict = {
                'x': subgraph_features,
                'y': subgraph_labels,
                'original_indices': node_subset
            }

            path_matrices = {
                'mathcal_in': self.mathcal_A_in, 'mathcal_out': self.mathcal_A_out,
                'undirected_norm': self.A_undirected_norm_sparse
            }
            for name, matrix in path_matrices.items():
                if matrix is not None:
                    edge_index, edge_weight = subgraph(
                        node_subset, matrix.indices(), matrix.values(),
                        relabel_nodes=True, num_nodes=self.number_of_nodes
                    )
                    subgraph_data_dict[f'edge_index_{name}'] = edge_index
                    subgraph_data_dict[f'edge_weight_{name}'] = edge_weight

            # --- ANTICIPATORY DEBUGGING: Conditionally add the new matrices if they were generated. ---
            # This makes the function flexible for both heterophilic and homophilic graphs.
            if A_homo_norm is not None and A_hetero_norm is not None:
                homo_edge_index, homo_edge_weight = subgraph(node_subset, A_homo_norm.indices(), A_homo_norm.values(),
                                                             relabel_nodes=True, num_nodes=self.number_of_nodes)
                hetero_edge_index, hetero_edge_weight = subgraph(node_subset, A_hetero_norm.indices(),
                                                                 A_hetero_norm.values(),
                                                                 relabel_nodes=True, num_nodes=self.number_of_nodes)
                subgraph_data_dict.update({
                    'edge_index_homo_norm': homo_edge_index, 'edge_weight_homo_norm': homo_edge_weight,
                    'edge_index_hetero_norm': hetero_edge_index, 'edge_weight_hetero_norm': hetero_edge_weight
                })
            return Data.from_dict(subgraph_data_dict)
        else:
            # Default for other GNNs (uses undirected graph)
            edge_index, edge_weight = subgraph(node_subset, self.A_undirected_norm_sparse.indices(),
                                               self.A_undirected_norm_sparse.values(),
                                               relabel_nodes=True, num_nodes=self.number_of_nodes)
            return Data(x=subgraph_features, y=subgraph_labels, edge_index=edge_index, edge_attr=edge_weight,
                        original_indices=node_subset)
