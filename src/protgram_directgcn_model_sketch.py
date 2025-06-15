from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd

import networkx as nx
import tensorflow as tf

# 1- DATA MODEL
class Graph:
    """A base class for representing a generic graph structure with nodes and edges."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        self.nodes = nodes if nodes is not None else {}
        self.node_sequences: List[str] = []
        self.original_edges = edges if edges is not None else []
        self.edges = list(self.original_edges)
        self.edges_weight_lookup: Dict[str, Any] = {}


        self._create_node_indices()
        self._index_edges()
        self.integrity_check()

    def _create_node_indices(self):
        self.node_to_idx = {}
        self.idx_to_node = {}
        node_keys = set(self.nodes.keys())
        if not node_keys and self.edges:
            for pair in self.original_edges:
                node_keys.add(str(pair[0]))
                node_keys.add(str(pair[1]))
        sorted_nodes = sorted(list(node_keys))
        for i, node_name in enumerate(sorted_nodes):
            self.node_to_idx[node_name] = i
            self.idx_to_node[i] = node_name
        self.number_of_nodes = len(self.node_to_idx)
        if self.number_of_nodes > 0:
            self.node_sequences = [self.idx_to_node[i] for i in range(self.number_of_nodes)]
        else:
            self.node_sequences = []

    def _index_edges(self):
        indexed_edge_list = []
        for es in self.edges:
            s_idx = self.node_to_idx.get(str(es[0]))
            t_idx = self.node_to_idx.get(str(es[1]))
            self.edges_weight_lookup[str(s_idx) + "-" + str(t_idx)] = float(es[2])
            if s_idx is not None and t_idx is not None:
                indexed_edge_list.append((s_idx, t_idx) + es[2:])
        self.edges = indexed_edge_list
        self.number_of_edges = len(self.edges)

    def integrity_check(self):
        if not self.edges and self.number_of_nodes == 0: return
        if not self.edges and self.number_of_nodes > 0: return
        if not self.edges: return
        weightless_edges = [(e[0], e[1]) for e in self.edges]
        try:
            graph_nx = nx.Graph()
            if self.number_of_nodes > 0:
                graph_nx.add_nodes_from(range(self.number_of_nodes))
            graph_nx.add_edges_from(weightless_edges)
        except Exception as e:
            print(f"WARNING: Could not perform NetworkX integrity check during Graph init. Error: {e}")


class DirectedGraph(Graph):
    """
    A specialized directed graph object for the ProtGram-DirectGCN model. It stores:
    - Weighted in edges and out edges.
    - Raw weighted adjacency matrices (A_out_w, A_in_w).
    - Preprocessed propagation matrices (mathcal_A_out, mathcal_A_in) using symmetric/skew-symmetric decomposition.
    - Binary adjacency matrices (out_adjacency_matrix, in_adjacency_matrix).
    """

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple], epsilon_propagation: float = 1e-9):
        super().__init__(nodes=nodes, edges=edges)
        self.epsilon_propagation = epsilon_propagation

        if self.number_of_nodes > 0:
            self._create_raw_weighted_adj_matrices()  # A_out_w, A_in_w
            self._create_binary_adj_matrices()  # out_adjacency_matrix, in_adjacency_matrix
            self._create_propagation_matrices()  # mathcal_A_out, mathcal_A_in
        else:
            self.A_out_w = np.array([], dtype=np.float32).reshape(0, 0)
            self.A_in_w = np.array([], dtype=np.float32).reshape(0, 0)
            self.out_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.in_adjacency_matrix = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_out = np.array([], dtype=np.float32).reshape(0, 0)
            self.mathcal_A_in = np.array([], dtype=np.float32).reshape(0, 0)

    def _create_in_out_edges(self):
        column_names = ['source', 'target']
        edges_df = pd.DataFrame(self.edges, columns=column_names)
        result_dict = edges_df.groupby('source')['target'].apply(list).to_dict()


    def _create_raw_weighted_adj_matrices(self):
        self.A_out_w = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_idx, t_idx, w, *_ in self.edges:
            self.A_out_w[s_idx, t_idx] = w
        self.A_in_w = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_idx, t_idx, w, *_ in self.edges:
            self.A_in_w[s_idx, t_idx] = w

    def _create_binary_adj_matrices(self):
        self.out_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        self.in_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_n, t_n, *_ in self.edges:
            self.out_adjacency_matrix[s_n, t_n] = 1
            self.in_adjacency_matrix[t_n, s_n] = 1

    def _calculate_single_propagation_matrix(self, A_w: np.ndarray) -> np.ndarray:
        """
        Helper function to calculate the propagation matrix from a weighted adjacency matrix.
        Steps:
        1. Row-wise degree normalization: A_n = D^-1 * A_w
        2. Symmetric-like component: S = (A_n + A_n.T) / 2
        3. Skew-symmetric-like component: K = (A_n - A_n.T) / 2
        4. Propagation matrix: mathcal_A_uv = sqrt(S_uv^2 + K_uv^2 + epsilon)
        """
        if A_w.shape[0] == 0:  # Handle empty matrix
            return np.array([], dtype=np.float32).reshape(0, 0)

        # 1. Degree normalization (row-wise)
        # For A_out_w, D_out_ii = sum_j (A_out_w)_ij. This is row sum.
        # For A_in_w, D_in_ii = sum_j (A_in_w)_ij. This is also row sum of A_in_w.
        row_sum = A_w.sum(axis=1)
        D_inv_diag = np.zeros_like(row_sum, dtype=np.float32)
        non_zero_degrees = row_sum != 0
        D_inv_diag[non_zero_degrees] = 1.0 / row_sum[non_zero_degrees]

        # Efficient way to do D_inv @ A_w for diagonal D_inv
        A_n = np.diag(D_inv_diag) @ A_w
        # Or element-wise: A_n = D_inv_diag[:, np.newaxis] * A_w

        # 2. Symmetric-like component
        S = (A_n + A_n.T) / 2.0

        # 3. Skew-symmetric-like component
        K = (A_n - A_n.T) / 2.0

        # 4. Propagation matrix
        mathcal_A = np.sqrt(np.square(S) + np.square(K) + self.epsilon_propagation)

        return mathcal_A

    def _create_propagation_matrices(self):
        """
        Creates the preprocessed propagation matrices mathcal_A_out and mathcal_A_in.
        """
        print(f"Creating propagation matrices (mathcal_A) with epsilon={self.epsilon_propagation}...")
        self.mathcal_A_out = self._calculate_single_propagation_matrix(self.A_out_w)
        self.mathcal_A_in = self._calculate_single_propagation_matrix(self.A_in_w)
        print("Propagation matrices mathcal_A_out and mathcal_A_in created.")
#############################################################################################

# 2- AI MODEL


import tensorflow as tf
from typing import List, Optional


class DirectGCNLayer(tf.keras.layers.Layer):
    """
    The custom directed GCN layer in TensorFlow.

    This layer implements the logic described in the user's original PyTorch
    code, with explicit pathways for incoming and outgoing edges. Message
    passing is performed using efficient sparse matrix multiplication.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.use_vector_coeffs = use_vector_coeffs

        # --- Intial Path Weights (equivalent to nn.Linear with bias=False) ---
        self.W_initial_in = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer='glorot_uniform')
        self.W_initial_out = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer='glorot_uniform')

        # --- Main Path Weights (equivalent to nn.Linear with bias=False) ---
        self.W_main_in = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer='glorot_uniform')
        self.W_main_out = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer='glorot_uniform')

        # --- Shared Path Weights ---
        self.W_shared = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer='glorot_uniform')

    def build(self, input_shape):
        """
        Create the layer's weights and biases. This method is called automatically
        by Keras the first time the layer is used.
        """

        # --- Initial Path Biases (equivalent to nn.Parameter) ---
        self.bias_initial_in = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_initial_in')
        self.bias_initial_out = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_initial_out')

        # --- Main Path Biases (equivalent to nn.Parameter) ---
        self.bias_main_in = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_main_in')
        self.bias_main_out = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_main_out')

        # --- Shared Path Biases ---
        self.bias_shared_in = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_shared_in')
        self.bias_shared_out = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias_shared_out')

        # --- Adaptive Coefficients ---
        if self.use_vector_coeffs:
            self.C_in_vec = self.add_weight(shape=(self.num_nodes, 1), initializer='ones', trainable=True, name='C_in_vec')
            self.C_out_vec = self.add_weight(shape=(self.num_nodes, 1), initializer='ones', trainable=True, name='C_out_vec')
        else:
            self.C_in = self.add_weight(shape=(1,), initializer='ones', trainable=True, name='C_in')
            self.C_out = self.add_weight(shape=(1,), initializer='ones', trainable=True, name='C_out')

        super().build(input_shape)

    # def _propagate(self, x: tf.Tensor, edge_index: tf.Tensor, edge_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    #     """
    #     Performs message passing using sparse matrix multiplication.
    #
    #     Args:
    #         x: Node features tensor of shape [num_nodes, num_features].
    #         edge_index: Edge indices tensor of shape [2, num_edges].
    #         edge_weight: Edge weights tensor of shape [num_edges,].
    #
    #     Returns:
    #         The aggregated messages for each node.
    #     """
    #     if edge_weight is None:
    #         edge_weight = tf.ones(tf.shape(edge_index)[1])
    #
    #     # Create a SparseTensor for the adjacency matrix
    #     adj_sparse = tf.SparseTensor(
    #         indices=tf.transpose(edge_index),  # Shape: [num_edges, 2]
    #         values=edge_weight,
    #         dense_shape=(self.num_nodes, self.num_nodes)
    #     )
    #
    #     # Perform the graph convolution: A * X
    #     return tf.sparse.sparse_dense_matmul(adj_sparse, x)

    def call(self, X: tf.Tensor, A_in_binary: tf.Tensor, A_in_weighted: tf.Tensor, A_out_binary: tf.Tensor, A_out_weighted: tf.Tensor, N_in: tf.Tensor, N_out: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the DirectGCN layer.
        """

        tf.sparse.sparse_dense_matmul(tf.SparseTensor(indices=tf.transpose(edge_index), values=edge_weight, dense_shape=(self.num_nodes, self.num_nodes)), x)

        # --- Incoming Edges Path ---
        weights_in = (self.W_initial_in(X) + self.bias_initial_in) + (A_in_binary * (self.W_main_in(X) + self.bias_main_in)) + (A_in_binary * (self.W_shared(X) + self.bias_shared_in))




        # --- Incoming Edges Path ---
        h_main_in = self._propagate(self.lin_main_in(x), edge_index_in, edge_weight_in)
        h_shared_in = self._propagate(self.lin_shared(x), edge_index_in, edge_weight_in)
        ic_combined = (h_main_in + self.bias_main_in) + (h_shared_in + self.bias_shared_in)

        # --- Outgoing Edges Path ---
        h_main_out = self._propagate(self.lin_main_out(x), edge_index_out, edge_weight_out)
        h_shared_out = self._propagate(self.lin_shared(x), edge_index_out, edge_weight_out)
        oc_combined = (h_main_out + self.bias_main_out) + (h_shared_out + self.bias_shared_out)

        # --- Combine Paths with Adaptive Coefficients ---
        c_in = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out = self.C_out_vec if self.use_vector_coeffs else self.C_out

        return c_in * ic_combined + c_out * oc_combined

    class DirectGCNModel(tf.keras.Model):
        """
        The main GCN model in TensorFlow, with a dynamic number of GCN and
        residual layers.
        """

        def __init__(self, layer_dims: List[int], num_graph_nodes: int, task_num_output_classes: int, n_gram_len: int, one_gram_dim: int, max_pe_len: int, dropout: float, use_vector_coeffs: bool, l2_eps: float = 1e-12,
                     **kwargs):
            super().__init__(**kwargs)
            self.n_gram_len = n_gram_len
            self.one_gram_dim = one_gram_dim
            self.l2_eps = l2_eps

            # --- Layer Definitions ---
            self.pe_layer = tf.keras.layers.Embedding(max_pe_len, one_gram_dim) if one_gram_dim > 0 and max_pe_len > 0 else None

            self.convs = []
            self.res_projs = []

            for i in range(len(layer_dims) - 1):
                in_dim = layer_dims[i]
                out_dim = layer_dims[i + 1]
                self.convs.append(DirectGCNLayer(in_dim, out_dim, num_graph_nodes, use_vector_coeffs))
                # Use a Dense layer for projection if dimensions differ, else identity
                self.res_projs.append(
                    tf.keras.layers.Dense(out_dim) if in_dim != out_dim else tf.keras.layers.Lambda(lambda t: t)
                )

            self.dropout_layer = tf.keras.layers.Dropout(dropout)
            self.decoder_fc = tf.keras.layers.Dense(task_num_output_classes)

        def _apply_pe(self, x: tf.Tensor) -> tf.Tensor:
            """Applies positional embeddings to the input tensor."""
            if self.pe_layer is None or self.n_gram_len == 0 or self.one_gram_dim == 0:
                return x

            expected_dim = self.n_gram_len * self.one_gram_dim
            if x.shape[1] != expected_dim:
                # Silently skip if dimensions do not match the expected structure
                return x

            x_reshaped = tf.reshape(x, (-1, self.n_gram_len, self.one_gram_dim))

            # Number of positions to apply encoding to
            pos_to_enc = min(self.n_gram_len, self.pe_layer.input_dim)
            if pos_to_enc > 0:
                pos_indices = tf.range(pos_to_enc, dtype=tf.int32)
                pe = self.pe_layer(pos_indices)  # Shape: [pos_to_enc, one_gram_dim]

                # Add positional embeddings
                x_pe_part = x_reshaped[:, :pos_to_enc, :] + pe
                x_reshaped = tf.concat([x_pe_part, x_reshaped[:, pos_to_enc:, :]], axis=1)

            return tf.reshape(x_reshaped, (-1, expected_dim))

        def call(self, inputs, training=False):
            """
            Forward pass for the ProtGramDirectGCN model.

            Args:
                inputs: A tuple or dictionary containing the feature matrix and edge information.
                        Expected keys: 'x', 'edge_index_in', 'edge_weight_in',
                                       'edge_index_out', 'edge_weight_out'.
                training: A boolean flag passed by Keras to indicate if the model
                          is in training mode (for dropout).

            Returns:
                A tuple containing the log-softmaxed task logits and the
                L2-normalized final embeddings.
            """
            x, ei_in, ew_in, ei_out, ew_out = inputs['x'], inputs['edge_index_in'], inputs['edge_weight_in'], inputs['edge_index_out'], inputs['edge_weight_out']

            h = self._apply_pe(x)

            for i in range(len(self.convs)):
                h_res = h
                # Apply GCN layer, residual connection, and activation
                gcn_out = self.convs[i](h, ei_in, ew_in, ei_out, ew_out)
                res_out = self.res_projs[i](h_res)
                h = tf.nn.tanh(gcn_out + res_out)
                # Apply dropout
                h = self.dropout_layer(h, training=training)

            final_embed_for_task = h
            task_logits = self.decoder_fc(final_embed_for_task)

            # L2 normalize the final embeddings
            final_normalized_embeddings = tf.math.l2_normalize(final_embed_for_task, axis=-1, epsilon=self.l2_eps)

            return tf.nn.log_softmax(task_logits, axis=-1), final_normalized_embeddings

# 3- DATA BUILDER

# 4- MODEL TRAINER

# 5- MAIN PIPELINE

# 6- RESULTS