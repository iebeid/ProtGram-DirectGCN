import tensorflow as tf

# 1- DATA MODEL
class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[str, Any], edges: List[Tuple]):
        self.nodes_map = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []
        self.edges = list(self.original_edges)
        self.node_sequences: List[str] = []

        self._create_node_indices()
        self._index_edges()
        self.integrity_check()

    def _create_node_indices(self):
        self.node_to_idx = {}
        self.idx_to_node = {}
        node_keys = set(self.nodes_map.keys())
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
            print(f"WARNING: Could not perform NetworkX integrity check during NgramGraph init. Error: {e}")


class DirectedGraph(Graph):
    """
    A specialized graph object for the ProtDiGCN model. It stores:
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

    def _create_raw_weighted_adj_matrices(self):
        self.A_out_w = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.float32)
        for s_idx, t_idx, w, *_ in self.edges:
            self.A_out_w[s_idx, t_idx] = w
        self.A_in_w = self.A_out_w.T

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

# 2- STATISTICAL MODEL



class DirectGCNLayer(tf.keras.layers.Layer):
    """
    The custom directed GCN layer in TensorFlow.

    This layer implements an architecture with separate main and shared pathways
    for both incoming and outgoing edges, along with adaptive coefficients.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int, use_vector_coeffs: bool = True, **kwargs):
        super(DirectGCNLayer, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.use_vector_coeffs = use_vector_coeffs

        # --- Weight Initializers ---
        self.xavier_uniform_initializer = tf.keras.initializers.GlorotUniform()
        self.zeros_initializer = tf.keras.initializers.Zeros()
        self.ones_initializer = tf.keras.initializers.Ones()

        # --- Main Path Weights ---
        # Dense layer for the linear transformation without bias
        self.lin_main_in = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer=self.xavier_uniform_initializer)
        self.lin_main_out = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer=self.xavier_uniform_initializer)

        # Bias terms for the main paths
        self.bias_main_in = self.add_weight(shape=(out_channels,), initializer=self.zeros_initializer, name='bias_main_in')
        self.bias_main_out = self.add_weight(shape=(out_channels,), initializer=self.zeros_initializer, name='bias_main_out')

        # --- Shared Path (Skip/W_all) Weights ---
        self.lin_shared = tf.keras.layers.Dense(out_channels, use_bias=False, kernel_initializer=self.xavier_uniform_initializer)

        # Bias terms for the shared paths
        self.bias_shared_in = self.add_weight(shape=(out_channels,), initializer=self.zeros_initializer, name='bias_shared_in')
        self.bias_shared_out = self.add_weight(shape=(out_channels,), initializer=self.zeros_initializer, name='bias_shared_out')

        # --- Adaptive Coefficients ---
        if self.use_vector_coeffs:
            self.C_in_vec = self.add_weight(shape=(self.num_nodes, 1), initializer=self.ones_initializer, name='C_in_vec')
            self.C_out_vec = self.add_weight(shape=(self.num_nodes, 1), initializer=self.ones_initializer, name='C_out_vec')
        else:
            self.C_in = self.add_weight(shape=(1,), initializer=self.ones_initializer, name='C_in')
            self.C_out = self.add_weight(shape=(1,), initializer=self.ones_initializer, name='C_out')

    def propagate(self, x, edge_index, edge_weight):
        """
        Manually implements the message passing mechanism.

        Args:
            x (tf.Tensor): The input node features.
            edge_index (tf.Tensor): The edge index tensor of shape (2, num_edges).
            edge_weight (tf.Tensor): The edge weights.

        Returns:
            tf.Tensor: The aggregated messages for each node.
        """
        # Source nodes are at index 0, target nodes are at index 1
        source_nodes, target_nodes = edge_index[0], edge_index[1]

        # Gather features of source nodes for message creation
        messages = tf.gather(x, source_nodes)

        # Apply edge weights to messages
        if edge_weight is not None:
            messages *= tf.expand_dims(edge_weight, axis=-1)

        # Aggregate messages at target nodes
        # Use segment_sum to sum messages for each target node
        aggregated_messages = tf.math.segment_sum(messages, target_nodes)

        # Ensure the output tensor has the correct shape for all nodes
        return tf.scatter_nd(tf.expand_dims(tf.unique(target_nodes)[0], axis=1), aggregated_messages, shape=(self.num_nodes, self.out_channels))

    def call(self, x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out):
        """
        Forward pass for the DirectGCNLayer.

        Args:
            x (tf.Tensor): Input node features.
            edge_index_in (tf.Tensor): Edge index for incoming edges.
            edge_weight_in (tf.Tensor): Edge weights for incoming edges.
            edge_index_out (tf.Tensor): Edge index for outgoing edges.
            edge_weight_out (tf.Tensor): Edge weights for outgoing edges.

        Returns:
            tf.Tensor: The output node features.
        """
        # --- Incoming Edges Path ---
        h_main_in = self.propagate(self.lin_main_in(x), edge_index_in, edge_weight_in)
        h_shared_in = self.propagate(self.lin_shared(x), edge_index_in, edge_weight_in)
        ic_combined = (h_main_in + self.bias_main_in) + (h_shared_in + self.bias_shared_in)

        # --- Outgoing Edges Path ---
        h_main_out = self.propagate(self.lin_main_out(x), edge_index_out, edge_weight_out)
        h_shared_out = self.propagate(self.lin_shared(x), edge_index_out, edge_weight_out)
        oc_combined = (h_main_out + self.bias_main_out) + (h_shared_out + self.bias_shared_out)

        # --- Adaptive Coefficients ---
        c_in = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out = self.C_out_vec if self.use_vector_coeffs else self.C_out

        # Combine the pathways
        return c_in * ic_combined + c_out * oc_combined

# 3- DATA BUILDER

# 4- MODEL TRAINER

# 5- MAIN PIPELINE

# 6- RESULTS