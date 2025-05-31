import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, RGCNConv
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time
import os
import h5py
import pickle
import json
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import gc
from typing import List, Tuple, Dict, Iterator, Callable, Optional, Union

# Attempt to import DiGCNConv
try:
    from torch_geometric_signed_directed.nn.directed import DiGCNConv

    HAS_TONG_DiGCN_LIB = True
except ImportError:
    HAS_TONG_DiGCN_LIB = False
    print("Warning: 'torch_geometric_signed_directed' not found. 'Tong_Library_DiGCN' will be skipped.")
    DiGCNConv = None

# Attempt to import PyG's label_propagation
try:
    from torch_geometric.utils import label_propagation

    PYG_LABEL_PROPAGATION_AVAILABLE = True
except ImportError:
    print(
        "Warning: torch_geometric.utils.label_propagation not found. Will use NetworkX for community detection if needed.")
    PYG_LABEL_PROPAGATION_AVAILABLE = False
    label_propagation = None

# --- 0. Configuration & Setup ---
SEED = 42
KFOLDS = 10
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.005
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_DROPOUT = 0.3
INITIAL_CHAR_FEATURE_DIM = 64

OUTPUT_BASE_DIR_FOR_THIS_SCRIPT = "C:/tmp/Models/"
CHAR_GRAPH_BASE_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/models/"

CHAR_GRAPH_INPUT_DIR = os.path.normpath(os.path.join(CHAR_GRAPH_BASE_INPUT_DIR, "output_global_character_graphs"))
GLOBAL_UNDIRECTED_GRAPH_PKL = os.path.normpath(
    os.path.join(CHAR_GRAPH_INPUT_DIR, "global_undirected_character_graph.pkl"))
GLOBAL_DIRECTED_GRAPH_PKL = os.path.normpath(os.path.join(CHAR_GRAPH_INPUT_DIR, "global_directed_character_graph.pkl"))
CHAR_VOCAB_JSON = os.path.normpath(os.path.join(CHAR_GRAPH_INPUT_DIR, "global_char_graph_vocab_mappings.json"))

DATASET_TAG_PREFIX = "GlobalCharGraph_RandInitFeat_v2"
SAVE_EMBEDDINGS = True
EMBEDDINGS_OUTPUT_DIR = os.path.normpath(
    os.path.join(OUTPUT_BASE_DIR_FOR_THIS_SCRIPT, f"output_char_embeddings_{DATASET_TAG_PREFIX}"))
PLOTS_OUTPUT_DIR = os.path.normpath(os.path.join(OUTPUT_BASE_DIR_FOR_THIS_SCRIPT,
                                                 f"training_history_plots_{DATASET_TAG_PREFIX}"))  # Specific dir for these plots
RESULTS_TABLE_FILE = os.path.normpath(
    os.path.join(OUTPUT_BASE_DIR_FOR_THIS_SCRIPT, f"results_summary_char_gnns_{DATASET_TAG_PREFIX}.txt"))

COMMON_EMBEDDING_DIM_PCA = 64
APPLY_PCA_TO_GNN_EMBEDDINGS = True
PLOT_TRAINING_HISTORY = True

torch.manual_seed(SEED);
np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Shape:
    def __init__(self, i, o, b): self.in_size, self.out_size, self.batch_size = i, o, b

    def __str__(self): return f"(In: {self.in_size}, Out: {self.out_size}, Batch: {self.batch_size})"


class Graph:
    def __init__(self, nodes=None, edges=None):
        self.original_nodes_dict = nodes if nodes is not None else {};
        self.original_edges = edges if edges is not None else []
        self.nodes, self.edges, self.node_index, self.node_inverted_index = {}, [], {}, {}
        self.number_of_nodes, self.number_of_edges, self._features, self.dimensions = 0, 0, None, 0
        self.node_indices();
        self.edge_indices()

    def node_indices(self):
        current_map = self.original_nodes_dict
        if not current_map and self.original_edges:
            all_ids = set(str(e[i]) for e in self.original_edges if len(e) >= 2 for i in range(2))
            current_map = {nid: nid for nid in sorted(list(all_ids))};
            self.original_nodes_dict = current_map
        sorted_ids = sorted([str(k) for k in current_map.keys()])
        self.nodes.clear();
        self.node_index.clear();
        self.node_inverted_index.clear()
        for i, id_str in enumerate(sorted_ids):
            self.node_index[id_str] = i;
            self.node_inverted_index[i] = id_str
            self.nodes[i] = current_map.get(id_str, id_str)
        self.number_of_nodes = len(self.nodes)

    def edge_indices(self):
        if not self.node_index or self.number_of_nodes == 0: self.edges, self.number_of_edges = [], 0;return
        idx_edges = []
        for e_orig in self.original_edges:
            if len(e_orig) >= 2:
                s_orig, t_orig = str(e_orig[0]), str(e_orig[1])
                if s_orig in self.node_index and t_orig in self.node_index:
                    s_idx, t_idx = self.node_index[s_orig], self.node_index[t_orig]
                    idx_edges.append((s_idx, t_idx) + tuple(e_orig[2:]))
        self.edges, self.number_of_edges = idx_edges, len(idx_edges)

    @property
    def features(self):
        if self._features is None: self.set_node_features(
            np.eye(self.number_of_nodes, dtype=np.float32) if self.number_of_nodes > 0 else np.array([],
                                                                                                     dtype=np.float32).reshape(
                0, 0))
        return self._features

    def set_node_features(self, features_array):
        self._features, self.dimensions = features_array, (
            features_array.shape[1] if features_array.ndim == 2 and features_array.shape[0] > 0 else 0)


class UndirectedGraph(Graph):
    def __init__(self, edges=None, nodes=None):
        super().__init__(nodes=nodes, edges=edges); self._adjacency, self._degree_normalized_adjacency = None, None

    @property
    def adjacency(self):
        if self._adjacency is None:
            if self.number_of_nodes == 0: self._adjacency = np.array([], dtype=float).reshape(0,
                                                                                              0); return self._adjacency
            adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            for s, t, w, *_ in self.edges: adj[s, t] += float(w); adj[t, s] += float(w)
            self._adjacency = adj
        return self._adjacency

    @property
    def degree_normalized_adjacency(self):
        if self._degree_normalized_adjacency is None:
            if self.number_of_nodes == 0: self._degree_normalized_adjacency = np.array([], dtype=float).reshape(0,
                                                                                                                0); return self._degree_normalized_adjacency
            adj, I = self.adjacency.copy(), np.eye(self.number_of_nodes, dtype=float);
            adj_tilde = adj + I
            deg_tilde_vec = np.sum(adj_tilde, axis=0);
            inv_sqrt_deg = np.power(deg_tilde_vec, -0.5, where=deg_tilde_vec != 0, out=np.zeros_like(deg_tilde_vec))
            D_inv_sqrt = np.diag(inv_sqrt_deg);
            self._degree_normalized_adjacency = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
        return self._degree_normalized_adjacency


class DirectedGraph(Graph):
    def __init__(self, edges=None, nodes=None):
        super().__init__(nodes=nodes, edges=edges); self._out_adjacency = None

    @property
    def out_adjacency(self):
        if self._out_adjacency is None:
            if self.number_of_nodes == 0: self._out_adjacency = np.array([], dtype=float).reshape(0,
                                                                                                  0); return self._out_adjacency
            adj = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            for s, t, w, *_ in self.edges: adj[s, t] += float(w)
            self._out_adjacency = adj
        return self._out_adjacency


def load_pickled_graph(filepath: str):
    filepath = os.path.normpath(filepath);
    print(f"Loading pickled graph from: {filepath}")
    if not os.path.exists(filepath): print(f"Error: Pickled graph file not found at {filepath}");return None
    try:
        with open(filepath, 'rb') as f:
            graph_obj = pickle.load(f)
        if not (hasattr(graph_obj, 'node_indices') and hasattr(graph_obj, 'edge_indices')): print(
            f"Error: Loaded object from {filepath} not valid graph.");return None
        graph_obj.node_indices();
        graph_obj.edge_indices()
        print(
            f"Successfully loaded graph: {type(graph_obj).__name__} with {graph_obj.number_of_nodes} nodes and {graph_obj.number_of_edges} valid edge entries.")
        return graph_obj
    except Exception as e:
        print(f"Error loading pickled graph from {filepath}: {e}");return None


def load_char_vocabulary(filepath: str):
    filepath = os.path.normpath(filepath);
    print(f"Loading character vocabulary from: {filepath}")
    if not os.path.exists(filepath): print(f"Error: Character vocabulary file not found at {filepath}");return None
    try:
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        print(f"Successfully loaded character vocabulary (size: {len(vocab_data.get('ascii_char_to_int_id', {}))}).")
        return vocab_data
    except Exception as e:
        print(f"Error loading character vocabulary from {filepath}: {e}");return None


def convert_custom_char_graph_to_pyg_data(custom_graph_obj, char_vocab_data: Dict, dataset_name="GlobalCharGraph"):
    if custom_graph_obj is None or char_vocab_data is None: return None, 0
    print(f"Converting custom graph '{dataset_name}' to PyG Data object...")
    num_nodes = custom_graph_obj.number_of_nodes
    if num_nodes == 0: print(f"Error: Graph {dataset_name} has 0 nodes.");return None, 0
    print(f"  Initializing node features. Dim: {INITIAL_CHAR_FEATURE_DIM}")
    x = torch.empty(num_nodes, INITIAL_CHAR_FEATURE_DIM, dtype=torch.float);
    torch.nn.init.xavier_uniform_(x) if INITIAL_CHAR_FEATURE_DIM > 0 else None
    edges_data = [(e[0], e[1], float(e[2]), int(e[3]) if len(e) > 3 else 0) for e in custom_graph_obj.edges]
    edge_list_u, edge_list_v, edge_weights, edge_types_for_rgcn = zip(*edges_data) if edges_data else ([], [], [], [])
    edge_index = torch.tensor([edge_list_u, edge_list_v], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) if edge_weights else None
    pyg_edge_type = torch.tensor(edge_types_for_rgcn, dtype=torch.long) if edge_types_for_rgcn else None
    print(f"  Applying community detection for node labels on {dataset_name}...")
    num_classes = 0;
    y = torch.zeros(num_nodes, dtype=torch.long);
    community_detection_attempted = False
    if num_nodes > 1 and edge_index.numel() > 0 and edge_index.size(1) > 0:
        if edge_index.max() >= num_nodes:
            print(f"  Warning ({dataset_name}): Max edge idx out of bounds. Skipping community detection.")
        elif PYG_LABEL_PROPAGATION_AVAILABLE and label_propagation is not None:
            community_detection_attempted = True
            try:
                lp_labels = label_propagation(edge_index, num_nodes=num_nodes, max_iter=100)
                unique_labels = lp_labels.unique(sorted=True);
                num_classes = unique_labels.size(0)
                label_map = {label_val.item(): i for i, label_val in enumerate(unique_labels)}
                y = torch.tensor([label_map[label_val.item()] for label_val in lp_labels], dtype=torch.long)
                print(f"  PyG Label Propagation found {num_classes} communities for {dataset_name}.")
                if num_classes == 0: print(f"  Warning: PyG Label Propagation resulted in 0 communities.")
            except Exception as e:
                print(
                    f"  PyG Label Propagation failed for {dataset_name}: {e}. Trying NetworkX fallback.");num_classes = 0;community_detection_attempted = False
        if not community_detection_attempted or num_classes == 0:
            print(f"  Attempting NetworkX community detection for {dataset_name}...")
            nx_data = Data(edge_index=edge_index, num_nodes=num_nodes)
            nx_g = torch_geometric.utils.to_networkx(nx_data, to_undirected=True)
            if nx_g.number_of_nodes() > 0 and nx_g.number_of_edges() > 0:
                try:
                    communities_generator = nx.community.greedy_modularity_communities(nx_g, weight=None)
                    communities = [list(c) for c in communities_generator]
                    if communities and isinstance(communities, list) and len(communities) > 0:
                        num_classes = len(communities);
                        [y.scatter_(0, torch.tensor(list(c), dtype=torch.long), i_comm) for i_comm, c in
                         enumerate(communities)]
                        print(f"  NetworkX Greedy Modularity found {num_classes} communities for {dataset_name}.")
                    else:
                        print(
                            f"  NetworkX Greedy Modularity returned no communities for {dataset_name}.");num_classes = 0
                except Exception as e_nx:
                    print(f"  NetworkX Community detection failed for {dataset_name}: {e_nx}.");num_classes = 0
            else:
                print(f"  NX graph empty for NetworkX community detection on {dataset_name}.");num_classes = 0
    if num_classes == 0 and num_nodes > 0:
        if num_nodes == 1:
            y[0] = 0;num_classes = 1
        else:
            y = torch.arange(num_nodes, dtype=torch.long);num_classes = num_nodes
        print(f"  Using node indices as fallback labels for {dataset_name} ({num_classes} classes).")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y);
    data.num_nodes = num_nodes
    if pyg_edge_type is not None and pyg_edge_type.numel() > 0: data.edge_type = pyg_edge_type
    data.node_original_ids = [custom_graph_obj.node_inverted_index[i] for i in range(num_nodes)]
    print(f"  {dataset_name} PyG Data: {data.num_nodes}N, {data.num_edges}E, {data.num_features}F, {num_classes}C.")
    return data, num_classes


class BaseGNN(nn.Module):
    def __init__(self): super().__init__();self.embedding_output = None

    def get_embeddings(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        _, embedding = self.forward(x, edge_index, edge_weight=edge_weight, edge_type=edge_type, num_nodes=num_nodes)
        return embedding


class DirectionalConvLayer_PyTorch(nn.Module):
    def __init__(self, i, o): super().__init__();self.W_all = nn.Parameter(
        torch.Tensor(i, o));self.b_all = nn.Parameter(torch.Tensor(o));self.W_in = nn.Parameter(
        torch.Tensor(i, o));self.b_in = nn.Parameter(torch.Tensor(o));self.W_out = nn.Parameter(
        torch.Tensor(i, o));self.b_out = nn.Parameter(torch.Tensor(o));self.C_in = nn.Parameter(
        torch.Tensor(1));self.C_out = nn.Parameter(torch.Tensor(1));self.reset_parameters()

    def reset_parameters(self): [nn.init.xavier_uniform_(w) for w in [self.W_all, self.W_in, self.W_out]];[
        nn.init.zeros_(b) for b in [self.b_all, self.b_in, self.b_out]];[nn.init.xavier_uniform_(c.unsqueeze(0)) for c
                                                                         in [self.C_in, self.C_out]]

    def _apply_conv(self, a, f, W, b, We, be): mf = f @ W;mp = a @ mf if not a.is_sparse else torch.sparse.mm(a,
                                                                                                              mf);m = mp + b;ef = f @ We;ep = a @ ef if not a.is_sparse else torch.sparse.mm(
        a, ef);e = ep + be;return m + e

    def forward(self, x, ai, ao): ic = self._apply_conv(ai, x, self.W_in, self.b_in, self.W_all,
                                                        self.b_all);oc = self._apply_conv(ao, x, self.W_out, self.b_out,
                                                                                          self.W_all,
                                                                                          self.b_all);return (
                self.C_in * ic) + (self.C_out * oc)


class UserCustomDiGCN_PyTorch(BaseGNN):
    def __init__(self, nf, nc, hc1=16, hc2=16, dp=0.5, norm_in=True):
        super().__init__();
        self.norm_in = norm_in;
        self.dp = dp;
        self.dgl1 = DirectionalConvLayer_PyTorch(nf, hc1);
        self.dgl2 = DirectionalConvLayer_PyTorch(hc1, hc2);
        self.out_linear = nn.Linear(hc2, nc);
        self._key = None;
        self._adj_in = None;
        self._adj_out = None

    def _prep_adjs(self, x, ei, nn, ew=None, dev='cpu'):
        cew = ew;
        if nn == 0: return torch.empty((0, 0), device=dev), torch.empty((0, 0), device=dev)
        if cew is None and ei.numel() > 0:
            cew = torch.ones(ei.size(1), dtype=x.dtype, device=dev)
        elif ei.numel() == 0:
            return torch.zeros((nn, nn), dtype=x.dtype, device=dev), torch.zeros((nn, nn), dtype=x.dtype, device=dev)
        a_out_w = torch_geometric.utils.to_dense_adj(ei, edge_attr=cew, max_num_nodes=nn)[0]
        a_in_w = torch_geometric.utils.to_dense_adj(ei[[1, 0]], edge_attr=cew, max_num_nodes=nn)[0]
        D_out_d = a_out_w.sum(1);
        D_in_d = a_in_w.sum(1)
        D_out_inv_d = torch.zeros_like(D_out_d);
        D_out_inv_d[D_out_d != 0] = 1. / D_out_d[D_out_d != 0]
        D_in_inv_d = torch.zeros_like(D_in_d);
        D_in_inv_d[D_in_d != 0] = 1. / D_in_d[D_in_d != 0]
        A_out_n = D_out_inv_d.unsqueeze(-1) * a_out_w;
        A_in_n = D_in_inv_d.unsqueeze(-1) * a_in_w
        ir = (A_in_n + A_in_n.T) / 2;
        ii = (A_in_n - A_in_n.T) / 2;
        fai = torch.sqrt(ir.pow(2) + ii.pow(2) + 1e-9)
        our = (A_out_n + A_out_n.T) / 2;
        oui = (A_out_n - A_out_n.T) / 2;
        fao = torch.sqrt(our.pow(2) + oui.pow(2) + 1e-9)
        return fai.to(dev), fao.to(dev)

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        nn_curr = num_nodes if num_nodes is not None else (x.size(0) if x is not None and hasattr(x, 'size') else 0)
        if nn_curr == 0:
            out_feat_dgl2 = self.out_linear.in_features
            self.embedding_output = torch.empty(0, out_feat_dgl2, device=x.device if x is not None else 'cpu')
            return torch.empty(0, self.out_linear.out_features,
                               device=x.device if x is not None else 'cpu'), self.embedding_output
        current_x = x if x is not None else torch.empty(nn_curr, 0, device='cpu')
        key = (id(edge_index), nn_curr, id(current_x), id(edge_weight))
        if self._key != key or self._adj_in is None or self._adj_out is None or \
                (self._adj_in is not None and self._adj_in.size(0) != nn_curr):
            self._adj_in, self._adj_out = self._prep_adjs(current_x, edge_index, nn_curr, edge_weight,
                                                          current_x.device);
            self._key = key
        adj_in, adj_out = self._adj_in, self._adj_out
        if self.norm_in and current_x.numel() > 0 and current_x.size(1) > 0: xn = torch.linalg.norm(current_x, 2, 1,
                                                                                                    True);current_x = current_x / (
                    xn + 1e-9)
        h = self.dgl1(current_x, adj_in, adj_out);
        h = torch.tanh(h);
        h = F.dropout(h, p=self.dp, training=self.training)
        emb_from_dgl2 = self.dgl2(h, adj_in, adj_out)
        if emb_from_dgl2.ndim == 3 and emb_from_dgl2.size(0) == nn_curr and emb_from_dgl2.size(1) == nn_curr:
            emb_from_dgl2 = emb_from_dgl2.mean(dim=1)
        elif emb_from_dgl2.ndim != 2 and emb_from_dgl2.size(0) == nn_curr:
            emb_from_dgl2 = emb_from_dgl2.view(nn_curr, -1)
            if emb_from_dgl2.size(1) != self.out_linear.in_features:
                emb_from_dgl2 = torch.zeros(nn_curr, self.out_linear.in_features, device=h.device)
        elif emb_from_dgl2.ndim != 2:
            emb_from_dgl2 = torch.zeros(nn_curr, self.out_linear.in_features, device=h.device)
        self.embedding_output = torch.tanh(emb_from_dgl2)
        out = self.out_linear(self.embedding_output)
        return out, self.embedding_output


class CustomGCN(BaseGNN):
    def __init__(self, nf, nc, hc1=64, hc2=32, dp=DEFAULT_DROPOUT): super().__init__();self.c1 = GCNConv(nf,
                                                                                                         hc1);self.c2 = GCNConv(
        hc1, hc2);self.c3 = GCNConv(hc2, nc);self.dp = dp

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        if x.size(0) == 0: self.embedding_output = torch.empty(0, self.c2.out_channels,
                                                               device=x.device);return torch.empty(0,
                                                                                                   self.c3.out_channels,
                                                                                                   device=x.device), self.embedding_output
        h = self.c1(x, edge_index, edge_weight);
        h = torch.tanh(h);
        h = F.dropout(h, p=self.dp, training=self.training)
        self.embedding_output = self.c2(h, edge_index, edge_weight)
        h_activated_emb = torch.tanh(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c3(h_emb_dropped, edge_index, edge_weight)
        return logits, self.embedding_output


class GATNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, h=8, dp=DEFAULT_DROPOUT): super().__init__();self.dp = dp;self.c1 = GATConv(nf,
                                                                                                                  hc,
                                                                                                                  heads=h,
                                                                                                                  dropout=dp);self.c2 = GATConv(
        hc * h, nc, heads=1, concat=False, dropout=dp)

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        if x.size(0) == 0: self.embedding_output = torch.empty(0,
                                                               self.c1.out_channels * self.c1.heads if hasattr(self.c1,
                                                                                                               'heads') else self.c1.out_channels,
                                                               device=x.device);return torch.empty(0,
                                                                                                   self.c2.out_channels,
                                                                                                   device=x.device), self.embedding_output
        h = F.dropout(x, p=self.dp, training=self.training);
        self.embedding_output = self.c1(h, edge_index, edge_attr=edge_weight)
        h_activated_emb = F.elu(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c2(h_emb_dropped, edge_index, edge_attr=edge_weight)
        return logits, self.embedding_output


class GraphSAGENet(BaseGNN):  # Ensured this class is defined
    def __init__(self, nf, nc, hc=128, dp=DEFAULT_DROPOUT):
        super().__init__();
        self.dp = dp;
        self.c1 = SAGEConv(nf, hc);
        self.c2 = SAGEConv(hc, nc)

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        if x.size(0) == 0: self.embedding_output = torch.empty(0, self.c1.out_channels,
                                                               device=x.device); return torch.empty(0,
                                                                                                    self.c2.out_channels,
                                                                                                    device=x.device), self.embedding_output
        self.embedding_output = self.c1(x, edge_index)
        h_activated_emb = F.relu(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c2(h_emb_dropped, edge_index)
        return logits, self.embedding_output


class GINNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, dp=DEFAULT_DROPOUT):
        super().__init__();
        self.dp = dp;
        mlp_hidden_dim = hc
        self.m1 = nn.Sequential(nn.Linear(nf, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, hc));
        self.c1 = GINConv(self.m1, train_eps=True)
        self.m2 = nn.Sequential(nn.Linear(hc, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, nc));
        self.c2 = GINConv(self.m2, train_eps=True)

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
        if x.size(0) == 0: self.embedding_output = torch.empty(0, self.m1[-1].out_features,
                                                               device=x.device);return torch.empty(0, self.m2[
            -1].out_features, device=x.device), self.embedding_output
        self.embedding_output = self.c1(x, edge_index)
        h_activated_emb = F.relu(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c2(h_emb_dropped, edge_index)
        return logits, self.embedding_output


if HAS_TONG_DiGCN_LIB and DiGCNConv is not None:
    class TongLibraryDiGCNNet(BaseGNN):
        def __init__(self, nf, nc, hc1=64, hc2=32, dp=DEFAULT_DROPOUT):
            super().__init__()
            self.nf = nf;
            self.nc = nc;
            self.hc1 = hc1;
            self.hc2 = hc2;
            self.dp = dp  # Corrected assignments
            self.c1 = DiGCNConv(nf, hc1);
            self.c2 = DiGCNConv(hc1, hc2);
            self.c3 = DiGCNConv(hc2, nc)

        def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):
            if x.size(0) == 0: out_c2 = self.hc2;out_c3 = self.nc;self.embedding_output = torch.empty(0, out_c2,
                                                                                                      device=x.device);return torch.empty(
                0, out_c3, device=x.device), self.embedding_output
            if edge_index.numel() == 0: out_c2 = self.hc2;out_c3 = self.nc;self.embedding_output = torch.zeros(
                x.size(0), out_c2, device=x.device);logits = torch.zeros(x.size(0), out_c3,
                                                                         device=x.device);return logits, self.embedding_output
            cew = edge_weight;
            if cew is None and edge_index.numel() > 0: cew = torch.ones(edge_index.size(1), dtype=x.dtype,
                                                                        device=x.device)
            h = self.c1(x, edge_index, edge_weight=cew);
            h = torch.tanh(h);
            h = F.dropout(h, p=self.dp, training=self.training)
            self.embedding_output = self.c2(h, edge_index, edge_weight=cew)
            h_activated_emb = torch.tanh(self.embedding_output);
            h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
            logits = self.c3(h_emb_dropped, edge_index, edge_weight=cew)
            return logits, self.embedding_output
else:
    TongLibraryDiGCNNet = None


class RGCNNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, num_relations=1, dp=DEFAULT_DROPOUT):
        super().__init__();
        self.dp = dp;
        self.c1 = RGCNConv(nf, hc, num_relations=num_relations);
        self.c2 = RGCNConv(hc, nc, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type, edge_weight=None, num_nodes=None):
        if x.size(0) == 0: self.embedding_output = torch.empty(0, self.c1.out_channels,
                                                               device=x.device);return torch.empty(0,
                                                                                                   self.c2.out_channels,
                                                                                                   device=x.device), self.embedding_output
        current_edge_type = edge_type
        if current_edge_type is None: current_edge_type = torch.zeros(edge_index.size(1), dtype=torch.long,
                                                                      device=edge_index.device) if edge_index.numel() > 0 else torch.empty(
            0, dtype=torch.long, device=edge_index.device)
        if edge_index.numel() == 0 and current_edge_type.numel() != 0: current_edge_type = torch.empty(0,
                                                                                                       dtype=torch.long,
                                                                                                       device=edge_index.device)
        self.embedding_output = self.c1(x, edge_index, current_edge_type)
        h_activated_emb = F.relu(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c2(h_emb_dropped, edge_index, current_edge_type)
        return logits, self.embedding_output


def get_model(model_name, num_node_features, num_classes_val, device, num_relations_for_rgcn=1):
    dp = DEFAULT_DROPOUT;
    hc1 = 64;
    hc2 = 32;
    hc_large = 128;
    model = None
    if model_name == "CustomGCN":
        model = CustomGCN(num_node_features, num_classes_val, hc1, hc2, dp)
    elif model_name == "GAT":
        gat_heads = 4;model = GATNet(num_node_features, num_classes_val, hc=hc1 // gat_heads, h=gat_heads, dp=0.6)
    elif model_name == "GraphSAGE":
        model = GraphSAGENet(num_node_features, num_classes_val, hc_large, dp)
    elif model_name == "GIN":
        model = GINNet(num_node_features, num_classes_val, hc1, dp)
    elif model_name == "Tong_Library_DiGCN" and HAS_TONG_DiGCN_LIB and TongLibraryDiGCNNet is not None:
        model = TongLibraryDiGCNNet(num_node_features, num_classes_val, hc1, hc2, dp)
    elif model_name == "UserCustomDiGCN":
        model = UserCustomDiGCN_PyTorch(num_node_features, num_classes_val, hc1=hc1, hc2=hc2, dp=dp, norm_in=True)
    elif model_name == "RGCN":
        model = RGCNNet(num_node_features, num_classes_val, hc=hc1, num_relations=num_relations_for_rgcn, dp=dp)
    if model is None: raise ValueError(
        f"Unknown model:{model_name} or library/class definition missing (HAS_TONG_DiGCN_LIB:{HAS_TONG_DiGCN_LIB}, TongClassDefined: {TongLibraryDiGCNNet is not None}).")
    return model.to(device)


def reduce_dimensionality_with_pca(embeddings_array_np, target_dim, model_name_for_log=""):
    if embeddings_array_np.ndim == 1: embeddings_array_np = embeddings_array_np.reshape(-1, 1)
    n_samples, original_dim = embeddings_array_np.shape
    if n_samples == 0 or original_dim == 0: print(
        f"PCA: Empty/0-dim for {model_name_for_log}.Skipped.");return embeddings_array_np, original_dim
    effective_target_dim = min(target_dim, n_samples - 1 if n_samples > 1 else 1, original_dim)
    if n_samples <= 1: effective_target_dim = min(original_dim, 1)
    if original_dim <= target_dim: print(
        f"PCA: Orig dim {original_dim}<=user target {target_dim} for {model_name_for_log}.No PCA.");return embeddings_array_np, original_dim
    if effective_target_dim < 1 or effective_target_dim >= original_dim: print(
        f"PCA: No reduction for {model_name_for_log} (orig:{original_dim},eff_target:{effective_target_dim},samples:{n_samples}).PCA skipped.");return embeddings_array_np, original_dim
    print(
        f"PCA:Applying to {model_name_for_log}.Orig:{original_dim},Target:{effective_target_dim},Samples:{n_samples}.")
    pca = PCA(n_components=effective_target_dim, random_state=SEED)
    try:
        transformed_embeddings = pca.fit_transform(embeddings_array_np);print(
            f"PCA applied for {model_name_for_log}.New dim:{transformed_embeddings.shape[1]}");return transformed_embeddings, \
        transformed_embeddings.shape[1]
    except Exception as e:
        print(
            f"PCA Error for {model_name_for_log}(target_dim {effective_target_dim}):{e}.Returning original.");return embeddings_array_np, original_dim


def save_node_embeddings_to_h5(embeddings_dict_to_save: Dict[str, np.ndarray], output_h5_path: str, model_name_tag: str,
                               dataset_name_tag: str, final_dim: int):
    output_h5_path = os.path.normpath(output_h5_path)
    print(f"Saving node (character) embeddings for {model_name_tag} (dim: {final_dim}) to {output_h5_path}...")
    try:
        parent_dir_h5 = os.path.dirname(output_h5_path)
        if not os.path.exists(parent_dir_h5): os.makedirs(parent_dir_h5, exist_ok=True); print(
            f"Created HDF5 parent directory: {parent_dir_h5}")
        if not os.path.isdir(parent_dir_h5): print(
            f"Error: HDF5 parent path {parent_dir_h5} is not a directory."); return False
        with h5py.File(output_h5_path, 'w') as hf:
            if not embeddings_dict_to_save:
                hf.attrs['status'] = f'No embeddings for {model_name_tag} for dataset {dataset_name_tag}'
            else:
                for char_ascii_str_val, embedding_vector in embeddings_dict_to_save.items(): hf.create_dataset(
                    str(char_ascii_str_val), data=embedding_vector)
            hf.attrs.update({'embedding_type': f'{model_name_tag}_character_node_embeddings', 'vector_size': final_dim,
                             'dataset_tag': dataset_name_tag, 'num_embeddings_saved': len(embeddings_dict_to_save)})
        print(
            f"Successfully saved {len(embeddings_dict_to_save)} {model_name_tag} char embeddings for {dataset_name_tag} to {output_h5_path}");
        return True
    except Exception as e:
        print(f"Error saving {model_name_tag} char HDF5 for {dataset_name_tag} at {output_h5_path}: {e}"); return False


def train_epoch(model, data, train_idx, optimizer, criterion, model_name="", edge_type=None):
    model.train();
    optimizer.zero_grad()
    if train_idx.numel() == 0: return 0.0, 0.0
    ew = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    try:
        logits, _ = model(data.x, data.edge_index, edge_weight=ew, edge_type=edge_type, num_nodes=data.num_nodes)
        if logits is None or logits.numel() == 0 or (
                train_idx.numel() > 0 and logits.size(0) <= train_idx.max()): return 0.0, 0.0
        if data.y is None or data.y.numel() == 0 or (
                train_idx.numel() > 0 and data.y.size(0) <= train_idx.max()): return 0.0, 0.0
        current_logits = logits[train_idx]
        # Removed problematic debug print here, rely on test_model_and_get_embeddings for shape checks if UserCustomDiGCN still an issue
        loss = criterion(current_logits, data.y[train_idx]);
        loss.backward();
        optimizer.step()
        pred = current_logits.argmax(dim=1)
        acc = (pred == data.y[train_idx]).sum().item() / len(train_idx) if len(train_idx) > 0 else 0.0
        return loss.item(), acc
    except RuntimeError as e_rt:
        print(f"RuntimeError during training for {model_name}: {e_rt}")
        if ("Expected target size" in str(e_rt) or "size mismatch" in str(e_rt)) and 'logits' in locals() and hasattr(
                data, 'y'):
            print(
                f"  Debug info for {model_name} train error: Logits shape: {locals().get('logits').shape if hasattr(locals().get('logits'), 'shape') else 'N/A'}, Targets for criterion: {data.y[train_idx].shape}")
        return 0.0, 0.0
    except Exception as e:
        print(f"General error during train_epoch for {model_name}: {e}");return 0.0, 0.0


def test_model_and_get_embeddings(model, data, idx_for_metrics, model_name="", edge_type=None,
                                  get_final_embeddings=False):
    model.eval();
    metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0};
    final_embeddings_numpy = None
    if not (hasattr(data, 'num_nodes') and data.num_nodes > 0 and hasattr(data,
                                                                          'x') and data.x is not None and data.x.numel() > 0): return metrics, None
    current_idx = idx_for_metrics  # Use the passed idx_for_metrics directly
    if current_idx.numel() == 0:  # If idx_for_metrics is empty, cannot compute metrics
        if get_final_embeddings:  # Still try to get full graph embeddings if requested
            with torch.no_grad():
                ew = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
                _, embeddings = model(data.x, data.edge_index, edge_weight=ew, edge_type=edge_type,
                                      num_nodes=data.num_nodes)
                if embeddings is not None: final_embeddings_numpy = embeddings.cpu().numpy()
        return metrics, final_embeddings_numpy  # Return default metrics & any embeddings

    with torch.no_grad():
        ew = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        logits_full, embeddings = model(data.x, data.edge_index, edge_weight=ew, edge_type=edge_type,
                                        num_nodes=data.num_nodes)
        if get_final_embeddings:
            if embeddings is not None:
                final_embeddings_numpy = embeddings.cpu().numpy()
            elif logits_full is not None:
                final_embeddings_numpy = logits_full.cpu().numpy()  # Fallback

        if logits_full is None or logits_full.numel() == 0 or data.y is None or data.y.numel() == 0 or \
                (current_idx.max() >= logits_full.size(0) or current_idx.max() >= data.y.size(0)):
            print(f"Warning: Cannot compute metrics for {model_name}. Logits/labels invalid or idx out of bounds.")
            return metrics, final_embeddings_numpy

        selected_logits = logits_full[current_idx]
        if selected_logits.ndim != 2:
            print(
                f"Warning: selected_logits for {model_name} is not 2D! Shape: {selected_logits.shape}. Full logits: {logits_full.shape}. Skipping metrics.");
            return metrics, final_embeddings_numpy
        pred = selected_logits.argmax(dim=1);
        true = data.y[current_idx]
        if pred.shape != true.shape: print(
            f"Shape mismatch (pred/true) for {model_name} metrics: pred {pred.shape},true {true.shape}.");return metrics, final_embeddings_numpy
        metrics['accuracy'] = (pred == true).sum().item() / len(current_idx)
        pred_cpu, true_cpu = pred.cpu().numpy(), true.cpu().numpy()
        if len(true_cpu) > 0:
            unique_true = np.unique(true_cpu)
            if len(unique_true) > 1 or (len(unique_true) == 1 and len(np.unique(pred_cpu)) > 1):
                metrics.update({'precision': precision_score(true_cpu, pred_cpu, average='macro', zero_division=0),
                                'recall': recall_score(true_cpu, pred_cpu, average='macro', zero_division=0),
                                'f1': f1_score(true_cpu, pred_cpu, average='macro', zero_division=0)})
            elif len(unique_true) == 1:
                is_correct = 1.0 if metrics['accuracy'] == 1.0 else 0.0;metrics.update(
                    {'precision': is_correct, 'recall': is_correct, 'f1': is_correct})
    return metrics, final_embeddings_numpy


def run_gnn_on_char_graph(model_name, dataset_print_name, char_graph_pyg_data, num_classes_val, device,
                          num_folds_config=KFOLDS, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
                          weight_decay=DEFAULT_WEIGHT_DECAY):
    metric_lists = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []};
    run_loss_history, run_acc_history = [], []
    if char_graph_pyg_data.num_nodes == 0: return {f'avg_{k}': 0 for k in metric_lists} | {f'std_{k}': 0 for k in
                                                                                           metric_lists}, None, {
        'loss': [], 'accuracy': []}
    num_relations_for_rgcn_model = 1;
    edge_type_attr = getattr(char_graph_pyg_data, 'edge_type', None)
    if model_name == "RGCN" and edge_type_attr is not None and edge_type_attr.numel() > 0: num_relations_for_rgcn_model = edge_type_attr.max().item() + 1
    edge_type_for_model_run = edge_type_attr.to(device) if model_name == "RGCN" and edge_type_attr is not None else None
    if model_name == "RGCN" and edge_type_for_model_run is None and char_graph_pyg_data.num_edges > 0:
        edge_type_for_model_run = torch.zeros(char_graph_pyg_data.num_edges, dtype=torch.long).to(device)
    elif model_name == "RGCN" and char_graph_pyg_data.num_edges == 0:
        edge_type_for_model_run = torch.empty(0, dtype=torch.long).to(device)
    if char_graph_pyg_data.num_edges == 0 and model_name == "Tong_Library_DiGCN" and HAS_TONG_DiGCN_LIB and TongLibraryDiGCNNet is not None:
        print(f"Skipping {model_name} on {dataset_print_name}: Model needs edges.");
        return {f'avg_{k}': 0 for k in metric_lists} | {f'std_{k}': 0 for k in metric_lists}, None, {'loss': [],
                                                                                                     'accuracy': []}
    final_trained_model = None;
    actual_num_folds = num_folds_config
    if char_graph_pyg_data.num_nodes < 2 or char_graph_pyg_data.num_nodes < num_folds_config:
        actual_num_folds = 1;
        print(f"  Info: Dataset {dataset_print_name} has {char_graph_pyg_data.num_nodes} node(s). Single train/eval.")
        current_data_on_device = char_graph_pyg_data.to(device)
        num_node_features = current_data_on_device.num_features if current_data_on_device.x is not None else 0
        model = get_model(model_name, num_node_features, num_classes_val, device,
                          num_relations_for_rgcn=num_relations_for_rgcn_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay);
        criterion = torch.nn.CrossEntropyLoss()
        all_node_indices = torch.arange(current_data_on_device.num_nodes, dtype=torch.long).to(device)
        if all_node_indices.numel() == 0: return {f'avg_{k}': 0 for k in metric_lists} | {f'std_{k}': 0 for k in
                                                                                          metric_lists}, None, {
            'loss': [], 'accuracy': []}
        for ep in range(1, epochs + 1):
            loss, acc = train_epoch(model, current_data_on_device, all_node_indices, optimizer, criterion, model_name,
                                    edge_type_for_model_run)
            if PLOT_TRAINING_HISTORY: run_loss_history.append(
                loss);eval_metrics_epoch, _ = test_model_and_get_embeddings(model, current_data_on_device,
                                                                            all_node_indices, model_name,
                                                                            edge_type_for_model_run);run_acc_history.append(
                eval_metrics_epoch['accuracy'])
        final_metrics, _ = test_model_and_get_embeddings(model, current_data_on_device, all_node_indices, model_name,
                                                         edge_type_for_model_run)
        for k, v in final_metrics.items(): metric_lists[k].append(v)
        final_trained_model = model
    else:
        kf = KFold(n_splits=actual_num_folds, shuffle=True, random_state=SEED)
        node_indices_np = np.arange(char_graph_pyg_data.num_nodes);
        y_for_split = char_graph_pyg_data.y.cpu().numpy() if hasattr(char_graph_pyg_data,
                                                                     'y') and char_graph_pyg_data.y is not None and char_graph_pyg_data.y.numel() == char_graph_pyg_data.num_nodes else node_indices_np
        can_stratify = False
        if hasattr(char_graph_pyg_data,
                   'y') and char_graph_pyg_data.y is not None and char_graph_pyg_data.y.numel() == char_graph_pyg_data.num_nodes and actual_num_folds > 1:
            class_counts = np.bincount(y_for_split);
            if len(class_counts) > 0 and np.all(class_counts >= actual_num_folds):
                can_stratify = True
            else:
                print(
                    f"  Warning for {dataset_print_name}: Not enough samples for stratified KFold (K={actual_num_folds}). Using regular KFold.")
        print(
            f"  Training {model_name} on {dataset_print_name} ({char_graph_pyg_data.num_nodes}N, {char_graph_pyg_data.num_edges}E, {char_graph_pyg_data.num_features}F, {num_classes_val}C) [{actual_num_folds} folds]")
        splitter = kf.split(node_indices_np, y_for_split) if can_stratify else kf.split(node_indices_np)
        for fold, (train_idx_np, test_idx_np) in enumerate(splitter):
            if len(test_idx_np) == 0 or len(train_idx_np) == 0: print(
                f"  Skipping fold {fold + 1} due to empty split.");continue
            train_idx = torch.tensor(train_idx_np, dtype=torch.long).to(device);
            test_idx = torch.tensor(test_idx_np, dtype=torch.long).to(device)
            current_data_on_device = char_graph_pyg_data.to(device)
            num_node_features = current_data_on_device.num_features if current_data_on_device.x is not None else 0
            model = get_model(model_name, num_node_features, num_classes_val, device,
                              num_relations_for_rgcn=num_relations_for_rgcn_model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay);
            criterion = torch.nn.CrossEntropyLoss()
            for ep in range(1, epochs + 1):
                loss, acc = train_epoch(model, current_data_on_device, train_idx, optimizer, criterion, model_name,
                                        edge_type_for_model_run)
                if fold == 0 and PLOT_TRAINING_HISTORY: run_loss_history.append(
                    loss);eval_metrics_epoch, _ = test_model_and_get_embeddings(model, current_data_on_device, test_idx,
                                                                                model_name,
                                                                                edge_type_for_model_run);run_acc_history.append(
                    eval_metrics_epoch['accuracy'])
            eval_metrics, _ = test_model_and_get_embeddings(model, current_data_on_device, test_idx, model_name,
                                                            edge_type_for_model_run)
            for k, v in eval_metrics.items(): metric_lists[k].append(v)
            if fold == actual_num_folds - 1: final_trained_model = model
    fold_train_history_to_return = {'loss': run_loss_history, 'accuracy': run_acc_history}
    results = {};
    if not any(v for v_list in metric_lists.values() for v in v_list): print(
        f"  No valid metrics recorded for {model_name} on {dataset_print_name}.")
    for k_metric, val_list in metric_lists.items(): valid_vals = [v for v in val_list if not np.isnan(v)];results[
        f'avg_{k_metric}'] = np.mean(valid_vals) if valid_vals else 0.0;results[f'std_{k_metric}'] = np.std(
        valid_vals) if len(valid_vals) > 1 else 0.0
    print(
        f"  Metrics for {model_name} on {dataset_print_name} ({actual_num_folds}-fold run): Acc={results.get('avg_accuracy', 0):.4f}(+/-{results.get('std_accuracy', 0):.4f}), F1M={results.get('avg_f1', 0):.4f}(+/-{results.get('std_f1', 0):.4f})")
    all_char_node_embeddings_np = None
    if SAVE_EMBEDDINGS and final_trained_model is not None and hasattr(char_graph_pyg_data,
                                                                       'x') and char_graph_pyg_data.x is not None and char_graph_pyg_data.num_nodes > 0:
        print(f"  Extracting final character node embeddings for {model_name} from {dataset_print_name}...")
        full_data_on_device = char_graph_pyg_data.to(device);
        all_nodes_idx = torch.arange(full_data_on_device.num_nodes).to(device)
        _, all_char_node_embeddings_np = test_model_and_get_embeddings(final_trained_model, full_data_on_device,
                                                                       all_nodes_idx, model_name,
                                                                       edge_type_for_model_run,
                                                                       get_final_embeddings=True)
    return results, all_char_node_embeddings_np, fold_train_history_to_return


def plot_training_history_torch(history_data: Dict[str, List[float]], model_name_str: str, dataset_name_str: str,
                                save_to_directory: str):  # Parameter name changed
    if not PLOT_TRAINING_HISTORY or not history_data or not history_data.get('loss') or not history_data.get(
        'accuracy'): print(f"Plot: No/empty history data for {model_name_str} on {dataset_name_str}.");return
    len_loss, len_acc = len(history_data['loss']), len(history_data['accuracy'])
    if len_loss == 0 or len_acc == 0 or len_loss != len_acc: print(
        f"Plot Warning: Mismatched/empty history for {model_name_str}. Loss:{len_loss},Acc:{len_acc}. Skipping.");return

    plot_path = os.path.normpath(save_to_directory)  # Use the passed directory directly
    try:
        if not os.path.exists(plot_path): os.makedirs(plot_path, exist_ok=True);
        if not os.path.isdir(plot_path): print(f"Error: Plot save path {plot_path} not a directory."); return
    except OSError as e:
        print(f"Error creating plot directory {plot_path}: {e}."); return

    epochs_range = range(1, len_loss + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4));
    fig.suptitle(f"Training History: {model_name_str} - {dataset_name_str}", fontsize=14)
    axes[0].plot(epochs_range, history_data['loss'], label='Loss', marker='.');
    axes[0].set_title('Loss');
    axes[0].set_xlabel('Epoch');
    axes[0].set_ylabel('Loss');
    axes[0].legend()
    axes[1].plot(epochs_range, history_data['accuracy'], label='Accuracy', marker='.');
    axes[1].set_title('Accuracy');
    axes[1].set_xlabel('Epoch');
    axes[1].set_ylabel('Accuracy');
    axes[1].legend();
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_filename = f"training_history_{dataset_name_str.replace(' ', '_').replace('/', '_')}_{model_name_str}.png"
    full_plot_save_path = os.path.normpath(os.path.join(plot_path, plot_filename))
    try:
        plt.savefig(full_plot_save_path); print(f"Saved training plot to {full_plot_save_path}")
    except Exception as e:
        print(f"Error saving plot {full_plot_save_path}: {e}")
    plt.close(fig)


def display_results_table_pandas(all_results_list_of_dicts, kfolds_val_config, output_file_path=None):
    if not all_results_list_of_dicts: print("No results for table."); return
    print(f"\n\n--- Final Summary of Average Metrics (Configured for {kfolds_val_config}-Fold CV) ---")
    table_rows = []
    for result_item in all_results_list_of_dicts:
        model_name = result_item.get("model_name", "UnknownModel");
        dataset_name = result_item.get("dataset_name", "UnknownDataset");
        metrics = result_item.get("metrics", {});
        row = {"Dataset": dataset_name, "Model": model_name}
        if isinstance(metrics.get('avg_accuracy'), str) and "Error" in metrics.get('avg_accuracy'):
            row.update({"Avg Acc": metrics.get('avg_accuracy'), "Std Acc": metrics.get('std_accuracy', "N/A"),
                        "Avg F1(M)": "Error"})
        else:
            row.update({"Avg Acc": f"{metrics.get('avg_accuracy', 0.):.4f}",
                        "Std Acc": f"{metrics.get('std_accuracy', 0.):.4f}",
                        "Avg F1(M)": f"{metrics.get('avg_f1', 0.):.4f}",
                        "Std F1(M)": f"{metrics.get('std_f1', 0.):.4f}",
                        "Avg Precision(M)": f"{metrics.get('avg_precision', 0.):.4f}",
                        "Avg Recall(M)": f"{metrics.get('avg_recall', 0.):.4f}"})
        table_rows.append(row)
    if table_rows:
        df = pd.DataFrame(table_rows);
        col_order = ["Dataset", "Model", "Avg Acc", "Std Acc", "Avg F1(M)", "Std F1(M)", "Avg Precision(M)",
                     "Avg Recall(M)"]
        df = df[[c for c in col_order if c in df.columns]];
        table_string = df.to_string(index=False);
        print(table_string)
        if output_file_path:
            output_file_path_norm = os.path.normpath(output_file_path)
            try:
                os.makedirs(os.path.dirname(output_file_path_norm),
                            exist_ok=True)  # Ensure directory for results file exists
                with open(output_file_path_norm, 'w') as f:
                    f.write(f"--- Final Summary of Average Metrics ({kfolds_val_config}-Fold CV) ---\n\n");f.write(
                        table_string)
                print(f"Results table saved to {output_file_path_norm}")
            except Exception as e:
                print(f"Error saving results table to {output_file_path_norm}:{e}")
    else:
        print("No results to display in table.")


if __name__ == '__main__':
    print(f"PyTorch Geometric Signed Directed Library (for Tong DiGCN) available: {HAS_TONG_DiGCN_LIB}")
    os.makedirs(OUTPUT_BASE_DIR_FOR_THIS_SCRIPT, exist_ok=True)
    os.makedirs(EMBEDDINGS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    print(f"Using output base directory: {OUTPUT_BASE_DIR_FOR_THIS_SCRIPT}")
    print(f"Embeddings will be saved to: {EMBEDDINGS_OUTPUT_DIR}")
    print(f"Plots will be saved to: {PLOTS_OUTPUT_DIR}")

    all_model_run_results = []
    print("--- Loading Pre-generated Global Character Graph Data ---")
    char_vocab_data = load_char_vocabulary(CHAR_VOCAB_JSON)
    undirected_char_graph_custom_orig = load_pickled_graph(GLOBAL_UNDIRECTED_GRAPH_PKL)
    directed_char_graph_custom_orig = load_pickled_graph(GLOBAL_DIRECTED_GRAPH_PKL)
    if char_vocab_data is None: print("Error: Character vocabulary not loaded. Exiting."); exit()

    # Prepare dataset variants
    datasets_to_process_variants = {}
    if undirected_char_graph_custom_orig:
        data_undir_raw, num_classes_undir = convert_custom_char_graph_to_pyg_data(undirected_char_graph_custom_orig,
                                                                                  char_vocab_data,
                                                                                  "GlobalCharGraph_Undir_Raw")
        if data_undir_raw and data_undir_raw.num_nodes > 0 and num_classes_undir > 0:
            datasets_to_process_variants["GlobalCharGraph_Undirected"] = (data_undir_raw, num_classes_undir)

    if directed_char_graph_custom_orig:
        data_dir_raw, num_classes_dir = convert_custom_char_graph_to_pyg_data(directed_char_graph_custom_orig,
                                                                              char_vocab_data,
                                                                              "GlobalCharGraph_Dir_Raw")
        if data_dir_raw and data_dir_raw.num_nodes > 0 and num_classes_dir > 0:
            datasets_to_process_variants["GlobalCharGraph_Directed"] = (data_dir_raw, num_classes_dir)

    if not datasets_to_process_variants: print(
        "Error: No PyG Data objects could be created from character graphs. Exiting."); exit()

    models_to_run = ["CustomGCN", "GAT", "GraphSAGE", "GIN", "UserCustomDiGCN", "RGCN"]
    if HAS_TONG_DiGCN_LIB and TongLibraryDiGCNNet is not None: models_to_run.append("Tong_Library_DiGCN")

    for dataset_variant_name, (data_object, num_actual_classes) in datasets_to_process_variants.items():
        print(f"\n--- Processing Dataset Variant: {dataset_variant_name} ---")
        current_epochs, current_lr, current_wd = DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY
        for model_name_to_run in models_to_run:
            # Removed the skip logic to run all models on both graph types
            print(f"  --- Model: {model_name_to_run} on {dataset_variant_name} ---")
            start_run_time = time.time()
            try:
                metrics_summary, node_embeddings_np, train_history = run_gnn_on_char_graph(
                    model_name_to_run, dataset_variant_name, data_object, num_actual_classes, device,
                    KFOLDS, current_epochs, current_lr, current_wd
                )
                all_model_run_results.append(
                    {"dataset_name": dataset_variant_name, "model_name": model_name_to_run, "metrics": metrics_summary})
                if SAVE_EMBEDDINGS and node_embeddings_np is not None and node_embeddings_np.size > 0:
                    emb_dim = node_embeddings_np.shape[1];
                    processed_embeddings_np = node_embeddings_np
                    if APPLY_PCA_TO_GNN_EMBEDDINGS and emb_dim > COMMON_EMBEDDING_DIM_PCA and COMMON_EMBEDDING_DIM_PCA > 0 and \
                            processed_embeddings_np.shape[0] > 1:
                        processed_embeddings_np, emb_dim = reduce_dimensionality_with_pca(node_embeddings_np,
                                                                                          COMMON_EMBEDDING_DIM_PCA,
                                                                                          f"{model_name_to_run}_{dataset_variant_name}")
                    char_embeddings_to_save_dict = {}
                    if hasattr(data_object, 'node_original_ids') and len(data_object.node_original_ids) == \
                            processed_embeddings_np.shape[0] and processed_embeddings_np.shape[0] > 0:
                        for i in range(processed_embeddings_np.shape[0]): char_embeddings_to_save_dict[
                            str(data_object.node_original_ids[i])] = processed_embeddings_np[i]
                    else:
                        print(
                            f"Warning: Embedding mapping issue for {dataset_variant_name}, model {model_name_to_run}.")
                    if char_embeddings_to_save_dict:
                        h5_filename = f"{dataset_variant_name.replace(' ', '_').replace('/', '_')}_{model_name_to_run}_CHAR_EMBS_dim{emb_dim}_{DATASET_TAG_PREFIX}.h5"
                        h5_path = os.path.normpath(os.path.join(EMBEDDINGS_OUTPUT_DIR, h5_filename))
                        save_node_embeddings_to_h5(embeddings_dict_to_save=char_embeddings_to_save_dict,
                                                   output_h5_path=h5_path, model_name_tag=model_name_to_run,
                                                   dataset_name_tag=dataset_variant_name, final_dim=emb_dim)
                    else:
                        print(
                            f"Warning: Embedding dictionary empty for {model_name_to_run} on {dataset_variant_name}. H5 save skipped.")

                # Corrected check for plotting history
                if PLOT_TRAINING_HISTORY and train_history and (
                        train_history.get('loss') or train_history.get('accuracy')):
                    plot_training_history_torch(train_history, model_name_to_run, dataset_variant_name,
                                                PLOTS_OUTPUT_DIR)  # Use PLOTS_OUTPUT_DIR
            except Exception as e_run:
                print(f"  ERROR running {model_name_to_run} on {dataset_variant_name}: {e_run}");
                import traceback;

                traceback.print_exc()
                all_model_run_results.append({"dataset_name": dataset_variant_name, "model_name": model_name_to_run,
                                              "metrics": {"avg_accuracy": f"Error: {type(e_run).__name__}",
                                                          "std_accuracy": str(e_run)}})
            end_run_time = time.time()
            if all_model_run_results and all_model_run_results[-1]["model_name"] == model_name_to_run and \
                    all_model_run_results[-1]["dataset_name"] == dataset_variant_name and not (
                    isinstance(all_model_run_results[-1]["metrics"].get('avg_accuracy'), str) and "Error" in
                    all_model_run_results[-1]["metrics"].get('avg_accuracy')):
                print(
                    f"  Time taken for {model_name_to_run} on {dataset_variant_name}:{end_run_time - start_run_time:.2f} seconds")
            gc.collect()
    display_results_table_pandas(all_model_run_results, KFOLDS,
                                 output_file_path=RESULTS_TABLE_FILE)  # Pass file path for saving table
    print("\nScript finished.")