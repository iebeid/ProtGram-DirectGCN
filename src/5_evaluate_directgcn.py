import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, RGCNConv
from torch_geometric.datasets import KarateClub, Planetoid, WebKB
from torch_geometric.utils import to_undirected  # For creating undirected versions
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time
import os
import h5py
import matplotlib.pyplot as plt
import gc
from typing import Optional, Union, List, Dict

# Attempt to import DiGCNConv
try:
    from torch_geometric_signed_directed.nn.directed import DiGCNConv

    HAS_TONG_DiGCN_LIB = True
except ImportError:
    HAS_TONG_DiGCN_LIB = False
    print("Warning: 'torch_geometric_signed_directed' not found. 'Tong_Library_DiGCN' will be skipped.")
    DiGCNConv = None

# --- 0. Configuration & Setup ---
SEED = 42
KFOLDS = 5
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.01
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_DROPOUT = 0.5

# Output paths
OUTPUT_BASE_DIR = "C:/tmp/Models/gnn_evaluation_v5/"  # New version subfolder
EMBEDDINGS_OUTPUT_DIR = os.path.normpath(os.path.join(OUTPUT_BASE_DIR, "embeddings"))
PLOTS_OUTPUT_DIR = os.path.normpath(os.path.join(OUTPUT_BASE_DIR, "plots"))
RESULTS_TABLE_FILE = os.path.normpath(os.path.join(OUTPUT_BASE_DIR, "all_results_summary_v5.txt"))

# Embedding Saving Configuration
SAVE_EMBEDDINGS = True
COMMON_EMBEDDING_DIM_PCA = 64
APPLY_PCA_TO_GNN_EMBEDDINGS = True

PLOT_TRAINING_HISTORY = True

# Paths for custom dataset
CUSTOM_DATA_NAME = "MyCustomGraph"
CUSTOM_EDGE_FILE = None
CUSTOM_NODE_FEATURES_FILE = None
CUSTOM_NODE_LABELS_FILE = None

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 1. Custom Dataset Loading ---
def load_custom_dataset_from_edges(
        edge_list_path, delimiter=',', header_edge_list=None,
        source_col_name_or_idx=0, target_col_name_or_idx=1,
        edge_attr_col_names_or_indices=None,
        node_feature_path=None, node_label_path=None,
        feature_generation_strategy='identity', default_feature_dim=64,
        feature_header=0, feature_index_col=0,
        label_header=0, label_index_col=0, label_target_col=1,
        normalize_features=False, dataset_name="CustomDataset"
):
    edge_list_path = os.path.normpath(edge_list_path)
    if node_feature_path: node_feature_path = os.path.normpath(node_feature_path)
    if node_label_path: node_label_path = os.path.normpath(node_label_path)
    print(f"Attempting to load custom dataset: {dataset_name} from {edge_list_path}")
    try:
        df_edges = pd.read_csv(edge_list_path, delimiter=delimiter, header=header_edge_list,
                               dtype={source_col_name_or_idx: str, target_col_name_or_idx: str})
        source_nodes_orig = df_edges.iloc[:, source_col_name_or_idx if isinstance(source_col_name_or_idx,
                                                                                  int) else df_edges.columns.get_loc(
            source_col_name_or_idx)].values
        target_nodes_orig = df_edges.iloc[:, target_col_name_or_idx if isinstance(target_col_name_or_idx,
                                                                                  int) else df_edges.columns.get_loc(
            target_col_name_or_idx)].values
        all_nodes_orig = sorted(list(set(source_nodes_orig) | set(target_nodes_orig)))
        node_to_idx = {node_id: i for i, node_id in enumerate(all_nodes_orig)};
        num_nodes = len(all_nodes_orig)
        if num_nodes == 0: print(f"Warning: No nodes found in {dataset_name}.");return None, 0
        print(f"  {dataset_name}: Found {num_nodes} unique nodes.")
        source_indices = [node_to_idx[s] for s in source_nodes_orig];
        target_indices = [node_to_idx[t] for t in target_nodes_orig]
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long);
        edge_attr = None
        if edge_attr_col_names_or_indices:
            attrs_data = df_edges.iloc[:,
                         edge_attr_col_names_or_indices if isinstance(edge_attr_col_names_or_indices, list) else [
                             edge_attr_col_names_or_indices]].values
            edge_attr = torch.tensor(attrs_data, dtype=torch.float)
            if edge_attr.ndim == 1: edge_attr = edge_attr.unsqueeze(1)
            print(f"  {dataset_name}: Loaded {edge_attr.shape[1]} edge attributes.")
        x = None
        if node_feature_path and os.path.exists(node_feature_path):
            try:
                df_features = pd.read_csv(node_feature_path, header=feature_header)
                feature_node_ids = df_features.iloc[:, feature_index_col].astype(str).values
                feature_values = df_features.iloc[:, [c_idx for c_idx, c in enumerate(df_features.columns) if
                                                      c_idx != feature_index_col]].values.astype(np.float32)
                feature_matrix_ordered = np.zeros((num_nodes, feature_values.shape[1]), dtype=np.float32);
                found_count = 0
                for i, orig_node_id_feat in enumerate(feature_node_ids):
                    if orig_node_id_feat in node_to_idx: idx = node_to_idx[orig_node_id_feat];feature_matrix_ordered[
                        idx] = feature_values[i];found_count += 1
                x = torch.tensor(feature_matrix_ordered, dtype=torch.float)
                print(f"  {dataset_name}: Loaded features for {found_count}/{num_nodes} nodes. Dim: {x.shape[1]}.")
            except Exception as e:
                print(f"  Warning: Could not load features from {node_feature_path}: {e}. Fallback.")
        if x is None and feature_generation_strategy:
            print(f"  {dataset_name}: Generating features via '{feature_generation_strategy}'.")
            if feature_generation_strategy == 'identity':
                x = torch.eye(num_nodes, dtype=torch.float)
            elif feature_generation_strategy == 'ones':
                x = torch.ones((num_nodes, default_feature_dim), dtype=torch.float)
            elif feature_generation_strategy == 'random':
                x = torch.randn((num_nodes, default_feature_dim), dtype=torch.float)
        elif x is None:
            x = torch.eye(num_nodes, dtype=torch.float)
        if normalize_features and x is not None: x = F.normalize(x, p=2, dim=1);print(
            f"  {dataset_name}: Node features L2 normalized.")
        y = None;
        num_classes = 0
        if node_label_path and os.path.exists(node_label_path):
            try:
                df_labels = pd.read_csv(node_label_path, header=label_header)
                label_node_ids = df_labels.iloc[:, label_index_col].astype(str).values
                label_targets_orig = df_labels.iloc[:, label_target_col].values
                unique_cls_lbls = sorted(list(set(label_targets_orig)));
                cls_to_int = {lbl: i for i, lbl in enumerate(unique_cls_lbls)};
                num_classes = len(unique_cls_lbls)
                y_mapped = torch.full((num_nodes,), -1, dtype=torch.long);
                found_lbl_count = 0
                for i, orig_node_id_lbl in enumerate(label_node_ids):
                    if orig_node_id_lbl in node_to_idx: y_mapped[node_to_idx[orig_node_id_lbl]] = cls_to_int[
                        label_targets_orig[i]];found_lbl_count += 1
                if found_lbl_count > 0:
                    y = y_mapped;print(
                        f"  {dataset_name}: Labels for {found_lbl_count}/{num_nodes} nodes. Classes: {num_classes}.")
                else:
                    y = None;num_classes = 0
            except Exception as e:
                print(f"  Warn: Could not load labels from {node_label_path}: {e}.");y = None;num_classes = 0
        if y is None: y = torch.zeros(num_nodes, dtype=torch.long);num_classes = 1 if num_nodes > 0 else 0;print(
            f"  {dataset_name}: No labels loaded. Using dummy labels. Num_classes: {num_classes}.")
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y);
        data.num_nodes = num_nodes
        print(f"  {dataset_name}: Successfully created PyG Data object.")
        return data, num_classes
    except FileNotFoundError:
        print(f"Error: Edge list file not found at {edge_list_path}");return None, 0
    except Exception as e:
        print(f"Error loading custom dataset {dataset_name}: {e}");import traceback;traceback.print_exc();return None, 0


class CustomDatasetWrapper:
    def __init__(self, data_obj, num_classes_val, name="Custom"): self._data = [
        data_obj];self.num_classes = num_classes_val;self.name = name

    def __getitem__(self, idx): return self._data[idx]

    def __len__(self): return len(self._data)

    def __repr__(self): return f"{self.name}({len(self._data)})"


# --- 2. GNN Model Definitions ---
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

    def forward(self, x, edge_index, edge_weight=None, edge_type=None, num_nodes=None):  # Standardized signature
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
        if x.size(0) == 0: self.embedding_output = torch.empty(0, self.c1.out_channels * self.c1.heads,
                                                               device=x.device);return torch.empty(0,
                                                                                                   self.c2.out_channels,
                                                                                                   device=x.device), self.embedding_output
        h = F.dropout(x, p=self.dp, training=self.training);
        self.embedding_output = self.c1(h, edge_index, edge_attr=edge_weight)
        h_activated_emb = F.elu(self.embedding_output);
        h_emb_dropped = F.dropout(h_activated_emb, p=self.dp, training=self.training);
        logits = self.c2(h_emb_dropped, edge_index, edge_attr=edge_weight)
        return logits, self.embedding_output


class GraphSAGENet(BaseGNN):
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
            super().__init__();
            self.nf = nf;
            self.nc = nc;
            self.hc1 = hc1;
            self.hc2 = hc2;
            self.dp = dp  # Individual assignments
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

    def forward(self, x, edge_index, edge_type, edge_weight=None, num_nodes=None):  # Standardized signature
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
    hc1_3layer = 64;
    hc2_3layer = 32;
    hc_2layer = 64;
    model = None
    if model_name == "CustomGCN":
        model = CustomGCN(num_node_features, num_classes_val, hc1_3layer, hc2_3layer, dp)
    elif model_name == "GAT":
        gat_hc = 32;
        gat_heads = 4
        if num_node_features > 1000 and num_classes_val < 10: gat_hc = max(16, num_classes_val);gat_heads = 2
        model = GATNet(num_node_features, num_classes_val, hc=gat_hc, h=gat_heads, dp=0.6)
    elif model_name == "GraphSAGE":
        sage_hc = max(128, num_classes_val * 4) if num_node_features < 1000 else hc_2layer
        model = GraphSAGENet(num_node_features, num_classes_val, hc=sage_hc, dp=dp)
    elif model_name == "GIN":
        model = GINNet(num_node_features, num_classes_val, hc_2layer, dp)
    elif model_name == "Tong_Library_DiGCN" and HAS_TONG_DiGCN_LIB and TongLibraryDiGCNNet is not None:
        model = TongLibraryDiGCNNet(num_node_features, num_classes_val, hc1_3layer, hc2_3layer, dp)
    elif model_name == "UserCustomDiGCN":
        model = UserCustomDiGCN_PyTorch(num_node_features, num_classes_val, hc1=16, hc2=16, dp=dp, norm_in=True)
    elif model_name == "RGCN":
        model = RGCNNet(num_node_features, num_classes_val, hc=hc_2layer, num_relations=num_relations_for_rgcn, dp=dp)
    if model is None: raise ValueError(f"Unknown model:{model_name} or library/class not available.")
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


def save_embeddings_to_h5(embeddings_dict_numpy, output_h5_path, model_name, dataset_tag_name, final_dim):
    output_h5_path = os.path.normpath(output_h5_path);
    print(f"Saving {model_name} embeddings (dim:{final_dim}) to {output_h5_path}...")
    try:
        parent_dir_h5 = os.path.dirname(output_h5_path)
        if not os.path.exists(parent_dir_h5): os.makedirs(parent_dir_h5, exist_ok=True);print(
            f"Created HDF5 parent directory: {parent_dir_h5}")
        if not os.path.isdir(parent_dir_h5): print(
            f"Error:HDF5 parent path {parent_dir_h5} not a directory.");return False
        with h5py.File(output_h5_path, 'w') as hf:
            if not embeddings_dict_numpy: hf.attrs['status'] = f'No embeddings for {model_name}'
            for node_id, embedding_vec in embeddings_dict_numpy.items(): hf.create_dataset(str(node_id),
                                                                                           data=embedding_vec)
            hf.attrs.update({'embedding_type': f'{model_name}_node_embeddings', 'vector_size': final_dim,
                             'dataset_tag': dataset_tag_name, 'num_embeddings_saved': len(embeddings_dict_numpy)})
        print(f"Successfully saved {model_name} embeddings to {output_h5_path}");
        return True
    except Exception as e:
        print(f"Error saving {model_name} HDF5 for {dataset_tag_name} at {output_h5_path}:{e}");return False


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
        loss = criterion(current_logits, data.y[train_idx]);
        loss.backward();
        optimizer.step()
        pred = current_logits.argmax(dim=1)
        acc = (pred == data.y[train_idx]).sum().item() / len(train_idx) if len(train_idx) > 0 else 0.0
        return loss.item(), acc
    except RuntimeError as e_rt:
        print(f"RuntimeError during training for {model_name}:{e_rt}");return 0.0, 0.0
    except Exception as e:
        print(f"General error during train_epoch for {model_name}:{e}");return 0.0, 0.0


def test_model_and_get_embeddings(model, data, test_idx, model_name="", edge_type=None, get_final_embeddings=False):
    model.eval();
    metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0};
    final_embeddings_numpy = None
    if not (hasattr(data, 'num_nodes') and data.num_nodes > 0 and hasattr(data,
                                                                          'x') and data.x is not None and data.x.numel() > 0): return metrics, None
    with torch.no_grad():
        ew = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        logits, embeddings = model(data.x, data.edge_index, edge_weight=ew, edge_type=edge_type,
                                   num_nodes=data.num_nodes)
        if get_final_embeddings and embeddings is not None:
            final_embeddings_numpy = embeddings.cpu().numpy()
        elif get_final_embeddings and logits is not None:
            print(f"Warning: {model_name} using logits as embeddings fallback.")
            final_embeddings_numpy = logits.cpu().numpy()
        if len(test_idx) > 0 and logits is not None and logits.numel() > 0 and data.y is not None and data.y.numel() > 0 and \
                not (test_idx.max() >= logits.size(0) or test_idx.max() >= data.y.size(0)):
            selected_logits = logits[test_idx]
            if selected_logits.ndim != 2:
                print(
                    f"Warning: Logits for metric calc in {model_name} not 2D! Shape: {selected_logits.shape}. Full: {logits.shape}. Skipping metrics.");
                return metrics, final_embeddings_numpy
            pred = selected_logits.argmax(dim=1);
            true = data.y[test_idx]
            if pred.shape != true.shape: print(
                f"Shape mismatch (pred/true) for {model_name}: pred {pred.shape},true {true.shape}.");return metrics, final_embeddings_numpy
            metrics['accuracy'] = (pred == true).sum().item() / len(test_idx)
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


def run_cv_for_model_dataset(model_name, dataset_name_full, data_obj, num_classes_val, device,
                             num_folds=KFOLDS, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    metric_lists = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []};
    run_loss_history, run_acc_history = [], []
    if data_obj.num_nodes == 0: return {f'avg_{k}': 0 for k in metric_lists} | {f'std_{k}': 0 for k in
                                                                                metric_lists}, None, {'loss': [],
                                                                                                      'accuracy': []}
    node_indices = np.arange(data_obj.num_nodes)
    print(
        f"  Training {model_name} on {dataset_name_full} ({data_obj.num_nodes}N, {data_obj.num_edges}E, {data_obj.num_features}F, {num_classes_val}C) [{num_folds} folds]")
    num_relations_for_rgcn_model = 1;
    edge_type_for_model_run = None
    if model_name == "RGCN":
        edge_type_attr = getattr(data_obj, 'edge_type', None)
        if edge_type_attr is not None and edge_type_attr.numel() > 0:
            num_relations_for_rgcn_model = edge_type_attr.max().item() + 1;edge_type_for_model_run = edge_type_attr.to(
                device)
        elif data_obj.num_edges > 0:
            edge_type_for_model_run = torch.zeros(data_obj.num_edges, dtype=torch.long).to(device)
        else:
            edge_type_for_model_run = torch.empty(0, dtype=torch.long).to(device)
    final_model_for_embedding_extraction = None
    for fold, (train_idx_np, test_idx_np) in enumerate(kf.split(node_indices,
                                                                data_obj.y.cpu().numpy() if hasattr(data_obj,
                                                                                                    'y') and data_obj.y is not None else None)):
        if len(test_idx_np) == 0 or len(train_idx_np) == 0: print(
            f"Skipping fold {fold + 1} due to empty split.");continue
        train_idx = torch.tensor(train_idx_np, dtype=torch.long).to(device);
        test_idx = torch.tensor(test_idx_np, dtype=torch.long).to(device)
        current_data_on_device = data_obj.to(device)
        model = get_model(model_name, current_data_on_device.num_features, num_classes_val, device,
                          num_relations_for_rgcn=num_relations_for_rgcn_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay);
        criterion = torch.nn.CrossEntropyLoss()
        for ep in range(1, epochs + 1):
            loss, acc = train_epoch(model, current_data_on_device, train_idx, optimizer, criterion, model_name,
                                    edge_type_for_model_run)
            if fold == 0 and PLOT_TRAINING_HISTORY: run_loss_history.append(
                loss);eval_metrics_epoch, _ = test_model_and_get_embeddings(model, current_data_on_device, test_idx,
                                                                            model_name, edge_type_for_model_run,
                                                                            get_final_embeddings=False);run_acc_history.append(
                eval_metrics_epoch['accuracy'])
        eval_metrics, _ = test_model_and_get_embeddings(model, current_data_on_device, test_idx, model_name,
                                                        edge_type_for_model_run, get_final_embeddings=False)
        for k, v in eval_metrics.items(): metric_lists[k].append(v)
        if fold == num_folds - 1: final_model_for_embedding_extraction = model
    fold_train_history_to_return = {'loss': run_loss_history, 'accuracy': run_acc_history}
    results = {};
    if not metric_lists['accuracy']:
        for k_metric in metric_lists: results[f'avg_{k_metric}'] = 0.0;results[f'std_{k_metric}'] = 0.0
    else:
        for k_metric, val_list in metric_lists.items(): results[f'avg_{k_metric}'] = np.mean(val_list);results[
            f'std_{k_metric}'] = np.std(val_list)
    print(
        f"  Metrics for {model_name} on {dataset_name_full}: Acc={results.get('avg_accuracy', 0):.4f}(+/-{results.get('std_accuracy', 0):.4f}), F1M={results.get('avg_f1', 0):.4f}(+/-{results.get('std_f1', 0):.4f})")
    all_node_embeddings_np = None
    if SAVE_EMBEDDINGS and final_model_for_embedding_extraction is not None:
        print(f"  Extracting final node embeddings for {model_name} on {dataset_name_full}...")
        full_data_on_device = data_obj.to(device)
        _, all_node_embeddings_np = test_model_and_get_embeddings(final_model_for_embedding_extraction,
                                                                  full_data_on_device,
                                                                  torch.arange(full_data_on_device.num_nodes).to(
                                                                      device), model_name, edge_type_for_model_run,
                                                                  get_final_embeddings=True)
    return results, all_node_embeddings_np, fold_train_history_to_return


def plot_training_history_torch(history_data: Dict[str, List[float]], model_name_str: str, dataset_name_str: str,
                                fold_id: str = "Fold1"):
    if not history_data or not history_data.get('loss') or not history_data.get('accuracy') or not history_data[
        'loss'] or not history_data['accuracy']:
        print(f"Plot: No/empty history for {model_name_str} on {dataset_name_str} ({fold_id}).");
        return
    len_loss, len_acc = len(history_data['loss']), len(history_data['accuracy'])
    if len_loss != len_acc:
        print(
            f"Plot Warning: Mismatched history len for {model_name_str}. L:{len_loss},A:{len_acc}. Trying shortest.");min_len = min(
            len_loss, len_acc);history_data['loss'] = history_data['loss'][:min_len];history_data['accuracy'] = \
        history_data['accuracy'][:min_len];epochs_range = range(1, min_len + 1)
    else:
        epochs_range = range(1, len_loss + 1)
    if not list(epochs_range): print(f"Plot: No epochs to plot for {model_name_str}."); return
    plt.figure(figsize=(12, 5));
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_data['loss'], label='Training Loss', marker='.');
    plt.title(f'Loss:{model_name_str}');
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_data['accuracy'], label='Validation Accuracy', marker='.');
    plt.title(f'Accuracy:{model_name_str}');
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.legend();
    plt.ylim(0, 1.05)
    plt.suptitle(f"History: {model_name_str} - {dataset_name_str} ({fold_id})", fontsize=16);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plot_filename = f"history_{dataset_name_str.replace(' ', '_').replace('/', '_')}_{model_name_str}_{fold_id}.png"
    full_plot_path = os.path.normpath(os.path.join(PLOTS_OUTPUT_DIR, plot_filename))
    try:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        plt.savefig(full_plot_path);
        print(f"Saved training history plot to {full_plot_path}")
    except Exception as e:
        print(f"Error saving plot to {full_plot_path}: {e}")
    plt.close()


def display_results_table_pandas(all_results_dict_of_dicts, kfolds_val, output_file_path=None):
    print(f"\n\n--- Final Summary of Average Metrics ({kfolds_val}-Fold CV) ---")
    table_rows = []
    for d_name, model_metrics_dict in all_results_dict_of_dicts.items():
        for m_name, metrics_summary in model_metrics_dict.items():
            row = {"Dataset": d_name, "Model": m_name}
            if metrics_summary.get('avg_accuracy') == "Error":
                row.update({"Avg Accuracy": "Error", "Std Acc": metrics_summary.get('std_accuracy', "N/A"),
                            "Avg F1 (M)": "Error", "Std F1 (M)": "N/A"})
            else:
                row.update({"Avg Accuracy": f"{metrics_summary.get('avg_accuracy', 0.):.4f}",
                            "Std Acc": f"{metrics_summary.get('std_accuracy', 0.):.4f}",
                            "Avg F1 (M)": f"{metrics_summary.get('avg_f1', 0.):.4f}",
                            "Std F1 (M)": f"{metrics_summary.get('std_f1', 0.):.4f}",
                            "Avg Precision (M)": f"{metrics_summary.get('avg_precision', 0.):.4f}",
                            "Avg Recall (M)": f"{metrics_summary.get('avg_recall', 0.):.4f}"})
            table_rows.append(row)
    if table_rows:
        results_df = pd.DataFrame(table_rows);
        cols_ordered = ["Dataset", "Model", "Avg Accuracy", "Std Acc", "Avg F1 (M)", "Std F1 (M)", "Avg Precision (M)",
                        "Avg Recall (M)"]
        results_df = results_df[[c for c in cols_ordered if c in results_df.columns]];
        table_string = results_df.to_string(index=False);
        print(table_string)
        if output_file_path:
            output_file_path_norm = os.path.normpath(output_file_path)
            try:
                os.makedirs(os.path.dirname(output_file_path_norm), exist_ok=True)
                with open(output_file_path_norm, 'w') as f:
                    f.write(f"--- Final Summary of Average Metrics ({kfolds_val}-Fold CV) ---\n\n");f.write(
                        table_string)
                print(f"Results table saved to {output_file_path_norm}")
            except Exception as e:
                print(f"Error saving results table to {output_file_path_norm}:{e}")
    else:
        print("No results to display in table.")


if __name__ == '__main__':
    print(f"PyTorch Geometric Signed Directed Library (for Tong DiGCN) available: {HAS_TONG_DiGCN_LIB}")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    print(f"Output base directory: {OUTPUT_BASE_DIR}")

    datasets_registry = {}
    print("Loading standard datasets...")
    try:
        karate_data = KarateClub()[0]
        datasets_registry["KarateClub"] = (karate_data.cpu(),
                                           karate_data.y.max().item() + 1 if karate_data.y is not None else 1)
        for name in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(root='/tmp/' + name, name=name)
            datasets_registry[name] = (dataset[0].cpu(), dataset.num_classes)
        for name in ["Cornell", "Texas", "Wisconsin"]:
            dataset = WebKB(root='/tmp/' + name, name=name)
            datasets_registry[name] = (dataset[0].cpu(), dataset.num_classes)
    except Exception as e:
        print(f"Error loading standard datasets: {e}")
    print(f"Standard datasets loaded: {list(datasets_registry.keys())}")

    if CUSTOM_EDGE_FILE and os.path.exists(os.path.normpath(CUSTOM_EDGE_FILE)):
        custom_data_obj, custom_num_cls = load_custom_dataset_from_edges(CUSTOM_EDGE_FILE,
                                                                         dataset_name=CUSTOM_DATA_NAME,
                                                                         node_feature_path=CUSTOM_NODE_FEATURES_FILE,
                                                                         node_label_path=CUSTOM_NODE_LABELS_FILE)
        if custom_data_obj is not None and custom_num_cls > 0: datasets_registry[CUSTOM_DATA_NAME] = (custom_data_obj,
                                                                                                      custom_num_cls)
        print(f"{CUSTOM_DATA_NAME} added to registry with {custom_num_cls} classes.")
    else:
        print(f"Custom edge file '{CUSTOM_EDGE_FILE}' not found or not specified. Skipping custom dataset.")

    models_to_run = ["CustomGCN", "GAT", "GraphSAGE", "GIN", "UserCustomDiGCN", "RGCN"]
    if HAS_TONG_DiGCN_LIB and TongLibraryDiGCNNet is not None:
        models_to_run.append("Tong_Library_DiGCN")
    else:
        print("'Tong_Library_DiGCN' will be skipped as library/class not available.")

    all_results_summary = {}

    for d_name_orig, (data_orig_cpu, num_classes) in datasets_registry.items():
        print(f"\n--- Processing Dataset Original: {d_name_orig} ---")
        if data_orig_cpu is None or not hasattr(data_orig_cpu, 'x') or data_orig_cpu.x is None: print(
            f"  Skip {d_name_orig}:no data/features.");continue
        if not hasattr(data_orig_cpu, 'y') or data_orig_cpu.y is None: print(
            f"  Skip {d_name_orig}:no labels.");continue
        data_orig_cpu.x = data_orig_cpu.x.float()
        if hasattr(data_orig_cpu,
                   'edge_attr') and data_orig_cpu.edge_attr is not None: data_orig_cpu.edge_attr = data_orig_cpu.edge_attr.float()
        if num_classes <= 1 and d_name_orig != "KarateClub": print(
            f"  Skip {d_name_orig}:has {num_classes} class(es).");continue

        graph_variants_to_process = {}
        # 1. Original graph (can be treated as directed by some models)
        graph_variants_to_process[f"{d_name_orig}_OriginalDirected"] = data_orig_cpu.clone()

        # 2. Explicitly Undirected version
        try:
            # For to_undirected, edge_attr handling:
            # If reduce="add", attributes of coalesced edges are summed.
            # If reduce="mean", attributes are averaged (default).
            # If reduce="max" or "min", those are taken.
            # If edge_attr has multiple features, reduce applies element-wise.
            # For this script, "mean" is a reasonable default for undirected conversion.
            # If edge_attr is None, it remains None.
            edge_idx_undir, edge_attr_undir = to_undirected(
                data_orig_cpu.edge_index,
                edge_attr=data_orig_cpu.edge_attr,
                num_nodes=data_orig_cpu.num_nodes,
                reduce="mean"
            )
            data_undir = Data(x=data_orig_cpu.x, edge_index=edge_idx_undir, y=data_orig_cpu.y,
                              num_nodes=data_orig_cpu.num_nodes)
            if edge_attr_undir is not None: data_undir.edge_attr = edge_attr_undir
            # edge_type for undirected is often not used or set to a single type if original was multi-relational.
            # For simplicity, we'll let RGCN handle a None edge_type for this variant if it runs.
            graph_variants_to_process[f"{d_name_orig}_Undirected"] = data_undir
        except Exception as e_undir:
            print(f"Could not create undirected version for {d_name_orig}: {e_undir}")

        for variant_name, data_variant in graph_variants_to_process.items():
            if data_variant is None: continue
            print(f"\n  --- Graph Variant: {variant_name} ---")
            if variant_name not in all_results_summary: all_results_summary[variant_name] = {}

            current_epochs, current_lr, current_wd = DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY
            # (Dataset specific HPs can be set here based on d_name_orig if needed)

            for model_name_to_run in models_to_run:
                # Basic compatibility: directed models might behave differently on undirected graph representations
                # but we will run them to see, as per request.
                print(f"    --- Model: {model_name_to_run} ---")
                start_run_time = time.time()
                try:
                    metrics, node_embeddings, history = run_cv_for_model_dataset(
                        model_name_to_run, variant_name, data_variant, num_classes, device,
                        KFOLDS, current_epochs, current_lr, current_wd
                    )
                    all_results_summary[variant_name][model_name_to_run] = metrics
                    if SAVE_EMBEDDINGS and node_embeddings is not None and node_embeddings.size > 0:
                        emb_dim = node_embeddings.shape[1];
                        processed_embeddings_np = node_embeddings
                        if APPLY_PCA_TO_GNN_EMBEDDINGS and emb_dim > COMMON_EMBEDDING_DIM_PCA and COMMON_EMBEDDING_DIM_PCA > 0 and \
                                processed_embeddings_np.shape[0] > 1:
                            processed_embeddings_np, emb_dim = reduce_dimensionality_with_pca(node_embeddings,
                                                                                              COMMON_EMBEDDING_DIM_PCA,
                                                                                              f"{model_name_to_run}_{variant_name}")
                        embeddings_to_save_dict = {str(i): processed_embeddings_np[i] for i in
                                                   range(processed_embeddings_np.shape[0])}
                        h5_filename = f"{variant_name.replace(' ', '_')}_{model_name_to_run}_embeddings_dim{emb_dim}.h5"
                        h5_path = os.path.join(EMBEDDINGS_OUTPUT_DIR, h5_filename)
                        save_embeddings_to_h5(embeddings_dict_numpy=embeddings_to_save_dict, output_h5_path=h5_path,
                                              model_name=model_name_to_run, dataset_tag_name=variant_name,
                                              final_dim=emb_dim)
                    if PLOT_TRAINING_HISTORY and history and (history.get('loss') or history.get('accuracy')):
                        plot_training_history_torch(history_data=history, model_name_str=model_name_to_run,
                                                    dataset_name_str=variant_name, fold_id="Fold1_CV")
                except Exception as e_run:
                    print(f"  ERROR running {model_name_to_run} on {variant_name}:{e_run}");
                    import traceback;

                    traceback.print_exc()
                    all_results_summary[variant_name][model_name_to_run] = {
                        "avg_accuracy": f"Error: {type(e_run).__name__}", "std_accuracy": str(e_run)}
                end_run_time = time.time()
                if model_name_to_run in all_results_summary[variant_name] and all_results_summary[variant_name][
                    model_name_to_run].get('avg_accuracy') != "Error":
                    print(
                        f"  Time taken for {model_name_to_run} on {variant_name}:{end_run_time - start_run_time:.2f} seconds")
                gc.collect()
    display_results_table_pandas(all_results_summary, KFOLDS, output_file_path=RESULTS_TABLE_FILE)
    print("\nScript finished.")