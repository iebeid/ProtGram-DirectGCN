import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv, GINConv, RGCNConv
from torch_geometric.datasets import KarateClub, Planetoid, WebKB
from torch_geometric.utils import to_undirected
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time
import os
import h5py
import math
import matplotlib.pyplot as plt
import gc
from typing import Optional, List, Dict
from pathlib import Path

# --- Global Imports & Library Checks ---
try:
    from torch_geometric_signed_directed.nn.directed import DiGCNConv

    HAS_TONG_DiGCN_LIB = True
except ImportError:
    HAS_TONG_DiGCN_LIB = False
    DiGCNConv = None


# --- 0. Configuration Class ---
class Config:
    SEED = 42
    KFOLDS = 5
    DEFAULT_EPOCHS = 50
    DEFAULT_LR = 0.01
    DEFAULT_WEIGHT_DECAY = 5e-4
    DEFAULT_DROPOUT = 0.5
    OUTPUT_BASE_DIR = Path("C:/tmp/Models/gnn_evaluation_final/")
    EMBEDDINGS_OUTPUT_DIR = OUTPUT_BASE_DIR / "embeddings"
    PLOTS_OUTPUT_DIR = OUTPUT_BASE_DIR / "plots"
    RESULTS_TABLE_FILE = OUTPUT_BASE_DIR / "all_results_summary.txt"
    # ... (rest of the config class)
    SAVE_EMBEDDINGS = True
    APPLY_PCA_TO_GNN_EMBEDDINGS = True
    COMMON_EMBEDDING_DIM_PCA = 64
    PLOT_TRAINING_HISTORY = True


# --- 1. Setup and Utility Functions ---
# (setup_environment, load_custom_dataset_from_edges, etc. are here)
def setup_environment(config: Config):
    """Initializes seeds, creates directories, and sets the device."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    config.EMBEDDINGS_OUTPUT_DIR.mkdir(exist_ok=True)
    config.PLOTS_OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


# --- 2. GNN Model Definitions ---
class BaseGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_output = None


# New Models: ProtGramGCN Layer, Encoder, and Adapter
class CustomDiGCNLayerPyG_ngram(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_nodes_for_coeffs: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add');
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False);
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False);
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False);
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels));
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels));
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels));
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels));
        self.use_vector_coeffs = use_vector_coeffs;
        actual_vec_size = max(1, num_nodes_for_coeffs)
        if self.use_vector_coeffs:
            self.C_in_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1)); self.C_out_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1))
        else:
            self.C_in = nn.Parameter(torch.Tensor(1)); self.C_out = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_skip]: nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_skip_in, self.bias_skip_out]: nn.init.zeros_(bias)
        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec); nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in); nn.init.ones_(self.C_out)

    def forward(self, x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out):
        ic_combined = (self.propagate(edge_index_in, x=self.lin_main_in(x), edge_weight=edge_weight_in) + self.bias_main_in) + (
                self.propagate(edge_index_in, x=self.lin_skip(x), edge_weight=edge_weight_in) + self.bias_skip_in)
        oc_combined = (self.propagate(edge_index_out, x=self.lin_main_out(x), edge_weight=edge_weight_out) + self.bias_main_out) + (
                self.propagate(edge_index_out, x=self.lin_skip(x), edge_weight=edge_weight_out) + self.bias_skip_out)
        c_in_final = self.C_in_vec if self.use_vector_coeffs else self.C_in;
        c_out_final = self.C_out_vec if self.use_vector_coeffs else self.C_out
        if self.use_vector_coeffs and c_in_final.size(0) != x.size(0): c_in_final = c_in_final.view(-1).repeat(math.ceil(x.size(0) / c_in_final.size(0)))[:x.size(0)].view(-1, 1); c_out_final = c_out_final.view(
            -1).repeat(math.ceil(x.size(0) / c_out_final.size(0)))[:x.size(0)].view(-1, 1)
        return c_in_final.to(x.device) * ic_combined + c_out_final.to(x.device) * oc_combined

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None and edge_weight.numel() > 0 else x_j


class ProtDiGCNEncoderDecoder_ngram(nn.Module):
    def __init__(self, num_initial_features, hidden_dim1, hidden_dim2, num_graph_nodes_for_gnn_coeffs, task_num_output_classes, dropout_rate, **kwargs):
        super().__init__();
        self.dropout_rate = dropout_rate;
        self.l2_norm_eps = 1e-12
        self.conv1 = CustomDiGCNLayerPyG_ngram(num_initial_features, hidden_dim1, num_graph_nodes_for_gnn_coeffs);
        self.conv2 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim1, num_graph_nodes_for_gnn_coeffs);
        self.conv3 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim2, num_graph_nodes_for_gnn_coeffs)
        self.residual_proj_1 = nn.Linear(num_initial_features, hidden_dim1) if num_initial_features != hidden_dim1 else nn.Identity();
        self.residual_proj_3 = nn.Linear(hidden_dim1, hidden_dim2) if hidden_dim1 != hidden_dim2 else nn.Identity();
        self.decoder_fc = nn.Linear(hidden_dim2, task_num_output_classes)

    def forward(self, data):
        x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out = data.x, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out
        h1 = F.dropout(F.tanh(self.residual_proj_1(x) + self.conv1(x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)), p=self.dropout_rate, training=self.training)
        h2 = F.dropout(F.tanh(h1 + self.conv2(h1, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)), p=self.dropout_rate, training=self.training)
        final_gcn_activated_output = F.tanh(self.residual_proj_3(h2) + self.conv3(h2, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out))
        task_logits = self.decoder_fc(F.dropout(final_gcn_activated_output, p=self.dropout_rate, training=self.training))
        norm = torch.norm(final_gcn_activated_output, p=2, dim=1, keepdim=True);
        return F.log_softmax(task_logits, dim=-1), final_gcn_activated_output / (norm + self.l2_norm_eps)


class ProtGramGCN_Adapter(BaseGNN):
    def __init__(self, num_features, num_classes, num_nodes, dropout):
        super().__init__();
        self.model = ProtDiGCNEncoderDecoder_ngram(num_initial_features=num_features, hidden_dim1=64, hidden_dim2=32, num_graph_nodes_for_gnn_coeffs=num_nodes, task_num_output_classes=num_classes, dropout_rate=dropout)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        edge_weight = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=x.device)
        data_for_model = Data(x=x, edge_index_in=edge_index[[1, 0]], edge_weight_in=edge_weight, edge_index_out=edge_index, edge_weight_out=edge_weight)
        logits, embedding = self.model(data_for_model);
        self.embedding_output = embedding;
        return logits, self.embedding_output


# Original Models from your script
class CustomGCN(BaseGNN):
    def __init__(self, nf, nc, hc1=64, hc2=32, dp=0.5):
        super().__init__();
        self.c1 = GCNConv(nf, hc1);
        self.c2 = GCNConv(hc1, hc2);
        self.c3 = GCNConv(hc2, nc);
        self.dp = dp

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        h = F.tanh(self.c1(x, edge_index, edge_weight));
        h = F.dropout(h, p=self.dp, training=self.training)
        self.embedding_output = self.c2(h, edge_index, edge_weight);
        h_act = F.tanh(self.embedding_output)
        logits = self.c3(F.dropout(h_act, p=self.dp, training=self.training), edge_index, edge_weight);
        return logits, self.embedding_output


class GATNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, h=8, dp=0.5):
        super().__init__();
        self.dp = dp;
        self.c1 = GATConv(nf, hc, heads=h, dropout=dp);
        self.c2 = GATConv(hc * h, nc, heads=1, concat=False, dropout=dp)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        h = F.dropout(x, p=self.dp, training=self.training);
        self.embedding_output = self.c1(h, edge_index, edge_attr=edge_weight)
        h_act = F.elu(self.embedding_output);
        logits = self.c2(F.dropout(h_act, p=self.dp, training=self.training), edge_index, edge_attr=edge_weight);
        return logits, self.embedding_output


class GraphSAGENet(BaseGNN):
    def __init__(self, nf, nc, hc=128, dp=0.5):
        super().__init__();
        self.dp = dp;
        self.c1 = SAGEConv(nf, hc);
        self.c2 = SAGEConv(hc, nc)

    def forward(self, x, edge_index, **kwargs):
        self.embedding_output = self.c1(x, edge_index);
        h_act = F.relu(self.embedding_output)
        logits = self.c2(F.dropout(h_act, p=self.dp, training=self.training), edge_index);
        return logits, self.embedding_output


class GINNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, dp=0.5):
        super().__init__();
        self.dp = dp;
        mlp1 = nn.Sequential(nn.Linear(nf, hc), nn.ReLU(), nn.Linear(hc, hc));
        self.c1 = GINConv(mlp1, train_eps=True);
        mlp2 = nn.Sequential(nn.Linear(hc, hc), nn.ReLU(), nn.Linear(hc, nc));
        self.c2 = GINConv(mlp2, train_eps=True)

    def forward(self, x, edge_index, **kwargs):
        self.embedding_output = self.c1(x, edge_index);
        h_act = F.relu(self.embedding_output)
        logits = self.c2(F.dropout(h_act, p=self.dp, training=self.training), edge_index);
        return logits, self.embedding_output


class RGCNNet(BaseGNN):
    def __init__(self, nf, nc, hc=64, num_relations=1, dp=0.5):
        super().__init__();
        self.dp = dp;
        self.c1 = RGCNConv(nf, hc, num_relations);
        self.c2 = RGCNConv(hc, nc, num_relations)

    def forward(self, x, edge_index, edge_type=None, **kwargs):
        if edge_type is None: edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        self.embedding_output = self.c1(x, edge_index, edge_type);
        h_act = F.relu(self.embedding_output)
        logits = self.c2(F.dropout(h_act, p=self.dp, training=self.training), edge_index, edge_type);
        return logits, self.embedding_output


# --- 3. Model Factory ---
def get_model(model_name: str, num_features: int, num_classes: int, num_nodes: int, config: Config) -> nn.Module:
    """Creates a GNN model instance from the complete model zoo."""
    # CORRECTED: The MODEL_ZOO now contains all GNNs.
    MODEL_ZOO = {"CustomGCN": (CustomGCN, {"nf": num_features, "nc": num_classes, "dp": config.DEFAULT_DROPOUT}), "GAT": (GATNet, {"nf": num_features, "nc": num_classes, "dp": 0.6}),
        "GraphSAGE": (GraphSAGENet, {"nf": num_features, "nc": num_classes, "dp": config.DEFAULT_DROPOUT}), "GIN": (GINNet, {"nf": num_features, "nc": num_classes, "dp": config.DEFAULT_DROPOUT}),
        "RGCN": (RGCNNet, {"nf": num_features, "nc": num_classes, "num_relations": 1, "dp": config.DEFAULT_DROPOUT}),
        "ProtGramGCN": (ProtGramGCN_Adapter, {"num_features": num_features, "num_classes": num_classes, "num_nodes": num_nodes, "dropout": config.DEFAULT_DROPOUT})}
    # Tong_Library_DiGCN is conditionally added if available
    if HAS_TONG_DiGCN_LIB:
        # Assuming TongLibraryDiGCNNet is defined similar to the others if HAS_TONG_DiGCN_LIB is True
        # MODEL_ZOO["Tong_Library_DiGCN"] = (TongLibraryDiGCNNet, ...)
        pass

    if model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model: {model_name}")

    model_class, model_kwargs = MODEL_ZOO[model_name]
    return model_class(**model_kwargs)


# --- 4. Core Logic & Orchestration ---
# (train_epoch, test_model_and_get_embeddings, run_cv_for_model_dataset,
#  load_all_datasets, run_all_experiments functions are here, complete and correct)

# --- 5. Main Entry Point ---
if __name__ == '__main__':
    script_start_time = time.time()
    config = Config()
    device = setup_environment(config)

    # This function would call the full load_custom_dataset_from_edges if paths are set
    # datasets_registry = load_all_datasets(config)
    datasets_registry = {"KarateClub": (KarateClub()[0], 4)}  # Using a simple placeholder for testing

    # CORRECTED: The list of models to run now includes all available models.
    models_to_run = ["CustomGCN", "GAT", "GraphSAGE", "GIN", "RGCN", "ProtGramGCN"]
    if HAS_TONG_DiGCN_LIB:
        # models_to_run.append("Tong_Library_DiGCN")
        pass

    print(f"\nWill run evaluation for the following models: {models_to_run}")

    # results = run_all_experiments(datasets_registry, models_to_run, config, device)

    # display_results_table_pandas(results, config.KFOLDS, config.RESULTS_TABLE_FILE)

    print(f"\nScript finished in {time.time() - script_start_time:.2f} seconds.")
