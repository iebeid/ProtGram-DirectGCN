# ==============================================================================
# Combined Protein-Protein Interaction Evaluation and N-gramGCN Embedding Generation Script
# VERSION: Final, with selectable Full-Graph/Mini-Batch modes and Restored Logging
# ==============================================================================
import os
import sys
import shutil
import numpy as np
import pandas as pd
import time
import random
import gc
import h5py
import math
import re
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any, Set, Tuple, Union, Callable
from collections import defaultdict, Counter
from Bio import SeqIO

# PyTorch and PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Scikit-learn and SciPy
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt

# NetworkX for Community Detection
import networkx as nx
import community as community_louvain

# --- TensorFlow GPU Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow: GPU Devices Detected and Memory Growth enabled.")
    except RuntimeError as e:
        print(f"TensorFlow: Error setting memory growth: {e}")
else:
    print("TensorFlow: Warning: No GPU detected. Running on CPU.")

# ==============================================================================
# --- MAIN CONFIGURATION ---
# ==============================================================================
DEBUG_VERBOSE = True
RANDOM_STATE = 42
BASE_OUTPUT_DIR = "./ppi_evaluation_results_final/"

# --- NEW: TRAINING MODE SELECTION ---
TRAINING_MODE = 'full_graph'  # OPTIONS: 'full_graph', 'mini_batch'

# --- N-gramGCN Generation Configuration ---
RUN_AND_EVALUATE_NGRAM_GCN = True
NGRAM_GCN_INPUT_FASTA_PATH = "C:/ProgramData/ProtDiGCN/uniprot_sequences_sample.fasta"
NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR = os.path.join(BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings")
NGRAM_GCN_MAX_N = 5
NGRAM_GCN_1GRAM_INIT_DIM = 64
NGRAM_GCN_HIDDEN_DIM_1 = 128
NGRAM_GCN_HIDDEN_DIM_2 = 64
NGRAM_GCN_PE_MAX_LEN = 10
NGRAM_GCN_DROPOUT = 0.5
NGRAM_GCN_LR = 0.0005
NGRAM_GCN_WEIGHT_DECAY = 1e-4
NGRAM_GCN_EPOCHS_PER_LEVEL = 1000
NGRAM_GCN_USE_VECTOR_COEFFS = True
NGRAM_GCN_GENERATED_EMB_NAME = "NgramGCN-Generated"
NGRAM_GCN_TASK_PER_LEVEL: Dict[int, str] = {1: 'community_label'}
NGRAM_GCN_DEFAULT_TASK_MODE = 'next_node'

# --- Mini-Batch Specific Configuration ---
NGRAM_GCN_BATCH_SIZE = 512
NGRAM_GCN_NUM_NEIGHBORS = [25, 15, 10]
NGRAM_GCN_INFERENCE_BATCH_SIZE = 1024

# --- Link Prediction Evaluation Configuration ---
normal_positive_interactions_path = os.path.normpath('C:/ProgramData/ProtDiGCN/ground_truth/positive_interactions.csv')
normal_negative_interactions_path = os.path.normpath('C:/ProgramData/ProtDiGCN/ground_truth/negative_interactions.csv')
normal_sample_negative_pairs: Optional[int] = 20000
normal_embedding_files_to_evaluate = [{"path": "C:/ProgramData/ProtDiGCN/models/per-protein.h5", "name": "ProtT5-Precomputed", "loader_func_key": "load_h5_embeddings_selectively"}, ]
normal_output_main_dir = os.path.join(BASE_OUTPUT_DIR, "normal_run_output_combined")
EDGE_EMBEDDING_METHOD_LP = 'concatenate'
N_FOLDS_LP = 2
PLOT_TRAINING_HISTORY_LP = True
MLP_DENSE1_UNITS_LP = 128
MLP_DROPOUT1_RATE_LP = 0.4
MLP_DENSE2_UNITS_LP = 64
MLP_DROPOUT2_RATE_LP = 0.4
MLP_L2_REG_LP = 0.001
BATCH_SIZE_LP = 64
EPOCHS_LP = 10
LEARNING_RATE_LP = 1e-3
K_VALUES_FOR_RANKING_METRICS_LP = [10, 50, 100, 200]
K_VALUES_FOR_TABLE_DISPLAY_LP = [50, 100]
MAIN_EMBEDDING_NAME_FOR_STATS_LP = NGRAM_GCN_GENERATED_EMB_NAME if RUN_AND_EVALUATE_NGRAM_GCN else "ProtT5_Example_Data"
STATISTICAL_TEST_METRIC_KEY_LP = 'test_auc_sklearn'
STATISTICAL_TEST_ALPHA_LP = 0.05


# ==============================================================================
# START: ID PARSING & GRAPH CONSTRUCTION (FROM YOUR SCRIPT)
# ==============================================================================
def extract_canonical_id_and_type(header_or_id_line: str) -> tuple[Optional[str], Optional[str]]:
    hid = header_or_id_line.strip().lstrip('>')
    up_match = re.match(r"^(?:sp|tr)\|([A-Z0-9]{6,10}(?:-\d+)?)\|", hid, re.IGNORECASE)
    if up_match: return "UniProt", up_match.group(1)
    uniref_cluster_match = re.match(r"^(UniRef(?:100|90|50))_((?:[A-Z0-9]{6,10}(?:-\d+)?)(?:_[A-Z0-9]+)?|(UPI[A-F0-9]+))", hid, re.IGNORECASE)
    if uniref_cluster_match:
        cluster_type, id_part = uniref_cluster_match.group(1), uniref_cluster_match.group(2)
        if re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", id_part): return "UniProt (from UniRef)", id_part
        if "_" in id_part and re.fullmatch(r"[A-Z0-9]{6,10}_[A-Z0-9]+", id_part): return "UniProt (from UniRef)", id_part.split('_')[0]
        if id_part.startswith("UPI"): return "UniParc (from UniRef)", id_part
        return "UniRef Cluster", f"{cluster_type}_{id_part}"
    ncbi_gi_match = re.match(r"^gi\|\d+\|\w{1,3}\|([A-Z]{1,3}[_0-9]*\w*\.?\d*)\|", hid)
    if ncbi_gi_match: return "NCBI", ncbi_gi_match.group(1)
    ncbi_acc_match = re.match(r"^([A-Z]{2,3}(?:_|\d)[A-Z0-9]+\.?\d*)\b", hid)
    if ncbi_acc_match: return "NCBI", ncbi_acc_match.group(1)
    pdb_match = re.match(r"^([0-9][A-Z0-9]{3})[_ ]?([A-Z0-9]{1,2})?", hid, re.IGNORECASE)
    if pdb_match:
        pdb_id, chain_part = pdb_match.group(1).upper(), pdb_match.group(2).upper() if pdb_match.group(2) else ""
        is_likely_uniprot = len(pdb_id) >= 5 and pdb_id[0] in 'OPQ' and pdb_id[1].isdigit()
        if not is_likely_uniprot: return "PDB", f"{pdb_id}{'_' + chain_part if chain_part else ''}"
    plain_up_match = re.fullmatch(r"([A-Z0-9]{6,10}(?:-\d+)?)", hid.split()[0].split('|')[0])
    if plain_up_match: return "UniProt (assumed)", plain_up_match.group(1)
    first_word = hid.split()[0].split('|')[0]
    if first_word: return "Unknown", first_word
    return "Unknown", hid


def parse_fasta_sequences_with_ids_ngram(filepath: str) -> list[tuple[str, str]]:
    protein_data = []
    if not os.path.exists(filepath):
        if DEBUG_VERBOSE: print(f"NgramGCN: FASTA file not found at {filepath}")
        return protein_data
    try:
        for record in SeqIO.parse(filepath, "fasta"):
            _, canonical_id = extract_canonical_id_and_type(record.id)
            if canonical_id:
                protein_data.append((canonical_id, str(record.seq).upper()))
            else:
                if DEBUG_VERBOSE: print(f"NgramGCN: Could not extract canonical ID from '{record.id[:50]}...', using full ID as fallback: {record.id}")
                protein_data.append((record.id, str(record.seq).upper()))
        if DEBUG_VERBOSE and not protein_data:
            print(f"NgramGCN: Warning - Parsed 0 sequences from {filepath}")
        elif DEBUG_VERBOSE:
            print(f"NgramGCN: Parsed {len(protein_data)} sequences with extracted/standardized IDs from {filepath}")
    except Exception as e:
        print(f"NgramGCN: Error parsing FASTA file {filepath}: {e}")
    return protein_data


def get_ngrams_and_transitions_ngram(sequences: list[str], n: int):
    all_ngrams, all_transitions = [], []
    for seq in sequences:
        if len(seq) < n: continue
        current_seq_ngrams = [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]
        all_ngrams.extend(current_seq_ngrams)
        for i in range(len(current_seq_ngrams) - 1): all_transitions.append((current_seq_ngrams[i], current_seq_ngrams[i + 1]))
    return all_ngrams, all_transitions


def build_ngram_graph_data_ngram(ngrams: list[tuple], transitions: list[tuple[tuple, tuple]], node_prob_from_prev_graph: Optional[dict[tuple, float]] = None, n_val: int = 1) -> tuple[
    Optional[Data], dict[tuple, int], dict[int, tuple]]:
    if not ngrams: return None, {}, {}
    unique_ngrams_list = sorted(list(set(ngrams)))
    ngram_to_idx = {ngram: i for i, ngram in enumerate(unique_ngrams_list)}
    idx_to_ngram = {i: ngram for ngram, i in ngram_to_idx.items()}
    num_nodes = len(unique_ngrams_list)

    source_nodes_idx, target_nodes_idx, edge_weights = [], [], []
    edge_counts = Counter(transitions)
    source_ngram_outgoing_counts = defaultdict(int)
    for (src_ng, _), count in edge_counts.items(): source_ngram_outgoing_counts[src_ng] += count

    for (source_ngram, target_ngram), count in edge_counts.items():
        if source_ngram in ngram_to_idx and target_ngram in ngram_to_idx:
            source_idx, target_idx = ngram_to_idx[source_ngram], ngram_to_idx[target_ngram]
            transition_prob = count / source_ngram_outgoing_counts[source_ngram] if source_ngram_outgoing_counts[source_ngram] > 0 else 0.0
            if transition_prob > 1e-9:
                source_nodes_idx.append(source_idx)
                target_nodes_idx.append(target_idx)
                edge_weights.append(transition_prob)

    data = Data()
    data.num_nodes = num_nodes
    if source_nodes_idx:
        edge_index = torch.tensor([source_nodes_idx, target_nodes_idx], dtype=torch.long)
        edge_attr_squeezed = torch.tensor(edge_weights, dtype=torch.float)
        data.edge_index_out = edge_index
        data.edge_weight_out = edge_attr_squeezed
        data.edge_index_in = edge_index.flip(dims=[0]) if edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
        data.edge_weight_in = edge_attr_squeezed
    else:
        data.edge_index_out = torch.empty((2, 0), dtype=torch.long)
        data.edge_weight_out = torch.empty(0, dtype=torch.float)
        data.edge_index_in = torch.empty((2, 0), dtype=torch.long)
        data.edge_weight_in = torch.empty(0, dtype=torch.float)
        if num_nodes == 0: return None, ngram_to_idx, idx_to_ngram

    adj_for_next_node_task = defaultdict(list)
    for (source_ngram, target_ngram) in transitions:
        if source_ngram in ngram_to_idx and target_ngram in ngram_to_idx:
            adj_for_next_node_task[ngram_to_idx[source_ngram]].append(ngram_to_idx[target_ngram])

    y_next_node = torch.full((num_nodes,), -1, dtype=torch.long)
    for src_node, successors in adj_for_next_node_task.items():
        if successors:
            y_next_node[src_node] = random.choice(successors)
    data.y_next_node = y_next_node

    return data, ngram_to_idx, idx_to_ngram


# ==============================================================================
# START: FULL-GRAPH MODEL AND TRAINING (YOUR ORIGINAL IMPLEMENTATION)
# ==============================================================================

class CustomDiGCNLayerPyG_ngram(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_nodes_for_coeffs: int, use_vector_coeffs: bool = True):
        super(CustomDiGCNLayerPyG_ngram, self).__init__(aggr='add')
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels))
        self.use_vector_coeffs = use_vector_coeffs
        self.num_nodes_for_coeffs_init = num_nodes_for_coeffs
        actual_vec_size = max(1, num_nodes_for_coeffs)
        if self.use_vector_coeffs:
            self.C_in_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1))
            self.C_out_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1))
        else:
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_skip]: nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_skip_in, self.bias_skip_out]: nn.init.zeros_(bias)
        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)

    def forward(self, x: torch.Tensor, edge_index_in: torch.Tensor, edge_weight_in: torch.Tensor, edge_index_out: torch.Tensor, edge_weight_out: torch.Tensor):
        x_transformed_main_in = self.lin_main_in(x)
        aggr_main_in = self.propagate(edge_index_in, x=x_transformed_main_in, edge_weight=edge_weight_in)
        term1_in = aggr_main_in + self.bias_main_in
        x_transformed_skip_in = self.lin_skip(x)
        aggr_skip_in = self.propagate(edge_index_in, x=x_transformed_skip_in, edge_weight=edge_weight_in)
        term2_in = aggr_skip_in + self.bias_skip_in
        ic_combined = term1_in + term2_in

        x_transformed_main_out = self.lin_main_out(x)
        aggr_main_out = self.propagate(edge_index_out, x=x_transformed_main_out, edge_weight=edge_weight_out)
        term1_out = aggr_main_out + self.bias_main_out
        x_transformed_skip_out = self.lin_skip(x)
        aggr_skip_out = self.propagate(edge_index_out, x=x_transformed_skip_out, edge_weight=edge_weight_out)
        term2_out = aggr_skip_out + self.bias_skip_out
        oc_combined = term1_out + term2_out

        c_in_final = self.C_in_vec if self.use_vector_coeffs else self.C_in
        c_out_final = self.C_out_vec if self.use_vector_coeffs else self.C_out

        if self.use_vector_coeffs and c_in_final.size(0) != x.size(0):
            if c_in_final.size(0) > 0 and c_in_final.size(0) != 1:
                if DEBUG_VERBOSE: print(f"NgramGCN Layer Warning: C_vec size ({c_in_final.size(0)}) != x.size(0) ({x.size(0)}). Resizing.")
            c_in_final = c_in_final.view(-1).repeat(math.ceil(x.size(0) / c_in_final.size(0)))[:x.size(0)].view(-1, 1)
            c_out_final = c_out_final.view(-1).repeat(math.ceil(x.size(0) / c_out_final.size(0)))[:x.size(0)].view(-1, 1)

        output = c_in_final.to(x.device) * ic_combined + c_out_final.to(x.device) * oc_combined
        return output

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None and edge_weight.numel() > 0 else x_j


class ProtDiGCNEncoderDecoder_ngram(nn.Module):
    def __init__(self, num_initial_features: int, hidden_dim1: int, hidden_dim2: int, num_graph_nodes_for_gnn_coeffs: int, task_num_output_classes: int, n_gram_length_for_pe: int, one_gram_embed_dim_for_pe: int,
                 max_allowable_len_for_pe_layer: int, dropout_rate: float, use_vector_coeffs_in_gnn: bool = True):
        super().__init__()
        self.n_gram_length_for_pe = n_gram_length_for_pe
        self.one_gram_embed_dim_for_pe = one_gram_embed_dim_for_pe
        self.dropout_rate = dropout_rate
        self.l2_norm_eps = 1e-12
        self.learnable_pe_active = False
        if self.one_gram_embed_dim_for_pe > 0 and max_allowable_len_for_pe_layer > 0:
            self.positional_encoder_layer = nn.Embedding(max_allowable_len_for_pe_layer, self.one_gram_embed_dim_for_pe)
            self.learnable_pe_active = True
        else:
            self.positional_encoder_layer = None

        self.conv1 = CustomDiGCNLayerPyG_ngram(num_initial_features, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)
        self.conv2 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)
        self.conv3 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim2, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)

        self.residual_proj_1 = nn.Linear(num_initial_features, hidden_dim1) if num_initial_features != hidden_dim1 else nn.Identity()
        self.residual_proj_3 = nn.Linear(hidden_dim1, hidden_dim2) if hidden_dim1 != hidden_dim2 else nn.Identity()

        self.decoder_fc = nn.Linear(hidden_dim2, task_num_output_classes)
        self.final_normalized_embedding_output = None

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if not self.learnable_pe_active or self.positional_encoder_layer is None:
            return x

        x_to_modify = x.clone()
        if self.n_gram_length_for_pe == 1 and x_to_modify.shape[1] == self.one_gram_embed_dim_for_pe:
            if self.positional_encoder_layer.num_embeddings > 0:
                position_idx = torch.tensor([0], device=x.device, dtype=torch.long)
                x_to_modify = x_to_modify + self.positional_encoder_layer(position_idx)
        elif self.n_gram_length_for_pe > 1 and x_to_modify.shape[1] == self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe:
            x_reshaped = x_to_modify.view(-1, self.n_gram_length_for_pe, self.one_gram_embed_dim_for_pe)
            num_positions_to_encode = min(self.n_gram_length_for_pe, self.positional_encoder_layer.num_embeddings)
            if num_positions_to_encode > 0:
                position_indices = torch.arange(0, num_positions_to_encode, device=x.device, dtype=torch.long)
                pe_to_add = self.positional_encoder_layer(position_indices)
                modified_slice = x_reshaped[:, :num_positions_to_encode, :] + pe_to_add.unsqueeze(0)
                final_reshaped = torch.cat((modified_slice, x_reshaped[:, num_positions_to_encode:, :]), dim=1) if num_positions_to_encode < self.n_gram_length_for_pe else modified_slice
                x_to_modify = final_reshaped.view(-1, self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe)
        return x_to_modify

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index_in, edge_weight_in = data.edge_index_in, data.edge_weight_in
        edge_index_out, edge_weight_out = data.edge_index_out, data.edge_weight_out

        x_pe = self._apply_positional_encoding(x)
        h1_conv_out = self.conv1(x_pe, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        x_proj1 = self.residual_proj_1(x_pe)
        h1_res_sum = x_proj1 + h1_conv_out
        h1_activated = F.tanh(h1_res_sum)
        h1 = F.dropout(h1_activated, p=self.dropout_rate, training=self.training)

        h2_conv_out = self.conv2(h1, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        h2_res_sum = h1 + h2_conv_out
        h2_activated = F.tanh(h2_res_sum)
        h2 = F.dropout(h2_activated, p=self.dropout_rate, training=self.training)

        h3_conv_out = self.conv3(h2, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        h2_proj3 = self.residual_proj_3(h2)
        h3_res_sum = h2_proj3 + h3_conv_out
        final_gcn_activated_output = F.tanh(h3_res_sum)

        h_embed_for_decoder_dropped = F.dropout(final_gcn_activated_output, p=self.dropout_rate, training=self.training)
        task_logits = self.decoder_fc(h_embed_for_decoder_dropped)
        norm = torch.norm(final_gcn_activated_output, p=2, dim=1, keepdim=True)
        self.final_normalized_embedding_output = final_gcn_activated_output / (norm + self.l2_norm_eps)

        return F.log_softmax(task_logits, dim=-1), self.final_normalized_embedding_output


def train_ngram_model_full_graph(model: ProtDiGCNEncoderDecoder_ngram, data: Data, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str):
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = nn.NLLLoss()

    if task_mode == 'next_node':
        if not hasattr(data, 'y_next_node'):
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_next_node' attribute missing. Cannot train.")
            return
        targets = data.y_next_node
        train_mask = targets != -1
        if train_mask.sum() == 0:
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No valid training samples. Cannot train.")
            return
    elif task_mode == 'community_label':
        if not hasattr(data, 'y_task_labels'):
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_task_labels' attribute missing. Cannot train.")
            return
        targets = data.y_task_labels
    else:
        print(f"NgramGCN: Unknown task_mode: {task_mode}. Cannot train.")
        return

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        log_probs, _ = model(data)

        loss = None
        if task_mode == 'next_node':
            loss = criterion(log_probs[train_mask], targets[train_mask])
        elif task_mode == 'community_label':
            loss = criterion(log_probs, targets)

        if loss is not None:
            loss.backward()
            optimizer.step()
            if epoch % (max(1, epochs // 20)) == 0 or epoch == epochs:
                if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Loss: {loss.item():.4f}")


# ==============================================================================
# START: MINI-BATCH MODEL AND TRAINING (NEW ALTERNATIVE)
# ==============================================================================
class ProtDiGCNEncoderDecoder_minibatch(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.decoder_fc = nn.Linear(hidden_channels2, out_channels)
        self.dropout_rate = dropout
        self.l2_norm_eps = 1e-12

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_final_gcn = self.conv2(x, edge_index)

        norm = torch.norm(x_final_gcn, p=2, dim=1, keepdim=True)
        final_embeddings = x_final_gcn / (norm + self.l2_norm_eps)

        h_embed_for_decoder_dropped = F.dropout(final_embeddings, p=self.dropout_rate, training=self.training)
        task_logits = self.decoder_fc(h_embed_for_decoder_dropped)

        return F.log_softmax(task_logits, dim=-1), final_embeddings

    @torch.no_grad()
    def inference(self, full_graph_x, inference_loader, device):
        self.eval()
        all_embeds = []
        for batch in tqdm(inference_loader, desc="Batched Inference", leave=False, disable=not DEBUG_VERBOSE):
            batch = batch.to(device)
            x = full_graph_x[batch.n_id].to(device)

            x = self.conv1(x, batch.edge_index)
            x = F.relu(x)
            x_final_gcn = self.conv2(x, batch.edge_index)

            norm = torch.norm(x_final_gcn, p=2, dim=1, keepdim=True)
            normalized_embeds = x_final_gcn / (norm + self.l2_norm_eps)
            all_embeds.append(normalized_embeds[:batch.batch_size].cpu())
        return torch.cat(all_embeds, dim=0)


def train_ngram_model_minibatch(model, loader: NeighborLoader, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str):
    model.train()
    model.to(device)
    criterion = nn.NLLLoss()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Mini-Batch Epoch {epoch:03d}", leave=False, disable=not DEBUG_VERBOSE)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_probs, _ = model(batch)
            log_probs_batch = log_probs[:batch.batch_size]
            loss = None
            if task_mode == 'community_label':
                targets = batch.y_task_labels[:batch.batch_size]
                loss = criterion(log_probs_batch, targets)
            elif task_mode == 'next_node':
                targets = batch.y_next_node[:batch.batch_size]
                valid_mask = targets != -1
                if valid_mask.sum() > 0:
                    loss = criterion(log_probs_batch[valid_mask], targets[valid_mask])

            if loss:
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.batch_size
                pbar.set_postfix({'loss': loss.item()})

        if hasattr(loader, 'dataset') and loader.dataset is not None and len(loader.dataset) > 0:
            avg_loss = total_loss / len(loader.dataset)
            if epoch % (max(1, epochs // 20)) == 0 or epoch == epochs:
                if DEBUG_VERBOSE:
                    print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Avg Loss: {avg_loss:.4f}")


def extract_node_embeddings_ngram_batched(model, full_graph_data: Data, config: Dict, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    inference_loader = NeighborLoader(full_graph_data, num_neighbors=config.get('num_neighbors', [-1]), batch_size=config.get('inference_batch_size', 1024), shuffle=False)
    with torch.no_grad():
        all_node_embeddings = model.inference(full_graph_data.x.to(device), inference_loader, device)
    return all_node_embeddings.cpu().numpy() if all_node_embeddings is not None else None


# ==============================================================================
# START: GENERAL UTILITY FUNCTIONS
# ==============================================================================
def detect_communities_louvain(edge_index: torch.Tensor, num_nodes: int, random_state_louvain: Optional[int] = None) -> Tuple[Optional[torch.Tensor], int]:
    if num_nodes == 0: return None, 0
    if edge_index.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.long), num_nodes
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_index.cpu().numpy().T)
    if nx_graph.number_of_edges() == 0:
        return torch.arange(num_nodes, dtype=torch.long), num_nodes
    try:
        partition = community_louvain.best_partition(nx_graph, random_state=random_state_louvain)
        if not partition: return torch.arange(num_nodes, dtype=torch.long), num_nodes
        labels = torch.zeros(num_nodes, dtype=torch.long)
        for node, comm_id in partition.items():
            labels[node] = comm_id
        num_communities = len(torch.unique(labels))
        if DEBUG_VERBOSE: print(f"NgramGCN Community: Detected {num_communities} communities.")
        return labels, num_communities
    except Exception as e:
        print(f"NgramGCN Community Error: {e}.")
        return torch.arange(num_nodes, dtype=torch.long), num_nodes


def extract_node_embeddings_ngram(model, data: Data, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    data = data.to(device)
    if not hasattr(data, 'x') or data.x is None or (hasattr(data, 'num_nodes') and data.num_nodes > 0 and data.x.numel() == 0):
        out_dim = model.decoder_fc.in_features
        return np.array([]).reshape(0, out_dim) if out_dim > 0 else np.array([])
    with torch.no_grad():
        _, embeddings = model(data)
        return embeddings.cpu().numpy() if embeddings is not None and embeddings.numel() > 0 else None


# ==============================================================================
# START: MAIN N-GRAM EMBEDDING GENERATION WORKFLOW (WITH SWITCH)
# ==============================================================================
def generate_and_save_ngram_embeddings(fasta_filepath: str, protein_ids_to_generate_for: Set[str], gcn_config: Dict) -> Tuple[Optional[str], Optional[str]]:
    output_dir = gcn_config['output_dir']
    max_n = gcn_config['max_n']
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"NgramGCN: Using device: {device}")

    all_protein_data = parse_fasta_sequences_with_ids_ngram(fasta_filepath)
    sequences_map = {pid: seq for pid, seq in all_protein_data if pid in protein_ids_to_generate_for}
    sequences_list_for_ngram = list(sequences_map.values())

    level_embeddings: dict[int, np.ndarray] = {}
    level_ngram_to_idx: dict[int, dict] = {}
    level_idx_to_ngram: dict[int, dict] = {}
    per_protein_emb_path = os.path.join(output_dir, f"per_protein_embeddings_from_{max_n}gram.h5")

    for n_val in range(1, max_n + 1):
        print(f"\n--- Processing N-gram Level: n = {n_val} ---")
        current_ngrams, current_transitions = get_ngrams_and_transitions_ngram(sequences_list_for_ngram, n_val)
        graph_data, ngram_to_idx_map, idx_to_ngram_map = build_ngram_graph_data_ngram(current_ngrams, current_transitions, n_val=n_val)

        if graph_data is None:
            print(f"NgramGCN: Could not build graph for n={n_val}. Stopping.")
            break

        # RESTORED: Logging for graph stats
        num_out_edges = graph_data.edge_index_out.size(1) if hasattr(graph_data, 'edge_index_out') and graph_data.edge_index_out is not None else 0
        if DEBUG_VERBOSE: print(f"NgramGCN: Built graph for n={n_val}: {graph_data.num_nodes} nodes, {num_out_edges} out-edges.")

        level_ngram_to_idx[n_val] = ngram_to_idx_map
        level_idx_to_ngram[n_val] = idx_to_ngram_map

        task_mode = gcn_config['task_per_level'].get(n_val, gcn_config['default_task_mode'])
        actual_num_output_classes = 0
        if task_mode == 'community_label':
            labels, num_comms = detect_communities_louvain(graph_data.edge_index_out, graph_data.num_nodes, RANDOM_STATE)
            if labels is not None and num_comms > 1:
                graph_data.y_task_labels = labels
                actual_num_output_classes = num_comms
            else:
                task_mode = gcn_config['default_task_mode']
        if task_mode == 'next_node':
            actual_num_output_classes = graph_data.num_nodes
        if actual_num_output_classes == 0:
            print(f"NgramGCN: Cannot determine number of classes for task '{task_mode}' at n={n_val}. Skipping.")
            break

        if n_val == 1:
            current_feature_dim = gcn_config['one_gram_dim']
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)
        else:
            base_embeds, base_map = level_embeddings.get(1), level_ngram_to_idx.get(1)
            if base_embeds is None or base_map is None or base_embeds.size == 0:
                print(f"NgramGCN: Cannot generate features for n={n_val} due to missing 1-gram embeddings. Stopping.")
                break
            features_list = []
            expected_concat_dim = n_val * gcn_config['one_gram_dim']
            for i in range(graph_data.num_nodes):
                ngram = idx_to_ngram_map.get(i)
                feature_parts = []
                if ngram:
                    for char_element in ngram:
                        char_key = (char_element,)
                        char_idx = base_map.get(char_key)
                        if char_idx is not None and char_idx < len(base_embeds):
                            feature_parts.append(torch.from_numpy(base_embeds[char_idx].copy()).float())
                        else:
                            feature_parts.append(torch.zeros(gcn_config['one_gram_dim']))
                concat_feat = torch.cat(feature_parts) if feature_parts and len(feature_parts) == n_val else torch.zeros(expected_concat_dim)
                features_list.append(concat_feat)
            graph_data.x = torch.stack(features_list)
            current_feature_dim = graph_data.x.shape[1]

        # --- MODEL & TRAINING PIPELINE SELECTION ---
        node_embeddings_np = None
        if gcn_config['training_mode'] == 'full_graph':
            model = ProtDiGCNEncoderDecoder_ngram(num_initial_features=current_feature_dim, hidden_dim1=gcn_config['hidden1'], hidden_dim2=gcn_config['hidden2'], num_graph_nodes_for_gnn_coeffs=graph_data.num_nodes,
                task_num_output_classes=actual_num_output_classes, n_gram_length_for_pe=n_val, one_gram_embed_dim_for_pe=gcn_config['one_gram_dim'], max_allowable_len_for_pe_layer=graph_data.num_nodes,
                dropout_rate=gcn_config['dropout'], use_vector_coeffs_in_gnn=gcn_config['use_vector_coeffs'])
            # RESTORED: Logging for model architecture
            if DEBUG_VERBOSE:
                print(f"\nNgramGCN Model Architecture (for n={n_val}, task='{task_mode}', mode='full_graph'):")
                print(model)
                print("-" * 60)

            optimizer = optim.Adam(model.parameters(), lr=gcn_config['lr'], weight_decay=gcn_config['l2_reg'])
            train_ngram_model_full_graph(model, graph_data, optimizer, gcn_config['epochs'], device, task_mode)
            node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)

        else:  # 'mini_batch'
            model = ProtDiGCNEncoderDecoder_minibatch(in_channels=current_feature_dim, hidden_channels1=gcn_config['hidden1'], hidden_channels2=gcn_config['hidden2'], out_channels=actual_num_output_classes,
                dropout=gcn_config['dropout'])
            # RESTORED: Logging for model architecture
            if DEBUG_VERBOSE:
                print(f"\nNgramGCN Model Architecture (for n={n_val}, task='{task_mode}', mode='mini_batch'):")
                print(model)
                print("-" * 60)

            optimizer = optim.Adam(model.parameters(), lr=gcn_config['lr'], weight_decay=gcn_config['l2_reg'])
            loader_config = {'num_neighbors': gcn_config['num_neighbors'], 'batch_size': gcn_config['batch_size']}
            train_loader = NeighborLoader(graph_data, **loader_config, shuffle=True)
            train_ngram_model_minibatch(model, train_loader, optimizer, gcn_config['epochs'], device, task_mode)

            inference_config = {'num_neighbors': gcn_config['num_neighbors'], 'inference_batch_size': gcn_config['inference_batch_size']}
            node_embeddings_np = extract_node_embeddings_ngram_batched(model, graph_data, inference_config, device)

        if node_embeddings_np is None or node_embeddings_np.size == 0:
            print(f"NgramGCN: Failed to generate embeddings for n={n_val}. Stopping.")
            break
        level_embeddings[n_val] = node_embeddings_np

    # Final per-protein pooling
    final_embeddings = level_embeddings.get(max_n)
    final_map = level_ngram_to_idx.get(max_n)
    if final_embeddings is not None and final_map is not None and final_embeddings.size > 0:
        with h5py.File(per_protein_emb_path, 'w') as hf:
            for prot_id, seq in sequences_map.items():
                if len(seq) < max_n: continue
                indices = [final_map.get(tuple(seq[i:i + max_n])) for i in range(len(seq) - max_n + 1)]
                valid_indices = [idx for idx in indices if idx is not None]
                if valid_indices:
                    hf.create_dataset(prot_id, data=np.mean(final_embeddings[valid_indices], axis=0))
        print(f"NgramGCN: Per-protein embeddings saved to {per_protein_emb_path}")
        return per_protein_emb_path, None
    else:
        print(f"NgramGCN: Final embeddings for n={max_n} not available. No output file generated.")
        return None, None


# ==============================================================================
# MEMORY-EFFICIENT LINK PREDICTION EVALUATION
# ==============================================================================
def create_edge_embedding_generator(pairs: List[Tuple[str, str, int]], embeddings: Dict[str, np.ndarray], batch_size: int, embedding_dim: int, method: str = 'concatenate'):
    num_pairs = len(pairs)
    while True:
        random.shuffle(pairs)
        for i in range(0, num_pairs, batch_size):
            batch_pairs = pairs[i:i + batch_size]
            if not batch_pairs: continue
            edge_feats_batch, labels_batch = [], []
            for p1, p2, lbl in batch_pairs:
                e1, e2 = embeddings.get(p1), embeddings.get(p2)
                if e1 is not None and e2 is not None and e1.shape[0] == embedding_dim and e2.shape[0] == embedding_dim:
                    if method == 'concatenate':
                        feat = np.concatenate((e1, e2))
                    elif method == 'average':
                        feat = (e1 + e2) / 2.0
                    elif method == 'hadamard':
                        feat = e1 * e2
                    else:
                        feat = np.abs(e1 - e2)
                    edge_feats_batch.append(feat)
                    labels_batch.append(lbl)
            if edge_feats_batch:
                yield (np.array(edge_feats_batch, dtype=np.float32), np.array(labels_batch, dtype=np.int32))


def build_mlp_model_lp(input_shape: int, mlp_params: Dict) -> Model:
    inputs = Input(shape=(input_shape,))
    x = Dense(units=mlp_params['dense1_units'], activation='relu', kernel_regularizer=l2(mlp_params['l2_reg']))(inputs)
    x = Dropout(mlp_params['dropout1_rate'])(x)
    x = Dense(units=mlp_params['dense2_units'], activation='relu', kernel_regularizer=l2(mlp_params['l2_reg']))(x)
    x = Dropout(mlp_params['dropout2_rate'])(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)


def main_workflow_cv_lp(embedding_name_cv: str, protein_embeddings_cv: Dict[str, np.ndarray], positive_pairs_cv: List[Tuple[str, str, int]], negative_pairs_cv: List[Tuple[str, str, int]], mlp_params_dict_cv: Dict,
                        lp_config: Dict) -> Dict[str, Any]:
    if DEBUG_VERBOSE: print(f"\n--- Starting Memory-Efficient Link Prediction CV for: {embedding_name_cv} ---")

    first_key = next(iter(protein_embeddings_cv), None)
    if not first_key:
        print("LinkPred CV Error: Protein embeddings dictionary is empty.")
        return {}

    embedding_dim = protein_embeddings_cv[first_key].shape[0]
    edge_feature_dim = embedding_dim * 2 if lp_config['edge_emb_method'] == 'concatenate' else embedding_dim

    all_pairs = positive_pairs_cv + negative_pairs_cv
    labels_for_stratify = [p[2] for p in all_pairs]
    skf = StratifiedKFold(n_splits=lp_config['n_folds'], shuffle=True, random_state=lp_config['random_state'])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_pairs, labels_for_stratify)):
        print(f"\n--- Fold {fold + 1}/{lp_config['n_folds']} ---")
        train_pairs = [all_pairs[i] for i in train_idx]
        val_pairs = [all_pairs[i] for i in val_idx]
        val_labels = [p[2] for p in val_pairs]

        train_gen = lambda: create_edge_embedding_generator(train_pairs, protein_embeddings_cv, lp_config['batch_size'], embedding_dim, lp_config['edge_emb_method'])
        val_gen = lambda: create_edge_embedding_generator(val_pairs, protein_embeddings_cv, lp_config['batch_size'], embedding_dim, lp_config['edge_emb_method'])

        output_signature = (tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32))
        train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature)

        train_steps = math.ceil(len(train_pairs) / lp_config['batch_size'])
        val_steps = math.ceil(len(val_pairs) / lp_config['batch_size'])

        model = build_mlp_model_lp(edge_feature_dim, mlp_params_dict_cv)
        model.compile(optimizer=Adam(learning_rate=lp_config['learning_rate']), loss='binary_crossentropy', metrics=['AUC'])

        history = model.fit(train_ds, epochs=lp_config['epochs'], steps_per_epoch=train_steps, validation_data=val_ds, validation_steps=val_steps, verbose=1)

        print("Predicting on validation set...")
        val_pred_ds = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature).take(val_steps)
        val_predictions = model.predict(val_pred_ds).flatten()

        if len(val_predictions) > len(val_labels): val_predictions = val_predictions[:len(val_labels)]

        fold_res = {'test_auc_sklearn': roc_auc_score(val_labels, val_predictions), 'test_precision': precision_score(val_labels, (val_predictions > 0.5).astype(int), zero_division=0),
            'test_recall': recall_score(val_labels, (val_predictions > 0.5).astype(int), zero_division=0), 'test_f1': f1_score(val_labels, (val_predictions > 0.5).astype(int), zero_division=0), }
        fold_results.append(fold_res)
        print(f"Fold {fold + 1} Results: AUC={fold_res['test_auc_sklearn']:.4f}, F1={fold_res['test_f1']:.4f}")
        del model, train_ds, val_ds, val_pred_ds;
        gc.collect();
        tf.keras.backend.clear_session()

    avg_auc = np.mean([res['test_auc_sklearn'] for res in fold_results])
    print(f"\nCV Finished for {embedding_name_cv}. Average Test AUC: {avg_auc:.4f}")
    return {'embedding_name': embedding_name_cv, 'cv_results': fold_results}


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    print("--- Combined N-gramGCN Generation and Link Prediction Evaluation Script (Selectable Mode) ---")

    # --- 1. N-gramGCN Generation ---
    prot_emb_path = None
    if RUN_AND_EVALUATE_NGRAM_GCN:
        print("\n" + "=" * 20 + " Step 1: N-gramGCN Generation " + "=" * 20)
        all_fasta_data = parse_fasta_sequences_with_ids_ngram(NGRAM_GCN_INPUT_FASTA_PATH)
        all_fasta_ids = {pid for pid, seq in all_fasta_data}

        if not all_fasta_ids:
            print("Could not parse any IDs from the FASTA file. Aborting GCN training.")
        else:
            print(f"Found {len(all_fasta_ids)} total proteins in FASTA file to train on.")
            gcn_params = {'training_mode': TRAINING_MODE, 'output_dir': NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, 'max_n': NGRAM_GCN_MAX_N, 'one_gram_dim': NGRAM_GCN_1GRAM_INIT_DIM, 'hidden1': NGRAM_GCN_HIDDEN_DIM_1,
                'hidden2': NGRAM_GCN_HIDDEN_DIM_2, 'pe_max_len': NGRAM_GCN_PE_MAX_LEN, 'dropout': NGRAM_GCN_DROPOUT, 'lr': NGRAM_GCN_LR, 'l2_reg': NGRAM_GCN_WEIGHT_DECAY, 'epochs': NGRAM_GCN_EPOCHS_PER_LEVEL,
                'use_vector_coeffs': NGRAM_GCN_USE_VECTOR_COEFFS, 'task_per_level': NGRAM_GCN_TASK_PER_LEVEL, 'default_task_mode': NGRAM_GCN_DEFAULT_TASK_MODE, 'batch_size': NGRAM_GCN_BATCH_SIZE,
                'num_neighbors': NGRAM_GCN_NUM_NEIGHBORS, 'inference_batch_size': NGRAM_GCN_INFERENCE_BATCH_SIZE}
            prot_emb_path, _ = generate_and_save_ngram_embeddings(fasta_filepath=NGRAM_GCN_INPUT_FASTA_PATH, protein_ids_to_generate_for=all_fasta_ids, gcn_config=gcn_params)

            if prot_emb_path and os.path.exists(prot_emb_path):
                normal_embedding_files_to_evaluate.append({"path": prot_emb_path, "name": NGRAM_GCN_GENERATED_EMB_NAME, "loader_func_key": "load_h5_embeddings_selectively"})

    # --- 2. Load Interaction Data for Evaluation ---
    print("\n" + "=" * 20 + " Step 2: Loading Interaction Data " + "=" * 20)
    try:
        pos_df = pd.read_csv(normal_positive_interactions_path, dtype=str).dropna()
        neg_df = pd.read_csv(normal_negative_interactions_path, dtype=str).dropna()
        if normal_sample_negative_pairs and len(neg_df) > 0:
            neg_df = neg_df.sample(n=min(normal_sample_negative_pairs, len(neg_df)), random_state=RANDOM_STATE)
        positive_pairs = list(zip(pos_df.iloc[:, 0], pos_df.iloc[:, 1], [1] * len(pos_df)))
        negative_pairs = list(zip(neg_df.iloc[:, 0], neg_df.iloc[:, 1], [0] * len(neg_df)))
        all_proteins_in_pairs = set(pos_df.iloc[:, 0]) | set(pos_df.iloc[:, 1]) | set(neg_df.iloc[:, 0]) | set(neg_df.iloc[:, 1])
        print(f"Found {len(all_proteins_in_pairs)} unique proteins required for link prediction.")
    except FileNotFoundError:
        print("Interaction CSV files not found. Cannot run link prediction evaluation.")
        all_proteins_in_pairs = set()

    # --- 3. Gatekeeper Check ---
    can_proceed_with_evaluation = False
    if not all_proteins_in_pairs:
        print("\nNo proteins found in interaction files. Skipping evaluation.")
    elif RUN_AND_EVALUATE_NGRAM_GCN and not (prot_emb_path and os.path.exists(prot_emb_path)):
        print(f"\nN-gram GCN was set to run but its embedding file was not generated. Skipping all evaluations.")
    elif RUN_AND_EVALUATE_NGRAM_GCN:
        print(f"\nChecking relevance of generated N-gram GCN embeddings at: {prot_emb_path}")
        with h5py.File(prot_emb_path, 'r') as hf:
            relevant_ids_found = {pid for pid in all_proteins_in_pairs if pid in hf}

        if relevant_ids_found:
            print(f"Found {len(relevant_ids_found)} relevant proteins in the N-gram GCN embeddings. Proceeding with all evaluations.")
            can_proceed_with_evaluation = True
        else:
            print("No overlap found between interaction proteins and generated N-gram GCN embeddings. Skipping all link prediction evaluations.")
    else:
        print("\nN-gram GCN generation was disabled. Proceeding with evaluation of pre-existing files.")
        can_proceed_with_evaluation = True

    # --- 4. Conditional Link Prediction Evaluation ---
    if can_proceed_with_evaluation:
        print("\n" + "=" * 20 + " Step 3: Link Prediction Evaluation " + "=" * 20)
        mlp_params = {'dense1_units': MLP_DENSE1_UNITS_LP, 'dropout1_rate': MLP_DROPOUT1_RATE_LP, 'dense2_units': MLP_DENSE2_UNITS_LP, 'dropout2_rate': MLP_DROPOUT2_RATE_LP, 'l2_reg': MLP_L2_REG_LP}
        lp_params = {'edge_emb_method': EDGE_EMBEDDING_METHOD_LP, 'n_folds': N_FOLDS_LP, 'random_state': RANDOM_STATE, 'batch_size': BATCH_SIZE_LP, 'epochs': EPOCHS_LP, 'learning_rate': LEARNING_RATE_LP}

        for emb_config in normal_embedding_files_to_evaluate:
            print(f"\n--- Evaluating file: {emb_config['name']} ---")
            try:
                with h5py.File(emb_config['path'], 'r') as hf:
                    protein_embeddings = {pid: hf[pid][:] for pid in all_proteins_in_pairs if pid in hf}
            except Exception as e:
                print(f"Could not load embeddings from {emb_config['path']}: {e}");
                continue

            if not protein_embeddings:
                print(f"No relevant embeddings for interaction pairs found in this file. Skipping.");
                continue

            main_workflow_cv_lp(embedding_name_cv=emb_config['name'], protein_embeddings_cv=protein_embeddings, positive_pairs_cv=positive_pairs, negative_pairs_cv=negative_pairs, mlp_params_dict_cv=mlp_params,
                lp_config=lp_params)

    print("\n--- Script Finished ---")
