# Combined Protein-Protein Interaction Evaluation and N-gramGCN Embedding Generation Script
import os
import sys
import shutil
import numpy as np
import pandas as pd
import time
import gc
import h5py
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

import tensorflow as tf
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any, Set, Tuple, Union, Callable
from collections import defaultdict, Counter
from Bio import SeqIO

# NEW Imports for community detection
import networkx as nx
import community as community_louvain  # For Louvain community detection (install with: pip install python-louvain)

# --- TensorFlow GPU Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"TensorFlow: GPU Devices Detected: {gpu_devices}")
else:
    print("TensorFlow: Warning: No GPU detected. Running on CPU.")

# --- General Configuration & Script Behavior ---
DEBUG_VERBOSE = True
RANDOM_STATE = 42
BASE_OUTPUT_DIR = "C:/ProgramData/ProtDiGCN/ppi_evaluation_results_combined_with_ngramgcn/"

# --- N-gramGCN Generation Configuration ---
RUN_AND_EVALUATE_NGRAM_GCN = True
NGRAM_GCN_INPUT_FASTA_PATH = "C:/ProgramData/ProtDiGCN/uniprot_sequences_sample.fasta"  # User specified, do not change
NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR = os.path.join(BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings")  #
NGRAM_GCN_MAX_N = 5  #
NGRAM_GCN_1GRAM_INIT_DIM = 64  # Dimension for each component of PE and for 1-gram features
NGRAM_GCN_HIDDEN_DIM_1 = 128  #
NGRAM_GCN_HIDDEN_DIM_2 = 64  #
NGRAM_GCN_PE_MAX_LEN = 10  #
NGRAM_GCN_DROPOUT = 0.5  #
NGRAM_GCN_LR = 0.0005  #
NGRAM_GCN_WEIGHT_DECAY = 1e-4  #
NGRAM_GCN_EPOCHS_PER_LEVEL = 1000  #
NGRAM_GCN_USE_VECTOR_COEFFS = True  #
NGRAM_GCN_GENERATED_EMB_NAME = "NgramGCN-Generated"  #

NGRAM_GCN_TASK_PER_LEVEL: Dict[int, str] = {1: 'community_label', }
NGRAM_GCN_DEFAULT_TASK_MODE = 'next_node'

# --- Link Prediction Evaluation Configuration ---
normal_positive_interactions_path = os.path.normpath('C:/ProgramData/ProtDiGCN/ground_truth/positive_interactions.csv')  #
normal_negative_interactions_path = os.path.normpath('C:/ProgramData/ProtDiGCN/ground_truth/negative_interactions.csv')  #
normal_sample_negative_pairs: Optional[int] = 20000  #
normal_embedding_files_to_evaluate = [{"path": "C:/ProgramData/ProtDiGCN/models/per-protein.h5", "name": "ProtT5-Precomputed", "loader_func_key": "load_h5_embeddings_selectively"}, ]  #
normal_output_main_dir = os.path.join(BASE_OUTPUT_DIR, "normal_run_output_combined")  #
EDGE_EMBEDDING_METHOD_LP = 'concatenate'  #
N_FOLDS_LP = 2  #
MAX_TRAIN_SAMPLES_CV_LP = 100000  #
MAX_VAL_SAMPLES_CV_LP = 20000  #
MAX_SHUFFLE_BUFFER_SIZE_LP = 200000  #
PLOT_TRAINING_HISTORY_LP = True  #
MLP_DENSE1_UNITS_LP = 128  #
MLP_DROPOUT1_RATE_LP = 0.4  #
MLP_DENSE2_UNITS_LP = 64  #
MLP_DROPOUT2_RATE_LP = 0.4  #
MLP_L2_REG_LP = 0.001  #
BATCH_SIZE_LP = 64  #
EPOCHS_LP = 10  #
LEARNING_RATE_LP = 1e-3  #
K_VALUES_FOR_RANKING_METRICS_LP = [10, 50, 100, 200]  #
K_VALUES_FOR_TABLE_DISPLAY_LP = [50, 100]  #
MAIN_EMBEDDING_NAME_FOR_STATS_LP = NGRAM_GCN_GENERATED_EMB_NAME if RUN_AND_EVALUATE_NGRAM_GCN else "ProtT5_Example_Data"  #
STATISTICAL_TEST_METRIC_KEY_LP = 'test_auc_sklearn'  #
STATISTICAL_TEST_ALPHA_LP = 0.05  #


# ==============================================================================
# START: ID Parsing and N-gramGCN Code Integration
# ==============================================================================

def extract_canonical_id_and_type(header_or_id_line: str) -> tuple[Optional[str], Optional[str]]:  #
    hid = header_or_id_line.strip().lstrip('>')  #
    up_match = re.match(r"^(?:sp|tr)\|([A-Z0-9]{6,10}(?:-\d+)?)\|", hid, re.IGNORECASE)  #
    if up_match: return "UniProt", up_match.group(1)  #
    uniref_cluster_match = re.match(r"^(UniRef(?:100|90|50))_((?:[A-Z0-9]{6,10}(?:-\d+)?)(?:_[A-Z0-9]+)?|(UPI[A-F0-9]+))", hid, re.IGNORECASE)  #
    if uniref_cluster_match:  #
        cluster_type, id_part = uniref_cluster_match.group(1), uniref_cluster_match.group(2)  #
        if re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", id_part): return "UniProt (from UniRef)", id_part  #
        if "_" in id_part and re.fullmatch(r"[A-Z0-9]{6,10}_[A-Z0-9]+", id_part): return "UniProt (from UniRef)", id_part.split('_')[0]  #
        if id_part.startswith("UPI"): return "UniParc (from UniRef)", id_part  #
        return "UniRef Cluster", f"{cluster_type}_{id_part}"  #
    ncbi_gi_match = re.match(r"^gi\|\d+\|\w{1,3}\|([A-Z]{1,3}[_0-9]*\w*\.?\d*)\|", hid)  #
    if ncbi_gi_match: return "NCBI", ncbi_gi_match.group(1)  #
    ncbi_acc_match = re.match(r"^([A-Z]{2,3}(?:_|\d)[A-Z0-9]+\.?\d*)\b", hid)  #
    if ncbi_acc_match: return "NCBI", ncbi_acc_match.group(1)  #
    pdb_match = re.match(r"^([0-9][A-Z0-9]{3})[_ ]?([A-Z0-9]{1,2})?", hid, re.IGNORECASE)  #
    if pdb_match:  #
        pdb_id, chain_part = pdb_match.group(1).upper(), pdb_match.group(2).upper() if pdb_match.group(2) else ""  #
        is_likely_uniprot = len(pdb_id) >= 5 and pdb_id[0] in 'OPQ' and pdb_id[1].isdigit()  #
        if not is_likely_uniprot: return "PDB", f"{pdb_id}{'_' + chain_part if chain_part else ''}"  #
    plain_up_match = re.fullmatch(r"([A-Z0-9]{6,10}(?:-\d+)?)", hid.split()[0].split('|')[0])  #
    if plain_up_match: return "UniProt (assumed)", plain_up_match.group(1)  #
    first_word = hid.split()[0].split('|')[0]  #
    if first_word: return "Unknown", first_word  #
    return "Unknown", hid  #


def parse_fasta_sequences_with_ids_ngram(filepath: str) -> list[tuple[str, str]]:  #
    protein_data = []  #
    if not os.path.exists(filepath):  #
        if DEBUG_VERBOSE: print(f"NgramGCN: FASTA file not found at {filepath}")  #
        return protein_data  #
    try:  #
        for record in SeqIO.parse(filepath, "fasta"):  #
            _, canonical_id = extract_canonical_id_and_type(record.id)  #
            if canonical_id:  #
                protein_data.append((canonical_id, str(record.seq).upper()))  #
            else:  #
                if DEBUG_VERBOSE: print(f"NgramGCN: Could not extract canonical ID from '{record.id[:50]}...', using full ID as fallback: {record.id}")  #
                protein_data.append((record.id, str(record.seq).upper()))  #
        if DEBUG_VERBOSE: print(f"NgramGCN: Parsed {len(protein_data)} sequences with extracted/standardized IDs from {filepath}")  #
    except Exception as e:  #
        print(f"NgramGCN: Error parsing FASTA file {filepath}: {e}")  #
    return protein_data  #


def get_ngrams_and_transitions_ngram(sequences: list[str], n: int):  #
    all_ngrams, all_transitions = [], []  #
    for seq in sequences:  #
        if len(seq) < n: continue  #
        current_seq_ngrams = [tuple(seq[i: i + n]) for i in range(len(seq) - n + 1)]  #
        all_ngrams.extend(current_seq_ngrams)  #
        for i in range(len(current_seq_ngrams) - 1): all_transitions.append((current_seq_ngrams[i], current_seq_ngrams[i + 1]))  #
    return all_ngrams, all_transitions  #


def build_ngram_graph_data_ngram(ngrams: list[tuple], transitions: list[tuple[tuple, tuple]], node_prob_from_prev_graph: Optional[dict[tuple, float]] = None, n_val: int = 1) -> tuple[  #
    Optional[Data], dict[tuple, int], dict[int, tuple], dict[tuple, float]]:  #
    if not ngrams: return None, {}, {}, {}  #
    unique_ngrams_list = sorted(list(set(ngrams)))  #
    ngram_to_idx = {ngram: i for i, ngram in enumerate(unique_ngrams_list)}  #
    idx_to_ngram = {i: ngram for ngram, i in ngram_to_idx.items()}  #
    num_nodes = len(unique_ngrams_list)  #
    ngram_counts = Counter(ngrams)  #
    ngram_probabilities = {ngram: count / len(ngrams) if len(ngrams) > 0 else 0 for ngram, count in ngram_counts.items()}  #
    source_nodes_idx, target_nodes_idx, edge_weights = [], [], []  #
    edge_counts = Counter(transitions)  #
    source_ngram_outgoing_counts = defaultdict(int)  #
    for (src_ng, _), count in edge_counts.items(): source_ngram_outgoing_counts[src_ng] += count  #
    for (source_ngram, target_ngram), count in edge_counts.items():  #
        if source_ngram in ngram_to_idx and target_ngram in ngram_to_idx:  #
            source_idx, target_idx = ngram_to_idx[source_ngram], ngram_to_idx[target_ngram]  #
            transition_prob = count / source_ngram_outgoing_counts[source_ngram] if source_ngram_outgoing_counts[source_ngram] > 0 else 0.0  #
            if transition_prob > 1e-9:  #
                source_nodes_idx.append(source_idx);  #
                target_nodes_idx.append(target_idx);  #
                edge_weights.append(transition_prob)  #
    data = Data();  #
    data.num_nodes = num_nodes  #
    if source_nodes_idx:  #
        edge_index = torch.tensor([source_nodes_idx, target_nodes_idx], dtype=torch.long)  #
        edge_attr_squeezed = torch.tensor(edge_weights, dtype=torch.float)  #
        data.edge_index_out = edge_index;  #
        data.edge_weight_out = edge_attr_squeezed  #
        data.edge_index_in = edge_index.flip(dims=[0]) if edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long)  #
        data.edge_weight_in = edge_attr_squeezed  #
    else:  #
        if DEBUG_VERBOSE and n_val > 0 and num_nodes > 0: print(f"NgramGCN: Warning: No edges created for n={n_val} graph (num_nodes: {num_nodes}).")  #
        data.edge_index_out = torch.empty((2, 0), dtype=torch.long);  #
        data.edge_weight_out = torch.empty(0, dtype=torch.float)  #
        data.edge_index_in = torch.empty((2, 0), dtype=torch.long);  #
        data.edge_weight_in = torch.empty(0, dtype=torch.float)  #
        if num_nodes == 0: return None, ngram_to_idx, idx_to_ngram, ngram_probabilities  #
    training_samples = []  # For 'next_node' task #
    if hasattr(data, 'edge_index_out') and data.edge_index_out is not None and data.edge_index_out.numel() > 0:  #
        for i in range(data.edge_index_out.size(1)): training_samples.append((data.edge_index_out[0, i].item(), data.edge_index_out[1, i].item()))  #
    data.training_samples = training_samples  #
    return data, ngram_to_idx, idx_to_ngram, ngram_probabilities  #


class CustomDiGCNLayerPyG_ngram(MessagePassing):  #
    def __init__(self, in_channels: int, out_channels: int, num_nodes_for_coeffs: int, use_vector_coeffs: bool = True):  #
        super(CustomDiGCNLayerPyG_ngram, self).__init__(aggr='add')  #
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)  #
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)  #
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)  #
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))  #
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))  #
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels))  #
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels))  #
        self.use_vector_coeffs = use_vector_coeffs  #
        self.num_nodes_for_coeffs_init = num_nodes_for_coeffs  #
        actual_vec_size = max(1, num_nodes_for_coeffs)  #
        if self.use_vector_coeffs:  #
            self.C_in_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1))  #
            self.C_out_vec = nn.Parameter(torch.Tensor(actual_vec_size, 1))  #
        else:  #
            self.C_in = nn.Parameter(torch.Tensor(1));  #
            self.C_out = nn.Parameter(torch.Tensor(1))  #
        self.reset_parameters()  #

    def reset_parameters(self):  #
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_skip]: nn.init.xavier_uniform_(lin.weight)  #
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_skip_in, self.bias_skip_out]: nn.init.zeros_(bias)  #
        if self.use_vector_coeffs:  #
            nn.init.ones_(self.C_in_vec);  #
            nn.init.ones_(self.C_out_vec)  #
        else:  #
            nn.init.ones_(self.C_in);  #
            nn.init.ones_(self.C_out)  #

    def forward(self, x: torch.Tensor, edge_index_in: torch.Tensor, edge_weight_in: torch.Tensor, edge_index_out: torch.Tensor, edge_weight_out: torch.Tensor):  #
        x_transformed_main_in = self.lin_main_in(x);  #
        aggr_main_in = self.propagate(edge_index_in, x=x_transformed_main_in, edge_weight=edge_weight_in)  #
        term1_in = aggr_main_in + self.bias_main_in;  #
        x_transformed_skip_in = self.lin_skip(x)  #
        aggr_skip_in = self.propagate(edge_index_in, x=x_transformed_skip_in, edge_weight=edge_weight_in)  #
        term2_in = aggr_skip_in + self.bias_skip_in;  #
        ic_combined = term1_in + term2_in  #
        x_transformed_main_out = self.lin_main_out(x);  #
        aggr_main_out = self.propagate(edge_index_out, x=x_transformed_main_out, edge_weight=edge_weight_out)  #
        term1_out = aggr_main_out + self.bias_main_out;  #
        x_transformed_skip_out = self.lin_skip(x)  #
        aggr_skip_out = self.propagate(edge_index_out, x=x_transformed_skip_out, edge_weight=edge_weight_out)  #
        term2_out = aggr_skip_out + self.bias_skip_out;  #
        oc_combined = term1_out + term2_out  #
        if self.use_vector_coeffs:  #
            if self.C_in_vec.size(0) != x.size(0):  #
                if DEBUG_VERBOSE and self.C_in_vec.size(0) > 0 and self.C_in_vec.size(0) != 1:  # Avoid warning if C_vec is scalar-like due to num_nodes_for_coeffs=0/1 #
                    print(f"NgramGCN Layer Warning: C_vec size ({self.C_in_vec.size(0)}) != x.size(0) ({x.size(0)}). "  #
                          f"Ensure num_nodes_for_coeffs ({self.num_nodes_for_coeffs_init}) passed at init matches current graph's data.num_nodes.")  #
            output = self.C_in_vec.to(x.device) * ic_combined + self.C_out_vec.to(x.device) * oc_combined  #
        else:  #
            output = self.C_in.to(x.device) * ic_combined + self.C_out.to(x.device) * oc_combined  #
        return output  #

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:  #
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None and edge_weight.numel() > 0 else x_j  #


class ProtDiGCNEncoderDecoder_ngram(nn.Module):  #
    def __init__(self, num_initial_features: int, hidden_dim1: int, hidden_dim2: int, num_graph_nodes_for_gnn_coeffs: int,  # For CustomDiGCNLayerPyG_ngram C_vec sizing #
                 task_num_output_classes: int,  # For decoder output (num_nodes or num_communities) #
                 n_gram_length_for_pe: int,  # Current n_val (length of n-gram, e.g., 1, 2, 3...) #
                 one_gram_embed_dim_for_pe: int,  # Dimension of each positional vector #
                 max_allowable_len_for_pe_layer: int,  # Size of nn.Embedding for PEs (now graph_data.num_nodes) #
                 dropout_rate: float, use_vector_coeffs_in_gnn: bool = True):  #
        super().__init__()  #
        self.n_gram_length_for_pe = n_gram_length_for_pe  #
        self.one_gram_embed_dim_for_pe = one_gram_embed_dim_for_pe  #
        self.dropout_rate = dropout_rate  #
        self.l2_norm_eps = 1e-12  #

        self.learnable_pe_active = False  #
        if self.one_gram_embed_dim_for_pe > 0 and max_allowable_len_for_pe_layer > 0:  #
            self.positional_encoder_layer = nn.Embedding(max_allowable_len_for_pe_layer, self.one_gram_embed_dim_for_pe)  #
            self.learnable_pe_active = True  #
            if DEBUG_VERBOSE: print(f"NgramGCN Model: Initialized Learnable PE Layer with size: ({max_allowable_len_for_pe_layer}, {self.one_gram_embed_dim_for_pe}) for n_val={n_gram_length_for_pe}")  #
        else:  #
            self.positional_encoder_layer = None  #
            if DEBUG_VERBOSE: print(
                f"NgramGCN Model: Learnable PE Layer NOT initialized for n_val={n_gram_length_for_pe} (one_gram_embed_dim={one_gram_embed_dim_for_pe}, pe_layer_size={max_allowable_len_for_pe_layer})")  #

        self.conv1 = CustomDiGCNLayerPyG_ngram(num_initial_features, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)  #
        self.conv2 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)  #
        self.conv3 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim2, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)  #

        if num_initial_features == hidden_dim1:
            self.residual_proj_1 = nn.Identity()  #
        else:
            self.residual_proj_1 = nn.Linear(num_initial_features, hidden_dim1)  #
        if hidden_dim1 == hidden_dim2:
            self.residual_proj_3 = nn.Identity()  #
        else:
            self.residual_proj_3 = nn.Linear(hidden_dim1, hidden_dim2)  #

        self.decoder_fc = nn.Linear(hidden_dim2, task_num_output_classes)  #
        self.final_normalized_embedding_output = None  #

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:  #
        if not self.learnable_pe_active or self.positional_encoder_layer is None or self.n_gram_length_for_pe == 0 or self.one_gram_embed_dim_for_pe == 0:  #
            return x  #

        current_device = x.device  #
        # Create a clone to ensure operations are not in-place on the original input tensor x #
        x_to_modify = x.clone()  #

        if self.n_gram_length_for_pe == 1:  # For 1-grams #
            if x_to_modify.shape[1] == self.one_gram_embed_dim_for_pe:  #
                if self.positional_encoder_layer.num_embeddings > 0:  # Check if PE layer is valid #
                    position_idx = torch.tensor([0], device=current_device, dtype=torch.long)  #
                    try:  #
                        pe_for_pos_0 = self.positional_encoder_layer(position_idx).squeeze(0)  #
                        x_to_modify = x_to_modify + pe_for_pos_0.unsqueeze(0)  # Not in-place for x_to_modify's data #
                    except IndexError:  #
                        if DEBUG_VERBOSE: print(f"NgramGCN PE Warning: Index [0] out of bounds for PE layer (size {self.positional_encoder_layer.num_embeddings}, n_val=1).")  #

        elif self.n_gram_length_for_pe > 1:  # For n-grams with n > 1 #
            expected_feature_dim = self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe  #
            if x_to_modify.shape[1] == expected_feature_dim:  #
                x_reshaped_original = x_to_modify.view(-1, self.n_gram_length_for_pe, self.one_gram_embed_dim_for_pe)  #
                # Create a clone of the reshaped view if we are going to modify slices, to avoid modifying x_to_modify potentially shared view memory directly before reassignment
                x_reshaped_for_modification = x_reshaped_original.clone()

                num_positions_in_ngram = self.n_gram_length_for_pe  #
                num_positions_to_encode = min(num_positions_in_ngram, self.positional_encoder_layer.num_embeddings)  #

                if num_positions_in_ngram > self.positional_encoder_layer.num_embeddings:
                    if DEBUG_VERBOSE: print(f"NgramGCN PE Warning: n-gram length {num_positions_in_ngram} > PE layer table size {self.positional_encoder_layer.num_embeddings}. PE will be partial based on PE layer size.")

                if num_positions_to_encode > 0:  #
                    position_indices = torch.arange(0, num_positions_to_encode, device=current_device, dtype=torch.long)  #
                    try:  #
                        pe_to_add = self.positional_encoder_layer(position_indices)  # Shape: (num_positions_to_encode, one_gram_embed_dim_for_pe) #

                        # Apply PE out-of-place to the relevant part of the cloned reshaped tensor
                        modified_slice = x_reshaped_for_modification[:, :num_positions_to_encode, :] + pe_to_add.unsqueeze(0)
                        x_reshaped_for_modification[:, :num_positions_to_encode, :] = modified_slice  # Assign back to the slice of the clone

                        x_to_modify = x_reshaped_for_modification.view(-1, expected_feature_dim)  # Update x_to_modify with the changes

                    except IndexError:  #
                        if DEBUG_VERBOSE: print(  #
                            f"NgramGCN PE Warning: Index out of bounds for PE lookup (n_val={self.n_gram_length_for_pe}). Max idx: {num_positions_to_encode - 1}, PE layer: {self.positional_encoder_layer.num_embeddings}")  #  # else: shape mismatch, x_to_modify (clone of x) is returned as is. #

        return x_to_modify  #

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:  #
        current_device = data.x.device if hasattr(data, 'x') and data.x is not None else (data.edge_index_out.device if hasattr(data, 'edge_index_out') and data.edge_index_out is not None else 'cpu')  #
        x_on_device = data.x.to(current_device) if hasattr(data, 'x') and data.x is not None else torch.empty((data.num_nodes, 0), device=current_device)  #
        if x_on_device.numel() == 0 and data.num_nodes > 0 and hasattr(self.conv1, 'lin_main_in') and self.conv1.lin_main_in.in_features > 0:  #
            if 'DEBUG_VERBOSE' in globals() and DEBUG_VERBOSE: print(f"NgramGCN Model: Warning - data.x is empty for {data.num_nodes} nodes. Using random features of dim {self.conv1.lin_main_in.in_features}.")  #
            x_on_device = torch.randn(data.num_nodes, self.conv1.lin_main_in.in_features, device=current_device)  #
        x_pe = self._apply_positional_encoding(x_on_device)  #
        edge_index_in = data.edge_index_in.to(current_device) if hasattr(data, 'edge_index_in') and data.edge_index_in is not None else torch.empty((2, 0), dtype=torch.long, device=current_device)  #
        edge_index_out = data.edge_index_out.to(current_device) if hasattr(data, 'edge_index_out') and data.edge_index_out is not None else torch.empty((2, 0), dtype=torch.long, device=current_device)  #
        edge_weight_in = data.edge_weight_in.to(current_device) if hasattr(data, 'edge_weight_in') and data.edge_weight_in is not None and data.edge_weight_in.numel() > 0 else torch.ones(edge_index_in.size(1),
                                                                                                                                                                                           device=current_device)  #
        edge_weight_out = data.edge_weight_out.to(current_device) if hasattr(data, 'edge_weight_out') and data.edge_weight_out is not None and data.edge_weight_out.numel() > 0 else torch.ones(edge_index_out.size(1),
                                                                                                                                                                                                device=current_device)  #
        if x_pe.numel() == 0 and data.num_nodes > 0:  #
            print(f"NgramGCN Model Error: Input features 'x' are empty for graph with {data.num_nodes} nodes after PE. Cannot proceed.")  #
            empty_logits = torch.empty((data.num_nodes, self.decoder_fc.out_features), device=current_device)  #
            empty_embeds_dim = self.conv3.lin_main_in.out_features if hasattr(self.conv3, 'lin_main_in') else self.decoder_fc.in_features  #
            empty_embeds = torch.empty((data.num_nodes, empty_embeds_dim), device=current_device)  #
            return empty_logits, empty_embeds  #
        h1_conv_out = self.conv1(x_pe, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)  #
        x_proj1 = self.residual_proj_1(x_pe);  #
        h1_res_sum = x_proj1 + h1_conv_out  #
        h1_activated = F.tanh(h1_res_sum);  #
        h1 = F.dropout(h1_activated, p=self.dropout_rate, training=self.training)  #
        h2_conv_out = self.conv2(h1, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)  #
        h2_res_sum = h1 + h2_conv_out;  #
        h2_activated = F.tanh(h2_res_sum)  #
        h2 = F.dropout(h2_activated, p=self.dropout_rate, training=self.training)  #
        h3_conv_out = self.conv3(h2, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)  #
        h2_proj3 = self.residual_proj_3(h2);  #
        h3_res_sum = h2_proj3 + h3_conv_out  #
        final_gcn_activated_output = F.tanh(h3_res_sum)  #
        h_embed_for_decoder_dropped = F.dropout(final_gcn_activated_output, p=self.dropout_rate, training=self.training)  #
        task_logits = self.decoder_fc(h_embed_for_decoder_dropped)  #
        norm = torch.norm(final_gcn_activated_output, p=2, dim=1, keepdim=True)  #
        self.final_normalized_embedding_output = final_gcn_activated_output / (norm + self.l2_norm_eps)  #
        return F.log_softmax(task_logits, dim=-1), self.final_normalized_embedding_output  #


def detect_communities_louvain(edge_index: torch.Tensor, num_nodes: int, random_state_louvain: Optional[int] = None) -> Tuple[Optional[torch.Tensor], int]:  #
    if num_nodes == 0:  #
        if DEBUG_VERBOSE: print("NgramGCN Community: num_nodes is 0, cannot detect communities.")  #
        return None, 0  #
    if edge_index.numel() == 0:  #
        if DEBUG_VERBOSE: print("NgramGCN Community: No edges in graph. Assigning each node to its own community.")  #
        labels_tensor = torch.arange(num_nodes, dtype=torch.long)  #
        return labels_tensor, num_nodes  #
    adj = defaultdict(set)  #
    edges = edge_index.cpu().numpy().T  #
    for u, v in edges: adj[u].add(v); adj[v].add(u)  #
    nx_graph = nx.Graph();  #
    nx_graph.add_nodes_from(range(num_nodes))  #
    for u, neighbors in adj.items():  #
        for v in neighbors: nx_graph.add_edge(u, v)  #
    if nx_graph.number_of_nodes() == 0:  #
        if DEBUG_VERBOSE: print("NgramGCN Community: NetworkX graph has no nodes after conversion."); return None, 0  #
    if nx_graph.number_of_edges() == 0 and nx_graph.number_of_nodes() > 0:  #
        if DEBUG_VERBOSE: print("NgramGCN Community: NetworkX graph has nodes but no edges. Assigning each node its own community.")  #
        labels_tensor = torch.arange(num_nodes, dtype=torch.long);  #
        return labels_tensor, num_nodes  #
    try:  #
        partition = community_louvain.best_partition(nx_graph, random_state=random_state_louvain)  #
        if not partition:  # Handle empty partition if Louvain fails unexpectedly #
            if DEBUG_VERBOSE: print("NgramGCN Community: Louvain returned empty partition. Assigning own communities.")  #
            labels_tensor = torch.arange(num_nodes, dtype=torch.long);  #
            return labels_tensor, num_nodes  #
        community_labels_list = [-1] * num_nodes  # Default for nodes not in partition (e.g. isolated) #
        for node_idx_from_partition, community_id in partition.items():  #
            if 0 <= node_idx_from_partition < num_nodes: community_labels_list[node_idx_from_partition] = community_id  #
        max_existing_comm_id = -1  #
        if partition: max_existing_comm_id = max(partition.values()) if partition.values() else -1  # Max community ID from partition #
        current_new_comm_id = max_existing_comm_id + 1  #
        for i in range(num_nodes):  # Assign labels to any nodes missed by partition (e.g. isolated) #
            if community_labels_list[i] == -1: community_labels_list[i] = current_new_comm_id; current_new_comm_id += 1  #
        community_labels = torch.tensor(community_labels_list, dtype=torch.long)  #
        num_communities = len(torch.unique(community_labels))  # Recalculate based on the full list #
        if DEBUG_VERBOSE: print(f"NgramGCN Community: Detected {num_communities} communities for {num_nodes} nodes.")  #
        if num_communities == 1 and num_nodes > 1:  #
            if DEBUG_VERBOSE: print(f"NgramGCN Community: Info - Louvain process resulted in a single community for {num_nodes} nodes.")  #
        return community_labels, num_communities  #
    except Exception as e:  #
        print(f"NgramGCN Community: Error during Louvain: {e}. Assigning own communities as fallback.")  #
        labels_tensor = torch.arange(num_nodes, dtype=torch.long);  #
        return labels_tensor, num_nodes  #


def train_ngram_model_ngram(model: ProtDiGCNEncoderDecoder_ngram, data: Data, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str):  #
    model.train();  #
    model.to(device);  #
    data = data.to(device)  #
    criterion = nn.NLLLoss()  #
    if task_mode == 'next_node':  #
        if not hasattr(data, 'training_samples') or not data.training_samples:  #
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No 'training_samples'. Cannot train."); return  #
        source_indices = torch.tensor([s for s, t in data.training_samples], dtype=torch.long).to(device)  #
        target_indices = torch.tensor([t for s, t in data.training_samples], dtype=torch.long).to(device)  #
        if source_indices.numel() == 0:  #
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Source indices empty. Cannot train."); return  #
    elif task_mode == 'community_label':  #
        if not hasattr(data, 'y_task_labels') or data.y_task_labels is None:  #
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No 'y_task_labels'. Cannot train."); return  #
        target_community_labels = data.y_task_labels.to(device).long()  #
        if data.num_nodes == 0:  #
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No nodes for training. Cannot train."); return  #
    else:  #
        print(f"NgramGCN: Unknown task_mode: {task_mode}. Cannot train.");  #
        return  #

    for epoch in range(1, epochs + 1):  #
        optimizer.zero_grad()  #
        log_probs, _ = model(data)  #
        if log_probs.numel() == 0:  #
            if DEBUG_VERBOSE: print(f"NgramGCN: Epoch {epoch:03d}, Task {task_mode}: Empty logits. Skip loss."); continue  #
        loss = None  #
        try:  #
            if task_mode == 'next_node':  #
                if source_indices.max() >= log_probs.size(0): print(f"NgramGCN ({task_mode}) Error: Max source idx {source_indices.max()} OOB log_probs[0] ({log_probs.size(0)})"); return  #
                if target_indices.max() >= log_probs.size(1) or target_indices.min() < 0: print(  #
                    f"NgramGCN ({task_mode}) Error: Target idx OOB. Max: {target_indices.max()}, Min: {target_indices.min()} vs Decoder Classes: {log_probs.size(1)}"); return  #
                loss = criterion(log_probs[source_indices], target_indices)  #
            elif task_mode == 'community_label':  #
                if log_probs.size(0) != data.num_nodes: print(f"NgramGCN ({task_mode}) Error: log_probs num_nodes mismatch. Expected {data.num_nodes}, Got {log_probs.size(0)}"); return  #
                if target_community_labels.max() >= log_probs.size(1) or target_community_labels.min() < 0:  #
                    print(
                        f"NgramGCN ({task_mode}) Error: Target community label OOB. Max: {target_community_labels.max()}, Min: {target_community_labels.min()} vs Num Output Classes (Communities): {log_probs.size(1)}");  #
                    return  #
                loss = criterion(log_probs, target_community_labels)  # Predict community for all nodes #
        except IndexError as e:  #
            print(f"NgramGCN ({task_mode}) Error: IndexError in loss. Epoch {epoch}. {e}");  #
            return  # Stop training #
        if loss is not None:  #
            loss.backward();  #
            optimizer.step()  #
            if epoch % (max(1, epochs // 10 if epochs >= 10 else 1)) == 0 or epoch == epochs:  #
                if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Loss: {loss.item():.4f}")  #
        else:  #
            if DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Epoch {epoch:03d}, Loss None."); break  #


def extract_node_embeddings_ngram(model: ProtDiGCNEncoderDecoder_ngram, data: Data, device: torch.device) -> Optional[np.ndarray]:  #
    model.eval();  #
    model.to(device);  #
    data = data.to(device)  #
    if not hasattr(data, 'x') or data.x is None or (hasattr(data, 'num_nodes') and data.num_nodes > 0 and data.x.numel() == 0):  #
        if DEBUG_VERBOSE: print("NgramGCN Extract: data.x missing/empty for non-empty graph.")  #
        out_dim = model.decoder_fc.in_features  #
        return np.array([]).reshape(0, out_dim) if out_dim > 0 else np.array([])  #
    with torch.no_grad():  #
        _, embeddings = model(data)  #
        return embeddings.cpu().numpy() if embeddings is not None and embeddings.numel() > 0 else None  #


def generate_and_save_ngram_embeddings(fasta_filepath: str, protein_ids_to_generate_for: Set[str], max_n_for_ngram: int, output_dir_for_this_run: str, one_gram_init_embed_dim: int, hidden_dim1: int, hidden_dim2: int,  #
                                       pe_max_len_config: int,  # This is the original NGRAM_GCN_PE_MAX_LEN from global config. It's passed but not directly used for PE layer sizing anymore.
                                       dropout: float, lr: float, epochs_per_level: int, use_vector_gnn_coeffs: bool, ngram_l2_reg: float, task_modes_per_level: Dict[int, str], default_task_mode: str) -> Tuple[  #
    Optional[str], Optional[str]]:  #
    if DEBUG_VERBOSE: print("NgramGCN: Starting N-gram Embedding Generation Workflow...")  #
    os.makedirs(output_dir_for_this_run, exist_ok=True)  #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
    if DEBUG_VERBOSE: print(f"NgramGCN: Using device: {device}")  #
    all_protein_data_from_fasta = parse_fasta_sequences_with_ids_ngram(fasta_filepath)  #
    if not all_protein_data_from_fasta: print("NgramGCN: No sequences parsed. Cannot generate."); return None, None  #
    sequences_to_process_map = {pid: seq for pid, seq in all_protein_data_from_fasta if pid in protein_ids_to_generate_for}  #
    if not sequences_to_process_map: print(f"NgramGCN: No sequences match 'protein_ids_to_generate_for'. Cannot generate."); return None, None  #
    sequences_list_for_ngram = list(sequences_to_process_map.values())  #
    protein_ids_for_final_pooling = list(sequences_to_process_map.keys())  #
    if DEBUG_VERBOSE: print(f"NgramGCN: Will process {len(sequences_list_for_ngram)} sequences.")  #
    level_embeddings: dict[int, np.ndarray] = {};  #
    level_ngram_to_idx: dict[int, dict[tuple, int]] = {};  #
    level_idx_to_ngram: dict[int, dict[int, tuple]] = {}  #
    per_residue_emb_path = os.path.join(output_dir_for_this_run, "per_residue_embeddings_from_1gram.h5")  #
    per_protein_emb_path = os.path.join(output_dir_for_this_run, f"per_protein_embeddings_from_{max_n_for_ngram}gram.h5")  #

    for n_val in range(1, max_n_for_ngram + 1):  #
        if DEBUG_VERBOSE: print(f"\nNgramGCN: --- Processing N-gram Level: n = {n_val} ---")  #
        current_ngrams, current_transitions = get_ngrams_and_transitions_ngram(sequences_list_for_ngram, n_val)  #
        if not current_ngrams: print(f"NgramGCN: No {n_val}-grams. Stopping."); level_embeddings[n_val] = np.array([]); break  #
        graph_data, ngram_to_idx_map, idx_to_ngram_map, _ = build_ngram_graph_data_ngram(current_ngrams, current_transitions, None, n_val)  #
        if graph_data is None or graph_data.num_nodes == 0: print(f"NgramGCN: No graph for n={n_val}. Stopping."); level_embeddings[n_val] = np.array([]); break  #
        level_ngram_to_idx[n_val] = ngram_to_idx_map;  #
        level_idx_to_ngram[n_val] = idx_to_ngram_map  #
        num_out_edges = graph_data.edge_index_out.size(1) if hasattr(graph_data, 'edge_index_out') and graph_data.edge_index_out is not None else 0  #
        if DEBUG_VERBOSE: print(f"NgramGCN: Built graph for n={n_val}: {graph_data.num_nodes} nodes, {num_out_edges} out-edges.")  #
        current_task_mode = task_modes_per_level.get(n_val, default_task_mode)  #
        if DEBUG_VERBOSE: print(f"NgramGCN: Task mode for n={n_val} is '{current_task_mode}'.")  #
        actual_num_output_classes_for_model = 0  #
        if current_task_mode == 'community_label':  #
            community_labels, num_communities = detect_communities_louvain(graph_data.edge_index_out, graph_data.num_nodes, random_state_louvain=RANDOM_STATE)  #
            if community_labels is not None and num_communities > 0:  #
                graph_data.y_task_labels = community_labels;  #
                actual_num_output_classes_for_model = num_communities  #
                if num_communities <= 1 and graph_data.num_nodes > 1: print(f"NgramGCN: Warn: Community detection for n={n_val} -> {num_communities} comm (nodes: {graph_data.num_nodes}).")  #
            else:  #
                print(f"NgramGCN: Warn: Community detection failed for n={n_val}. Reverting to '{default_task_mode}'.");  #
                current_task_mode = default_task_mode  #
        if current_task_mode == 'next_node': actual_num_output_classes_for_model = graph_data.num_nodes  #
        if actual_num_output_classes_for_model == 0 and graph_data.num_nodes > 0:  #
            print(f"NgramGCN: Critical: 0 output classes for task '{current_task_mode}' n={n_val}. Defaulting to num_nodes & next_node task.")  #
            actual_num_output_classes_for_model = graph_data.num_nodes;  #
            current_task_mode = 'next_node'  #
            if DEBUG_VERBOSE: print(f"NgramGCN: Task mode for n={n_val} forced to '{current_task_mode}'.")  #
        current_feature_dim: int  #
        if n_val == 1:  #
            current_feature_dim = one_gram_init_embed_dim;  #
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)  #
            if DEBUG_VERBOSE: print(f"NgramGCN: Initialized 1-gram features randomly (dim: {current_feature_dim}).")  #
        else:  #
            one_gram_embeds_arr = level_embeddings.get(1);  #
            one_gram_map = level_ngram_to_idx.get(1)  #
            if one_gram_embeds_arr is None or one_gram_map is None or one_gram_embeds_arr.size == 0: print(f"NgramGCN: Error: 1-gram base data missing for n={n_val}. Stopping."); level_embeddings[n_val] = np.array(  #
                []); break  #
            features_list = [];  #
            current_lvl_idx_map = level_idx_to_ngram[n_val];  #
            expected_concat_dim = n_val * one_gram_init_embed_dim  #
            for i in range(graph_data.num_nodes):  #
                ngram = current_lvl_idx_map.get(i);  #
                feature_parts = []  #
                if ngram:  #
                    for char_element in ngram:  #
                        char_key = (char_element,);  #
                        char_idx = one_gram_map.get(char_key)  #
                        if char_idx is not None and char_idx < len(one_gram_embeds_arr):
                            feature_parts.append(torch.from_numpy(one_gram_embeds_arr[char_idx].copy()).float())  #
                        else:
                            feature_parts.append(torch.zeros(one_gram_init_embed_dim))  #
                concat_feat = torch.cat(feature_parts) if feature_parts else torch.zeros(expected_concat_dim)  #
                if concat_feat.shape[0] != expected_concat_dim:  #
                    pad_needed = expected_concat_dim - concat_feat.shape[0]  #
                    if pad_needed > 0:
                        concat_feat = torch.cat((concat_feat, torch.zeros(pad_needed)))  #
                    else:
                        concat_feat = concat_feat[:expected_concat_dim]  #
                features_list.append(concat_feat)  #
            graph_data.x = torch.stack(features_list) if features_list else torch.empty((graph_data.num_nodes, expected_concat_dim))  #
            current_feature_dim = graph_data.x.shape[1] if graph_data.x.numel() > 0 else 0  #
            if DEBUG_VERBOSE: print(f"NgramGCN: Initialized n={n_val} features (dim: {current_feature_dim}).")  #

        if graph_data.num_nodes > 0 and current_feature_dim > 0 and actual_num_output_classes_for_model > 0:  #
            can_train = (current_task_mode == 'next_node' and hasattr(graph_data, 'training_samples') and graph_data.training_samples) or (
                        current_task_mode == 'community_label' and hasattr(graph_data, 'y_task_labels') and graph_data.y_task_labels is not None)  #

            model_pe_layer_size = graph_data.num_nodes  # PE layer sized by current graph's node count

            model = ProtDiGCNEncoderDecoder_ngram(  #
                num_initial_features=current_feature_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2,  #
                num_graph_nodes_for_gnn_coeffs=graph_data.num_nodes,  #
                task_num_output_classes=actual_num_output_classes_for_model,  #
                n_gram_length_for_pe=n_val,  #
                one_gram_embed_dim_for_pe=one_gram_init_embed_dim,  #
                max_allowable_len_for_pe_layer=model_pe_layer_size,  # THIS IS THE DYNAMIC SIZE
                dropout_rate=dropout, use_vector_coeffs_in_gnn=use_vector_gnn_coeffs)  #

            if DEBUG_VERBOSE: print(f"\nNgramGCN Model Architecture (for n={n_val}, task='{current_task_mode}', out_classes={actual_num_output_classes_for_model}, PE layer size={model_pe_layer_size}):"); print(  #
                model); print("-" * 50)  #

            if not can_train:  #
                if DEBUG_VERBOSE: print(f"NgramGCN: Cannot train model for n={n_val} (task='{current_task_mode}'). Extracting untrain_embeddings.")  #
                node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)  #
                level_embeddings[n_val] = node_embeddings_np if node_embeddings_np is not None and node_embeddings_np.size > 0 else np.array([])  #
                if n_val == 1 and current_task_mode == 'next_node' and num_out_edges == 0: print("NgramGCN: Critical: 1-gram 'next_node' task with no edges. Subsequent levels affected.");  # break #
            else:  #
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=ngram_l2_reg)  #
                if DEBUG_VERBOSE: print(f"NgramGCN: Training model for n={n_val}, task='{current_task_mode}'...")  #
                train_ngram_model_ngram(model, graph_data, optimizer, epochs_per_level, device, current_task_mode)  #
                node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)  #
                if node_embeddings_np is None or node_embeddings_np.size == 0: print(f"NgramGCN: Failed to extract embeds for n={n_val}. Stopping."); level_embeddings[n_val] = np.array([]); break  #
                level_embeddings[n_val] = node_embeddings_np  #
                if DEBUG_VERBOSE: print(f"NgramGCN: Extracted {node_embeddings_np.shape[0]} node embeds (dim {node_embeddings_np.shape[1]}) for n={n_val}.")  #

            if n_val == 1 and level_embeddings.get(1, np.array([])).size > 0:  #
                with h5py.File(per_residue_emb_path, 'w') as hf:  #
                    for idx, ngram_tuple in level_idx_to_ngram[1].items():  #
                        if idx < len(level_embeddings[1]): hf.create_dataset(ngram_tuple[0], data=level_embeddings[1][idx])  #
                print(f"NgramGCN: Saved per-residue (1-gram) embeddings to {per_residue_emb_path}")  #
        elif graph_data.num_nodes == 0:  #
            print(f"NgramGCN: No nodes for n={n_val}. Skip model.");  #
            level_embeddings[n_val] = np.array([])  #
        else:  #
            print(f"NgramGCN: Features/Classes zero for n={n_val}. Skip model.");  #
            level_embeddings[n_val] = np.array([])  #

    final_embeddings_to_pool_from = level_embeddings.get(max_n_for_ngram)  #
    final_idx_map_to_use = level_ngram_to_idx.get(max_n_for_ngram)  #
    protein_count_final = 0  #
    if final_embeddings_to_pool_from is None or final_embeddings_to_pool_from.size == 0 or final_idx_map_to_use is None:  #
        print(f"NgramGCN: Final n={max_n_for_ngram} embeds not available. Cannot gen per-protein embeds.")  #
    else:  #
        if DEBUG_VERBOSE: print(f"\nNgramGCN: --- Generating Per-Protein Embeddings (from n={max_n_for_ngram} graph) ---")  #
        with h5py.File(per_protein_emb_path, 'w') as hf_protein:  #
            for protein_id in tqdm(protein_ids_for_final_pooling, desc="NgramGCN: Pooling protein embeds", disable=not DEBUG_VERBOSE):  #
                protein_sequence = sequences_to_process_map.get(protein_id)  #
                if not protein_sequence or len(protein_sequence) < max_n_for_ngram: continue  #
                protein_specific_ngram_embeds = []  #
                for i in range(len(protein_sequence) - max_n_for_ngram + 1):  #
                    ngram_tuple = tuple(protein_sequence[i: i + max_n_for_ngram]);  #
                    ngram_idx = final_idx_map_to_use.get(ngram_tuple)  #
                    if ngram_idx is not None and ngram_idx < len(final_embeddings_to_pool_from): protein_specific_ngram_embeds.append(final_embeddings_to_pool_from[ngram_idx])  #
                if protein_specific_ngram_embeds:  #
                    valid_embeds = [emb.flatten() for emb in protein_specific_ngram_embeds if emb is not None and emb.ndim > 0 and emb.size > 0]  # ensure emb is not None #
                    if valid_embeds:  #
                        first_dim = valid_embeds[0].shape[0]  #
                        if all(e.shape[0] == first_dim for e in valid_embeds):  #
                            try:  #
                                hf_protein.create_dataset(protein_id, data=np.mean(np.stack(valid_embeds), axis=0));  #
                                protein_count_final += 1  #
                            except Exception as e_h5:  #
                                if DEBUG_VERBOSE:  # CORRECTED if AND INDENTATION
                                    print(f"NgramGCN: H5 Write Err for {protein_id}: {e_h5}")  #
            # CORRECTED INDENTATION for this summary print line
            print(f"NgramGCN: Saved {protein_count_final} per-protein embeddings to {per_protein_emb_path}")  #
    res_path_final = per_residue_emb_path if os.path.exists(per_residue_emb_path) and level_embeddings.get(1, np.array([])).size > 0 else None  #
    prot_path_final = per_protein_emb_path if os.path.exists(per_protein_emb_path) and protein_count_final > 0 else None  #
    return prot_path_final, res_path_final  #


# ============================================================================
# (Link Prediction Evaluator Code: ProteinFileOps, FileOps, Graph_LP, etc. and
#  main_workflow_cv_lp, run_evaluation_pipeline plotting/table functions remain unchanged from your script)
# ============================================================================
# ... (The existing Link Prediction code from your script is assumed to be here, unchanged) ...
class ProteinFileOps:  #
    @staticmethod  #
    def load_interaction_pairs(filepath: str, label: int, sample_n: Optional[int] = None, random_state_for_sampling: Optional[int] = None) -> List[Tuple[str, str, int]]:  #
        filepath = os.path.normpath(filepath)  #
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""  #
        if DEBUG_VERBOSE: print(f"LinkPred: Loading pairs from: {filepath} (label: {label}){sampling_info}...")  #
        if not os.path.exists(filepath): print(f"LinkPred: Warning: File not found: {filepath}"); return []  #
        try:  #
            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str)  #
            if sample_n is not None and 0 < sample_n < len(df):  #
                df = df.sample(n=sample_n, random_state=random_state_for_sampling)  #
            elif sample_n is not None and sample_n <= 0:  #
                df = df.iloc[0:0]  #

            pairs = []  #
            for _, r in df.iterrows():  #
                p1_raw = str(r.protein1).strip()  #
                p2_raw = str(r.protein2).strip()  #
                _, p1_acc = extract_canonical_id_and_type(p1_raw)  #
                _, p2_acc = extract_canonical_id_and_type(p2_raw)  #

                if p1_acc and p2_acc:  #
                    pairs.append((p1_acc, p2_acc, label))  #
                elif DEBUG_VERBOSE:  #
                    if p1_raw or p2_raw:  # Only warn if original strings were not empty #
                        print(f"LinkPred: Warning: Could not properly parse/standardize IDs for pair ('{p1_raw}', '{p2_raw}') in {os.path.basename(filepath)}. Skipping pair.")  #
            if DEBUG_VERBOSE: print(f"LinkPred: Loaded and standardized {len(pairs)} pairs from {os.path.basename(filepath)}.")  #
            return pairs  #
        except Exception as e:  #
            print(f"LinkPred: Error loading {filepath}: {e}");  #
            return []  #


class FileOps:  #
    @staticmethod  #
    def load_h5_embeddings_selectively(h5_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:  #
        h5_path = os.path.normpath(h5_path)  #
        if not os.path.exists(h5_path): print(f"LinkPred: Warning: Embedding file not found: {h5_path}"); return {}  #
        protein_embeddings: Dict[str, np.ndarray] = {}  #
        loaded_count = 0  #
        try:  #
            with h5py.File(h5_path, 'r') as hf:  #
                keys_in_file_raw = list(hf.keys())  #

                h5_key_to_standardized_id: Dict[str, str] = {}  #
                standardized_ids_in_h5_set: Set[str] = set()  #

                for raw_key in keys_in_file_raw:  #
                    _, std_id = extract_canonical_id_and_type(raw_key)  # Standardize H5 keys #
                    if std_id:  #
                        h5_key_to_standardized_id[raw_key] = std_id  #
                        standardized_ids_in_h5_set.add(std_id)  #

                keys_to_load_final_std = standardized_ids_in_h5_set  #
                if required_ids is not None:  #
                    keys_to_load_final_std = standardized_ids_in_h5_set.intersection(required_ids)  #

                if not keys_to_load_final_std and required_ids and standardized_ids_in_h5_set and DEBUG_VERBOSE:  #
                    print(f"  LinkPred: No standardized keys in {os.path.basename(h5_path)} match the provided required_ids set.")  #

                for raw_key in tqdm(keys_in_file_raw, desc=f"  LinkPred: Reading {os.path.basename(h5_path)}", leave=False, unit="protein", disable=not DEBUG_VERBOSE or not keys_in_file_raw):  #
                    standardized_id_for_key = h5_key_to_standardized_id.get(raw_key)  #
                    if not standardized_id_for_key: continue  #

                    if standardized_id_for_key in keys_to_load_final_std:  # Check against the desired set of *standardized* IDs #
                        if isinstance(hf[raw_key], h5py.Dataset):  #
                            try:  #
                                protein_embeddings[standardized_id_for_key] = hf[raw_key][:].astype(np.float32)  # Store with standardized ID #
                                loaded_count += 1  #
                            except Exception as eL:  #
                                if DEBUG_VERBOSE: print(f"LinkPred: Warn: Could not load data for key '{raw_key}' (std ID: {standardized_id_for_key}) from {os.path.basename(h5_path)}: {eL}")  #
            if DEBUG_VERBOSE: print(f"LinkPred: Loaded {loaded_count} embeddings (keyed by standardized ID) from {os.path.basename(h5_path)} (based on required IDs if provided).")  #
        except Exception as e:  #
            print(f"LinkPred: Error processing HDF5 {h5_path}: {e}");  #
            return {}  #
        return protein_embeddings  #

    @staticmethod  #
    def load_custom_embeddings(p_path: str, r_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:  # Matches user script #
        print(f"LinkPred: Placeholder load_custom_embeddings called for {p_path}. Returning empty dict.")  #
        return {}  #


class Graph_LP:  #
    def __init__(self):  #
        self.embedding_dim: Optional[int] = None  #

    def get_embedding_dimension(self, p_embs: Dict[str, np.ndarray]) -> int:  #
        if not p_embs: self.embedding_dim = 0; return 0  #
        for emb_v in p_embs.values():  #
            if emb_v is not None and hasattr(emb_v, 'shape') and len(emb_v.shape) > 0 and emb_v.shape[-1] > 0:  #
                self.embedding_dim = emb_v.shape[-1]  #
                if DEBUG_VERBOSE: print(f"LinkPred: Inferred embed dim: {self.embedding_dim}"); return self.embedding_dim  #
        self.embedding_dim = 0;  #
        print("LinkPred: Warn: Could not infer valid embed dim.");  #
        return 0  #

    def create_edge_embeddings(self, i_pairs: List[Tuple[str, str, int]], p_embs: Dict[str, np.ndarray], method: str = 'concatenate') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:  #
        if DEBUG_VERBOSE: print(f"LinkPred: Creating edge embeddings (method: {method})...")  #
        if not p_embs: print("LinkPred: Embeddings dict empty for edge creation."); return None, None  #
        if self.embedding_dim is None or self.embedding_dim == 0: self.get_embedding_dimension(p_embs)  #
        if self.embedding_dim == 0: print("LinkPred: Embed dim is 0. Cannot create edge features."); return None, None  #
        edge_feats, labels, skipped = [], [], 0  #
        for p1, p2, lbl in tqdm(i_pairs, desc="LinkPred: Edge Feats", leave=False, disable=not DEBUG_VERBOSE or not i_pairs):  #
            e1, e2 = p_embs.get(p1), p_embs.get(p2)  #
            if e1 is not None and e2 is not None:  #
                if e1.ndim > 1: e1 = e1.flatten();  #
                if e2.ndim > 1: e2 = e2.flatten()  #
                if e1.shape[0] != self.embedding_dim or e2.shape[0] != self.embedding_dim: skipped += 1; continue  #
                if method == 'concatenate':  #
                    feat = np.concatenate((e1, e2))  #
                elif method == 'average':  #
                    feat = (e1 + e2) / 2.0  #
                elif method == 'hadamard':  #
                    feat = e1 * e2  #
                elif method == 'subtract':  #
                    feat = np.abs(e1 - e2)  #
                else:  # Default to concatenate #
                    feat = np.concatenate((e1, e2))  #
                edge_feats.append(feat);  #
                labels.append(lbl)  #
            else:  #
                skipped += 1  #
        if skipped > 0 and DEBUG_VERBOSE: print(f"LinkPred: Skipped {skipped}/{len(i_pairs)} pairs for edge feats (missing embeds/dim mismatch).")  #
        if not edge_feats: print("LinkPred: No edge features created (all pairs might have been skipped or input empty)."); return None, None  #
        if DEBUG_VERBOSE and edge_feats: print(f"LinkPred: Created {len(edge_feats)} edge features (dim: {edge_feats[0].shape[0]}).")  #
        return np.array(edge_feats, dtype=np.float32), np.array(labels, dtype=np.int32)  #


def build_mlp_model_lp(input_shape: int, lr_lp: float, mlp_p_dict: Dict[str, Any]) -> tf.keras.Model:  #
    model = tf.keras.Sequential(  #
        [tf.keras.layers.InputLayer(input_shape=(input_shape,)), tf.keras.layers.Dense(mlp_p_dict['dense1_units'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(mlp_p_dict['l2_reg'])),  #
         tf.keras.layers.Dropout(mlp_p_dict['dropout1_rate']), tf.keras.layers.Dense(mlp_p_dict['dense2_units'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(mlp_p_dict['l2_reg'])),  #
         tf.keras.layers.Dropout(mlp_p_dict['dropout2_rate']), tf.keras.layers.Dense(1, activation='sigmoid')])  #
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_lp), loss=tf.keras.losses.BinaryCrossentropy(),  #
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc_keras'), tf.keras.metrics.Precision(name='precision_keras'), tf.keras.metrics.Recall(name='recall_keras')])  #
    return model  #


def plot_training_history(history_dict: Dict[str, Any], model_name: str, plots_output_dir: str, fold_num: Optional[int] = None):  #
    title_suffix = f" (Fold {fold_num})" if fold_num is not None else " (Representative Fold)"  #
    if not history_dict or not any(isinstance(val_list, list) and len(val_list) > 0 for val_list in history_dict.values()):  #
        if DEBUG_VERBOSE: print(f"LinkPred Plot: No history data for {model_name}{title_suffix}.")  #
        return  #
    os.makedirs(plots_output_dir, exist_ok=True)  #
    plot_filename = os.path.join(plots_output_dir, f"lp_history_{model_name.replace(' / ', '_').replace(':', '-')}{'_F' + str(fold_num) if fold_num else ''}.png")  #
    plt.figure(figsize=(12, 5))  #
    plotted_loss, plotted_acc = False, False  #
    if 'loss' in history_dict and history_dict['loss']:  #
        plt.subplot(1, 2, 1);  #
        plt.plot(history_dict['loss'], label='Train Loss', marker='.' if len(history_dict['loss']) < 15 else None);  #
        plotted_loss = True  #
    if 'val_loss' in history_dict and history_dict['val_loss']:  #
        if not plotted_loss: plt.subplot(1, 2, 1)  #
        plt.plot(history_dict['val_loss'], label='Val Loss', marker='.' if len(history_dict['val_loss']) < 15 else None);  #
        plotted_loss = True  #
    if plotted_loss: plt.title(f'LinkPred MLP Loss: {model_name}{title_suffix}'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)  #
    if 'accuracy' in history_dict and history_dict['accuracy']:  #
        plt.subplot(1, 2, 2);  #
        plt.plot(history_dict['accuracy'], label='Train Acc', marker='.' if len(history_dict['accuracy']) < 15 else None);  #
        plotted_acc = True  #
    if 'val_accuracy' in history_dict and history_dict['val_accuracy']:  #
        if not plotted_acc: plt.subplot(1, 2, 2)  #
        plt.plot(history_dict['val_accuracy'], label='Val Acc', marker='.' if len(history_dict['val_accuracy']) < 15 else None);  #
        plotted_acc = True  #
    if plotted_acc: plt.title(f'LinkPred MLP Acc: {model_name}{title_suffix}'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)  #
    if not (plotted_loss or plotted_acc): plt.close(); return  #
    plt.suptitle(f"LinkPred MLP History: {model_name}{title_suffix}", fontsize=16);  #
    plt.tight_layout(rect=[0, 0, 1, 0.95])  #
    try:  #
        plt.savefig(plot_filename);
        print(f"  LinkPred: Saved history plot: {plot_filename}")  #
    except Exception as e:  #
        print(f"  LinkPred: Error saving plot {plot_filename}: {e}")  #
    plt.close()  #


def plot_roc_curves(results_list: List[Dict[str, Any]], plots_output_dir: str):  #
    plt.figure(figsize=(10, 8));  #
    plotted_anything = False  #
    for result in results_list:  #
        roc_data = result.get('roc_data_representative')  #
        if roc_data:  #
            fpr, tpr, auc_val = roc_data  #
            avg_auc = result.get('test_auc_sklearn', auc_val if auc_val is not None else 0.0)  #
            if fpr is not None and tpr is not None and avg_auc is not None and hasattr(fpr, '__len__') and len(fpr) > 0 and hasattr(tpr, '__len__') and len(tpr) > 0:  #
                plt.plot(fpr, tpr, lw=2, label=f"{result.get('embedding_name', 'Unknown')} (Avg AUC={avg_auc:.3f})");  #
                plotted_anything = True  #
    if not plotted_anything:  #
        if DEBUG_VERBOSE: print("LinkPred Plot: No valid ROC data."); plt.close(); return  #
    os.makedirs(plots_output_dir, exist_ok=True)  #
    plot_filename = os.path.join(plots_output_dir, "lp_comparison_roc_curves.png")  #
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance');  #
    plt.xlim([0.0, 1.0]);  #
    plt.ylim([0.0, 1.05])  #
    plt.xlabel('FPR');  #
    plt.ylabel('TPR');  #
    plt.title('LinkPred ROC Curves');  #
    plt.legend(loc="lower right");  #
    plt.grid(True)  #
    try:  #
        plt.savefig(plot_filename);
        print(f"  LinkPred: Saved ROC plot: {plot_filename}")  #
    except Exception as e:  #
        print(f"  LinkPred: Error saving ROC plot {plot_filename}: {e}")  #
    plt.close()  #


def plot_comparison_charts(results_list: List[Dict[str, Any]], plots_output_dir: str, k_vals_table: List[int]):  #
    if not results_list:  #
        if DEBUG_VERBOSE: print("LinkPred Plot: No results for comparison charts."); return  #
    metrics_map = {'Accuracy': 'test_accuracy_keras', 'Precision': 'test_precision_sklearn', 'Recall': 'test_recall_sklearn', 'F1-Score': 'test_f1_sklearn', 'AUC': 'test_auc_sklearn'}  #
    if k_vals_table:  #
        if len(k_vals_table) > 0: metrics_map[f'Hits@{k_vals_table[0]}'] = f'test_hits_at_{k_vals_table[0]}'; metrics_map[f'NDCG@{k_vals_table[0]}'] = f'test_ndcg_at_{k_vals_table[0]}'  #
        if len(k_vals_table) > 1: metrics_map[f'Hits@{k_vals_table[1]}'] = f'test_hits_at_{k_vals_table[1]}'; metrics_map[f'NDCG@{k_vals_table[1]}'] = f'test_ndcg_at_{k_vals_table[1]}'  #
    emb_names = [res.get('embedding_name', f'Run {i + 1}') for i, res in enumerate(results_list)]  #
    num_metrics = len(metrics_map);  #
    cols = min(3, num_metrics + 1);  #
    rows = math.ceil((num_metrics + 1) / cols)  #
    plt.figure(figsize=(max(15, cols * 5), rows * 4));  #
    plot_idx = 1  #
    for disp_name, metric_key in metrics_map.items():  #
        plt.subplot(rows, cols, plot_idx);  #
        values = [res.get(metric_key, 0.0) for res in results_list]  #
        values = [v if isinstance(v, (int, float)) and not np.isnan(v) else 0.0 for v in values]  #
        bars = plt.bar(emb_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(emb_names))))  #
        plt.title(disp_name);  #
        plt.ylabel('Score');  #
        plt.xticks(rotation=30, ha="right")  #
        max_val = max(values) if values else 0.0;  #
        up_lim = 1.05  #
        if "Hits@" in disp_name:  #
            up_lim = max(max_val * 1.15 if max_val > 0 else 10, 10)  #
        elif values and max_val > 0:  #
            up_lim = max(1.05 if max_val <= 1 else max_val * 1.15, 0.1)  #
        else:  #
            up_lim = 1.05 if "Hits@" not in disp_name else max(10, max_val * 1.15)  #
        plt.ylim(0, up_lim)  #
        for bar_i, bar_obj in enumerate(bars): yv = values[bar_i]; plt.text(bar_obj.get_x() + bar_obj.get_width() / 2.0, yv + 0.01 * up_lim, f'{yv:.3f}', ha='center', va='bottom')  #
        plot_idx += 1  #
    if plot_idx <= rows * cols:  #
        plt.subplot(rows, cols, plot_idx);  #
        times = [res.get('training_time', 0.0) for res in results_list]  #
        times = [t if isinstance(t, (int, float)) and not np.isnan(t) else 0.0 for t in times]  #
        bars = plt.bar(emb_names, times, color=plt.cm.plasma(np.linspace(0, 1, len(emb_names))))  #
        plt.title('Avg Train Time/Fold');  #
        plt.ylabel('Secs');  #
        plt.xticks(rotation=30, ha="right")  #
        max_t = max(times) if times else 1.0;  #
        up_lim_t = max_t * 1.15 if max_t > 0 else 10;  #
        plt.ylim(0, up_lim_t)  #
        for bar_i, bar_obj in enumerate(bars): yv = times[bar_i]; plt.text(bar_obj.get_x() + bar_obj.get_width() / 2.0, yv + 0.01 * up_lim_t, f'{yv:.2f}s', ha='center', va='bottom')  #
    os.makedirs(plots_output_dir, exist_ok=True)  #
    plot_filename = os.path.join(plots_output_dir, "lp_comparison_metrics.png")  #
    plt.suptitle("LinkPred Model Perf Comparison (Avg over Folds)", fontsize=18);  #
    plt.tight_layout(rect=[0, 0, 1, 0.95])  #
    try:  #
        plt.savefig(plot_filename, dpi=150);
        print(f"  LinkPred: Saved metrics chart: {plot_filename}")  #
    except Exception as e:  #
        print(f"  LinkPred: Error saving metrics chart {plot_filename}: {e}")  #
    plt.close()  #


def print_results_table(results_list: List[Dict[str, Any]], k_vals_table: List[int], is_cv: bool = False, output_dir: str = ".", filename: str = "lp_results_summary_table.txt"):  #
    if not results_list:  #
        if DEBUG_VERBOSE: print("\nLinkPred Table: No results to display."); return  #
    sfx = " (Avg over Folds)" if is_cv else "";  #
    hdr_txt = f"--- LinkPred Overall Perf Table{sfx} ---";  #
    print(f"\n\n{hdr_txt}")  #
    mkeys_hdrs = [('embedding_name', "Embedding Name"), ('training_time', f"Train Time(s){sfx}"), ('test_loss', f"Val Loss{sfx}"), ('test_accuracy_keras', f"Accuracy{sfx}"), ('test_precision_sklearn', f"Precision{sfx}"),
                  #
                  ('test_recall_sklearn', f"Recall{sfx}"), ('test_f1_sklearn', f"F1-Score{sfx}"), ('test_auc_sklearn', f"AUC{sfx}")]  #
    for k_ in k_vals_table: mkeys_hdrs.append((f'test_hits_at_{k_}', f"Hits@{k_}{sfx}"))  #
    for k_ in k_vals_table: mkeys_hdrs.append((f'test_ndcg_at_{k_}', f"NDCG@{k_}{sfx}"))  #
    if is_cv: mkeys_hdrs.append(('test_f1_sklearn_std', "F1 StdDev")); mkeys_hdrs.append(('test_auc_sklearn_std', "AUC StdDev"))  #
    hdrs = [h for _, h in mkeys_hdrs];  #
    mkeys_extract = [k for k, _ in mkeys_hdrs];  #
    table_data = [hdrs]  #
    for res_d in results_list:  #
        row_vals = []  #
        for key_ in mkeys_extract:  #
            val_ = res_d.get(key_);  #
            is_ph = False  #
            if isinstance(val_, (int, float)):  #
                if val_ == -1.0 and 'loss' not in key_: is_ph = True  #
                if val_ == 0 and ('hits_at_' in key_ or 'ndcg_at_' in key_) and res_d.get('notes') and ("Single class" in res_d['notes'] or "Empty" in res_d['notes']): is_ph = True  #
            if val_ is None or is_ph or (isinstance(val_, float) and np.isnan(val_)):  #
                row_vals.append("N/A")  #
            elif isinstance(val_, float):  #
                row_vals.append(f"{val_:.2f}" if key_ == 'training_time' or '_std' in key_ else f"{val_:.4f}")  #
            elif isinstance(val_, int) and "hits_at_" in key_:  #
                row_vals.append(str(val_))  #
            else:  #
                row_vals.append(str(val_))  #
        table_data.append(row_vals)  #
    if len(table_data) <= 1:  #
        if DEBUG_VERBOSE: print("LinkPred Table: No data rows."); return  #
    col_w = [max(len(str(item)) for item in col) for col in zip(*table_data)]  #
    fmt = " | ".join([f"{{:<{w_}}}" for w_ in col_w]);  #
    tbl_str = fmt.format(*hdrs) + "\n" + "-+-".join(["-" * w_ for w_ in col_w]) + "\n"  #
    for i_ in range(1, len(table_data)): tbl_str += fmt.format(*table_data[i_]) + "\n"  #
    print(tbl_str)  #
    if filename:  #
        fpath = os.path.normpath(os.path.join(output_dir, filename));  #
        os.makedirs(output_dir, exist_ok=True)  #
        try:  #
            with open(fpath, 'w') as fH:  #
                fH.write(hdr_txt + "\n\n" + tbl_str)  #
            print(f"LinkPred: Results table saved to: {fpath}")  #
        except Exception as eS:  #
            print(f"LinkPred: Error saving results table to {fpath}: {eS}")  #


def perform_statistical_tests(results_list: List[Dict[str, Any]], main_emb_name: str, metric_key: str, alpha_val: float):  #
    if not main_emb_name: print("\nLinkPred Stats: Main embedding name not specified. Skipping."); return  #
    if len(results_list) < 2: print("\nLinkPred Stats: Need at least two methods to compare."); return  #
    print(f"\n\n--- LinkPred Stats vs '{main_emb_name}' on '{metric_key}' (Alpha={alpha_val}) ---")  #
    key_fold_scores = 'fold_auc_scores' if metric_key == 'test_auc_sklearn' else ('fold_f1_scores' if metric_key == 'test_f1_sklearn' else None)  #
    if not key_fold_scores: print(f"LinkPred Stats: Metric key '{metric_key}' not supported."); return  #
    main_res = next((r for r in results_list if r['embedding_name'] == main_emb_name), None)  #
    if not main_res: print(f"LinkPred Stats: Main model '{main_emb_name}' not found."); return  #
    main_scores_all = main_res.get(key_fold_scores, []);  #
    main_scores_valid = [s for s in main_scores_all if not np.isnan(s)]  #
    if len(main_scores_valid) < 2: print(f"LinkPred Stats: Not enough valid scores ({len(main_scores_valid)}) for main model '{main_emb_name}'. Min 2 required."); return  #
    other_results = [r for r in results_list if r['embedding_name'] != main_emb_name]  #
    if not other_results: print("LinkPred Stats: No other models to compare."); return  #
    hdr_parts = [f"{'Compared Embedding':<30}", f"{'Wilcoxon p-val':<15}", f"{'Signif. (p<{alpha_val})':<20}", f"{'Pearson r':<10}", f"{'r-squared':<10}"]  #
    hdr_str = " | ".join(hdr_parts);  #
    print(hdr_str);  #
    print("-" * len(hdr_str))  #
    for other_res in other_results:  #
        other_name = other_res['embedding_name'];  #
        other_scores_all = other_res.get(key_fold_scores, [])  #
        other_scores_valid = [s for s in other_scores_all if not np.isnan(s)]  #
        if len(main_scores_valid) != len(other_scores_valid) or len(main_scores_valid) < 2:  #
            print(f"{other_name:<30} | {'N/A (scores mismatch/few)':<15} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10}");  #
            continue  #
        main_np, other_np = np.array(main_scores_valid, dtype=float), np.array(other_scores_valid, dtype=float)  #
        p_wil_str, p_r_str, r2_str = "N/A", "N/A", "N/A";  #
        sig_diff = "N/A";  #
        corr_note = ""  #
        try:  #
            if not np.allclose(main_np, other_np):  #
                _, p_w = wilcoxon(main_np, other_np, alternative='two-sided', zero_method='pratt')  #
                p_wil_str = f"{p_w:.4f}"  #
                if p_w < alpha_val:  #
                    m_main, m_other = np.mean(main_np), np.mean(other_np);
                    sig_diff = f"Yes ({'Main Better' if m_main > m_other else ('Main Worse' if m_main < m_other else 'Diff, Means Eq.')})"  #
                else:  #
                    sig_diff = "No"  #
            else:  #
                p_wil_str = "1.0000 (Identical)";
                sig_diff = "No (Identical Scores)"  #
        except ValueError as eW:  #
            p_wil_str = "Error";
            sig_diff = "N/A (Wilcoxon Err)"  #
        try:  #
            if len(np.unique(main_np)) > 1 and len(np.unique(other_np)) > 1:  #
                r_val, p_c = pearsonr(main_np, other_np);  #
                p_r_str = f"{r_val:.4f}";  #
                r2_str = f"{r_val ** 2:.4f}"  #
                if p_c < alpha_val: corr_note = f"(Corr. p={p_c:.2e})"  #
        except Exception as eC:  #
            p_r_str, r2_str = "Error", "Error"  #
        print(f"{other_name:<30} | {p_wil_str:<15} | {sig_diff:<20} | {p_r_str:<10} | {r2_str:<10} {corr_note}")  #
    print("-" * len(hdr_str));  #
    print("Note: Wilcoxon tests difference. Pearson r for linear correlation.")  #


def main_workflow_cv_lp(embedding_name_cv: str, protein_embeddings_cv: Dict[str, np.ndarray], positive_pairs_cv: List[Tuple[str, str, int]], negative_pairs_cv: List[Tuple[str, str, int]],  #
                        mlp_params_dict_cv: Dict[str, Any], edge_emb_method_cv: str, num_folds_cv: int, rand_state_cv: int, max_train_samp_cv_lp: Optional[int], max_val_samp_cv_lp: Optional[int],
                        max_shuff_buffer_cv: int,  #
                        batch_size_cv: int, epochs_cv: int, learning_rate_cv: float, k_vals_ranking_cv: List[int]) -> Dict[str, Any]:  #
    agg_res: Dict[str, Any] = {'embedding_name': embedding_name_cv, 'training_time': 0.0, 'history_dict_fold1': {}, 'roc_data_representative': (np.array([]), np.array([]), 0.0), 'notes': "", 'fold_f1_scores': [],  #
                               'fold_auc_scores': [], **{k: 0.0 for k in  #
                                                         ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras', 'test_recall_keras', 'test_precision_sklearn', 'test_recall_sklearn',  #
                                                          'test_f1_sklearn', 'test_auc_sklearn']}, **{f'test_hits_at_{kv}': 0.0 for kv in k_vals_ranking_cv}, **{f'test_ndcg_at_{kv}': 0.0 for kv in k_vals_ranking_cv}}  #
    if not protein_embeddings_cv: agg_res['notes'] = "No embeds for CV."; return agg_res  #
    all_i_pairs = positive_pairs_cv + negative_pairs_cv  #
    if not all_i_pairs: agg_res['notes'] = "No interactions for CV."; return agg_res  #
    graph_proc_lp = Graph_LP();  #
    X_full, y_full = graph_proc_lp.create_edge_embeddings(all_i_pairs, protein_embeddings_cv, method=edge_emb_method_cv)  #
    if X_full is None or y_full is None or len(X_full) == 0: agg_res['notes'] = "Dataset creation failed for CV."; return agg_res  #
    if DEBUG_VERBOSE: print(f"LinkPred CV: Samples for {embedding_name_cv}: {len(y_full)} (+:{np.sum(y_full == 1)}, -:{np.sum(y_full == 0)})")  #
    if len(np.unique(y_full)) < 2: agg_res['notes'] = "Single class in y_full for CV."; return agg_res  #
    skf = StratifiedKFold(n_splits=num_folds_cv, shuffle=True, random_state=rand_state_cv)  #
    fold_metrics, total_train_t = [], 0.0  #
    for fold_n, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):  #
        if DEBUG_VERBOSE: print(f"\nLinkPred CV: --- Fold {fold_n + 1}/{num_folds_cv} for {embedding_name_cv} ---")  #
        X_tr, y_tr = X_full[train_idx], y_full[train_idx];  #
        X_v, y_v = X_full[val_idx], y_full[val_idx]  #
        X_tr_use, y_tr_use = X_tr, y_tr  #
        if max_train_samp_cv_lp and X_tr.shape[0] > max_train_samp_cv_lp:  #
            s_idx = np.random.choice(X_tr.shape[0], max_train_samp_cv_lp, replace=False);  #
            X_tr_use, y_tr_use = X_tr[s_idx], y_tr[s_idx]  #
        X_v_use, y_v_use = X_v, y_v  #
        if max_val_samp_cv_lp and X_v.shape[0] > max_val_samp_cv_lp:  #
            s_idx = np.random.choice(X_v.shape[0], max_val_samp_cv_lp, replace=False);  #
            X_v_use, y_v_use = X_v[s_idx], y_v[s_idx]  #
        curr_fold_m: Dict[str, Any] = {'fold': fold_n + 1}  #
        if X_tr_use.shape[0] == 0:  #
            if DEBUG_VERBOSE: print(f"LinkPred CV: Fold {fold_n + 1} - Training data empty. Skipping.")  #
            for k_m_key in agg_res:  #
                if 'test_' in k_m_key or 'hits_at' in k_m_key or 'ndcg_at' in k_m_key: curr_fold_m[k_m_key] = np.nan  #
            fold_metrics.append(curr_fold_m);  #
            continue  #

        shuf_buf = min(X_tr_use.shape[0], max_shuff_buffer_cv)  #
        train_ds = tf.data.Dataset.from_tensor_slices((X_tr_use, y_tr_use)).shuffle(shuf_buf).batch(batch_size_cv).prefetch(tf.data.AUTOTUNE)  #
        val_ds = tf.data.Dataset.from_tensor_slices((X_v_use, y_v_use)).batch(batch_size_cv).prefetch(tf.data.AUTOTUNE) if X_v_use.shape[0] > 0 else None  #

        model_lp = build_mlp_model_lp(X_tr_use.shape[1], learning_rate_cv, mlp_p_dict=mlp_params_dict_cv)  #
        if fold_n == 0 and DEBUG_VERBOSE: model_lp.summary(print_fn=lambda x_print: print(x_print))  #
        if DEBUG_VERBOSE: print(f"LinkPred CV: Training Fold {fold_n + 1} ({X_tr_use.shape[0]} train, {X_v_use.shape[0]} val samples)...")  #

        s_t = time.time();  #
        hist = model_lp.fit(train_ds, epochs=epochs_cv, validation_data=val_ds, verbose=1 if DEBUG_VERBOSE else 0)  #
        fold_train_t = time.time() - s_t;  #
        total_train_t += fold_train_t;  #
        curr_fold_m['training_time'] = fold_train_t  #
        if fold_n == 0: agg_res['history_dict_fold1'] = hist.history  #

        y_v_eval = np.array(y_v_use).flatten()  #
        if X_v_use.shape[0] > 0 and val_ds:  #
            eval_r = model_lp.evaluate(val_ds, verbose=0)  #
            k_keys = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras', 'test_recall_keras']  #
            for nm, vl in zip(k_keys, eval_r): curr_fold_m[nm] = vl  #
            y_pred_p = model_lp.predict(X_v_use, batch_size=batch_size_cv).flatten();  #
            y_pred_c = (y_pred_p > 0.5).astype(int)  #
            curr_fold_m.update({'test_precision_sklearn': precision_score(y_v_eval, y_pred_c, zero_division=0), 'test_recall_sklearn': recall_score(y_v_eval, y_pred_c, zero_division=0),  #
                                'test_f1_sklearn': f1_score(y_v_eval, y_pred_c, zero_division=0)})  #
            if len(np.unique(y_v_eval)) > 1:  #
                curr_fold_m['test_auc_sklearn'] = roc_auc_score(y_v_eval, y_pred_p)  #
                if fold_n == 0: fpr, tpr, _ = roc_curve(y_v_eval, y_pred_p); agg_res['roc_data_representative'] = (fpr, tpr, curr_fold_m['test_auc_sklearn'])  #
            else:  #
                curr_fold_m['test_auc_sklearn'] = 0.5;  #
            if fold_n == 0 and len(np.unique(y_v_eval)) <= 1: agg_res['roc_data_representative'] = (np.array([]), np.array([]), 0.5)  #

            desc_idx = np.argsort(y_pred_p)[::-1];  #
            sorted_y_v = y_v_eval[desc_idx]  #
            for k_r in k_vals_ranking_cv:  #
                eff_k = min(k_r, len(sorted_y_v))  #
                curr_fold_m[f'test_hits_at_{k_r}'] = np.sum(sorted_y_v[:eff_k] == 1) if eff_k > 0 else 0  #
                curr_fold_m[f'test_ndcg_at_{k_r}'] = ndcg_score(np.asarray([y_v_eval]), np.asarray([y_pred_p]), k=eff_k, ignore_ties=True) if eff_k > 0 and len(np.unique(y_v_eval)) > 1 else 0.0  #
        else:  #
            if DEBUG_VERBOSE: print(f"LinkPred CV: Fold {fold_n + 1} - Eval skipped (no val data). Metrics NaN.")  #
            nan_keys = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras', 'test_recall_keras', 'test_precision_sklearn', 'test_recall_sklearn', 'test_f1_sklearn', 'test_auc_sklearn'] + [  #
                f'test_hits_at_{k_val}' for k_val in k_vals_ranking_cv] + [f'test_ndcg_at_{k_val}' for k_val in k_vals_ranking_cv]  #
            for kN in nan_keys: curr_fold_m[kN] = np.nan  #
        fold_metrics.append(curr_fold_m)  #
        del model_lp, hist, train_ds, val_ds;  #
        gc.collect();  #
        tf.keras.backend.clear_session()  #

    if not fold_metrics: agg_res['notes'] = "No folds completed for CV."; return agg_res  #

    for key_avg in agg_res.keys():  #
        if key_avg not in ['embedding_name', 'history_dict_fold1', 'roc_data_representative', 'notes', 'fold_f1_scores', 'fold_auc_scores', 'training_time', 'test_f1_sklearn_std', 'test_auc_sklearn_std']:  #
            v_fold_vals = [fm.get(key_avg) for fm in fold_metrics if fm.get(key_avg) is not None and not np.isnan(fm.get(key_avg))]  #
            agg_res[key_avg] = np.mean(v_fold_vals) if v_fold_vals else 0.0  #
    agg_res['training_time'] = total_train_t / len(fold_metrics) if fold_metrics else 0.0  #
    agg_res['fold_f1_scores'] = [fm.get('test_f1_sklearn', np.nan) for fm in fold_metrics]  #
    agg_res['fold_auc_scores'] = [fm.get('test_auc_sklearn', np.nan) for fm in fold_metrics]  #
    f1_v = [s for s in agg_res['fold_f1_scores'] if not np.isnan(s)];  #
    agg_res['test_f1_sklearn_std'] = np.std(f1_v) if len(f1_v) > 1 else 0.0  #
    auc_v = [s for s in agg_res['fold_auc_scores'] if not np.isnan(s)];  #
    agg_res['test_auc_sklearn_std'] = np.std(auc_v) if len(auc_v) > 1 else 0.0  #
    if DEBUG_VERBOSE: print(f"LinkPred CV: ===== Finished CV for {embedding_name_cv} =====")  #
    return agg_res  #


def run_evaluation_pipeline(positive_interactions_fp: str, negative_interactions_fp: str, embedding_configs_input: List[Dict[str, Any]], output_root_dir: str, run_ngram_gcn_gen: bool, ngram_gcn_config: Dict[str, Any],  #
                            lp_random_state: int, lp_sample_negative_pairs: Optional[int], lp_edge_embedding_method: str, lp_n_folds: int, lp_max_train_samples_cv: Optional[int], lp_max_val_samples_cv: Optional[int],  #
                            lp_max_shuffle_buffer_size: int, lp_plot_training_history: bool, lp_mlp_params_dict: Dict[str, Any], lp_batch_size: int, lp_epochs: int, lp_learning_rate: float, lp_k_vals_ranking: List[int],
                            #
                            lp_k_vals_table: List[int], lp_main_embedding_name_stats: str, lp_stat_test_metric: str, lp_stat_test_alpha: float):  #
    print(f"\n--- Starting Combined Evaluation Pipeline ---")  #
    print(f"Output root: {output_root_dir}")  #
    plots_dir = os.path.join(output_root_dir, "plots_link_prediction")  #
    os.makedirs(output_root_dir, exist_ok=True);  #
    os.makedirs(plots_dir, exist_ok=True)  #

    loader_func_map: Dict[str, Callable] = {'load_h5_embeddings_selectively': FileOps.load_h5_embeddings_selectively, 'load_h5_embeddings': FileOps.load_h5_embeddings_selectively,  #
                                            'load_custom_embeddings': FileOps.load_custom_embeddings}  #
    default_loader_key = 'load_h5_embeddings_selectively'  #

    current_embedding_configs_to_process = list(embedding_configs_input)  #

    # --- N-gramGCN Embedding Generation Step (Happens FIRST) --- #
    if run_ngram_gcn_gen:  #
        print("\n" + "-" * 20 + " N-gramGCN Embedding Generation " + "-" * 20)  #
        ngram_fasta_path = ngram_gcn_config.get('input_fasta_path')  #
        ngram_output_base = ngram_gcn_config.get('output_embeddings_dir')  #

        if not ngram_fasta_path or not os.path.exists(ngram_fasta_path) or not ngram_output_base:  #
            print(f"NgramGCN Error: Valid 'input_fasta_path' ({ngram_fasta_path}) and 'output_embeddings_dir' ({ngram_output_base}) must be in ngram_gcn_config. Skipping N-gramGCN generation.")  #
        else:  #
            run_specific_ngram_name = f"NgramGCN_N{ngram_gcn_config.get('max_n_val')}_H2D{ngram_gcn_config.get('hidden_dim2')}_E{ngram_gcn_config.get('epochs')}"  #
            ngram_run_specific_output_dir = os.path.join(ngram_output_base, run_specific_ngram_name)  #

            all_ids_in_ngram_fasta_parsed = {pid for pid, seq in parse_fasta_sequences_with_ids_ngram(ngram_fasta_path) if pid is not None}  #

            if not all_ids_in_ngram_fasta_parsed:  #
                print(f"NgramGCN: No protein IDs found in the NgramGCN input FASTA: {ngram_fasta_path}. Skipping N-gramGCN generation.")  #
            else:  #
                if DEBUG_VERBOSE:  #
                    print(f"NgramGCN: Will generate embeddings for all {len(all_ids_in_ngram_fasta_parsed)} proteins found in its input FASTA ('{os.path.basename(ngram_fasta_path)}').")  #

                generated_prot_h5, _ = generate_and_save_ngram_embeddings(fasta_filepath=ngram_fasta_path, protein_ids_to_generate_for=all_ids_in_ngram_fasta_parsed,  # Generate for ALL in its FASTA #
                                                                          max_n_for_ngram=ngram_gcn_config.get('max_n_val'), output_dir_for_this_run=ngram_run_specific_output_dir,
                                                                          one_gram_init_embed_dim=ngram_gcn_config.get('init_dim'),  #
                                                                          hidden_dim1=ngram_gcn_config.get('hidden_dim1'), hidden_dim2=ngram_gcn_config.get('hidden_dim2'),
                                                                          pe_max_len_config=ngram_gcn_config.get('pe_max_len'), dropout=ngram_gcn_config.get('dropout'),  #
                                                                          lr=ngram_gcn_config.get('lr'), epochs_per_level=ngram_gcn_config.get('epochs'), use_vector_gnn_coeffs=ngram_gcn_config.get('use_vector_coeffs'), #
                                                                          ngram_l2_reg=ngram_gcn_config.get('weight_decay', 0.0), task_modes_per_level=ngram_gcn_config.get('task_per_level', {}),
                                                                          default_task_mode=ngram_gcn_config.get('default_task_mode', 'next_node'))  #
                if generated_prot_h5 and os.path.exists(generated_prot_h5):  #
                    print(f"NgramGCN: Successfully generated per-protein embeddings at: {generated_prot_h5}")  #
                    current_embedding_configs_to_process.append({"path": generated_prot_h5, "name": ngram_gcn_config.get('generated_emb_name', "NgramGCN-DefaultName"), "loader_func_key": default_loader_key})  #
                else:  #
                    print(f"NgramGCN: Failed to generate per-protein H5 file. These NgramGCN embeddings will not be evaluated.")  #
        print("-" * 20 + " N-gramGCN Generation Finished " + "-" * 20 + "\n")  #

    # --- Load Interaction Data and Determine Common IDs for Evaluation --- #
    if not current_embedding_configs_to_process:  #
        print("LinkPred CRITICAL: No embedding files to evaluate (neither pre-existing nor generated after NgramGCN step). Exiting.")  #
        return  #

    print("LinkPred: Loading all interaction pairs for evaluation...")  #
    pos_pairs_all_raw = ProteinFileOps.load_interaction_pairs(positive_interactions_fp, 1, random_state_for_sampling=lp_random_state)  #
    neg_pairs_all_raw = ProteinFileOps.load_interaction_pairs(negative_interactions_fp, 0, sample_n=lp_sample_negative_pairs, random_state_for_sampling=lp_random_state)  #

    initial_interacting_proteins = set(p_id for p1, p2, _ in (pos_pairs_all_raw + neg_pairs_all_raw) for p_id in (p1, p2) if p_id is not None)  #
    if not initial_interacting_proteins:  #
        print("LinkPred CRITICAL: No valid protein IDs found in interaction files after parsing. Exiting.");  #
        return  #
    if DEBUG_VERBOSE: print(f"LinkPred: Initial {len(initial_interacting_proteins)} unique protein IDs (parsed as accessions) from interaction files.")  #

    print("LinkPred: Determining FINAL common protein IDs across ALL H5 files and interaction data for FAIR evaluation...")  #
    common_protein_ids_for_eval = set(initial_interacting_proteins)  #

    valid_embedding_configs_for_common_set = []  #
    for config_item in current_embedding_configs_to_process:  #
        path_lp = config_item.get('path')  #
        name_lp = config_item.get('name', os.path.basename(path_lp) if path_lp else "UnknownEmb")  #
        config_item['name'] = name_lp  #

        if not path_lp or not os.path.exists(os.path.normpath(path_lp)):  #
            print(f"LinkPred: Path for '{name_lp}' is invalid/missing ('{path_lp}'). It will be EXCLUDED.")  #
            continue  #

        try:  #
            with h5py.File(os.path.normpath(path_lp), 'r') as hf:  #
                # Standardize keys FROM THIS H5 FILE before intersection #
                ids_in_this_h5_standardized = {extract_canonical_id_and_type(key)[1] for key in hf.keys() if extract_canonical_id_and_type(key)[1] is not None}  #

                if not ids_in_this_h5_standardized:  #
                    if DEBUG_VERBOSE: print(f"  LinkPred: No valid standardized IDs found in H5 keys of '{name_lp}'. Excluding it from common set for now.")  #
                    continue  # This file contributes no usable IDs, skip to next config item #

                original_common_set_size = len(common_protein_ids_for_eval)  #
                common_protein_ids_for_eval.intersection_update(ids_in_this_h5_standardized)  #

                if DEBUG_VERBOSE:  #
                    print(f"  LinkPred: After '{name_lp}' (std H5 keys: {len(ids_in_this_h5_standardized)}), common set size: {len(common_protein_ids_for_eval)}")  #

                if not common_protein_ids_for_eval and original_common_set_size > 0:  #
                    print(f"LinkPred: No common protein IDs remaining after processing '{name_lp}'. Further common set determination stopped for subsequent files.");  #
                    break  #

                valid_embedding_configs_for_common_set.append(config_item)  #
        except Exception as e:  #
            print(f"LinkPred: Error reading H5 '{name_lp}' at {path_lp} to get keys: {e}. This file will be excluded from fair evaluation.");  #

    current_embedding_configs_to_process = valid_embedding_configs_for_common_set  #

    if not common_protein_ids_for_eval:  #
        print("LinkPred CRITICAL: No common protein IDs found for fair evaluation after checking all H5 files. Cannot proceed.");  #
        return  #

    final_required_ids_for_evaluation = common_protein_ids_for_eval  #
    if DEBUG_VERBOSE: print(f"LinkPred: FINAL common set for evaluation (standardized accessions): {len(final_required_ids_for_evaluation)} IDs.")  #

    final_pos_pairs = [(p1, p2, lbl) for p1, p2, lbl in pos_pairs_all_raw if p1 in final_required_ids_for_evaluation and p2 in final_required_ids_for_evaluation]  #
    final_neg_pairs = [(p1, p2, lbl) for p1, p2, lbl in neg_pairs_all_raw if p1 in final_required_ids_for_evaluation and p2 in final_required_ids_for_evaluation]  #

    if not final_pos_pairs and not final_neg_pairs:  #
        print(f"LinkPred CRITICAL: No interaction pairs remain after FINAL filtering by common protein IDs (L_pos={len(final_pos_pairs)}, L_neg={len(final_neg_pairs)}). Cannot proceed.");  #
        return  #
    if DEBUG_VERBOSE: print(f"LinkPred: Final Pos pairs for eval: {len(final_pos_pairs)}, Final Neg pairs for eval: {len(final_neg_pairs)}")  #

    processed_emb_configs_final_lp: List[Dict[str, Any]] = []  #
    for item_cfg in current_embedding_configs_to_process:  #
        cfg_lp = {};  #
        path_lp = item_cfg.get('path');  #
        name_lp = item_cfg.get('name');  #
        loader_k = item_cfg.get('loader_func_key', default_loader_key)  #
        cfg_lp['path'] = os.path.normpath(path_lp);  #
        cfg_lp['name'] = name_lp  #
        act_loader = loader_func_map.get(loader_k)  #
        if not act_loader: print(f"LinkPred: Loader '{loader_k}' missing for {cfg_lp['name']}. Skipping."); continue  #
        cfg_lp['loader_func'] = act_loader;  #
        processed_emb_configs_final_lp.append(cfg_lp)  #

    if not processed_emb_configs_final_lp:  #
        print("LinkPred: No embedding configurations left to process for Link Prediction after all checks. Exiting.")  #
        return  #

    all_cv_results_lp: List[Dict[str, Any]] = []  #
    for cfg_item_lp in processed_emb_configs_final_lp:  #
        if DEBUG_VERBOSE: print(f"\nLinkPred CV: {'=' * 15} Eval: {cfg_item_lp['name']} ({cfg_item_lp['path']}) {'=' * 15}")  #
        prot_embs_lp = cfg_item_lp['loader_func'](cfg_item_lp['path'], final_required_ids_for_evaluation)  #

        if prot_embs_lp and len(prot_embs_lp) > 0:  #
            loaded_and_relevant_ids = set(prot_embs_lp.keys()).intersection(final_required_ids_for_evaluation)  #
            if not loaded_and_relevant_ids and (len(final_pos_pairs) > 0 or len(final_neg_pairs) > 0):  #
                if DEBUG_VERBOSE: print(f"LinkPred CV: Skipping {cfg_item_lp['name']}: No embeddings were loaded for the FINAL common proteins from THIS file.")  #
                all_cv_results_lp.append({'embedding_name': cfg_item_lp['name'], 'notes': f"No embeds loaded for FINAL common IDs from this file.", 'fold_f1_scores': [], 'fold_auc_scores': []});  #
                continue  #

            cv_res = main_workflow_cv_lp(embedding_name_cv=cfg_item_lp['name'], protein_embeddings_cv=prot_embs_lp, positive_pairs_cv=final_pos_pairs, negative_pairs_cv=final_neg_pairs,  #
                                         mlp_params_dict_cv=lp_mlp_params_dict, edge_emb_method_cv=lp_edge_embedding_method, num_folds_cv=lp_n_folds, rand_state_cv=lp_random_state,
                                         max_train_samp_cv_lp=lp_max_train_samples_cv,  #
                                         max_val_samp_cv_lp=lp_max_val_samples_cv, max_shuff_buffer_cv=lp_max_shuffle_buffer_size, batch_size_cv=lp_batch_size, epochs_cv=lp_epochs, learning_rate_cv=lp_learning_rate,  #
                                         k_vals_ranking_cv=lp_k_vals_ranking)  #
            if cv_res:  #
                all_cv_results_lp.append(cv_res)  #
                hist_f1 = cv_res.get('history_dict_fold1', {})  #
                if lp_plot_training_history and hist_f1 and any(isinstance(vL, list) and len(vL) > 0 for vL in hist_f1.values()):  #
                    plot_training_history(hist_f1, cv_res['embedding_name'], plots_dir, fold_num=1)  #
        else:  #
            print(f"LinkPred CV: Skipping {cfg_item_lp['name']}: Failed to load any embeddings for the FINAL common protein set from this file.")  #
            all_cv_results_lp.append({'embedding_name': cfg_item_lp['name'], 'notes': "No embeds for FINAL common proteins from this file.", 'fold_f1_scores': [], 'fold_auc_scores': []})  #
        del prot_embs_lp;  #
        gc.collect()  #

    if all_cv_results_lp:  #
        print("\nLinkPred: Generating aggregate comparison plots & table...")  #
        valid_roc_exists = any(  #
            isinstance(r.get('roc_data_representative'), tuple) and len(r['roc_data_representative']) == 3 and r['roc_data_representative'][0] is not None and hasattr(r['roc_data_representative'][0], '__len__') and len(#
                r['roc_data_representative'][0]) > 0 for r in all_cv_results_lp)  #
        if valid_roc_exists:  #
            plot_roc_curves(all_cv_results_lp, plots_dir)  #
        else:  #
            print("LinkPred Plot: No valid ROC data for plotting.")  #
        plot_comparison_charts(all_cv_results_lp, plots_dir, lp_k_vals_table)  #
        print_results_table(all_cv_results_lp, lp_k_vals_table, is_cv=True, output_dir=output_root_dir, filename="linkpred_summary_table.txt")  #
        perform_statistical_tests(all_cv_results_lp, lp_main_embedding_name_stats, lp_stat_test_metric, lp_stat_test_alpha)  #
    else:  #
        print("\nLinkPred: No results from any configs to plot/tabulate.")  #
    print(f"--- Combined Evaluation Pipeline Finished. Results in: {output_root_dir} ---")  #


# ============================================================================
# END: Evaluater Code
# ============================================================================

if __name__ == '__main__':  #
    print(f"Combined N-gramGCN Generation and Link Prediction Evaluation Script")  #
    print(f"Python Version: {sys.version.split()[0]}")  #
    print(f"NumPy Version: {np.__version__}")  #
    print(f"Pandas Version: {pd.__version__}")  #
    print(f"TensorFlow Version: {tf.__version__}")  #
    print(f"Torch Version: {torch.__version__}")  #
    print(f"Scikit-learn Version: {sklearn.__version__}")  #
    print(f"H5Py Version: {h5py.__version__}")  #
    try:  #
        print(f"BioPython Version: {SeqIO.__version__}")  #
    except AttributeError:  #
        import Bio;  #

        print(f"BioPython Version: {Bio.__version__}")  #
    try:  #
        print(f"TQDM Version: {tqdm.__version__}")  #
    except AttributeError:  #
        pass  #
    try:  #
        print(f"NetworkX Version: {nx.__version__}")  #
    except NameError:  #
        print("NetworkX not imported or found")  #
    except AttributeError:  #
        pass  # In case __version__ is not set #
    try:  #
        print(f"Community (Louvain) Version: {community_louvain.__version__}")  #
    except NameError:  #
        print("Community (Louvain) not imported or found")  #
    except AttributeError:  #
        pass  # In case __version__ is not set #

    mlp_params_lp = {'dense1_units': MLP_DENSE1_UNITS_LP, 'dropout1_rate': MLP_DROPOUT1_RATE_LP, 'dense2_units': MLP_DENSE2_UNITS_LP, 'dropout2_rate': MLP_DROPOUT2_RATE_LP, 'l2_reg': MLP_L2_REG_LP}  #
    ngram_gcn_hparams = {'input_fasta_path': NGRAM_GCN_INPUT_FASTA_PATH, 'output_embeddings_dir': NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, 'max_n_val': NGRAM_GCN_MAX_N, 'init_dim': NGRAM_GCN_1GRAM_INIT_DIM,  #
                         'hidden_dim1': NGRAM_GCN_HIDDEN_DIM_1, 'hidden_dim2': NGRAM_GCN_HIDDEN_DIM_2, 'pe_max_len': NGRAM_GCN_PE_MAX_LEN, 'dropout': NGRAM_GCN_DROPOUT, 'lr': NGRAM_GCN_LR,  #
                         'weight_decay': NGRAM_GCN_WEIGHT_DECAY, 'epochs': NGRAM_GCN_EPOCHS_PER_LEVEL, 'use_vector_coeffs': NGRAM_GCN_USE_VECTOR_COEFFS, 'generated_emb_name': NGRAM_GCN_GENERATED_EMB_NAME,  #
                         'task_per_level': NGRAM_GCN_TASK_PER_LEVEL, 'default_task_mode': NGRAM_GCN_DEFAULT_TASK_MODE}  #
    print("\n" + "=" * 30 + " RUNNING NORMAL EVALUATION CASE " + "=" * 30)  #
    essential_files_exist = True  #
    if not os.path.exists(normal_positive_interactions_path): print(f"Missing NORMAL RUN positive interactions: {normal_positive_interactions_path}"); essential_files_exist = False  #
    if not os.path.exists(normal_negative_interactions_path): print(f"Missing NORMAL RUN negative interactions: {normal_negative_interactions_path}"); essential_files_exist = False  #
    if RUN_AND_EVALUATE_NGRAM_GCN and (not ngram_gcn_hparams.get('input_fasta_path') or not os.path.exists(str(ngram_gcn_hparams.get('input_fasta_path')))):  #
        print(f"Missing or invalid NORMAL RUN NgramGCN FASTA: {ngram_gcn_hparams.get('input_fasta_path')}");  #
        essential_files_exist = False  #
    for emb_file_config in normal_embedding_files_to_evaluate:  #
        emb_path = emb_file_config.get("path")  #
        if not emb_path or not os.path.exists(os.path.normpath(str(emb_path))):  #
            print(f"Missing pre-computed embedding file for NORMAL RUN: {emb_path}");  #
            essential_files_exist = False  #
    if not essential_files_exist:  #
        print(f"CRITICAL ERROR: One or more essential input files for the normal run are missing or invalid. Skipping normal evaluation.")  #
    else:  #
        run_evaluation_pipeline(positive_interactions_fp=normal_positive_interactions_path, negative_interactions_fp=normal_negative_interactions_path, embedding_configs_input=normal_embedding_files_to_evaluate,  #
                                output_root_dir=normal_output_main_dir, run_ngram_gcn_gen=RUN_AND_EVALUATE_NGRAM_GCN, ngram_gcn_config=ngram_gcn_hparams, lp_random_state=RANDOM_STATE,  #
                                lp_sample_negative_pairs=normal_sample_negative_pairs, lp_edge_embedding_method=EDGE_EMBEDDING_METHOD_LP, lp_n_folds=N_FOLDS_LP, lp_max_train_samples_cv=MAX_TRAIN_SAMPLES_CV_LP,  #
                                lp_max_val_samples_cv=MAX_VAL_SAMPLES_CV_LP, lp_max_shuffle_buffer_size=MAX_SHUFFLE_BUFFER_SIZE_LP, lp_plot_training_history=PLOT_TRAINING_HISTORY_LP, lp_mlp_params_dict=mlp_params_lp,  #
                                lp_batch_size=BATCH_SIZE_LP, lp_epochs=EPOCHS_LP, lp_learning_rate=LEARNING_RATE_LP, lp_k_vals_ranking=K_VALUES_FOR_RANKING_METRICS_LP, lp_k_vals_table=K_VALUES_FOR_TABLE_DISPLAY_LP,  #
                                lp_main_embedding_name_stats=MAIN_EMBEDDING_NAME_FOR_STATS_LP, lp_stat_test_metric=STATISTICAL_TEST_METRIC_KEY_LP, lp_stat_test_alpha=STATISTICAL_TEST_ALPHA_LP)  #
    print("\nCombined Script finished.")  #
