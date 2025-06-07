# ==============================================================================
# Combined Protein-Protein Interaction Evaluation and N-gramGCN Embedding Generation Script
# VERSION: 5.0 (Final, Fully Integrated and Corrected)
# ==============================================================================
import os
import sys
import shutil
import multiprocessing
from itertools import repeat
import numpy as np
import pandas as pd
import time
import gc
import h5py
import math
import re
import random
import traceback
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any, Set, Tuple, Union, Callable
from collections import defaultdict
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt

# NetworkX for Community Detection
import networkx as nx
import community as community_louvain

# Dask for memory-efficient graph construction
try:
    import dask
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

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
# --- MAIN CONFIGURATION CLASS ---
# ==============================================================================
class ScriptConfig:
    """
    Centralized configuration class for the entire script.
    Modify parameters here.
    """

    def __init__(self):
        # --- GENERAL SETTINGS ---
        self.DEBUG_VERBOSE = True
        self.RANDOM_STATE = 42

        # !!! IMPORTANT: SET YOUR BASE DIRECTORIES HERE !!!
        self.BASE_DATA_DIR = "C:/ProgramData/ProtDiGCN/"
        self.BASE_OUTPUT_DIR = os.path.join(self.BASE_DATA_DIR, "ppi_evaluation_results_final_dummy")

        # --- Master Switches for Workflow Control ---
        self.RUN_NGRAM_GCN_GENERATION = True
        self.RUN_LINK_PREDICTION_EVALUATION = True

        # --- N-gramGCN Generation Configuration ---
        self.NGRAM_GCN_INPUT_FASTA_PATH = os.path.join(self.BASE_DATA_DIR, "uniprot_sequences_sample.fasta")
        self.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings")
        self.TRAINING_MODE = 'full_graph'
        self.NGRAM_GCN_MAX_N = 3
        self.NGRAM_GCN_1GRAM_INIT_DIM = 64
        self.NGRAM_GCN_HIDDEN_DIM_1 = 128
        self.NGRAM_GCN_HIDDEN_DIM_2 = 64
        self.NGRAM_GCN_EPOCHS_PER_LEVEL = 500
        self.NGRAM_GCN_LR = 0.001
        self.NGRAM_GCN_PE_MAX_LEN = 10
        self.NGRAM_GCN_DROPOUT = 0.5
        self.NGRAM_GCN_WEIGHT_DECAY = 1e-4
        self.NGRAM_GCN_USE_VECTOR_COEFFS = True
        self.NGRAM_GCN_GENERATED_EMB_NAME = "NgramGCN-Generated"
        self.NGRAM_GCN_TASK_PER_LEVEL: Dict[int, str] = {1: 'community_label'}
        self.NGRAM_GCN_DEFAULT_TASK_MODE = 'next_node'
        self.NGRAM_GCN_BATCH_SIZE = 512
        self.NGRAM_GCN_NUM_NEIGHBORS = [25, 15, 10]
        self.NGRAM_GCN_INFERENCE_BATCH_SIZE = 1024

        # --- Parallel & Graph Construction Config ---
        self.DASK_CHUNK_SIZE = 2000000
        self.TEMP_FILE_DIR = os.path.join(self.BASE_OUTPUT_DIR, "temp")
        self.PARALLEL_CONSTRUCTION_WORKERS: Optional[int] = 16

        # --- Link Prediction Evaluation Configuration ---
        self.LP_POSITIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/positive_interactions.csv')
        self.LP_NEGATIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/negative_interactions.csv')
        self.LP_DESIRED_POS_TO_NEG_RATIO: float = 1.0
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [{"path": os.path.join(self.BASE_DATA_DIR, "models/per-protein.h5"), "name": "ProtT5-Precomputed"},
                                               {"path": os.path.join(self.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, "per_protein_embeddings_from_3gram.h5"), "name": "NgramGCN-Generated"}, ]
        self.LP_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.LP_N_FOLDS = 5
        self.LP_MLP_DENSE1_UNITS = 128
        self.LP_MLP_DROPOUT1_RATE = 0.4
        self.LP_MLP_DENSE2_UNITS = 64
        self.LP_MLP_DROPOUT2_RATE = 0.4
        self.LP_MLP_L2_REG = 0.001
        self.LP_BATCH_SIZE = 128
        self.LP_EPOCHS = 10
        self.LP_LEARNING_RATE = 1e-3

        # --- Reporting & Analysis Configuration ---
        self.LP_PLOT_TRAINING_HISTORY = True
        self.LP_K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
        self.LP_K_VALUES_FOR_TABLE_DISPLAY = [50, 100]
        self.LP_MAIN_EMBEDDING_NAME_FOR_STATS = "ProtT5-Precomputed"
        self.LP_STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'
        self.LP_STATISTICAL_TEST_ALPHA = 0.05


# ==============================================================================
# --- N-GRAM GCN GENERATION (Core Functions) ---
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
        if not (len(pdb_id) >= 5 and pdb_id[0] in 'OPQ' and pdb_id[1].isdigit()): return "PDB", f"{pdb_id}{'_' + chain_part if chain_part else ''}"
    plain_up_match = re.fullmatch(r"([A-Z0-9]{6,10}(?:-\d+)?)", hid.split()[0].split('|')[0])
    if plain_up_match: return "UniProt (assumed)", plain_up_match.group(1)
    first_word = hid.split()[0].split('|')[0]
    return ("Unknown", first_word) if first_word else ("Unknown", hid)


def parse_fasta_sequences_with_ids_ngram(filepath: str, config: ScriptConfig) -> list[tuple[str, str]]:
    protein_data = []
    if not os.path.exists(filepath):
        if config.DEBUG_VERBOSE: print(f"NgramGCN: FASTA file not found at {filepath}")
        return protein_data
    try:
        for record in SeqIO.parse(filepath, "fasta"):
            _, canonical_id = extract_canonical_id_and_type(record.id)
            if canonical_id:
                protein_data.append((canonical_id, str(record.seq).upper()))
            else:
                if config.DEBUG_VERBOSE: print(f"NgramGCN: Could not extract canonical ID from '{record.id[:50]}...', using full ID as fallback: {record.id}")
                protein_data.append((record.id, str(record.seq).upper()))
        if config.DEBUG_VERBOSE: print(f"NgramGCN: Parsed {len(protein_data)} sequences from {filepath}")
    except Exception as e:
        print(f"NgramGCN: Error parsing FASTA file {filepath}: {e}")
    return protein_data


def stream_ngram_chunks_from_fasta(fasta_path: str, n: int, chunk_size: int):
    ngrams_buffer = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) >= n:
            for i in range(len(seq) - n + 1):
                ngrams_buffer.append("".join(seq[i: i + n]))
                if len(ngrams_buffer) >= chunk_size:
                    yield pd.DataFrame(ngrams_buffer, columns=['ngram']).astype('string')
                    ngrams_buffer = []
    if ngrams_buffer:
        yield pd.DataFrame(ngrams_buffer, columns=['ngram']).astype('string')


def stream_transitions_from_fasta(fasta_path: str, n: int):
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) >= n + 1:
            current_seq_ngrams = ["".join(seq[i: i + n]) for i in range(len(seq) - n + 1)]
            for i in range(len(current_seq_ngrams) - 1):
                yield (current_seq_ngrams[i], current_seq_ngrams[i + 1])


def build_node_map_with_dask(fasta_path: str, n: int, output_parquet_path: str, chunk_size: int):
    if not DASK_AVAILABLE: raise ImportError("Dask is not installed.")
    print("Pass 1: Discovering unique n-grams with Dask...")
    lazy_chunks = [dask.delayed(chunk) for chunk in stream_ngram_chunks_from_fasta(fasta_path, n, chunk_size)]
    ddf = dd.from_delayed(lazy_chunks, meta={'ngram': 'string'})
    unique_ngrams_ddf = ddf.drop_duplicates().reset_index(drop=True)
    unique_ngrams_ddf['id'] = 1
    unique_ngrams_ddf['id'] = (unique_ngrams_ddf['id'].cumsum() - 1).astype('int64')
    print("Executing Dask computation and writing to Parquet...")
    unique_ngrams_ddf.to_parquet(output_parquet_path, engine='pyarrow', write_index=False, overwrite=True, compression=None)
    print(f"Pass 1 Complete. N-gram map saved to: {output_parquet_path}")


def build_edge_file_from_stream(fasta_path: str, n: int, ngram_to_idx_series: pd.Series, output_edge_path: str):
    print(f"Pass 2: Generating edge list and saving to {output_edge_path}...")
    with open(output_edge_path, 'w') as f:
        for source_ngram, target_ngram in tqdm(stream_transitions_from_fasta(fasta_path, n), desc="Generating edges"):
            source_id = ngram_to_idx_series.get(source_ngram)
            target_id = ngram_to_idx_series.get(target_ngram)
            if source_id is not None and target_id is not None:
                f.write(f"{int(source_id)},{int(target_id)}\n")
    print("Pass 2 Complete: Edge file has been created.")


def build_graph_from_disk(parquet_path: str, edge_file_path: str) -> Optional[Data]:
    print("Building final graph object from disk files...")
    if not os.path.exists(parquet_path) or not os.path.exists(edge_file_path):
        print("Error: Graph disk files not found.")
        return None
    map_df = pd.read_parquet(parquet_path)
    num_nodes = len(map_df)
    if num_nodes == 0: return None
    ngram_to_idx = pd.Series(map_df.id.values, index=map_df.ngram).to_dict()
    idx_to_ngram = {v: k for k, v in ngram_to_idx.items()}
    print(f"Reading the edge file from {edge_file_path} into memory...")
    edge_df = pd.read_csv(edge_file_path, header=None, names=['source', 'target'])
    print(f"Finished reading {len(edge_df)} total edges.")
    print("Aggregating edges to find unique transitions and their counts...")
    edge_counts = edge_df.groupby(['source', 'target']).size()
    unique_edges_df = edge_counts.reset_index(name='count')
    print(f"Found {len(unique_edges_df)} unique directed edges.")
    print("Calculating total outgoing transitions per source node...")
    source_outgoing_total_counts = edge_df.groupby('source').size()
    print("Calculating final edge weights...")
    source_totals_for_unique_edges = unique_edges_df['source'].map(source_outgoing_total_counts)
    transition_probabilities = unique_edges_df['count'] / source_totals_for_unique_edges
    edge_weights_tensor = torch.tensor(transition_probabilities.values, dtype=torch.float)
    source_nodes = torch.tensor(unique_edges_df['source'].values, dtype=torch.long)
    target_nodes = torch.tensor(unique_edges_df['target'].values, dtype=torch.long)
    directed_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    data = Data(num_nodes=num_nodes)
    data.ngram_to_idx = ngram_to_idx
    data.idx_to_ngram = idx_to_ngram
    data.edge_index_out = directed_edge_index
    data.edge_weight_out = edge_weights_tensor
    data.edge_index_in = directed_edge_index.flip(dims=[0])
    data.edge_weight_in = edge_weights_tensor
    data.edge_index = to_undirected(directed_edge_index, num_nodes=num_nodes)
    print("Building adjacency list for next-node task (this may take a moment)...")
    adj_for_next_node_task = edge_df.groupby('source')['target'].apply(list).to_dict()
    adj_for_next_node_task = defaultdict(list, adj_for_next_node_task)
    print("Assigning labels for next-node task...")
    y_next_node = torch.full((num_nodes,), -1, dtype=torch.long)
    for src_node, successors in adj_for_next_node_task.items():
        if successors: y_next_node[src_node] = random.choice(successors)
    data.y_next_node = y_next_node
    del edge_df, edge_counts, unique_edges_df, source_outgoing_total_counts
    gc.collect()
    print("Compressed graph object created successfully.")
    return data


class CustomDiGCNLayerPyG_ngram(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_nodes_for_coeffs: int, use_vector_coeffs: bool = True):
        super().__init__(aggr='add')
        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)
        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels))
        self.use_vector_coeffs = use_vector_coeffs
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
            nn.init.ones_(self.C_in_vec);
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in);
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
            c_in_final = c_in_final.view(-1).repeat(math.ceil(x.size(0) / c_in_final.size(0)))[:x.size(0)].view(-1, 1)
            c_out_final = c_out_final.view(-1).repeat(math.ceil(x.size(0) / c_out_final.size(0)))[:x.size(0)].view(-1, 1)
        output = c_in_final.to(x.device) * ic_combined + c_out_final.to(x.device) * oc_combined
        return output

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None and edge_weight.numel() > 0 else x_j


class ProtDiGCNEncoderDecoder_ngram(nn.Module):
    def __init__(self, num_initial_features: int, hidden_dim1: int, hidden_dim2: int, num_graph_nodes_for_gnn_coeffs: int, task_num_output_classes: int, n_gram_length_for_pe: int, one_gram_embed_dim_for_pe: int,
                 max_len_for_pe: int, dropout_rate: float, use_vector_coeffs_in_gnn: bool = True):
        super().__init__()
        self.n_gram_length_for_pe = n_gram_length_for_pe
        self.one_gram_embed_dim_for_pe = one_gram_embed_dim_for_pe
        self.dropout_rate = dropout_rate
        self.l2_norm_eps = 1e-12
        self.positional_encoder_layer = nn.Embedding(max_len_for_pe, self.one_gram_embed_dim_for_pe) if one_gram_embed_dim_for_pe > 0 and max_len_for_pe > 0 else None
        self.conv1 = CustomDiGCNLayerPyG_ngram(num_initial_features, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)
        self.conv2 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim1, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)
        self.conv3 = CustomDiGCNLayerPyG_ngram(hidden_dim1, hidden_dim2, num_graph_nodes_for_gnn_coeffs, use_vector_coeffs_in_gnn)
        self.residual_proj_1 = nn.Linear(num_initial_features, hidden_dim1) if num_initial_features != hidden_dim1 else nn.Identity()
        self.residual_proj_3 = nn.Linear(hidden_dim1, hidden_dim2) if hidden_dim1 != hidden_dim2 else nn.Identity()
        self.decoder_fc = nn.Linear(hidden_dim2, task_num_output_classes)

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_encoder_layer is None: return x
        x_pe = x.clone()
        try:
            x_reshaped = x_pe.view(-1, self.n_gram_length_for_pe, self.one_gram_embed_dim_for_pe)
        except RuntimeError:
            return x
        num_positions_to_encode = min(self.n_gram_length_for_pe, self.positional_encoder_layer.num_embeddings)
        if num_positions_to_encode > 0:
            position_indices = torch.arange(0, num_positions_to_encode, device=x.device, dtype=torch.long)
            pe_to_add = self.positional_encoder_layer(position_indices)
            x_reshaped[:, :num_positions_to_encode, :] = x_reshaped[:, :num_positions_to_encode, :] + pe_to_add.unsqueeze(0)
        return x_reshaped.view(-1, self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out = data.x, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out
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
        final_normalized_embedding_output = final_gcn_activated_output / (norm + self.l2_norm_eps)
        return F.log_softmax(task_logits, dim=-1), final_normalized_embedding_output


def train_ngram_model_full_graph(model: ProtDiGCNEncoderDecoder_ngram, data: Data, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str, config: ScriptConfig):
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = nn.NLLLoss()
    if task_mode == 'next_node':
        if not hasattr(data, 'y_next_node'):
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_next_node' attribute missing. Cannot train.")
            return
        targets = data.y_next_node
        train_mask = targets != -1
        if train_mask.sum() == 0:
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No valid training samples. Cannot train.")
            return
    elif task_mode == 'community_label':
        if not hasattr(data, 'y_task_labels'):
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_task_labels' attribute missing. Cannot train.")
            return
        targets = data.y_task_labels
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool)  # All nodes have a label
    else:
        print(f"NgramGCN: Unknown task_mode: {task_mode}. Cannot train.");
        return
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        log_probs, _ = model(data)
        loss = criterion(log_probs[train_mask], targets[train_mask])
        if loss is not None:
            loss.backward()
            optimizer.step()
            if epoch % (max(1, epochs // 20)) == 0 or epoch == epochs:
                if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Loss: {loss.item():.4f}")


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
        x = F.relu(self.conv1(x, edge_index))
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
        for batch in tqdm(inference_loader, desc="Batched Inference", leave=False):
            batch = batch.to(device)
            x_batch = full_graph_x[batch.n_id].to(device)
            x = F.relu(self.conv1(x_batch, batch.edge_index))
            x_final_gcn = self.conv2(x, batch.edge_index)
            norm = torch.norm(x_final_gcn, p=2, dim=1, keepdim=True)
            normalized_embeds = x_final_gcn / (norm + self.l2_norm_eps)
            all_embeds.append(normalized_embeds[:batch.batch_size].cpu())
        return torch.cat(all_embeds, dim=0)


def train_ngram_model_minibatch(model, loader: NeighborLoader, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str, config: ScriptConfig):
    model.train()
    model.to(device)
    criterion = nn.NLLLoss()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Mini-Batch Epoch {epoch:03d}", leave=False, disable=not config.DEBUG_VERBOSE)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_probs, _ = model(batch)
            log_probs_batch = log_probs[:batch.batch_size]
            loss = None
            if task_mode == 'community_label':
                loss = criterion(log_probs_batch, batch.y_task_labels[:batch.batch_size])
            elif task_mode == 'next_node':
                targets = batch.y_next_node[:batch.batch_size]
                valid_mask = targets != -1
                if valid_mask.sum() > 0: loss = criterion(log_probs_batch[valid_mask], targets[valid_mask])
            if loss:
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.batch_size
                pbar.set_postfix({'loss': loss.item()})
        if hasattr(loader, 'dataset') and len(loader.dataset) > 0:
            avg_loss = total_loss / len(loader.dataset)
            if (epoch % (max(1, epochs // 20)) == 0 or epoch == epochs) and config.DEBUG_VERBOSE:
                print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Avg Loss: {avg_loss:.4f}")


def extract_node_embeddings_ngram_batched(model, full_graph_data: Data, config: ScriptConfig, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    inference_loader = NeighborLoader(full_graph_data, num_neighbors=config.NGRAM_GCN_NUM_NEIGHBORS, batch_size=config.NGRAM_GCN_INFERENCE_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        all_node_embeddings = model.inference(full_graph_data.x.to(device), inference_loader, device)
    return all_node_embeddings.cpu().numpy() if all_node_embeddings is not None else None


def detect_communities_louvain(edge_index: torch.Tensor, num_nodes: int, config: ScriptConfig) -> Tuple[Optional[torch.Tensor], int]:
    if num_nodes == 0: return None, 0
    if edge_index.numel() == 0: return torch.arange(num_nodes, dtype=torch.long), num_nodes
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_index.cpu().numpy().T)
    if nx_graph.number_of_edges() == 0: return torch.arange(num_nodes, dtype=torch.long), num_nodes
    try:
        partition = community_louvain.best_partition(nx_graph, random_state=config.RANDOM_STATE)
        if not partition: return torch.arange(num_nodes, dtype=torch.long), num_nodes
        labels = torch.zeros(num_nodes, dtype=torch.long)
        for node, comm_id in partition.items(): labels[node] = comm_id
        num_communities = len(torch.unique(labels))
        if config.DEBUG_VERBOSE: print(f"NgramGCN Community: Detected {num_communities} communities.")
        return labels, num_communities
    except Exception as e:
        print(f"NgramGCN Community Error: {e}.")
        return torch.arange(num_nodes, dtype=torch.long), num_nodes


def extract_node_embeddings_ngram(model, data: Data, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data)
        return embeddings.cpu().numpy() if embeddings is not None and embeddings.numel() > 0 else None


def build_graph_files_for_level_n(n_val: int, config: ScriptConfig):
    if DASK_AVAILABLE: dask.config.set(scheduler='synchronous')
    print(f"[Worker n={n_val}]: Starting graph file construction.")
    try:
        parquet_path = os.path.join(config.TEMP_FILE_DIR, f"ngram_map_n{n_val}.parquet")
        build_node_map_with_dask(config.NGRAM_GCN_INPUT_FASTA_PATH, n_val, parquet_path, config.DASK_CHUNK_SIZE)
        edge_path = os.path.join(config.TEMP_FILE_DIR, f"edge_list_n{n_val}.txt")
        map_df = pd.read_parquet(parquet_path)
        ngram_map_series = pd.Series(map_df.id.values, index=map_df.ngram)
        del map_df;
        gc.collect()
        build_edge_file_from_stream(config.NGRAM_GCN_INPUT_FASTA_PATH, n_val, ngram_map_series, edge_path)
        print(f"[Worker n={n_val}]: Successfully completed graph file construction.")
        return n_val, True
    except Exception as e:
        print(f"[Worker n={n_val}]: FAILED with error: {e}")
        traceback.print_exc()
        return n_val, False


def generate_and_save_ngram_embeddings_sequential_training(config: ScriptConfig, protein_ids_to_generate_for: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Main orchestrator for the N-gramGCN embedding generation process.
    MODIFIED: Now includes more verbose logging about sequence mapping.
    """
    os.makedirs(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"NgramGCN: Using device: {device}", flush=True)

    all_protein_data = parse_fasta_sequences_with_ids_ngram(config.NGRAM_GCN_INPUT_FASTA_PATH, config)

    # --- NEW: More detailed logging to diagnose data overlap issues ---
    sequences_map = {pid: seq for pid, seq in all_protein_data if pid in protein_ids_to_generate_for}
    if not sequences_map:
        print("CRITICAL ERROR: No overlap found between proteins in the interaction files and proteins in the FASTA file.", flush=True)
        print(f"Interaction files require {len(protein_ids_to_generate_for)} proteins, but none were found in the parsed FASTA.", flush=True)
        return None, "No overlapping proteins between interaction data and FASTA file."
    else:
        print(f"Found {len(sequences_map)} matching proteins between interaction data and the FASTA file. Proceeding with training.", flush=True)

    level_embeddings: dict[int, np.ndarray] = {}
    level_ngram_to_idx: dict[int, dict] = {}
    level_idx_to_ngram: dict[int, dict] = {}
    per_protein_emb_path = os.path.join(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, f"per_protein_embeddings_from_{config.NGRAM_GCN_MAX_N}gram.h5")

    for n_val in range(1, config.NGRAM_GCN_MAX_N + 1):
        print(f"\n--- Processing N-gram Level: n = {n_val} ---")
        parquet_path = os.path.join(config.TEMP_FILE_DIR, f"ngram_map_n{n_val}.parquet")
        edge_path = os.path.join(config.TEMP_FILE_DIR, f"edge_list_n{n_val}.txt")
        print(f"Loading pre-built graph for n={n_val} from disk...")
        graph_data = build_graph_from_disk(parquet_path, edge_path)
        if graph_data is None: return None, "Graph loading failed."
        if config.DEBUG_VERBOSE: print(f"NgramGCN: Built graph for n={n_val}: {graph_data.num_nodes} nodes, {graph_data.edge_index_out.size(1)} out-edges.")
        level_ngram_to_idx[n_val] = graph_data.ngram_to_idx
        level_idx_to_ngram[n_val] = graph_data.idx_to_ngram
        task_mode = config.NGRAM_GCN_TASK_PER_LEVEL.get(n_val, config.NGRAM_GCN_DEFAULT_TASK_MODE)
        actual_num_output_classes = 0
        if task_mode == 'community_label':
            labels, num_comms = detect_communities_louvain(graph_data.edge_index, graph_data.num_nodes, config)
            if labels is not None and num_comms > 1:
                graph_data.y_task_labels = labels;
                actual_num_output_classes = num_comms
            else:
                task_mode = config.NGRAM_GCN_DEFAULT_TASK_MODE
        if task_mode == 'next_node':
            actual_num_output_classes = graph_data.num_nodes
        if actual_num_output_classes == 0: print(f"Cannot determine classes. Skipping."); break
        if n_val == 1:
            current_feature_dim = config.NGRAM_GCN_1GRAM_INIT_DIM
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)
        else:
            prev_embeds_np = level_embeddings.get(n_val - 1)
            prev_map = level_ngram_to_idx.get(n_val - 1)
            if prev_embeds_np is None or prev_map is None: print("Missing previous embeddings. Stopping."); break
            expected_concat_dim = 2 * prev_embeds_np.shape[1]
            current_idx_to_ngram_map = level_idx_to_ngram[n_val]
            features_list = []
            for i in tqdm(range(graph_data.num_nodes), desc=f"Generating features for n={n_val}"):
                ngram_str = current_idx_to_ngram_map.get(i)
                if ngram_str and len(ngram_str) == n_val:
                    idx1, idx2 = prev_map.get(ngram_str[:-1]), prev_map.get(ngram_str[1:])
                    if idx1 is not None and idx2 is not None:
                        features_list.append(torch.from_numpy(np.concatenate([prev_embeds_np[idx1], prev_embeds_np[idx2]])).float())
                    else:
                        features_list.append(torch.zeros(expected_concat_dim))
                else:
                    features_list.append(torch.zeros(expected_concat_dim))
            graph_data.x = torch.stack(features_list)
            current_feature_dim = graph_data.x.shape[1]
        node_embeddings_np = None
        if config.TRAINING_MODE == 'full_graph':
            model = ProtDiGCNEncoderDecoder_ngram(num_initial_features=current_feature_dim, hidden_dim1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_dim2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                  num_graph_nodes_for_gnn_coeffs=graph_data.num_nodes, task_num_output_classes=actual_num_output_classes, n_gram_length_for_pe=n_val,
                                                  one_gram_embed_dim_for_pe=(config.NGRAM_GCN_1GRAM_INIT_DIM if n_val == 1 else 0), max_len_for_pe=config.NGRAM_GCN_PE_MAX_LEN, dropout_rate=config.NGRAM_GCN_DROPOUT,
                                                  use_vector_coeffs_in_gnn=config.NGRAM_GCN_USE_VECTOR_COEFFS)
            if config.DEBUG_VERBOSE: print(f"Model (n={n_val}):\n{model}")
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)
            train_ngram_model_full_graph(model, graph_data, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)
        else:  # Minibatch
            model = ProtDiGCNEncoderDecoder_minibatch(in_channels=current_feature_dim, hidden_channels1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_channels2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                      out_channels=actual_num_output_classes, dropout=config.NGRAM_GCN_DROPOUT)
            if config.DEBUG_VERBOSE: print(f"Model (n={n_val}):\n{model}")
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)
            train_loader = NeighborLoader(graph_data, num_neighbors=config.NGRAM_GCN_NUM_NEIGHBORS, batch_size=config.NGRAM_GCN_BATCH_SIZE, shuffle=True)
            train_ngram_model_minibatch(model, train_loader, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram_batched(model, graph_data, config, device)
        if node_embeddings_np is None or node_embeddings_np.size == 0: print(f"Failed to generate embeddings for n={n_val}."); break
        level_embeddings[n_val] = node_embeddings_np
        del graph_data, model, optimizer, node_embeddings_np;
        gc.collect()
    final_embeddings = level_embeddings.get(config.NGRAM_GCN_MAX_N)
    final_map = level_ngram_to_idx.get(config.NGRAM_GCN_MAX_N)
    if final_embeddings is not None and final_map is not None:
        with h5py.File(per_protein_emb_path, 'w') as hf:
            print(f"Aggregating final embeddings...")
            for prot_id, seq in tqdm(sequences_map.items(), desc="Pooling Protein Embeddings"):
                if len(seq) >= config.NGRAM_GCN_MAX_N:
                    indices = [final_map.get("".join(seq[i:i + config.NGRAM_GCN_MAX_N])) for i in range(len(seq) - config.NGRAM_GCN_MAX_N + 1)]
                    valid_indices = [idx for idx in indices if idx is not None]
                    if valid_indices: hf.create_dataset(prot_id, data=np.mean(final_embeddings[valid_indices], axis=0))
        return per_protein_emb_path, None
    return None, f"Embeddings for n={config.NGRAM_GCN_MAX_N} not available."


# ==============================================================================
# --- LINK PREDICTION EVALUATION (Core Functions) ---
# ==============================================================================
def load_h5_embeddings_selectively(filepath: str, protein_ids: Set[str]) -> Dict[str, np.ndarray]:
    embeddings = {}
    try:
        with h5py.File(filepath, 'r') as hf:
            keys_to_load = [pid for pid in protein_ids if pid in hf]
            for prot_id in keys_to_load:
                embeddings[prot_id] = hf[prot_id][:]
    except Exception as e:
        print(f"Error loading H5 file {filepath}: {e}")
    return embeddings


def create_edge_embedding_generator(pairs_df: pd.DataFrame, embeddings: Dict[str, np.ndarray], batch_size: int, embed_method: str) -> Callable:
    num_samples = len(pairs_df)

    def generator():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_pairs = pairs_df.iloc[start:end]
            batch_embeds1 = [embeddings.get(p1) for p1 in batch_pairs['protein1']]
            batch_embeds2 = [embeddings.get(p2) for p2 in batch_pairs['protein2']]
            labels = batch_pairs['label'].values
            valid_indices = [i for i, (e1, e2) in enumerate(zip(batch_embeds1, batch_embeds2)) if e1 is not None and e2 is not None]
            if not valid_indices: continue
            valid_embeds1 = np.array([batch_embeds1[i] for i in valid_indices])
            valid_embeds2 = np.array([batch_embeds2[i] for i in valid_indices])
            valid_labels = labels[valid_indices]
            if embed_method == 'concatenate':
                edge_features = np.concatenate([valid_embeds1, valid_embeds2], axis=1)
            elif embed_method == 'hadamard':
                edge_features = valid_embeds1 * valid_embeds2
            elif embed_method == 'average':
                edge_features = (valid_embeds1 + valid_embeds2) / 2
            else:
                edge_features = np.concatenate([valid_embeds1, valid_embeds2], axis=1)
            yield edge_features, valid_labels

    return generator


def build_mlp_model_lp(input_dim: int, config: ScriptConfig) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(config.LP_MLP_DENSE1_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(inp)
    x = Dropout(config.LP_MLP_DROPOUT1_RATE)(x)
    x = Dense(config.LP_MLP_DENSE2_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(x)
    x = Dropout(config.LP_MLP_DROPOUT2_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    optimizer = Adam(learning_rate=config.LP_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc_keras')])
    return model


# ==============================================================================
# --- REPORTING, PLOTTING, AND ANALYSIS FUNCTIONS (Integrated) ---
# ==============================================================================
def plot_training_history(history_dict: dict, model_name: str, fold_num: Optional[int] = None):
    title_suffix = f" (Fold {fold_num})" if fold_num is not None else ""
    if not history_dict: return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict: plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss: {model_name}{title_suffix}');
    plt.xlabel('Epoch');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict: plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy: {model_name}{title_suffix}');
    plt.xlabel('Epoch');
    plt.legend();
    plt.grid(True)
    plt.suptitle(f"Training History: {model_name}", fontsize=16);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.show()


def plot_roc_curves(results_list: list):
    plt.figure(figsize=(10, 8))
    for result in results_list:
        if 'roc_data_representative' in result:
            fpr, tpr, auc_val = result['roc_data_representative']
            if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0:
                plt.plot(fpr, tpr, lw=2, label=f"{result['embedding_name']} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curves Comparison');
    plt.legend(loc="lower right");
    plt.grid(True);
    plt.show()


def plot_comparison_charts(results_list: list, config: ScriptConfig):
    metrics_to_plot = {'AUC': 'test_auc_sklearn', 'F1-Score': 'test_f1_sklearn', 'Precision': 'test_precision_sklearn', 'Recall': 'test_recall_sklearn'}
    for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY:
        metrics_to_plot[f'Hits@{k}'] = f'test_hits_at_{k}';
        metrics_to_plot[f'NDCG@{k}'] = f'test_ndcg_at_{k}'
    embedding_names = [res['embedding_name'] for res in results_list]
    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics)
    rows = math.ceil(num_metrics / cols)
    plt.figure(figsize=(cols * 6, rows * 5))
    for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        plt.subplot(rows, cols, i + 1)
        values = [res.get(metric_key, 0) for res in results_list]
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(embedding_names))))
        plt.ylabel('Score');
        plt.title(f'{metric_name} Comparison');
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, max(values) * 1.15 if any(v > 0 for v in values) else 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
    plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.show()


def print_results_table(results_list: list, config: ScriptConfig):
    if not results_list: return
    print("\n\n--- Overall Performance Comparison Table (Averaged over Folds) ---")
    headers = ["Embedding Name", "AUC", "F1", "Precision", "Recall"]
    for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY: headers.extend([f"Hits@{k}", f"NDCG@{k}"])
    headers.extend(["AUC StdDev", "F1 StdDev"])

    table_data = []
    for res in results_list:
        row = [res['embedding_name'], f"{res.get('test_auc_sklearn', 0):.4f}", f"{res.get('test_f1_sklearn', 0):.4f}", f"{res.get('test_precision_sklearn', 0):.4f}", f"{res.get('test_recall_sklearn', 0):.4f}"]
        for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY:
            row.append(f"{res.get(f'test_hits_at_{k}', 0)}")
            row.append(f"{res.get(f'test_ndcg_at_{k}', 0):.4f}")
        row.append(f"{res.get('test_auc_sklearn_std', 0):.4f}")
        row.append(f"{res.get('test_f1_sklearn_std', 0):.4f}")
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=headers)
    print(df.to_string(index=False))


def perform_statistical_tests(results_list: list, config: ScriptConfig):
    main_emb_name = config.LP_MAIN_EMBEDDING_NAME_FOR_STATS
    metric_key = config.LP_STATISTICAL_TEST_METRIC_KEY
    alpha = config.LP_STATISTICAL_TEST_ALPHA
    fold_scores_key = 'fold_auc_scores' if 'auc' in metric_key else 'fold_f1_scores'

    if len(results_list) < 2: return
    print(f"\n\n--- Statistical Comparison vs '{main_emb_name}' on '{metric_key}' (Alpha={alpha}) ---")

    main_res = next((r for r in results_list if r['embedding_name'] == main_emb_name), None)
    if not main_res or fold_scores_key not in main_res:
        print(f"Main model '{main_emb_name}' or its fold scores not found. Skipping tests.");
        return

    main_scores = main_res[fold_scores_key]
    print(f"{'Compared Embedding':<30} | {'p-value':<12} | {'Significant?':<15}")
    print("-" * 65)
    for other_res in [r for r in results_list if r['embedding_name'] != main_emb_name]:
        other_scores = other_res.get(fold_scores_key, [])
        if len(main_scores) != len(other_scores) or len(main_scores) == 0:
            print(f"{other_res['embedding_name']:<30} | {'N/A (score mismatch)':<12} | {'-':<15}")
            continue

        stat, p_value = wilcoxon(main_scores, other_scores)
        conclusion = "Yes" if p_value < alpha else "No"
        print(f"{other_res['embedding_name']:<30} | {p_value:<12.4f} | {conclusion:<15}")


# ==============================================================================
# --- MAIN EVALUATION WORKFLOW (Upgraded) ---
# ==============================================================================
def run_evaluation_for_one_embedding(config: ScriptConfig, eval_pairs_df: pd.DataFrame, emb_name: str, embeddings: Dict[str, np.ndarray]):
    if eval_pairs_df.empty: return None

    aggregated_results = {'embedding_name': emb_name, 'history_dict_fold1': {}, 'roc_data_representative': (None, None, 0.0)}
    skf = StratifiedKFold(n_splits=config.LP_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(eval_pairs_df, eval_pairs_df['label'])):
        print(f"\n-- Fold {fold + 1}/{config.LP_N_FOLDS} for {emb_name} --")
        train_df, test_df = eval_pairs_df.iloc[train_idx], eval_pairs_df.iloc[test_idx]

        embed_dim = next(iter(embeddings.values())).shape[0]
        input_dim = embed_dim * 2 if config.LP_EDGE_EMBEDDING_METHOD == 'concatenate' else embed_dim
        model = build_mlp_model_lp(input_dim, config)

        train_gen = create_edge_embedding_generator(train_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)()
        test_gen = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)()
        train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape([None, input_dim]), tf.TensorShape([None, ])))
        test_dataset = tf.data.Dataset.from_generator(lambda: test_gen, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape([None, input_dim]), tf.TensorShape([None, ])))

        history = model.fit(train_dataset, epochs=config.LP_EPOCHS, validation_data=test_dataset, verbose=1 if config.DEBUG_VERBOSE else 0)
        if fold == 0: aggregated_results['history_dict_fold1'] = history.history

        y_pred_probs = model.predict(test_dataset).flatten()
        y_true_gen = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)()
        y_true_aligned = np.concatenate([labels for _, labels in y_true_gen])

        current_fold_metrics = {}
        y_pred_class = (y_pred_probs > 0.5).astype(int)
        current_fold_metrics['test_precision_sklearn'] = precision_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_recall_sklearn'] = recall_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_f1_sklearn'] = f1_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_auc_sklearn'] = roc_auc_score(y_true_aligned, y_pred_probs) if len(np.unique(y_true_aligned)) > 1 else 0.5

        if fold == 0 and len(np.unique(y_true_aligned)) > 1:
            fpr, tpr, _ = roc_curve(y_true_aligned, y_pred_probs)
            aggregated_results['roc_data_representative'] = (fpr, tpr, current_fold_metrics['test_auc_sklearn'])

        desc_indices = np.argsort(y_pred_probs)[::-1]
        sorted_y_true = y_true_aligned[desc_indices]
        for k in config.LP_K_VALUES_FOR_RANKING_METRICS:
            eff_k = min(k, len(sorted_y_true))
            current_fold_metrics[f'test_hits_at_{k}'] = np.sum(sorted_y_true[:eff_k]) if eff_k > 0 else 0
            current_fold_metrics[f'test_ndcg_at_{k}'] = ndcg_score([y_true_aligned], [y_pred_probs], k=eff_k) if eff_k > 0 and len(np.unique(y_true_aligned)) > 1 else 0.0

        fold_metrics_list.append(current_fold_metrics)
        print(f"Fold {fold + 1} Results: AUC={current_fold_metrics['test_auc_sklearn']:.4f}, F1={current_fold_metrics['test_f1_sklearn']:.4f}")

    if not fold_metrics_list: return None

    # Aggregate results across folds
    for key in fold_metrics_list[0].keys():
        values = [d[key] for d in fold_metrics_list]
        aggregated_results[key] = np.mean(values)
        aggregated_results[f"{key}_std"] = np.std(values)

    aggregated_results['fold_auc_scores'] = [d.get('test_auc_sklearn', 0.0) for d in fold_metrics_list]
    aggregated_results['fold_f1_scores'] = [d.get('test_f1_sklearn', 0.0) for d in fold_metrics_list]

    return aggregated_results


# ==============================================================================
# --- MAIN EXECUTION BLOCK (Final Integrated Version) ---
# ==============================================================================
if __name__ == '__main__':

    print("--- Modular N-gramGCN and Link Prediction Script ---")
    config = ScriptConfig()

    # ==========================================================================
    # --- PART 1: N-GRAM GCN EMBEDDING GENERATION ---
    # ==========================================================================
    if config.RUN_NGRAM_GCN_GENERATION:
        print("\n" + "#" * 20 + " RUNNING: N-gramGCN Generation " + "#" * 20)
        print("\n" + "=" * 20 + " Phase 1: Parallel Graph Construction " + "=" * 20)
        os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
        levels_to_process = list(range(1, config.NGRAM_GCN_MAX_N + 1))

        if config.PARALLEL_CONSTRUCTION_WORKERS is None:
            num_workers = min(len(levels_to_process), max(1, multiprocessing.cpu_count() - 2))
            print(f"Auto-detecting workers: {num_workers} cores.")
        else:
            num_workers = config.PARALLEL_CONSTRUCTION_WORKERS
            print(f"Using user-specified workers: {num_workers}.")

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(build_graph_files_for_level_n, zip(levels_to_process, repeat(config)))

        # --- NEW: Added flush=True to ensure messages print immediately ---
        if not all(success for _, success in results):
            print("\nGraph construction failed. Aborting.", flush=True)
            sys.exit(1)
        else:
            print("\nAll graph files pre-built successfully.", flush=True)

        print("\n" + "=" * 20 + " Phase 2: Sequential Model Training " + "=" * 20, flush=True)
        try:
            pos_df_gcn = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, dtype=str).dropna()
            neg_df_gcn = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, dtype=str).dropna()
            proteins_in_pairs_for_gcn = set(pos_df_gcn.iloc[:, 0]) | set(pos_df_gcn.iloc[:, 1]) | set(neg_df_gcn.iloc[:, 0]) | set(neg_df_gcn.iloc[:, 1])
        except FileNotFoundError:
            print("Interaction files not found. Aborting.", flush=True)
            sys.exit(1)

        generated_prot_emb_path, error_msg = generate_and_save_ngram_embeddings_sequential_training(config, proteins_in_pairs_for_gcn)

        if generated_prot_emb_path:
            print(f"\nSUCCESS: N-gramGCN Generation complete.\nEmbeddings saved to: {generated_prot_emb_path}", flush=True)
            print("To evaluate, manually add path to 'LP_EMBEDDING_FILES_TO_EVALUATE' in config.", flush=True)
        else:
            print(f"\nFAILED: Training phase failed. Reason: {error_msg}", flush=True)
            sys.exit(1)

    if config.RUN_LINK_PREDICTION_EVALUATION:
        print("\n" + "#" * 20 + " RUNNING: Link Prediction Evaluation " + "#" * 20)
        try:
            all_pos_df = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()
            all_neg_df = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()
            all_pos_df['label'] = 1;
            all_neg_df['label'] = 0
            print(f"Loaded {len(all_pos_df)} total positive and {len(all_neg_df)} total negative candidate pairs.")
        except FileNotFoundError:
            print("Interaction CSV files not found. Aborting.");
            sys.exit(1)

        all_cv_results = []
        loader_map = {"load_h5_embeddings_selectively": load_h5_embeddings_selectively}

        for emb_config in config.LP_EMBEDDING_FILES_TO_EVALUATE:
            path = emb_config.get("path")
            name = emb_config.get("name", os.path.basename(str(path)) if path else "Unknown")
            loader_func = loader_map.get(emb_config.get("loader_func_key", "load_h5_embeddings_selectively"))

            print(f"\n--- Evaluating file: {name} ---")
            if not path or not os.path.exists(path): print(f"File not found: {path}. Skipping."); continue

            all_proteins_in_network = set(all_pos_df['protein1']) | set(all_pos_df['protein2']) | set(all_neg_df['protein1']) | set(all_neg_df['protein2'])
            protein_embeddings = loader_func(path, all_proteins_in_network)
            if not protein_embeddings: print(f"No relevant embeddings loaded. Skipping."); continue

            valid_pos_mask = all_pos_df['protein1'].isin(protein_embeddings.keys()) & all_pos_df['protein2'].isin(protein_embeddings.keys())
            valid_pos_df = all_pos_df[valid_pos_mask]
            valid_neg_mask = all_neg_df['protein1'].isin(protein_embeddings.keys()) & all_neg_df['protein2'].isin(protein_embeddings.keys())
            valid_neg_df = all_neg_df[valid_neg_mask]

            target_neg_count = int(len(valid_pos_df) / config.LP_DESIRED_POS_TO_NEG_RATIO)
            print(f"Found {len(valid_pos_df)} valid positive pairs. Sampling {target_neg_count} negative pairs from {len(valid_neg_df)} valid candidates.")

            if len(valid_neg_df) >= target_neg_count:
                sampled_neg_df = valid_neg_df.sample(n=target_neg_count, random_state=config.RANDOM_STATE)
            else:
                print(f"Warning: Not enough valid negative pairs. Using all {len(valid_neg_df)}.");
                sampled_neg_df = valid_neg_df

            eval_pairs_df = pd.concat([valid_pos_df, sampled_neg_df], ignore_index=True)
            eval_pairs_df = eval_pairs_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
            print(f"Created and shuffled evaluation set with {len(eval_pairs_df)} pairs.")

            results_for_emb = run_evaluation_for_one_embedding(config, eval_pairs_df, name, protein_embeddings)
            if results_for_emb:
                all_cv_results.append(results_for_emb)
                if config.LP_PLOT_TRAINING_HISTORY:
                    plot_training_history(results_for_emb.get('history_dict_fold1', {}), name, fold_num=1)

        if all_cv_results:
            print("\n\n" + "#" * 20 + " FINAL AGGREGATE RESULTS " + "#" * 20)
            print_results_table(all_cv_results, config)
            plot_comparison_charts(all_cv_results, config)
            plot_roc_curves(all_cv_results)
            perform_statistical_tests(all_cv_results, config)
        else:
            print("\nNo embeddings were successfully evaluated.")

    print("\n--- Script Finished ---")
