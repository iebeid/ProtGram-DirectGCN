# ==============================================================================
# Combined Protein-Protein Interaction Evaluation and N-gramGCN Embedding Generation Script
# VERSION: 3.0 (Main Block Refactored for Compatibility)
# ==============================================================================
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
import random
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

# Dask for memory-efficient graph construction
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client
    import pyarrow

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
        # This is the root directory where your input data (FASTA, interaction files) is located.
        self.BASE_DATA_DIR = "C:/ProgramData/ProtDiGCN/"
        # This is the root directory where all outputs will be saved.
        self.BASE_OUTPUT_DIR = "C:/ProgramData/ProtDiGCN/ppi_evaluation_results_final/"

        # --- TRAINING MODE SELECTION ---
        # 'full_graph': Original method, loading the entire graph. May learn better but uses more memory.
        # 'mini_batch': Uses NeighborLoader for memory efficiency. Necessary for large graphs.
        self.TRAINING_MODE = 'full_graph'  # OPTIONS: 'full_graph', 'mini_batch'

        # --- N-gramGCN Generation Configuration ---
        self.RUN_AND_EVALUATE_NGRAM_GCN = True
        self.NGRAM_GCN_INPUT_FASTA_PATH = os.path.join(self.BASE_DATA_DIR, "uniprot_sequences_sample.fasta")
        self.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings")
        self.NGRAM_GCN_MAX_N = 5
        self.NGRAM_GCN_1GRAM_INIT_DIM = 64
        self.NGRAM_GCN_HIDDEN_DIM_1 = 128
        self.NGRAM_GCN_HIDDEN_DIM_2 = 64
        self.NGRAM_GCN_PE_MAX_LEN = 10  # Max positional encoding length within an n-gram
        self.NGRAM_GCN_DROPOUT = 0.5
        self.NGRAM_GCN_LR = 0.001
        self.NGRAM_GCN_WEIGHT_DECAY = 1e-4
        self.NGRAM_GCN_EPOCHS_PER_LEVEL = 1000
        self.NGRAM_GCN_USE_VECTOR_COEFFS = True
        self.NGRAM_GCN_GENERATED_EMB_NAME = "NgramGCN-Generated"
        self.NGRAM_GCN_TASK_PER_LEVEL: Dict[int, str] = {1: 'community_label'}
        self.NGRAM_GCN_DEFAULT_TASK_MODE = 'next_node'

        # --- Mini-Batch Specific Configuration (only used if TRAINING_MODE = 'mini_batch') ---
        self.NGRAM_GCN_BATCH_SIZE = 512
        self.NGRAM_GCN_NUM_NEIGHBORS = [25, 15, 10]
        self.NGRAM_GCN_INFERENCE_BATCH_SIZE = 1024

        # --- Efficient Graph Construction Config ---
        self.DASK_CHUNK_SIZE = 2000000
        self.TEMP_FILE_DIR = os.path.join(self.BASE_OUTPUT_DIR, "temp")

        # --- Link Prediction Evaluation Configuration ---
        self.LP_POSITIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/positive_interactions.csv')
        self.LP_NEGATIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/negative_interactions.csv')
        self.LP_SAMPLE_NEGATIVE_PAIRS: Optional[int] = 20000
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [{"path": os.path.join(self.BASE_DATA_DIR, "models/per-protein.h5"), "name": "ProtT5-Precomputed", "loader_func_key": "load_h5_embeddings_selectively"}, ]
        self.LP_OUTPUT_MAIN_DIR = os.path.join(self.BASE_OUTPUT_DIR, "link_prediction_evaluation")
        self.LP_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.LP_N_FOLDS = 2
        self.LP_PLOT_TRAINING_HISTORY = True
        self.LP_MLP_DENSE1_UNITS = 128
        self.LP_MLP_DROPOUT1_RATE = 0.4
        self.LP_MLP_DENSE2_UNITS = 64
        self.LP_MLP_DROPOUT2_RATE = 0.4
        self.LP_MLP_L2_REG = 0.001
        self.LP_BATCH_SIZE = 64
        self.LP_EPOCHS = 10
        self.LP_LEARNING_RATE = 1e-3
        self.LP_K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
        self.LP_K_VALUES_FOR_TABLE_DISPLAY = [50, 100]
        self.LP_STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'
        self.LP_STATISTICAL_TEST_ALPHA = 0.05


# (The rest of the script, including all function definitions from the previous turn, remains unchanged here...)
# ==============================================================================
# START: ID PARSING & EFFICIENT GRAPH CONSTRUCTION
# ==============================================================================
def extract_canonical_id_and_type(header_or_id_line: str) -> tuple[Optional[str], Optional[str]]:
    """Extracts a canonical protein identifier (like a UniProt ID) from a FASTA header."""
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


def parse_fasta_sequences_with_ids_ngram(filepath: str, config: ScriptConfig) -> list[tuple[str, str]]:
    """Parses a FASTA file to extract canonical IDs and sequences."""
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
        if config.DEBUG_VERBOSE and not protein_data:
            print(f"NgramGCN: Warning - Parsed 0 sequences from {filepath}")
        elif config.DEBUG_VERBOSE:
            print(f"NgramGCN: Parsed {len(protein_data)} sequences with extracted/standardized IDs from {filepath}")
    except Exception as e:
        print(f"NgramGCN: Error parsing FASTA file {filepath}: {e}")
    return protein_data


def stream_ngram_chunks_from_fasta(fasta_path: str, n: int, chunk_size: int):
    """Yields chunks of n-grams from a FASTA file for memory-efficient processing."""
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
    """Yields n-gram to n-gram transitions from sequences in a FASTA file."""
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) >= n + 1:
            current_seq_ngrams = ["".join(seq[i: i + n]) for i in range(len(seq) - n + 1)]
            for i in range(len(current_seq_ngrams) - 1):
                yield (current_seq_ngrams[i], current_seq_ngrams[i + 1])


def build_node_map_with_dask(fasta_path: str, n: int, output_parquet_path: str, chunk_size: int):
    """
    Pass 1 of graph construction. Uses Dask to find all unique n-grams in a FASTA file
    and assigns a unique integer ID to each, saving the map to a Parquet file.
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask is required for graph construction but not installed.")

    client = None
    try:
        client = Client()
        print(f"Dask client dashboard available at: {client.dashboard_link}")
        print("Pass 1: Discovering unique n-grams with Dask...")
        lazy_chunks = [dask.delayed(chunk) for chunk in stream_ngram_chunks_from_fasta(fasta_path, n, chunk_size)]
        ddf = dd.from_delayed(lazy_chunks, meta={'ngram': 'string'})

        unique_ngrams_ddf = ddf.drop_duplicates().reset_index(drop=True)
        unique_ngrams_ddf['id'] = 1
        unique_ngrams_ddf['id'] = (unique_ngrams_ddf['id'].cumsum() - 1).astype('int64')

        print("Executing Dask computation and writing to Parquet...")
        unique_ngrams_ddf.to_parquet(output_parquet_path, engine='pyarrow', write_index=False, overwrite=True, compression=None)
        print(f"Pass 1 Complete. N-gram map saved to: {output_parquet_path}")
    finally:
        if client:
            client.close()


def build_edge_file_from_stream(fasta_path: str, n: int, ngram_to_idx_series: pd.Series, output_edge_path: str):
    """
    Pass 2 of graph construction. Streams n-gram transitions, converts them to integer IDs
    using the pre-computed map, and writes the edge list to a CSV file.
    """
    print(f"Pass 2: Generating edge list and saving to {output_edge_path}...")
    with open(output_edge_path, 'w') as f:
        for source_ngram, target_ngram in tqdm(stream_transitions_from_fasta(fasta_path, n), desc="Generating edges"):
            source_id = ngram_to_idx_series.get(source_ngram)
            target_id = ngram_to_idx_series.get(target_ngram)
            if source_id is not None and target_id is not None:
                f.write(f"{int(source_id)},{int(target_id)}\n")
    print("Pass 2 Complete: Edge file has been created.")


def build_graph_from_disk(parquet_path: str, edge_file_path: str) -> Optional[Data]:
    """
    Constructs the final PyTorch Geometric `Data` object from the n-gram map (Parquet)
    and the edge list (CSV). Calculates transition probabilities for edge weights.
    """
    print("Building final graph object from disk files...")
    if not os.path.exists(parquet_path) or not os.path.exists(edge_file_path):
        print("Error: Graph disk files not found.")
        return None

    map_df = pd.read_parquet(parquet_path)
    num_nodes = len(map_df)
    if num_nodes == 0: return None

    ngram_to_idx = pd.Series(map_df.id.values, index=map_df.ngram).to_dict()
    idx_to_ngram = {v: k for k, v in ngram_to_idx.items()}

    edge_df = pd.read_csv(edge_file_path, header=None, names=['source', 'target'])

    data = Data(num_nodes=num_nodes)
    data.ngram_to_idx = ngram_to_idx
    data.idx_to_ngram = idx_to_ngram

    source_nodes = torch.tensor(edge_df['source'].values, dtype=torch.long)
    target_nodes = torch.tensor(edge_df['target'].values, dtype=torch.long)
    directed_edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    # Efficiently calculate edge weights (transition probabilities)
    edge_counts = edge_df.groupby(['source', 'target']).size()
    source_outgoing_counts = edge_df.groupby('source').size()
    edge_weights_series = edge_counts / source_outgoing_counts.loc[edge_counts.index.get_level_values('source')].values
    edge_weights_df = edge_weights_series.reset_index(name='weight')

    # Align weights with the original edge_df order
    merged_df = pd.merge(edge_df.reset_index(), edge_weights_df, on=['source', 'target'], how='left')
    merged_df = merged_df.sort_values('index').set_index('index')

    edge_weights_tensor = torch.tensor(merged_df['weight'].values, dtype=torch.float)

    data.edge_index_out = directed_edge_index
    data.edge_weight_out = edge_weights_tensor
    data.edge_index_in = directed_edge_index.flip(dims=[0])
    data.edge_weight_in = edge_weights_tensor  # Assuming symmetric weights for in-edges for this model
    data.edge_index = to_undirected(directed_edge_index, num_nodes=num_nodes)

    # Prepare labels for the 'next_node' prediction task
    adj_for_next_node_task = defaultdict(list)
    for _, row in edge_df.iterrows():
        adj_for_next_node_task[row['source']].append(row['target'])

    y_next_node = torch.full((num_nodes,), -1, dtype=torch.long)
    for src_node, successors in adj_for_next_node_task.items():
        if successors: y_next_node[src_node] = random.choice(successors)
    data.y_next_node = y_next_node

    print("Graph object created successfully.")
    return data


# ==============================================================================
# START: FULL-GRAPH MODEL AND TRAINING (YOUR ORIGINAL IMPLEMENTATION)
# ==============================================================================

class CustomDiGCNLayerPyG_ngram(MessagePassing):
    """Custom Directed GCN Layer for the full-graph model."""

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

        # This logic handles potential size mismatches, especially during mini-batching if this layer were used there.
        if self.use_vector_coeffs and c_in_final.size(0) != x.size(0):
            # Fallback to repeating the coefficients if sizes don't match
            c_in_final = c_in_final.view(-1).repeat(math.ceil(x.size(0) / c_in_final.size(0)))[:x.size(0)].view(-1, 1)
            c_out_final = c_out_final.view(-1).repeat(math.ceil(x.size(0) / c_out_final.size(0)))[:x.size(0)].view(-1, 1)

        output = c_in_final.to(x.device) * ic_combined + c_out_final.to(x.device) * oc_combined
        return output

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None and edge_weight.numel() > 0 else x_j


class ProtDiGCNEncoderDecoder_ngram(nn.Module):
    """The main GCN model for full-graph training, incorporating residual connections and positional encoding."""

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
        """
        Applies learnable positional encoding to the input features.
        Crucially, it clones the input tensor to avoid in-place modification.
        """
        if self.positional_encoder_layer is None:
            return x

        # --- FIX: Clone the tensor to prevent modifying the original data.x ---
        x_pe = x.clone()

        # Reshape for PE: (num_nodes, n * one_gram_dim) -> (num_nodes, n, one_gram_dim)
        try:
            x_reshaped = x_pe.view(-1, self.n_gram_length_for_pe, self.one_gram_embed_dim_for_pe)
        except RuntimeError:
            # This can happen if the feature dimension doesn't match the expected PE dimensions,
            # especially for n > 1. In that case, we skip PE.
            return x

        # Get number of positions to encode, cannot exceed n-gram length or embedding layer size
        num_positions_to_encode = min(self.n_gram_length_for_pe, self.positional_encoder_layer.num_embeddings)

        if num_positions_to_encode > 0:
            position_indices = torch.arange(0, num_positions_to_encode, device=x.device, dtype=torch.long)
            pe_to_add = self.positional_encoder_layer(position_indices)  # Shape: (num_pos, one_gram_dim)

            # Add PE to the corresponding positions in each n-gram
            x_reshaped[:, :num_positions_to_encode, :] = x_reshaped[:, :num_positions_to_encode, :] + pe_to_add.unsqueeze(0)

        # Reshape back to original feature dimension
        return x_reshaped.view(-1, self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index_in, edge_weight_in = data.edge_index_in, data.edge_weight_in
        edge_index_out, edge_weight_out = data.edge_index_out, data.edge_weight_out

        x_pe = self._apply_positional_encoding(x)

        # Layer 1
        h1_conv_out = self.conv1(x_pe, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        x_proj1 = self.residual_proj_1(x_pe)
        h1_res_sum = x_proj1 + h1_conv_out
        h1_activated = F.tanh(h1_res_sum)
        h1 = F.dropout(h1_activated, p=self.dropout_rate, training=self.training)

        # Layer 2
        h2_conv_out = self.conv2(h1, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        h2_res_sum = h1 + h2_conv_out  # Residual connection
        h2_activated = F.tanh(h2_res_sum)
        h2 = F.dropout(h2_activated, p=self.dropout_rate, training=self.training)

        # Layer 3
        h3_conv_out = self.conv3(h2, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out)
        h2_proj3 = self.residual_proj_3(h2)
        h3_res_sum = h2_proj3 + h3_conv_out
        final_gcn_activated_output = F.tanh(h3_res_sum)

        # Decoder and Final Embeddings
        h_embed_for_decoder_dropped = F.dropout(final_gcn_activated_output, p=self.dropout_rate, training=self.training)
        task_logits = self.decoder_fc(h_embed_for_decoder_dropped)

        norm = torch.norm(final_gcn_activated_output, p=2, dim=1, keepdim=True)
        final_normalized_embedding_output = final_gcn_activated_output / (norm + self.l2_norm_eps)

        return F.log_softmax(task_logits, dim=-1), final_normalized_embedding_output


def train_ngram_model_full_graph(model: ProtDiGCNEncoderDecoder_ngram, data: Data, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str, config: ScriptConfig):
    """Training loop for the full-graph model."""
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = nn.NLLLoss()

    # Determine the target labels based on the task
    if task_mode == 'next_node':
        if not hasattr(data, 'y_next_node'):
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_next_node' attribute missing. Cannot train.")
            return
        targets = data.y_next_node
        train_mask = targets != -1  # Only train on nodes with successors
        if train_mask.sum() == 0:
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): No valid training samples. Cannot train.")
            return
    elif task_mode == 'community_label':
        if not hasattr(data, 'y_task_labels'):
            if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): 'y_task_labels' attribute missing. Cannot train.")
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
                if config.DEBUG_VERBOSE: print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Loss: {loss.item():.4f}")


# ==============================================================================
# START: MINI-BATCH MODEL AND TRAINING (NEW ALTERNATIVE)
# ==============================================================================
class ProtDiGCNEncoderDecoder_minibatch(nn.Module):
    """A standard GCN model designed for mini-batch training using PyG's NeighborLoader."""

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
    def inference(self, full_graph_x, inference_loader, device, config: ScriptConfig):
        """Performs batched inference to generate embeddings for all nodes in the graph."""
        self.eval()
        all_embeds = []
        for batch in tqdm(inference_loader, desc="Batched Inference", leave=False, disable=not config.DEBUG_VERBOSE):
            batch = batch.to(device)
            # We need the features of all nodes in the current computation graph (batch + neighbors)
            x_batch = full_graph_x[batch.n_id].to(device)

            # The model's forward pass is essentially run here
            x = self.conv1(x_batch, batch.edge_index)
            x = F.relu(x)
            x_final_gcn = self.conv2(x, batch.edge_index)

            norm = torch.norm(x_final_gcn, p=2, dim=1, keepdim=True)
            normalized_embeds = x_final_gcn / (norm + self.l2_norm_eps)

            # We only keep the embeddings for the root nodes of the batch
            all_embeds.append(normalized_embeds[:batch.batch_size].cpu())

        return torch.cat(all_embeds, dim=0)


def train_ngram_model_minibatch(model, loader: NeighborLoader, optimizer: optim.Optimizer, epochs: int, device: torch.device, task_mode: str, config: ScriptConfig):
    """Training loop for the mini-batch model."""
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
                if config.DEBUG_VERBOSE:
                    print(f"NgramGCN ({task_mode}): Epoch: {epoch:03d}, Avg Loss: {avg_loss:.4f}")


def extract_node_embeddings_ngram_batched(model, full_graph_data: Data, config: ScriptConfig, device: torch.device) -> Optional[np.ndarray]:
    """Wrapper function to perform batched inference for the mini-batch model."""
    model.eval()
    model.to(device)
    inference_loader = NeighborLoader(full_graph_data, num_neighbors=config.NGRAM_GCN_NUM_NEIGHBORS, batch_size=config.NGRAM_GCN_INFERENCE_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        all_node_embeddings = model.inference(full_graph_data.x.to(device), inference_loader, device, config)
    return all_node_embeddings.cpu().numpy() if all_node_embeddings is not None else None


# ==============================================================================
# START: GENERAL UTILITY FUNCTIONS
# ==============================================================================
def detect_communities_louvain(edge_index: torch.Tensor, num_nodes: int, config: ScriptConfig) -> Tuple[Optional[torch.Tensor], int]:
    """Detects node communities using the Louvain algorithm."""
    if num_nodes == 0: return None, 0
    if edge_index.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.long), num_nodes

    # Use the undirected graph for community detection
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_index.cpu().numpy().T)

    if nx_graph.number_of_edges() == 0:
        return torch.arange(num_nodes, dtype=torch.long), num_nodes

    try:
        partition = community_louvain.best_partition(nx_graph, random_state=config.RANDOM_STATE)
        if not partition: return torch.arange(num_nodes, dtype=torch.long), num_nodes

        labels = torch.zeros(num_nodes, dtype=torch.long)
        for node, comm_id in partition.items():
            labels[node] = comm_id

        num_communities = len(torch.unique(labels))
        if config.DEBUG_VERBOSE: print(f"NgramGCN Community: Detected {num_communities} communities.")
        return labels, num_communities
    except Exception as e:
        print(f"NgramGCN Community Error: {e}.")
        return torch.arange(num_nodes, dtype=torch.long), num_nodes


def extract_node_embeddings_ngram(model, data: Data, device: torch.device) -> Optional[np.ndarray]:
    """Extracts node embeddings from the full-graph model."""
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data)
        return embeddings.cpu().numpy() if embeddings is not None and embeddings.numel() > 0 else None


# ==============================================================================
# START: MAIN N-GRAM EMBEDDING GENERATION WORKFLOW
# ==============================================================================
def generate_and_save_ngram_embeddings(config: ScriptConfig, protein_ids_to_generate_for: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Main orchestrator for the N-gramGCN embedding generation process. It iterates from n=1 to max_n,
    building a graph, training a GCN model, and generating embeddings at each level.
    The embeddings from level n-1 are used as features for level n.

    Returns:
        The file path to the final per-protein H5 embedding file.
    """
    os.makedirs(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"NgramGCN: Using device: {device}")

    all_protein_data = parse_fasta_sequences_with_ids_ngram(config.NGRAM_GCN_INPUT_FASTA_PATH, config)
    sequences_map = {pid: seq for pid, seq in all_protein_data if pid in protein_ids_to_generate_for}

    level_embeddings: dict[int, np.ndarray] = {}
    level_ngram_to_idx: dict[int, dict] = {}
    level_idx_to_ngram: dict[int, dict] = {}
    per_protein_emb_path = os.path.join(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, f"per_protein_embeddings_from_{config.NGRAM_GCN_MAX_N}gram.h5")

    for n_val in range(1, config.NGRAM_GCN_MAX_N + 1):
        print(f"\n--- Processing N-gram Level: n = {n_val} ---")

        # --- Unified Efficient Graph Construction ---
        parquet_path = os.path.join(config.TEMP_FILE_DIR, f"ngram_map_n{n_val}.parquet")
        edge_path = os.path.join(config.TEMP_FILE_DIR, f"edge_list_n{n_val}.txt")
        build_node_map_with_dask(config.NGRAM_GCN_INPUT_FASTA_PATH, n_val, parquet_path, config.DASK_CHUNK_SIZE)

        map_df = pd.read_parquet(parquet_path)
        ngram_map_series = pd.Series(map_df.id.values, index=map_df.ngram)
        del map_df
        gc.collect()

        build_edge_file_from_stream(config.NGRAM_GCN_INPUT_FASTA_PATH, n_val, ngram_map_series, edge_path)
        graph_data = build_graph_from_disk(parquet_path, edge_path)

        if graph_data is None:
            print(f"NgramGCN: Could not build graph for n={n_val}. Stopping.")
            return None, "Graph construction failed."

        num_out_edges = graph_data.edge_index_out.size(1) if hasattr(graph_data, 'edge_index_out') and graph_data.edge_index_out is not None else 0
        if config.DEBUG_VERBOSE: print(f"NgramGCN: Built graph for n={n_val}: {graph_data.num_nodes} nodes, {num_out_edges} out-edges.")

        level_ngram_to_idx[n_val] = graph_data.ngram_to_idx
        level_idx_to_ngram[n_val] = graph_data.idx_to_ngram

        # --- Task and Feature Setup ---
        task_mode = config.NGRAM_GCN_TASK_PER_LEVEL.get(n_val, config.NGRAM_GCN_DEFAULT_TASK_MODE)
        actual_num_output_classes = 0
        if task_mode == 'community_label':
            # Use the undirected edge_index for Louvain
            labels, num_comms = detect_communities_louvain(graph_data.edge_index, graph_data.num_nodes, config)
            if labels is not None and num_comms > 1:
                graph_data.y_task_labels = labels
                actual_num_output_classes = num_comms
            else:
                print(f"NgramGCN: Community detection failed or found <= 1 community. Falling back to '{config.NGRAM_GCN_DEFAULT_TASK_MODE}' task.")
                task_mode = config.NGRAM_GCN_DEFAULT_TASK_MODE

        if task_mode == 'next_node':
            actual_num_output_classes = graph_data.num_nodes

        if actual_num_output_classes == 0:
            print(f"NgramGCN: Cannot determine number of classes for task '{task_mode}' at n={n_val}. Skipping.")
            break

        # --- Feature Generation ---
        if n_val == 1:
            current_feature_dim = config.NGRAM_GCN_1GRAM_INIT_DIM
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)
        else:
            prev_embeds_np = level_embeddings.get(n_val - 1)
            prev_map = level_ngram_to_idx.get(n_val - 1)
            prev_embed_dim = prev_embeds_np.shape[1]

            if prev_embeds_np is None or prev_map is None:
                print(f"NgramGCN: Cannot generate features for n={n_val} due to missing (n-1)-gram embeddings. Stopping.")
                break

            # Corrected logic: an n-gram is composed of two overlapping (n-1)-grams.
            expected_concat_dim = 2 * prev_embed_dim

            # Create features by concatenating the embeddings of the two (n-1)-grams that form the n-gram.
            current_idx_to_ngram_map = level_idx_to_ngram[n_val]

            features_list = []
            for i in tqdm(range(graph_data.num_nodes), desc=f"Generating features for n={n_val}"):
                ngram_str = current_idx_to_ngram_map.get(i)
                if ngram_str and len(ngram_str) == n_val:
                    sub_gram1 = ngram_str[:-1]
                    sub_gram2 = ngram_str[1:]

                    idx1 = prev_map.get(sub_gram1)
                    idx2 = prev_map.get(sub_gram2)

                    if idx1 is not None and idx2 is not None:
                        feat = np.concatenate([prev_embeds_np[idx1], prev_embeds_np[idx2]])
                        features_list.append(torch.from_numpy(feat).float())
                    else:
                        features_list.append(torch.zeros(expected_concat_dim))
                else:
                    features_list.append(torch.zeros(expected_concat_dim))

            graph_data.x = torch.stack(features_list)
            current_feature_dim = graph_data.x.shape[1]

        # --- MODEL & TRAINING PIPELINE SELECTION ---
        node_embeddings_np = None
        model = None

        if config.TRAINING_MODE == 'full_graph':
            model = ProtDiGCNEncoderDecoder_ngram(num_initial_features=current_feature_dim, hidden_dim1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_dim2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                  num_graph_nodes_for_gnn_coeffs=graph_data.num_nodes, task_num_output_classes=actual_num_output_classes, n_gram_length_for_pe=n_val,
                                                  one_gram_embed_dim_for_pe=(config.NGRAM_GCN_1GRAM_INIT_DIM if n_val == 1 else 0),  # PE only on 1-grams
                                                  max_len_for_pe=config.NGRAM_GCN_PE_MAX_LEN, dropout_rate=config.NGRAM_GCN_DROPOUT, use_vector_coeffs_in_gnn=config.NGRAM_GCN_USE_VECTOR_COEFFS)
            if config.DEBUG_VERBOSE:
                print(f"\nNgramGCN Model Architecture (for n={n_val}, task='{task_mode}', mode='full_graph'):")
                print(model)
                print("-" * 60)
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)
            train_ngram_model_full_graph(model, graph_data, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)

        else:  # 'mini_batch'
            model = ProtDiGCNEncoderDecoder_minibatch(in_channels=current_feature_dim, hidden_channels1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_channels2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                      out_channels=actual_num_output_classes, dropout=config.NGRAM_GCN_DROPOUT)
            if config.DEBUG_VERBOSE:
                print(f"\nNgramGCN Model Architecture (for n={n_val}, task='{task_mode}', mode='mini_batch'):")
                print(model)
                print("-" * 60)
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)

            train_loader = NeighborLoader(graph_data, num_neighbors=config.NGRAM_GCN_NUM_NEIGHBORS, batch_size=config.NGRAM_GCN_BATCH_SIZE, shuffle=True)
            train_ngram_model_minibatch(model, train_loader, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram_batched(model, graph_data, config, device)

        if node_embeddings_np is None or node_embeddings_np.size == 0:
            print(f"NgramGCN: Failed to generate embeddings for n={n_val}. Stopping.")
            break

        level_embeddings[n_val] = node_embeddings_np
        del graph_data, model, optimizer, node_embeddings_np
        gc.collect()

    # --- Final Per-Protein Embedding Aggregation ---
    final_embeddings = level_embeddings.get(config.NGRAM_GCN_MAX_N)
    final_map = level_ngram_to_idx.get(config.NGRAM_GCN_MAX_N)
    if final_embeddings is not None and final_map is not None and final_embeddings.size > 0:
        with h5py.File(per_protein_emb_path, 'w') as hf:
            print(f"Aggregating final {config.NGRAM_GCN_MAX_N}-gram embeddings to per-protein embeddings...")
            for prot_id, seq in tqdm(sequences_map.items(), desc="Pooling Protein Embeddings"):
                if len(seq) < config.NGRAM_GCN_MAX_N: continue

                indices = [final_map.get("".join(seq[i:i + config.NGRAM_GCN_MAX_N])) for i in range(len(seq) - config.NGRAM_GCN_MAX_N + 1)]
                valid_indices = [idx for idx in indices if idx is not None]

                if valid_indices:
                    # Mean pooling of n-gram embeddings for the protein
                    protein_embedding = np.mean(final_embeddings[valid_indices], axis=0)
                    hf.create_dataset(prot_id, data=protein_embedding)
        print(f"NgramGCN: Per-protein embeddings saved to {per_protein_emb_path}")
        return per_protein_emb_path, None
    else:
        error_msg = f"NgramGCN: Final embeddings for n={config.NGRAM_GCN_MAX_N} not available. No output file generated."
        print(error_msg)
        return None, error_msg


# ==============================================================================
# MEMORY-EFFICIENT LINK PREDICTION EVALUATION
# ==============================================================================

def load_h5_embeddings_selectively(filepath: str, protein_ids: Set[str]) -> Dict[str, np.ndarray]:
    """Loads specific protein embeddings from an H5 file."""
    embeddings = {}
    try:
        with h5py.File(filepath, 'r') as hf:
            loaded_ids = set(hf.keys())
            ids_to_load = list(protein_ids & loaded_ids)
            for prot_id in ids_to_load:
                embeddings[prot_id] = hf[prot_id][:]
    except Exception as e:
        print(f"Error loading H5 file {filepath}: {e}")
    return embeddings


def create_edge_embedding_generator(pairs_df: pd.DataFrame, embeddings: Dict[str, np.ndarray], batch_size: int, embed_method: str) -> Callable:
    """
    Creates a Python generator to yield batches of edge embeddings for training/testing the MLP.
    This is memory-efficient as it doesn't store all edge embeddings in memory at once.
    """
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
            else:  # Default to concatenate
                edge_features = np.concatenate([valid_embeds1, valid_embeds2], axis=1)

            yield edge_features, valid_labels

    return generator


def build_mlp_model_lp(input_dim: int, config: ScriptConfig) -> Model:
    """Builds the Keras MLP model for link prediction."""
    inp = Input(shape=(input_dim,))
    x = Dense(config.LP_MLP_DENSE1_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(inp)
    x = Dropout(config.LP_MLP_DROPOUT1_RATE)(x)
    x = Dense(config.LP_MLP_DENSE2_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(x)
    x = Dropout(config.LP_MLP_DROPOUT2_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    optimizer = Adam(learning_rate=config.LP_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_evaluation_for_one_embedding(config: ScriptConfig, all_pairs_df: pd.DataFrame, emb_name: str, embeddings: Dict[str, np.ndarray]):
    """
    Performs K-fold cross-validation for a single, pre-loaded set of embeddings.
    """
    # Filter pairs where both proteins have an embedding
    valid_pairs_mask = all_pairs_df['protein1'].isin(embeddings.keys()) & all_pairs_df['protein2'].isin(embeddings.keys())
    eval_pairs_df = all_pairs_df[valid_pairs_mask].reset_index(drop=True)

    print(f"Found embeddings for {len(eval_pairs_df)} / {len(all_pairs_df)} pairs in '{emb_name}'.")
    if eval_pairs_df.empty:
        print("No valid pairs with embeddings to evaluate. Skipping.")
        return None

    skf = StratifiedKFold(n_splits=config.LP_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(eval_pairs_df, eval_pairs_df['label'])):
        print(f"\n-- Fold {fold + 1}/{config.LP_N_FOLDS} --")
        train_df = eval_pairs_df.iloc[train_idx]
        test_df = eval_pairs_df.iloc[test_idx]

        # Get embedding dimension from the first available embedding
        embed_dim = next(iter(embeddings.values())).shape[0]
        input_dim = embed_dim * 2 if config.LP_EDGE_EMBEDDING_METHOD == 'concatenate' else embed_dim

        model = build_mlp_model_lp(input_dim, config)

        train_gen_func = create_edge_embedding_generator(train_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)
        test_gen_func = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)

        output_types = (tf.float32, tf.int32)
        output_shapes = (tf.TensorShape([None, input_dim]), tf.TensorShape([None, ]))

        train_dataset = tf.data.Dataset.from_generator(train_gen_func, output_types=output_types, output_shapes=output_shapes)
        test_dataset = tf.data.Dataset.from_generator(test_gen_func, output_types=output_types, output_shapes=output_shapes)

        history = model.fit(train_dataset, epochs=config.LP_EPOCHS, verbose=1 if config.DEBUG_VERBOSE else 0)

        # Evaluation on Test Set
        y_pred_probs = model.predict(test_dataset).flatten()

        # Since the generator skips invalid pairs, we must align predictions with true labels
        test_gen_for_labels = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)
        y_true_aligned = np.concatenate([labels for _, labels in test_gen_for_labels()])

        auc = roc_auc_score(y_true_aligned, y_pred_probs)
        precision = precision_score(y_true_aligned, y_pred_probs > 0.5)
        recall = recall_score(y_true_aligned, y_pred_probs > 0.5)
        f1 = f1_score(y_true_aligned, y_pred_probs > 0.5)

        fold_res = {'fold': fold, 'test_auc_sklearn': auc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}
        fold_results.append(fold_res)
        print(f"Fold {fold + 1} Results for '{emb_name}': AUC={auc:.4f}, F1={f1:.4f}")

    # Aggregate results for the current embedding
    avg_results = pd.DataFrame(fold_results).mean().to_dict()
    avg_results['embedding_name'] = emb_name
    return avg_results


# ==============================================================================
# MAIN EXECUTION BLOCK (REFACTORED)
# ==============================================================================
if __name__ == '__main__':
    print("--- Combined N-gramGCN Generation and Link Prediction Evaluation Script (Selectable Mode) ---")
    config = ScriptConfig()

    # --- 1. N-gramGCN Generation ---
    generated_prot_emb_path = None
    if config.RUN_AND_EVALUATE_NGRAM_GCN:
        print("\n" + "=" * 20 + " Step 1: N-gramGCN Generation " + "=" * 20)

        # To generate relevant embeddings, we first find which proteins are in our evaluation set
        try:
            pos_df_gcn = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, dtype=str).dropna()
            neg_df_gcn = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, dtype=str).dropna()
            proteins_in_pairs_for_gcn = set(pos_df_gcn.iloc[:, 0]) | set(pos_df_gcn.iloc[:, 1]) | set(neg_df_gcn.iloc[:, 0]) | set(neg_df_gcn.iloc[:, 1])
        except FileNotFoundError:
            print("Interaction files not found. Cannot determine which proteins to generate. Aborting GCN step.")
            proteins_in_pairs_for_gcn = set()

        if not proteins_in_pairs_for_gcn:
            print("Could not identify any proteins from interaction files. Aborting GCN training.")
        else:
            print(f"Found {len(proteins_in_pairs_for_gcn)} unique proteins in interaction data to target for embedding generation.")
            generated_prot_emb_path, error_msg = generate_and_save_ngram_embeddings(config=config, protein_ids_to_generate_for=proteins_in_pairs_for_gcn)

            if generated_prot_emb_path and os.path.exists(generated_prot_emb_path):
                print(f"N-gramGCN embedding generation successful. File saved to: {generated_prot_emb_path}")
                # Prepend the newly generated embeddings to the list of files to be evaluated
                new_embedding_info = {"path": generated_prot_emb_path, "name": config.NGRAM_GCN_GENERATED_EMB_NAME, "loader_func_key": "load_h5_embeddings_selectively"}
                config.LP_EMBEDDING_FILES_TO_EVALUATE.insert(0, new_embedding_info)
            else:
                print(f"N-gramGCN embedding generation failed. Reason: {error_msg}")

    # --- 2. Load Interaction Data for Evaluation ---
    print("\n" + "=" * 20 + " Step 2: Loading Interaction Data " + "=" * 20)
    all_interaction_pairs_df = None
    all_proteins_in_pairs = set()
    try:
        pos_df = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()
        neg_df = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()

        if config.LP_SAMPLE_NEGATIVE_PAIRS and len(neg_df) > 0:
            neg_df = neg_df.sample(n=min(config.LP_SAMPLE_NEGATIVE_PAIRS, len(neg_df)), random_state=config.RANDOM_STATE)

        pos_df['label'] = 1
        neg_df['label'] = 0
        all_interaction_pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
        all_proteins_in_pairs = set(all_interaction_pairs_df['protein1']) | set(all_interaction_pairs_df['protein2'])
        print(f"Loaded {len(pos_df)} positive and {len(neg_df)} negative pairs.")
        print(f"Found {len(all_proteins_in_pairs)} unique proteins required for link prediction.")
    except FileNotFoundError:
        print("Interaction CSV files not found. Cannot run link prediction evaluation.")

    # --- 3. Gatekeeper Check ---
    print("\n" + "=" * 20 + " Step 3: Gatekeeper Check " + "=" * 20)
    can_proceed_with_evaluation = False
    if all_interaction_pairs_df is None or all_interaction_pairs_df.empty:
        print("No interaction data loaded. Skipping evaluation.")
    elif config.RUN_AND_EVALUATE_NGRAM_GCN and not (generated_prot_emb_path and os.path.exists(generated_prot_emb_path)):
        print("N-gram GCN was set to run but its embedding file was not generated. Skipping all evaluations.")
    elif config.RUN_AND_EVALUATE_NGRAM_GCN:
        print(f"Checking relevance of generated N-gram GCN embeddings at: {generated_prot_emb_path}")
        with h5py.File(generated_prot_emb_path, 'r') as hf:
            relevant_ids_found = {pid for pid in all_proteins_in_pairs if pid in hf}

        if relevant_ids_found:
            print(f"Found {len(relevant_ids_found)} relevant proteins in the N-gram GCN embeddings. Proceeding with all evaluations.")
            can_proceed_with_evaluation = True
        else:
            print("No overlap found between interaction proteins and generated N-gram GCN embeddings. Skipping all link prediction evaluations.")
    else:
        print("N-gram GCN generation was disabled. Proceeding with evaluation of pre-existing files.")
        can_proceed_with_evaluation = True

    # --- 4. Conditional Link Prediction Evaluation ---
    if can_proceed_with_evaluation:
        print("\n" + "=" * 20 + " Step 4: Link Prediction Evaluation " + "=" * 20)
        all_results = []
        embedding_loader_map = {"load_h5_embeddings_selectively": load_h5_embeddings_selectively}

        for emb_config in config.LP_EMBEDDING_FILES_TO_EVALUATE:
            emb_path = emb_config['path']
            emb_name = emb_config['name']
            loader_func = embedding_loader_map[emb_config['loader_func_key']]

            print(f"\n--- Evaluating file: {emb_name} from {emb_path} ---")

            if not os.path.exists(emb_path):
                print("File not found. Skipping.")
                continue

            try:
                protein_embeddings = loader_func(emb_path, all_proteins_in_pairs)
            except Exception as e:
                print(f"Could not load embeddings from {emb_path}: {e}")
                continue

            if not protein_embeddings:
                print(f"No relevant embeddings for interaction pairs found in this file. Skipping.")
                continue

            # Run evaluation for this specific embedding file
            results_for_emb = run_evaluation_for_one_embedding(config, all_interaction_pairs_df, emb_name, protein_embeddings)
            if results_for_emb:
                all_results.append(results_for_emb)

        # Display Final Summary
        if all_results:
            results_df = pd.DataFrame(all_results)
            print("\n\n--- Final Link Prediction Evaluation Summary ---")
            print(results_df[['embedding_name', 'test_auc_sklearn', 'test_f1', 'test_precision', 'test_recall']].round(4))
            print("-" * 50)
        else:
            print("\nNo embeddings were successfully evaluated.")

    print("\n--- Script Finished ---")
