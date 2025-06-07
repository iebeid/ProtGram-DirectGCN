# ==============================================================================
# SCRIPT 2: N-gram GCN Model Training
# PURPOSE: Loads pre-built graphs, trains the model, and saves embeddings.
# VERSION: 2.0 (Complete, No Placeholders)
# ==============================================================================

import os
import sys
import gc
import h5py
import math
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict
from Bio import SeqIO
import networkx as nx
import community as community_louvain


# ==============================================================================
# --- CONFIGURATION CLASS ---
# ==============================================================================
class ScriptConfig:
    """
    Centralized configuration class for the model training script.
    """

    def __init__(self):
        # --- GENERAL SETTINGS ---
        self.DEBUG_VERBOSE = True
        self.RANDOM_STATE = 42

        # !!! IMPORTANT: SET YOUR BASE DIRECTORIES HERE !!!
        self.BASE_DATA_DIR = "C:/ProgramData/ProtDiGCN/"
        self.BASE_OUTPUT_DIR = os.path.join(self.BASE_DATA_DIR, "ppi_evaluation_results_final_dummy")

        # --- INPUT DIRECTORIES ---
        self.GRAPH_INPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, "constructed_graphs")
        self.TEMP_FILE_DIR = os.path.join(self.BASE_OUTPUT_DIR, "temp_graph_files")  # Needed for raw edge lists
        self.NGRAM_GCN_INPUT_FASTA_PATH = os.path.join(self.BASE_DATA_DIR, "uniprot_sequences_sample.fasta")
        self.LP_POSITIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/positive_interactions.csv')
        self.LP_NEGATIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/negative_interactions.csv')

        # --- OUTPUT DIRECTORY ---
        self.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings")

        # --- N-gramGCN Training Configuration ---
        self.TRAINING_MODE = 'full_graph'  # 'full_graph' or 'minibatch'
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
        self.NGRAM_GCN_TASK_PER_LEVEL: Dict[int, str] = {1: 'community_label'}
        self.NGRAM_GCN_DEFAULT_TASK_MODE = 'next_node'
        self.NGRAM_GCN_BATCH_SIZE = 512
        self.NGRAM_GCN_NUM_NEIGHBORS = [25, 15, 10]
        self.NGRAM_GCN_INFERENCE_BATCH_SIZE = 1024


# ==============================================================================
# --- PYTORCH MODELS & LAYERS ---
# ==============================================================================

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


# ==============================================================================
# --- TRAINING, HELPER, and UTILITY FUNCTIONS ---
# ==============================================================================

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


def extract_node_embeddings_ngram(model, data: Data, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data)
        return embeddings.cpu().numpy() if embeddings is not None and embeddings.numel() > 0 else None


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


def load_and_prepare_graph_for_training(n_val: int, config: ScriptConfig) -> Optional[Data]:
    """
    Loads the basic graph object from Script 1 and enriches it with the
    detailed edge/label information required for GCN training.
    """
    print(f"Loading and preparing graph for n={n_val}...")
    base_graph_path = os.path.join(config.GRAPH_INPUT_DIR, f"graph_n{n_val}.pt")
    edge_file_path = os.path.join(config.TEMP_FILE_DIR, f"edge_list_n{n_val}.txt")

    if not os.path.exists(base_graph_path) or not os.path.exists(edge_file_path):
        print(f"Error: Missing required files for n={n_val}. Cannot prepare graph.")
        return None

    # Load the base graph object with nodes and n-gram maps
    data = torch.load(base_graph_path)
    num_nodes = data.num_nodes

    # Load the raw edge list and process it to get weights, directed edges, etc.
    print(f"Reading the edge file from {edge_file_path} to build training graph...")
    edge_df = pd.read_csv(edge_file_path, header=None, names=['source', 'target'])
    print(f"Aggregating {len(edge_df)} total edges...")
    edge_counts = edge_df.groupby(['source', 'target']).size()
    unique_edges_df = edge_counts.reset_index(name='count')

    source_outgoing_total_counts = edge_df.groupby('source').size()
    source_totals_for_unique_edges = unique_edges_df['source'].map(source_outgoing_total_counts)
    transition_probabilities = unique_edges_df['count'] / source_totals_for_unique_edges

    edge_weights_tensor = torch.tensor(transition_probabilities.values, dtype=torch.float)
    source_nodes = torch.tensor(unique_edges_df['source'].values, dtype=torch.long)
    target_nodes = torch.tensor(unique_edges_df['target'].values, dtype=torch.long)
    directed_edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    # Add the required attributes to the Data object
    data.edge_index_out = directed_edge_index
    data.edge_weight_out = edge_weights_tensor
    data.edge_index_in = directed_edge_index.flip(dims=[0])
    data.edge_weight_in = edge_weights_tensor
    # Note: data.edge_index (undirected) is already present from Script 1

    # Create labels for the 'next_node' prediction task
    adj_for_next_node_task = defaultdict(list, edge_df.groupby('source')['target'].apply(list).to_dict())
    y_next_node = torch.full((num_nodes,), -1, dtype=torch.long)
    for src_node, successors in adj_for_next_node_task.items():
        if successors: y_next_node[src_node] = random.choice(successors)
    data.y_next_node = y_next_node

    del edge_df, edge_counts, unique_edges_df, source_outgoing_total_counts
    gc.collect()
    print("Graph fully prepared for training.")
    return data


# ==============================================================================
# --- MAIN TRAINING ORCHESTRATOR ---
# ==============================================================================

def train_and_generate_embeddings(config: ScriptConfig, protein_ids_to_generate_for: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Main orchestrator for the N-gramGCN embedding generation process.
    """
    os.makedirs(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    all_protein_data = parse_fasta_sequences_with_ids_ngram(config.NGRAM_GCN_INPUT_FASTA_PATH, config)
    sequences_map = {pid: seq for pid, seq in all_protein_data if pid in protein_ids_to_generate_for}

    if not sequences_map:
        print("CRITICAL ERROR: No overlap found between proteins in interaction files and FASTA file.")
        return None, "No overlapping proteins found."

    level_embeddings: dict[int, np.ndarray] = {}
    level_ngram_to_idx: dict[int, dict] = {}
    per_protein_emb_path = os.path.join(config.NGRAM_GCN_OUTPUT_EMBEDDINGS_DIR, f"per_protein_embeddings_from_{config.NGRAM_GCN_MAX_N}gram.h5")

    for n_val in range(1, config.NGRAM_GCN_MAX_N + 1):
        print(f"\n--- Processing N-gram Level: n = {n_val} ---")

        # MODIFIED PART: Load pre-built and prepared graph
        graph_data = load_and_prepare_graph_for_training(n_val, config)
        if graph_data is None: return None, f"Graph loading failed for n={n_val}."

        level_ngram_to_idx[n_val] = graph_data.ngram_to_idx

        # Determine task for this level
        task_mode = config.NGRAM_GCN_TASK_PER_LEVEL.get(n_val, config.NGRAM_GCN_DEFAULT_TASK_MODE)
        actual_num_output_classes = 0
        if task_mode == 'community_label':
            undirected_edge_index = to_undirected(graph_data.edge_index_out, num_nodes=graph_data.num_nodes)
            labels, num_comms = detect_communities_louvain(undirected_edge_index, graph_data.num_nodes, config)
            if labels is not None and num_comms > 1:
                graph_data.y_task_labels = labels;
                actual_num_output_classes = num_comms
            else:
                task_mode = config.NGRAM_GCN_DEFAULT_TASK_MODE

        if task_mode == 'next_node':
            actual_num_output_classes = graph_data.num_nodes

        if actual_num_output_classes == 0:
            print("Cannot determine output classes for task. Skipping training.");
            break

        # Prepare initial features
        if n_val == 1:
            current_feature_dim = config.NGRAM_GCN_1GRAM_INIT_DIM
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)
        else:
            prev_embeds_np = level_embeddings.get(n_val - 1)
            prev_map = level_ngram_to_idx.get(n_val - 1)
            if prev_embeds_np is None or prev_map is None: break
            expected_concat_dim = 2 * prev_embeds_np.shape[1]
            current_idx_to_ngram_map = graph_data.idx_to_ngram
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

        # Select training mode and run
        node_embeddings_np = None
        if config.TRAINING_MODE == 'full_graph':
            model = ProtDiGCNEncoderDecoder_ngram(num_initial_features=current_feature_dim, hidden_dim1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_dim2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                  num_graph_nodes_for_gnn_coeffs=graph_data.num_nodes, task_num_output_classes=actual_num_output_classes, n_gram_length_for_pe=n_val,
                                                  one_gram_embed_dim_for_pe=(config.NGRAM_GCN_1GRAM_INIT_DIM if n_val == 1 else 0), max_len_for_pe=config.NGRAM_GCN_PE_MAX_LEN, dropout_rate=config.NGRAM_GCN_DROPOUT,
                                                  use_vector_coeffs_in_gnn=config.NGRAM_GCN_USE_VECTOR_COEFFS)
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)
            train_ngram_model_full_graph(model, graph_data, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram(model, graph_data, device)
        else:  # Minibatch
            model = ProtDiGCNEncoderDecoder_minibatch(in_channels=current_feature_dim, hidden_channels1=config.NGRAM_GCN_HIDDEN_DIM_1, hidden_channels2=config.NGRAM_GCN_HIDDEN_DIM_2,
                                                      out_channels=actual_num_output_classes, dropout=config.NGRAM_GCN_DROPOUT)
            optimizer = optim.Adam(model.parameters(), lr=config.NGRAM_GCN_LR, weight_decay=config.NGRAM_GCN_WEIGHT_DECAY)
            train_loader = NeighborLoader(graph_data, num_neighbors=config.NGRAM_GCN_NUM_NEIGHBORS, batch_size=config.NGRAM_GCN_BATCH_SIZE, shuffle=True)
            train_ngram_model_minibatch(model, train_loader, optimizer, config.NGRAM_GCN_EPOCHS_PER_LEVEL, device, task_mode, config)
            node_embeddings_np = extract_node_embeddings_ngram_batched(model, graph_data, config, device)

        if node_embeddings_np is None or node_embeddings_np.size == 0:
            print(f"Failed to generate embeddings for n={n_val}.");
            break
        level_embeddings[n_val] = node_embeddings_np
        del graph_data, model, optimizer, node_embeddings_np;
        gc.collect()

    # Save final protein embeddings
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
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    print("--- SCRIPT 2: N-gram GCN Model Training ---")
    config = ScriptConfig()

    print("\n" + "=" * 20 + " Phase 1: Loading Protein Set for Generation " + "=" * 20, flush=True)
    try:
        # Assuming headerless CSVs with protein IDs in the first two columns
        pos_df_gcn = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, header=None, dtype=str).dropna()
        neg_df_gcn = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, header=None, dtype=str).dropna()
        proteins_in_pairs_for_gcn = set(pos_df_gcn.iloc[:, 0]) | set(pos_df_gcn.iloc[:, 1]) | set(neg_df_gcn.iloc[:, 0]) | set(neg_df_gcn.iloc[:, 1])
        print(f"Found {len(proteins_in_pairs_for_gcn)} unique proteins in interaction files.")
    except FileNotFoundError:
        print("Interaction files not found. Aborting.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading interaction files: {e}. Aborting.")
        sys.exit(1)

    print("\n" + "=" * 20 + " Phase 2: Sequential Model Training " + "=" * 20, flush=True)
    generated_prot_emb_path, error_msg = train_and_generate_embeddings(config, proteins_in_pairs_for_gcn)

    if generated_prot_emb_path:
        print(f"\nSUCCESS: N-gramGCN training complete.")
        print(f"Embeddings saved to: {generated_prot_emb_path}")
    else:
        print(f"\nFAILED: Training phase failed. Reason: {error_msg}")
        sys.exit(1)

    print("\n--- Script 2 Finished ---")
