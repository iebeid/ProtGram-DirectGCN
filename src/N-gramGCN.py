import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import degree # Not explicitly used in the GNN layer provided, but good for PyG
import numpy as np
from collections import Counter, defaultdict
import math
import os
import h5py
from Bio import SeqIO  # Standard library for FASTA parsing
from tqdm import tqdm  # For progress bars
from typing import Optional

# --- Configuration Constants (assuming these are defined elsewhere or default) ---
DEFAULT_EMBEDDING_DIM = 256  # Matching your output for 1-gram
DEFAULT_HIDDEN_DIM_1 = 512
DEFAULT_HIDDEN_DIM_2 = 256  # Matching your output for 1-gram to make sense for next layer
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_EPOCHS_PER_NGRAM_MODEL = 5  # From your output
MAX_NRAM_LEN_FOR_PE = 10


# --- 1. FASTA Parsing (Modified to return IDs) ---
def parse_fasta_sequences_with_ids(filepath: str) -> list[tuple[str, str]]:
    """Parses a FASTA file and returns a list of (protein_id, sequence) tuples."""
    protein_data = []
    if not os.path.exists(filepath):
        print(f"Error: FASTA file not found at {filepath}")
        return protein_data
    try:
        for record in SeqIO.parse(filepath, "fasta"):
            protein_data.append((record.id, str(record.seq).upper()))  # Store ID and sequence
        print(f"Parsed {len(protein_data)} sequences with IDs from {filepath}")
    except Exception as e:
        print(f"Error parsing FASTA file {filepath}: {e}")
    return protein_data


# --- 2. N-gram Graph Construction (Modified to return idx_to_ngram_map) ---
def get_ngrams_and_transitions(sequences: list[str], n: int):
    """
    Extracts n-grams and their transitions from a list of sequences.
    An n-gram is a tuple of characters.
    A transition is (ngram_source, ngram_target).
    """
    all_ngrams = []
    all_transitions = []
    for seq in sequences:
        if len(seq) < n:
            continue

        current_seq_ngrams = []
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i: i + n])
            current_seq_ngrams.append(ngram)
            all_ngrams.append(ngram)

        for i in range(len(current_seq_ngrams) - 1):
            source_ngram = current_seq_ngrams[i]
            target_ngram = current_seq_ngrams[i + 1]
            all_transitions.append((source_ngram, target_ngram))

    return all_ngrams, all_transitions


def build_ngram_graph_data(
        ngrams: list[tuple],
        transitions: list[tuple[tuple, tuple]],
        node_prob_from_prev_graph: Optional[dict[tuple, float]] = None,
        n_val: int = 1
) -> tuple[Optional[Data], dict[tuple, int], dict[int, tuple], dict[tuple, float]]:
    """
    Builds a PyTorch Geometric Data object for an n-gram graph.
    Returns: Data object, ngram_to_idx map, idx_to_ngram map, ngram_probabilities.
    """
    if not ngrams:
        return None, {}, {}, {}

    unique_ngrams_list = sorted(list(set(ngrams)))
    ngram_to_idx = {ngram: i for i, ngram in enumerate(unique_ngrams_list)}
    idx_to_ngram = {i: ngram for ngram, i in ngram_to_idx.items()}  # Store this mapping
    num_nodes = len(unique_ngrams_list)

    ngram_counts = Counter(ngrams)
    ngram_probabilities = {
        ngram: count / len(ngrams) for ngram, count in ngram_counts.items()
    }

    source_nodes_idx, target_nodes_idx, edge_weights = [], [], []
    edge_counts = Counter(transitions)
    source_ngram_outgoing_counts = defaultdict(int)
    for (src_ng, _), count in edge_counts.items():
        source_ngram_outgoing_counts[src_ng] += count

    for (source_ngram, target_ngram), count in edge_counts.items():
        if source_ngram in ngram_to_idx and target_ngram in ngram_to_idx:
            source_idx = ngram_to_idx[source_ngram]
            target_idx = ngram_to_idx[target_ngram]
            transition_prob = count / source_ngram_outgoing_counts[source_ngram] if source_ngram_outgoing_counts[source_ngram] > 0 else 0.0
            current_edge_weight = transition_prob
            # Placeholder for incorporating node_prob_from_prev_graph if logic is defined
            # if n_val > 1 and node_prob_from_prev_graph:
            #     prefix_ngram = source_ngram[:-1]
            #     if prefix_ngram in node_prob_from_prev_graph:
            #         current_edge_weight *= node_prob_from_prev_graph[prefix_ngram]
            if current_edge_weight > 1e-9:  # Avoid tiny/zero weights if not meaningful
                source_nodes_idx.append(source_idx)
                target_nodes_idx.append(target_idx)
                edge_weights.append(current_edge_weight)

    data = Data()
    data.num_nodes = num_nodes

    if source_nodes_idx:
        edge_index = torch.tensor([source_nodes_idx, target_nodes_idx], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        data.edge_index_out = edge_index
        data.edge_weight_out = edge_attr.squeeze()
        data.edge_index_in = edge_index.flip(dims=[0])
        data.edge_weight_in = edge_attr.squeeze()  # Assuming symmetric weight use for now
    else:
        print(f"Warning: No edges created for n={n_val} graph.")
        data.edge_index_out = torch.empty((2, 0), dtype=torch.long)
        data.edge_weight_out = torch.empty(0, dtype=torch.float)
        data.edge_index_in = torch.empty((2, 0), dtype=torch.long)
        data.edge_weight_in = torch.empty(0, dtype=torch.float)
        if num_nodes == 0:  # If no nodes either, return None for data
            return None, ngram_to_idx, idx_to_ngram, ngram_probabilities

    training_samples = []
    if hasattr(data, 'edge_index_out') and data.edge_index_out.numel() > 0:
        for i in range(data.edge_index_out.size(1)):
            src = data.edge_index_out[0, i].item()
            tgt = data.edge_index_out[1, i].item()
            training_samples.append((src, tgt))
    data.training_samples = training_samples

    return data, ngram_to_idx, idx_to_ngram, ngram_probabilities


# --- 3. Positional Embeddings (Assumed to be defined as in the initial script) ---
def get_sinusoidal_positional_embeddings(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term[:-1]) if div_term.size(0) > 1 and div_term.size(0) - 1 > 0 else torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# --- 4. Modified DiGCN Model (Assumed to be defined as in the initial script) ---
class CustomDiGCNLayerPyG(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_nodes_for_coeffs: int, use_vector_coeffs: bool = True):
        super(CustomDiGCNLayerPyG, self).__init__(aggr='add')

        self.lin_main_in = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_main_out = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=False)

        self.bias_main_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_main_out = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_in = nn.Parameter(torch.Tensor(out_channels))
        self.bias_skip_out = nn.Parameter(torch.Tensor(out_channels))

        self.use_vector_coeffs = use_vector_coeffs
        self.num_nodes = num_nodes_for_coeffs

        if self.use_vector_coeffs:
            self.C_in_vec = nn.Parameter(torch.Tensor(self.num_nodes, 1))
            self.C_out_vec = nn.Parameter(torch.Tensor(self.num_nodes, 1))
        else:
            self.C_in = nn.Parameter(torch.Tensor(1))
            self.C_out = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.lin_main_in, self.lin_main_out, self.lin_skip]:
            nn.init.xavier_uniform_(lin.weight)
        for bias in [self.bias_main_in, self.bias_main_out, self.bias_skip_in, self.bias_skip_out]:
            nn.init.zeros_(bias)

        if self.use_vector_coeffs:
            nn.init.ones_(self.C_in_vec)
            nn.init.ones_(self.C_out_vec)
        else:
            nn.init.ones_(self.C_in)
            nn.init.ones_(self.C_out)

    def forward(self, x: torch.Tensor,
                edge_index_in: torch.Tensor, edge_weight_in: torch.Tensor,
                edge_index_out: torch.Tensor, edge_weight_out: torch.Tensor):

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

        if self.use_vector_coeffs:
            C_in_vec_ = self.C_in_vec.to(x.device)
            C_out_vec_ = self.C_out_vec.to(x.device)
            if C_in_vec_.size(0) != x.size(0):
                raise ValueError(f"C_in_vec num_nodes ({C_in_vec_.size(0)}) mismatch with input x num_nodes ({x.size(0)})")
            output = C_in_vec_ * ic_combined + C_out_vec_ * oc_combined
        else:
            C_in_ = self.C_in.to(x.device)
            C_out_ = self.C_out.to(x.device)
            output = C_in_ * ic_combined + C_out_ * oc_combined

        return output

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class ProtDiGCNEncoderDecoder(nn.Module):
    def __init__(self, num_initial_features: int,
                 hidden_dim1: int, hidden_dim2: int,
                 num_nodes_in_graph: int,
                 n_gram_length_for_pe: int,
                 one_gram_embed_dim_for_pe: int,
                 max_allowable_ngram_len_for_pe: int,
                 dropout_rate: float,
                 use_vector_coeffs_in_gnn: bool = True):
        super().__init__()

        self.n_gram_length_for_pe = n_gram_length_for_pe
        self.one_gram_embed_dim_for_pe = one_gram_embed_dim_for_pe
        self.dropout_rate = dropout_rate
        self.num_nodes_in_graph = num_nodes_in_graph

        self.positional_encoder = None
        if self.one_gram_embed_dim_for_pe > 0 and max_allowable_ngram_len_for_pe > 0:
            self.positional_encoder = get_sinusoidal_positional_embeddings(
                max_allowable_ngram_len_for_pe,
                self.one_gram_embed_dim_for_pe
            )

        current_feature_dim = num_initial_features

        self.conv1 = CustomDiGCNLayerPyG(current_feature_dim, hidden_dim1,
                                         num_nodes_for_coeffs=num_nodes_in_graph,
                                         use_vector_coeffs=use_vector_coeffs_in_gnn)
        self.conv2 = CustomDiGCNLayerPyG(hidden_dim1, hidden_dim2,
                                         num_nodes_for_coeffs=num_nodes_in_graph,
                                         use_vector_coeffs=use_vector_coeffs_in_gnn)

        self.decoder_fc = nn.Linear(hidden_dim2, num_nodes_in_graph)
        self.embedding_output_from_conv2 = None

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_encoder is None or self.n_gram_length_for_pe == 0 or self.one_gram_embed_dim_for_pe == 0:
            return x

        if self.n_gram_length_for_pe == 1:
            if x.shape[1] == self.one_gram_embed_dim_for_pe:
                pe_to_add = self.positional_encoder[0, :].to(x.device)
                x = x + pe_to_add.unsqueeze(0)
                # else: # This warning was a bit noisy during testing, can be re-enabled
                # print(f"Warning: PE not applied for n=1. Feature dim {x.shape[1]} != PE dim {self.one_gram_embed_dim_for_pe}")
        elif self.n_gram_length_for_pe > 1:
            if x.shape[1] == self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe:
                x_reshaped = x.view(-1, self.n_gram_length_for_pe, self.one_gram_embed_dim_for_pe)
                pe_len_to_use = min(self.n_gram_length_for_pe, self.positional_encoder.shape[0])
                pe_to_add = self.positional_encoder[:pe_len_to_use, :].to(x.device)

                x_reshaped_pe_part = x_reshaped[:, :pe_len_to_use, :] + pe_to_add.unsqueeze(0)

                if pe_len_to_use < self.n_gram_length_for_pe:
                    x_with_pe_reshaped = torch.cat(
                        (x_reshaped_pe_part, x_reshaped[:, pe_len_to_use:, :]), dim=1
                    )
                else:
                    x_with_pe_reshaped = x_reshaped_pe_part
                x = x_with_pe_reshaped.view(-1, self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe)
            # else: # This warning was a bit noisy
            # print(f"Warning: PE not applied for n={self.n_gram_length_for_pe}. Feature dim {x.shape[1]} != expected {self.n_gram_length_for_pe * self.one_gram_embed_dim_for_pe}")
        return x

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        x = self._apply_positional_encoding(x)

        h = self.conv1(x, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out)
        h = F.tanh(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        self.embedding_output_from_conv2 = self.conv2(h, data.edge_index_in, data.edge_weight_in, data.edge_index_out, data.edge_weight_out)
        h_embed_activated = F.tanh(self.embedding_output_from_conv2)
        h_embed_dropped = F.dropout(h_embed_activated, p=self.dropout_rate, training=self.training)

        next_hop_logits = self.decoder_fc(h_embed_dropped)

        return F.log_softmax(next_hop_logits, dim=-1), self.embedding_output_from_conv2


# --- 5. Training and Embedding Extraction (Assumed to be defined as in the initial script) ---
def train_ngram_model(model: ProtDiGCNEncoderDecoder, data: Data, optimizer: optim.Optimizer, epochs: int, device: torch.device):
    model.train()
    model.to(device)
    data = data.to(device)

    if not hasattr(data, 'training_samples') or not data.training_samples:
        print("No training_samples (source,target pairs) in data object. Cannot train.")
        return

    source_indices = torch.tensor([s for s, t in data.training_samples], dtype=torch.long).to(device)
    target_indices = torch.tensor([t for s, t in data.training_samples], dtype=torch.long).to(device)

    if source_indices.numel() == 0:
        print("Source indices for training are empty. Cannot train.")
        return

    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        log_probs, _ = model(data)

        if source_indices.max() >= log_probs.size(0):
            print(f"Error: Max source index {source_indices.max()} out of bounds for log_probs size {log_probs.size(0)}")
            return
        if target_indices.max() >= log_probs.size(1):  # Decoder outputs logits for num_nodes_in_graph
            print(f"Error: Max target index {target_indices.max()} out of bounds for decoder output size {log_probs.size(1)}")
            return

        loss = criterion(log_probs[source_indices], target_indices)
        loss.backward()
        optimizer.step()
        if epoch % (max(1, epochs // 10)) == 0 or epoch == epochs:  # Ensure printing for few epochs
            print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")


def extract_node_embeddings(model: ProtDiGCNEncoderDecoder, data: Data, device: torch.device) -> Optional[np.ndarray]:
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data)
        if embeddings is not None:
            return embeddings.cpu().numpy()
    return None


# --- 6. Main Workflow (Corrected Section) ---
def run_hierarchical_ngram_embedding_workflow(
        fasta_filepath: str,
        max_n_for_ngram: int,
        output_dir: str,
        one_gram_init_embed_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim1: int = DEFAULT_HIDDEN_DIM_1,
        hidden_dim2: int = DEFAULT_HIDDEN_DIM_2,
        pe_max_len: int = MAX_NRAM_LEN_FOR_PE,
        dropout: float = DEFAULT_DROPOUT_RATE,
        lr: float = DEFAULT_LEARNING_RATE,
        epochs_per_level: int = DEFAULT_EPOCHS_PER_NGRAM_MODEL,
        use_vector_gnn_coeffs: bool = True
):
    print("Starting Hierarchical N-gram Embedding Workflow...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use new parser to get (protein_id, sequence)
    protein_id_sequence_pairs = parse_fasta_sequences_with_ids(fasta_filepath)
    if not protein_id_sequence_pairs:
        print("No sequences parsed. Exiting.")
        return

    sequences = [seq for pid, seq in protein_id_sequence_pairs]  # Keep separate list of sequences for n-gram processing

    level_embeddings: dict[int, np.ndarray] = {}
    level_ngram_to_idx: dict[int, dict[tuple, int]] = {}
    level_idx_to_ngram: dict[int, dict[int, tuple]] = {}  # Store idx_to_ngram for each level
    level_node_probabilities: dict[int, dict[tuple, float]] = {}

    per_residue_emb_path = os.path.join(output_dir, "per_residue_embeddings_from_1gram.h5")
    per_protein_emb_path = os.path.join(output_dir, f"per_protein_embeddings_from_{max_n_for_ngram}gram.h5")

    for n_val in range(1, max_n_for_ngram + 1):
        print(f"\n--- Processing N-gram Level: n = {n_val} ---")

        current_ngrams, current_transitions = get_ngrams_and_transitions(sequences, n_val)
        if not current_ngrams:
            print(f"No {n_val}-grams found. Stopping hierarchical process.")
            break

        prev_n_gram_probs = level_node_probabilities.get(n_val - 1) if n_val > 1 else None

        # build_ngram_graph_data now returns idx_to_ngram_map as the third item
        graph_data, ngram_to_idx_map, idx_to_ngram_map, ngram_probs = build_ngram_graph_data(
            current_ngrams, current_transitions,
            node_prob_from_prev_graph=prev_n_gram_probs,
            n_val=n_val
        )

        if graph_data is None or graph_data.num_nodes == 0:
            print(f"Could not build graph for n={n_val}. Stopping.")
            break

        level_ngram_to_idx[n_val] = ngram_to_idx_map
        level_idx_to_ngram[n_val] = idx_to_ngram_map  # Store it
        level_node_probabilities[n_val] = ngram_probs
        print(f"Built graph for n={n_val}: {graph_data.num_nodes} nodes, {graph_data.edge_index_out.size(1)} out-edges.")

        current_feature_dim: int
        if n_val == 1:
            current_feature_dim = one_gram_init_embed_dim
            graph_data.x = torch.randn(graph_data.num_nodes, current_feature_dim)
            print(f"  Initialized 1-gram features randomly (dim: {current_feature_dim}).")
        else:
            one_gram_embeds_arr = level_embeddings.get(1)
            one_gram_idx_to_char_map = level_idx_to_ngram.get(1)  # map int index to 1-gram tuple e.g. {0: ('A',)}

            if one_gram_embeds_arr is None or one_gram_idx_to_char_map is None:
                print(f"Error: 1-gram embeddings/map not found for constructing n={n_val} features. Stopping.")
                break

            # We need a map from 1-gram char to its index in the 1-gram embedding array
            # The level_ngram_to_idx[1] gives {('A',): 0, ('C',): 1, ...}
            one_gram_char_tuple_to_idx_map = level_ngram_to_idx.get(1)
            if one_gram_char_tuple_to_idx_map is None:  # Should not happen if check above passed
                print(f"Error: 1-gram char_tuple_to_idx map not found. Stopping.")
                break

            features_list = []
            # current_idx_to_ngram_map is for the *current* n_val level
            current_idx_to_ngram_map = level_idx_to_ngram[n_val]

            for i in range(graph_data.num_nodes):  # i is the node index in the current n-val graph
                current_multichar_ngram_tuple = current_idx_to_ngram_map[i]  # e.g., ('A', 'C', 'G') for n=3

                feature_parts = []
                valid_ngram_feature = True
                for residue_char in current_multichar_ngram_tuple:  # e.g., residue_char is 'A', then 'C', then 'G'
                    residue_1gram_tuple_key = (residue_char,)  # Key for 1-gram map, e.g., ('A',)

                    if residue_1gram_tuple_key in one_gram_char_tuple_to_idx_map:
                        idx_in_1gram_embeddings = one_gram_char_tuple_to_idx_map[residue_1gram_tuple_key]
                        feature_parts.append(torch.tensor(one_gram_embeds_arr[idx_in_1gram_embeddings]))
                    else:
                        # This case should be rare if all residues are in the 1-gram vocab
                        print(f"Warning: Residue '{residue_char}' from {n_val}-gram not in 1-gram map. Using zeros.")
                        feature_parts.append(torch.zeros(one_gram_embeds_arr.shape[1]))
                        valid_ngram_feature = False

                if feature_parts:  # If any parts were collected (should always be true for n > 0)
                    features_list.append(torch.cat(feature_parts))
                else:  # Fallback, should ideally not be reached if n_val > 0
                    print(f"Error: No feature parts for n-gram {current_multichar_ngram_tuple}. Using zeros.")
                    features_list.append(torch.zeros(n_val * one_gram_embeds_arr.shape[1]))

            if not features_list:  # If somehow no features were created for any node
                print(f"Error: Failed to create any node features for n={n_val}. Stopping.")
                break

            graph_data.x = torch.stack(features_list)
            current_feature_dim = graph_data.x.shape[1]
            print(f"  Initialized n={n_val} features by concatenating 1-gram embeddings (new dim: {current_feature_dim}).")

        model = ProtDiGCNEncoderDecoder(
            num_initial_features=current_feature_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            num_nodes_in_graph=graph_data.num_nodes,
            n_gram_length_for_pe=n_val,
            one_gram_embed_dim_for_pe=one_gram_init_embed_dim,
            max_allowable_ngram_len_for_pe=pe_max_len,
            dropout_rate=dropout,
            use_vector_coeffs_in_gnn=use_vector_gnn_coeffs
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"  Training model for n={n_val}...")
        train_ngram_model(model, graph_data, optimizer, epochs_per_level, device)

        node_embeddings_np = extract_node_embeddings(model, graph_data, device)
        if node_embeddings_np is None:
            print(f"Failed to extract embeddings for n={n_val}. Stopping.")
            break

        level_embeddings[n_val] = node_embeddings_np
        print(f"  Extracted {node_embeddings_np.shape[0]} node embeddings of dim {node_embeddings_np.shape[1]} for n={n_val}.")

        if n_val == 1:
            with h5py.File(per_residue_emb_path, 'w') as hf:
                # Use level_idx_to_ngram[1] and level_embeddings[1]
                idx_map_1gram = level_idx_to_ngram[1]
                embeds_1gram = level_embeddings[1]
                for idx, ngram_tuple in idx_map_1gram.items():
                    hf.create_dataset(ngram_tuple[0], data=embeds_1gram[idx])  # Key is the char
                print(f"Saved per-residue (1-gram) embeddings to {per_residue_emb_path}")

    if max_n_for_ngram not in level_embeddings or max_n_for_ngram not in level_ngram_to_idx:
        print(f"Embeddings for final n-gram level (n={max_n_for_ngram}) not available. Cannot generate per-protein embeddings.")
    else:
        print(f"\n--- Generating Per-Protein Embeddings (from n={max_n_for_ngram} graph) ---")
        final_n_gram_embeddings = level_embeddings[max_n_for_ngram]
        final_n_gram_to_idx_map = level_ngram_to_idx[max_n_for_ngram]  # ngram_tuple -> int_idx

        with h5py.File(per_protein_emb_path, 'w') as hf_protein:
            protein_count = 0
            # Iterate using protein_id_sequence_pairs to get actual protein IDs
            for protein_id, protein_sequence in tqdm(protein_id_sequence_pairs, desc="Pooling protein embeddings"):
                if len(protein_sequence) < max_n_for_ngram:
                    continue

                protein_specific_ngram_embeddings = []
                for i in range(len(protein_sequence) - max_n_for_ngram + 1):
                    ngram_tuple = tuple(protein_sequence[i: i + max_n_for_ngram])
                    if ngram_tuple in final_n_gram_to_idx_map:
                        ngram_idx = final_n_gram_to_idx_map[ngram_tuple]
                        protein_specific_ngram_embeddings.append(final_n_gram_embeddings[ngram_idx])

                if protein_specific_ngram_embeddings:
                    pooled_embedding = np.mean(protein_specific_ngram_embeddings, axis=0)
                    hf_protein.create_dataset(protein_id, data=pooled_embedding)  # Use actual protein_id
                    protein_count += 1
            print(f"Saved {protein_count} per-protein embeddings to {per_protein_emb_path}")

    print("\nWorkflow finished.")


if __name__ == '__main__':
    DUMMY_FASTA_PATH = "C:/ProgramData/ProtDiGCN/uniprot_sequences_sample.txt"  # Using your sample path
    DUMMY_OUTPUT_DIR = "C:/ProgramData/ProtDiGCN/hierarchical_ngram_embeddings_output"  # Using your sample path

    # It's good practice to ensure the dummy FASTA exists if this is a test,
    # or rely on the user providing it. For now, I'll assume it exists at the path.
    if not os.path.exists(DUMMY_FASTA_PATH):
        print(f"Error: Dummy FASTA file not found at {DUMMY_FASTA_PATH}. Please create it or update the path.")
        # Example of creating a minimal one if it's missing and you want to auto-generate for testing:
        # with open(DUMMY_FASTA_PATH, "w") as f:
        #     f.write(">protein1_test_id\nACGCGTCGACGTACGTAGCATCGATCG\n")
        #     f.write(">protein2_another_id\nTTTTCCCCGGGGAAAATTTCCCGGAA\n")
        #     f.write(">P12345\nAGCTAGCTAGCTNNNAGCTAGCT\n")
        #     f.write(">Q00001\nACGT\n")
        # exit()

    run_hierarchical_ngram_embedding_workflow(
        fasta_filepath=DUMMY_FASTA_PATH,
        max_n_for_ngram=2,  # Changed from 3 to 2 to match your error point for quicker verification
        output_dir=DUMMY_OUTPUT_DIR,
        one_gram_init_embed_dim=DEFAULT_EMBEDDING_DIM,
        hidden_dim1=DEFAULT_HIDDEN_DIM_1,
        hidden_dim2=DEFAULT_HIDDEN_DIM_2,
        pe_max_len=MAX_NRAM_LEN_FOR_PE,
        epochs_per_level=DEFAULT_EPOCHS_PER_NGRAM_MODEL,
        lr=DEFAULT_LEARNING_RATE,
        dropout=DEFAULT_DROPOUT_RATE,
        use_vector_gnn_coeffs=True  # As per requirements
    )

    print(f"\nExample run complete. Check outputs in '{DUMMY_OUTPUT_DIR}'.")
