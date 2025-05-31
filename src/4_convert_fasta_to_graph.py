import glob
import json
import os
import pickle
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Iterator, Callable, Optional  # Added Optional

import dask.bag as db
import numpy as np
import tensorflow as tf
from dask.diagnostics import ProgressBar

# --- Configuration ---
UNIREF_FASTA_DIRECTORY = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/"
UNIREF_FASTA_PATHS = None

OUTPUT_GRAPH_DIR = "./output_global_character_graphs"
GLOBAL_DIRECTED_GRAPH_FILENAME = "global_directed_character_graph.pkl"
GLOBAL_UNDIRECTED_GRAPH_FILENAME = "global_undirected_character_graph.pkl"
CHAR_VOCAB_AND_MAPPINGS_FILENAME = "global_char_graph_vocab_mappings.json"

SPACE_SEPARATOR_CHAR = ' '
VALID_AA_CHARACTERS = "ACDEFGHIKLMNPQRSTVWYXBUZO"
DASK_N_PARTITIONS = None
random_seed = 123
np.random.seed(random_seed)


# --- Helper Functions ---

def strip_non_alphanumeric_biohelper(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-zA-Z0-9]', '', text.rstrip().lstrip())


def encode_characters_biohelper(character_sequence: List[str]) -> List[str]:
    ascii_sequence = []
    for c in character_sequence:
        try:
            if isinstance(c, str) and len(c) == 1:
                ascii_sequence.append(str(ord(c)))
            else:
                ascii_sequence.append("ERR_ENC_CHAR")
        except TypeError:
            ascii_sequence.append("ERR_ENC_TYPE")
    return ascii_sequence


def fasta_parser_for_dask(file_handle_iterator) -> Iterator[Tuple[str, str]]:
    current_header_id = None
    sequence_parts = []
    for line_content in file_handle_iterator:
        line = line_content.strip()
        if not line: continue
        if line.startswith('>'):
            if current_header_id and sequence_parts: yield current_header_id, "".join(sequence_parts)
            current_header_id = line[1:].split(maxsplit=1)[0]
            sequence_parts = []
        elif current_header_id:
            sequence_parts.append(line.upper())
    if current_header_id and sequence_parts:
        yield current_header_id, "".join(sequence_parts)


def process_fasta_chunk_to_bigrams(
        sequences_in_chunk: List[str],
        *,  # Force subsequent arguments to be keyword-only
        space_separator: str = SPACE_SEPARATOR_CHAR,
        valid_aa_chars: str = VALID_AA_CHARACTERS,
        # strip_algo is not used here as sequences_in_chunk are already stripped/cleaned by caller
        encode_algo: Callable[[List[str]], List[str]] = encode_characters_biohelper
) -> List[Tuple[str, str]]:
    """
    Processes a CHUNK of already cleaned protein sequences.
    Joins them with space_separator, then encodes and generates bigrams.
    """
    # For Debugging what this function receives:
    # print(f"DEBUG process_fasta_chunk_to_bigrams: received {len(sequences_in_chunk)} sequences. space='{space_separator}', valid='{valid_aa_chars}', encode_algo={encode_algo}")

    if not sequences_in_chunk:
        return []

    mega_sequence_chunk_string = space_separator.join(sequences_in_chunk)
    final_character_walk: List[str] = list(mega_sequence_chunk_string)
    encoded_walk: List[str] = encode_algo(final_character_walk)

    bigrams = []
    if len(encoded_walk) >= 2:
        for i in range(len(encoded_walk) - 1):
            if encoded_walk[i] not in ["ERR_ENC_CHAR", "ERR_ENC_TYPE"] and \
                    encoded_walk[i + 1] not in ["ERR_ENC_CHAR", "ERR_ENC_TYPE"]:
                bigrams.append((encoded_walk[i], encoded_walk[i + 1]))
    return bigrams


def map_file_to_chunked_bigrams(
        fasta_filepath: str,
        sequences_per_chunk: int = 1000,
        space_separator: str = SPACE_SEPARATOR_CHAR,
        valid_aa_chars: str = VALID_AA_CHARACTERS,
        strip_algo: Callable[[str], str] = strip_non_alphanumeric_biohelper,
        encode_algo: Callable[[List[str]], List[str]] = encode_characters_biohelper
) -> Iterator[Tuple[str, str]]:
    # This print is helpful to see if Dask calls this function as expected for each file
    print(f"  Processing file (chunked yield): {os.path.basename(fasta_filepath)}")

    current_chunk_cleaned_sequences = []
    seq_count_in_file = 0
    try:
        with open(fasta_filepath, 'r', encoding='utf-8') as f_handle:
            for _, seq_str in fasta_parser_for_dask(f_handle):
                seq_count_in_file += 1
                processed_seq_str = strip_algo(seq_str)
                cleaned_seq = "".join(c.upper() for c in processed_seq_str if c.upper() in valid_aa_chars)
                if cleaned_seq:
                    current_chunk_cleaned_sequences.append(cleaned_seq)

                if len(current_chunk_cleaned_sequences) >= sequences_per_chunk:
                    # Explicit keyword arguments for the inner call
                    yield from process_fasta_chunk_to_bigrams(
                        sequences_in_chunk=current_chunk_cleaned_sequences,
                        space_separator=space_separator,
                        valid_aa_chars=valid_aa_chars,
                        encode_algo=encode_algo
                    )
                    current_chunk_cleaned_sequences = []

            if current_chunk_cleaned_sequences:
                yield from process_fasta_chunk_to_bigrams(
                    sequences_in_chunk=current_chunk_cleaned_sequences,
                    space_separator=space_separator,
                    valid_aa_chars=valid_aa_chars,
                    encode_algo=encode_algo
                )
    except Exception as e:
        # This is where your traceback shows the error is caught
        print(f"Error during chunked processing of {fasta_filepath}: {e}")
        # For more detail from Dask worker, you might need to configure Dask logging
        # or use a try-except inside the lambda if this doesn't show enough.

    print(f"  Finished yielding bigrams from {seq_count_in_file} sequences in {os.path.basename(fasta_filepath)}")


def get_first_last_char_from_dataset(
        sorted_fasta_files: List[str],
        valid_aa_chars: str,
        strip_algo: Callable[[str], str]
) -> Tuple[Optional[str], Optional[str]]:
    first_char = None;
    last_char = None
    if not sorted_fasta_files: return None, None
    try:
        with open(sorted_fasta_files[0], 'r', encoding='utf-8') as f:
            for _, seq_str in fasta_parser_for_dask(f):
                processed_seq = strip_algo(seq_str)
                cleaned_seq = "".join(c.upper() for c in processed_seq if c.upper() in valid_aa_chars)
                if cleaned_seq: first_char = cleaned_seq[0]; break
            if first_char: print(f"    Global first character: {first_char}")
    except Exception as e:
        print(f"Could not determine first character: {e}")
    try:
        temp_last_char_of_file = None
        with open(sorted_fasta_files[-1], 'r', encoding='utf-8') as f:
            last_valid_seq_in_file = None
            for _, seq_str in fasta_parser_for_dask(f):
                processed_seq = strip_algo(seq_str)
                cleaned_seq = "".join(c.upper() for c in processed_seq if c.upper() in valid_aa_chars)
                if cleaned_seq: last_valid_seq_in_file = cleaned_seq
            if last_valid_seq_in_file: temp_last_char_of_file = last_valid_seq_in_file[-1]
        if temp_last_char_of_file: last_char = temp_last_char_of_file; print(f"    Global last character: {last_char}")
    except Exception as e:
        print(f"Could not determine last character: {e}")
    return first_char, last_char


# --- TensorFlow Graph Class Definitions ---
class Shape:
    def __init__(self, in_size, out_size,
                 batch_size): self.in_size = in_size; self.out_size = out_size; self.batch_size = batch_size

    def __str__(self): return f"(In: {self.in_size}, Out: {self.out_size}, Batch: {self.batch_size})"


class Graph:
    def __init__(self, nodes=None, edges=None):
        self.original_nodes_dict = nodes if nodes is not None else {};
        self.original_edges = edges if edges is not None else []
        self.nodes = {};
        self.edges = [];
        self.node_index = {};
        self.node_inverted_index = {}
        self.number_of_nodes = 0;
        self.number_of_edges = 0;
        self._features = None;
        self.dimensions = 0
        self.node_indices();
        self.edge_indices()

    def node_indices(self):
        current_map = self.original_nodes_dict
        if not current_map and self.original_edges:
            all_ids = set();
            [all_ids.add(str(e[0])) or all_ids.add(str(e[1])) for e in self.original_edges if len(e) >= 2]
            current_map = {nid: nid for nid in sorted(list(all_ids))};
            self.original_nodes_dict = current_map
        sorted_ids = sorted([str(k) for k in current_map.keys()])
        self.nodes = {};
        self.node_index = {};
        self.node_inverted_index = {}
        for i, id_str in enumerate(sorted_ids): self.node_index[id_str] = i; self.node_inverted_index[i] = id_str;
        self.nodes[i] = current_map[id_str]
        self.number_of_nodes = len(self.nodes)

    def edge_indices(self):
        if not self.node_index: print(f"W({type(self).__name__}): No node_idx."); self.edges = deepcopy(
            self.original_edges); self.number_of_edges = len(self.edges); return
        idx_edges = [];
        [idx_edges.append((self.node_index[str(e[0])], self.node_index[str(e[1])]) + tuple(e[2:])) for e in
         self.original_edges if len(e) >= 2 and str(e[0]) in self.node_index and str(e[1]) in self.node_index]
        self.edges = idx_edges;
        self.number_of_edges = len(self.edges)

    def set_node_features(self, features_np):
        if features_np is not None and isinstance(features_np, np.ndarray) and (
                features_np.shape[0] == self.number_of_nodes or self.number_of_nodes == 0):
            self._features = tf.constant(features_np, dtype=tf.float32);
            self.dimensions = features_np.shape[1] if features_np.ndim > 1 and features_np.shape[0] > 0 else 0
        else:
            self._features = None; self.dimensions = 0

    @property
    def features(self):
        if self._features is None and self.number_of_nodes > 0:
            self.set_node_features(np.eye(self.number_of_nodes, dtype=np.float32))
        elif self._features is None and self.number_of_nodes == 0:
            self.set_node_features(np.array([], dtype=np.float32).reshape(0, 0))
        return self._features


class UndirectedGraph(Graph):
    def __init__(self, edges=None, nodes=None):
        super().__init__(nodes=nodes, edges=edges); self._adjacency = None; self._degree_normalized_adjacency = None

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
            adj = self.adjacency.copy();
            adj_tilde = adj + np.eye(self.number_of_nodes, dtype=float)
            deg_tilde_vec = np.sum(adj_tilde, axis=0)
            inv_sqrt_deg = np.power(deg_tilde_vec, -0.5, where=deg_tilde_vec != 0, out=np.zeros_like(deg_tilde_vec))
            D_inv_sqrt = np.diag(inv_sqrt_deg)
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
    # Add other properties like in_adjacency, normalized versions if needed by your specific GNNs


# --- Main Dask Workflow Script ---
def build_global_graph_workflow():
    print("Starting global character graph construction using Dask...")
    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

    actual_fasta_files = []
    if UNIREF_FASTA_PATHS and isinstance(UNIREF_FASTA_PATHS, list) and len(UNIREF_FASTA_PATHS) > 0:
        actual_fasta_files = [f for f in UNIREF_FASTA_PATHS if os.path.isfile(f)]
    elif isinstance(UNIREF_FASTA_DIRECTORY, str) and os.path.isdir(UNIREF_FASTA_DIRECTORY):
        print(f"Scanning directory '{UNIREF_FASTA_DIRECTORY}' for FASTA files (*.fasta, *.fas, *.fa, *.fna)...")
        for ext in ('*.fasta', '*.fas', '*.fa', '*.fna'):
            actual_fasta_files.extend(glob.glob(os.path.join(UNIREF_FASTA_DIRECTORY, ext)))

    if not actual_fasta_files: print(f"Error: No FASTA files found."); return
    actual_fasta_files.sort()
    print(f"Found {len(actual_fasta_files)} FASTA file(s) to process.")

    file_paths_bag = db.from_sequence(actual_fasta_files,
                                      npartitions=DASK_N_PARTITIONS or max(1, len(actual_fasta_files)))

    print("Mapping file processing to Dask Bag (extracting bigrams per file using chunked processing)...")

    # Define the arguments that are constant for all calls to map_file_to_chunked_bigrams
    # These will be passed as keyword arguments via the lambda
    fixed_args_for_map_file = {
        'sequences_per_chunk': 1000,  # You can make this a global config
        'space_separator': SPACE_SEPARATOR_CHAR,
        'valid_aa_chars': VALID_AA_CHARACTERS,
        'strip_algo': strip_non_alphanumeric_biohelper,
        'encode_algo': encode_characters_biohelper
    }

    bigrams_iterators_bag = file_paths_bag.map(
        lambda fp: map_file_to_chunked_bigrams(fasta_filepath=fp, **fixed_args_for_map_file)
    )

    all_encoded_bigrams_flat_bag = bigrams_iterators_bag.flatten()

    print("Computing global bigram frequencies with Dask (this may take time)...")
    with ProgressBar():
        num_dask_workers = 60  # Hardcoded as requested
        print(f"Dask: Using {num_dask_workers} workers for compute().")
        try:
            computed_bigram_frequencies_list = all_encoded_bigrams_flat_bag.frequencies().compute(
                scheduler='processes', num_workers=num_dask_workers
            )
        except ValueError as e:
            if "max_workers must be <= 61" in str(e) and num_dask_workers > 61:  # Check if it was actually over limit
                print(f"ValueError: max_workers ({num_dask_workers}) > 61. Capping at 60 and retrying with processes.")
                computed_bigram_frequencies_list = all_encoded_bigrams_flat_bag.frequencies().compute(
                    scheduler='processes', num_workers=60)
            elif "max_workers must be <= 61" in str(e):  # If it failed even at 60 or less for other reasons
                print(f"ValueError related to max_workers. Trying threaded scheduler.")
                computed_bigram_frequencies_list = all_encoded_bigrams_flat_bag.frequencies().compute(
                    scheduler='threads')
            else:
                raise e  # Re-raise other ValueErrors
        except Exception as e_gen:  # Catch other potential errors during compute
            print(f"An unexpected error occurred during Dask compute: {e_gen}")
            computed_bigram_frequencies_list = []  # Ensure it's a list

    bigram_counts_dict = defaultdict(int)
    for (item1, item2), count in computed_bigram_frequencies_list:
        bigram_counts_dict[(item1, item2)] += count

    print("Adjusting counts for global start/end spaces...")
    first_char_overall, last_char_overall = get_first_last_char_from_dataset(
        actual_fasta_files, VALID_AA_CHARACTERS, strip_non_alphanumeric_biohelper
    )
    # Ensure encode_algo is available here or pass it
    encode_algo_for_ends = encode_characters_biohelper
    encoded_space_list = encode_algo_for_ends([SPACE_SEPARATOR_CHAR])

    if not encoded_space_list or encoded_space_list[0] in ["ERR_ENC_CHAR", "ERR_ENC_TYPE"]:
        print("Error encoding space character. Cannot adjust global start/end counts.")
    else:
        encoded_space = encoded_space_list[0]
        if first_char_overall:
            encoded_first_char_list = encode_algo_for_ends([first_char_overall])
            if encoded_first_char_list and encoded_first_char_list[0] not in ["ERR_ENC_CHAR", "ERR_ENC_TYPE"]:
                encoded_first_char = encoded_first_char_list[0]
                bigram_counts_dict[(encoded_space, encoded_first_char)] += 1
                print(
                    f"  Incremented count for global start: (Space ('{encoded_space}') -> {first_char_overall} ('{encoded_first_char}'))")

        if last_char_overall:
            encoded_last_char_list = encode_algo_for_ends([last_char_overall])
            if encoded_last_char_list and encoded_last_char_list[0] not in ["ERR_ENC_CHAR", "ERR_ENC_TYPE"]:
                encoded_last_char = encoded_last_char_list[0]
                bigram_counts_dict[(encoded_last_char, encoded_space)] += 1
                print(
                    f"  Incremented count for global end: ({last_char_overall} ('{encoded_last_char}') -> Space ('{encoded_space}'))")

    if not bigram_counts_dict: print("No bigrams were extracted/counted. Cannot build graph."); return
    print(f"Total {len(bigram_counts_dict)} unique weighted character (ASCII string) edges globally after adjustments.")

    all_ascii_char_nodes = set()
    for (ascii_c1, ascii_c2), count in bigram_counts_dict.items():
        all_ascii_char_nodes.add(ascii_c1);
        all_ascii_char_nodes.add(ascii_c2)

    if encoded_space != "ERR_ENC_CHAR": all_ascii_char_nodes.add(encoded_space)

    sorted_ascii_chars = sorted(list(all_ascii_char_nodes))
    ascii_char_to_int_id = {ascii_char: i for i, ascii_char in enumerate(sorted_ascii_chars)}
    int_id_to_ascii_char = {i: ascii_char for ascii_char, i in ascii_char_to_int_id.items()}

    print(f"Global character (ASCII string) vocabulary size: {len(ascii_char_to_int_id)}")
    if not ascii_char_to_int_id: print("Vocabulary is empty. Exiting."); return

    try:
        vocab_path = os.path.join(OUTPUT_GRAPH_DIR, CHAR_VOCAB_AND_MAPPINGS_FILENAME)
        with open(vocab_path, 'w') as f:
            json.dump({'ascii_char_to_int_id': ascii_char_to_int_id, 'int_id_to_ascii_char': int_id_to_ascii_char,
                       'sorted_ascii_chars_list': sorted_ascii_chars}, f, indent=4)
        print(f"Character vocabulary saved to {vocab_path}")
    except Exception as e:
        print(f"Error saving char vocabulary: {e}")

    nodes_dict_for_graph = {ascii_char: ascii_char for ascii_char in sorted_ascii_chars}
    edge_list_for_graph = []
    for (ascii_c1, ascii_c2), count in bigram_counts_dict.items():
        edge_list_for_graph.append((ascii_c1, ascii_c2, float(count), 0))

    print("Instantiating global character graph objects...")
    global_directed_char_graph = DirectedGraph(nodes=deepcopy(nodes_dict_for_graph),
                                               edges=deepcopy(edge_list_for_graph))
    global_undirected_char_graph = UndirectedGraph(nodes=deepcopy(nodes_dict_for_graph),
                                                   edges=deepcopy(edge_list_for_graph))

    print(
        f"  Directed graph: {global_directed_char_graph.number_of_nodes} nodes, {global_directed_char_graph.number_of_edges} unique directed edge types.")
    print(
        f"  Undirected graph: {global_undirected_char_graph.number_of_nodes} nodes, {global_undirected_char_graph.number_of_edges} (original directed edge types passed).")

    print("Saving graph objects...")
    try:
        with open(os.path.join(OUTPUT_GRAPH_DIR, GLOBAL_DIRECTED_GRAPH_FILENAME), "wb") as f:
            pickle.dump(global_directed_char_graph, f)
        print(f"Saved directed graph to {os.path.join(OUTPUT_GRAPH_DIR, GLOBAL_DIRECTED_GRAPH_FILENAME)}")
        with open(os.path.join(OUTPUT_GRAPH_DIR, GLOBAL_UNDIRECTED_GRAPH_FILENAME), "wb") as f:
            pickle.dump(global_undirected_char_graph, f)
        print(f"Saved undirected graph to {os.path.join(OUTPUT_GRAPH_DIR, GLOBAL_UNDIRECTED_GRAPH_FILENAME)}")
    except Exception as e:
        print(f"Error saving graph objects: {e}")
    print("Global character graph construction complete.")


if __name__ == "__main__":
    if UNIREF_FASTA_DIRECTORY == "path/to/your/uniref_fasta_files/":
        print("WARNING: UNIREF_FASTA_DIRECTORY is a placeholder. Creating a single dummy FASTA file for testing.")
        dummy_dir = "temp_uniref_fasta_dir_for_char_graph"
        os.makedirs(dummy_dir, exist_ok=True)
        dummy_file_path = os.path.join(dummy_dir, "dummy_uniref_combined.fasta")
        with open(dummy_file_path, "w") as f:
            f.write(">protein1 test protein one\nACGTACGT\n")
            f.write(">protein2 another one\nCGTACGTAAAAC\n")
        UNIREF_FASTA_DIRECTORY = dummy_dir

    build_global_graph_workflow()
