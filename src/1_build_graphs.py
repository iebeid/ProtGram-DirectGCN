# ==============================================================================
# SCRIPT 1: N-gram Graph Construction and Analysis
# PURPOSE: Constructs n-gram graphs, performs analysis, and saves them.
# ==============================================================================

import os
import sys
import shutil
import multiprocessing
from itertools import repeat
import pandas as pd
import torch
import gc
import re
import traceback
from tqdm.auto import tqdm
from typing import Optional, Tuple
from collections import defaultdict

# PyG and NetworkX
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx
import community as community_louvain

# Dask for memory-efficient graph construction
try:
    import dask
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from Bio import SeqIO


# ==============================================================================
# --- CONFIGURATION CLASS ---
# ==============================================================================
class ScriptConfig:
    """
    Centralized configuration class for the graph building script.
    """

    def __init__(self):
        # --- GENERAL SETTINGS ---
        self.DEBUG_VERBOSE = True
        self.RANDOM_STATE = 42

        # !!! IMPORTANT: SET YOUR BASE DIRECTORIES HERE !!!
        self.BASE_DATA_DIR = "C:/ProgramData/ProtDiGCN/"
        self.BASE_OUTPUT_DIR = os.path.join(self.BASE_DATA_DIR, "ppi_evaluation_results_final_dummy")

        # --- Graph Construction Configuration ---
        self.NGRAM_GCN_INPUT_FASTA_PATH = os.path.join(self.BASE_DATA_DIR, "uniprot_sequences_sample.fasta")
        self.GRAPH_OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, "constructed_graphs")  # Directory to save final graph objects
        self.NGRAM_GCN_MAX_N = 3

        # --- Parallel & Dask Configuration ---
        self.DASK_CHUNK_SIZE = 2000000
        self.TEMP_FILE_DIR = os.path.join(self.BASE_OUTPUT_DIR, "temp_graph_files")
        self.PARALLEL_CONSTRUCTION_WORKERS: Optional[int] = 16


# ==============================================================================
# --- HELPER & GRAPH CONSTRUCTION FUNCTIONS (from original script) ---
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
    source_nodes = torch.tensor(unique_edges_df['source'].values, dtype=torch.long)
    target_nodes = torch.tensor(unique_edges_df['target'].values, dtype=torch.long)
    directed_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    data = Data(num_nodes=num_nodes)
    data.ngram_to_idx = ngram_to_idx
    data.idx_to_ngram = idx_to_ngram
    data.edge_index = to_undirected(directed_edge_index, num_nodes=num_nodes)  # Storing undirected for analysis
    del edge_df, edge_counts, unique_edges_df
    gc.collect()
    print("Graph object created successfully for analysis.")
    return data


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


# ==============================================================================
# --- NEW: GRAPH ANALYSIS FUNCTIONS ---
# ==============================================================================

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
        return labels, num_communities
    except Exception as e:
        print(f"Community Detection Error: {e}.")
        return torch.arange(num_nodes, dtype=torch.long), num_nodes


def analyze_and_print_graph_stats(graph_data: Data, graph_name: str, config: ScriptConfig):
    """Performs and prints key statistics for a given graph."""
    if not isinstance(graph_data, Data):
        print(f"Analysis for {graph_name}: Invalid graph data provided.")
        return

    print("\n" + "-" * 25)
    print(f"ANALYSIS for: {graph_name}")
    print(f"  - Total Nodes: {graph_data.num_nodes}")
    print(f"  - Total Edges (undirected): {graph_data.edge_index.size(1) // 2}")

    if graph_data.num_nodes > 0 and graph_data.edge_index.numel() > 0:
        # Build NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(range(graph_data.num_nodes))
        G.add_edges_from(graph_data.edge_index.cpu().numpy().T)

        # 1. Graph Density
        density = nx.density(G)
        print(f"  - Graph Density: {density:.6f}")

        # 2. Louvain Community Detection
        _, num_communities = detect_communities_louvain(graph_data.edge_index, graph_data.num_nodes, config)
        print(f"  - Louvain Communities Detected: {num_communities}")

        # 3. Degree Centrality
        if graph_data.num_nodes < 50000:  # Avoid long computation on huge graphs
            print("  - Degree Centrality (Top 5 Nodes):")
            degree_centrality = nx.degree_centrality(G)
            sorted_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
            for i, (node_idx, score) in enumerate(sorted_centrality[:5]):
                ngram = graph_data.idx_to_ngram.get(node_idx, "N/A")
                print(f"    {i + 1}. N-gram '{ngram}' (ID: {node_idx}) - Score: {score:.4f}")
        else:
            print("  - Degree Centrality: Skipped for large graph ( >50,000 nodes).")

    print("-" * 25 + "\n")


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    print("--- SCRIPT 1: N-gram Graph Construction and Analysis ---")
    config = ScriptConfig()

    # Create necessary directories
    os.makedirs(config.TEMP_FILE_DIR, exist_ok=True)
    os.makedirs(config.GRAPH_OUTPUT_DIR, exist_ok=True)

    # --- PART 1: Parallel construction of node maps and edge lists ---
    print("\n" + "=" * 20 + " Phase 1: Parallel Graph File Construction " + "=" * 20)
    levels_to_process = list(range(1, config.NGRAM_GCN_MAX_N + 1))

    if config.PARALLEL_CONSTRUCTION_WORKERS is None:
        num_workers = min(len(levels_to_process), max(1, multiprocessing.cpu_count() - 2))
    else:
        num_workers = config.PARALLEL_CONSTRUCTION_WORKERS
    print(f"Using {num_workers} worker processes.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(build_graph_files_for_level_n, zip(levels_to_process, repeat(config)))

    if not all(success for _, success in results):
        print("\nERROR: Graph file construction failed for one or more levels. Aborting.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All raw graph files (node maps, edge lists) pre-built successfully.")

    # --- PART 2: Load, Analyze, and Save Graph Objects ---
    print("\n" + "=" * 20 + " Phase 2: Loading, Analyzing, and Saving Graphs " + "=" * 20)
    for n_val in levels_to_process:
        print(f"\n--- Processing n = {n_val} ---")
        parquet_path = os.path.join(config.TEMP_FILE_DIR, f"ngram_map_n{n_val}.parquet")
        edge_path = os.path.join(config.TEMP_FILE_DIR, f"edge_list_n{n_val}.txt")

        # Load graph from the temporary files
        graph_data = build_graph_from_disk(parquet_path, edge_path)

        if graph_data:
            # Perform and print analysis
            analyze_and_print_graph_stats(graph_data, f"{n_val}-gram Graph", config)

            # Save the final PyG Data object
            output_filepath = os.path.join(config.GRAPH_OUTPUT_DIR, f"graph_n{n_val}.pt")
            torch.save(graph_data, output_filepath)
            print(f"SUCCESS: Graph for n={n_val} saved as a checkpoint file:")
            print(f"  >> {output_filepath}")
        else:
            print(f"FAILED: Could not build the graph object for n={n_val}.")

    # --- PART 3: Cleanup ---
    print("\n" + "=" * 20 + " Phase 3: Cleaning Up Temporary Files " + "=" * 20)
    try:
        shutil.rmtree(config.TEMP_FILE_DIR)
        print(f"Successfully removed temporary directory: {config.TEMP_FILE_DIR}")
    except OSError as e:
        print(f"Error removing temporary directory {config.TEMP_FILE_DIR}: {e.strerror}")

    print("\n--- Script 1 Finished ---")
