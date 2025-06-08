# ==============================================================================
# MODULE: pipeline/graph_builder.py
# PURPOSE: Builds n-gram graphs from a FASTA file and saves them as objects.
# ==============================================================================

import os
import sys
import shutil
import multiprocessing
from itertools import repeat
import pandas as pd
import pickle
import gc
import traceback
from tqdm.auto import tqdm
from typing import Optional

# Dask for memory-efficient graph construction
try:
    import dask
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from Bio import SeqIO

# Import from our new project structure
from src.utils import graph_processor
from src.config import Config


# --- Helper Functions for Graph Construction ---

def _stream_ngram_chunks(fasta_path: str, n: int, chunk_size: int):
    """Yields chunks of n-grams from a FASTA file."""
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


def _build_node_map_dask(fasta_path: str, n: int, output_path: str, chunk_size: int):
    """Uses Dask to find unique n-grams and save them to a Parquet file."""
    if not DASK_AVAILABLE: raise ImportError("Dask is required for this step.")
    print(f"  Pass 1 (n={n}): Discovering unique n-grams with Dask...")
    lazy_chunks = [dask.delayed(chunk) for chunk in _stream_ngram_chunks(fasta_path, n, chunk_size)]
    ddf = dd.from_delayed(lazy_chunks, meta={'ngram': 'string'})
    unique_ngrams_ddf = ddf.drop_duplicates()
    unique_ngrams_ddf.to_parquet(output_path, engine='pyarrow', write_index=False, overwrite=True)
    print(f"  Pass 1 (n={n}) Complete. Unique n-grams saved to: {output_path}")


def _stream_transitions(fasta_path: str, n: int):
    """Yields n-gram to n-gram transitions from a FASTA file."""
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) >= n + 1:
            for i in range(len(seq) - n):
                yield ("".join(seq[i: i + n]), "".join(seq[i + 1: i + 1 + n]))


def _build_edge_file(fasta_path: str, n: int, output_path: str):
    """Creates a CSV file of all n-gram transitions."""
    print(f"  Pass 2 (n={n}): Generating edge list and saving to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("source,target\n")
        for source_ngram, target_ngram in tqdm(_stream_transitions(fasta_path, n), desc=f"  Generating edges for n={n}", leave=False):
            f.write(f"{source_ngram},{target_ngram}\n")
    print(f"  Pass 2 (n={n}) Complete: Edge file has been created.")


def _build_intermediate_files_for_level(n_val: int, config: Config, temp_dir: str):
    """Worker function to create the intermediate node and edge files for a given n-gram level."""
    if DASK_AVAILABLE: dask.config.set(scheduler='synchronous')
    print(f"[Worker n={n_val}]: Starting intermediate file construction.")
    try:
        node_map_path = os.path.join(temp_dir, f"ngram_map_n{n_val}.parquet")
        _build_node_map_dask(config.GCN_INPUT_FASTA_PATH, n_val, node_map_path, config.DASK_CHUNK_SIZE)

        edge_path = os.path.join(temp_dir, f"edge_list_n{n_val}.txt")
        _build_edge_file(config.GCN_INPUT_FASTA_PATH, n_val, edge_path)

        print(f"[Worker n={n_val}]: Successfully completed file construction.")
        return True
    except Exception as e:
        print(f"[Worker n={n_val}]: FAILED with error: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# --- Main Orchestration Function for this Module ---
# ==============================================================================

def run_graph_building(config: Config):
    """
    The main entry point for the graph building pipeline step.
    This function creates and saves graph objects for each n-gram level.
    """
    print("\n" + "=" * 80)
    print("### PIPELINE STEP 1: Building N-gram Graphs ###")
    print("=" * 80)

    # Use a temporary directory for intermediate files
    temp_dir = os.path.join(config.BASE_OUTPUT_DIR, "temp_graph_builder")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(config.GRAPH_OBJECTS_DIR, exist_ok=True)

    # --- PART 1: Parallel construction of intermediate files ---
    print("\n>>> Phase 1: Creating intermediate node and edge files in parallel...")
    levels_to_process = list(range(1, config.GCN_NGRAM_MAX_N + 1))

    num_workers = config.GRAPH_BUILDER_WORKERS or min(len(levels_to_process), max(1, os.cpu_count() - 2))
    print(f"Using {num_workers} worker processes.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # We use partial to pass the config and temp_dir to the worker function
        from functools import partial
        worker_func = partial(_build_intermediate_files_for_level, config=config, temp_dir=temp_dir)
        results = pool.map(worker_func, levels_to_process)

    if not all(results):
        print("\nERROR: Intermediate file construction failed for one or more levels. Aborting.")
        shutil.rmtree(temp_dir)
        sys.exit(1)
    else:
        print("\nSUCCESS: All intermediate files created successfully.")

    # --- PART 2: Load files and build final Graph objects ---
    print("\n>>> Phase 2: Building and saving final graph objects...")
    for n_val in levels_to_process:
        print(f"\n--- Processing n = {n_val} ---")
        node_map_path = os.path.join(temp_dir, f"ngram_map_n{n_val}.parquet")
        edge_path = os.path.join(temp_dir, f"edge_list_n{n_val}.txt")

        # 1. Load nodes and edges from intermediate files
        nodes_df = pd.read_parquet(node_map_path)
        nodes_dict = {row['ngram']: "N/A" for _, row in nodes_df.iterrows()}

        edge_df = pd.read_csv(edge_path)
        edge_counts = edge_df.groupby(['source', 'target']).size().reset_index(name='count')

        # Calculate transition probabilities for edge weights
        source_totals = edge_df.groupby('source').size().to_dict()
        edge_counts['weight'] = edge_counts.apply(lambda row: row['count'] / source_totals.get(row['source'], 1), axis=1)

        weighted_edge_list = list(edge_counts[['source', 'target', 'weight']].to_records(index=False))

        # 2. Instantiate the DirectedNgramGraph object
        print(f"Instantiating DirectedNgramGraph object for n={n_val}...")
        graph_object = graph_processor.DirectedNgramGraph(nodes=nodes_dict, edges=weighted_edge_list)

        # 3. Save the final graph object using pickle
        output_filepath = os.path.join(config.GRAPH_OBJECTS_DIR, f"graph_n{n_val}.pkl")
        with open(output_filepath, 'wb') as f:
            pickle.dump(graph_object, f, pickle.HIGHEST_PROTOCOL)
        print(f"SUCCESS: Graph object for n={n_val} saved to: {output_filepath}")

        del nodes_df, edge_df, graph_object;
        gc.collect()

    # --- PART 3: Cleanup ---
    print("\n>>> Phase 3: Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
        print(f"Successfully removed temporary directory: {temp_dir}")
    except OSError as e:
        print(f"Error removing temporary directory {temp_dir}: {e.strerror}")

    print("\n### PIPELINE STEP 1 FINISHED ###")
