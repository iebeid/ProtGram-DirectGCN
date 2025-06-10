# ==============================================================================
# MODULE: pipeline/graph_builder.py
# PURPOSE: Main function to orchestrate the graph building process.
# VERSION: 3.0 (Merged: Dask + ProtDiGCN Logic)
# ==============================================================================

import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from tqdm import tqdm
import pickle
import shutil
import time

# Correctly import the updated graph processor
from src.utils.graph_processor import DirectedNgramGraph
from src.config import Config


def save_object(obj, filename):
    """Saves a Python object to a file using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def _create_intermediate_files(n, temp_dir, protein_sequence_file, chunk_size):
    """
    Creates intermediate node and edge files using Dask for parallel processing.
    This version uses ASCII values for n-gram IDs to ensure consistency.
    """
    print(f"[Worker n={n}]: Starting intermediate file construction.")

    output_ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
    output_edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

    print(f"  Pass 1 (n={n}): Discovering unique n-grams with Dask...")
    df = dd.read_csv(protein_sequence_file, header=None, names=['sequence'], blocksize=chunk_size, sample=1000000)

    def get_ngrams(sequence):
        if not isinstance(sequence, str): return []
        return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]

    # Use Dask to find all unique n-grams in parallel
    all_ngrams = df['sequence'].dropna().map_partitions(lambda s: s.apply(get_ngrams), meta=(None, 'object')).explode().unique()
    unique_ngrams_df = all_ngrams.to_frame(name='ngram').compute()

    # Generate a unique, stable ID for each n-gram based on its ASCII representation
    # Using a simple integer index after sorting ensures consistency
    unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
    unique_ngrams_df['id'] = unique_ngrams_df.index

    unique_ngrams_df[['id', 'ngram']].to_parquet(output_ngram_map_file)
    print(f"  Pass 1 (n={n}) Complete. Unique n-gram map saved.")

    print(f"  Pass 2 (n={n}): Generating raw edge list...")
    ngram_to_id_map = pd.read_parquet(output_ngram_map_file).set_index('ngram')['id'].to_dict()

    def row_to_edges(row):
        sequence = row['sequence']
        if not isinstance(sequence, str) or len(sequence) < n + 1: return []
        edges = []
        # Generate transitions (edges)
        for i in range(len(sequence) - n):
            source_ngram = sequence[i:i + n]
            target_ngram = sequence[i + 1:i + 1 + n]
            source_id = ngram_to_id_map.get(source_ngram)
            target_id = ngram_to_id_map.get(target_ngram)
            if source_id is not None and target_id is not None:
                edges.append(f"{source_id} {target_id}\n")
        return edges

    with open(output_edge_file, 'w') as f:
        # Process the dataframe in partitions to manage memory
        for partition in tqdm(df.partitions, desc=f"Processing Partitions (n={n})"):
            computed_part = partition.compute()
            for _, row in computed_part.iterrows():
                edges = row_to_edges(row)
                f.writelines(edges)

    print(f"  Pass 2 (n={n}) Complete. Raw edge file created.")


def run_graph_building(config: Config):
    """Main function to orchestrate the graph building process."""
    # Configuration setup
    protein_sequence_file = config.GCN_INPUT_FASTA_PATH
    output_dir = config.GRAPH_OBJECTS_DIR
    n_max = config.GCN_NGRAM_MAX_N
    num_workers = config.GRAPH_BUILDER_WORKERS
    chunk_size = config.DASK_CHUNK_SIZE
    temp_dir = os.path.join(config.BASE_OUTPUT_DIR, "temp_graph_builder")

    print_header("PIPELINE STEP 1: Building N-gram Graphs")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    n_values = range(1, n_max + 1)

    print(f"\n>>> Phase 1: Creating intermediate files with {num_workers} workers...")
    # Setup and use a local Dask cluster for parallel file generation
    with LocalCluster(n_workers=num_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
        print(f"Dask dashboard link: {client.dashboard_link}")
        tasks = [client.submit(_create_intermediate_files, n, temp_dir, protein_sequence_file, chunk_size) for n in n_values]
        for future in tqdm(tasks, desc="Dask Workers Progress"):
            future.result()  # Wait for each parallel task to complete

    print("\n>>> Phase 2: Building and saving final graph objects...")
    for n in n_values:
        print(f"\n--- Processing n = {n} ---")
        ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
        edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

        # Create node mappings from the parquet file
        nodes_df = pd.read_parquet(ngram_map_file)
        idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()

        # Read the raw edge list
        edge_df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int)

        # Aggregate to get unique edges with raw counts (weights)
        weighted_edge_list = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
        print(f"Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_list)} unique weighted edges.")

        print(f"Instantiating DirectedNgramGraph object for n={n}...")
        # The graph object now only needs the node map and the raw weighted edges
        graph_object = DirectedNgramGraph(nodes=idx_to_node, edges=weighted_edge_list)

        output_path = os.path.join(output_dir, f'ngram_graph_n{n}.pkl')
        save_object(graph_object, output_path)
        print(f"Graph for n={n} saved to {output_path}")

    print("\n>>> Phase 3: Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")
    print_header("N-gram Graph Building FINISHED")


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}\n### {title} ###\n{border}\n")
