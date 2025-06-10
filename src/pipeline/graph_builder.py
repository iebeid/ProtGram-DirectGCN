# ==============================================================================
# MODULE: pipeline/graph_builder.py
# PURPOSE: Main function to orchestrate the graph building process.
# VERSION: 3.1 (Merged with error handling and empty DataFrame checks)
# ==============================================================================

import os
import shutil
import pickle
import pandas as pd
import pyarrow
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

# Assuming these are correctly located in your project structure
from src.utils.graph_processor import DirectedNgramGraph
from src.config import Config


def save_object(obj, filename):
    """Saves a Python object to a file using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}\n### {title} ###\n{border}\n")


def _create_intermediate_files(n, temp_dir, protein_sequence_file, chunk_size):
    """
    Creates intermediate node and edge files using Dask for parallel processing.
    """
    print(f"[Worker n={n}]: Starting intermediate file construction.")

    output_ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
    output_edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

    print(f"  Pass 1 (n={n}): Discovering unique n-grams with Dask...")
    df = dd.read_csv(protein_sequence_file, header=None, names=['sequence'], blocksize=chunk_size, sample=1000000)

    def get_ngrams(sequence):
        if not isinstance(sequence, str):
            return []
        return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]

    all_ngrams = df['sequence'].dropna().map_partitions(lambda s: s.apply(get_ngrams), meta=(None, 'object')).explode().unique()
    unique_ngrams_df = all_ngrams.to_frame(name='ngram').compute()

    # =================================================================================
    # ✨ NEW: Check if the n-gram DataFrame is empty before saving.
    # =================================================================================
    if unique_ngrams_df.empty:
        print(f"⚠️ Warning: No n-grams of size n={n} were generated. This may be expected for small proteins or large n.")
        print(f"An empty map file will be created at: {output_ngram_map_file}")
        # Create an empty but valid Parquet file to prevent downstream errors
        empty_df = pd.DataFrame({'id': pd.Series(dtype='int'), 'ngram': pd.Series(dtype='str')})
        empty_df.to_parquet(output_ngram_map_file)
        # Create an empty edge file as well
        open(output_edge_file, 'w').close()
        print(f"  Pass 1 & 2 (n={n}) Complete. Empty intermediate files created.")
        return  # Exit the function early for this n-gram size

    unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
    unique_ngrams_df['id'] = unique_ngrams_df.index
    unique_ngrams_df[['id', 'ngram']].to_parquet(output_ngram_map_file)
    print(f"  Pass 1 (n={n}) Complete. Unique n-gram map saved.")

    print(f"  Pass 2 (n={n}): Generating raw edge list...")
    ngram_to_id_map = pd.read_parquet(output_ngram_map_file).set_index('ngram')['id'].to_dict()

    def row_to_edges(row):
        sequence = row['sequence']
        if not isinstance(sequence, str) or len(sequence) < n + 1:
            return []
        edges = []
        for i in range(len(sequence) - n):
            source_ngram = sequence[i:i + n]
            target_ngram = sequence[i + 1:i + 1 + n]
            source_id = ngram_to_id_map.get(source_ngram)
            target_id = ngram_to_id_map.get(target_ngram)
            if source_id is not None and target_id is not None:
                edges.append(f"{source_id} {target_id}\n")
        return edges

    with open(output_edge_file, 'w') as f:
        for partition in tqdm(df.partitions, desc=f"Processing Partitions (n={n})"):
            computed_part = partition.compute()
            for _, row in computed_part.iterrows():
                edges = row_to_edges(row)
                f.writelines(edges)

    print(f"  Pass 2 (n={n}) Complete. Raw edge file created.")


def run_graph_building(config: Config):
    """Main function to orchestrate the graph building process."""
    protein_sequence_file = config.GCN_INPUT_FASTA_PATH
    output_dir = config.GRAPH_OBJECTS_DIR
    n_max = config.GCN_NGRAM_MAX_N
    num_workers = config.GRAPH_BUILDER_WORKERS
    chunk_size = config.DASK_CHUNK_SIZE
    temp_dir = os.path.join(config.BASE_OUTPUT_DIR, "temp_graph_builder")

    print_header("PIPELINE STEP 1: Building N-gram Graphs")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    n_values = range(1, n_max + 1)

    print(f"\n>>> Phase 1: Creating intermediate files with {num_workers} workers...")
    with LocalCluster(n_workers=num_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
        print(f"Dask dashboard link: {client.dashboard_link}")
        tasks = [client.submit(_create_intermediate_files, n, temp_dir, protein_sequence_file, chunk_size) for n in n_values]
        for future in tqdm(tasks, desc="Dask Workers Progress"):
            future.result()

    print("\n>>> Phase 2: Building and saving final graph objects...")
    for n in n_values:
        print(f"\n--- Processing n = {n} ---")
        ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
        edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

        # =================================================================================
        # ✨ NEW: Add robust error handling for reading Parquet files.
        # =================================================================================
        try:
            nodes_df = pd.read_parquet(ngram_map_file)
        except pyarrow.lib.ArrowInvalid as e:
            print(f"❌ Error: Could not read the Parquet file for n={n} at: {ngram_map_file}")
            print(f"   ArrowInvalid Error Details: {e}")
            print("   This file might be corrupted or empty in an invalid format. Skipping this n-gram size.")
            continue

        if nodes_df.empty:
            print(f"ℹ️ Info: The n-gram map for n={n} is empty. No graph will be generated. Skipping.")
            continue

        idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()
        edge_df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int)

        if edge_df.empty:
            print(f"ℹ️ Info: No edges found for n={n}. Creating a graph with nodes but no edges.")
            weighted_edge_list = pd.DataFrame(columns=['source', 'target', 'weight'])
        else:
            weighted_edge_list = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
            print(f"Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_list)} unique weighted edges.")

        print(f"Instantiating DirectedNgramGraph object for n={n}...")
        graph_object = DirectedNgramGraph(nodes=idx_to_node, edges=weighted_edge_list)

        output_path = os.path.join(output_dir, f'ngram_graph_n{n}.pkl')
        save_object(graph_object, output_path)
        print(f"Graph for n={n} saved to {output_path}")

    print("\n>>> Phase 3: Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Cleanup complete.")
    print_header("N-gram Graph Building FINISHED")
