import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from tqdm import tqdm
import pickle
import shutil
import time

# Correctly import the Config class and the graph utility
from src.utils.graph_processor import DirectedNgramGraph
from src.config import Config


def save_object(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def _create_intermediate_files(n, temp_dir, protein_sequence_file, chunk_size):
    """
    Creates intermediate node and edge files using Dask for parallel processing.
    """
    print(f"[Worker n={n}]: Starting intermediate file construction.")

    output_ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
    output_edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

    print(f"  Pass 1 (n={n}): Discovering unique n-grams with Dask...")
    df = dd.read_csv(protein_sequence_file, header=None, names=['sequence'], blocksize=chunk_size)

    def get_ngrams(sequence):
        if not isinstance(sequence, str): return []
        return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]

    all_ngrams = df['sequence'].dropna().map_partitions(lambda s: s.apply(get_ngrams), meta=(None, 'object')).explode().unique()
    unique_ngrams_df = all_ngrams.to_frame(name='ngram').compute()
    unique_ngrams_df['id'] = range(len(unique_ngrams_df))
    unique_ngrams_df.to_parquet(output_ngram_map_file)
    print(f"  Pass 1 (n={n}) Complete. Unique n-grams saved to: {output_ngram_map_file}")

    print(f"  Pass 2 (n={n}): Generating raw edge list and saving to {output_edge_file}...")
    ngram_map = pd.read_parquet(output_ngram_map_file).set_index('ngram')['id'].to_dict()

    def row_to_edges(row):
        sequence = row['sequence']
        if not isinstance(sequence, str): return []

        edges = []
        ngrams = [sequence[i:i + n] for i in range(len(sequence) - n + 1)]
        for i in range(len(ngrams) - 1):
            source_ngram = ngrams[i]
            target_ngram = ngrams[i + 1]
            if source_ngram in ngram_map and target_ngram in ngram_map:
                edges.append(f"{ngram_map[source_ngram]} {ngram_map[target_ngram]}\n")
        return edges

    with open(output_edge_file, 'w') as f:
        for part in df.partitions:
            computed_part = part.compute()
            for _, row in computed_part.iterrows():
                edges = row_to_edges(row)
                f.writelines(edges)

    print(f"  Pass 2 (n={n}) Complete: Raw edge file has been created.")
    print(f"[Worker n={n}]: Successfully completed file construction.")


def run_graph_building(config: Config):
    """
    Main function to orchestrate the graph building process.
    """


    protein_sequence_file = config.GCN_INPUT_FASTA_PATH
    output_dir = config.GRAPH_OBJECTS_DIR
    n_max = config.GCN_NGRAM_MAX_N
    num_workers = config.GRAPH_BUILDER_WORKERS
    chunk_size = config.DASK_CHUNK_SIZE

    # Create a temporary directory for intermediate files
    temp_dir = os.path.join(config.BASE_OUTPUT_DIR, "temp_graph_builder")

    print_header("PIPELINE STEP 1: Building N-gram Graphs")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    n_values = range(1, n_max + 1)

    print("\n>>> Phase 1: Creating intermediate node and edge files in parallel...")
    print(f"Using {num_workers} worker processes.")

    with LocalCluster(n_workers=num_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
        print(f"Dask dashboard link: {client.dashboard_link}")
        tasks = [client.submit(_create_intermediate_files, n, temp_dir, protein_sequence_file, chunk_size) for n in n_values]
        for future in tqdm(tasks, desc="Dask Workers Progress"):
            future.result()

    print("\nSUCCESS: All intermediate files created successfully.")

    print("\n>>> Phase 2: Building and saving final graph objects...")
    for n in n_values:
        print(f"\n--- Processing n = {n} ---")

        ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
        edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

        nodes_df = pd.read_parquet(ngram_map_file)
        nodes_dict = nodes_df.set_index('id')['ngram'].to_dict()
        print(f"Created node indices for {len(nodes_dict)} unique n-gram nodes.")

        print("Calculating edge transition frequencies...")
        edge_df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'])
        weighted_edge_list = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')

        print(f"Loaded {len(weighted_edge_list)} unique weighted edges.")

        print(f"Instantiating DirectedNgramGraph object for n={n}...")
        graph_object = DirectedNgramGraph(nodes=nodes_dict, edges=weighted_edge_list)

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
    print(f"\n{border}")
    print(f"### {title} ###")
    print(f"{border}\n")
