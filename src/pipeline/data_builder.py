# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 4.0 (Refactored into GraphBuilderPipeline class)
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
from src.utils.graph_utils import DirectedNgramGraphForGCN  # Corrected from graph_utils to graph
from src.config import Config
from src.utils.data_utils import DataUtils, DataLoader  # Corrected from data_utils to data_loader


class GraphBuilder:
    """
    Orchestrates the n-gram graph building process using Dask for parallel processing.
    """

    def __init__(self, config: Config):
        """
        Initializes the GraphBuilderPipeline.

        Args:
            config (Config): The configuration object for the pipeline.
        """
        self.config = config
        self.protein_sequence_file = str(config.GCN_INPUT_FASTA_PATH)  # Ensure path is string
        self.output_dir = str(config.GRAPH_OBJECTS_DIR)
        self.n_max = config.GCN_NGRAM_MAX_N
        self.num_workers = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.chunk_size = config.DASK_CHUNK_SIZE
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        # Epsilon for graph object, ensure it's in config or provide a default
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

    @staticmethod
    def _create_intermediate_files(n, temp_dir, protein_sequence_file, chunk_size):
        """
        Creates intermediate node and edge files using Dask for parallel processing.
        This method is static as it's intended to be submitted to Dask workers.
        """
        print(f"[Worker n={n}]: Starting intermediate file construction.")

        output_ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n}.parquet')
        output_edge_file = os.path.join(temp_dir, f'edge_list_n{n}.txt')

        print(f"  Pass 1 (n={n}): Discovering unique n-grams with Dask...")
        # Ensure protein_sequence_file is a string for dd.read_csv
        df = dd.read_csv(str(protein_sequence_file), header=None, names=['sequence'], blocksize=chunk_size, sample=1000000)

        def get_ngrams(sequence):
            if not isinstance(sequence, str):
                return []
            return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]

        all_ngrams = df['sequence'].dropna().map_partitions(lambda s: s.apply(get_ngrams), meta=(None, 'object')).explode().unique()
        unique_ngrams_df = all_ngrams.to_frame(name='ngram').compute()

        if unique_ngrams_df.empty:
            print(f"⚠️ Warning: No n-grams of size n={n} were generated. This may be expected for small proteins or large n.")
            print(f"An empty map file will be created at: {output_ngram_map_file}")
            empty_df = pd.DataFrame({'id': pd.Series(dtype='int'), 'ngram': pd.Series(dtype='str')})
            empty_df.to_parquet(output_ngram_map_file)
            open(output_edge_file, 'w').close()
            print(f"  Pass 1 & 2 (n={n}) Complete. Empty intermediate files created.")
            return

        unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
        unique_ngrams_df['id'] = unique_ngrams_df.index
        unique_ngrams_df[['id', 'ngram']].to_parquet(output_ngram_map_file)
        print(f"  Pass 1 (n={n}) Complete. Unique n-gram map saved.")

        print(f"  Pass 2 (n={n}): Generating raw edge list...")
        ngram_to_id_map = pd.read_parquet(output_ngram_map_file).set_index('ngram')['id'].to_dict()

        def row_to_edges(row_data):  # Renamed 'row' to 'row_data' to avoid conflict
            sequence = row_data['sequence']
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
                for _, r in computed_part.iterrows():  # Renamed 'row' to 'r'
                    edges = row_to_edges(r)
                    f.writelines(edges)

        print(f"  Pass 2 (n={n}) Complete. Raw edge file created.")

    def run(self):
        """Main function to orchestrate the graph building process."""
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs")

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        n_values = range(1, self.n_max + 1)

        print(f"\n>>> Phase 1: Creating intermediate files with {self.num_workers} workers...")
        # Ensure self.num_workers is at least 1
        effective_num_workers = max(1, self.num_workers)

        with LocalCluster(n_workers=effective_num_workers, threads_per_worker=1, silence_logs='error') as cluster, Client(cluster) as client:
            print(f"Dask dashboard link: {client.dashboard_link}")
            # Pass class static method to Dask
            tasks = [client.submit(GraphBuilder._create_intermediate_files,
                                   n, self.temp_dir, self.protein_sequence_file, self.chunk_size)
                     for n in n_values]
            for future in tqdm(tasks, desc="Dask Workers Progress"):
                future.result()  # Wait for each Dask task to complete

        print("\n>>> Phase 2: Building and saving final graph objects...")
        for n in n_values:
            print(f"\n--- Processing n = {n} ---")
            ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
            edge_file = os.path.join(self.temp_dir, f'edge_list_n{n}.txt')

            try:
                nodes_df = pd.read_parquet(ngram_map_file)
            except pyarrow.lib.ArrowInvalid as e:
                print(f"❌ Error: Could not read the Parquet file for n={n} at: {ngram_map_file}")
                print(f"   ArrowInvalid Error Details: {e}")
                print("   This file might be corrupted or empty in an invalid format. Skipping this n-gram size.")
                continue
            except FileNotFoundError:
                print(f"❌ Error: Parquet file not found for n={n} at: {ngram_map_file}. Skipping this n-gram size.")
                continue

            if nodes_df.empty:
                print(f"ℹ️ Info: The n-gram map for n={n} is empty. No graph will be generated. Skipping.")
                continue

            idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()

            try:
                edge_df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int)
            except pd.errors.EmptyDataError:
                print(f"ℹ️ Info: Edge file for n={n} is empty or contains no valid edges. Creating graph with no edges.")
                edge_df = pd.DataFrame(columns=['source', 'target'])  # Ensure edge_df is an empty DataFrame
            except FileNotFoundError:
                print(f"❌ Error: Edge file not found for n={n} at: {edge_file}. Assuming no edges.")
                edge_df = pd.DataFrame(columns=['source', 'target'])

            if edge_df.empty:
                print(f"ℹ️ Info: No edges found for n={n}. Creating a graph with nodes but no edges.")
                # Ensure weighted_edge_list is a list of tuples for DirectedNgramGraphForGCN
                weighted_edge_list_tuples = []
            else:
                weighted_edge_df = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
                print(f"Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_df)} unique weighted edges.")
                # Convert DataFrame to list of tuples: (source_idx, target_idx, weight)
                weighted_edge_list_tuples = [tuple(x) for x in weighted_edge_df[['source', 'target', 'weight']].to_numpy()]

            print(f"Instantiating DirectedNgramGraph object for n={n}...")
            # Pass epsilon from self.config (via self.gcn_propagation_epsilon)
            graph_object = DirectedNgramGraphForGCN(nodes=idx_to_node,
                                                    edges=weighted_edge_list_tuples,
                                                    epsilon_propagation=self.gcn_propagation_epsilon)

            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"Graph for n={n} saved to {output_path}")

        print("\n>>> Phase 3: Cleaning up temporary files...")
        if os.path.exists(self.temp_dir):  # Check if temp_dir exists before trying to remove
            shutil.rmtree(self.temp_dir)
            print("Cleanup complete.")
        else:
            print("Temporary directory not found, no cleanup needed or already cleaned.")
        DataUtils.print_header("N-gram Graph Building FINISHED")
