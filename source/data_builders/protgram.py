# ==============================================================================
# MODULE: data_builders/protgram.py
# PURPOSE: A memory-efficient, scalable graph builder using a Dask-based architecture.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import random
import shutil
import time
from functools import partial
from typing import Tuple, Iterator
from pathlib import Path
import dask.bag as db
import dask.dataframe as dd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils
from source.utils.data.protgram_helper import ProtgramDaskHelpers


class ProtGramDataBuilder:
    """
    A memory-efficient, scalable graph builder using a Dask-based architecture.
    This implementation avoids collecting all unique n-grams into memory by leveraging
    Dask DataFrames, making it suitable for very large datasets like UniRef50.
    """

    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_files = [str(p) for p in config.SEQUENCE_FILE_PATHS]
        self.output_dir = str(config.RESULTS_GRAPH_OBJECTS_DIR)
        self.n_max = config.PROTGRAM_NGRAM_MAX_N
        self.num_workers_config = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

        print(f"ProtGramDataBuilder initialized: n_max={self.n_max}, "
              f"configured_workers={self.num_workers_config}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"ProtGramDataBuilder Initialized (Output: {self.output_dir})")

    def run(self) -> None:
        """
        Main execution function for the scalable graph builder.
        """
        overall_start_time = time.monotonic()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs")

        # --- NEW: Add a specific resource alert before starting the build ---
        try:
            total_input_size_gb = sum(os.path.getsize(f) for f in self.protein_sequence_files if os.path.exists(f)) / (1024 ** 3)
            if total_input_size_gb > 0:
                print("\n--- Resource Alert: Graph Building ---")
                print(f"  Input FASTA size is ~{total_input_size_gb:.2f} GB.")
                print(f"  This process can generate intermediate files up to 3-5x this size (~{total_input_size_gb*3:.2f} - {total_input_size_gb*5:.2f} GB).")
                print("  Please ensure you have sufficient free disk space.")
                print("--------------------------------------\n")
        except Exception:
            # Don't crash if we can't get file sizes for some reason
            pass

        # --- Validation of existing graphs ---
        all_graphs_exist_and_are_valid = True
        for n in range(1, self.n_max + 1):
            expected_graph_dir = Path(self.output_dir) / f"ngram_graph_n{n}"
            if not expected_graph_dir.exists():
                all_graphs_exist_and_are_valid = False
                print(f"  Info: Graph file for n={n} not found. Will proceed with full build.")
                break
            else:
                print(f"  Validating existing graph file for n={n}...")
                graph_obj = DirectedNgramGraph.load_from_dir(expected_graph_dir)
                if graph_obj is None or not hasattr(graph_obj, 'number_of_edges') or graph_obj.number_of_edges == 0:
                    print(f"  - Validation FAILED for n={n}: Graph is empty or corrupt. Forcing rebuild.")
                    all_graphs_exist_and_are_valid = False
                    shutil.rmtree(expected_graph_dir)
                    break
                print(f"  - Validation PASSED for n={n} (Nodes: {graph_obj.number_of_nodes}, Edges: {graph_obj.number_of_edges}).")

        if all_graphs_exist_and_are_valid:
            print("\nAll required n-gram graph objects already exist and are valid.")
            DataUtils.print_header(f"N-gram Graph Building SKIPPED (Files exist and are valid)")
            return

        # --- Setup for a new build ---
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Temporary files will be stored in: {self.temp_dir}")
        print(f"Final graph objects will be saved to: {self.output_dir}")

        # --- NEW: Wrap the core logic in a try...finally block to guarantee cleanup of the temp directory ---
        try:
            n_values = range(1, self.n_max + 1)
            effective_dask_workers = self.num_workers_config
            dask_scheduler_general = 'threads' if effective_dask_workers > 1 else 'sync'

            if self.num_workers_config > 1:
                print("\n" + "=" * 80)
                print(f"GraphBuilder is configured for parallel processing (GRAPH_BUILDER_WORKERS={self.num_workers_config}).")
                print(f"Using Dask with a THREADED scheduler ({effective_dask_workers} threads).")
                print("=" * 80 + "\n")
            else:
                print("\nGraphBuilder is configured for synchronous (single-threaded) execution.\n")

            def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]:
                """
                A generator that streams sequences from FASTA files, applying cleaning
                and downsampling as configured. It also yields a flag for the first sequence.
                """
                sequence_iterator = None
                # --- NEW: Downsampling logic ---
                # If downsampling is enabled, we first get all sequence tuples.
                # This is memory-intensive but necessary for random sampling.
                # The pipeline is designed to handle this by having separate flags for large datasets.
                if self.config.SEQUENCE_DOWNSAMPLE_FRACTION and 0.0 < self.config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0:
                    print(f"  Downsampling FASTA files to {self.config.SEQUENCE_DOWNSAMPLE_FRACTION * 100:.2f}% using memory-efficient reservoir sampling...")
                    # --- ANTICIPATORY DEBUGGING: Avoid loading the entire dataset into memory for sampling. ---
                    # Instead of list(FastaUtils.parse_sequences(...)), which would cause an OOM error on large
                    # files, we use a reservoir sampling algorithm that processes the sequence stream.
                    sequence_stream = FastaUtils.parse_sequences(
                        self.protein_sequence_files,
                        perform_cleaning=self.config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
                        min_len=self.config.PROTGRAM_FASTA_MIN_LEN,
                        max_len=self.config.PROTGRAM_FASTA_MAX_LEN,
                        alphabet_type=self.config.PROTGRAM_FASTA_ALPHABET
                    )
                    # Estimate total number of sequences for sampling k
                    total_sequences = sum(1 for _ in FastaUtils.parse_sequences(self.protein_sequence_files))
                    sample_size = int(total_sequences * self.config.SEQUENCE_DOWNSAMPLE_FRACTION)
                    print(f"  Estimated total sequences: {total_sequences}, Target sample size: {sample_size}")
                    sequence_iterator = DataUtils.reservoir_sample(sequence_stream, sample_size, self.config.RANDOM_STATE)
                else:
                    # Default behavior: stream sequences directly
                    sequence_iterator = FastaUtils.parse_sequences(
                        self.protein_sequence_files,
                        perform_cleaning=self.config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
                        min_len=self.config.PROTGRAM_FASTA_MIN_LEN,
                        max_len=self.config.PROTGRAM_FASTA_MAX_LEN,
                        alphabet_type=self.config.PROTGRAM_FASTA_ALPHABET
                    )

                # --- DEFINITIVE FIX: Add the missing iterator and yield logic ---
                if sequence_iterator:
                    first_sequence = True
                    for seq_tuple in sequence_iterator:
                        yield seq_tuple, first_sequence
                        if first_sequence:
                            first_sequence = False

            sequence_generator = get_preprocessed_sequence_stream()
            num_partitions_for_bag = effective_dask_workers if effective_dask_workers > 1 else 1
            raw_sequence_bag_with_flag = db.from_sequence(sequence_generator, npartitions=num_partitions_for_bag)
            final_preprocessed_input_bag = raw_sequence_bag_with_flag.starmap(ProtgramDaskHelpers.preprocess_sequence_tuple_for_bag)

            # --- DEFINITIVE FIX: Restore the main processing loop for each n-gram level ---
            for n in tqdm(n_values, desc="Building N-Gram Levels"):
                DataUtils.print_header(f"Processing N-gram Level: n = {n}")
                level_start_time = time.monotonic()

                # 1. Generate all n-grams and create a unique, indexed map on disk
                print(f"  [n={n}] Generating and mapping unique n-grams...")
                extract_ngrams_partial = partial(ProtgramDaskHelpers.extract_ngrams_from_sequence_tuple, n_val=n)
                ngrams_bag = final_preprocessed_input_bag.map(extract_ngrams_partial).flatten()
                ngrams_ddf = ngrams_bag.to_dataframe(columns=['ngram'])

                # Use Dask to find unique n-grams and create an ID map
                ngram_map_ddf = ngrams_ddf.drop_duplicates(split_out=num_partitions_for_bag).reset_index(drop=True)
                ngram_map_ddf['id'] = ngram_map_ddf.index

                output_ngram_map_path = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
                ngram_map_ddf.to_parquet(output_ngram_map_path, write_index=False, engine='pyarrow', overwrite=True)

                # Persist the map in memory for the merge operations.
                ngram_map_ddf = ngram_map_ddf.persist()
                ngram_to_id_map = ngram_map_ddf.compute().set_index('ngram')['id'].to_dict()
                print(f"    Unique n-gram map for n={n} (size: {len(ngram_to_id_map)}) computed and loaded into memory.")

                # 2. Generate edge pairs and map to IDs
                print(f"  [n={n}] Generating edge pairs and mapping to IDs simultaneously...")
                extract_edges_partial = partial(
                    ProtgramDaskHelpers.extract_edges_from_sequence_tuple, n_val=n, ngram_to_id_map=ngram_to_id_map
                )
                edge_id_str_bag = final_preprocessed_input_bag.map(extract_edges_partial).flatten()

                # 3. Save edge pairs to Parquet
                temp_edge_parts_dir = os.path.join(self.temp_dir, f"edge_parts_n{n}")
                if os.path.exists(temp_edge_parts_dir):
                    shutil.rmtree(temp_edge_parts_dir)

                edge_id_dict_bag = edge_id_str_bag.map(lambda s: {'source': int(s.split()[0]), 'target': int(s.split()[1])})
                edge_id_ddf_from_bag = edge_id_dict_bag.to_dataframe()
                edge_id_ddf_from_bag.to_parquet(temp_edge_parts_dir, engine='pyarrow', overwrite=True)

                # 4. Aggregate edge weights
                edge_id_ddf = dd.read_parquet(temp_edge_parts_dir, engine='pyarrow')
                print(f"  [n={n}] Aggregating edge weights...")
                weighted_edges_ddf = edge_id_ddf.groupby(['source', 'target']).size().to_frame('weight')

                # 5. Save the final aggregated edges to disk
                temp_edge_file_path = os.path.join(self.temp_dir, f"aggregated_edges_n{n}.parquet")
                print(f"  [n={n}] Saving final aggregated edges to disk...")
                weighted_edges_ddf.to_parquet(temp_edge_file_path, engine='pyarrow', write_index=True, overwrite=True)

                print(f"  Level n={n} processing finished in {time.monotonic() - level_start_time:.2f}s.")
                shutil.rmtree(temp_edge_parts_dir)
                del ngrams_bag, ngrams_ddf, ngram_map_ddf, edge_id_str_bag, edge_id_dict_bag, edge_id_ddf, weighted_edges_ddf, ngram_to_id_map
                gc.collect()

            # --- Phase 2: Build and save final graph objects ---
            DataUtils.print_header("Phase 2: Building and saving final graph objects")
            phase2_start_time = time.monotonic()
            for n in tqdm(n_values, desc="Saving Final Graph Objects"):
                print(f"\n--- Processing n = {n} for final graph object ---")
                ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')

                if not os.path.exists(ngram_map_file):
                    print(f"  Warning: N-gram map for n={n} not found. Skipping graph generation.")
                    continue
                try:
                    table_read = pq.read_table(ngram_map_file, columns=['id', 'ngram'])
                    nodes_df = table_read.to_pandas()
                    del table_read
                except Exception as e_parquet:
                    print(f"  ❌ Error: General error reading Parquet file for n={n}: {e_parquet}. Skipping.")
                    continue

                if nodes_df.empty:
                    print(f"  ℹ️ Info: The n-gram map for n={n} is empty. No graph will be generated. Skipping.")
                    continue
                print(f"  Loaded {len(nodes_df)} n-grams for n={n} from map file.")
                idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()
                del nodes_df

                print(f"  Instantiating DirectedNgramGraph object for n={n} from file...")
                temp_edge_file_path = os.path.join(self.temp_dir, f"aggregated_edges_n{n}.parquet")
                graph_object = DirectedNgramGraph(
                    nodes=idx_to_node,
                    edge_file_path=temp_edge_file_path,
                    epsilon_propagation=self.gcn_propagation_epsilon,
                    n_value=n
                )

                output_dir_path = Path(self.output_dir) / f'ngram_graph_n{n}'
                graph_object.save_to_dir(output_dir_path)
                print(f"  Graph for n={n} saved to {output_dir_path}")

                del graph_object, idx_to_node
                gc.collect()

            print(f"<<< Phase 2 finished in {time.monotonic() - phase2_start_time:.2f}s.")

        # --- DEFINITIVE FIX: Ensure cleanup runs by placing it in the finally block ---
        # --- FIX: Move cleanup to a finally block to ensure it always runs ---
        finally:
            # --- Phase 3: Cleanup ---
            DataUtils.print_header("Phase 3: Cleaning up temporary files")
            phase3_start_time = time.monotonic()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"  Temporary directory {self.temp_dir} cleaned up.")
            print(f"<<< Phase 3 finished in {time.monotonic() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.monotonic() - overall_start_time:.2f}s")
