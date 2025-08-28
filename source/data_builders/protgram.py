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
import time # noqa
from functools import partial
from typing import Tuple, Iterator, List, Dict
from pathlib import Path
import dask.bag as db
import dask.dataframe as dd
import pyarrow.parquet as pq
import pandas as pd
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
        self._resource_alert()

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
        os.makedirs(self.temp_dir, exist_ok=True) # noqa
        os.makedirs(self.output_dir, exist_ok=True) # noqa
        print(f"Temporary files will be stored in: {self.temp_dir}")
        print(f"Final graph objects will be saved to: {self.output_dir}")

        # --- NEW: Wrap the core logic in a try...finally block to guarantee cleanup of the temp directory ---
        try:
            n_values = range(1, self.n_max + 1)
            effective_dask_workers = self.num_workers_config if self.num_workers_config > 1 else 1 # noqa

            def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]: # noqa
                """A generator that streams sequences from FASTA files, applying cleaning as configured."""
                sequence_iterator = FastaUtils.parse_sequences(
                    self.protein_sequence_files,
                    perform_cleaning=self.config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
                    min_len=self.config.PROTGRAM_FASTA_MIN_LEN,
                    max_len=self.config.PROTGRAM_FASTA_MAX_LEN,
                    alphabet_type=self.config.PROTGRAM_FASTA_ALPHABET
                )

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

            # --- FIX: Add a check for empty input after downsampling/filtering ---
            # This prevents Dask from erroring on empty partitions, which can happen
            # in tests or if the downsample fraction is very small.
            try:
                if final_preprocessed_input_bag.count().compute() == 0:
                    print("\n--- ⚠️ WARNING: No sequences remained after downsampling/filtering. ---")
                    print("--- The input FASTA file might be too small for the specified SEQUENCE_DOWNSAMPLE_FRACTION,")
                    print("--- or all sequences were filtered out by length constraints.")
                    print("--- Skipping n-gram graph generation. ---")
                    return # The 'finally' block will still run for cleanup.
            except Exception:
                print("--- WARNING: Could not compute initial sequence count. Proceeding with build. ---")

            # --- REFACTOR: Implement a 2-pass Dask pipeline for massive performance improvement ---
            # Pass 1: Generate all n-gram maps for all levels in a single pass over the data.
            DataUtils.print_header("Phase 1: Generating All N-Gram Maps")
            phase1_start_time = time.monotonic() # --- DEFINITIVE FIX: Remove redundant local functions and use the centralized helpers ---
            extract_all_ngrams_partial = partial(ProtgramDaskHelpers.extract_all_ngrams_from_sequence_tuple, n_max=self.n_max) # noqa
            all_ngrams_bag = final_preprocessed_input_bag.map(extract_all_ngrams_partial).flatten() # noqa
            all_ngrams_ddf = all_ngrams_bag.to_dataframe(meta={'n': 'i4', 'ngram': 'str'}) # noqa

            for n in tqdm(n_values, desc="Building N-Gram Levels"):
                print(f"  - Creating map for n={n}...") # noqa
                ngrams_for_n_ddf = all_ngrams_ddf[all_ngrams_ddf['n'] == n]
                ngram_map_ddf = ngrams_for_n_ddf[['ngram']].drop_duplicates().reset_index(drop=True)
                ngram_map_ddf['id'] = ngram_map_ddf.index
                output_ngram_map_path = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
                # --- DEFINITIVE FIX: Write the map to disk but DO NOT load it into memory ---
                # The .compute() call was the source of the OOM error on large datasets.
                # We will now read these maps back from disk in a scalable way in Phase 2.
                ngram_map_ddf.to_parquet(output_ngram_map_path, write_index=False, engine='pyarrow', overwrite=True) # noqa
                # We can get the size from metadata without loading the whole file.
                num_unique_ngrams = len(pd.read_parquet(output_ngram_map_path, columns=['id']))
                print(f"    Unique n-gram map for n={n} (size: {num_unique_ngrams:,}) created.")
            print(f"<<< Phase 1 finished in {time.monotonic() - phase1_start_time:.2f}s.")

            # Pass 2: Generate all edges for all levels in a single pass over the data.
            DataUtils.print_header("Phase 2: Generating All Edges")
            phase2_start_time = time.monotonic()
            # --- REFACTOR: Replace in-memory mapping with scalable Dask joins ---
            for n in tqdm(n_values, desc="Aggregating Edges"):
                print(f"  - Aggregating edges for n={n}...")
                # 1. Extract string-based edges for the current n-gram level
                extract_edges_for_n_partial = partial(ProtgramDaskHelpers.extract_string_edges_for_n, n=n)
                string_edges_ddf = final_preprocessed_input_bag.map(extract_edges_for_n_partial).flatten().to_dataframe(meta={'source_str': 'str', 'target_str': 'str'})

                # 2. Load the corresponding n-gram map from disk
                ngram_map_path = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
                ngram_map_ddf = dd.read_parquet(ngram_map_path)

                # 3. Perform two joins to map string edges to integer IDs
                # Join for source nodes
                merged_source = string_edges_ddf.merge(ngram_map_ddf, left_on='source_str', right_on='ngram', how='inner')
                merged_source = merged_source.rename(columns={'id': 'source'}).drop(columns=['ngram', 'source_str'])
                # Join for target nodes
                merged_target = merged_source.merge(ngram_map_ddf, left_on='target_str', right_on='ngram', how='inner')
                edges_for_n_ddf = merged_target.rename(columns={'id': 'target'}).drop(columns=['ngram', 'target_str'])

                # 4. Aggregate and save the weighted edges
                weighted_edges_ddf = edges_for_n_ddf.groupby(['source', 'target']).size().to_frame('weight')
                temp_edge_file_path = os.path.join(self.temp_dir, f"aggregated_edges_n{n}.parquet")
                weighted_edges_ddf.to_parquet(temp_edge_file_path, engine='pyarrow', write_index=True, overwrite=True)
            print(f"<<< Phase 2 finished in {time.monotonic() - phase2_start_time:.2f}s.")

            # --- Phase 2: Build and save final graph objects ---
            DataUtils.print_header("Phase 3: Building and saving final graph objects")
            phase3_start_time = time.monotonic()
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

                del graph_object, idx_to_node # noqa
                gc.collect()

            print(f"<<< Phase 3 finished in {time.monotonic() - phase3_start_time:.2f}s.")

        # --- DEFINITIVE FIX: Ensure cleanup runs by placing it in the finally block ---
        # --- FIX: Move cleanup to a finally block to ensure it always runs ---
        finally:
            # --- Phase 3: Cleanup ---
            DataUtils.print_header("Phase 4: Cleaning up temporary files")
            phase4_start_time = time.monotonic()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"  Temporary directory {self.temp_dir} cleaned up.")
            print(f"<<< Phase 4 finished in {time.monotonic() - phase4_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.monotonic() - overall_start_time:.2f}s")

    def _resource_alert(self):
        """Prints a warning about potential disk usage."""
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
