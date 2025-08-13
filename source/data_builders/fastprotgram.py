# ==============================================================================
# MODULE: data_builders/fastprotgram.py
# PURPOSE: A memory-efficient, scalable graph builder using a fully Dask-based architecture.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Iterator

import dask.bag as db
import dask.dataframe as dd
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from configuration.config import Config
from source.data_structures.graph import DirectedNgramGraph
from source.utils.data import DataUtils, FastaUtils, ProtgramDaskHelpers


class FastProtGramDataBuilder:
    """
    A memory-efficient, scalable graph builder using a fully Dask-based architecture.
    This implementation avoids in-memory collection of unique n-grams and uses
    parallel DataFrame merges to construct the graph, making it suitable for
    very large datasets like UniRef50.
    """

    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_files = [str(p) for p in config.SEQUENCE_FILE_PATHS]
        self.output_dir = str(config.RESULTS_GRAPH_OBJECTS_DIR)
        self.n_max = config.PROTGRAM_NGRAM_MAX_N
        self.num_workers_config = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

        print(
            f"FastGraphBuilder initialized: n_max={self.n_max}, configured_workers={self.num_workers_config}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"FastGraphBuilder Initialized (Output: {self.output_dir})")

    def run(self) -> None:
        """
        Main execution function for the scalable graph builder.
        """
        overall_start_time = time.monotonic()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs (Scalable Mode)")

        # --- Validation of existing graphs ---
        all_graphs_exist_and_are_valid = True
        for n in range(1, self.n_max + 1):
            expected_graph_file = os.path.join(self.output_dir, f"ngram_graph_n{n}.pkl")
            if not os.path.exists(expected_graph_file):
                all_graphs_exist_and_are_valid = False
                print(f"  Info: Graph file for n={n} not found. Will proceed with full build.")
                break
            else:
                print(f"  Validating existing graph file for n={n}...")
                graph_obj = DataUtils.load_object(expected_graph_file)
                if graph_obj is None or not hasattr(graph_obj, 'number_of_edges') or graph_obj.number_of_edges == 0:
                    print(f"  - Validation FAILED for n={n}: Graph is empty or corrupt. Forcing rebuild.")
                    all_graphs_exist_and_are_valid = False
                    os.remove(expected_graph_file)
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

        n_values = range(1, self.n_max + 1)
        effective_dask_workers = self.num_workers_config
        dask_scheduler_general = 'threads' if effective_dask_workers > 1 else 'sync'

        if self.num_workers_config > 1:
            print("\n" + "=" * 80)
            print(f"FastGraphBuilder is configured for parallel processing (GRAPH_BUILDER_WORKERS={self.num_workers_config}).")
            print(f"Using Dask with a THREADED scheduler ({effective_dask_workers} threads).")
            print("=" * 80 + "\n")
        else:
            print("\nFastGraphBuilder is configured for synchronous (single-threaded) execution.\n")

        def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]:
            first_sequence = True
            for seq_tuple in FastaUtils.parse_sequences(self.protein_sequence_files):
                yield seq_tuple, first_sequence
                if first_sequence:
                    first_sequence = False

        sequence_generator = get_preprocessed_sequence_stream()
        num_partitions_for_bag = effective_dask_workers if effective_dask_workers > 1 else 1
        raw_sequence_bag_with_flag = db.from_sequence(sequence_generator, npartitions=num_partitions_for_bag)
        final_preprocessed_input_bag = raw_sequence_bag_with_flag.starmap(ProtgramDaskHelpers.preprocess_sequence_tuple_for_bag)

        # --- ARCHITECTURAL REFACTOR: Process all n-gram levels in a scalable Dask pipeline ---
        for n in n_values:
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

            # Persist the map in memory for the merge operations. This is a calculated risk;
            # for extremely large vocabularies, this could still be a bottleneck, but it's
            # far more efficient than passing a dict to each task.
            ngram_map_ddf = ngram_map_ddf.persist()
            # --- DEFINITIVE FIX for OOM Crash: Compute the map once and use it directly ---
            # This avoids expensive, memory-intensive merge/join operations on string columns.
            ngram_to_id_map = ngram_map_ddf.compute().set_index('ngram')['id'].to_dict()
            print(f"    Unique n-gram map for n={n} (size: {len(ngram_to_id_map)}) computed and loaded into memory.")

            # 2. Generate edge pairs (source_ngram, target_ngram)
            print(f"  [n={n}] Generating edge pairs and mapping to IDs simultaneously...")
            # This new helper function yields integer pairs directly, which is far more memory-efficient.
            extract_edges_partial = partial(
                ProtgramDaskHelpers.extract_edges_from_sequence_tuple,
                n_val=n,
                ngram_to_id_map=ngram_to_id_map
            )
            # The bag now contains strings like "source_id target_id"
            edge_id_str_bag = final_preprocessed_input_bag.map(extract_edges_partial).flatten()

            # --- DEFINITIVE FIX for Scalability & Stability: Use Parquet for intermediate storage ---
            # Instead of a complex in-memory conversion or inefficient text files, we use Dask's
            # optimized to_parquet/read_parquet workflow. This is both memory-safe and fast.
            temp_edge_parts_dir = os.path.join(self.temp_dir, f"edge_parts_n{n}")
            if os.path.exists(temp_edge_parts_dir):
                shutil.rmtree(temp_edge_parts_dir)

            # Map the bag of strings to a bag of dictionaries, the required format for to_parquet
            edge_id_dict_bag = edge_id_str_bag.map(
                lambda s: {'source': int(s.split()[0]), 'target': int(s.split()[1])}
            )
            # Write the bag of dictionaries to a Parquet dataset.
            edge_id_dict_bag.to_parquet(temp_edge_parts_dir, compute=True, engine='pyarrow')

            # Read the Parquet dataset back into a Dask DataFrame.
            edge_id_ddf = dd.read_parquet(temp_edge_parts_dir, engine='pyarrow')

            # 4. Aggregate edge weights
            print(f"  [n={n}] Aggregating edge weights...")
            weighted_edges_ddf = edge_id_ddf.groupby(['source', 'target']).size().to_frame('weight')

            # 5. Save the final aggregated edges to disk
            temp_edge_file_path = os.path.join(self.temp_dir, f"aggregated_edges_n{n}.parquet")
            print(f"  [n={n}] Saving final aggregated edges to disk...")
            weighted_edges_ddf.to_parquet(temp_edge_file_path, engine='pyarrow', write_index=True, overwrite=True)

            print(f"  Level n={n} processing finished in {time.monotonic() - level_start_time:.2f}s.")
            # Clean up the intermediate text files
            shutil.rmtree(temp_edge_parts_dir)
            del ngrams_bag, ngrams_ddf, ngram_map_ddf, edge_id_str_bag, edge_id_dict_bag, edge_id_ddf, weighted_edges_ddf, ngram_to_id_map, temp_edge_parts_dir
            gc.collect()

        # --- Phase 2: Build and save final graph objects ---
        DataUtils.print_header("Phase 2: Building and saving final graph objects")
        phase2_start_time = time.monotonic()
        for n in n_values:
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

            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"  Graph for n={n} saved to {output_path}")

            DataUtils.log_graph_statistics(graph_object, n)

            del graph_object, idx_to_node
            gc.collect()

        print(f"<<< Phase 2 finished in {time.monotonic() - phase2_start_time:.2f}s.")

        # --- Phase 3: Cleanup ---
        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.monotonic()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        print(f"<<< Phase 3 finished in {time.monotonic() - phase3_start_time:.2f}s.")
        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.monotonic() - overall_start_time:.2f}s")