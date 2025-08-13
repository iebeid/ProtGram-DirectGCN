# ==============================================================================
# MODULE: data_builders/protgram.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 8.1 (Aligned run signature with main pipeline and removed obsolete evaluation logic)
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


class ProtGramDataBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_files = [str(p) for p in config.SEQUENCE_FILE_PATHS]
        self.output_dir = str(config.RESULTS_GRAPH_OBJECTS_DIR)
        self.n_max = config.PROTGRAM_NGRAM_MAX_N  # This is now the default max, can be overridden
        self.num_workers_config = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

        print(
            f"GraphBuilder initialized: n_max={self.n_max}, configured_workers={self.num_workers_config}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"GraphBuilder Initialized (Output: {self.output_dir})")

    def run(self) -> None:
        """
        Main execution function for the graph builder.

        This method orchestrates the entire graph construction process, from
        reading sequences to building and saving the final graph objects for each n-gram level.

        Returns:
            None
        """
        overall_start_time = time.monotonic()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs")

        # --- Check if all final graph objects already exist ---
        all_graphs_exist = True
        for n in range(1, self.n_max + 1):
            expected_graph_file = os.path.join(self.output_dir, f"ngram_graph_n{n}.pkl")
            if not os.path.exists(expected_graph_file):
                all_graphs_exist = False
                print(f"  Info: Graph file for n={n} not found. Will proceed with full build.")
                break

        if all_graphs_exist:
            print("\nAll required n-gram graph objects already exist in the output directory.")
            DataUtils.print_header(f"N-gram Graph Building SKIPPED (Files exist)")
            return
        # --- END Check ---

        if os.path.exists(self.temp_dir):
            print(f"Cleaning up existing temporary directory: {self.temp_dir}")
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
            print(
                f"GraphBuilder is configured for parallel processing (GRAPH_BUILDER_WORKERS={self.num_workers_config}).")
            print(
                f"Attempting to use Dask with a THREADED scheduler ({effective_dask_workers} threads) for most ops.")
            print("This aims to improve speed while potentially avoiding multiprocessing-related memory issues.")
            print("=" * 80 + "\n")
        else:
            print("\nGraphBuilder is configured for synchronous (single-threaded) execution.\n")

        def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]:
            first_sequence = True
            for seq_tuple in FastaUtils.parse_sequences(self.protein_sequence_files):
                yield seq_tuple, first_sequence
                if first_sequence:
                    first_sequence = False

        # --- FIX: Do not load the entire FASTA file into a list in memory. ---
        # Pass the generator directly to Dask for lazy, scalable processing.
        sequence_generator = get_preprocessed_sequence_stream()
        # The FastaUtils.parse_sequences handles file not found errors internally.
        # An empty generator will be handled gracefully by Dask and downstream logic.

        num_partitions_for_bag = effective_dask_workers if effective_dask_workers > 1 else 1
        raw_sequence_bag_with_flag = db.from_sequence(sequence_generator, npartitions=num_partitions_for_bag)
        preprocessed_sequence_bag_unpersisted = raw_sequence_bag_with_flag.starmap(
            ProtgramDaskHelpers.preprocess_sequence_tuple_for_bag)

        # --- DEFINITIVE FIX for OOM Killer: Do NOT persist the bag for large datasets. ---
        # This streams the data from disk for each n-gram level, trading speed for memory stability.
        final_preprocessed_input_bag = preprocessed_sequence_bag_unpersisted

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if dask_scheduler_general != 'sync':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations.")

        for n_val_loop in n_values:
            DataUtils.print_header(
                f"Processing N-gram Level n = {n_val_loop} (Dask scheduler general: {dask_scheduler_general})")
            phase1_level_start_time = time.monotonic()

            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            temp_edge_output_dir = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}_parts')

            print(f"  [n={n_val_loop}] Generating n-grams using Dask Bag (source: final_preprocessed_input_bag)...")
            sys.stdout.flush()

            extract_ngrams_partial = partial(ProtgramDaskHelpers.extract_ngrams_from_sequence_tuple, n_val=n_val_loop)
            all_ngrams_bag_flattened = final_preprocessed_input_bag.map(extract_ngrams_partial).flatten()

            print(f"    Computing unique n-grams using Dask Bag's distinct()...")
            try:
                unique_ngrams_list = list(all_ngrams_bag_flattened.distinct().compute(
                    scheduler=dask_scheduler_general, num_workers=effective_dask_workers
                ))
                print(f"    Found {len(unique_ngrams_list)} unique {n_val_loop}-grams.")

                if not unique_ngrams_list:
                    unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
                else:
                    unique_ngrams_df = pd.DataFrame(sorted(unique_ngrams_list), columns=['ngram'])
            except MemoryError as e_mem_distinct_ngrams:
                print(
                    f"  [n={n_val_loop}] MEMORY ERROR during Dask compute for distinct n-grams: {e_mem_distinct_ngrams}")
                continue
            except Exception as e_compute_distinct_ngrams:
                print(
                    f"  [n={n_val_loop}] ERROR during Dask compute for distinct n-grams: {e_compute_distinct_ngrams}")
                continue

            unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
            unique_ngrams_df['id'] = unique_ngrams_df.index

            try:
                table_to_write = pyarrow.Table.from_pandas(unique_ngrams_df[['id', 'ngram']], preserve_index=False)
                pq.write_table(table_to_write, output_ngram_map_file)
                del table_to_write
                print(f"  [n={n_val_loop}] N-gram map saved: {os.path.basename(output_ngram_map_file)}")
            except Exception as e_parquet_write:
                print(f"  [n={n_val_loop}] ERROR writing Parquet map: {e_parquet_write}")
                continue

            if unique_ngrams_df.empty:
                os.makedirs(temp_edge_output_dir, exist_ok=True)
                print(f"  [n={n_val_loop}] No n-grams, so no edges will be generated.")
                print(f"  Level n={n_val_loop} (Phase 1) finished in {time.monotonic() - phase1_level_start_time:.2f}s.")
                continue
            del unique_ngrams_df

            try:
                map_table_read = pq.read_table(output_ngram_map_file, columns=['ngram', 'id'])
                map_df_read = map_table_read.to_pandas()
                ngram_to_id_map = map_df_read.set_index('ngram')['id'].to_dict()
                del map_table_read, map_df_read
            except Exception as e_read_map:
                print(f"  [n={n_val_loop}] ERROR reading back ngram map: {e_read_map}. Skipping edge generation.")
                continue

            print(f"  [n={n_val_loop}] Generating edge strings using Dask Bag (source: final_preprocessed_input_bag)...")
            sys.stdout.flush()

            extract_edges_partial = partial(ProtgramDaskHelpers.extract_edges_from_sequence_tuple,
                                            n_val=n_val_loop,
                                            ngram_to_id_map=ngram_to_id_map)
            all_edges_str_bag_flattened = final_preprocessed_input_bag.map(extract_edges_partial).flatten()

            if os.path.exists(temp_edge_output_dir):
                shutil.rmtree(temp_edge_output_dir)

            # --- FIX: Prevent Dask race condition by pre-creating the directory ---
            # This ensures the directory exists before any worker tries to write to it.
            os.makedirs(temp_edge_output_dir, exist_ok=True)
            scheduler_for_to_textfiles = 'sync'
            print(
                f"    Writing edge strings to directory: {temp_edge_output_dir} using Dask scheduler: '{scheduler_for_to_textfiles}'...")

            try:
                all_edges_str_bag_flattened.to_textfiles(
                    os.path.join(temp_edge_output_dir, 'part-*.txt'),
                    compute=True,
                    scheduler=scheduler_for_to_textfiles
                )
                print(f"    Edge parts saved to directory {temp_edge_output_dir}.")
            except Exception as e_write_edges:
                print(f"  [n={n_val_loop}] ERROR writing edge list parts: {e_write_edges}")
                os.makedirs(temp_edge_output_dir, exist_ok=True)

            del ngram_to_id_map
            gc.collect()
            print(f"  Level n={n_val_loop} (Phase 1) finished in {time.monotonic() - phase1_level_start_time:.2f}s.")

        if dask_scheduler_general != 'sync':
            if original_cuda_visible_devices is None:
                if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
            print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")

        DataUtils.print_header("Phase 2: Building and saving final graph objects")
        phase2_start_time = time.monotonic()
        for n in n_values:
            print(f"\n--- Processing n = {n} for final graph object ---")
            ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
            edge_parts_dir = os.path.join(self.temp_dir, f'edge_list_n{n}_parts')

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

            print(f"  Loading and aggregating raw edges for n={n} using Dask DataFrame from parts...")
            weighted_edge_df_computed = pd.DataFrame(columns=['source', 'target', 'weight'])
            try:
                temp_edge_file_path = os.path.join(self.temp_dir, f"aggregated_edges_n{n}.parquet")
                edge_parts_glob = os.path.join(edge_parts_dir, 'part-*.txt')
                if os.path.exists(edge_parts_dir) and any(Path(edge_parts_dir).glob('part-*.txt')):
                    # --- FIX: Warn on bad lines instead of skipping silently to aid debugging. ---
                    ddf = dd.read_csv(edge_parts_glob, sep=' ', header=None, names=['source', 'target'], dtype=int, on_bad_lines='warn', blocksize='128MB')
                    print(f"    Dask DataFrame created for n={n} from part-files with {ddf.npartitions} partitions.") # --- DEFINITIVE FIX for Multi-Index Error: Perform groupby directly. ---
                    # The set_index call is not supported for multi-column indexes in Dask.
                    # The groupby operation itself will trigger the necessary shuffle, and to_parquet handles the memory.
                    print(f"    Aggregating edge weights...")
                    weighted_ddf = ddf.groupby(['source', 'target']).size().to_frame('weight')
                    # Write the final result directly to a Parquet file.
                    weighted_ddf.to_parquet(temp_edge_file_path, engine='pyarrow', write_index=True, compute=True)
                    print(f"    Finished computing aggregated weighted edges for n={n}.")
                else:
                    print(f"  ℹ️ Info: Edge parts directory for n={n} is empty or not found.")
            except Exception as e_ddf:
                print(f"  ❌ Error: Dask DataFrame processing error for edges n={n}: {e_ddf}. Assuming no edges.")

            print(f"  Instantiating DirectedNgramGraph object for n={n} from file...")
            graph_object = DirectedNgramGraph(
                nodes=idx_to_node,
                edge_file_path=temp_edge_file_path,
                epsilon_propagation=self.gcn_propagation_epsilon,
                n_value=n
            )

            if os.path.exists(temp_edge_file_path):
                os.remove(temp_edge_file_path)
            # --- END MEMORY OPTIMIZATION ---

            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"  Graph for n={n} saved to {output_path}")

            print(f"    --- Graph Statistics for n={n} ---")
            num_nodes = graph_object.number_of_nodes
            num_edges = graph_object.number_of_edges
            print(f"      Nodes: {num_nodes}")
            print(f"      Edges (unique weighted): {num_edges}")
            if num_nodes > 1:
                possible_edges_no_self_loops = num_nodes * (num_nodes - 1)
                density = num_edges / possible_edges_no_self_loops if possible_edges_no_self_loops > 0 else 0
                print(f"      Density (E / N(N-1)): {density:.4f}")

            if num_nodes > 0 and num_edges > 0:
                print("      Skipping detailed NetworkX stats for large graphs to save time/memory.")
            print(f"    --- End of Graph Statistics for n={n} ---\n")

            del graph_object, idx_to_node

            gc.collect()

        print(f"<<< Phase 2 finished in {time.monotonic() - phase2_start_time:.2f}s.")

        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.monotonic()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        print(f"<<< Phase 3 finished in {time.monotonic() - phase3_start_time:.2f}s.")
        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.monotonic() - overall_start_time:.2f}s")
