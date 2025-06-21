# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 6.4 (Saves aggregated edges to file to break memory chain)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, Iterator

import dask.bag as db
import dask.dataframe as dd
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from config import Config
from src.utils.data_utils import DataUtils, DataLoader
from src.utils.graph_utils import DirectedNgramGraph


# ... (Helper functions _preprocess_sequence_tuple_for_bag, etc. remain the same) ...
def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
    pid, seq_text = seq_tuple
    modified_seq_text = str(seq_text)
    if add_initial_space:
        modified_seq_text = " " + modified_seq_text
    modified_seq_text = modified_seq_text + " "
    return pid, modified_seq_text


def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[str]:
    _, processed_seq_text = seq_tuple
    if len(processed_seq_text) >= n_val:
        for i in range(len(processed_seq_text) - n_val + 1):
            yield processed_seq_text[i:i + n_val]


def _extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int, ngram_to_id_map: Dict[str, int]) -> Iterator[str]:
    _, processed_seq_text = seq_tuple
    if len(processed_seq_text) >= n_val + 1:
        for i in range(len(processed_seq_text) - n_val):
            source_ngram = processed_seq_text[i:i + n_val]
            target_ngram = processed_seq_text[i + 1:i + 1 + n_val]
            source_id = ngram_to_id_map.get(source_ngram)
            target_id = ngram_to_id_map.get(target_ngram)
            if source_id is not None and target_id is not None:
                yield f"{source_id} {target_id}\n"


class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_file = str(config.GCN_INPUT_FASTA_PATH)
        self.output_dir = str(config.GRAPH_OBJECTS_DIR)
        self.n_max = config.GCN_NGRAM_MAX_N
        self.num_workers_config = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

        print(f"GraphBuilder initialized: n_max={self.n_max}, configured_workers={self.num_workers_config}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"GraphBuilder Initialized (Output: {self.output_dir})")

    def run(self):
        # ... (Phase 1 logic remains the same as v6.3) ...
        overall_start_time = time.monotonic()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs")

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
            print(f"GraphBuilder is configured for parallel processing (GRAPH_BUILDER_WORKERS={self.num_workers_config}).")
            print(f"Attempting to use Dask with a THREADED scheduler ({effective_dask_workers} threads) for most ops.")
            print("This aims to improve speed while potentially avoiding multiprocessing-related memory issues.")
            print("=" * 80 + "\n")
        else:
            print("\nGraphBuilder is configured for synchronous (single-threaded) execution.\n")

        def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]:
            first_sequence = True
            for seq_tuple in DataLoader.parse_sequences(self.protein_sequence_file):
                yield seq_tuple, first_sequence
                if first_sequence:
                    first_sequence = False

        try:
            sequence_stream_list = list(get_preprocessed_sequence_stream())
            if not sequence_stream_list:
                print("ERROR: No sequences found in the FASTA file. Cannot proceed.")
                return
            print(f"  Loaded {len(sequence_stream_list)} sequences from FASTA.")
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.protein_sequence_file}")
            return
        except MemoryError as e_mem_seq_list:
            print(f"ERROR: Memory error creating sequence list from FASTA: {e_mem_seq_list}.")
            return

        num_partitions_for_bag = effective_dask_workers if effective_dask_workers > 1 else 1
        raw_sequence_bag_with_flag = db.from_sequence(sequence_stream_list, npartitions=num_partitions_for_bag)
        preprocessed_sequence_bag_unpersisted = raw_sequence_bag_with_flag.starmap(_preprocess_sequence_tuple_for_bag)

        final_preprocessed_input_bag = preprocessed_sequence_bag_unpersisted
        if dask_scheduler_general != 'sync':
            print("  Persisting preprocessed_sequence_bag in Dask memory...")
            final_preprocessed_input_bag = preprocessed_sequence_bag_unpersisted.persist(
                scheduler=dask_scheduler_general, num_workers=effective_dask_workers
            )
            try:
                persisted_bag_count = final_preprocessed_input_bag.count().compute(scheduler=dask_scheduler_general, num_workers=effective_dask_workers)
                print(f"  DEBUG: Count of items in persisted preprocessed_sequence_bag: {persisted_bag_count}")
            except Exception as e_count:
                print(f"  DEBUG: Error counting persisted bag items: {e_count}")
            print(f"  Preprocessed bag persisted. Type: {type(final_preprocessed_input_bag)}")
        else:
            print("  Using unpersisted preprocessed_sequence_bag for synchronous execution.")

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if dask_scheduler_general != 'sync':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations.")

        for n_val_loop in n_values:
            DataUtils.print_header(f"Processing N-gram Level n = {n_val_loop} (Dask scheduler general: {dask_scheduler_general})")
            phase1_level_start_time = time.monotonic()

            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            temp_edge_output_dir = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}_parts')

            print(f"  [n={n_val_loop}] Generating n-grams using Dask Bag (source: final_preprocessed_input_bag)...")
            sys.stdout.flush()

            extract_ngrams_partial = partial(_extract_ngrams_from_sequence_tuple, n_val=n_val_loop)
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
                print(f"  [n={n_val_loop}] MEMORY ERROR during Dask compute for distinct n-grams: {e_mem_distinct_ngrams}")
                continue
            except Exception as e_compute_distinct_ngrams:
                print(f"  [n={n_val_loop}] ERROR during Dask compute for distinct n-grams: {e_compute_distinct_ngrams}")
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

            extract_edges_partial = partial(_extract_edges_from_sequence_tuple,
                                            n_val=n_val_loop,
                                            ngram_to_id_map=ngram_to_id_map)
            all_edges_str_bag_flattened = final_preprocessed_input_bag.map(extract_edges_partial).flatten()

            if os.path.exists(temp_edge_output_dir):
                shutil.rmtree(temp_edge_output_dir)

            scheduler_for_to_textfiles = 'sync'
            print(f"    Writing edge strings to directory: {temp_edge_output_dir} using Dask scheduler: '{scheduler_for_to_textfiles}'...")

            try:
                all_edges_str_bag_flattened.to_textfiles(
                    os.path.join(temp_edge_output_dir, 'part-*.txt'),
                    compute=True,
                    scheduler=scheduler_for_to_textfiles,
                    num_workers=None
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
                edge_parts_glob = os.path.join(edge_parts_dir, 'part-*.txt')
                if os.path.exists(edge_parts_dir) and any(Path(edge_parts_dir).glob('part-*.txt')):
                    ddf = dd.read_csv(edge_parts_glob, sep=' ', header=None, names=['source', 'target'], dtype=int,
                                      on_bad_lines='skip', blocksize='128MB')
                    print(f"    Dask DataFrame created for n={n} from part-files with {ddf.npartitions} partitions.")
                    weighted_ddf_series = ddf.groupby(['source', 'target']).size()
                    weighted_ddf = weighted_ddf_series.to_frame(name='weight').reset_index()
                    print(f"    Computing aggregated weighted edges for n={n}...")
                    weighted_edge_df_computed = weighted_ddf.compute(scheduler=dask_scheduler_general, num_workers=effective_dask_workers)
                    print(f"    Finished computing aggregated weighted edges for n={n}.")
                else:
                    print(f"  ℹ️ Info: Edge parts directory for n={n} is empty or not found.")
            except Exception as e_ddf:
                print(f"  ❌ Error: Dask DataFrame processing error for edges n={n}: {e_ddf}. Assuming no edges.")

            # --- MODIFICATION: Save to file to break memory chain ---
            temp_edge_file_path = os.path.join(self.temp_dir, f'aggregated_edges_n{n}.parquet')
            if not weighted_edge_df_computed.empty:
                print(f"  Aggregated raw transitions into {len(weighted_edge_df_computed)} unique weighted edges for n={n}.")
                print(f"  Saving aggregated edges to temporary file: {os.path.basename(temp_edge_file_path)}")
                try:
                    weighted_edge_df_computed.to_parquet(temp_edge_file_path, index=False)
                except Exception as e:
                    print(f"  ❌ Error saving temporary aggregated edge file: {e}. Skipping graph creation for n={n}.")
                    continue
            else:
                if os.path.exists(temp_edge_file_path):
                    os.remove(temp_edge_file_path)
                print(f"  ℹ️ Info: No unique edges found for n={n}.")

            del weighted_edge_df_computed
            gc.collect()

            print(f"  Instantiating DirectedNgramGraph object for n={n} from file...")
            graph_object = DirectedNgramGraph(
                nodes=idx_to_node,
                edge_file_path=temp_edge_file_path,
                epsilon_propagation=self.gcn_propagation_epsilon,
                n_value=n
            )

            if os.path.exists(temp_edge_file_path):
                os.remove(temp_edge_file_path)
            # --- END MODIFICATION ---

            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"  Graph for n={n} saved to {output_path}")

            # ... (Statistics generation logic remains the same) ...
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
