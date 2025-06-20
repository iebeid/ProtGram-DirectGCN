# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 6.0 (Added diagnostic for part-file line counts before concatenation)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import shutil
import sys
import time  # Ensure time is imported for time.monotonic()
from functools import partial
from typing import Tuple, Dict, Iterator

import community as community_louvain  # For Louvain community detection
import dask.bag as db
import dask.dataframe as dd  # Import Dask DataFrame
import networkx as nx
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm.auto import tqdm  # Import tqdm for progress bar during concatenation

from config import Config
from src.utils.data_utils import DataUtils, DataLoader
from src.utils.graph_utils import DirectedNgramGraph

# TensorFlow import guard
_tf = None


def get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# --- Helper functions for Dask Bag processing ---
def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
    """Adds space tokens to a single sequence tuple."""
    pid, seq_text = seq_tuple
    modified_seq_text = str(seq_text)
    if add_initial_space:
        modified_seq_text = " " + modified_seq_text
    modified_seq_text = modified_seq_text + " "
    return pid, modified_seq_text


def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[str]:
    """Extracts n-grams from a single preprocessed sequence tuple."""
    _, processed_seq_text = seq_tuple
    if len(processed_seq_text) >= n_val:
        for i in range(len(processed_seq_text) - n_val + 1):
            yield processed_seq_text[i:i + n_val]


def _extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int, ngram_to_id_map: Dict[str, int]) -> Iterator[str]:
    """
    Extracts edges (as "src_id tgt_id\n" strings) from a single
    preprocessed sequence tuple using the provided ngram_to_id_map.
    """
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
        # General dask_scheduler for most operations
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
            print(f"ERROR: Memory error creating sequence list from FASTA: {e_mem_seq_list}. The FASTA file might be too large to load sequence tuples into memory.")
            print("Consider processing the FASTA in chunks or using Dask's read_text directly if possible.")
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
                if persisted_bag_count != len(sequence_stream_list):
                    print(f"  CRITICAL WARNING: Mismatch! Initial sequence list had {len(sequence_stream_list)} items, but persisted bag has {persisted_bag_count}.")
            except Exception as e_count:
                print(f"  DEBUG: Error counting persisted bag items: {e_count}")
            print(f"  Preprocessed bag persisted. Type: {type(final_preprocessed_input_bag)}")
        else:
            print("  Using unpersisted preprocessed_sequence_bag for synchronous execution.")

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if dask_scheduler_general != 'sync':  # Only hide if Dask might use other contexts
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations.")

        for n_val_loop in n_values:
            DataUtils.print_header(f"Processing N-gram Level n = {n_val_loop} (Dask scheduler general: {dask_scheduler_general})")
            phase1_level_start_time = time.monotonic()

            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            output_edge_file = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}.txt')

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
                    print(f"  [n={n_val_loop}] No n-grams generated. Creating empty map file.")
                    unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
                else:
                    unique_ngrams_df = pd.DataFrame(sorted(unique_ngrams_list), columns=['ngram'])
            except MemoryError as e_mem_distinct_ngrams:
                print(f"  [n={n_val_loop}] MEMORY ERROR during Dask compute for distinct n-grams: {e_mem_distinct_ngrams}")
                print("  The number of unique n-grams might be too large to fit in memory even after using distinct().")
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue
            except Exception as e_compute_distinct_ngrams:
                print(f"  [n={n_val_loop}] ERROR during Dask compute for distinct n-grams: {e_compute_distinct_ngrams}")
                import traceback
                traceback.print_exc(file=sys.stderr)
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
                open(output_edge_file, 'w').close()
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

            if n_val_loop == 1 and self.config.DEBUG_VERBOSE:  # Keep this for n=1
                try:
                    print(f"    DEBUG: Attempting to count items in all_edges_str_bag_flattened for n={n_val_loop} before to_textfiles...")
                    intermediate_edge_count = all_edges_str_bag_flattened.count().compute(scheduler=dask_scheduler_general, num_workers=effective_dask_workers)
                    print(f"    DEBUG: Count from all_edges_str_bag_flattened for n={n_val_loop}: {intermediate_edge_count}")
                except Exception as e_dbg_count:
                    print(f"    DEBUG: Error counting intermediate edge bag for n={n_val_loop}: {e_dbg_count}")

            temp_edge_output_dir = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}_parts')
            if os.path.exists(temp_edge_output_dir):
                shutil.rmtree(temp_edge_output_dir)

            scheduler_for_to_textfiles = 'sync'  # Keep 'sync' for to_textfiles for diagnostic stability
            print(f"    Writing edge strings to directory: {temp_edge_output_dir} using Dask scheduler: '{scheduler_for_to_textfiles}' (DIAGNOSTIC)...")

            edge_count_from_concatenation = 0  # Renamed to avoid conflict with any prior edge_count
            try:
                all_edges_str_bag_flattened.to_textfiles(
                    os.path.join(temp_edge_output_dir, 'part-*.txt'),
                    compute=True,
                    scheduler=scheduler_for_to_textfiles,
                    num_workers=effective_dask_workers if scheduler_for_to_textfiles != 'sync' else None
                )
                print(f"    Edge parts saved to directory {temp_edge_output_dir}.")

                # *** ADDED DIAGNOSTIC FOR PART FILES (APPLIES TO ALL N) ***
                if self.config.DEBUG_VERBOSE and os.path.exists(temp_edge_output_dir):
                    part_files_check = sorted([  # Sort for consistent order if it matters for debugging
                        os.path.join(temp_edge_output_dir, f)
                        for f in os.listdir(temp_edge_output_dir)
                        if f.startswith('part-') and f.endswith('.txt')
                    ])
                    print(f"    DEBUG (n={n_val_loop}): Found {len(part_files_check)} part-files in {temp_edge_output_dir}.")
                    total_lines_in_parts = 0
                    for i_part, pf_path in enumerate(part_files_check):
                        try:
                            with open(pf_path, 'r', encoding='utf-8') as temp_pf:  # Added encoding
                                lines_in_this_part = sum(1 for _ in temp_pf)
                                total_lines_in_parts += lines_in_this_part
                                if i_part < 3 or (len(part_files_check) > 5 and i_part > len(part_files_check) - 3):  # Print for first/last few parts
                                    print(f"      DEBUG (n={n_val_loop}): Part-file '{os.path.basename(pf_path)}' has {lines_in_this_part} lines.")
                        except Exception as e_pf_read:
                            print(f"      DEBUG (n={n_val_loop}): Error reading part-file {pf_path}: {e_pf_read}")
                    print(f"    DEBUG (n={n_val_loop}): Total lines counted across all part-files: {total_lines_in_parts}")
                # *** END ADDED DIAGNOSTIC ***

                print(f"    Concatenating edge parts into {os.path.basename(output_edge_file)}...")
                edge_count_from_concatenation = 0  # Reset before this specific loop
                if not os.path.exists(temp_edge_output_dir):
                    print(f"    Warning: Dask did not create the edge parts directory: {temp_edge_output_dir}. Assuming no edges were generated.")
                    open(output_edge_file, 'w').close()
                else:
                    # Use the already sorted list if available from diagnostic, else re-list and sort
                    part_files_for_concat = part_files_check if 'part_files_check' in locals() and self.config.DEBUG_VERBOSE else sorted([
                        os.path.join(temp_edge_output_dir, f)
                        for f in os.listdir(temp_edge_output_dir)
                        if f.startswith('part-') and f.endswith('.txt')
                    ])

                    if not part_files_for_concat:
                        print(f"    Warning: No edge part-files found in {temp_edge_output_dir} for concatenation. Output edge file will be empty.")

                    with open(output_edge_file, 'w', encoding='utf-8') as outfile:  # Added encoding
                        for part_file_name in tqdm(part_files_for_concat, desc=f"    Concatenating parts (n={n_val_loop})", leave=False, disable=not self.config.DEBUG_VERBOSE):
                            with open(part_file_name, 'r', encoding='utf-8') as infile:  # Added encoding
                                for line in infile:
                                    outfile.write(line)
                                    edge_count_from_concatenation += 1
                print(f"    Successfully wrote {edge_count_from_concatenation} raw edge transitions to {os.path.basename(output_edge_file)}.")
                if os.path.exists(temp_edge_output_dir):
                    shutil.rmtree(temp_edge_output_dir)

            except MemoryError as e_mem:
                print(f"  [n={n_val_loop}] MEMORY ERROR during edge string computation or writing (to_textfiles): {e_mem}")
                print("  This can happen if even a single partition's result is too large for a worker, or during Dask's internal operations for to_textfiles.")
                import traceback
                traceback.print_exc(file=sys.stderr)
                open(output_edge_file, 'w').close()
                print(f"  Created empty edge file for n={n_val_loop} due to memory error.")
                if "Killed" in str(e_mem) or "Cannot allocate memory" in str(e_mem):
                    print(f"  OOM likely killed a Dask worker or the process at n={n_val_loop} during edge processing. Further processing for this n-level may be unreliable.")
            except Exception as e_write_edges:
                print(f"  [n={n_val_loop}] ERROR writing edge list file or during edge computation: {e_write_edges}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                open(output_edge_file, 'w').close()
                print(f"  Created empty edge file for n={n_val_loop} due to error.")

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
            edge_file = os.path.join(self.temp_dir, f'edge_list_n{n}.txt')

            if not os.path.exists(ngram_map_file) or not os.path.exists(edge_file):
                print(f"  Warning: Intermediate files for n={n} not found (likely due to error in Phase 1). Skipping graph generation for this level.")
                continue
            try:
                table_read = pq.read_table(ngram_map_file, columns=['id', 'ngram'])
                nodes_df = table_read.to_pandas()
                del table_read
            except pyarrow.lib.ArrowInvalid as e:
                print(f"  ❌ Error: Could not read Parquet file for n={n} at: {ngram_map_file}. Details: {e}. Skipping.")
                continue
            except Exception as e_parquet:
                print(f"  ❌ Error: General error reading Parquet file for n={n} at: {ngram_map_file}. Details: {e_parquet}. Skipping.")
                continue

            if nodes_df.empty:
                print(f"  ℹ️ Info: The n-gram map for n={n} is empty. No graph will be generated. Skipping.")
                continue
            print(f"  Loaded {len(nodes_df)} n-grams for n={n} from map file.")
            idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()
            del nodes_df

            print(f"  Loading and aggregating raw edges for n={n} using Dask DataFrame...")
            weighted_edge_df_computed = pd.DataFrame(columns=['source', 'target', 'weight'])
            raw_edge_file_was_empty = False
            try:
                if not os.path.exists(edge_file) or os.path.getsize(edge_file) == 0:
                    print(f"  ℹ️ Info: Edge file for n={n} is empty or not found. Creating graph with no edges.")
                    raw_edge_file_was_empty = True
                else:
                    ddf = dd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int,
                                      on_bad_lines='skip', blocksize='128MB')  # Added on_bad_lines
                    print(f"    Dask DataFrame created for n={n} with {ddf.npartitions} partitions.")

                    weighted_ddf_series = ddf.groupby(['source', 'target']).size()
                    weighted_ddf = weighted_ddf_series.to_frame(name='weight').reset_index()

                    print(f"    Computing aggregated weighted edges for n={n}...")
                    weighted_edge_df_computed = weighted_ddf.compute(scheduler=dask_scheduler_general, num_workers=effective_dask_workers)
                    print(f"    Finished computing aggregated weighted edges for n={n}.")

                if weighted_edge_df_computed.empty and not raw_edge_file_was_empty:
                    print(f"  Warning: Dask DataFrame aggregation resulted in empty weighted edges for n={n}, though raw edge file was not empty.")

            except FileNotFoundError:
                print(f"  ❌ Error: Edge file not found for n={n} at: {edge_file} (during Dask DF load). Assuming no edges.")
                raw_edge_file_was_empty = True
            except Exception as e_ddf:
                print(f"  ❌ Error: Dask DataFrame processing error for edges n={n}: {e_ddf}. Assuming no edges.")
                import traceback
                traceback.print_exc(file=sys.stderr)
                raw_edge_file_was_empty = True

            weighted_edge_list_tuples = []
            if weighted_edge_df_computed.empty:
                if not raw_edge_file_was_empty:  # Only print if raw file wasn't empty but aggregation is
                    print(f"  ℹ️ Info: No unique edges found after aggregation for n={n}.")
            else:
                if self.config.DEBUG_VERBOSE:
                    print(f"  [DEBUG] Weighted edge_df (from Dask DF) for n={n}:")
                    print(weighted_edge_df_computed.head())
                # The Dask DataFrame aggregation now gives the count of unique weighted edges.
                # The number of raw transitions processed by Dask DF would be weighted_edge_df_computed['weight'].sum()
                # if we wanted to report that, but the log from Phase 1 (edge_count_from_concatenation)
                # already gives the number of lines in the concatenated file.
                print(f"  Aggregated raw transitions into {len(weighted_edge_df_computed)} unique weighted edges for n={n}.")
                weighted_edge_list_tuples = [tuple(x) for x in weighted_edge_df_computed[['source', 'target', 'weight']].to_numpy()]

            del weighted_edge_df_computed
            gc.collect()

            if self.config.DEBUG_VERBOSE:
                print(f"  [DEBUG] weighted_edge_list_tuples (sample): {weighted_edge_list_tuples[:5] if weighted_edge_list_tuples else 'Empty'}")

            print(f"  Instantiating DirectedNgramGraph object for n={n}...")
            graph_object = DirectedNgramGraph(nodes=idx_to_node, edges=weighted_edge_list_tuples, epsilon_propagation=self.gcn_propagation_epsilon)
            graph_object.n_value = n
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
            else:
                print("      Density: N/A (graph has <= 1 node)")
            is_complete = False
            if num_nodes > 1:
                if num_edges == num_nodes * (num_nodes - 1): is_complete = True
            elif num_nodes == 1 and num_edges == 0:
                is_complete = True
            print(f"      Is Complete (all N*(N-1) directed edges present): {is_complete}")
            if num_nodes > 0:
                G_nx = nx.DiGraph()
                G_nx.add_nodes_from(range(num_nodes))
                edges_for_nx = [(u, v, {'weight': w}) for u, v, w in graph_object.edges]
                G_nx.add_edges_from(edges_for_nx)
                num_wcc = nx.number_weakly_connected_components(G_nx)
                print(f"      Weakly Connected Components: {num_wcc}")
                if num_edges > 0:
                    num_scc = nx.number_strongly_connected_components(G_nx)
                    print(f"      Strongly Connected Components: {num_scc}")
                    G_undirected_nx = G_nx.to_undirected()
                    if G_undirected_nx.number_of_edges() > 0:
                        partition = community_louvain.best_partition(G_undirected_nx, random_state=self.config.RANDOM_STATE)
                        num_communities = len(set(partition.values()))
                        print(f"      Louvain Communities (on undirected graph): {num_communities}")
                    else:
                        print("      Louvain Communities: N/A (no edges in undirected version)")
                else:
                    print("      Strongly Connected Components: N/A (no edges)")
                    print("      Louvain Communities: N/A (no edges)")
                avg_in_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
                avg_out_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
                print(f"      Average In-Degree: {avg_in_degree:.2f}")
                print(f"      Average Out-Degree: {avg_out_degree:.2f}")
                if num_nodes > 1:
                    in_degree_centrality = nx.in_degree_centrality(G_nx)
                    avg_in_degree_centrality = sum(in_degree_centrality.values()) / num_nodes
                    print(f"      Avg. In-Degree Centrality (normalized): {avg_in_degree_centrality:.4f}")
                    out_degree_centrality = nx.out_degree_centrality(G_nx)
                    avg_out_degree_centrality = sum(out_degree_centrality.values()) / num_nodes
                    print(f"      Avg. Out-Degree Centrality (normalized): {avg_out_degree_centrality:.4f}")
                else:
                    print("      Avg. Degree Centrality: N/A (graph has <= 1 node)")
            print(f"    --- End of Graph Statistics for n={n} ---\n")
            del graph_object, idx_to_node, weighted_edge_list_tuples, G_nx
            gc.collect()

        print(f"<<< Phase 2 finished in {time.monotonic() - phase2_start_time:.2f}s.")

        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.monotonic()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        else:
            print("  Temporary directory not found, no cleanup needed.")
        print(f"<<< Phase 3 finished in {time.monotonic() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.monotonic() - overall_start_time:.2f}s")
