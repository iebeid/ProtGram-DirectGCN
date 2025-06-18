# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 5.2 (Fixed UnboundLocalError for weighted_edge_list_tuples)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
import sys
import time
from functools import partial
from typing import List, Tuple, Dict, Iterator

import community as community_louvain  # For Louvain community detection
import dask.bag as db
import networkx as nx
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

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


# --- Helper functions for Dask Bag processing (used by synchronous path) ---
def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
    """Adds space tokens to a single sequence tuple."""
    pid, seq_text = seq_tuple
    modified_seq_text = str(seq_text)
    if add_initial_space:
        modified_seq_text = " " + modified_seq_text
    modified_seq_text = modified_seq_text + " "
    return pid, modified_seq_text


def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> List[str]:
    """Extracts n-grams from a single preprocessed sequence tuple."""
    _, processed_seq_text = seq_tuple
    ngrams = []
    if len(processed_seq_text) >= n_val:
        for i in range(len(processed_seq_text) - n_val + 1):
            ngrams.append(processed_seq_text[i:i + n_val])
    return ngrams


def _extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int, ngram_to_id_map: Dict[str, int]) -> List[str]:
    """
    Extracts edges (as "src_id tgt_id\n" strings) from a single
    preprocessed sequence tuple using the provided ngram_to_id_map.
    """
    _, processed_seq_text = seq_tuple
    edge_strs = []
    if len(processed_seq_text) >= n_val + 1:
        for i in range(len(processed_seq_text) - n_val):
            source_ngram = processed_seq_text[i:i + n_val]
            target_ngram = processed_seq_text[i + 1:i + 1 + n_val]
            source_id = ngram_to_id_map.get(source_ngram)
            target_id = ngram_to_id_map.get(target_ngram)
            if source_id is not None and target_id is not None:
                edge_strs.append(f"{source_id} {target_id}\n")
    return edge_strs


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
        overall_start_time = time.time()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs")

        if os.path.exists(self.temp_dir):
            print(f"Cleaning up existing temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Temporary files will be stored in: {self.temp_dir}")
        print(f"Final graph objects will be saved to: {self.output_dir}")

        n_values = range(1, self.n_max + 1)

        if self.num_workers_config > 1:
            print("\n" + "=" * 80)
            print("WARNING: GraphBuilder is configured for parallel processing "
                  f"(GRAPH_BUILDER_WORKERS={self.num_workers_config}),")
            print("         but due to persistent multiprocessing instability (e.g., 'double free' errors),")
            print("         this stage will be FORCED TO RUN SYNCHRONOUSLY (num_workers=1).")
            print("         This will be slower but aims for stability.")
            print("         To re-attempt parallel graph building, you'll need to resolve the underlying")
            print("         environment or library conflicts related to multiprocessing.")
            print("=" * 80 + "\n")

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
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.protein_sequence_file}")
            return

        num_partitions_for_synchronous_bag = 1
        raw_sequence_bag_with_flag = db.from_sequence(sequence_stream_list, npartitions=num_partitions_for_synchronous_bag)
        preprocessed_sequence_bag = raw_sequence_bag_with_flag.starmap(_preprocess_sequence_tuple_for_bag)

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations (even if synchronous).")

        for n_val_loop in n_values:
            DataUtils.print_header(f"Processing N-gram Level n = {n_val_loop} (Synchronously)")
            phase1_level_start_time = time.time()

            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            output_edge_file = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}.txt')

            print(f"  [n={n_val_loop}] Generating all n-grams (including duplicates) using Dask Bag (single-threaded)...")
            sys.stdout.flush()

            extract_ngrams_partial = partial(_extract_ngrams_from_sequence_tuple, n_val=n_val_loop)
            all_ngrams_bag_flattened = preprocessed_sequence_bag.map(extract_ngrams_partial).flatten()

            all_ngrams_list_with_duplicates = []
            try:
                print("    Computing all n-grams (with duplicates) synchronously...")
                all_ngrams_list_with_duplicates = all_ngrams_bag_flattened.compute(scheduler='single-threaded')
                print(f"    Computed {len(all_ngrams_list_with_duplicates)} n-grams (including duplicates).")
            except Exception as e_compute_ngrams:
                print(f"  [n={n_val_loop}] ERROR during Dask compute for all n-grams: {e_compute_ngrams}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue

            print(f"    Finding unique n-grams and saving map sequentially...")
            if not all_ngrams_list_with_duplicates:
                print(f"  [n={n_val_loop}] No n-grams generated. Creating empty map file.")
                unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
            else:
                unique_ngrams_set = set(all_ngrams_list_with_duplicates)
                print(f"    Found {len(unique_ngrams_set)} unique {n_val_loop}-grams.")
                unique_ngrams_df = pd.DataFrame(sorted(list(unique_ngrams_set)), columns=['ngram'])

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
                print(f"  Level n={n_val_loop} (Phase 1) finished in {time.time() - phase1_level_start_time:.2f}s.")
                continue

            try:
                map_table_read = pq.read_table(output_ngram_map_file, columns=['ngram', 'id'])
                map_df_read = map_table_read.to_pandas()
                ngram_to_id_map = map_df_read.set_index('ngram')['id'].to_dict()
                del map_table_read, map_df_read
            except Exception as e_read_map:
                print(f"  [n={n_val_loop}] ERROR reading back ngram map: {e_read_map}. Skipping edge generation.")
                continue

            print(f"  [n={n_val_loop}] Generating all edge strings using Dask Bag (single-threaded)...")
            sys.stdout.flush()

            extract_edges_partial = partial(_extract_edges_from_sequence_tuple, n_val=n_val_loop, ngram_to_id_map=ngram_to_id_map)
            all_edges_str_bag_flattened = preprocessed_sequence_bag.map(extract_edges_partial).flatten()

            all_edges_str_list = []
            try:
                print("    Computing all edge strings synchronously...")
                all_edges_str_list = all_edges_str_bag_flattened.compute(scheduler='single-threaded')
                print(f"    Computed {len(all_edges_str_list)} edge strings.")
            except Exception as e_compute_edges:
                print(f"  [n={n_val_loop}] ERROR during Dask compute for edge strings: {e_compute_edges}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue

            print(f"    Saving edge list sequentially...")
            try:
                with open(output_edge_file, 'w') as outfile:
                    outfile.writelines(all_edges_str_list)
                print(f"  [n={n_val_loop}] Edge list saved: {os.path.basename(output_edge_file)}")
            except Exception as e_write_edges:
                print(f"  [n={n_val_loop}] ERROR writing edge list file: {e_write_edges}")
                continue

            print(f"  Level n={n_val_loop} (Phase 1) finished in {time.time() - phase1_level_start_time:.2f}s.")

        if original_cuda_visible_devices is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")

        DataUtils.print_header("Phase 2: Building and saving final graph objects")
        phase2_start_time = time.time()
        for n in n_values:
            print(f"\n--- Processing n = {n} for final graph object ---")
            ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
            edge_file = os.path.join(self.temp_dir, f'edge_list_n{n}.txt')

            if not os.path.exists(ngram_map_file) or not os.path.exists(edge_file):
                print(f"  Warning: Intermediate files for n={n} not found. Skipping graph generation.")
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

            try:
                edge_df = pd.read_csv(edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int)
                print(f"  Loaded {len(edge_df)} raw edges for n={n} from edge list file.")
            except pd.errors.EmptyDataError:
                print(f"  ℹ️ Info: Edge file for n={n} is empty. Creating graph with no edges.")
                edge_df = pd.DataFrame(columns=['source', 'target'])
            except FileNotFoundError:
                print(f"  ❌ Error: Edge file not found for n={n} at: {edge_file}. Assuming no edges.")
                edge_df = pd.DataFrame(columns=['source', 'target'])
            except Exception as e_csv:
                print(f"  ❌ Error: General error reading edge CSV file for n={n} at: {edge_file}. Details: {e_csv}. Assuming no edges.")
                edge_df = pd.DataFrame(columns=['source', 'target'])

            # Initialize weighted_edge_list_tuples *before* the if/else block
            weighted_edge_list_tuples = []
            if edge_df.empty:
                print(f"  ℹ️ Info: No edges found for n={n}. Creating a graph with nodes but no edges.")
                # weighted_edge_list_tuples remains empty as initialized
            else:
                weighted_edge_df = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
                print(f"  [DEBUG] Weighted edge_df for n={n}:")
                print(weighted_edge_df.head())
                print(f"  [DEBUG] Number of unique edges in weighted_edge_df: {len(weighted_edge_df)}")
                weighted_edge_list_tuples = [tuple(x) for x in weighted_edge_df[['source', 'target', 'weight']].to_numpy()]
                print(f"  Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_df)} unique weighted edges for n={n}.")

            # This debug print can now safely access weighted_edge_list_tuples
            print(f"  [DEBUG] weighted_edge_list_tuples (sample): {weighted_edge_list_tuples[:5] if weighted_edge_list_tuples else 'Empty'}")

            print(f"  Instantiating DirectedNgramGraph object for n={n}...")
            graph_object = DirectedNgramGraph(nodes=idx_to_node, edges=weighted_edge_list_tuples, epsilon_propagation=self.gcn_propagation_epsilon)
            graph_object.n_value = n
            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"  Graph for n={n} saved to {output_path}")

            # --- Calculate and Print Graph Statistics ---
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
                if num_edges == num_nodes * (num_nodes - 1):
                    is_complete = True
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

        print(f"<<< Phase 2 finished in {time.time() - phase2_start_time:.2f}s.")

        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.time()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        else:
            print("  Temporary directory not found, no cleanup needed.")
        print(f"<<< Phase 3 finished in {time.time() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.time() - overall_start_time:.2f}s")
