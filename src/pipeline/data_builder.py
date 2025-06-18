# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 4.8 (Refactor to use Dask Bag more like dask_test.py for stability)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
import sys
import time
from typing import List, Tuple, Dict, Iterator  # Added Iterator
from functools import partial  # For passing args to map functions

import dask
import dask.bag as db
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm

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
# These functions are designed to be simple and stateless for Dask workers

def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
    """Adds space tokens to a single sequence tuple."""
    pid, seq_text = seq_tuple
    modified_seq_text = str(seq_text)
    if add_initial_space:  # This flag needs to be true only for the very first sequence in the entire dataset
        modified_seq_text = " " + modified_seq_text
    modified_seq_text = modified_seq_text + " "
    return pid, modified_seq_text


def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> List[str]:
    """Extracts n-grams from a single preprocessed sequence tuple."""
    _, processed_seq_text = seq_tuple  # ID is not used here, only sequence
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
        self.num_workers = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        # SEQUENCES_PER_DASK_CHUNK is not directly used in this Dask Bag approach,
        # as Dask Bag handles partitioning of the sequence stream.
        # However, repartitioning can be influenced by num_workers.

        print(f"GraphBuilder initialized: n_max={self.n_max}, num_workers={self.num_workers}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"GraphBuilder Initialized (Output: {self.output_dir})")

    def run(self):
        overall_start_time = time.time()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs (Dask Bag Approach)")

        if os.path.exists(self.temp_dir):
            print(f"Cleaning up existing temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Temporary files will be stored in: {self.temp_dir}")
        print(f"Final graph objects will be saved to: {self.output_dir}")

        n_values = range(1, self.n_max + 1)
        effective_num_workers = max(1, self.num_workers)

        # Create a Dask Bag from the FASTA sequence stream
        # DataLoader.parse_sequences yields (pid, seq_text)
        # We need to preprocess this stream (add space tokens)
        # The first sequence in the entire dataset needs special handling for the initial space.

        # Create a generator for sequence tuples with preprocessing information
        def get_preprocessed_sequence_stream() -> Iterator[Tuple[Tuple[str, str], bool]]:
            first_sequence = True
            for seq_tuple in DataLoader.parse_sequences(self.protein_sequence_file):
                yield seq_tuple, first_sequence
                if first_sequence:
                    first_sequence = False

        # Create the Dask Bag from this generator.
        # For process-based scheduler, Dask might try to pickle the generator or its items.
        # It's generally better if the items are simple and picklable.
        # `db.from_sequence` with `npartitions` helps Dask manage this.
        # The number of partitions should ideally be related to `effective_num_workers`.
        num_partitions_for_bag = effective_num_workers * 2  # Heuristic

        # This sequence_bag will contain tuples: ((pid, raw_seq_text), is_first_sequence_flag)
        raw_sequence_bag_with_flag = db.from_sequence(list(get_preprocessed_sequence_stream()),  # Materialize for from_sequence if iterator causes issues
            npartitions=num_partitions_for_bag)

        # Preprocess the sequences within the Dask Bag
        # `starmap` unpacks the inner tuple for `_preprocess_sequence_tuple_for_bag`
        preprocessed_sequence_bag = raw_sequence_bag_with_flag.starmap(_preprocess_sequence_tuple_for_bag)

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # For the main process and Dask workers
        print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations.")

        for n_val_loop in n_values:
            DataUtils.print_header(f"Processing N-gram Level n = {n_val_loop}")
            phase1_level_start_time = time.time()

            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            output_edge_file = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}.txt')

            # --- Step 1: Generate and Save Unique N-grams ---
            print(f"  [n={n_val_loop}] Generating unique n-grams using Dask Bag...")
            sys.stdout.flush()

            # Use partial to fix the n_val argument for the map function
            extract_ngrams_partial = partial(_extract_ngrams_from_sequence_tuple, n_val=n_val_loop)

            all_ngrams_bag = preprocessed_sequence_bag.map(extract_ngrams_partial).flatten()

            try:
                if effective_num_workers == 1:
                    print("    Computing unique n-grams synchronously...")
                    unique_ngrams_list = all_ngrams_bag.distinct().compute(scheduler='single-threaded')
                else:
                    print(f"    Computing unique n-grams with 'processes' scheduler ({effective_num_workers} workers)...")
                    unique_ngrams_list = all_ngrams_bag.distinct().compute(scheduler='processes', num_workers=effective_num_workers)
                print(f"    Found {len(unique_ngrams_list)} unique {n_val_loop}-grams.")
            except Exception as e_compute_ngrams:
                print(f"  [n={n_val_loop}] ERROR during Dask compute for unique n-grams: {e_compute_ngrams}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue  # Skip to next n-gram level

            if not unique_ngrams_list:
                print(f"  [n={n_val_loop}] No unique n-grams found. Creating empty map file.")
                unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
            else:
                unique_ngrams_df = pd.DataFrame(sorted(unique_ngrams_list), columns=['ngram'])

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
                open(output_edge_file, 'w').close()  # Create empty edge file
                print(f"  [n={n_val_loop}] No n-grams, so no edges will be generated.")
                print(f"  Level n={n_val_loop} (Phase 1) finished in {time.time() - phase1_level_start_time:.2f}s.")
                continue

            # Load the created map for edge generation
            try:
                map_table_read = pq.read_table(output_ngram_map_file, columns=['ngram', 'id'])
                map_df_read = map_table_read.to_pandas()
                ngram_to_id_map = map_df_read.set_index('ngram')['id'].to_dict()
                del map_table_read, map_df_read
            except Exception as e_read_map:
                print(f"  [n={n_val_loop}] ERROR reading back ngram map: {e_read_map}. Skipping edge generation.")
                continue

            # --- Step 2: Generate and Save Edge List ---
            print(f"  [n={n_val_loop}] Generating edge list using Dask Bag...")
            sys.stdout.flush()

            extract_edges_partial = partial(_extract_edges_from_sequence_tuple, n_val=n_val_loop, ngram_to_id_map=ngram_to_id_map)
            all_edges_str_bag = preprocessed_sequence_bag.map(extract_edges_partial).flatten()

            temp_edge_parts_dir_n = os.path.join(self.temp_dir, f"edge_parts_n{n_val_loop}")
            if os.path.exists(temp_edge_parts_dir_n): shutil.rmtree(temp_edge_parts_dir_n)
            # os.makedirs(temp_edge_parts_dir_n, exist_ok=True) # to_textfiles creates it

            try:
                if effective_num_workers == 1:
                    print("    Saving edge parts synchronously...")
                    all_edges_str_bag.to_textfiles(os.path.join(temp_edge_parts_dir_n, 'part-*.txt'), compute_kwargs={'scheduler': 'single-threaded'})
                else:
                    print(f"    Saving edge parts with 'processes' scheduler ({effective_num_workers} workers)...")
                    all_edges_str_bag.to_textfiles(os.path.join(temp_edge_parts_dir_n, 'part-*.txt'), compute_kwargs={'scheduler': 'processes', 'num_workers': effective_num_workers})
                print(f"    Edge parts saved to: {temp_edge_parts_dir_n}")
            except Exception as e_to_textfiles:
                print(f"  [n={n_val_loop}] ERROR during Dask to_textfiles for edges: {e_to_textfiles}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue

            # Concatenate edge part files (sequentially)
            print(f"    Concatenating edge parts for n={n_val_loop}...")
            try:
                with open(output_edge_file, 'w') as outfile:
                    if os.path.exists(temp_edge_parts_dir_n):  # Check if dir was created
                        part_files = sorted([os.path.join(temp_edge_parts_dir_n, f) for f in os.listdir(temp_edge_parts_dir_n) if f.startswith('part-') and f.endswith('.txt')])
                        for fname in part_files:
                            with open(fname, 'r') as infile:
                                shutil.copyfileobj(infile, outfile)
                if os.path.exists(temp_edge_parts_dir_n): shutil.rmtree(temp_edge_parts_dir_n)
                print(f"  [n={n_val_loop}] Edge list saved: {os.path.basename(output_edge_file)}")
            except Exception as e_concat:
                print(f"  [n={n_val_loop}] ERROR concatenating edge files: {e_concat}")
                continue

            print(f"  Level n={n_val_loop} (Phase 1) finished in {time.time() - phase1_level_start_time:.2f}s.")

        # Restore CUDA_VISIBLE_DEVICES
        if original_cuda_visible_devices is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")
        # No distinct Phase 1.5 anymore, aggregation is per-level

        # --- Phase 2: Building and saving final graph objects (remains mostly the same) ---
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

            if edge_df.empty:
                print(f"  ℹ️ Info: No edges found for n={n}. Creating a graph with nodes but no edges.")
                weighted_edge_list_tuples = []
            else:
                weighted_edge_df = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
                print(f"  Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_df)} unique weighted edges for n={n}.")
                weighted_edge_list_tuples = [tuple(x) for x in weighted_edge_df[['source', 'target', 'weight']].to_numpy()]

            print(f"  Instantiating DirectedNgramGraph object for n={n}...")
            graph_object = DirectedNgramGraph(nodes=idx_to_node, edges=weighted_edge_list_tuples, epsilon_propagation=self.gcn_propagation_epsilon)
            graph_object.n_value = n
            output_path = os.path.join(self.output_dir, f'ngram_graph_n{n}.pkl')
            DataUtils.save_object(graph_object, output_path)
            print(f"  Graph for n={n} saved to {output_path}")
        print(f"<<< Phase 2 finished in {time.time() - phase2_start_time:.2f}s.")

        # --- Phase 3: Cleaning up temporary files ---
        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.time()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        else:
            print("  Temporary directory not found, no cleanup needed.")
        print(f"<<< Phase 3 finished in {time.time() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.time() - overall_start_time:.2f}s")
