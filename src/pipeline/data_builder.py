# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 4.7 (Chunked FASTA processing with dask.delayed to reduce memory)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
import sys
import time
from typing import List, Tuple, Dict

import dask
# import dask.bag as db # Not used by _process_sequence_chunk
import pandas as pd
import pyarrow  # For type hinting and direct use
import pyarrow.parquet as pq  # For direct Parquet I/O
# from dask.distributed import Client, LocalCluster # Keep for sync path
from tqdm import tqdm

from config import Config
from src.utils.data_utils import DataUtils, DataLoader  # DataLoader.parse_sequences
from src.utils.graph_utils import DirectedNgramGraph

# TensorFlow import guard
_tf = None


def get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# Helper function to parse a specific chunk of sequences from a FASTA file
def _parse_fasta_chunk(fasta_filepath: str, chunk_id: int, sequences_per_chunk: int) -> List[Tuple[str, str]]:
    """
    Parses a specific chunk of sequences from a FASTA file.
    A simple way to define chunks is by sequence count.
    """
    sequences_in_chunk = []
    start_index = chunk_id * sequences_per_chunk
    end_index = start_index + sequences_per_chunk

    current_index = 0
    for seq_id, seq_text in DataLoader.parse_sequences(fasta_filepath):
        if current_index >= end_index:
            break
        if current_index >= start_index:
            sequences_in_chunk.append((seq_id, seq_text))
        current_index += 1
    return sequences_in_chunk


def _preprocess_sequences_in_chunk(sequences: List[Tuple[str, str]], is_first_chunk_overall: bool) -> List[Tuple[str, str]]:
    """Adds space tokens to sequences within a chunk."""
    processed_sequences = []
    for i, (pid, seq_text) in enumerate(sequences):
        modified_seq_text = str(seq_text)
        # The "first sequence in the file" space addition needs careful handling with chunking.
        # If chunk_id == 0 and i == 0, it's the very first sequence.
        if is_first_chunk_overall and i == 0:
            modified_seq_text = " " + modified_seq_text
        modified_seq_text = modified_seq_text + " "
        processed_sequences.append((pid, modified_seq_text))
    return processed_sequences


class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_file = str(config.GCN_INPUT_FASTA_PATH)
        self.output_dir = str(config.GRAPH_OBJECTS_DIR)
        self.n_max = config.GCN_NGRAM_MAX_N
        self.num_workers = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        # Define a reasonable number of sequences per chunk for Dask tasks
        self.sequences_per_dask_chunk = getattr(config, 'SEQUENCES_PER_DASK_CHUNK', 500)

        print(f"GraphBuilder initialized: n_max={self.n_max}, num_workers={self.num_workers}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"GraphBuilder Initialized (Output: {self.output_dir})")

    @staticmethod
    def _process_sequence_chunk_for_n_level(
            fasta_filepath: str,
            n_value: int,
            chunk_id: int,  # To identify the chunk
            sequences_per_chunk: int,  # How many sequences this chunk should process
            is_first_chunk_overall: bool  # True if this is chunk_id 0 of the entire dataset
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Processes a single chunk of sequences for a given n-gram level.
        Returns:
            - List of unique n-gram strings found in this chunk.
            - List of (source_ngram_str, target_ngram_str) edge tuples from this chunk.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # print(f"[ChunkTask n={n_value}, chunk={chunk_id}, PID={os.getpid()}]: Starting processing.", flush=True)

        # 1. Parse only the sequences for this chunk
        sequences_this_chunk_raw = _parse_fasta_chunk(fasta_filepath, chunk_id, sequences_per_chunk)
        if not sequences_this_chunk_raw:
            # print(f"[ChunkTask n={n_value}, chunk={chunk_id}, PID={os.getpid()}]: No sequences in this chunk.", flush=True)
            return [], []

        # 2. Preprocess (add space tokens)
        sequences_this_chunk_processed = _preprocess_sequences_in_chunk(sequences_this_chunk_raw, is_first_chunk_overall)

        # 3. Generate n-grams for this chunk
        chunk_ngrams: List[str] = []
        for _, seq_text in sequences_this_chunk_processed:
            if len(seq_text) >= n_value:
                for i in range(len(seq_text) - n_value + 1):
                    chunk_ngrams.append(seq_text[i:i + n_value])

        unique_chunk_ngrams = sorted(list(set(chunk_ngrams)))

        # 4. Generate edges (as n-gram strings) for this chunk
        chunk_edges_str_pairs: List[Tuple[str, str]] = []
        for _, seq_text in sequences_this_chunk_processed:
            if len(seq_text) >= n_value + 1:
                for i in range(len(seq_text) - n_value):
                    source_ngram = seq_text[i:i + n_value]
                    target_ngram = seq_text[i + 1:i + 1 + n_value]
                    chunk_edges_str_pairs.append((source_ngram, target_ngram))

        # print(f"[ChunkTask n={n_value}, chunk={chunk_id}, PID={os.getpid()}]: Found {len(unique_chunk_ngrams)} unique ngrams, {len(chunk_edges_str_pairs)} edges.", flush=True)
        return unique_chunk_ngrams, chunk_edges_str_pairs

    def run(self):
        overall_start_time = time.time()
        DataUtils.print_header("PIPELINE STEP 1: Building N-gram Graphs (Chunked Processing)")

        if os.path.exists(self.temp_dir):
            print(f"Cleaning up existing temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Temporary files will be stored in: {self.temp_dir}")
        print(f"Final graph objects will be saved to: {self.output_dir}")

        n_values = range(1, self.n_max + 1)
        effective_num_workers = max(1, self.num_workers)

        # Determine total number of sequences for chunking without loading all into memory
        total_sequences = 0
        try:
            for _ in DataLoader.parse_sequences(self.protein_sequence_file):
                total_sequences += 1
            if total_sequences == 0:
                print("ERROR: No sequences found in the FASTA file. Cannot proceed.")
                return
            print(f"  Total sequences in FASTA file: {total_sequences}")
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.protein_sequence_file}. Cannot proceed.")
            return

        num_chunks = (total_sequences + self.sequences_per_dask_chunk - 1) // self.sequences_per_dask_chunk
        print(f"  Dividing into {num_chunks} chunks of approx {self.sequences_per_dask_chunk} sequences each.")

        DataUtils.print_header(f"Phase 1: Processing Chunks and N-gram Levels via Dask.delayed")
        phase1_start_time = time.time()
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask operations in main process.")

        all_delayed_tasks = []
        # Keep track of (n_value, chunk_id) for each task to map results later
        task_metadata = []

        for n_val_loop in n_values:
            for chunk_idx in range(num_chunks):
                is_first_overall = (chunk_idx == 0)  # Only the very first chunk of the file gets the initial space
                task = dask.delayed(GraphBuilder._process_sequence_chunk_for_n_level)(
                    self.protein_sequence_file,
                    n_val_loop,
                    chunk_idx,
                    self.sequences_per_dask_chunk,
                    is_first_overall
                )
                all_delayed_tasks.append(task)
                task_metadata.append({'n_value': n_val_loop, 'chunk_id': chunk_idx})

        results_from_chunks_raw = []
        if all_delayed_tasks:
            print(f"  Submitting {len(all_delayed_tasks)} delayed chunk processing tasks...")
            sys.stdout.flush()
            try:
                if effective_num_workers == 1:
                    print("  Running chunk processing synchronously (1 worker).")
                    # Use a simple loop for synchronous execution to mimic dask.compute with single worker
                    # This helps in debugging the _process_sequence_chunk_for_n_level function itself
                    # results_from_chunks_raw = [task.compute(scheduler='single-threaded') for task in tqdm(all_delayed_tasks, desc="Processing Chunks (Sync)")]

                    # For true synchronous, call directly without dask.delayed wrapper
                    # This part is tricky if we want to use the same `all_delayed_tasks` list.
                    # Let's stick to dask.compute for consistency, even for 1 worker.
                    results_from_chunks_raw = dask.compute(*all_delayed_tasks, scheduler='single-threaded')

                else:
                    print(f"  Running chunk processing with 'processes' scheduler ({effective_num_workers} workers).")
                    results_from_chunks_raw = dask.compute(*all_delayed_tasks, scheduler='processes', num_workers=effective_num_workers)

                print(f"  All {len(results_from_chunks_raw)} chunk processing tasks completed.")
            except Exception as e_delayed_chunks:
                print(f"ERROR during dask.delayed chunk processing: {e_delayed_chunks}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                # Clean up and exit if chunk processing fails
                if original_cuda_visible_devices is None:
                    if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
                return
        else:
            print("  No chunk processing tasks to submit.")

        # Restore CUDA_VISIBLE_DEVICES
        if original_cuda_visible_devices is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")
        print(f"<<< Phase 1 (Chunk Processing) finished in {time.time() - phase1_start_time:.2f}s.")

        # --- Phase 1.5: Aggregate results from chunks and create final intermediate files ---
        DataUtils.print_header("Phase 1.5: Aggregating Chunk Results & Creating Intermediate Files")
        aggregation_start_time = time.time()

        # Reorganize results: results_by_n_level[n_value] = [(unique_ngrams_chunk0, edges_chunk0), ...]
        results_by_n_level: Dict[int, List[Tuple[List[str], List[Tuple[str, str]]]]] = {n: [] for n in n_values}
        for i, raw_res_tuple in enumerate(results_from_chunks_raw):
            meta = task_metadata[i]
            # Ensure raw_res_tuple is indeed a tuple of two lists
            if isinstance(raw_res_tuple, tuple) and len(raw_res_tuple) == 2 and \
                    isinstance(raw_res_tuple[0], list) and isinstance(raw_res_tuple[1], list):
                results_by_n_level[meta['n_value']].append(raw_res_tuple)
            else:
                print(f"Warning: Unexpected result format for n={meta['n_value']}, chunk={meta['chunk_id']}. Result: {type(raw_res_tuple)}. Skipping this chunk's result.")

        for n_val_loop in tqdm(n_values, desc="Aggregating and Saving by N-gram Level"):
            output_ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n_val_loop}.parquet')
            output_edge_file = os.path.join(self.temp_dir, f'edge_list_n{n_val_loop}.txt')

            if not results_by_n_level.get(n_val_loop):
                print(f"  No chunk results found for n={n_val_loop}. Creating empty intermediate files.")
                pd.DataFrame({'id': pd.Series(dtype='int'), 'ngram': pd.Series(dtype='str')}).to_parquet(output_ngram_map_file)
                open(output_edge_file, 'w').close()
                continue

            # 1. Aggregate unique n-grams
            all_unique_ngrams_for_n_level = set()
            for unique_ngrams_chunk, _ in results_by_n_level[n_val_loop]:
                all_unique_ngrams_for_n_level.update(unique_ngrams_chunk)

            if not all_unique_ngrams_for_n_level:
                print(f"  No unique n-grams found for n={n_val_loop} after aggregation.")
                unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
            else:
                unique_ngrams_df = pd.DataFrame(sorted(list(all_unique_ngrams_for_n_level)), columns=['ngram'])

            unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
            unique_ngrams_df['id'] = unique_ngrams_df.index

            try:
                table_to_write = pyarrow.Table.from_pandas(unique_ngrams_df[['id', 'ngram']], preserve_index=False)
                pq.write_table(table_to_write, output_ngram_map_file)
                del table_to_write
                print(f"  [n={n_val_loop}] Aggregated {len(unique_ngrams_df)} unique n-grams. Saved map: {os.path.basename(output_ngram_map_file)}")
            except Exception as e_parquet_agg_write:
                print(f"  [n={n_val_loop}] ERROR writing aggregated Parquet map: {e_parquet_agg_write}")
                continue  # Skip to next n-value if map saving fails

            if unique_ngrams_df.empty:
                open(output_edge_file, 'w').close()
                print(f"  [n={n_val_loop}] No n-grams, so no edges will be generated.")
                continue

            ngram_to_global_id_map = unique_ngrams_df.set_index('ngram')['id'].to_dict()

            # 2. Aggregate edges and convert to global IDs
            final_edge_list_str = []
            total_edges_from_chunks = 0
            for _, edges_str_pairs_chunk in results_by_n_level[n_val_loop]:
                total_edges_from_chunks += len(edges_str_pairs_chunk)
                for src_ngram_str, tgt_ngram_str in edges_str_pairs_chunk:
                    src_id = ngram_to_global_id_map.get(src_ngram_str)
                    tgt_id = ngram_to_global_id_map.get(tgt_ngram_str)
                    if src_id is not None and tgt_id is not None:
                        final_edge_list_str.append(f"{src_id} {tgt_id}\n")

            try:
                with open(output_edge_file, 'w') as f_edge:
                    f_edge.writelines(final_edge_list_str)
                print(f"  [n={n_val_loop}] Aggregated {len(final_edge_list_str)} edges (from {total_edges_from_chunks} chunk edges). Saved list: {os.path.basename(output_edge_file)}")
            except Exception as e_edge_agg_write:
                print(f"  [n={n_val_loop}] ERROR writing aggregated edge list: {e_edge_agg_write}")

        print(f"<<< Phase 1.5 (Aggregation) finished in {time.time() - aggregation_start_time:.2f}s.")

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
            # We are not creating chunk-specific files in temp_dir anymore with this design,
            # only the aggregated ngram_map_nX.parquet and edge_list_nX.txt.
            # So, cleaning self.temp_dir is correct.
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        else:
            print("  Temporary directory not found, no cleanup needed.")
        print(f"<<< Phase 3 finished in {time.time() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.time() - overall_start_time:.2f}s")
