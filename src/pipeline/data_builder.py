# ==============================================================================
# MODULE: pipeline/data_builder.py
# PURPOSE: Main class to orchestrate the graph building process.
# VERSION: 4.1 (Enhanced logging, Dask Bag for FASTA, CUDA_VISIBLE_DEVICES for workers)
# AUTHOR: Islam Ebeid
# ==============================================================================

import logging
import os
import shutil
import time

import dask.bag as db
import pandas as pd
import pyarrow
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from config import Config
from src.utils.data_utils import DataUtils, DataLoader
from src.utils.graph_utils import DirectedNgramGraph

# --- Add this import guard ---
# This helps delay TensorFlow import until it's actually needed,
# potentially avoiding initialization in Dask workers if they don't need it.
# However, if other libraries implicitly import TF, this might not be enough.
_tf = None
def get_tf():
    global _tf
    if _tf is None:
        # print("Importing TensorFlow inside worker task...") # Debugging print
        import tensorflow as tf
        _tf = tf
    return _tf
# --- End import guard ---


class GraphBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.protein_sequence_file = str(config.GCN_INPUT_FASTA_PATH)
        self.output_dir = str(config.GRAPH_OBJECTS_DIR)
        self.n_max = config.GCN_NGRAM_MAX_N
        self.num_workers = config.GRAPH_BUILDER_WORKERS if config.GRAPH_BUILDER_WORKERS is not None else 1
        self.temp_dir = os.path.join(str(config.BASE_OUTPUT_DIR), "temp_graph_builder")
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        print(f"GraphBuilder initialized: n_max={self.n_max}, num_workers={self.num_workers}, output_dir='{self.output_dir}'")
        DataUtils.print_header(f"GraphBuilder Initialized (Output: {self.output_dir})")


    @staticmethod
    def _create_intermediate_files(n_value, temp_dir, protein_sequence_file_path, num_dask_partitions):
        # --- Add environment variable setting at the start of the worker task ---
        # This ensures the worker process itself has this setting applied,
        # regardless of what the main process did or how it was inherited.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # print(f"[Worker n={n_value}, PID={os.getpid()}]: CUDA_VISIBLE_DEVICES set to -1.") # Debugging print
        # --- End environment variable setting ---

        process_start_time = time.time()
        print(f"[Worker n={n_value}, PID={os.getpid()}]: Starting intermediate file construction for n-gram size {n_value}.")
        output_ngram_map_file = os.path.join(temp_dir, f'ngram_map_n{n_value}.parquet')
        output_edge_file = os.path.join(temp_dir, f'edge_list_n{n_value}.txt')

        print(f"  [Worker n={n_value}] Pass 1: Discovering unique n-grams from '{os.path.basename(protein_sequence_file_path)}'...")

        # DataLoader.parse_sequences is a static method, safe to call.
        # It doesn't seem to import TensorFlow.
        sequences = list(DataLoader.parse_sequences(str(protein_sequence_file_path)))
        if not sequences:
            print(f"  [Worker n={n_value}] ⚠️ Warning: No sequences found in {os.path.basename(protein_sequence_file_path)} for n={n_value}.")
            pd.DataFrame({'id': pd.Series(dtype='int'), 'ngram': pd.Series(dtype='str')}).to_parquet(output_ngram_map_file)
            open(output_edge_file, 'w').close()
            print(f"  [Worker n={n_value}] Pass 1 & 2 Complete for n={n_value}. Empty intermediate files created.")
            return

        # Dask Bag creation - this is where the issue might start
        seq_bag = db.from_sequence(sequences, npartitions=num_dask_partitions)

        def get_ngrams_from_seq_tuple(seq_tuple):
            _, sequence_text = seq_tuple
            if not isinstance(sequence_text, str) or len(sequence_text) < n_value:
                return []
            return [sequence_text[i:i + n_value] for i in range(len(sequence_text) - n_value + 1)]

        # --- Add debugging prints around the first Dask compute ---
        print(f"  [Worker n={n_value}] Before first Dask compute (map/flatten/distinct)...")
        try:
            unique_ngrams_series = seq_bag.map(get_ngrams_from_seq_tuple).flatten().distinct().compute()
            print(f"  [Worker n={n_value}] After first Dask compute. {len(unique_ngrams_series)} unique n-grams found.")
        except Exception as e_dask_compute1:
            print(f"  [Worker n={n_value}] ERROR during first Dask compute: {e_dask_compute1}")
            import traceback
            traceback.print_exc()
            # Re-raise or handle the error appropriately
            raise e_dask_compute1 # Re-raise to make the test fail

        if not unique_ngrams_series:
            print(f"  [Worker n={n_value}] ⚠️ Warning: No n-grams of size n={n_value} were generated.")
            unique_ngrams_df = pd.DataFrame({'ngram': pd.Series(dtype='str')})
        else:
            unique_ngrams_df = pd.DataFrame(unique_ngrams_series, columns=['ngram'])

        unique_ngrams_df = unique_ngrams_df.sort_values('ngram').reset_index(drop=True)
        unique_ngrams_df['id'] = unique_ngrams_df.index
        unique_ngrams_df[['id', 'ngram']].to_parquet(output_ngram_map_file)
        print(f"  [Worker n={n_value}] Pass 1 Complete for n={n_value}. {len(unique_ngrams_df)} unique n-grams saved to map file.")

        if unique_ngrams_df.empty:
            open(output_edge_file, 'w').close()
            print(f"  [Worker n={n_value}] No n-grams for n={n_value}, skipping edge generation.")
            print(f"[Worker n={n_value}] Finished n={n_value} in {time.time() - process_start_time:.2f}s.")
            return

        print(f"  [Worker n={n_value}] Pass 2: Generating raw edge list for n={n_value}...")
        # Read the map back - this uses Pandas/PyArrow, another potential source of C-level issues
        try:
            ngram_to_id_map = pd.read_parquet(output_ngram_map_file).set_index('ngram')['id'].to_dict()
            print(f"  [Worker n={n_value}] Ngram map loaded for edge generation.")
        except Exception as e_read_parquet:
             print(f"  [Worker n={n_value}] ERROR reading ngram map for edge generation: {e_read_parquet}")
             import traceback
             traceback.print_exc()
             # Handle error - maybe create empty edge file and return?
             open(output_edge_file, 'w').close()
             print(f"[Worker n={n_value}] Finished n={n_value} with error in reading map.")
             return


        def sequence_to_edges_str_list(seq_tuple):
            _, sequence = seq_tuple
            if not isinstance(sequence, str) or len(sequence) < n_value + 1:
                return []
            edges_str = []
            # Accessing ngram_to_id_map from the outer scope - Dask handles this via pickling/serialization
            # This is another area where complex objects or serialization issues can arise.
            for i in range(len(sequence) - n_value):
                source_ngram = sequence[i:i + n_value]
                target_ngram = sequence[i + 1:i + 1 + n_value]
                source_id = ngram_to_id_map.get(source_ngram)
                target_id = ngram_to_id_map.get(target_ngram)
                if source_id is not None and target_id is not None:
                    edges_str.append(f"{source_id} {target_id}\n")
            return edges_str

        edge_lists_bag = seq_bag.map(sequence_to_edges_str_list).flatten()
        temp_edge_parts_dir = os.path.join(temp_dir, f"edge_parts_n{n_value}")
        if os.path.exists(temp_edge_parts_dir): shutil.rmtree(temp_edge_parts_dir)

        # --- Add debugging prints around the second Dask compute (to_textfiles) ---
        print(f"  [Worker n={n_value}] Before second Dask compute (to_textfiles)...")
        try:
            edge_lists_bag.to_textfiles(os.path.join(temp_edge_parts_dir, 'part-*.txt'))
            print(f"  [Worker n={n_value}] After second Dask compute (to_textfiles).")
        except Exception as e_dask_compute2:
            print(f"  [Worker n={n_value}] ERROR during second Dask compute (to_textfiles): {e_dask_compute2}")
            import traceback
            traceback.print_exc()
            # Handle error - maybe leave temp files for inspection?
            print(f"[Worker n={n_value}] Finished n={n_value} with error in to_textfiles.")
            return # Stop processing this n_value

        # ... (rest of file concatenation and cleanup) ...
        with open(output_edge_file, 'w') as outfile:
            part_files = sorted([f for f in os.listdir(temp_edge_parts_dir) if f.startswith('part-') and f.endswith('.txt')])
            for fname in part_files:
                with open(os.path.join(temp_edge_parts_dir, fname), 'r') as infile:
                    shutil.copyfileobj(infile, outfile)
        shutil.rmtree(temp_edge_parts_dir)

        print(f"  [Worker n={n_value}] Pass 2 Complete for n={n_value}. Raw edge file created at {output_edge_file}.")
        print(f"[Worker n={n_value}] Finished intermediate file construction for n={n_value} in {time.time() - process_start_time:.2f}s.")

    # ... (rest of GraphBuilder class) ...


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

        DataUtils.print_header(f"Phase 1: Creating intermediate files with {self.num_workers} Dask workers")
        phase1_start_time = time.time()
        effective_num_workers = max(1, self.num_workers)

        try:
            num_sequences = sum(1 for _ in DataLoader.parse_sequences(self.protein_sequence_file))
            sequences_per_partition_target = 500
            num_dask_partitions = max(1, min(effective_num_workers * 4, (num_sequences + sequences_per_partition_target - 1) // sequences_per_partition_target))
            print(f"  Estimated {num_sequences} sequences. Using {num_dask_partitions} Dask Bag partitions.")
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.protein_sequence_file}. Cannot proceed with graph building.")
            return

        # In src/pipeline/data_builder.py, inside GraphBuilder.run()

        # ... (other code) ...
        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set for the main process before cluster starts
        print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for Dask workers to avoid GPU contention.")

        try:
            # Explicitly set multiprocessing_method to 'spawn'
            with LocalCluster(
                    n_workers=effective_num_workers,
                    threads_per_worker=1,
                    silence_logs=logging.ERROR,
                    multiprocessing_method='spawn'  # <--- ADD THIS
            ) as cluster, Client(cluster) as client:
                print(f"  Dask LocalCluster started with {effective_num_workers} workers using 'spawn' method.")
                print(f"  Dask dashboard link: {client.dashboard_link}")

                # The _create_intermediate_files method already sets CUDA_VISIBLE_DEVICES for the worker
                tasks = [client.submit(GraphBuilder._create_intermediate_files, n, self.temp_dir, self.protein_sequence_file, num_dask_partitions) for n in n_values]

                for future in tqdm(tasks, desc="Dask Workers Progress (Phase 1)"):
                    try:
                        future.result()  # Wait for task completion and retrieve result (or raise exception)
                    except Exception as e:
                        print(f"ERROR in Dask worker during Phase 1 for a task: {e}")
                        # Optionally, log more details or decide if the whole process should stop
                        # For now, it will just print and continue with other tasks if any

        except Exception as e_cluster:
            print(f"ERROR setting up or running Dask LocalCluster: {e_cluster}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore CUDA_VISIBLE_DEVICES for the main process
            if original_cuda_visible_devices is None:
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
            print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")
        # ... (rest of the run method) ...

        print(f"<<< Phase 1 finished in {time.time() - phase1_start_time:.2f}s.")

        DataUtils.print_header("Phase 2: Building and saving final graph objects")
        phase2_start_time = time.time()
        for n in n_values:
            print(f"\n--- Processing n = {n} for final graph object ---")
            ngram_map_file = os.path.join(self.temp_dir, f'ngram_map_n{n}.parquet')
            edge_file = os.path.join(self.temp_dir, f'edge_list_n{n}.txt')

            if not os.path.exists(ngram_map_file) or not os.path.exists(edge_file):
                print(f"  Warning: Intermediate files for n={n} not found. Skipping graph generation for this n-gram size.")
                continue

            try:
                nodes_df = pd.read_parquet(ngram_map_file)
            except pyarrow.lib.ArrowInvalid as e:
                print(f"  ❌ Error: Could not read Parquet file for n={n} at: {ngram_map_file}. Details: {e}. Skipping.")
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

        DataUtils.print_header("Phase 3: Cleaning up temporary files")
        phase3_start_time = time.time()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  Temporary directory {self.temp_dir} cleaned up.")
        else:
            print("  Temporary directory not found, no cleanup needed.")
        print(f"<<< Phase 3 finished in {time.time() - phase3_start_time:.2f}s.")

        DataUtils.print_header(f"N-gram Graph Building FINISHED in {time.time() - overall_start_time:.2f}s")
