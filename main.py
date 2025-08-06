# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 6.0 (Corrected pre-analysis and prompting workflow)
# AUTHOR: Islam Ebeid
# ==============================================================================

import copy
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import List, Dict

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Robustness Improvement: Configure GPU Memory Growth for TensorFlow ---
# This should be done early, before TensorFlow allocates any memory.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Warning: Could not set memory growth for GPUs: {e}")

# --- NEW: Suppress noisy TensorFlow informational logs ---
# Sets the log level to '2', which corresponds to WARNING.
# This will hide the benign 'I' (INFO) messages like 'Filling up shuffle buffer'
# and 'End of sequence', making the log easier to read.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('WARNING')
# --- END NEW ---

from configuration.config import Config
from configuration.data import setup_data
from source.data_builders.protgram import ProtGramBuilder
from source.benchmarkers.gnns import GNNBenchmarker
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker
from source.experiments.ppi_1 import PPIPipeline
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.lstm import LSTMBasedEmbedder
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.data import DataUtils, FastaUtils
from source.testers.unit_tests import run_all_tests
from source.utils.logging import FileLogger


def _run_main_embedding_pipelines(config: Config) -> List[Dict[str, str]]:
    """Runs the main, potentially long-running, embedding generation pipelines."""

    pipelines = [
        {"flag": "RUN_GCN_PIPELINE", "runner": lambda: ProtGramXGCNTrainer(config).run(),
         "formatter": lambda paths: [{"name": name, "path": path} for name, path in paths.items()]},
        {"flag": "RUN_WORD2VEC_PIPELINE", "runner": lambda: Word2VecEmbedder(config).run(),
         "formatter": lambda path: [{"name": "Word2Vec-Generated", "path": path}]},
        {"flag": "RUN_LSTM_PIPELINE", "runner": lambda: LSTMBasedEmbedder(config).run(),
         "formatter": lambda path: [{"name": "LSTM-Generated", "path": path}]},
        {"flag": "RUN_TRANSFORMER_PIPELINE", "runner": lambda: TransformerEmbedder(config).run(),
         "formatter": lambda paths: [{"name": f"{name}-Generated", "path": str(path)} for name, path in paths.items()]}
    ]

    generated_files = []
    for p_config in pipelines:
        if getattr(config, p_config["flag"], False):
            result = p_config["runner"]()
            if result:
                generated_files.extend(p_config["formatter"](result))

    return generated_files


def _get_fasta_files_to_process(config: Config, temp_dir: Path) -> List[Path]:
    """
    Handles the logic for downsampling FASTA files.
    Returns a list of paths to the files that should be processed in the main loop.
    """
    files_to_process = []

    # --- NEW: Interactive FASTA file selection ---
    if len(config.ORIGINAL_SEQUENCE_FILE_PATHS) > 1:
        print("\n--- Multiple FASTA files found. Please choose one to process for this run: ---")
        for i, path in enumerate(config.ORIGINAL_SEQUENCE_FILE_PATHS):
            print(f"  [{i + 1}] {path.name}")

        while True:
            try:
                choice = int(input(f"Enter number (1-{len(config.ORIGINAL_SEQUENCE_FILE_PATHS)}): "))
                if 1 <= choice <= len(config.ORIGINAL_SEQUENCE_FILE_PATHS):
                    chosen_path = config.ORIGINAL_SEQUENCE_FILE_PATHS[choice - 1]
                    print(f"You selected: {chosen_path.name}")
                    # The rest of the pipeline will now run on only this file.
                    config.ORIGINAL_SEQUENCE_FILE_PATHS = [chosen_path]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    # --- END NEW ---

    should_downsample = config.SEQUENCE_DOWNSAMPLE_FRACTION and 0 < config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0

    if should_downsample:
        DataUtils.print_header(f"Downsampling FASTA files ({config.SEQUENCE_DOWNSAMPLE_FRACTION:.1%})")
        random.seed(config.RANDOM_STATE)

        for original_path in config.ORIGINAL_SEQUENCE_FILE_PATHS:
            # More efficient: Read all sequences into memory once, then sample.
            all_sequences = list(FastaUtils.parse_sequences([original_path]))
            if not all_sequences:
                print(f"  - WARNING: No sequences found in {original_path.name}. Skipping.")
                continue

            sample_size = int(len(all_sequences) * config.SEQUENCE_DOWNSAMPLE_FRACTION)
            print(f"  - Sampling {sample_size} of {len(all_sequences)} sequences from {original_path.name}")
            sampled_sequences = random.sample(all_sequences, sample_size)

            temp_fasta_path = temp_dir / f"{original_path.stem}_sampled.fasta"
            with open(temp_fasta_path, "w") as f:
                for seq_id, sequence in sampled_sequences:
                    f.write(f">{seq_id}\n{sequence}\n")
            files_to_process.append(temp_fasta_path)
    else:
        print("\nNo downsampling requested. Using original FASTA files for experiments.")
        files_to_process = config.ORIGINAL_SEQUENCE_FILE_PATHS.copy()

    return files_to_process


def _launch_mlflow_ui(config: Config):
    """Starts the MLflow UI and opens a browser if in a desktop environment."""
    if not config.USE_MLFLOW:
        return

    DataUtils.print_header("Launching MLflow UI")
    tracking_uri = config.MLFLOW_TRACKING_URI
    is_desktop_env = os.environ.get('DISPLAY') or platform.system() == "Windows"

    if is_desktop_env:
        print("Desktop environment detected. Starting MLflow UI in the background...")
        # Use Popen to run in the background. Redirect output to hide it.
        subprocess.Popen(
            ["mlflow", "ui", "--backend-store-uri", tracking_uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Give the server a moment to start
        time.sleep(5)
        try:
            # webbrowser.open() returns True on success, False on failure.
            was_opened = webbrowser.open("http://127.0.0.1:5000")
            if not was_opened:
                print("\nCould not automatically open web browser.")
                print("Please open http://127.0.0.1:5000 manually to view results.")
            else:
                print("\nMLflow UI has been launched in your web browser.")
                print("The server is running in the background. It will terminate when you close this terminal.")
        except webbrowser.Error as e:
            print(f"\nCould not automatically open web browser due to an error: {e}")
            print("Please open http://127.0.0.1:5000 manually to view results.")
    else:
        print("--- Headless/SSH environment detected. ---")
        print("To view the MLflow UI, run the following command on your local machine:")
        print(f"\n  mlflow ui --backend-store-uri {tracking_uri}\n")
        print("If running on a remote server, you may need to use SSH port forwarding, for example:")
        print("  ssh -L 5000:localhost:5000 your_user@your_server")


def _display_aggregated_benchmark_summary(all_results: List[pd.DataFrame]):
    """
    Standardizes, concatenates, and displays a final summary of all benchmark results.
    """
    if not all_results:
        print("No benchmark results were generated to aggregate.")
        return

    try:
        # Concatenate all collected DataFrames
        final_summary_df = pd.concat(all_results, ignore_index=True)

        # Define the desired final column order
        final_columns = ['dataset', 'model', 'test_accuracy', 'best_val_accuracy', 'error']
        # Reorder and fill missing columns with NaN
        final_summary_df = final_summary_df.reindex(columns=final_columns)

        DataUtils.print_header("Aggregated Benchmark Summary")
        print(final_summary_df.to_string())
    except Exception as e:
        print(f"Could not generate aggregated benchmark summary due to an error: {e}")


def _run_pre_analysis_and_prompt(config: Config, fasta_file_path: Path) -> bool:
    """
    Runs all preliminary benchmarks and the singleton GCN evaluation,
    displays a summary, and prompts the user to continue.
    """
    all_benchmark_results = []

    # --- Step 1: Run standard GNN and Network Embedding benchmarks. ---
    if config.RUN_BENCHMARKING_PIPELINE:
        mlflow.set_experiment(config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="GNN_Benchmark_Suite"):
            gnn_results_df = GNNBenchmarker(config).run()
            if gnn_results_df is not None and not gnn_results_df.empty:
                all_benchmark_results.append(gnn_results_df)

    if config.RUN_NETWORK_EMBEDDING_BENCHMARKING:
        mlflow.set_experiment(config.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="NE_Benchmark_Suite"):
            ne_results_df = NetworkEmbeddingBenchmarker(config).run()
            if ne_results_df is not None and not ne_results_df.empty:
                # Standardize columns to match GNN benchmark format
                ne_results_df['best_val_accuracy'] = np.nan
                all_benchmark_results.append(ne_results_df)

    # --- Step 2: Run the n=1 ProtGramBuilder step and Singleton GCN Evaluation. ---
    if config.RUN_SINGLETON_GCN_EVAL:
        DataUtils.print_header("Running Singleton (n=1) Graph Evaluation")
        # Create a temporary config to only build the n=1 graph for this step
        singleton_config = copy.deepcopy(config)
        singleton_config.GCN_NGRAM_MAX_N = 1

        # The run method of ProtGramBuilder will build the n=1 graph and run the eval
        singleton_results_df = ProtGramBuilder(singleton_config).run(run_singleton_eval=True)

        if singleton_results_df is not None and not singleton_results_df.empty:
            # Standardize singleton results to fit the benchmark table
            singleton_results_df = singleton_results_df.rename(columns={'Model': 'model', 'Accuracy': 'test_accuracy'})
            singleton_results_df['dataset'] = f"ProtGram_n1_Singleton_{fasta_file_path.stem}"
            singleton_results_df['best_val_accuracy'] = np.nan
            singleton_results_df['error'] = None
            # Ensure the column order matches for concatenation
            all_benchmark_results.append(singleton_results_df[['dataset', 'model', 'test_accuracy', 'best_val_accuracy', 'error']])

    # --- Step 3: Display the aggregated summary. ---
    _display_aggregated_benchmark_summary(all_benchmark_results)

    # --- Step 4: Prompt the user to continue. ---
    # This prompt now appears after all preliminary results have been shown.
    response = input("\nDo you want to continue with the full, long-running pipelines for this dataset? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Skipping main pipeline as requested by user.")
        return False

    print("Continuing with the full pipeline...\n")
    return True


def main():
    script_start_time = time.monotonic()
    base_config = Config()
    logger = FileLogger(base_config.LOG_DIR, enabled=base_config.ENABLE_FILE_LOGGING)

    with logger:
        try:
            DataUtils.print_header("Starting Protein-Protein Interaction Meta-Pipeline")
            if base_config.DEBUG_VERBOSE:
                # FIX: Print all public configuration values for complete transparency.
                print("--- Full Configuration Values ---")
                for key, value in sorted(vars(base_config).items()):
                    # Exclude private/internal attributes for cleaner logs
                    if not key.startswith('_'):
                        print(f"  {key:<40} | {value}")
                print("--------------------------")

            if base_config.USE_MLFLOW:
                mlflow.set_tracking_uri(base_config.MLFLOW_TRACKING_URI)

            setup_data(base_config)

            if base_config.RUN_INTEGRATED_TESTS:
                DataUtils.print_header("Running Integrated Test Suite (Once at Startup)")
                gpu_is_ok = run_all_tests()
                DataUtils.print_header("Integrated Test Suite Finished. Continuing main pipeline...")
                if not gpu_is_ok:
                    print("\n" + "!" * 80)
                    print("!!! WARNING: GPU verification failed for PyTorch or TensorFlow. !!!")
                    print("!!! The pipeline can continue, but it will run on the CPU, which may be very slow. !!!")
                    print("!" * 80)
                    response = input("Do you want to continue with CPU-only execution? (y/n): ").lower().strip()
                    if response not in ['y', 'yes']:
                        print("Aborting as requested by user.")
                        sys.exit(1)

            with tempfile.TemporaryDirectory() as temp_dir:
                files_to_process = _get_fasta_files_to_process(base_config, Path(temp_dir))
                if not files_to_process:
                    print("\nERROR: No sequence files defined in config.SEQUENCE_FILE_PATHS. Cannot run experiments.")
                else:
                    print(f"\nFound {len(files_to_process)} dataset(s) to process for the main pipeline.")
                    for fasta_file_path in files_to_process:
                        # --- FIX: Create a dataset-specific config to prevent overwriting results ---
                        config = copy.deepcopy(base_config)
                        dataset_name = fasta_file_path.stem
                        DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")
                        config.SEQUENCE_FILE_PATHS = [fasta_file_path]
                        print(f"  - This run will process sequences from: {fasta_file_path}")

                        # Programmatically update all relevant output paths
                        paths_to_specialize = [
                            'RESULTS_GRAPH_OBJECTS_DIR', 'RESULTS_GCN_EMBEDDINGS_DIR',
                            'RESULTS_W2V_EMBEDDINGS_DIR', 'RESULTS_LSTM_EMBEDDINGS_DIR',
                            'RESULTS_TRANSFORMER_EMBEDDINGS_DIR', 'RESULTS_EVALUATION_DIR'
                        ]
                        for path_attr in paths_to_specialize:
                            if hasattr(config, path_attr):
                                # The base path is from the original config object
                                original_path = getattr(base_config, path_attr)
                                setattr(config, path_attr, original_path / dataset_name)
                        # --- END FIX ---

                        # Run pre-analysis and prompt the user. If they agree, run the main pipelines.
                        if _run_pre_analysis_and_prompt(config, fasta_file_path):
                            # If the user proceeds, we must now build the FULL set of graphs (n=1 to 3)
                            # before running the main embedding pipelines.
                            DataUtils.print_header("Building all n-gram graphs for the main pipeline")
                            ProtGramBuilder(config).run(run_singleton_eval=False)  # Re-run, but skip the now-redundant eval

                            generated_embedding_files = _run_main_embedding_pipelines(config)
                            final_evaluation_list = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_embedding_files
                            config.LP_EMBEDDING_FILES_TO_EVALUATE = final_evaluation_list

                            if config.RUN_MAIN_PPI_EVALUATION:
                                if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                    DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
                                    ppi_evaluator = PPIPipeline(config)
                                    if config.USE_MLFLOW:
                                        mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT_NAME}-{dataset_name}")
                                        with mlflow.start_run(run_name=f"PPI_Evaluation_Full_Run") as parent_run:
                                            mlflow.set_tag("dataset_name", dataset_name)
                                            ppi_evaluator.run(use_dummy_data=False, parent_run_id=parent_run.info.run_id)
                                    else:
                                        ppi_evaluator.run(use_dummy_data=False)
                        DataUtils.print_header(f"COMPLETED FULL PIPELINE FOR DATASET: {dataset_name.upper()}")

            DataUtils.print_header(f"Full Orchestration Finished in {time.monotonic() - script_start_time:.2f} seconds.")

            # Launch MLflow UI at the very end
            _launch_mlflow_ui(base_config)

        except Exception as e:
            print(f"\n--- PIPELINE FAILED ---")
            print(f"An error occurred during execution: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
