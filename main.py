# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 8.1 (Completed subprocess logic for automated model conversion)
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
import pandas as pd
import tensorflow as tf

# --- Robustness Improvement: Configure GPU Memory Growth for TensorFlow ---
# This should be done early, before TensorFlow allocates any memory.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # --- NEW: Enable mixed precision for performance ---
    # This uses float16 for computations on compatible GPUs (Tensor Cores),
    # which can significantly speed up inference and reduce memory usage.
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
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
from source.data_builders.fastprotgram import FastProtGramDataBuilder
from source.data_builders.protgram import ProtGramDataBuilder # Legacy
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
            # --- DEFINITIVE FIX for Memory Crash: Use Reservoir Sampling for large files ---
            # This avoids loading the entire FASTA file into memory for downsampling.
            print(f"  Counting sequences in {original_path.name} for sampling...")
            with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
                total_sequences = sum(1 for line in f if line.startswith('>'))

            if total_sequences == 0:
                print(f"  - WARNING: No sequences found in {original_path.name}. Skipping.")
                continue

            sample_size = int(total_sequences * config.SEQUENCE_DOWNSAMPLE_FRACTION)
            print(
                f"  - Sampling {sample_size} of {total_sequences} sequences from {original_path.name} using memory-efficient reservoir sampling...")

            # Create a fresh iterator and sample from it
            sequence_iterator = FastaUtils.parse_sequences([original_path])
            sampled_sequences = DataUtils.reservoir_sample(sequence_iterator, sample_size, config.RANDOM_STATE)

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
            webbrowser.open("http://127.0.0.1:5000")
            # --- FIX: Make background process behavior more explicit to the user ---
            print("\nMLflow UI has been launched in your web browser (or a new tab).")
            print("The MLflow server is running as a background process.")
            print("To stop it when you are finished, you may need to close this terminal or")
            print("manually find and stop the 'mlflow ui' process.")
        except webbrowser.Error as e:
            print(f"\nCould not automatically open web browser due to an error: {e}")
            print("Please open http://12.0.0.1:5000 manually to view results.")
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

    # --- FIX: Ensure the full table is always displayed without truncation ---
    # This prevents pandas from hiding columns or rows in a wide/long summary.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)  # Set a generous width for the console
    pd.set_option('display.max_colwidth', None)
    # --- END FIX ---

    try:
        # Concatenate all collected DataFrames
        final_summary_df = pd.concat(all_results, ignore_index=True)

        # Define the full set of columns for data processing
        all_columns = [
            'dataset', 'model',
            'Accuracy', 'F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)',
            'error'
        ]
        # Define the columns to actually display in the final table
        columns_to_print = [
            'model', 'Accuracy', 'F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'error'
        ]

        # Reorder and fill missing columns with NaN to ensure consistency
        final_summary_df = final_summary_df.reindex(columns=all_columns)

        # --- Grouping logic for clearer presentation ---
        def get_group_name(dataset_name):
            if not isinstance(dataset_name, str): return "Unknown"
            if 'ProtGram_n1_Singleton' in dataset_name:
                return dataset_name
            return dataset_name.replace('_Original', '')

        # --- FIX: Enforce a specific model order for consistency ---
        # This order reflects a logical grouping (e.g., standard GNNs, custom GNNs, NE models)
        model_order = [
            "GAT", "GraphSAGE", "GIN", "ChebNet", "Node2Vec", "GCN",
            "RGCN", "DirGNN", "DirectGCN"
        ]

        final_summary_df['dataset_group'] = final_summary_df['dataset'].apply(get_group_name)
        # Convert the 'model' column to a categorical type with the specified order.
        final_summary_df['model'] = pd.Categorical(final_summary_df['model'], categories=model_order, ordered=True)
        # Now sort by the dataset group and the new categorical model order
        final_summary_df = final_summary_df.sort_values(by=['dataset_group', 'model'])

        DataUtils.print_header("Aggregated Benchmark Summary")
        for group_name, group_df in final_summary_df.groupby('dataset_group', sort=False):
            print(f"\n--- Results for Dataset: {group_name} ---")
            # Format float columns for better readability
            formatted_group_df = group_df.copy()
            float_cols = formatted_group_df.select_dtypes(include=['float']).columns
            for col in float_cols:
                formatted_group_df[col] = formatted_group_df[col].map('{:.4f}'.format)

            print(formatted_group_df[columns_to_print].to_string(index=False, na_rep='NaN'))
    except Exception as e:
        print(f"Could not generate aggregated benchmark summary due to an error: {e}")


def _run_pre_analysis_and_prompt(config: Config, fasta_file_path: Path) -> bool:
    """
    Runs all preliminary benchmarks and the singleton GCN evaluation,
    displays a summary, and prompts the user to continue.
    """
    all_benchmark_results = []

    # --- Step 1: Run standard GNN and Network Embedding benchmarks ---
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
                all_benchmark_results.append(ne_results_df)

    # --- Step 2: Run the n=1 ProtGramBuilder and Singleton GCN Evaluation ---
    if config.RUN_SINGLETON_GCN_EVAL:
        DataUtils.print_header("Ensuring n=1 Graph is Built for Singleton Evaluation")
        singleton_config = copy.deepcopy(config)
        singleton_config.PROTGRAM_NGRAM_MAX_N = 1
        # This will build the n=1 graph or skip if it already exists using the configured builder.
        if config.USE_FAST_GRAPH_BUILDER:
            print("  Using FastProtGramDataBuilder for singleton graph.")
            FastProtGramDataBuilder(singleton_config).run()
        else:
            print("  Using legacy ProtGramDataBuilder for singleton graph.")
            ProtGramDataBuilder(singleton_config).run()

        # Now, explicitly load the graph and run the evaluation.
        n1_graph_path = singleton_config.RESULTS_GRAPH_OBJECTS_DIR / "ngram_graph_n1.pkl"
        if n1_graph_path.exists():
            print("  n=1 graph found. Proceeding with singleton evaluation...")
            from source.data_structures.graph import DirectedNgramGraph
            from source.trainers.singleton_xgcn import SingletonXGCNTrainer
            n1_graph: DirectedNgramGraph = DataUtils.load_object(str(n1_graph_path))
            if n1_graph:
                singleton_results_df = SingletonXGCNTrainer(singleton_config, n1_graph).run()
                if singleton_results_df is not None and not singleton_results_df.empty:
                    singleton_results_df = singleton_results_df.rename(columns={'Model': 'model'})
                    singleton_results_df['dataset'] = f"ProtGram_n1_Singleton_{fasta_file_path.stem}"
                    singleton_results_df['error'] = None
                    all_benchmark_results.append(singleton_results_df)
        else:
            print(f"  Warning: n=1 graph not found at {n1_graph_path}. Skipping singleton evaluation.")

    # --- Step 3: Display the aggregated summary ---
    _display_aggregated_benchmark_summary(all_benchmark_results)

    # --- Step 4: Prompt the user to continue, with a check for non-interactive sessions ---
    if not sys.stdin.isatty():
        print("\n--- Non-interactive session detected. Skipping user prompt and main pipelines. ---")
        print("--- To run the full pipeline, execute the script in an interactive terminal. ---")
        return False

    # --- FIX: Make the prompt more robust to I/O buffering issues from background processes. ---
    # By explicitly flushing stdout and reading directly from stdin, we can sometimes bypass
    # hangs caused by lingering resources from previous computational steps.
    print("\n" + "#" * 80)
    print("### PRELIMINARY ANALYSIS COMPLETE ###")
    print("#" * 80)
    print("\nDo you want to continue with the full, long-running pipelines for this dataset? (y/n): ")
    sys.stdout.flush()
    response = sys.stdin.readline().strip().lower()

    if response not in ['y', 'yes']:
        print("\nSkipping main pipeline as requested by user.")
        return False

    print("Continuing with the full pipeline...\n")
    return True


def main():
    script_start_time = time.monotonic()
    # Define project_root here to make the script self-contained
    project_root = Path(__file__).parent.resolve()
    base_config = Config()
    # --- NEW: Set all random seeds at the very beginning of the run ---
    # This is the primary fix for ensuring run-to-run reproducibility.
    DataUtils.set_seeds(base_config.RANDOM_STATE)
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
                # --- FIX: Make the consequence of a failed GPU test explicit to the user ---
                if not gpu_is_ok:
                    print("\n" + "!" * 80)
                    print("!!! WARNING: GPU verification failed for PyTorch or TensorFlow. !!!")
                    print("!!! The pipeline can continue, but it will run on the CPU, which may be very slow. !!!")
                    print("!" * 80 + "\n")
                    if sys.stdin.isatty(): # Only prompt in interactive sessions
                        try:
                            response = input("Do you want to continue with CPU-only execution? (y/n): ").lower().strip()
                            if response not in ['y', 'yes']:
                                print("Aborting as requested by user.")
                                sys.exit(1)
                        except (KeyboardInterrupt, EOFError):
                            print("Aborting as requested by user.")
                            sys.exit(1)

            with tempfile.TemporaryDirectory() as temp_dir:
                files_to_process = _get_fasta_files_to_process(base_config, Path(temp_dir))
                if not files_to_process:
                    print("\nERROR: No sequence files defined in config.SEQUENCE_FILE_PATHS. Cannot run experiments.")
                    return

                print(f"\nFound {len(files_to_process)} dataset(s) to process for the main pipeline.")

                # --- NEW: Pre-convert Transformer models by launching a separate, isolated process ---
                if base_config.RUN_TRANSFORMER_PIPELINE:
                    DataUtils.print_header("Verifying Transformer Model Availability")
                    from huggingface_hub import model_info
                    for model_cfg in base_config.TRANSFORMER_MODELS_TO_RUN:
                        model_id = model_cfg['hf_id']
                        local_path = base_config.DATA_MODELS_DIR / model_id
                        if not (local_path.exists() and (local_path / "tf_model.h5").exists()):
                            try:
                                info = model_info(model_id)
                                # A robust check for PyTorch-only models
                                is_pytorch_only = "tensorflow" not in info.tags and "tf" not in info.tags
                                if is_pytorch_only:
                                    print(f"\n--- ACTION: Model '{model_id}' is PyTorch-only and requires conversion. ---")
                                    print("  This is a one-time, memory-intensive step.")
                                    print("  Launching conversion in a separate, isolated process to prevent OOM errors...")
                                    # The conversion script is the models.py utility itself.
                                    conversion_script_path = project_root / "source" / "utils" / "models.py"
                                    try:
                                        # Run the conversion script as a separate process.
                                        # This is critical to isolate its memory usage from the main pipeline.
                                        subprocess.run(
                                            [sys.executable, str(conversion_script_path), model_id],
                                            check=True, text=True, capture_output=False # Stream output directly
                                        )
                                        print(f"  ✅ Conversion process for '{model_id}' completed successfully.")
                                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                                        print(f"  ❌ ERROR: The conversion process for '{model_id}' failed: {e}")
                                        print("       The main pipeline will continue, but may fail when trying to load this model.")
                            except Exception as e:
                                print(f"  Warning: Could not verify model '{model_id}' on Hugging Face Hub: {e}")
                # --- END NEW ---

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
                        if config.USE_FAST_GRAPH_BUILDER:
                            print("  Using FastProtGramDataBuilder for main graph build.")
                            FastProtGramDataBuilder(config).run()
                        else:
                            print("  Using legacy ProtGramDataBuilder for main graph build.")
                            ProtGramDataBuilder(config).run()

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