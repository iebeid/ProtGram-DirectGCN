# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 5.0 (Refactored pipeline execution to be data-driven)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import time
import mlflow
import random
import copy
import platform
import os
import tempfile
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Dict

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


def _run_embedding_generation_pipelines(config: Config) -> List[Dict[str, str]]:
    """Runs all configured embedding generation pipelines using a data-driven approach."""

    # Define all pipelines in a list of dictionaries for easy extension
    pipelines = [
        {"flag": "RUN_GCN_PIPELINE", "pre_runner": lambda: ProtGramBuilder(config).run(),
         "runner": lambda: ProtGramXGCNTrainer(config).run(),
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
            if "pre_runner" in p_config:
                p_config["pre_runner"]()

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


def run_pipeline_for_dataset(base_config: Config, fasta_path: Path):
    """
    Runs the entire end-to-end pipeline for a single FASTA file dataset.
    """
    config = copy.deepcopy(base_config)
    dataset_name = fasta_path.stem

    DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")

    config.SEQUENCE_FILE_PATHS = [fasta_path]
    print(f"  - This run will process sequences from: {fasta_path}")

    # Programmatically update all relevant output paths
    paths_to_specialize = [
        'RESULTS_GRAPH_OBJECTS_DIR', 'RESULTS_GCN_EMBEDDINGS_DIR',
        'RESULTS_W2V_EMBEDDINGS_DIR', 'RESULTS_LSTM_EMBEDDINGS_DIR',
        'RESULTS_TRANSFORMER_EMBEDDINGS_DIR', 'RESULTS_EVALUATION_DIR'
    ]
    for path_attr in paths_to_specialize:
        if hasattr(config, path_attr):
            original_path = getattr(config, path_attr)
            setattr(config, path_attr, original_path / dataset_name)


    generated_embedding_files = _run_embedding_generation_pipelines(config)

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
        else:
            print(f"Skipping Main PPI Evaluation for {dataset_name}: No embeddings were generated or specified.")
    else:
        print(f"Skipping Main PPI Evaluation for {dataset_name} as per configuration.")

    DataUtils.print_header(f"COMPLETED FULL PIPELINE FOR DATASET: {dataset_name.upper()}")


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
        webbrowser.open("http://127.0.0.1:5000")
        print("\nMLflow UI has been launched in your web browser.")
        print("The server is running in the background. It will terminate when you close this terminal.")
    else:
        print("--- Headless/SSH environment detected. ---")
        print("To view the MLflow UI, run the following command on your local machine:")
        print(f"\n  mlflow ui --backend-store-uri {tracking_uri}\n")
        print("If running on a remote server, you may need to use SSH port forwarding, for example:")
        print("  ssh -L 5000:localhost:5000 your_user@your_server")


def main():
    script_start_time = time.monotonic()
    base_config = Config()
    logger = FileLogger(base_config.LOG_DIR, enabled=base_config.ENABLE_FILE_LOGGING)

    with logger:
        try:
            DataUtils.print_header("Starting Protein-Protein Interaction Meta-Pipeline")
            if base_config.DEBUG_VERBOSE:
                print("--- Base Configuration Loaded ---")
                flags = {k: v for k, v in base_config.__dict__.items() if k.startswith("RUN_")}
                for key, value in flags.items():
                    print(f"  {key}: {value}")
                print("--------------------------")

            if base_config.USE_MLFLOW:
                mlflow.set_tracking_uri(base_config.MLFLOW_TRACKING_URI)

            setup_data(base_config)

            if base_config.RUN_INTEGRATED_TESTS:
                DataUtils.print_header("Running Integrated Test Suite (Once at Startup)")
                run_all_tests()
                DataUtils.print_header("Integrated Test Suite Finished. Continuing main pipeline...")

            if base_config.RUN_BENCHMARKING_PIPELINE:
                mlflow.set_experiment(base_config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
                with mlflow.start_run(run_name="GNN_Benchmark_Suite"):
                    GNNBenchmarker(base_config).run()

            if base_config.RUN_NETWORK_EMBEDDING_BENCHMARKING:
                mlflow.set_experiment(base_config.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME)
                with mlflow.start_run(run_name="NE_Benchmark_Suite"):
                    NetworkEmbeddingBenchmarker(base_config).run()

            with tempfile.TemporaryDirectory() as temp_dir:
                files_to_process = _get_fasta_files_to_process(base_config, Path(temp_dir))
                if not files_to_process:
                    print("\nERROR: No sequence files defined in config.SEQUENCE_FILE_PATHS. Cannot run experiments.")
                else:
                    print(f"\nFound {len(files_to_process)} dataset(s) to process.")
                    for fasta_file_path in files_to_process:
                        run_pipeline_for_dataset(base_config, fasta_file_path)

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