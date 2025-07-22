# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 2.3 (Corrected PPI pipeline instantiation order and improved clarity)
# AUTHOR: Islam Ebeid
# ==============================================================================

import time
import mlflow
import random
import copy
import platform
import os
import tempfile
from pathlib import Path
from typing import List, Dict

import tensorflow as tf

# --- Robustness Improvement: Configure GPU Memory Growth for TensorFlow ---
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
    """Runs all configured embedding generation pipelines and returns a list of generated file paths."""
    generated_files = []
    if config.RUN_GCN_PIPELINE:
        # First, build the graphs
        ProtGramBuilder(config).run()
        # Then, train the GCNs on those graphs
        if gcn_paths := ProtGramXGCNTrainer(config).run():
            for name, path in gcn_paths.items():
                generated_files.append({"name": name, "path": path})
    if config.RUN_WORD2VEC_PIPELINE:
        if w2v_path := Word2VecEmbedder(config).run():
            generated_files.append({"name": "Word2Vec-Generated", "path": w2v_path})
    if config.RUN_LSTM_PIPELINE:
        if lstm_path := LSTMBasedEmbedder(config).run():
            generated_files.append({"name": "LSTM-Generated", "path": lstm_path})
    if config.RUN_TRANSFORMER_PIPELINE:
        if transformer_paths := TransformerEmbedder(config).run():
            for name, path in transformer_paths.items():
                generated_files.append({"name": f"{name}-Generated", "path": str(path)})
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

        for original_path in config.SEQUENCE_FILE_PATHS:
            sequences_from_file = list(FastaUtils.parse_sequences([original_path]))
            if not sequences_from_file:
                print(f"  - WARNING: No sequences found in {original_path.name}. Skipping.")
                continue
            sample_size = int(len(sequences_from_file) * config.SEQUENCE_DOWNSAMPLE_FRACTION)
            print(f"  - Sampling {sample_size} of {len(sequences_from_file)} sequences from {original_path.name}")

            sampled_sequences = random.sample(sequences_from_file, sample_size)

            temp_fasta_path = temp_dir / f"{original_path.stem}_sampled.fasta"
            with open(temp_fasta_path, "w") as f:
                for seq_id, sequence in sampled_sequences:
                    f.write(f">{seq_id}\n{sequence}\n")
            files_to_process.append(temp_fasta_path)
    else:
        print("\nNo downsampling requested. Using original FASTA files for experiments.")
        files_to_process = config.SEQUENCE_FILE_PATHS.copy()

    return files_to_process


def run_pipeline_for_dataset(base_config: Config, fasta_path: Path):
    """
    Runs the entire end-to-end pipeline for a single FASTA file dataset.
    """
    config = copy.deepcopy(base_config)
    dataset_name = fasta_path.stem

    DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")

    # --- 1. Modify Configuration to be Dataset-Specific ---
    config.SEQUENCE_FILE_PATHS = [fasta_path]
    print(f"  - This run will process sequences from: {fasta_path}")

    config.RESULTS_GRAPH_OBJECTS_DIR /= dataset_name
    config.RESULTS_GCN_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_W2V_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_LSTM_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_EVALUATION_DIR /= dataset_name

    # --- 2. Run Embedding Generation Pipelines for this Dataset ---
    generated_embedding_files = _run_embedding_generation_pipelines(config)

    # --- 3. Run PPI Evaluation for this Dataset ---
    # CRITICAL FIX: Finalize the list of embeddings BEFORE instantiating the pipeline.
    final_evaluation_list = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_embedding_files
    config.LP_EMBEDDING_FILES_TO_EVALUATE = final_evaluation_list

    if config.RUN_MAIN_PPI_EVALUATION:
        if config.LP_EMBEDDING_FILES_TO_EVALUATE:
            DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
            # Instantiate the evaluator HERE, with the complete config
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


def main():
    script_start_time = time.monotonic()

    if platform.system() == "Linux":
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_bin_path = os.path.join(conda_prefix, "bin")
            cc_path = os.path.join(conda_bin_path, "x86_64-conda-linux-gnu-cc")
            cxx_path = os.path.join(conda_bin_path, "x86_64-conda-linux-gnu-c++")
            if os.path.exists(conda_bin_path) and os.path.exists(cxx_path):
                print("--- Forcing environment to use compilers from active Conda env ---")
                os.environ["PATH"] = conda_bin_path + os.pathsep + os.environ.get("PATH", "")
                os.environ["CC"] = cc_path
                os.environ["CXX"] = cxx_path

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
                mlruns_path = base_config.BASE_OUTPUT_DIR / "mlruns"
                mlruns_path.mkdir(parents=True, exist_ok=True)
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

        except Exception as e:
            print(f"\n--- PIPELINE FAILED ---")
            print(f"An error occurred during execution: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    main()