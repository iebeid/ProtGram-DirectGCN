# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 2.0 (Refactored to run as a meta-pipeline, one experiment per FASTA file)
# AUTHOR: Islam Ebeid
# ==============================================================================

import time
import mlflow
import random
import os
import copy
import shutil
from pathlib import Path

from configuration.config import Config
from configuration.data import setup_data
from source.data_builders.protgram import GraphBuilder
from source.benchmarkers.gnns import GNNBenchmarker
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker
from source.experiments.ppi_1 import PPIPipeline
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.lstm import LSTMBasedEmbedder
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.data import DataLoader
from source.testers.unit_tests import run_all_tests
from source.utils.data import DataUtils
from source.utils.logging import start_logging, stop_logging


def run_pipeline_for_dataset(base_config: Config, fasta_path: Path):
    """
    Runs the entire end-to-end pipeline for a single FASTA file dataset.
    """
    # Create a deep copy of the base config to ensure each run is isolated
    config = copy.deepcopy(base_config)
    dataset_name = fasta_path.stem  # e.g., 'uniprot_sprot'

    DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")

    # --- 1. Modify Configuration to be Dataset-Specific ---
    # Point all subsequent modules to the single FASTA file for this run
    config.SEQUENCE_FILE_PATHS = [fasta_path]

    # Create dataset-specific output directories to prevent overwriting results
    config.RESULTS_GRAPH_OBJECTS_DIR /= dataset_name
    config.RESULTS_GCN_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_W2V_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_LSTM_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR /= dataset_name
    config.RESULTS_EVALUATION_DIR /= dataset_name

    # --- 2. Run Embedding Generation Pipelines for this Dataset ---
    generated_embedding_files = []

    if config.RUN_GCN_PIPELINE:
        graph_builder = GraphBuilder(config)
        graph_builder.run()
        gcn_trainer = ProtGramXGCNTrainer(config)
        gcn_embedding_paths = gcn_trainer.run()
        if gcn_embedding_paths:
            for model_name, path in gcn_embedding_paths.items():
                generated_embedding_files.append({"name": f"{model_name}-ProtGram", "path": path})

    if config.RUN_WORD2VEC_PIPELINE:
        word2vec_embedder = Word2VecEmbedder(config)
        w2v_embedding_path = word2vec_embedder.run()
        if w2v_embedding_path:
            generated_embedding_files.append({"name": "Word2Vec-Generated", "path": w2v_embedding_path})

    if config.RUN_LSTM_PIPELINE:
        lstm_embedder = LSTMBasedEmbedder(config)
        lstm_embedding_path = lstm_embedder.run()
        if lstm_embedding_path:
            generated_embedding_files.append({"name": "LSTM-Generated", "path": lstm_embedding_path})

    if config.RUN_TRANSFORMER_PIPELINE:
        transformer_embedder = TransformerEmbedder(config)
        transformer_embedding_paths = transformer_embedder.run()
        if transformer_embedding_paths:
            for model_name, path in transformer_embedding_paths.items():
                generated_embedding_files.append({"name": f"{model_name}-Generated", "path": path})

    # --- 3. Run PPI Evaluation for this Dataset ---
    should_setup_ppi_evaluator = config.RUN_DUMMY_TEST or config.RUN_MAIN_PPI_EVALUATION

    # Combine external embeddings with newly generated ones for the main evaluation
    final_evaluation_list = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_embedding_files
    config.LP_EMBEDDING_FILES_TO_EVALUATE = final_evaluation_list  # Override the config

    if should_setup_ppi_evaluator:
        ppi_evaluator = PPIPipeline(config)

        if config.RUN_MAIN_PPI_EVALUATION:
            if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
                if config.USE_MLFLOW:
                    # Create a unique experiment name for each dataset's evaluation
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
    should_downsample = False  # Initialize before the try block
    base_config = Config()

    if base_config.ENABLE_FILE_LOGGING:
        start_logging(base_config.LOG_DIR)

    try:
        DataUtils.print_header("Starting Protein-Protein Interaction Meta-Pipeline")
        if base_config.DEBUG_VERBOSE:
            print("--- Base Configuration Loaded ---")
            # Display a subset of the config for brevity
            flags = {k: v for k, v in base_config.__dict__.items() if k.startswith("RUN_")}
            for key, value in flags.items(): print(f"  {key}: {value}")
            print("--------------------------")

        # --- 1. Initial Setup: Download all required data ---
        setup_data(base_config)

        # --- 2. Initial Setup: Run integrated test suite if enabled ---
        if base_config.RUN_INTEGRATED_TESTS:
            DataUtils.print_header("Running Integrated Test Suite (Once at Startup)")
            run_all_tests()
            DataUtils.print_header("Integrated Test Suite Finished. Continuing main pipeline...")

        # --- 3. Initial Setup: Run Benchmarking Suites (They are dataset-agnostic) ---
        if base_config.USE_MLFLOW:
            mlflow.set_tracking_uri(base_config.MLFLOW_TRACKING_URI)

        if base_config.RUN_BENCHMARKING_PIPELINE:
            mlflow.set_experiment(base_config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="GNN_Benchmark_Suite") as run:
                GNNBenchmarker(base_config).run()

        if base_config.RUN_NETWORK_EMBEDDING_BENCHMARKING:
            mlflow.set_experiment(base_config.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="NE_Benchmark_Suite") as run:
                NetworkEmbeddingBenchmarker(base_config).run()

        # --- 4. Setup for Main Experimental Loop: Handle Downsampling ---
        should_downsample = base_config.SEQUENCE_DOWNSAMPLE_FRACTION and 0 < base_config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0
        files_to_process = []

        if should_downsample:
            DataUtils.print_header(f"Downsampling FASTA files ({base_config.SEQUENCE_DOWNSAMPLE_FRACTION:.1%})")
            temp_dir = base_config.TEMP_SAMPLED_DIR
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)
            random.seed(base_config.RANDOM_STATE)

            for original_path in base_config.SEQUENCE_FILE_PATHS:
                sequences_from_file = list(DataLoader.parse_sequences([original_path]))
                if not sequences_from_file:
                    print(f"  - WARNING: No sequences found in {original_path.name}. Skipping.")
                    continue
                sample_size = int(len(sequences_from_file) * base_config.SEQUENCE_DOWNSAMPLE_FRACTION)
                print(f"  - Sampling {sample_size} of {len(sequences_from_file)} sequences from {original_path.name}")

                sampled_sequences = random.sample(sequences_from_file, sample_size)

                temp_fasta_path = temp_dir / f"{original_path.stem}_sampled.fasta"
                with open(temp_fasta_path, "w") as f:
                    for seq_id, sequence in sampled_sequences:
                        f.write(f">{seq_id}\n{sequence}\n")
                files_to_process.append(temp_fasta_path)
        else:
            print("\nNo downsampling requested. Using original FASTA files for experiments.")
            files_to_process = base_config.SEQUENCE_FILE_PATHS.copy()

        # --- 5. Main Experimental Loop: Iterate over each dataset (original or sampled) ---
        if not files_to_process:
            print("\nERROR: No sequence files defined in config.SEQUENCE_FILE_PATHS. Cannot run experiments.")
        else:
            print(f"\nFound {len(files_to_process)} dataset(s) to process.")
            for fasta_file_path in files_to_process:
                run_pipeline_for_dataset(base_config, fasta_file_path)

        DataUtils.print_header(f"Full Orchestration Finished in {time.monotonic() - script_start_time:.2f} seconds.")
    except Exception as e:
        # Log any exceptions that occur during the pipeline
        print(f"\n--- PIPELINE FAILED ---")
        print(f"An error occurred during execution: {e}")
        # Re-raise the exception to see the full traceback in the log
        raise

    finally:
        # This block will always run, even if an error occurs,
        # ensuring that logging is properly stopped.
        if should_downsample and base_config.TEMP_SAMPLED_DIR.exists():
            print(f"Cleaning up temporary downsampled FASTA directory: {base_config.TEMP_SAMPLED_DIR}")
            shutil.rmtree(base_config.TEMP_SAMPLED_DIR)

        if base_config.ENABLE_FILE_LOGGING:
            stop_logging()


if __name__ == '__main__':
    main()