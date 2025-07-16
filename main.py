# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 1.2 (Uses config.RUN_MAIN_PPI_EVALUATION directly)
# AUTHOR: Islam Ebeid
# ==============================================================================

import time

import mlflow

from configuration.config import Config
from configuration.data import setup_data
from source.data_builders.protgram import GraphBuilder
from source.benchmarkers.gnns import GNNBenchmarker
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker
from source.experiments.ppi_1 import PPIPipeline
from source.trainers.transformers import TransformerEmbedder
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.word2vec import Word2VecEmbedder
from source.testers.unit_tests import run_all_tests
from source.utils.data import DataUtils
from source.utils.logging import start_logging, stop_logging


def main():
    script_start_time = time.monotonic()
    DataUtils.print_header("Starting Protein-Protein Interaction Pipeline")

    config = Config()
    if config.DEBUG_VERBOSE:
        print("--- Configuration Loaded ---")
        for key, value in config.__dict__.items():
            if not key.startswith("__"):
                print(f"  {key}: {value}")
        print("--------------------------")

    # List to hold paths of embeddings generated during this run
    generated_embedding_files = []

    # Start logging at the very beginning of your script
    if config.ENABLE_FILE_LOGGING:
        start_logging(config.LOG_DIR)

    try:
        print("Starting the main pipeline...")
        setup_data(config)

        # --- Run Integrated Test Suite (Optional) ---
        if config.RUN_INTEGRATED_TESTS:
            DataUtils.print_header("Running Integrated Test Suite")
            run_all_tests()
            DataUtils.print_header("Integrated Test Suite Finished. Continuing main pipeline...")

        if config.USE_MLFLOW:
            print(f"MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            print(f"MLflow experiment for GNN Benchmarking: {config.MLFLOW_BENCHMARK_EXPERIMENT_NAME}")
            print(f"MLflow experiment for PPI Evaluation: {config.MLFLOW_EXPERIMENT_NAME}")

        if config.RUN_BENCHMARKING_PIPELINE:
            if config.USE_MLFLOW:
                mlflow.set_experiment(config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
                with mlflow.start_run(run_name="GNN_Benchmark_Suite_Parent") as benchmark_parent_run:
                    mlflow.set_tag("suite_type", "GNN Benchmarking")
                    print(f"MLflow Parent Run for Benchmarking: {benchmark_parent_run.info.run_id}")
                    benchmarker = GNNBenchmarker(config)
                    benchmarker.run()
            else:
                benchmarker = GNNBenchmarker(config)
                benchmarker.run()

        if config.RUN_NETWORK_EMBEDDING_BENCHMARKING:
            if config.USE_MLFLOW:
                mlflow.set_experiment(config.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME)
                with mlflow.start_run(run_name="NE_Benchmark_Suite_Parent") as ne_parent_run:
                    mlflow.set_tag("suite_type", "Network Embedding Benchmarking")
                    print(f"MLflow Parent Run for NE Benchmarking: {ne_parent_run.info.run_id}")
                    ne_benchmarker = NetworkEmbeddingBenchmarker(config)
                    ne_benchmarker.run()
            else:
                ne_benchmarker = NetworkEmbeddingBenchmarker(config)
                ne_benchmarker.run()

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

        if config.RUN_TRANSFORMER_PIPELINE:
            transformer_embedder = TransformerEmbedder(config)
            transformer_embedding_paths = transformer_embedder.run()
            if transformer_embedding_paths:
                for model_name, path in transformer_embedding_paths.items():
                    generated_embedding_files.append({"name": f"{model_name}-Generated", "path": path})

        # Determine if any evaluation (dummy or main) should be set up
        should_setup_ppi_evaluator = config.RUN_DUMMY_TEST or config.RUN_MAIN_PPI_EVALUATION

        # Combine external embeddings with newly generated ones for the main evaluation
        final_evaluation_list = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_embedding_files
        config.LP_EMBEDDING_FILES_TO_EVALUATE = final_evaluation_list  # Override the config

        if should_setup_ppi_evaluator:
            if config.USE_MLFLOW:
                mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            ppi_evaluator = PPIPipeline(config)

            if config.RUN_DUMMY_TEST:
                DataUtils.print_header("Running Dummy Evaluation for PPI Pipeline")
                if config.USE_MLFLOW:
                    with mlflow.start_run(run_name="PPI_Dummy_Evaluation_Parent") as parent_run:
                        mlflow.set_tag("run_type", "ppi_dummy_test")
                        ppi_evaluator.run(use_dummy_data=True, parent_run_id=parent_run.info.run_id)
                else:
                    ppi_evaluator.run(use_dummy_data=True)

            if config.RUN_MAIN_PPI_EVALUATION:
                if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                    DataUtils.print_header("Running Main Evaluation for PPI Pipeline")
                    print("\nNote: The main PPI evaluation will now run on the files specified in your config's LP_EMBEDDING_FILES_TO_EVALUATE list.")
                    if config.USE_MLFLOW:
                        with mlflow.start_run(run_name="PPI_Production_Evaluation_Parent") as parent_run:
                            mlflow.set_tag("run_type", "ppi_production_eval")
                            ppi_evaluator.run(use_dummy_data=False, parent_run_id=parent_run.info.run_id)
                    else:
                        ppi_evaluator.run(use_dummy_data=False)
                else:
                    print("Skipping Main PPI Evaluation: LP_EMBEDDING_FILES_TO_EVALUATE is empty, although RUN_MAIN_PPI_EVALUATION is True.")
            elif not config.RUN_DUMMY_TEST: # Only print if dummy wasn't run and main eval was skipped due to empty list
                 print("Skipping Main PPI Evaluation: RUN_MAIN_PPI_EVALUATION is False or LP_EMBEDDING_FILES_TO_EVALUATE is empty.")

        else:
            print("\nSkipping all PPI evaluation (Dummy and Main) as per configuration.")

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
        if config.ENABLE_FILE_LOGGING:
            stop_logging()


if __name__ == '__main__':
    main()