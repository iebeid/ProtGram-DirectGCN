# ==============================================================================
# MODULE: main.py
# PURPOSE: Pipeline entry point
# VERSION: 1.2 (Uses config.RUN_MAIN_PPI_EVALUATION directly)
# AUTHOR: Islam Ebeid
# ==============================================================================

import time

import mlflow

from src.benchmarks.gnn_benchmarker import GNNBenchmarker
from config import Config
from src.pipeline.data_builder import GraphBuilder
from src.pipeline.ppi_main import PPIPipeline
from src.pipeline.protgram_directgcn_trainer import ProtGramDirectGCNTrainer
from src.pipeline.transformer_embedder import TransformerEmbedder
from src.pipeline.word2vec_embedder import Word2VecEmbedder
from src.utils.data_utils import DataUtils


def main():
    script_start_time = time.time()
    DataUtils.print_header("Starting Protein-Protein Interaction Pipeline")

    config = Config()
    if config.DEBUG_VERBOSE:
        print("--- Configuration Loaded ---")
        for key, value in config.__dict__.items():
            if not key.startswith("__"):
                print(f"  {key}: {value}")
        print("--------------------------")

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

    if config.RUN_GCN_PIPELINE:
        graph_builder = GraphBuilder(config)
        graph_builder.run()
        gcn_trainer = ProtGramDirectGCNTrainer(config)
        gcn_trainer.run()

    if config.RUN_WORD2VEC_PIPELINE:
        word2vec_embedder = Word2VecEmbedder(config)
        word2vec_embedder.run()

    if config.RUN_TRANSFORMER_PIPELINE:
        transformer_embedder = TransformerEmbedder(config)
        transformer_embedder.run()

    # Determine if any evaluation (dummy or main) should be set up
    should_setup_ppi_evaluator = config.RUN_DUMMY_TEST or config.RUN_MAIN_PPI_EVALUATION

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

    DataUtils.print_header(f"Full Orchestration Finished in {time.time() - script_start_time:.2f} seconds.")


if __name__ == '__main__':
    main()