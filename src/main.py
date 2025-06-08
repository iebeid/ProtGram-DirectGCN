# In src/main.py

import time
import os
import mlflow  # Import mlflow

# Import the configuration and the main run function from each module
from src.config import Config
from src.utils.checker import check_gpu_environment
from src.pipeline.graph_builder import run_graph_building
from src.pipeline.prot_ngram_gcn_trainer import run_gcn_training
from src.pipeline.word2vec_embedder import run_word2vec_training
from src.pipeline.transformer_embedder import run_transformer_embedding_generation
from src.pipeline.evaluator import run_evaluation
from src.benchmarks.gnn_benchmarker import run_gnn_benchmarking


def main():
    """
    The main orchestration function that runs the selected pipelines based on config flags.
    """
    script_start_time = time.time()
    print("======================================================")
    print("### Starting Protein-Protein Interaction Pipeline ###")
    print("======================================================")

    # 1. Load configuration and check environment
    config = Config()
    check_gpu_environment()

    # --- MLFLOW SETUP ---
    if config.USE_MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # 2. Run the GNN Benchmarking Pipeline (Optional)
    if config.RUN_BENCHMARKING_PIPELINE:
        if config.USE_MLFLOW:
            mlflow.set_experiment(config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
        run_gnn_benchmarking(config)

    # 3. Run the N-gram GCN Pipeline to generate embeddings (Optional)
    if config.RUN_GCN_PIPELINE:
        # Step 3a: Build the n-gram graphs
        run_graph_building(config)
        # Step 3b: Train the GCN model to produce embeddings
        run_gcn_training(config)

    # 4. Run the Word2Vec Embedding Pipeline (Optional)
    if config.RUN_WORD2VEC_PIPELINE:
        run_word2vec_training(config)

    # 5. Run the Transformer Embedding Pipeline (Optional)
    if config.RUN_TRANSFORMER_PIPELINE:
        run_transformer_embedding_generation(config)

    # --- Set experiment for the main evaluation ---
    if config.USE_MLFLOW:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # 6. Run the Evaluation Pipeline
    # This step evaluates the embeddings created by the steps above.
    if config.RUN_DUMMY_TEST:
        # Tag dummy runs in MLflow to easily filter them
        with mlflow.start_run(run_name="Dummy_Run_Parent") as parent_run:
            mlflow.set_tag("run_type", "dummy_test")
            run_evaluation(config, use_dummy_data=True, parent_run_id=parent_run.info.run_id)

    # Always run the main evaluation on the user-configured files
    print("\nNote: The main evaluation will now run on the files specified in your config's LP_EMBEDDING_FILES_TO_EVALUATE list.")
    with mlflow.start_run(run_name="Production_Run_Parent") as parent_run:
        mlflow.set_tag("run_type", "production_eval")
        run_evaluation(config, use_dummy_data=False, parent_run_id=parent_run.info.run_id)

    print("\n======================================================")
    print(f"### Full Orchestration Finished in {time.time() - script_start_time:.2f} seconds. ###")
    print("======================================================")


if __name__ == '__main__':
    main()
