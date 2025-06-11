# In src/main.py

import time

import mlflow  # Import mlflow

from src.benchmarks.gnn_benchmarker import GNNBenchmarker
# Import the configuration
from src.config import Config
# Import the refactored pipeline classes
from src.pipeline.data_builder import GraphBuilder
from src.pipeline.ppi_main import PPIPipeline
from src.pipeline.protgram_directgcn_embedder import ProtGramDirectGCNEmbedder  # Assuming this class name
from src.pipeline.transformer_embedder import TransformerEmbedder
from src.pipeline.word2vec_embedder import Word2VecEmbedder  # Assuming this class name and it has a .run() method


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

    # --- MLFLOW SETUP ---
    if config.USE_MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # 2. Run the GNN Benchmarking Pipeline (Optional)
    if config.RUN_BENCHMARKING_PIPELINE:
        if config.USE_MLFLOW:
            mlflow.set_experiment(config.MLFLOW_BENCHMARK_EXPERIMENT_NAME)
        benchmarker = GNNBenchmarker(config)
        benchmarker.run()

    # 3. Run the N-gram GCN Pipeline to generate embeddings (Optional)
    if config.RUN_GCN_PIPELINE:
        # Step 3a: Build the n-gram graphs
        graph_builder = GraphBuilder(config)
        graph_builder.run()
        # Step 3b: Train the GCN model to produce embeddings
        gcn_embedder = ProtGramDirectGCNEmbedder(config)  # Assuming this class and its .run() method
        gcn_embedder.run()

    # 4. Run the Word2Vec Embedding Pipeline (Optional)
    if config.RUN_WORD2VEC_PIPELINE:
        word2vec_embedder = Word2VecEmbedder(config)  # Assuming this class and its .run() method
        word2vec_embedder.run()

    # 5. Run the Transformer Embedding Pipeline (Optional)
    if config.RUN_TRANSFORMER_PIPELINE:
        transformer_embedder = TransformerEmbedder(config)
        transformer_embedder.run()

    # --- Set experiment for the main evaluation ---
    if config.USE_MLFLOW:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # 6. Run the Evaluation Pipeline
    # This step evaluates the embeddings created by the steps above.
    ppi_evaluator = PPIPipeline(config)

    if config.RUN_DUMMY_TEST:
        # Tag dummy runs in MLflow to easily filter them
        with mlflow.start_run(run_name="Dummy_Run_Parent") as parent_run:
            mlflow.set_tag("run_type", "dummy_test")
            ppi_evaluator.run(use_dummy_data=True, parent_run_id=parent_run.info.run_id)

    # Always run the main evaluation on the user-configured files
    print("\nNote: The main evaluation will now run on the files specified in your config's LP_EMBEDDING_FILES_TO_EVALUATE list.")
    with mlflow.start_run(run_name="Production_Run_Parent") as parent_run:
        mlflow.set_tag("run_type", "production_eval")
        ppi_evaluator.run(use_dummy_data=False, parent_run_id=parent_run.info.run_id)

    print("\n======================================================")
    print(f"### Full Orchestration Finished in {time.time() - script_start_time:.2f} seconds. ###")
    print("======================================================")


if __name__ == '__main__':
    main()
