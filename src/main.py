# ==============================================================================
# SCRIPT: main_orchestrator.py
# PURPOSE: The single entry point for running all workflows in the PPI pipeline,
#          including GCN training, other embedding generation, and evaluation.
# ==============================================================================

import time

# Import the configuration and the main run function from each module
from config import Config
from utils.diagnostics import check_gpu_environment
from pipeline.build_graphs import run_graph_building
from pipeline.gcn_trainer import run_gcn_training
from pipeline.word2vec_embedder import run_word2vec_training
from pipeline.transformer_embedder import run_transformer_embedding_generation
from pipeline.evaluator import run_evaluation
from benchmarking.gnn_evaluator import run_gnn_benchmarking


def main():
    """
    The main orchestration function that runs the selected pipelines.
    """
    script_start_time = time.time()
    print("======================================================")
    print("### Starting Protein-Protein Interaction Pipeline ###")
    print("======================================================")

    # 1. Load configuration and check environment
    config = Config()
    check_gpu_environment()

    # 2. Run the GNN Benchmarking Pipeline (Optional)
    if config.RUN_BENCHMARKING_PIPELINE:
        run_gnn_benchmarking(config)

    # 3. Run the N-gram GCN Pipeline (Optional)
    if config.RUN_GCN_PIPELINE:
        run_graph_building(config)
        run_gcn_training(config)

    # 4. Run the Word2Vec Embedding Pipeline (Optional)
    if config.RUN_WORD2VEC_PIPELINE:
        run_word2vec_training(config)

    # 5. Run the Transformer Embedding Pipeline (Optional)
    if config.RUN_TRANSFORMER_PIPELINE:
        run_transformer_embedding_generation(config)

    # 6. Run the Evaluation Pipeline
    # This step evaluates the embeddings created by the steps above.
    # It includes its own dummy test case controlled from the config.
    if config.RUN_DUMMY_TEST:
        run_evaluation(config, use_dummy_data=True)

    # Always run the main evaluation on the configured files
    run_evaluation(config, use_dummy_data=False)

    print("\n======================================================")
    print(f"### Full Orchestration Finished in {time.time() - script_start_time:.2f} seconds. ###")
    print("======================================================")


if __name__ == '__main__':
    main()