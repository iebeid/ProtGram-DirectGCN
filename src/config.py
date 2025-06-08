# ==============================================================================
# MODULE: config.py
# PURPOSE: Centralized configuration for the entire PPI pipeline.
# ==============================================================================

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

class Config:
    def __init__(self):
        # --- 1. GENERAL & ORCHESTRATION SETTINGS ---
        self.RANDOM_STATE = 42
        self.DEBUG_VERBOSE = True

        # --- Workflow Control Flags ---
        # Master switches to run each part of the pipeline.
        self.RUN_GCN_PIPELINE = True
        self.RUN_WORD2VEC_PIPELINE = True
        self.RUN_TRANSFORMER_PIPELINE = True
        self.RUN_BENCHMARKING_PIPELINE = True
        self.RUN_DUMMY_TEST = True  # Run a quick test of the evaluation pipeline
        self.CLEANUP_DUMMY_DATA = True

        # --- PATH CONFIGURATION ---
        # Base directories
        self.BASE_DATA_DIR = "G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Data/"
        self.BASE_OUTPUT_DIR = os.path.join(self.BASE_DATA_DIR, "pipeline_output")

        # Input data paths
        self.GCN_INPUT_FASTA_PATH = os.path.join(self.BASE_DATA_DIR, "uniprot_sequences_sample.fasta")
        self.INTERACTIONS_POSITIVE_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/positive_interactions.csv')
        self.INTERACTIONS_NEGATIVE_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/negative_interactions.csv')

        # Output directories for each pipeline stage
        self.GRAPH_OBJECTS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "1_graph_objects")
        self.GCN_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "2_gcn_embeddings")
        self.WORD2VEC_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "2_word2vec_embeddings")
        self.TRANSFORMER_EMBEDDINGS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "2_transformer_embeddings")
        self.EVALUATION_RESULTS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "3_evaluation_results")
        self.BENCHMARKING_RESULTS_DIR = os.path.join(self.BASE_OUTPUT_DIR, "4_benchmarking_results")

        # --- 2. GCN PIPELINE PARAMETERS ---

        # --- Graph Building (pipeline/graph_builder.py) ---
        self.GCN_NGRAM_MAX_N = 3
        self.DASK_CHUNK_SIZE = 2000000
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, os.cpu_count() - 2)

        # --- Model Training (pipeline/2_gcn_trainer.py) ---
        # ID Mapping
        self.ID_MAPPING_MODE = 'regex'  # Options: 'none', 'regex', 'api'
        self.ID_MAPPING_OUTPUT_FILE = os.path.join(self.BASE_OUTPUT_DIR, "gcn_id_mapping.tsv")
        self.API_MAPPING_FROM_DB = "UniRef50"
        self.API_MAPPING_TO_DB = "UniProtKB"

        # GCN Model Hyperparameters
        self.GCN_1GRAM_INIT_DIM = 64
        self.GCN_HIDDEN_DIM_1 = 128
        self.GCN_HIDDEN_DIM_2 = 64
        self.GCN_EPOCHS_PER_LEVEL = 500
        self.GCN_LR = 0.001
        self.GCN_DROPOUT = 0.5
        self.GCN_WEIGHT_DECAY = 1e-4

        # Pooling & PCA
        self.POOLING_WORKERS: Optional[int] = max(1, os.cpu_count() - 8)
        self.APPLY_PCA_TO_GCN = True
        self.PCA_TARGET_DIMENSION = 64

        # --- 3. WORD2VEC PIPELINE PARAMETERS (pipeline/word2vec_embedder.py) ---
        self.W2V_INPUT_FASTA_DIR = os.path.join(self.BASE_DATA_DIR, self.GCN_INPUT_FASTA_PATH)  # Example path
        self.W2V_VECTOR_SIZE = 100
        self.W2V_WINDOW = 5
        self.W2V_MIN_COUNT = 1
        self.W2V_EPOCHS = 5
        self.W2V_WORKERS = max(1, os.cpu_count() - 2)
        self.W2V_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_W2V = True

        # --- 4. TRANSFORMER PIPELINE PARAMETERS (pipeline/transformer_embedder.py) ---
        self.TRANSFORMER_INPUT_FASTA_DIR = os.path.join(self.BASE_DATA_DIR, self.GCN_INPUT_FASTA_PATH)  # Example path
        self.TRANSFORMER_MODELS_TO_RUN = [{"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1}, ]
        self.TRANSFORMER_MAX_LENGTH = 1024
        self.TRANSFORMER_BASE_BATCH_SIZE = 16
        self.TRANSFORMER_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_TRANSFORMER = True

        # --- 5. EVALUATION PARAMETERS (pipeline/evaluator.py) ---
        self.PLOT_TRAINING_HISTORY = True  # Add this line
        self.EARLY_STOPPING_PATIENCE = 10  # Add this line

        self.PERFORM_H5_INTEGRITY_CHECK = True
        self.SAMPLE_NEGATIVE_PAIRS: Optional[int] = 500000
        self.TF_DATASET_STRATEGY = 'from_tensor_slices'  # Options: 'from_tensor_slices', 'from_generator'

        # --- ADD THIS VARIABLE ---
        # A list of dictionaries, where each dict defines an embedding file to be evaluated.
        # You must provide the 'name' for plotting and the 'path' to the H5 file.
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [{"name": "ProtT5-UniProt", "path": os.path.join(self.BASE_DATA_DIR, "models/per-protein.h5")},
            {"name": "ProtNgramGCN-n3-PCA64", "path": os.path.join(self.GCN_EMBEDDINGS_DIR, "gcn_n3_embeddings_pca64.h5")},
            {"name": "ProtBERT-Mean-PCA64", "path": os.path.join(self.TRANSFORMER_EMBEDDINGS_DIR, "ProtBERT_mean_pca64.h5")},
            {"name": "Word2Vec-Mean-PCA64", "path": os.path.join(self.WORD2VEC_EMBEDDINGS_DIR, "word2vec_dim100_mean_pca64.h5")}# Add any other embedding files you want to compare here.
        ]
        # --- END OF ADDITION ---

        # In src/config.py

        # --- 6. MLFLOW & EXPERIMENT TRACKING ---
        self.USE_MLFLOW = True  # Master switch to enable/disable MLflow

        # --- THIS IS THE CORRECTED PART ---
        # Use pathlib to create a robust, system-agnostic file URI
        mlruns_path = Path(self.BASE_OUTPUT_DIR) / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.resolve().as_uri()
        # --- END OF CORRECTION ---

        # Experiment name for the main link prediction evaluation
        self.MLFLOW_EXPERIMENT_NAME = "PPI-Link-Prediction"
        # Experiment name for the GNN benchmarking
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = "GNN-Benchmarking"

        # In src/config.py, inside the Config class __init__ method

        # Evaluation MLP Architecture & Training
        self.EVAL_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.EVAL_N_FOLDS = 5
        self.EVAL_MLP_DENSE1_UNITS = 128
        self.EVAL_MLP_DROPOUT1_RATE = 0.4
        self.EVAL_MLP_DENSE2_UNITS = 64
        self.EVAL_MLP_DROPOUT2_RATE = 0.4
        self.EVAL_MLP_L2_REG = 0.001
        self.EVAL_BATCH_SIZE = 64
        self.EVAL_EPOCHS = 10
        self.EVAL_LEARNING_RATE = 1e-3

        # Evaluation Reporting
        self.EVAL_K_VALUES_FOR_TABLE = [50, 100]
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtT5-Precomputed"  # A baseline to compare against
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05
