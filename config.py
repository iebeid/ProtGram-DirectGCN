# ==============================================================================
# MODULE: config.py
# PURPOSE: Centralized configuration for the entire PPI pipeline.
# VERSION: 1.4 (Added GNN benchmark embedding/PCA/undirected flags, split ratios)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


class Config:
    def __init__(self):
        # --- 1. GENERAL & ORCHESTRATION SETTINGS ---
        self.RANDOM_STATE = 42
        self.DEBUG_VERBOSE = True # Set to False for less verbose training logs

        # --- Workflow Control Flags ---
        self.RUN_GCN_PIPELINE = True
        self.RUN_WORD2VEC_PIPELINE = False
        self.RUN_TRANSFORMER_PIPELINE = False
        self.RUN_BENCHMARKING_PIPELINE = False
        self.RUN_MAIN_PPI_EVALUATION = False
        self.RUN_DUMMY_TEST = True
        self.CLEANUP_DUMMY_DATA = True

        # --- PATH CONFIGURATION ---
        # self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        self.PROJECT_ROOT = Path(".").resolve()
        print("PROJECT ROOT IS: " + str(self.PROJECT_ROOT))
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data"
        self.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_OUTPUT_DIR = self.BASE_DATA_DIR / "results"

        self.GCN_INPUT_FASTA_PATH = self.BASE_DATA_DIR / "sequences/uniprot_sequences_sample.fasta"
        self.INTERACTIONS_POSITIVE_PATH = self.BASE_DATA_DIR / 'ground_truth/positive_interactions.csv'
        self.INTERACTIONS_NEGATIVE_PATH = self.BASE_DATA_DIR / 'ground_truth/negative_interactions.csv'

        self.GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "1_graph_objects"
        self.GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_gcn_embeddings"
        self.WORD2VEC_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_word2vec_embeddings"
        self.TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_transformer_embeddings"
        self.EVALUATION_RESULTS_DIR = self.BASE_OUTPUT_DIR / "3_evaluation_results"
        self.BENCHMARKING_RESULTS_DIR = self.BASE_OUTPUT_DIR / "4_benchmarking_results"
        self.BENCHMARK_EMBEDDINGS_DIR = self.BENCHMARKING_RESULTS_DIR / "embeddings" # For GNN benchmark embeddings

        # --- GNN BENCHMARKING PARAMETERS ---
        self.BENCHMARK_NODE_CLASSIFICATION_DATASETS = [
            "KarateClub", "Cora", "CiteSeer", "PubMed",
            "Cornell", "Texas", "Wisconsin", "PPI",
        ]
        self.BENCHMARK_SAVE_EMBEDDINGS = True # Save embeddings from GNN benchmark models
        self.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = True # Apply PCA to saved benchmark embeddings
        self.BENCHMARK_PCA_TARGET_DIM = 64 # Target dim for PCA on benchmark embeddings
        self.BENCHMARK_TEST_ON_UNDIRECTED = True # Also test on undirected versions of graphs
        # Ratios for RandomNodeSplit if dataset has no predefined masks. Sum should be <= 1.0
        self.BENCHMARK_SPLIT_RATIOS: Dict[str, float] = {"train": 0.1, "val": 0.1, "test": 0.8} # Example for small datasets like Karate


        # --- 2. GCN PIPELINE PARAMETERS (Your custom GCN) ---
        self.GCN_NGRAM_MAX_N = 5
        self.DASK_CHUNK_SIZE = 2000000
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, os.cpu_count() - 8) if os.cpu_count() else 1
        self.GCN_HIDDEN_LAYER_DIMS = [128, 128, 128, 128, 64]

        self.ID_MAPPING_MODE = 'regex'
        self.ID_MAPPING_OUTPUT_FILE = self.BASE_OUTPUT_DIR / "mappings/gcn_id_mapping.tsv"
        self.API_MAPPING_FROM_DB = "UniRef50"
        self.API_MAPPING_TO_DB = "UniProtKB"

        self.GCN_1GRAM_INIT_DIM = 512
        self.GCN_EPOCHS_PER_LEVEL = 10
        self.GCN_LR = 0.001
        self.GCN_DROPOUT_RATE = 0.5
        self.GCN_WEIGHT_DECAY = 1e-4
        self.GCN_L2_REG_LAMBDA = 0.0001
        self.GCN_PROPAGATION_EPSILON = 1e-9
        self.GCN_MAX_PE_LEN = 512
        self.GCN_USE_VECTOR_COEFFS = True

        self.GCN_TASK_TYPES_PER_LEVEL: Dict[int, str] = {
            1: "next_node", 2: "next_node", 3: "next_node", 4: "closest_aa", 5: "community"
        }
        self.GCN_DEFAULT_TASK_TYPE: str = "community"
        self.GCN_CLOSEST_AA_K_HOPS: int = 3

        self.POOLING_WORKERS: Optional[int] = max(1, os.cpu_count() - 8) if os.cpu_count() else 1
        self.APPLY_PCA_TO_GCN = True # PCA for your main GCN pipeline embeddings
        self.PCA_TARGET_DIMENSION = 64 # PCA target for your main GCN pipeline embeddings

        # --- 3. WORD2VEC PIPELINE PARAMETERS ---
        self.W2V_INPUT_FASTA_DIR = Path(self.GCN_INPUT_FASTA_PATH).parent
        self.W2V_VECTOR_SIZE = 100
        self.W2V_WINDOW = 5
        self.W2V_MIN_COUNT = 1
        self.W2V_EPOCHS = 5
        self.W2V_WORKERS = max(1, os.cpu_count() - 2) if os.cpu_count() else 1
        self.W2V_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_W2V = True

        # --- 4. TRANSFORMER PIPELINE PARAMETERS ---
        self.TRANSFORMER_INPUT_FASTA_DIR = Path(self.GCN_INPUT_FASTA_PATH).parent
        self.TRANSFORMER_MODELS_TO_RUN = [
            {"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1}
        ]
        self.TRANSFORMER_MAX_LENGTH = 1024
        self.TRANSFORMER_BASE_BATCH_SIZE = 16
        self.TRANSFORMER_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_TRANSFORMER = True

        # --- 5. EVALUATION PARAMETERS (for ppi_main.py and GNN Benchmarker) ---
        self.PLOT_TRAINING_HISTORY = True
        self.EARLY_STOPPING_PATIENCE = 10
        self.PERFORM_H5_INTEGRITY_CHECK = True
        self.SAMPLE_NEGATIVE_PAIRS: Optional[int] = 500000
        self.TF_DATASET_STRATEGY = 'from_tensor_slices'
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": "ProtT5-UniProt", "path": self.BASE_DATA_DIR / "models/per-protein.h5"},
            # {"name": "ProtNgramGCN-n3-PCA64", "path": self.GCN_EMBEDDINGS_DIR / "gcn_n3_embeddings_pca64.h5"},
            # {"name": "ProtBERT-Mean-PCA64", "path": self.TRANSFORMER_EMBEDDINGS_DIR / "ProtBERT_mean_pca64.h5"},
            # {"name": "Word2Vec-Mean-PCA64", "path": self.WORD2VEC_EMBEDDINGS_DIR / "word2vec_dim100_mean_pca64.h5"}
        ]

        # --- 6. MLFLOW & EXPERIMENT TRACKING ---
        self.USE_MLFLOW = True
        mlruns_path = self.BASE_OUTPUT_DIR / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.resolve().as_uri()
        self.MLFLOW_EXPERIMENT_NAME = "PPI-Link-Prediction"
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = "GNN-Benchmarking"

        # Evaluation MLP Architecture & Training
        self.EVAL_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.EVAL_N_FOLDS = 5
        self.EVAL_MLP_DENSE1_UNITS = 128
        self.EVAL_MLP_DROPOUT1_RATE = 0.4
        self.EVAL_MLP_DENSE2_UNITS = 64
        self.EVAL_MLP_DROPOUT2_RATE = 0.4
        self.EVAL_MLP_L2_REG = 0.001
        self.EVAL_BATCH_SIZE = 64
        self.EVAL_EPOCHS = 5
        self.EVAL_LEARNING_RATE = 0.01

        # Evaluation Reporting
        self.EVAL_K_VALUES_FOR_TABLE = [50, 100]
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtT5-UniProt"
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05