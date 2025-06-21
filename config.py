# ==============================================================================
# MODULE: config.py
# PURPOSE: Centralized configuration for the entire PPI pipeline.
# VERSION: 1.13 (Automated cluster count based on target nodes per cluster)
# AUTHOR: Islam Ebeid
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
        self.RUN_GCN_PIPELINE = False
        self.RUN_WORD2VEC_PIPELINE = False
        self.RUN_TRANSFORMER_PIPELINE = False
        self.RUN_BENCHMARKING_PIPELINE = False
        self.RUN_MAIN_PPI_EVALUATION = True
        self.RUN_DUMMY_TEST = True
        self.CLEANUP_DUMMY_DATA = True

        # --- PATH CONFIGURATION ---
        self.PROJECT_ROOT = Path(".").resolve()
        print("PROJECT ROOT IS: " + str(self.PROJECT_ROOT))
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data"
        self.BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_OUTPUT_DIR = self.BASE_DATA_DIR / "results"

        self.GCN_INPUT_FASTA_PATH = self.BASE_DATA_DIR / "sequences/uniprot_sprot.fasta"
        self.INTERACTIONS_POSITIVE_PATH = self.BASE_DATA_DIR / 'ground_truth/positive_interactions.csv'
        self.INTERACTIONS_NEGATIVE_PATH = self.BASE_DATA_DIR / 'ground_truth/negative_interactions.csv'

        self.GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "1_graph_objects"
        self.GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_gcn_embeddings"
        self.WORD2VEC_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_word2vec_embeddings"
        self.TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_transformer_embeddings"
        self.EVALUATION_RESULTS_DIR = self.BASE_OUTPUT_DIR / "3_evaluation_results"
        self.BENCHMARKING_RESULTS_DIR = self.BASE_OUTPUT_DIR / "4_benchmarking_results"
        self.BENCHMARK_EMBEDDINGS_DIR = self.BENCHMARKING_RESULTS_DIR / "embeddings"
        self.PPI_EVALUATION_MODELS_DIR = self.BASE_DATA_DIR / "models"

        # --- GNN BENCHMARKING PARAMETERS ---
        self.BENCHMARK_NODE_CLASSIFICATION_DATASETS = [
            "KarateClub", "Cora", "CiteSeer", "PubMed",
            "Cornell", "Texas", "Wisconsin"
        ]
        self.BENCHMARK_SAVE_EMBEDDINGS = True
        self.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = True
        self.BENCHMARK_PCA_TARGET_DIM = 64
        self.BENCHMARK_TEST_ON_UNDIRECTED = True
        self.BENCHMARK_SPLIT_RATIOS: Dict[str, float] = {"train": 0.1, "val": 0.1, "test": 0.8}

        # --- 2. GCN PIPELINE PARAMETERS (Your custom GCN) ---
        self.GCN_NGRAM_MAX_N = 4
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1

        self.GCN_HIDDEN_LAYER_DIMS = [256, 128, 64]
        self.ID_MAPPING_MODE = 'regex'
        self.ID_MAPPING_OUTPUT_FILE = self.BASE_OUTPUT_DIR / "mappings/gcn_id_mapping.tsv"
        self.API_MAPPING_FROM_DB = "UniRef50"
        self.API_MAPPING_TO_DB = "UniProtKB"

        self.GCN_1GRAM_INIT_DIM = 512
        self.GCN_EPOCHS_PER_LEVEL = 500  # Can be higher now that training is faster
        self.GCN_LR = 0.001
        self.GCN_DROPOUT_RATE = 0.5
        self.GCN_WEIGHT_DECAY = 1e-4
        self.GCN_L2_REG_LAMBDA = 1e-7  # Keep L2 low

        self.GCN_PROPAGATION_EPSILON = 1e-9
        self.GCN_MAX_PE_LEN = 512
        self.GCN_USE_VECTOR_COEFFS = True

        self.GCN_TASK_TYPES_PER_LEVEL: Dict[int, str] = {
            1: "next_node",
            2: "next_node",
            3: "closest_aa",
            4: "community",
        }
        self.GCN_DEFAULT_TASK_TYPE: str = "community"
        self.GCN_CLOSEST_AA_K_HOPS: int = 3

        # --- NEW: Cluster-GCN Training Strategy ---
        self.GCN_USE_CLUSTER_TRAINING = True
        self.GCN_CLUSTER_TRAINING_THRESHOLD_NODES = 10000  # Apply clustering for graphs with > 10k nodes

        # New parameters for automatic cluster count
        self.GCN_TARGET_NODES_PER_CLUSTER = 500  # Aim for 500 nodes per cluster
        self.GCN_MIN_CLUSTERS = 2  # Ensure at least 2 clusters if clustering is enabled
        self.GCN_MAX_CLUSTERS = 500  # Cap the number of clusters to avoid excessive fragmentation

        self.POOLING_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1
        self.APPLY_PCA_TO_GCN = True
        self.PCA_TARGET_DIMENSION = 64

        # --- 3. WORD2VEC PIPELINE PARAMETERS ---
        self.W2V_INPUT_FASTA_DIR = self.GCN_INPUT_FASTA_PATH
        self.W2V_VECTOR_SIZE = 100
        self.W2V_WINDOW = 5
        self.W2V_MIN_COUNT = 1
        self.W2V_EPOCHS = 5
        self.W2V_WORKERS = 1
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
        self.SAMPLE_NEGATIVE_PAIRS: Optional[int] = 100000
        self.TF_DATASET_STRATEGY = 'from_tensor_slices'

        # Ensure these paths are correct for your generated embeddings
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": "ProtT5-UniProt-PCA64", "path": self.PPI_EVALUATION_MODELS_DIR / "prott5.h5"},
            {"name": "ProtGramDirectGCN-UniProt-PCA64-Old", "path": self.PPI_EVALUATION_MODELS_DIR / f"protgram_directgcn_1.h5"},
            {"name": "ProtGramDirectGCN-UniProt-PCA64-New", "path": self.PPI_EVALUATION_MODELS_DIR / f"protgram_directgcn_2.h5"},
            {"name": "Word2Vec-UniProt-PCA64", "path": self.PPI_EVALUATION_MODELS_DIR / f"word2vec.h5"}
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
        self.EVAL_MLP_L2_REG = 1e-5
        self.EVAL_BATCH_SIZE = 1024
        self.EVAL_EPOCHS = 5
        self.EVAL_LEARNING_RATE = 0.001

        # Evaluation Reporting
        self.EVAL_K_VALUES_FOR_TABLE = [50, 100]
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtGramDirectGCN-UniProt-PCA64-New"
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05