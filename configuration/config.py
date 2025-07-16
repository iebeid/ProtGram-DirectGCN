# ==============================================================================
# MODULE: config.py
# PURPOSE: Centralized configuration for the entire PPI training.
# VERSION: 1.13 (Automated cluster count based on target nodes per cluster and automated data download)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from typing import List, Optional, Dict, Any
from pathlib import Path


class Config:
    def __init__(self):
        # --- GENERAL & ORCHESTRATION SETTINGS ---
        self.RANDOM_STATE = 42
        self.DEBUG_VERBOSE = True

        # --- Main Pipeline Control Flags ---
        self.RUN_GCN_PIPELINE = True
        #self.RUN_WORD2VEC = True
        #self.RUN_BIOBERT = True
        #self.RUN_PROTT5 = True
        #self.RUN_HMM = True
        #self.RUN_RNN = True
        #self.RUN_LSTM = True
        self.RUN_WORD2VEC_PIPELINE = False
        self.RUN_TRANSFORMER_PIPELINE = False

        #self.RUN_GNN_BENCHMARKING_PIPELINE = True
        #self.RUN_LLM_BENCHMARKING_PIPELINE = True
        #self.RUN_ML_BENCHMARKING_PIPELINE = True
        self.RUN_BENCHMARKING_PIPELINE = True

        self.RUN_MAIN_PPI_EVALUATION = True
        #self.RUN_GNN_PPI_PIPELINE = True
        #self.RUN_LLM_PPI_PIPELINE = True
        #self.RUN_ML_PPI_PIPELINE = True
        self.RUN_DUMMY_TEST = True
        self.CLEANUP_DUMMY_DATA = True

        # --- PATH CONFIGURATION ---
        # Set the project root to be the parent directory of this file's location (the 'configuration' directory).
        # This is more robust than using a relative path like ".." which depends on the current working directory.
        self.PROJECT_ROOT = (Path(__file__).parent.parent).resolve()
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data"
        self.BASE_OUTPUT_DIR = self.PROJECT_ROOT / "results"
        self.BASE_SRC_DIR = self.PROJECT_ROOT / "source"
        self.BASE_CONFIG_DIR = self.PROJECT_ROOT / "configuration"

        # --- NEW: Logging Configuration ---
        self.ENABLE_FILE_LOGGING = True
        self.LOG_DIR = self.BASE_OUTPUT_DIR / "logs"

        # --- Define file keys and their relative paths to BASE_DATA_DIR ---
        # This provides a single source of truth for file locations.
        self.FILE_KEYS = {
            "UNIPROT_FASTA": "sequences/uniprot_sprot.fasta",
            "POS_INTERACTIONS": "ground_truth/positive_interactions.csv",
            "NEG_INTERACTIONS": "ground_truth/negative_interactions.csv",
            "PROTT5_MODEL": "models/prott5.h5",
            "ID_MAPPING_TSV": "mappings/uniref_to_uniprot.tsv",
            # "WORD2VEC_MODEL": "models/word2vec.h5", # Example for another model
        }

        # --- Dynamically create path attributes from keys for use in the pipeline ---
        self.GCN_INPUT_FASTA_PATH = self.BASE_DATA_DIR / self.FILE_KEYS["UNIPROT_FASTA"]
        self.INTERACTIONS_POSITIVE_PATH = self.BASE_DATA_DIR / self.FILE_KEYS["POS_INTERACTIONS"]
        self.INTERACTIONS_NEGATIVE_PATH = self.BASE_DATA_DIR / self.FILE_KEYS["NEG_INTERACTIONS"]

        self.GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "1_graph_objects"
        self.GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_gcn_embeddings"
        self.WORD2VEC_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_word2vec_embeddings"
        self.TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_transformer_embeddings"
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

        # --- Direct GCN PIPELINE PARAMETERS ---
        self.GCN_NGRAM_MAX_N = 3
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1

        self.GCN_HIDDEN_LAYER_DIMS = [256, 128, 64]
        self.ID_MAPPING_MODE = 'regex'
        # This path is used by the DataLoader to process ID mappings.
        self.ID_MAPPING_OUTPUT_FILE = self.BASE_DATA_DIR / self.FILE_KEYS["ID_MAPPING_TSV"]
        self.API_MAPPING_FROM_DB = "UniRef50"
        self.API_MAPPING_TO_DB = "UniProtKB"

        self.GCN_1GRAM_INIT_DIM = 512
        self.GCN_EPOCHS_PER_LEVEL = 300  # Can be higher now that training is faster
        self.GCN_LR = 0.001
        self.GCN_DROPOUT_RATE = 0.5
        self.GCN_WEIGHT_DECAY = 1e-4
        self.GCN_L2_REG_LAMBDA = 1e-7  # Keep L2 low

        # --- NEW: LR Scheduler and Early Stopping for GCN Training ---
        self.GCN_USE_LR_SCHEDULER = True
        self.GCN_LR_SCHEDULER_PATIENCE = 10  # Epochs to wait for improvement before reducing LR
        self.GCN_LR_SCHEDULER_FACTOR = 0.5  # Factor by which the learning rate will be reduced

        self.GCN_USE_EARLY_STOPPING = True
        self.GCN_EARLY_STOPPING_PATIENCE = 25  # Epochs to wait for improvement before stopping
        self.GCN_EARLY_STOPPING_MIN_DELTA = 1e-5  # Minimum change in the monitored quantity to qualify as an improvement

        self.GCN_PROPAGATION_EPSILON = 1e-9
        self.GCN_MAX_PE_LEN = 512
        self.GCN_USE_VECTOR_COEFFS = True

        self.GCN_TASK_TYPES_PER_LEVEL: Dict[int, str] = {
            1: "community",
            2: "next_node",
            3: "next_node",
        }
        self.GCN_DEFAULT_TASK_TYPE: str = "community"
        self.GCN_CLOSEST_AA_K_HOPS: int = 3

        # --- NEW: Cluster-GCN Training Strategy ---
        self.GCN_USE_CLUSTER_TRAINING = True
        self.GCN_CLUSTER_TRAINING_THRESHOLD_NODES = 10000  # Apply clustering for graphs with > 10k nodes

        # New parameters for automatic cluster count
        self.GCN_TARGET_NODES_PER_CLUSTER = 2000  # Aim for 500 nodes per cluster
        self.GCN_MIN_CLUSTERS = 2  # Ensure at least 2 clusters if clustering is enabled
        self.GCN_MAX_CLUSTERS = 500  # Cap the number of clusters to avoid excessive fragmentation

        self.POOLING_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1
        self.APPLY_PCA_TO_GCN = True
        self.PCA_TARGET_DIMENSION = 64

        # --- NEW: Sanity Check PPI Task ---
        self.GCN_RUN_SANITY_CHECK_PPI = False
        self.GCN_SANITY_CHECK_EPOCHS = 5  # Reduced for a faster check
        self.GCN_SANITY_CHECK_TEST_SPLIT = 0.2
        self.GCN_SANITY_CHECK_SAMPLE_SIZE = 2000  # NEW: Use only N positive pairs for a quick check.

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

        # --- 5. EVALUATION PARAMETERS (for ppi_experimenter.py and GNN Benchmarker) ---
        self.PLOT_TRAINING_HISTORY = True
        self.EARLY_STOPPING_PATIENCE = 10
        self.PERFORM_H5_INTEGRITY_CHECK = True
        self.SAMPLE_NEGATIVE_PAIRS: Optional[int] = 100000
        self.TF_DATASET_STRATEGY = 'from_tensor_slices'

        # Ensure these paths are correct for your generated embeddings
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": "ProtT5", "path": self.BASE_DATA_DIR / self.FILE_KEYS["PROTT5_MODEL"]},
            # {"name": "ProtGramDirectGCN-UniProt-PCA64-n3-old", "path": self.PPI_EVALUATION_MODELS_DIR / f"protgram_directgcn_old.h5"},
            {"name": "ProtGramDirectGCN", "path": self.PPI_EVALUATION_MODELS_DIR / "protgram_directgcn.h5"},
            # {"name": "ProtGramDirectGCN-UniProt-PCA64-n3-new", "path": self.PPI_EVALUATION_MODELS_DIR / f"protgram_directgcn_n3_new.h5"},
            # {"name": "Word2Vec", "path": self.PPI_EVALUATION_MODELS_DIR / "word2vec.h5"}
        ]

        # --- 6. MLFLOW & EXPERIMENT TRACKING ---
        self.USE_MLFLOW = False
        mlruns_path = self.BASE_OUTPUT_DIR / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.resolve().as_uri()
        self.MLFLOW_EXPERIMENT_NAME = "PPI-Link-Prediction"
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = "GNN-Benchmarking"

        # Evaluation MLP Architecture & Training
        self.EVAL_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.EVAL_N_FOLDS = 3
        self.EVAL_MLP_DENSE1_UNITS = 128
        self.EVAL_MLP_DROPOUT1_RATE = 0.4
        self.EVAL_MLP_DENSE2_UNITS = 64
        self.EVAL_MLP_DROPOUT2_RATE = 0.4
        self.EVAL_MLP_L2_REG = 1e-5
        self.EVAL_BATCH_SIZE = 2048
        self.EVAL_EPOCHS = 20
        self.EVAL_LEARNING_RATE = 0.001

        # Evaluation Reporting
        self.EVAL_K_VALUES_FOR_TABLE = [50, 100]
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtGramDirectGCN"
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05

        # --- 7. DATA SOURCES FOR AUTOMATIC DOWNLOAD ---
        # The key must match a key in self.FILE_KEYS.
        # The data manager script will use this to verify and download files.
        # 'checksum' is optional (e.g., "sha256:your_hash_here").
        # 'post_process' can be 'ungzip' for .gz files.
        self.DATA_SOURCES = {
            "UNIPROT_FASTA": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
                "post_process": "ungzip",
                "checksum": None  # Optional: Add checksum of the final (uncompressed) file if known
            },
            "POS_INTERACTIONS": {
                "url": "https://example.com/data/positive_interactions.csv",  # Replace with a real URL
                "post_process": None,
                "checksum": None
            },
            "NEG_INTERACTIONS": {
                "url": "https://example.com/data/negative_interactions.csv",  # Replace with a real URL
                "post_process": None,
                "checksum": None
            },
            "PROTT5_MODEL": {
                "url": "https://example.com/models/prott5.h5",  # Replace with a real URL
                "post_process": None,
                "checksum": None
            },
            "ID_MAPPING_TSV": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/v4/idmapping_selected.tab.gz",
                "post_process": "ungzip",
                "checksum": None
            }
        }