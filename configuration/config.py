# ==============================================================================
# MODULE: config.py
# PURPOSE: Centralized configuration for the entire PPI trainers.
# VERSION: 1.13 (Automated cluster count based on target nodes per cluster and automated data_builders download)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from typing import Optional, Dict
from pathlib import Path


class Config:
    def __init__(self):
        # --- 1. GENERAL SETTINGS ---
        self.RANDOM_STATE = 42
        self.DEBUG_VERBOSE = True

        # --- 2. PATHS & DIRECTORIES ---
        self._setup_paths()

        # --- 3. PIPELINE CONTROL FLAGS ---
        self._setup_pipeline_flags()

        # --- 4. DATA SOURCES FOR AUTOMATIC DOWNLOAD ---
        self._setup_data_sources()

        # --- 5. GNN BENCHMARKING PARAMETERS ---
        self._setup_benchmarking_params()

        # --- 6. ProtGram-DirectGCN PIPELINE PARAMETERS ---
        self._setup_gcn_params()

        # --- 7. WORD2VEC PIPELINE PARAMETERS ---
        self._setup_word2vec_params()

        # --- 8. TRANSFORMER PIPELINE PARAMETERS ---
        self._setup_transformer_params()

        # --- 9. PPI EVALUATION PARAMETERS ---
        self._setup_evaluation_params()

        # --- 10. MLFLOW & EXPERIMENT TRACKING ---
        self._setup_mlflow_params()

    def _setup_paths(self):
        """Sets up all base, data_builders, and results paths for the project."""
        # Base Paths
        self.PROJECT_ROOT = (Path(__file__).parent.parent).resolve()
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data_builders"
        self.BASE_OUTPUT_DIR = self.PROJECT_ROOT / "results"
        self.LOG_DIR = self.BASE_OUTPUT_DIR / "logs"

        # Data Subdirectories
        self.DATA_SEQUENCES_DIR = self.BASE_DATA_DIR / "sequences"
        self.DATA_GROUND_TRUTH_DIR = self.BASE_DATA_DIR / "ground_truth"
        self.DATA_MODELS_DIR = self.BASE_DATA_DIR / "models"
        self.DATA_MAPPINGS_DIR = self.BASE_DATA_DIR / "mappings"
        self.DATA_STANDARD_DATASETS_DIR = self.BASE_DATA_DIR / "standard_datasets_pyg"

        # Results Subdirectories
        self.RESULTS_GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "1_graph_objects"
        self.RESULTS_GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_gcn_embeddings"
        self.RESULTS_W2V_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_word2vec_embeddings"
        self.RESULTS_TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "2_transformer_embeddings"
        self.RESULTS_EVALUATION_DIR = self.BASE_OUTPUT_DIR / "3_evaluation_results"
        self.RESULTS_BENCHMARKING_DIR = self.BASE_OUTPUT_DIR / "4_benchmarking_results"
        self.RESULTS_BENCHMARK_EMBEDDINGS_DIR = self.RESULTS_BENCHMARKING_DIR / "embeddings"

        # Key File Paths
        self.UNIPROT_FASTA_PATH = self.DATA_SEQUENCES_DIR / "uniprot_sprot.fasta"
        self.POS_INTERACTIONS_PATH = self.DATA_GROUND_TRUTH_DIR / "positive_interactions.csv"
        self.NEG_INTERACTIONS_PATH = self.DATA_GROUND_TRUTH_DIR / "negative_interactions.csv"
        self.ID_MAPPING_PATH = self.DATA_MAPPINGS_DIR / "uniref_to_uniprot.tsv"
        self.PROTT5_MODEL_PATH = self.DATA_MODELS_DIR / "prott5.h5"

    def _setup_pipeline_flags(self):
        """Sets flags to control which parts of the main pipeline are executed."""
        self.RUN_GCN_PIPELINE = True
        self.RUN_WORD2VEC_PIPELINE = False
        self.RUN_TRANSFORMER_PIPELINE = False
        self.RUN_BENCHMARKING_PIPELINE = True
        self.RUN_MAIN_PPI_EVALUATION = True
        self.RUN_INTEGRATED_TESTS = True  # Runs all unit, smoke, and verification testers from unit_tests.py
        self.RUN_DUMMY_TEST = True  # Runs a quick evaluation on dummy data_builders
        self.CLEANUP_DUMMY_DATA = True
        self.ENABLE_FILE_LOGGING = True

    def _setup_data_sources(self):
        """
        Defines the data_builders sources for automatic download.
        The key is a unique identifier, and 'path' is the final destination.
        """
        self.DATA_SOURCES = {
            "UNIPROT_FASTA": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
                "path": self.UNIPROT_FASTA_PATH,
                "post_process": "ungzip",
                "checksum": None
            },
            "POS_INTERACTIONS": {
                "url": "https://example.com/data/positive_interactions.csv",
                "path": self.POS_INTERACTIONS_PATH,
                "post_process": None,
                "checksum": None
            },
            "NEG_INTERACTIONS": {
                "url": "https://example.com/data/negative_interactions.csv",
                "path": self.NEG_INTERACTIONS_PATH,
                "post_process": None,
                "checksum": None
            },
            "PROTT5_MODEL": {
                "url": "https://example.com/models/prott5.h5",
                "path": self.PROTT5_MODEL_PATH,
                "post_process": None,
                "checksum": None
            },
            "ID_MAPPING_TSV": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/v4/idmapping_selected.tab.gz",
                "path": self.ID_MAPPING_PATH,
                "post_process": "ungzip",
                "checksum": None
            }
        }

    def _setup_benchmarking_params(self):
        """Sets parameters for the GNN benchmarking suite."""
        self.BENCHMARK_NODE_CLASSIFICATION_DATASETS = [
            "KarateClub", "Cora", "CiteSeer", "PubMed",
            "Cornell", "Texas", "Wisconsin"
        ]
        self.BENCHMARK_SAVE_EMBEDDINGS = True
        self.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = True
        self.BENCHMARK_TEST_ON_UNDIRECTED = True
        self.BENCHMARK_SPLIT_RATIOS: Dict[str, float] = {"train": 0.1, "val": 0.1, "test": 0.8}
        self.BENCHMARK_PCA_TARGET_DIM = 64

    def _setup_gcn_params(self):
        """Sets parameters for the main ProtGram-DirectGCN pipeline."""
        # Graph Building
        self.GCN_NGRAM_MAX_N = 3
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1

        # ID Mapping
        self.ID_MAPPING_MODE = 'regex'
        self.API_MAPPING_FROM_DB = "UniRef50"
        self.API_MAPPING_TO_DB = "UniProtKB"

        # Model Architecture
        self.GCN_HIDDEN_LAYER_DIMS = [256, 128, 64]
        self.GCN_1GRAM_INIT_DIM = 512
        self.GCN_MAX_PE_LEN = 512
        self.GCN_USE_VECTOR_COEFFS = True

        # Training Hyperparameters
        self.GCN_EPOCHS_PER_LEVEL = 300
        self.GCN_LR = 0.001
        self.GCN_DROPOUT_RATE = 0.5
        self.GCN_WEIGHT_DECAY = 1e-4
        self.GCN_L2_REG_LAMBDA = 1e-7
        self.GCN_PROPAGATION_EPSILON = 1e-9

        # LR Scheduler & Early Stopping
        self.GCN_USE_LR_SCHEDULER = True
        self.GCN_LR_SCHEDULER_PATIENCE = 10
        self.GCN_LR_SCHEDULER_FACTOR = 0.5
        self.GCN_USE_EARLY_STOPPING = True
        self.GCN_EARLY_STOPPING_PATIENCE = 25
        self.GCN_EARLY_STOPPING_MIN_DELTA = 1e-5

        # Self-Supervised Tasks
        self.GCN_TASK_TYPES_PER_LEVEL: Dict[int, str] = {
            1: "community", 2: "next_node", 3: "next_node",
        }
        self.GCN_DEFAULT_TASK_TYPE: str = "community"
        self.GCN_CLOSEST_AA_K_HOPS: int = 3

        # Cluster-GCN Strategy
        self.GCN_USE_CLUSTER_TRAINING = True
        self.GCN_CLUSTER_TRAINING_THRESHOLD_NODES = 10000
        self.GCN_TARGET_NODES_PER_CLUSTER = 2000
        self.GCN_MIN_CLUSTERS = 2
        self.GCN_MAX_CLUSTERS = 500

        # Post-Processing
        self.POOLING_WORKERS: Optional[int] = max(1, os.cpu_count() - 4) if os.cpu_count() else 1
        self.APPLY_PCA_TO_GCN = True
        self.PCA_TARGET_DIMENSION = 64

        # Sanity Check
        self.GCN_RUN_SANITY_CHECK_PPI = False
        self.GCN_SANITY_CHECK_EPOCHS = 5
        self.GCN_SANITY_CHECK_TEST_SPLIT = 0.2
        self.GCN_SANITY_CHECK_SAMPLE_SIZE = 2000

    def _setup_word2vec_params(self):
        """Sets parameters for the Word2Vec pipeline."""
        self.W2V_VECTOR_SIZE = 100
        self.W2V_WINDOW = 5
        self.W2V_MIN_COUNT = 1
        self.W2V_EPOCHS = 5
        self.W2V_WORKERS = 1
        self.W2V_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_W2V = True

    def _setup_transformer_params(self):
        """Sets parameters for the Transformer (e.g., ProtBERT) pipeline."""
        self.TRANSFORMER_MODELS_TO_RUN = [
            {"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1}
        ]
        self.TRANSFORMER_MAX_LENGTH = 1024
        self.TRANSFORMER_BASE_BATCH_SIZE = 16
        self.TRANSFORMER_POOLING_STRATEGY = 'mean'
        self.APPLY_PCA_TO_TRANSFORMER = True

    def _setup_evaluation_params(self):
        """Sets parameters for the final PPI evaluation pipeline."""
        # General
        self.PLOT_TRAINING_HISTORY = True
        self.EARLY_STOPPING_PATIENCE = 10
        self.PERFORM_H5_INTEGRITY_CHECK = True
        self.SAMPLE_NEGATIVE_PAIRS: Optional[int] = 100000
        self.TF_DATASET_STRATEGY = 'from_tensor_slices'

        # Embedding Files for Evaluation
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": "ProtT5", "path": self.PROTT5_MODEL_PATH},
            {"name": "ProtGramDirectGCN", "path": self.RESULTS_GCN_EMBEDDINGS_DIR / f"gcn_n{self.GCN_NGRAM_MAX_N}_embeddings_pca{self.PCA_TARGET_DIMENSION}.h5"},
        ]

        # MLP Architecture & Training
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

        # Reporting
        self.EVAL_K_VALUES_FOR_TABLE = [50, 100]
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtGramDirectGCN"
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05

    def _setup_mlflow_params(self):
        """Sets parameters for MLflow experiment tracking."""
        self.USE_MLFLOW = False
        mlruns_path = self.BASE_OUTPUT_DIR / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.resolve().as_uri()
        self.MLFLOW_EXPERIMENT_NAME = "PPI-Link-Prediction"
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = "GNN-Benchmarking"