# ==============================================================================
# MODULE: configuration/config.py
# PURPOSE: Centralized configuration for the entire PPI trainers.
# VERSION: 2.0 (Centralized more parameters from trainers/benchmarkers)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from pathlib import Path
from typing import Optional, Dict, List


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

        # --- NEW: 4.5. DYNAMICALLY LINK DATA SOURCES TO ATTRIBUTES ---
        self._link_data_sources_to_attributes()

        # --- 5. GNN BENCHMARKING PARAMETERS ---
        self._setup_benchmarking_params()

        # --- 6. ProtGram-DirectGCN PIPELINE PARAMETERS ---
        self._setup_gcn_params()

        # --- 7. WORD2VEC PIPELINE PARAMETERS ---
        self._setup_word2vec_params()

        # --- 8. TRANSFORMER PIPELINE PARAMETERS ---
        self._setup_transformer_params()

        # --- 9. LSTM PIPELINE PARAMETERS ---
        self._setup_lstm_params()

        # --- NEW: 9.5. SINGLETON EVALUATION PARAMETERS ---
        self._setup_singleton_eval_params()

        # --- 10. PPI EVALUATION PARAMETERS ---
        self._setup_evaluation_params()

        # --- 11. MLFLOW & EXPERIMENT TRACKING ---
        self._setup_mlflow_params()

    def _setup_paths(self):
        """Sets up all base, data, and results paths for the project."""
        # Base Paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.resolve()
        self.BASE_CONFIG_DIR = self.PROJECT_ROOT / "configuration"
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data"
        self.BASE_SOURCE_DIR = self.PROJECT_ROOT / "source"
        self.BASE_OUTPUT_DIR = self.PROJECT_ROOT / "results"
        self.PERSISTENT_DATA_CACHE = Path.home() / ".cache" / "protgram_directgcn"
        self.LOG_DIR = self.BASE_OUTPUT_DIR / "logs"

        # Data Subdirectories
        self.DATA_SEQUENCES_DIR = self.BASE_DATA_DIR / "sequences"
        self.DATA_GROUND_TRUTH_DIR = self.BASE_DATA_DIR / "ground_truth"
        self.DATA_MODELS_DIR = self.BASE_DATA_DIR / "models"
        self.DATA_MAPPINGS_DIR = self.BASE_DATA_DIR / "mappings"
        self.DATA_STANDARD_DATASETS_DIR = self.BASE_DATA_DIR / "benchmarks"

        # Results Subdirectories
        self.RESULTS_GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "graph_objects"
        self.RESULTS_GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "gcn_embeddings"
        self.RESULTS_W2V_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "word2vec_embeddings"
        self.RESULTS_LSTM_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "lstm_embeddings"
        self.RESULTS_TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "transformer_embeddings"
        self.RESULTS_EVALUATION_DIR = self.BASE_OUTPUT_DIR / "evaluation_results"
        self.RESULTS_BENCHMARKING_DIR = self.BASE_OUTPUT_DIR / "benchmarking_results"
        self.RESULTS_BENCHMARK_EMBEDDINGS_DIR = self.RESULTS_BENCHMARKING_DIR / "embeddings"

        # --- NEW: Proactively create all necessary directories ---
        # This makes the configuration self-sufficient and prevents FileNotFoundError
        # in downstream modules if they are run in isolation.
        dirs_to_create = [
            self.LOG_DIR, self.DATA_SEQUENCES_DIR, self.DATA_GROUND_TRUTH_DIR,
            self.DATA_MODELS_DIR, self.DATA_MAPPINGS_DIR, self.DATA_STANDARD_DATASETS_DIR,
            self.RESULTS_GRAPH_OBJECTS_DIR, self.RESULTS_GCN_EMBEDDINGS_DIR,
            self.RESULTS_W2V_EMBEDDINGS_DIR, self.RESULTS_LSTM_EMBEDDINGS_DIR,
            self.RESULTS_TRANSFORMER_EMBEDDINGS_DIR, self.RESULTS_EVALUATION_DIR,
            self.RESULTS_BENCHMARKING_DIR, self.RESULTS_BENCHMARK_EMBEDDINGS_DIR
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_pipeline_flags(self):
        """Sets flags to control which parts of the main pipeline are executed."""
        self.RUN_GCN_PIPELINE = True
        self.RUN_LSTM_PIPELINE = False
        self.RUN_WORD2VEC_PIPELINE = False
        self.RUN_TRANSFORMER_PIPELINE = False
        self.RUN_BENCHMARKING_PIPELINE = False
        self.RUN_NETWORK_EMBEDDING_BENCHMARKING = False
        self.RUN_MAIN_PPI_EVALUATION = True
        self.RUN_INTEGRATED_TESTS = False  # Runs all unit, smoke, and verification testers
        self.RUN_SINGLETON_GCN_EVAL = False # Runs a fast evaluation on the n=1 graph for rapid prototyping
        self.RUN_DUMMY_TEST = False  # Runs a quick evaluation on dummy data
        self.SEQUENCE_DOWNSAMPLE_FRACTION: Optional[float] = 0.95  # e.g., 0.1 for 10%. Set to None or >= 1.0 to disable.
        self.CLEANUP_DUMMY_DATA = False
        self.ENABLE_FILE_LOGGING = True

    def _setup_data_sources(self):
        """
        Defines the data sources for automatic download.
        The key is a unique identifier, and 'path' is the final destination.
        """
        self.DATA_SOURCES: Dict[str, Dict] = {
            "UNIPROT_SPROT_FASTA": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
                "path": self.DATA_SEQUENCES_DIR / "uniprot_sprot.fasta",
                "post_process": "ungzip",
                "checksum": None,
                "cacheable": True
            },
            "UNIREF_50_FASTA": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
                "path": self.DATA_SEQUENCES_DIR / "uniref50.fasta",
                "post_process": "ungzip",
                "checksum": None,
                "cacheable": True
            },
            "POS_INTERACTIONS": {
                # --- FIX: Use direct download link for Google Drive to avoid downloading HTML page ---
                "url": "https://drive.google.com/uc?id=1vDDdeOVdyu00y5Qux6HRdWtzywj9w7z4",
                "path": self.DATA_GROUND_TRUTH_DIR / "positive_interactions.csv",
                "post_process": None,
                "checksum": None,
                "cacheable": True
            },
            "NEG_INTERACTIONS": {
                # --- FIX: Use direct download link for Google Drive to avoid downloading HTML page ---
                "url": "https://drive.google.com/uc?id=1AZJS5_1XLM-GWWDjLWRTQU_5WQRROyrg",
                "path": self.DATA_GROUND_TRUTH_DIR / "negative_interactions.csv",
                "post_process": None,
                "checksum": None,
                "cacheable": True
            },
            "PROTT5_MODEL": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/uniprot_sprot/per-protein.h5",
                "path": self.DATA_MODELS_DIR / "per-protein.h5",
                "post_process": None, "checksum": None,
                "cacheable": True  # This large file will be cached persistently
            },
            "ID_MAPPING_TSV": {
                "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping.dat.gz",
                "path": self.DATA_MAPPINGS_DIR / "idmapping.dat",
                "post_process": "ungzip",
                "checksum": None,
                "cacheable": True  # This large file will be cached persistently
            }
        }

    def _link_data_sources_to_attributes(self):
        """
        Dynamically creates key file path attributes from DATA_SOURCES.
        This ensures a single source of truth for all data paths, removing redundancy.
        """
        # Link specific, named file paths for easy access elsewhere in the code.
        self.POS_INTERACTIONS_PATH = self.DATA_SOURCES['POS_INTERACTIONS']['path']
        self.NEG_INTERACTIONS_PATH = self.DATA_SOURCES['NEG_INTERACTIONS']['path']
        self.ID_MAPPING_PATH = self.DATA_SOURCES['ID_MAPPING_TSV']['path']
        self.PROTT5_MODEL_PATH = self.DATA_SOURCES['PROTT5_MODEL']['path']

        # Dynamically build the list of all available FASTA files.
        self.ORIGINAL_SEQUENCE_FILE_PATHS = [
            Path(source_info['path'])
            for key, source_info in self.DATA_SOURCES.items()
            if 'path' in source_info and str(source_info['path']).endswith(('.fasta', '.fa'))
        ]
        # This is the "working" list of paths, which can be modified for downsampling during a run.
        self.SEQUENCE_FILE_PATHS = self.ORIGINAL_SEQUENCE_FILE_PATHS.copy()

    def _setup_benchmarking_params(self):
        """Sets parameters for the GNN benchmarking suite."""
        self.BENCHMARK_NODE_CLASSIFICATION_DATASETS = [
            "KarateClub", "Cora", "CiteSeer", "PubMed",
            "Cornell", "Texas", "Wisconsin"
        ]
        # List of GNN models to run in the benchmark suite.
        # Options: "GCN", "GAT", "GraphSAGE", "GIN", "ChebNet", "RGCN", "TongDiGCN", "DirectGCN"
        self.BENCHMARK_GNN_MODELS_TO_RUN: List[str] = ["GCN", "GAT", "GraphSAGE", "GIN", "ChebNet", "RGCN", "DirGNN", "DirectGCN"]
        self.BENCHMARK_SAVE_EMBEDDINGS = True
        self.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = True
        self.BENCHMARK_TEST_ON_UNDIRECTED = True
        self.BENCHMARK_SPLIT_RATIOS: Dict[str, float] = {"train": 0.1, "val": 0.1, "test": 0.8}
        self.BENCHMARK_PCA_TARGET_DIM = 64
        self.BENCHMARK_NE_MODELS_TO_RUN = ["Node2Vec"]
        # Number of epochs for the network embedding benchmark (Node2Vec, etc.)
        self.BENCHMARK_NE_EPOCHS = 5
        self.BENCHMARK_NE_EMBEDDING_DIM = 128
        self.BENCHMARK_NE_WALK_LENGTH = 20
        self.BENCHMARK_NE_CONTEXT_SIZE = 10

        # --- GNN Architecture for Benchmarking ---
        # These parameters control the architecture of GNNs used in the standard
        # node classification benchmarks (Cora, PubMed, etc.).
        self.BENCHMARK_GNN_HIDDEN_CHANNELS = 64
        self.BENCHMARK_GNN_NUM_LAYERS = 2
        self.BENCHMARK_GNN_DROPOUT_RATE = 0.5
        self.BENCHMARK_GAT_HEADS = 2
        self.BENCHMARK_GAT_DROPOUT_RATE = 0.5  # GAT often benefits from higher dropout
        self.BENCHMARK_CHEBNET_K = 3
        self.BENCHMARK_GNN_LEARNING_RATE = 0.01
        # --- NEW: Dedicated epochs for the GNN benchmark suite ---
        self.BENCHMARK_GNN_EPOCHS = 100
        self.BENCHMARK_RGCN_NUM_RELATIONS = 2
        # --- FIX: Align DirectGCN's benchmark architecture with other GNNs for a fair comparison. ---
        # It will now also be a 2-layer model with 64 hidden channels.
        self.BENCHMARK_DIRECTGCN_HIDDEN_LAYER_DIMS = [self.BENCHMARK_GNN_HIDDEN_CHANNELS] * self.BENCHMARK_GNN_NUM_LAYERS
        # --- NEW: Dedicated initial feature dimension for benchmark models ---
        self.BENCHMARK_GNN_INIT_DIM = 64

    def _setup_gcn_params(self):
        """Sets parameters for the main ProtGram-DirectGCN pipeline."""
        # --- ProtGram Graph Building --- #FIXME: This parameter is not used anywhere.
        self.PROTGRAM_NGRAM_MAX_N = 3
        # --- FIX: Safely handle os.cpu_count() returning None ---
        cpu_cores = os.cpu_count()
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, cpu_cores - 4) if cpu_cores is not None else 1

        # ID Mapping
        # Options: 'file' (recommended), 'regex', 'api', 'none'.
        # 'file': Uses the large idmapping.dat to create a robust local SQLite DB. Best for production.
        # 'regex': Fast, but relies on standard UniProt headers (e.g., >sp|P12345|...).
        # 'api': Uses the live UniProt API. Slow, for small-scale use only.
        self.ID_MAPPING_MODE = 'regex'
        # This value is now set dynamically in main.py based on the input FASTA file
        # to correctly handle different UniRef versions (e.g., UniRef50, UniRef100).
        self.API_MAPPING_FROM_DB: Optional[str] = None
        self.API_MAPPING_TO_DB = "UniProtKB"

        # Model Selection for ProtGram
        # Options: 'directgcn', 'rgcn', 'tongdigcn'
        # 'rgcn' treats in/out edges as 2 relations.
        self.PROTGRAM_MODELS_TO_TRAIN = ['directgcn']

        # --- ProtGram Model Architecture ---
        # Defines the architecture of the DirectGCN model. The length of the list determines
        # the model's depth, and each value specifies the output dimension of a GCN layer.
        # The final value is the dimension of the output node embeddings.
        self.DIRECTGCN_HIDDEN_LAYER_DIMS = [512, 256, 128, 64]
        self.PROTGRAM_1GRAM_INIT_DIM = 512
        self.PROTGRAM_MAX_PE_LEN = 512 # Max length for positional embeddings
        # --- NEW: Architecture for other GNNs in the ProtGram pipeline ---
        self.PROTGRAM_GNN_HIDDEN_CHANNELS = 64
        self.PROTGRAM_GNN_NUM_LAYERS = 2
        # Gating mode for DirectGCN.
        # 'scalar': One learnable scalar per path, per layer (shared by all nodes).
        # 'vector': One learnable scalar per path, per node, per layer (more expressive).
        # 'node_gate_vector': A learnable *vector* per path, per node, per layer for element-wise gating. Most expressive.
        # 'none': No gating, paths are simply added.
        self.PROTGRAM_GATING_COEFF_MODE = "vector"
        # NEW: Control whether to add positional embeddings in the DirectGCN model.
        self.PROTGRAM_USE_POSITIONAL_EMBEDDING: bool = False

        # --- ProtGram Training Hyperparameters ---
        self.PROTGRAM_EPOCHS_PER_LEVEL = 500
        self.PROTGRAM_LR = 0.005
        self.PROTGRAM_DROPOUT_RATE = 0.5
        self.PROTGRAM_WEIGHT_DECAY = 1e-4 # Standard L2 regularization
        self.PROTGRAM_USE_LR_SCHEDULER = True
        self.PROTGRAM_LR_SCHEDULER_PATIENCE = 10
        self.PROTGRAM_LR_SCHEDULER_FACTOR = 0.5
        self.PROTGRAM_USE_EARLY_STOPPING = True
        self.PROTGRAM_EARLY_STOPPING_PATIENCE = 50
        self.PROTGRAM_EARLY_STOPPING_MIN_DELTA = 1e-5

        # --- ProtGram Self-Supervised Tasks ---
        # --- FIX: Use 'community' for n=1 as 'masked_node' is unsolvable with random features ---
        self.PROTGRAM_TASK_TYPES_PER_LEVEL: Dict[int, str] = {
            1: "community", 2: "next_node", 3: "next_node"
        }
        self.PROTGRAM_DEFAULT_TASK_TYPE: str = "community"
        # --- NEW: Make the homophily threshold a configurable parameter ---
        self.GCN_HETEROPHILY_THRESHOLD: float = 0.6
        self.PROTGRAM_CLOSEST_AA_K_HOPS: int = 3
        self.PROTGRAM_MASKED_NODE_FRACTION: float = 0.15  # Fraction of nodes to mask for the masked_node task

        # --- ProtGram Cluster-GCN Strategy ---
        self.PROTGRAM_USE_CLUSTER_TRAINING = True
        self.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES = 5000
        self.PROTGRAM_TARGET_NODES_PER_CLUSTER = 2000
        self.PROTGRAM_MIN_CLUSTERS = 2
        self.PROTGRAM_MAX_CLUSTERS = 500

        # --- ProtGram Post-Processing ---
        # --- FIX: Safely handle os.cpu_count() returning None ---
        self.POOLING_WORKERS: Optional[int] = max(1, cpu_cores - 4) if cpu_cores is not None else 1
        self.PCA_TARGET_DIMENSION = 64
        # NEW: Protein-level pooling strategy
        # Strategy for pooling final n-gram embeddings to create a single protein embedding.
        # Options: 'mean' (fast), 'sum', 'max', 'attention' (slower, more expressive)
        self.PROTGRAM_PROTEIN_POOLING_STRATEGY = 'attention'
        # NEW: Hierarchical pooling strategy
        # Strategy for pooling (n-1)-gram embeddings to initialize n-gram features for n>1.
        # Options: 'mean', 'attention'
        self.PROTGRAM_HIERARCHICAL_POOLING_STRATEGY = 'attention'
        # NEW: Control whether to log potentially large attention files.
        # Set to True to generate attention plots, False to save memory/time.
        self.PROTGRAM_LOG_ATTENTION_WEIGHTS = True

        # --- ProtGram Sanity Check ---
        self.PROTGRAM_RUN_SANITY_CHECK_PPI = True
        self.PROTGRAM_SANITY_CHECK_EPOCHS = 5
        self.PROTGRAM_SANITY_CHECK_TEST_SPLIT = 0.2
        self.PROTGRAM_SANITY_CHECK_SAMPLE_SIZE = 2000

    def _setup_word2vec_params(self):
        """Sets parameters for the Word2Vec pipeline."""
        self.W2V_VECTOR_SIZE = 100
        self.W2V_WINDOW = 5
        self.W2V_MIN_COUNT = 1
        self.W2V_EPOCHS = 5
        # --- FIX: Safely handle os.cpu_count() returning None ---
        cpu_cores = os.cpu_count()
        self.W2V_WORKERS: Optional[int] = max(1, cpu_cores - 4) if cpu_cores is not None else 1
        self.W2V_POOLING_STRATEGY = 'mean' # Options: 'mean', 'sum', 'max'
        self.APPLY_PCA_TO_W2V = True # Apply PCA to match other embedding dimensions

    def _setup_transformer_params(self):
        """Sets parameters for the Transformer (e.g., ProtBERT) pipeline."""
        self.TRANSFORMER_MODELS_TO_RUN = [
            # --- DIAGNOSTIC STEP: Temporarily disable the large ProtBERT model to test for memory issues. ---
            # If the pipeline succeeds with only ESM2, the previous crash was due to RAM limitations.
            {"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1.0},
            {"name": "ESM2", "hf_id": "facebook/esm2_t6_8M_UR50D", "is_t5": False, "batch_size_multiplier": 1.0}
        ]
        self.TRANSFORMER_MAX_LENGTH = 1024
        self.TRANSFORMER_BASE_BATCH_SIZE = 16
        # Process the full FASTA file in chunks to avoid loading all sequences into memory at once.
        self.TRANSFORMER_CHUNK_SIZE = 10000
        self.TRANSFORMER_POOLING_STRATEGY = 'mean'
        self.USE_XLA_COMPILATION = True  # Set to True for faster inference

    def _setup_lstm_params(self):
        """Sets parameters for the LSTM embedding pipeline."""
        self.LSTM_EMBEDDING_DIM = 100
        self.LSTM_HIDDEN_DIM = 256
        self.LSTM_NUM_LAYERS = 2
        self.LSTM_EPOCHS = 5
        self.LSTM_BATCH_SIZE = 64
        self.LSTM_TRAIN_SEQ_LEN = 50
        self.LSTM_TRAIN_STEP = 50
        self.LSTM_LEARNING_RATE = 0.001
        self.LSTM_POOLING_STRATEGY = 'mean'

    def _setup_singleton_eval_params(self):
        """Sets parameters for the rapid, n=1 GCN evaluation."""
        self.SINGLETON_EVAL_EPOCHS = 100
        self.SINGLETON_EVAL_TEST_SPLIT = 0.2
        self.SINGLETON_EVAL_LR = 0.01
        # A list of models to test in the singleton evaluation.
        # Options: "GCN", "GAT", "GraphSAGE", "GIN", "ChebNet", "RGCN", "TongDiGCN", "DirectGCN"
        self.SINGLETON_EVAL_MODELS_TO_RUN: List[str] = ["DirectGCN", "GCN", "RGCN", "DirGNN"]
        # --- NEW: Dedicated architecture parameters for the singleton evaluation ---
        self.SINGLETON_GNN_HIDDEN_CHANNELS = 64
        self.SINGLETON_GNN_NUM_LAYERS = 2
        self.SINGLETON_GNN_DROPOUT_RATE = 0.5
        self.SINGLETON_GAT_HEADS = 2
        self.SINGLETON_GAT_DROPOUT_RATE = 0.5
        self.SINGLETON_CHEBNET_K = 3
        self.SINGLETON_RGCN_NUM_RELATIONS = 2
        # --- FIX: Align DirectGCN's singleton architecture with other GNNs for a fair comparison. ---
        # It will now also be a 2-layer model with 64 hidden channels.
        self.SINGLETON_DIRECTGCN_HIDDEN_LAYER_DIMS = [self.SINGLETON_GNN_HIDDEN_CHANNELS] * self.SINGLETON_GNN_NUM_LAYERS

    def _setup_evaluation_params(self):
        """Sets parameters for the final PPI evaluation pipeline."""
        # General
        self.PLOT_TRAINING_HISTORY = True
        self.PERFORM_H5_INTEGRITY_CHECK = True
        self.EVAL_GENERATE_SHAP_SUMMARY = True
        self.EARLY_STOPPING_PATIENCE = 10 # For the MLP classifier

        # List of pre-existing or external embedding files to include in evaluation.
        # Embeddings generated during the pipeline run will be added automatically.
        self.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE = [
            {"name": "ProtT5", "path": self.PROTT5_MODEL_PATH},
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
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = "ProtGramDirectgcn"
        self.EVAL_STATISTICAL_TEST_ALPHA = 0.05

    def _setup_mlflow_params(self):
        """Sets parameters for MLflow experiment tracking."""
        self.USE_MLFLOW = True
        mlruns_path = self.BASE_OUTPUT_DIR / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.as_uri()  # Use as_uri() for proper file URI scheme.
        self.MLFLOW_EXPERIMENT_NAME = "PPI-Link-Prediction"
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = "GNN-Benchmarking"
        self.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME = "Network_Embedding_Benchmarking"
