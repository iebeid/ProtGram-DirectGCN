# ==============================================================================
# MODULE: configuration/config.py
# PURPOSE: Centralized configuration loaded from a YAML file.
# VERSION: 3.6 (Corrected syntax errors and added requirements)
# VERSION: 4.0 (Integrated run-specific directory creation)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
import yaml
import sys
from pydantic import BaseModel, Field, ValidationError
import time


# ==============================================================================
# Pydantic Validation Schemas
# ==============================================================================

class DataUrls(BaseModel):
    UNIPROT_SPROT_FASTA: str
    UNIREF_50_FASTA: Optional[str] = None
    BIOGRID_INTERACTIONS: str
    PROTT5_MODEL: str
    UNIPROT_ID_MAPPING: str
    NEG_INTERACTIONS: List[str] = Field(default_factory=list)

class ResourceManagementParams(BaseModel):
    MEMORY_USAGE_STRATEGY: str
    CHECKSUM_SKIP_SIZE_MB: int = Field(default=500, gt=0)

class PipelineFlags(BaseModel):
    RUN_PROTGRAM_XGCN_PIPELINE: bool
    RUN_LSTM_PIPELINE: bool
    RUN_WORD2VEC_PIPELINE: bool
    RUN_TRANSFORMER_PIPELINE: bool
    RUN_BENCHMARKING_PIPELINE: bool # noqa
    RUN_PROTGRAM_PIPELINE: bool # noqa
    RUN_MAIN_PPI_EVALUATION: bool
    RUN_INTEGRATED_TESTS: bool
    RUN_SINGLETON_GCN_EVAL: bool
    SEQUENCE_DOWNSAMPLE_FRACTION: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    DISABLE_INTERACTIVE_PROMPTS: bool = Field(default=False)
    ENABLE_FILE_LOGGING: bool
    PROCESS_ID_MAPPING_FILE: bool

class BaseGNNTrainingParams(BaseModel):
    """A base model for shared GNN training hyperparameters to ensure consistency."""
    EPOCHS: int = Field(gt=0)
    LEARNING_RATE: float = Field(gt=0)
    WEIGHT_DECAY: float = Field(default=5e-4, ge=0.0)

class GNNBenchmarkingParams(BaseGNNTrainingParams):
    BENCHMARK_NODE_CLASSIFICATION_DATASETS: List[str] = Field(alias='DATASETS')
    GNN_MODELS_TO_RUN: List[str]
    NE_MODELS_TO_RUN: List[str]
    SAVE_EMBEDDINGS: bool
    TEST_ON_UNDIRECTED: bool
    SPLIT_RATIOS: Dict[str, float]
    PCA_TARGET_DIM: int = Field(gt=0)
    NE_LEARNING_RATE: float = Field(gt=0)
    NE_BATCH_SIZE: int = Field(gt=0)
    NE_EPOCHS: int = Field(gt=0)
    NE_EMBEDDING_DIM: int = Field(gt=0)
    NE_WALK_LENGTH: int = Field(gt=0)
    NE_CONTEXT_SIZE: int = Field(gt=0)
    NE_WALKS_PER_NODE: int = Field(gt=0)
    NE_NUM_NEGATIVE_SAMPLES: int = Field(gt=0)
    NE_CLASSIFIER_C: float = Field(gt=0)
    NE_CLASSIFIER_MAX_ITER: int = Field(gt=0)
    NE_CLASSIFIER_SOLVER: str
    GNN_HIDDEN_CHANNELS: int = Field(gt=0)
    GNN_NUM_LAYERS: int = Field(gt=0)
    GNN_DROPOUT_RATE: float = Field(ge=0.0, lt=1.0)
    GAT_HEADS: int = Field(gt=0)
    GAT_DROPOUT_RATE: float = Field(ge=0.0, lt=1.0)
    CHEBNET_K: int = Field(gt=0)
    RGCN_NUM_RELATIONS: int = Field(gt=0)
    DIRECTGCN_HIDDEN_LAYER_DIMS: List[int]
    GNN_INIT_DIM: int = Field(gt=0)

class ProtGramGCNParams(BaseModel):
    
    PROTGRAM_NGRAM_MAX_N: int = Field(gt=0, lt=5, description="Maximum n-gram size. Kept below 5 for memory efficiency.")
    FASTA_FILE_TO_PROCESS: str
    ID_MAPPING_MODE: str
    USE_CANONICAL_ID_MAPPING_FILE: bool
    REGEX_CONFIDENCE_THRESHOLD: float = Field(ge=0.0, le=1.0)
    REGEX_COMPATIBILITY_SAMPLE_SIZE: int = Field(gt=0)
    API_MAPPING_FROM_DB: Optional[str]
    API_MAPPING_TO_DB: str
    PROTGRAM_MODELS_TO_TRAIN: List[str]
    DIRECTGCN_HIDDEN_LAYER_DIMS: List[int]
    PROTGRAM_1GRAM_INIT_DIM: int = Field(gt=0)
    PROTGRAM_GNN_HIDDEN_CHANNELS: int = Field(gt=0)
    PROTGRAM_GNN_NUM_LAYERS: int = Field(gt=0)
    PROTGRAM_GATING_COEFF_MODE: str
    PROTGRAM_EPOCHS_PER_LEVEL: int = Field(gt=0)
    PROTGRAM_LR: float = Field(gt=0)
    PROTGRAM_DROPOUT_RATE: float = Field(ge=0.0, lt=1.0)
    PROTGRAM_WEIGHT_DECAY: float = Field(ge=0.0)
    PROTGRAM_USE_LR_SCHEDULER: bool
    PROTGRAM_LR_SCHEDULER_PATIENCE: int = Field(gt=0)
    PROTGRAM_LR_SCHEDULER_FACTOR: float = Field(gt=0, lt=1.0)
    PROTGRAM_USE_EARLY_STOPPING: bool
    PROTGRAM_EARLY_STOPPING_PATIENCE: int = Field(gt=0)
    PROTGRAM_EARLY_STOPPING_MIN_DELTA: float = Field(ge=0.0)
    PROTGRAM_GRADIENT_ACCUMULATION_STEPS: int = Field(default=1, ge=1)
    PROTGRAM_TASK_TYPES_PER_LEVEL: Dict[int, str]
    PROTGRAM_DEFAULT_TASK_TYPE: str
    GCN_HETEROPHILY_THRESHOLD: float = Field(ge=0.0, le=1.0)
    GCN_PROPAGATION_EPSILON: float = Field(gt=0, description="Small value to add for numerical stability in DirectGCN propagation.")
    PROTGRAM_CLOSEST_AA_K_HOPS: int = Field(gt=0)
    PROTGRAM_MASKED_NODE_FRACTION: float = Field(gt=0, lt=1.0)
    PROTGRAM_CLEAN_FASTA_ON_PARSE: bool
    PROTGRAM_FASTA_MIN_LEN: int = Field(gt=0)
    PROTGRAM_FASTA_MAX_LEN: int = Field(gt=0)
    PROTGRAM_FASTA_ALPHABET: str
    PROTGRAM_USE_CLUSTER_TRAINING: bool
    PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES: int = Field(gt=0)
    PROTGRAM_PARTITIONING_METHOD: str
    PROTGRAM_COARSENING_LEVEL_FOR_PARTITIONING: int = Field(default=1, ge=1)
    PROTGRAM_VALIDATE_COARSENING: bool
    PROTGRAM_TARGET_NODES_PER_CLUSTER: int = Field(gt=0)
    PROTGRAM_MIN_CLUSTERS: int = Field(gt=0)
    PROTGRAM_MAX_CLUSTERS: int = Field(gt=0)
    PROTGRAM_CLUSTER_GROUP_SIZE: int = Field(default=1, ge=1)
    PCA_TARGET_DIMENSION: int
    PROTGRAM_PROTEIN_POOLING_STRATEGY: str
    PROTGRAM_HIERARCHICAL_POOLING_STRATEGY: str
    PROTGRAM_LOG_ATTENTION_WEIGHTS: bool
    PROTGRAM_RUN_SANITY_CHECK_PPI: bool
    PROTGRAM_SANITY_CHECK_EPOCHS: int = Field(gt=0)
    PROTGRAM_SANITY_CHECK_TEST_SPLIT: float = Field(gt=0, lt=1.0)
    PROTGRAM_SANITY_CHECK_SAMPLE_SIZE: int = Field(gt=0)

class Word2VecParams(BaseModel):
    W2V_VECTOR_SIZE: int = Field(gt=0)
    W2V_WINDOW: int = Field(gt=0)
    W2V_MIN_COUNT: int = Field(ge=1)
    W2V_EPOCHS: int = Field(gt=0)
    W2V_POOLING_STRATEGY: str
    APPLY_PCA_TO_W2V: bool

class TransformerParams(BaseModel):
    MODELS_TO_RUN: List[Dict[str, Any]]
    MAX_LENGTH: int = Field(gt=0)
    BASE_BATCH_SIZE: int = Field(gt=0)
    CHUNK_SIZE: int = Field(gt=0)
    POOLING_STRATEGY: str
    USE_XLA_COMPILATION: bool
    TRANSFORMER_INFERENCE_SAMPLE_FRACTION: float = Field(default=1.0, ge=0.0, le=1.0)

class LSTMParams(BaseModel):
    EMBEDDING_DIM: int = Field(gt=0)
    HIDDEN_DIM: int = Field(gt=0)
    NUM_LAYERS: int = Field(gt=0)
    EPOCHS: int = Field(gt=0)
    BATCH_SIZE: int = Field(gt=0)
    TRAIN_SEQ_LEN: int = Field(gt=0)
    TRAIN_STEP: int = Field(gt=0)
    LEARNING_RATE: float = Field(gt=0)
    POOLING_STRATEGY: str
    VALIDATION_SPLIT: float = Field(gt=0, lt=1.0)
    USE_EARLY_STOPPING: bool
    EARLY_STOPPING_PATIENCE: int = Field(ge=0)
    DROPOUT_RATE: float = Field(default=0.5, ge=0.0, lt=1.0)
    EARLY_STOPPING_MIN_DELTA: float = Field(ge=0.0)

class SingletonEvalParams(BaseGNNTrainingParams):
    TEST_SPLIT: float = Field(gt=0, lt=1.0)
    MODELS_TO_RUN: List[str]
    GNN_HIDDEN_CHANNELS: int = Field(gt=0)
    GNN_NUM_LAYERS: int = Field(gt=0)
    GNN_DROPOUT_RATE: float = Field(ge=0.0, lt=1.0)
    GAT_HEADS: int = Field(gt=0)
    GAT_DROPOUT_RATE: float = Field(ge=0.0, lt=1.0)
    CHEBNET_K: int = Field(gt=0)
    RGCN_NUM_RELATIONS: int = Field(gt=0)
    GRADIENT_ACCUMULATION_STEPS: int = Field(default=1, ge=1)
    DIRECTGCN_HIDDEN_LAYER_DIMS: List[int]

class PPIEvaluationParams(BaseModel):
    PLOT_TRAINING_HISTORY: bool
    PERFORM_H5_INTEGRITY_CHECK: bool
    EVAL_GENERATE_SHAP_SUMMARY: bool
    EARLY_STOPPING_PATIENCE: int = Field(ge=0)
    EDGE_EMBEDDING_METHOD: str
    N_FOLDS: int = Field(gt=1)
    MLP_DENSE1_UNITS: int = Field(gt=0)
    MLP_DROPOUT1_RATE: float = Field(ge=0.0, lt=1.0)
    MLP_DENSE2_UNITS: int = Field(gt=0)
    MLP_DROPOUT2_RATE: float = Field(ge=0.0, lt=1.0)
    MLP_L2_REG: float = Field(ge=0.0)
    BATCH_SIZE: int = Field(gt=0)
    EPOCHS: int = Field(gt=0)
    MLP_LEARNING_RATE: float = Field(gt=0)
    K_VALUES_FOR_TABLE: List[int]
    MAIN_EMBEDDING_FOR_STATS: str
    STATISTICAL_TEST_ALPHA: float = Field(gt=0, lt=1.0)

class MLflowParams(BaseModel):
    USE_MLFLOW: bool
    EXPERIMENT_NAME: str
    BENCHMARK_EXPERIMENT_NAME: str
    LLMS_EXPERIMENT_NAME: str
    PROTGRAM_XGCN_EXPERIMENT_NAME: str
    INTERPRETABILITY_EXPERIMENT_NAME: str

class HPOSearchSpaceItem(BaseModel):
    type: str
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None

class HPOParams(BaseModel):
    RUN_HPO: bool
    HPO_N_TRIALS: int = Field(default=50, gt=0)
    HPO_TARGET_EMBEDDING_MODEL: str = ""
    HPO_PPI_MLP_SEARCH_SPACE: Dict[str, HPOSearchSpaceItem] = Field(default_factory=dict)

class ValidationSchema(BaseModel):
    """The root model for validating the entire
    config.yaml file."""
    RANDOM_STATE: int
    resource_management: ResourceManagementParams
    DEBUG_VERBOSE: bool
    data_urls: DataUrls
    pipeline_flags: PipelineFlags
    gnn_benchmarking: GNNBenchmarkingParams
    protgram_gcn: ProtGramGCNParams
    word2vec: Word2VecParams
    transformer: TransformerParams
    lstm: LSTMParams
    singleton_eval: SingletonEvalParams
    ppi_evaluation: PPIEvaluationParams
    mlflow: MLflowParams
    hyperparameter_optimization: HPOParams = Field(default_factory=HPOParams)

class Config:
    def __init__(self, config_path: str = 'configuration/config.yaml'):
        # --- 0. LOAD YAML CONFIG ---
        self._config = self._load_yaml_config(config_path)

        # --- 1. GENERAL SETTINGS (from YAML) ---
        # --- VALIDATE CONFIGURATION FIRST (Fail-Fast) ---
        self._validate_config()

        self.RANDOM_STATE: int = self._config['RANDOM_STATE']
        self.DEBUG_VERBOSE: bool = self._config['DEBUG_VERBOSE']

        # --- 2. PATHS & DIRECTORIES ---
        # --- DEFINITIVE FIX for Test Isolation ---
        # The project root is now set here, once. The _setup_paths method
        # will derive all other paths from this, allowing tests to override it
        # before calling _setup_paths to create an isolated environment.
        self.PROJECT_ROOT = Path(__file__).parent.parent.resolve()

        # --- DEFINITIVE FIX: Create a unique output directory for each run ---
        # This logic is now centralized within the Config class itself, ensuring
        # that all paths are correct from the moment the object is instantiated.
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.BASE_OUTPUT_DIR = self.PROJECT_ROOT / "results" / run_id

        # Now, set up all other paths based on this unique directory
        self._setup_paths()

        # --- NEW: Set up resource management parameters ---
        self._setup_resource_management_params()

        # --- 3. PIPELINE CONTROL FLAGS (from YAML) ---
        self._setup_pipeline_flags()

        # --- 4. GNN BENCHMARKING PARAMETERS (from YAML) ---
        self._setup_benchmarking_params()

        # --- 5. GCN PARAMETERS (from YAML) ---
        # This must be called before _setup_data_sources because the data source
        # logic depends on the USE_CANONICAL_ID_MAPPING_FILE flag.
        self._setup_gcn_params()

        # --- 6. DATA SOURCES (Dynamically set, logic from original file) ---
        self._setup_data_sources()

        # --- 7. WORD2VEC PIPELINE PARAMETERS (from YAML) ---
        self._setup_word2vec_params()

        # --- 8. TRANSFORMER PIPELINE PARAMETERS (from YAML) ---
        self._setup_transformer_params()

        # --- 9. LSTM PIPELINE PARAMETERS (from YAML) ---
        self._setup_lstm_params()

        # --- 10. SINGLETON EVALUATION PARAMETERS (from YAML) ---
        self._setup_singleton_eval_params()

        # --- 11. PPI EVALUATION PARAMETERS (from YAML) ---
        self._setup_evaluation_params()

        # --- 12. MLFLOW & EXPERIMENT TRACKING (from YAML) ---
        self._setup_mlflow_params()

        # --- 13. HYPERPARAMETER OPTIMIZATION (from YAML) ---
        self._setup_hpo_params()

        # --- LAST STEP: Link attributes now that all params are loaded ---
        self._link_data_sources_to_attributes()

    def _load_yaml_config(self, config_path: str) -> Dict:
        """Loads the YAML configuration file."""
        full_path = Path.cwd() / config_path
        if not full_path.exists():
            full_path = Path(__file__).parent.parent / config_path
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {full_path}")
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Sets up all base, data, and results paths for the project."""
        self.BASE_CONFIG_DIR = self.PROJECT_ROOT / "configuration"
        self.BASE_DATA_DIR = self.PROJECT_ROOT / "data"
        self.BASE_OUTPUT_DIR = self.PROJECT_ROOT / "results"
        self.PERSISTENT_DATA_CACHE = Path.home() / ".cache" / "protgram_directgcn"
        self.LOG_DIR = self.BASE_OUTPUT_DIR / "logs"
        self.DATA_SEQUENCES_DIR = self.BASE_DATA_DIR / "sequences"
        self.DATA_GROUND_TRUTH_DIR = self.BASE_DATA_DIR / "ground_truth"
        self.DATA_MODELS_DIR = self.BASE_DATA_DIR / "models"
        self.DATA_MAPPINGS_DIR = self.BASE_DATA_DIR / "mappings"
        self.DATA_STANDARD_DATASETS_DIR = self.BASE_DATA_DIR / "benchmarks"
        self.RESULTS_GRAPH_OBJECTS_DIR = self.BASE_OUTPUT_DIR / "graph_objects"
        self.RESULTS_GCN_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "gcn_embeddings"
        self.RESULTS_W2V_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "word2vec_embeddings"
        self.RESULTS_LSTM_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "lstm_embeddings"
        self.RESULTS_TRANSFORMER_EMBEDDINGS_DIR = self.BASE_OUTPUT_DIR / "transformer_embeddings"
        self.RESULTS_EVALUATION_DIR = self.BASE_OUTPUT_DIR / "evaluation_results"
        self.RESULTS_BENCHMARKING_DIR = self.BASE_OUTPUT_DIR / "benchmarking_results"
        self.RESULTS_BENCHMARK_EMBEDDINGS_DIR = self.RESULTS_BENCHMARKING_DIR / "embeddings"
        # --- DEFINITIVE FIX for Test Isolation ---
        # These paths depend on the base paths above. They must be re-calculated
        # whenever _setup_paths is called to ensure that unit tests using a
        # temporary project root are fully isolated.
        self.POS_INTERACTIONS_PATH = self.DATA_GROUND_TRUTH_DIR / "positive_interactions.parquet"
        self.NEG_INTERACTIONS_PATH = self.DATA_GROUND_TRUTH_DIR / "negative_interactions.parquet"
        self.ID_MAPPING_PATH = self.DATA_MAPPINGS_DIR / "id_mapping.parquet"
        self.DATA_BUNDLE_PATH = self.PERSISTENT_DATA_CACHE / "data_bundle.tar.gz"
        self.DATA_MANIFEST_PATH = self.PERSISTENT_DATA_CACHE / "data_manifest.json"

    def _setup_resource_management_params(self):
        """
        Sets the memory usage strategy from the config. The actual logic for how
        to use this strategy is handled by the components that need it (e.g., EmbeddingLoader).
        """
        params = self._config['resource_management']
        self.MEMORY_USAGE_STRATEGY = params['MEMORY_USAGE_STRATEGY']
        self.CHECKSUM_SKIP_SIZE_BYTES = params['CHECKSUM_SKIP_SIZE_MB'] * 1024 * 1024

    def _setup_pipeline_flags(self):
        """Sets flags statically from the YAML config."""
        flags = self._config['pipeline_flags']
        self.RUN_PROTGRAM_XGCN_PIPELINE = flags['RUN_PROTGRAM_XGCN_PIPELINE']
        self.RUN_LSTM_PIPELINE = flags['RUN_LSTM_PIPELINE']
        self.RUN_WORD2VEC_PIPELINE = flags['RUN_WORD2VEC_PIPELINE']
        self.RUN_TRANSFORMER_PIPELINE = flags['RUN_TRANSFORMER_PIPELINE']
        self.RUN_BENCHMARKING_PIPELINE = flags['RUN_BENCHMARKING_PIPELINE']
        self.RUN_PROTGRAM_PIPELINE = flags['RUN_PROTGRAM_PIPELINE']
        
        self.RUN_MAIN_PPI_EVALUATION = flags['RUN_MAIN_PPI_EVALUATION']
        self.RUN_INTEGRATED_TESTS = flags['RUN_INTEGRATED_TESTS']
        self.RUN_SINGLETON_GCN_EVAL = flags['RUN_SINGLETON_GCN_EVAL']
        self.SEQUENCE_DOWNSAMPLE_FRACTION = flags['SEQUENCE_DOWNSAMPLE_FRACTION'] # noqa
        self.DISABLE_INTERACTIVE_PROMPTS = flags['DISABLE_INTERACTIVE_PROMPTS']
        self.ENABLE_FILE_LOGGING = flags['ENABLE_FILE_LOGGING']
        self.PROCESS_ID_MAPPING_FILE = flags['PROCESS_ID_MAPPING_FILE']

    def _setup_benchmarking_params(self):
        """Sets GNN benchmarking parameters statically from the YAML config."""
        params = self._config['gnn_benchmarking']
        self.BENCHMARK_NODE_CLASSIFICATION_DATASETS = params['BENCHMARK_NODE_CLASSIFICATION_DATASETS']
        self.BENCHMARK_GNN_MODELS_TO_RUN = params['GNN_MODELS_TO_RUN']
        self.BENCHMARK_NE_MODELS_TO_RUN = params['NE_MODELS_TO_RUN']
        self.BENCHMARK_SAVE_EMBEDDINGS = params['SAVE_EMBEDDINGS']
        self.BENCHMARK_TEST_ON_UNDIRECTED = params['TEST_ON_UNDIRECTED']
        self.BENCHMARK_SPLIT_RATIOS = params['SPLIT_RATIOS']
        self.BENCHMARK_PCA_TARGET_DIM = params['PCA_TARGET_DIM']
        self.BENCHMARK_NE_LEARNING_RATE = params['NE_LEARNING_RATE']
        self.BENCHMARK_NE_BATCH_SIZE = params['NE_BATCH_SIZE']
        self.BENCHMARK_NE_EPOCHS = params['NE_EPOCHS']
        self.BENCHMARK_NE_EMBEDDING_DIM = params['NE_EMBEDDING_DIM']
        self.BENCHMARK_NE_WALK_LENGTH = params['NE_WALK_LENGTH']
        self.BENCHMARK_NE_CONTEXT_SIZE = params['NE_CONTEXT_SIZE']
        self.BENCHMARK_NE_WALKS_PER_NODE = params['NE_WALKS_PER_NODE']
        self.BENCHMARK_NE_NUM_NEGATIVE_SAMPLES = params['NE_NUM_NEGATIVE_SAMPLES']
        self.BENCHMARK_NE_CLASSIFIER_C = params['NE_CLASSIFIER_C']
        self.BENCHMARK_NE_CLASSIFIER_MAX_ITER = params['NE_CLASSIFIER_MAX_ITER']
        self.BENCHMARK_NE_CLASSIFIER_SOLVER = params['NE_CLASSIFIER_SOLVER']
        self.BENCHMARK_GNN_EPOCHS = params['EPOCHS']
        self.BENCHMARK_GNN_WEIGHT_DECAY = params['WEIGHT_DECAY']
        self.BENCHMARK_GNN_HIDDEN_CHANNELS = params['GNN_HIDDEN_CHANNELS']
        self.BENCHMARK_GNN_NUM_LAYERS = params['GNN_NUM_LAYERS']
        self.BENCHMARK_GNN_DROPOUT_RATE = params['GNN_DROPOUT_RATE']
        self.BENCHMARK_GAT_HEADS = params['GAT_HEADS']
        self.BENCHMARK_GAT_DROPOUT_RATE = params['GAT_DROPOUT_RATE']
        self.BENCHMARK_CHEBNET_K = params['CHEBNET_K']
        self.BENCHMARK_GNN_LEARNING_RATE = params['LEARNING_RATE']
        self.BENCHMARK_RGCN_NUM_RELATIONS = params['RGCN_NUM_RELATIONS']
        self.BENCHMARK_DIRECTGCN_HIDDEN_LAYER_DIMS = params['DIRECTGCN_HIDDEN_LAYER_DIMS']
        self.BENCHMARK_GNN_INIT_DIM = params['GNN_INIT_DIM']

    def _setup_data_sources(self):
        """Defines the data sources for automatic download using URLs from YAML config."""
        from urllib.parse import urlparse
        urls = self._config['data_urls']

        # --- DEFINITIVE FIX: Dynamically determine filename from URL ---
        # This makes the config robust to changes in the source filename (e.g., idmapping.dat vs idmapping_selected.tab)
        id_mapping_url = urls['UNIPROT_ID_MAPPING']
        id_mapping_filename_gz = Path(urlparse(id_mapping_url).path).name
        id_mapping_filename = id_mapping_filename_gz.replace('.gz', '')

        # Base sources that are always needed
        data_sources: Dict[str, Dict] = {
            "UNIPROT_SPROT_FASTA": {
                "url": urls['UNIPROT_SPROT_FASTA'],
                "type": "file", "path": self.DATA_SEQUENCES_DIR / "uniprot_sprot.fasta",
                "post_process": "ungzip", "checksum": None, "cacheable": True
            },
            "BIOGRID_INTERACTIONS": {
                "url": urls['BIOGRID_INTERACTIONS'],
                "type": "file", "path": self.DATA_GROUND_TRUTH_DIR / "BIOGRID-ALL-4.4.248.mitab.txt",
                "post_process": "unzip", "checksum": None, "cacheable": True
            },
            "PROTT5_MODEL": {
                "url": urls['PROTT5_MODEL'],
                "type": "file", "path": self.DATA_MODELS_DIR / "per-protein.h5",
                "post_process": None, "checksum": None, "cacheable": True
            }
        }

        # --- DEFINITIVE FIX: Conditionally add optional data sources ---
        # This prevents the pipeline from trying to download files that are not
        # specified in the config.yaml (e.g., if they are commented out).
        if urls.get('UNIREF_50_FASTA'):
            data_sources["UNIREF_50_FASTA"] = {
                "url": urls['UNIREF_50_FASTA'], "type": "file", "path": self.DATA_SEQUENCES_DIR / "uniref50.fasta",
                "post_process": "ungzip", "checksum": None, "cacheable": True
            }
        # --- DEFINITIVE FIX: Conditionally add the large ID mapping file as a data source ---
        if self.USE_CANONICAL_ID_MAPPING_FILE:
            print("  INFO: `USE_CANONICAL_ID_MAPPING_FILE` is true. The full UniProt ID mapping file will be downloaded.")
            data_sources["UNIPROT_ID_MAPPING"] = {
                "url": id_mapping_url,
                "type": "file", "path": self.DATA_MAPPINGS_DIR / id_mapping_filename,
                "post_process": "ungzip", "checksum": None, "cacheable": True
            }

        neg_urls = urls['NEG_INTERACTIONS']
        for i, url in enumerate(neg_urls):
            data_sources[f"NEG_INTERACTIONS_{i + 1}"] = {
                "url": url, "type": "file", "path": self.DATA_GROUND_TRUTH_DIR / f"neg_{i + 1}.mitab",
                "post_process": "ungzip", "checksum": None, "cacheable": True
            }

        for name in self.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            data_sources[f"BENCHMARK_{name.upper()}"] = {
                "type": "pyg_dataset", "name": name,
                "path": self.DATA_STANDARD_DATASETS_DIR / name, "cacheable": True
            }
        self.DATA_SOURCES = data_sources

    def _link_data_sources_to_attributes(self):
        """Dynamically creates key file path attributes."""
        self.PROTT5_MODEL_PATH = self.DATA_SOURCES['PROTT5_MODEL']['path']

        # --- DEFINITIVE FIX: Create a dedicated attribute for the raw mapping file path ---
        # This avoids hardcoding the filename in other parts of the application.
        if 'UNIPROT_ID_MAPPING' in self.DATA_SOURCES:
            self.ID_MAPPING_RAW_PATH = self.DATA_SOURCES['UNIPROT_ID_MAPPING']['path']

        # --- NEW: Define raw data paths for the processor ---
        # These attributes were being used by the DataProcessor but were never defined.
        self.BIOGRID_RAW_PATH = self.DATA_SOURCES['BIOGRID_INTERACTIONS']['path']
        self.NEG_INTERACTIONS_RAW_PATHS = [
            v['path'] for k, v in self.DATA_SOURCES.items() if k.startswith('NEG_INTERACTIONS')
        ]


        # --- REFACTOR: Use the configured FASTA file key to select the single file to process ---
        fasta_key = self.FASTA_FILE_TO_PROCESS
        if fasta_key in self.DATA_SOURCES and str(self.DATA_SOURCES[fasta_key]['path']).endswith(('.fasta', '.fa')):
            self.SEQUENCE_FILE_PATHS = [Path(self.DATA_SOURCES[fasta_key]['path'])]
        else:
            print(f"  - ❌ CONFIGURATION ERROR: The specified 'FASTA_FILE_TO_PROCESS' key '{fasta_key}' was not found in the defined DATA_SOURCES or is not a FASTA file. Aborting.")
            sys.exit(1)

        # This attribute is used by the UI manager for downsampling logic.
        self.ORIGINAL_SEQUENCE_FILE_PATHS = self.SEQUENCE_FILE_PATHS.copy()

        self.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE = [
            {"name": "ProtT5", "path": self.PROTT5_MODEL_PATH},
        ]
        # --- NEW: Define dependencies for smart data restoration ---
        # This maps a processed file to the raw source file(s) it replaces.
        # The key is the name of the processed file, the value is a list of raw file names.
        self.PROCESSED_FILE_DEPENDENCIES = {
            self.POS_INTERACTIONS_PATH.name: {
                "dependencies": [Path(self.DATA_SOURCES['BIOGRID_INTERACTIONS']['path']).name],
                "destination_dir_attr": "DATA_GROUND_TRUTH_DIR"
            },
            self.NEG_INTERACTIONS_PATH.name: {
                "dependencies": [v['path'].name for k, v in self.DATA_SOURCES.items() if k.startswith('NEG_INTERACTIONS')],
                "destination_dir_attr": "DATA_GROUND_TRUTH_DIR"
            }
        }
        # Conditionally add the ID mapping dependency
        if self.USE_CANONICAL_ID_MAPPING_FILE:
            self.PROCESSED_FILE_DEPENDENCIES[self.ID_MAPPING_PATH.name] = {
                "dependencies": [Path(self.DATA_SOURCES['UNIPROT_ID_MAPPING']['path']).name],
                "destination_dir_attr": "DATA_MAPPINGS_DIR"
            }

    def _setup_gcn_params(self):
        """Sets ProtGram-GCN parameters statically from the YAML config."""
        params = self._config['protgram_gcn']
        cpu_cores = os.cpu_count()
        self.PROTGRAM_NGRAM_MAX_N = params['PROTGRAM_NGRAM_MAX_N']
        self.FASTA_FILE_TO_PROCESS = params['FASTA_FILE_TO_PROCESS']
        self.DASK_N_PARTITIONS = os.cpu_count() or 1
        self.GRAPH_BUILDER_WORKERS: Optional[int] = max(1, cpu_cores - 1) if cpu_cores is not None else 1
        self.ID_MAPPING_MODE = params['ID_MAPPING_MODE']
        self.USE_CANONICAL_ID_MAPPING_FILE = params['USE_CANONICAL_ID_MAPPING_FILE']
        self.REGEX_CONFIDENCE_THRESHOLD = params['REGEX_CONFIDENCE_THRESHOLD']
        self.REGEX_COMPATIBILITY_SAMPLE_SIZE = params['REGEX_COMPATIBILITY_SAMPLE_SIZE']
        self.API_MAPPING_FROM_DB = params['API_MAPPING_FROM_DB']
        self.API_MAPPING_TO_DB = params['API_MAPPING_TO_DB']
        self.PROTGRAM_MODELS_TO_TRAIN = params['PROTGRAM_MODELS_TO_TRAIN']
        self.DIRECTGCN_HIDDEN_LAYER_DIMS = params['DIRECTGCN_HIDDEN_LAYER_DIMS']
        self.PROTGRAM_1GRAM_INIT_DIM = params['PROTGRAM_1GRAM_INIT_DIM']
        self.PROTGRAM_GNN_HIDDEN_CHANNELS = params['PROTGRAM_GNN_HIDDEN_CHANNELS']
        self.PROTGRAM_GNN_NUM_LAYERS = params['PROTGRAM_GNN_NUM_LAYERS']
        self.PROTGRAM_GATING_COEFF_MODE = params['PROTGRAM_GATING_COEFF_MODE']
        self.PROTGRAM_EPOCHS_PER_LEVEL = params['PROTGRAM_EPOCHS_PER_LEVEL']
        self.PROTGRAM_LR = params['PROTGRAM_LR']
        self.PROTGRAM_DROPOUT_RATE = params['PROTGRAM_DROPOUT_RATE']
        self.PROTGRAM_WEIGHT_DECAY = params['PROTGRAM_WEIGHT_DECAY']
        self.PROTGRAM_USE_LR_SCHEDULER = params['PROTGRAM_USE_LR_SCHEDULER']
        self.PROTGRAM_LR_SCHEDULER_PATIENCE = params['PROTGRAM_LR_SCHEDULER_PATIENCE']
        self.PROTGRAM_LR_SCHEDULER_FACTOR = params['PROTGRAM_LR_SCHEDULER_FACTOR']
        self.PROTGRAM_USE_EARLY_STOPPING = params['PROTGRAM_USE_EARLY_STOPPING']
        self.PROTGRAM_EARLY_STOPPING_PATIENCE = params['PROTGRAM_EARLY_STOPPING_PATIENCE']
        self.PROTGRAM_EARLY_STOPPING_MIN_DELTA = params['PROTGRAM_EARLY_STOPPING_MIN_DELTA']
        self.PROTGRAM_GRADIENT_ACCUMULATION_STEPS = params['PROTGRAM_GRADIENT_ACCUMULATION_STEPS']
        self.PROTGRAM_TASK_TYPES_PER_LEVEL = params['PROTGRAM_TASK_TYPES_PER_LEVEL']
        self.PROTGRAM_DEFAULT_TASK_TYPE = params['PROTGRAM_DEFAULT_TASK_TYPE']
        self.GCN_HETEROPHILY_THRESHOLD = params['GCN_HETEROPHILY_THRESHOLD']
        self.GCN_PROPAGATION_EPSILON = params['GCN_PROPAGATION_EPSILON']
        self.PROTGRAM_CLOSEST_AA_K_HOPS = params['PROTGRAM_CLOSEST_AA_K_HOPS']
        self.PROTGRAM_MASKED_NODE_FRACTION = params['PROTGRAM_MASKED_NODE_FRACTION']
        self.PROTGRAM_CLEAN_FASTA_ON_PARSE = params['PROTGRAM_CLEAN_FASTA_ON_PARSE']
        self.PROTGRAM_FASTA_MIN_LEN = params['PROTGRAM_FASTA_MIN_LEN']
        self.PROTGRAM_FASTA_MAX_LEN = params['PROTGRAM_FASTA_MAX_LEN']
        self.PROTGRAM_FASTA_ALPHABET = params['PROTGRAM_FASTA_ALPHABET']
        self.PROTGRAM_USE_CLUSTER_TRAINING = params['PROTGRAM_USE_CLUSTER_TRAINING']
        self.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES = params['PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES']
        self.PROTGRAM_PARTITIONING_METHOD = params['PROTGRAM_PARTITIONING_METHOD']
        self.PROTGRAM_COARSENING_LEVEL_FOR_PARTITIONING = params['PROTGRAM_COARSENING_LEVEL_FOR_PARTITIONING']
        self.PROTGRAM_VALIDATE_COARSENING = params['PROTGRAM_VALIDATE_COARSENING']
        self.PROTGRAM_TARGET_NODES_PER_CLUSTER = params['PROTGRAM_TARGET_NODES_PER_CLUSTER']
        self.PROTGRAM_MIN_CLUSTERS = params['PROTGRAM_MIN_CLUSTERS']
        self.PROTGRAM_MAX_CLUSTERS = params['PROTGRAM_MAX_CLUSTERS']
        self.PROTGRAM_CLUSTER_GROUP_SIZE = params['PROTGRAM_CLUSTER_GROUP_SIZE']
        self.POOLING_WORKERS: Optional[int] = max(1, cpu_cores - 1) if cpu_cores is not None else 1
        self.PCA_TARGET_DIMENSION = params['PCA_TARGET_DIMENSION']
        self.PROTGRAM_PROTEIN_POOLING_STRATEGY = params['PROTGRAM_PROTEIN_POOLING_STRATEGY']
        self.PROTGRAM_HIERARCHICAL_POOLING_STRATEGY = params['PROTGRAM_HIERARCHICAL_POOLING_STRATEGY']
        self.PROTGRAM_LOG_ATTENTION_WEIGHTS = params['PROTGRAM_LOG_ATTENTION_WEIGHTS']
        self.PROTGRAM_RUN_SANITY_CHECK_PPI = params['PROTGRAM_RUN_SANITY_CHECK_PPI']
        self.PROTGRAM_SANITY_CHECK_EPOCHS = params['PROTGRAM_SANITY_CHECK_EPOCHS']
        self.PROTGRAM_SANITY_CHECK_TEST_SPLIT = params['PROTGRAM_SANITY_CHECK_TEST_SPLIT']
        self.PROTGRAM_SANITY_CHECK_SAMPLE_SIZE = params['PROTGRAM_SANITY_CHECK_SAMPLE_SIZE']

    def _setup_word2vec_params(self):
        """Sets Word2Vec parameters statically from the YAML config."""
        params = self._config['word2vec']
        cpu_cores = os.cpu_count()
        self.W2V_VECTOR_SIZE = params['W2V_VECTOR_SIZE']
        self.W2V_WINDOW = params['W2V_WINDOW']
        self.W2V_MIN_COUNT = params['W2V_MIN_COUNT']
        self.W2V_EPOCHS = params['W2V_EPOCHS']
        self.W2V_WORKERS: Optional[int] = max(1, cpu_cores - 4) if cpu_cores is not None else 1
        self.W2V_POOLING_STRATEGY = params['W2V_POOLING_STRATEGY']
        self.APPLY_PCA_TO_W2V = params['APPLY_PCA_TO_W2V']

    def _setup_transformer_params(self):
        """Sets Transformer parameters statically from the YAML config."""
        params = self._config['transformer']
        self.TRANSFORMER_MODELS_TO_RUN = params['MODELS_TO_RUN']
        self.TRANSFORMER_MAX_LENGTH = params['MAX_LENGTH']
        self.TRANSFORMER_BASE_BATCH_SIZE = params['BASE_BATCH_SIZE']
        self.TRANSFORMER_CHUNK_SIZE = params['CHUNK_SIZE']
        self.TRANSFORMER_POOLING_STRATEGY = params['POOLING_STRATEGY']
        self.USE_XLA_COMPILATION = params['USE_XLA_COMPILATION']
        self.TRANSFORMER_INFERENCE_SAMPLE_FRACTION = params['TRANSFORMER_INFERENCE_SAMPLE_FRACTION']

    def _setup_lstm_params(self):
        """Sets LSTM parameters statically from the YAML config."""
        params = self._config['lstm']
        self.LSTM_EMBEDDING_DIM = params['EMBEDDING_DIM']
        self.LSTM_HIDDEN_DIM = params['HIDDEN_DIM']
        self.LSTM_NUM_LAYERS = params['NUM_LAYERS']
        self.LSTM_EPOCHS = params['EPOCHS']
        self.LSTM_BATCH_SIZE = params['BATCH_SIZE']
        self.LSTM_TRAIN_SEQ_LEN = params['TRAIN_SEQ_LEN']
        self.LSTM_TRAIN_STEP = params['TRAIN_STEP']
        self.LSTM_LEARNING_RATE = params['LEARNING_RATE']
        self.LSTM_POOLING_STRATEGY = params['POOLING_STRATEGY']
        self.LSTM_VALIDATION_SPLIT = params['VALIDATION_SPLIT']
        self.LSTM_USE_EARLY_STOPPING = params['USE_EARLY_STOPPING']
        self.LSTM_EARLY_STOPPING_PATIENCE = params['EARLY_STOPPING_PATIENCE']
        self.LSTM_DROPOUT_RATE = params['DROPOUT_RATE']
        self.LSTM_EARLY_STOPPING_MIN_DELTA = params['EARLY_STOPPING_MIN_DELTA']

    def _setup_singleton_eval_params(self):
        """Sets Singleton evaluation parameters statically from the YAML config."""
        params = self._config['singleton_eval']
        self.SINGLETON_EVAL_EPOCHS = params['EPOCHS'] # Keep this for clarity if needed elsewhere
        self.SINGLETON_EVAL_TEST_SPLIT = params['TEST_SPLIT']
        self.SINGLETON_EVAL_LR = params['LEARNING_RATE']
        self.SINGLETON_EVAL_MODELS_TO_RUN = params['MODELS_TO_RUN']
        self.SINGLETON_GNN_HIDDEN_CHANNELS = params['GNN_HIDDEN_CHANNELS']
        self.SINGLETON_GNN_NUM_LAYERS = params['GNN_NUM_LAYERS']
        self.SINGLETON_GNN_DROPOUT_RATE = params['GNN_DROPOUT_RATE']
        self.SINGLETON_GAT_HEADS = params['GAT_HEADS']
        self.SINGLETON_GAT_DROPOUT_RATE = params['GAT_DROPOUT_RATE']
        self.SINGLETON_CHEBNET_K = params['CHEBNET_K']
        self.SINGLETON_RGCN_NUM_RELATIONS = params['RGCN_NUM_RELATIONS']
        self.SINGLETON_EVAL_GRADIENT_ACCUMULATION_STEPS = params['GRADIENT_ACCUMULATION_STEPS']
        self.SINGLETON_DIRECTGCN_HIDDEN_LAYER_DIMS = params['DIRECTGCN_HIDDEN_LAYER_DIMS']

    def _setup_evaluation_params(self):
        """Sets PPI evaluation parameters statically from the YAML config."""
        params = self._config['ppi_evaluation']
        self.PLOT_TRAINING_HISTORY = params['PLOT_TRAINING_HISTORY']
        self.PERFORM_H5_INTEGRITY_CHECK = params['PERFORM_H5_INTEGRITY_CHECK']
        self.EVAL_GENERATE_SHAP_SUMMARY = params['EVAL_GENERATE_SHAP_SUMMARY']
        self.EARLY_STOPPING_PATIENCE = params['EARLY_STOPPING_PATIENCE']
        self.EVAL_EDGE_EMBEDDING_METHOD = params['EDGE_EMBEDDING_METHOD']
        self.EVAL_N_FOLDS = params['N_FOLDS']
        self.EVAL_MLP_DENSE1_UNITS = params['MLP_DENSE1_UNITS']
        self.EVAL_MLP_DROPOUT1_RATE = params['MLP_DROPOUT1_RATE']
        self.EVAL_MLP_DENSE2_UNITS = params['MLP_DENSE2_UNITS']
        self.EVAL_MLP_DROPOUT2_RATE = params['MLP_DROPOUT2_RATE']
        self.EVAL_MLP_L2_REG = params['MLP_L2_REG']
        self.EVAL_BATCH_SIZE = params['BATCH_SIZE']
        self.EVAL_EPOCHS = params['EPOCHS']
        self.EVAL_MLP_LEARNING_RATE = params['MLP_LEARNING_RATE']
        self.EVAL_K_VALUES_FOR_TABLE = params['K_VALUES_FOR_TABLE']
        self.EVAL_MAIN_EMBEDDING_FOR_STATS = params['MAIN_EMBEDDING_FOR_STATS']
        self.EVAL_STATISTICAL_TEST_ALPHA = params['STATISTICAL_TEST_ALPHA']

    def _setup_mlflow_params(self):
        """Sets MLflow parameters statically from the YAML config."""
        params = self._config['mlflow']
        self.USE_MLFLOW = params['USE_MLFLOW']
        mlruns_path = self.BASE_OUTPUT_DIR / "mlruns"
        self.MLFLOW_TRACKING_URI = mlruns_path.as_uri()
        self.MLFLOW_EXPERIMENT_NAME = params['EXPERIMENT_NAME']
        self.MLFLOW_BENCHMARK_EXPERIMENT_NAME = params['BENCHMARK_EXPERIMENT_NAME']
        self.MLFLOW_NE_BENCHMARK_EXPERIMENT_NAME = self.MLFLOW_BENCHMARK_EXPERIMENT_NAME
        self.MLFLOW_LLMS_EXPERIMENT_NAME = params['LLMS_EXPERIMENT_NAME']
        self.MLFLOW_PROTGRAM_XGCN_EXPERIMENT_NAME = params['PROTGRAM_XGCN_EXPERIMENT_NAME']
        self.MLFLOW_INTERPRETABILITY_EXPERIMENT_NAME = params['INTERPRETABILITY_EXPERIMENT_NAME']

    def _setup_hpo_params(self):
        """Sets HPO parameters statically from the YAML config."""
        # Defaults are handled by the Pydantic schema. If the section is missing,
        # an empty object with defaults is created during validation.
        params = self._config['hyperparameter_optimization']
        self.RUN_HPO = params['RUN_HPO']
        self.HPO_N_TRIALS = params['HPO_N_TRIALS']
        self.HPO_TARGET_EMBEDDING_MODEL = params['HPO_TARGET_EMBEDDING_MODEL']
        self.HPO_PPI_MLP_SEARCH_SPACE = params['HPO_PPI_MLP_SEARCH_SPACE']

    def _validate_config(self):
        """
        Performs validation of the entire configuration using Pydantic schemas.
        This provides clear, structured error messages if the config is invalid.
        """
        print("--- Validating configuration parameters using Pydantic schema... ---")
        try:
            self._config = ValidationSchema.model_validate(self._config).model_dump()
            print("  ✅ Configuration is valid.")
        except ValidationError as e:
            # --- DEFINITIVE FIX: Provide clear, actionable error messages and exit ---
            print("\n" + "="*80)
            print("--- ❌ CONFIGURATION ERROR ---")
            print("  Your 'config.yaml' file has one or more errors:")
            # Pydantic provides a nicely formatted error message.
            print(e)
            print("="*80)
            print("\n--- Please correct the configuration file and try again. ---")
            sys.exit(1)  # Exit with an error code
