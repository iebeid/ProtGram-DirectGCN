# ==============================================================================
# MODULE: source/entry/main.py
# PURPOSE: Main pipeline entry point and orchestrator.
# VERSION: 12.1 (Bug Fixes Applied)
# AUTHOR: Islam Ebeid
# ==============================================================================

import copy
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

# --- Add project root to sys.path to allow for relative imports ---
# This must be done BEFORE any local modules (like 'configuration') are imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from typing import List, Dict
import pandas as pd
from source.utils.post.embedding_processor import EmbeddingProcessor
import mlflow
import tensorflow as tf
# --- Local Application Imports ---
from configuration.config import Config
from source.benchmarkers.gnns import GNNBenchmarker
from source.data_builders.protgram import ProtGramDataBuilder
from configuration.manager import DataManager
from source.experiments.ppi_1 import PPIPipeline
from source.testers.unit_tests import run_all_tests
from source.trainers.lstm import LSTMBasedEmbedder
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.models.hyperparameter_optimizer import HyperparameterOptimizer
from source.utils.data.data_utils import DataUtils
from source.entry.checkpoints import CheckpointManager
from source.entry.ui import UIManager


class PipelineOrchestrator:
    """Orchestrates the entire data processing and model training pipeline."""

    def __init__(self, config: Config):
        self.project_root = Path(__file__).resolve().parents[2]
        self.base_config = config
        self.ui_manager = UIManager()

        self._configure_gpu()
        DataUtils.set_seeds(self.base_config.RANDOM_STATE)

    def _configure_gpu(self):
        """Configures GPU settings for TensorFlow."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            from tf_keras import mixed_precision

            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Warning: Could not set memory growth for GPUs: {e}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('WARNING')

    def _run_protgram_xgcn_trainer(self, config: Config):
        """
        Helper method to run ProtGramXGCNTrainer with a check for existing graph objects.
        """
        if not config.RUN_PROTGRAM_PIPELINE and not config.RESULTS_GRAPH_OBJECTS_DIR.exists():
            print(f"  Warning: Skipping ProtGram-XGCN pipeline. Graph objects not found in {config.RESULTS_GRAPH_OBJECTS_DIR}.")
            return None

        return ProtGramXGCNTrainer(config).run()

    def _run_main_embedding_pipelines(self, config: Config, checkpoint_manager: CheckpointManager) -> List[Dict[str, str]]:
        """Runs the main embedding generation pipelines with checkpointing and user prompts."""
        pipelines = [
            {"name": "ProtGram-XGCN", "flag": "RUN_PROTGRAM_XGCN_PIPELINE", "runner": lambda: self._run_protgram_xgcn_trainer(config),
             "formatter": lambda paths: [{"name": name, "path": path} for name, path in paths.items()]},
            {"name": "Word2Vec", "flag": "RUN_WORD2VEC_PIPELINE", "runner": lambda: Word2VecEmbedder(config).run(),
             "formatter": lambda paths: [{"name": name, "path": path} for name, path in paths.items()]},
            {"name": "LSTM", "flag": "RUN_LSTM_PIPELINE", "runner": lambda: LSTMBasedEmbedder(config).run(),
             "formatter": lambda path: [{"name": "LSTM-Generated", "path": str(path)}]},
            {"name": "Transformer", "flag": "RUN_TRANSFORMER_PIPELINE", "runner": lambda: TransformerEmbedder(config).run(),
             "formatter": lambda paths: [{"name": f"{name}-Generated", "path": str(path)} for name, path in paths.items()]}
        ]

        generated_files = []
        for p_config in pipelines:
            if getattr(config, p_config["flag"], False):
                DataUtils.print_header(f"Preparing Pipeline: {p_config['name']}")
                checkpoint_data = checkpoint_manager.get_checkpoint(p_config['name'])
                if checkpoint_data:
                    generated_files.extend(checkpoint_data)
                    continue
                try:
                    raw_result = p_config["runner"]()
                    if raw_result:
                        formatted_raw_results = p_config["formatter"](raw_result)

                        if config.USE_CANONICAL_ID_MAPPING_FILE:
                            print("  Standardizing generated embeddings using canonical ID map...")
                            standardized_files = []
                            for file_info in formatted_raw_results:
                                standardized_path = EmbeddingProcessor.standardize_embedding_file(file_info["path"])
                                standardized_files.append({"name": file_info["name"], "path": standardized_path})
                        else:
                            print("  Skipping standardization, assuming generated embeddings use correct IDs.")
                            standardized_files = formatted_raw_results

                        checkpoint_manager.save_checkpoint(p_config['name'], standardized_files)
                        generated_files.extend(standardized_files)

                        if not self.ui_manager.prompt_to_continue(p_config['name'], config):
                            sys.exit(0)
                    else:
                        print(f"  Pipeline '{p_config['name']}' did not produce any output files.")
                except Exception as e:
                    print(f"\n--- ❌ ERROR in pipeline: {p_config['name']} ---")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"--- Skipping pipeline {p_config['name']} and continuing. ---")

        return generated_files

    def _discover_existing_embeddings(self, config: Config, existing_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Scans output directories for pre-existing embeddings and adds them to the evaluation list."""
        print("  Scanning for existing, pre-generated embeddings...")
        discovered_files = []
        existing_paths = {f['path'] for f in existing_files}

        embedding_dirs = [
            config.RESULTS_GCN_EMBEDDINGS_DIR,
            config.RESULTS_W2V_EMBEDDINGS_DIR,
            config.RESULTS_LSTM_EMBEDDINGS_DIR,
            config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR,
            config.RESULTS_BENCHMARK_EMBEDDINGS_DIR  # Include benchmark-saved embeddings (e.g., DirectGCN)
        ]

        for emb_dir in embedding_dirs:
            if emb_dir.exists():
                for h5_file in emb_dir.glob('**/*.h5'):
                    if str(h5_file) not in existing_paths:
                        model_name = h5_file.stem.replace('_standardized', '')
                        print(f"    Discovered pre-existing embedding: {model_name}")
                        discovered_files.append({"name": model_name, "path": str(h5_file)})
        return discovered_files

    def _run_singleton_gcn_eval_for_n(self, n: int, config: Config, fasta_file_path: Path) -> pd.DataFrame:
        """Helper to run the singleton GCN evaluation for a single n-gram level."""
        from source.data_structures.direct_ngram_graph import DirectedNgramGraph
        from source.trainers.singleton_xgcn import SingletonXGCNTrainer

        DataUtils.print_header(f"Running Singleton (n={n}) GCN Evaluation")
        n_graph_dir = config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}"
        if not (n_graph_dir.exists() and n_graph_dir.is_dir()):
            print(f"  Warning: n={n} graph not found at {n_graph_dir}. It may not have been built. Skipping singleton evaluation.")
            return pd.DataFrame()

        print(f"  n={n} graph found. Proceeding...")
        n_graph: DirectedNgramGraph = DirectedNgramGraph.load_from_dir(n_graph_dir)
        if not n_graph:
            return pd.DataFrame()

        singleton_results_df = SingletonXGCNTrainer(config, n_graph).run()
        if singleton_results_df is not None and not singleton_results_df.empty:
            singleton_results_df = singleton_results_df.rename(columns={'Model': 'model'})
            singleton_results_df['dataset'] = f"ProtGram_n{n}_Singleton_{fasta_file_path.stem}"
            singleton_results_df['error'] = None
            return singleton_results_df
        return pd.DataFrame()

    def _run_pre_analysis_and_prompt(self, config: Config, fasta_file_path: Path) -> bool:
        """Runs preliminary benchmarks, displays a summary, and prompts the user to continue."""
        all_benchmark_results = []

        try:
            if config.RUN_BENCHMARKING_PIPELINE:
                with mlflow.start_run(run_name="Node_Classification_Benchmark_Suite"):
                    benchmark_results_df = GNNBenchmarker(config).run()
                    if benchmark_results_df is not None and not benchmark_results_df.empty:
                        all_benchmark_results.append(benchmark_results_df)
        except Exception as e:
            print(f"\n--- GNN BENCHMARKING FAILED: {e} ---\n")
            import traceback
            traceback.print_exc()

        try:
            if config.RUN_SINGLETON_GCN_EVAL:
                # Limit Singleton evaluation to n=1 as a lightweight smoke test
                for n in [1]:
                    singleton_results = self._run_singleton_gcn_eval_for_n(n, config, fasta_file_path)
                    if not singleton_results.empty:
                        all_benchmark_results.append(singleton_results)
        except Exception as e:
            print(f"\n--- SINGLETON GCN EVALUATION FAILED: {e} ---\n")
            import traceback
            traceback.print_exc()

        self.ui_manager.display_aggregated_benchmark_summary(all_benchmark_results)

        print("\n" + "#" * 80)
        print("### PRELIMINARY ANALYSIS COMPLETE ###")
        print("#" * 80)
        return self.ui_manager.prompt_to_continue("Preliminary Analysis", config)

    def run(self):
        """Executes the entire pipeline."""
        script_start_time = time.monotonic()
        try:
            DataUtils.print_header("Starting Protein-Protein Interaction Meta-Pipeline")

            data_manager = DataManager(self.base_config)
            print("\n--- Verifying local data integrity ---")
            if not data_manager.is_setup_complete():
                print("\n--- Local data is invalid or incomplete. Attempting to restore from cache... ---")
                if data_manager.restore_data_from_cache():
                    print("  - ✅ Successfully restored source data from cache.")
                else:
                    print("  - ℹ️ Could not restore from cache. Will proceed with full download.")

                print("\n--- Data setup is incomplete. Running full setup to generate derived files... ---")
                data_manager.run_full_setup()

                if not data_manager.is_setup_complete():
                    print("FATAL: Data validation failed even after a full setup. Please check logs.")
                    sys.exit(1)
            else:
                print("  - ✅ Local data is valid.")

            # Unified TEMP directories for this run
            self.base_config.TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            self.base_config.TEMP_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
            self.base_config.TEMP_TESTS_DIR.mkdir(parents=True, exist_ok=True)

            # Set Dask temp inside unified TEMP_PIPELINE_DIR
            dask_temp_dir = self.base_config.TEMP_PIPELINE_DIR / "dask"
            dask_temp_dir.mkdir(exist_ok=True, parents=True)
            os.environ['DASK_TEMPORARY_DIRECTORY'] = str(dask_temp_dir)
            print(f"  - Dask temporary directory set to: {dask_temp_dir}")

            # Ensure all temporary files live under the unified pipeline TEMP directory
            os.environ['TMPDIR'] = str(self.base_config.TEMP_PIPELINE_DIR)
            tempfile.tempdir = str(self.base_config.TEMP_PIPELINE_DIR)

            # Route HuggingFace/Transformers cache under results to avoid /tmp usage
            hf_cache = self.base_config.BASE_OUTPUT_DIR / "hf_cache"
            hf_cache.mkdir(exist_ok=True, parents=True)
            os.environ['TRANSFORMERS_CACHE'] = str(hf_cache)
            os.environ['HF_HOME'] = str(hf_cache)

            DataUtils.report_system_resources(self.base_config.BASE_OUTPUT_DIR)

            if self.base_config.DEBUG_VERBOSE:
                print("--- Full Configuration Values ---")
                for key, value in sorted(vars(self.base_config).items()):
                    if not key.startswith('_'): print(f"  {key:<40} | {value}")
                print("--------------------------")

            if self.base_config.USE_MLFLOW:
                mlflow.set_tracking_uri(self.base_config.MLFLOW_TRACKING_URI)

            if self.base_config.RUN_INTEGRATED_TESTS:
                DataUtils.print_header("Running Integrated Test Suite")
                gpu_is_ok = run_all_tests(config=self.base_config)
                DataUtils.print_header("Integrated Test Suite Finished.")
                if not gpu_is_ok:
                    print("\n" + "!" * 80)
                    print("!!! WARNING: GPU verification failed. Pipeline will run on CPU. !!!")
                    print("!" * 80 + "\n")
                    if sys.stdin.isatty():
                        response = input("Continue with CPU-only execution? (y/n): ").lower().strip()
                        if response not in ['y', 'yes']: sys.exit(0)
                if not self.ui_manager.prompt_to_continue("Integrated Tests", self.base_config): sys.exit(0)

            # Use unified pipeline TEMP directory for transient dataset files
            temp_dir_base = self.base_config.TEMP_PIPELINE_DIR
            temp_dir_base.mkdir(exist_ok=True, parents=True)
            with tempfile.TemporaryDirectory(dir=temp_dir_base) as temp_dir:
                files_to_process = self.ui_manager.get_fasta_files_to_process(self.base_config, Path(temp_dir))
                if not files_to_process:
                    print("\nERROR: No sequence files defined. Cannot run experiments.")
                    return

                if self.base_config.SEQUENCE_DOWNSAMPLE_FRACTION is not None and self.base_config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0:
                    DataUtils.print_header(f"FASTA Downsampling Active ({self.base_config.SEQUENCE_DOWNSAMPLE_FRACTION * 100:.1f}%)")
                else:
                    DataUtils.print_header("FASTA Downsampling Inactive (Processing Full Files)")

                if self.base_config.RUN_TRANSFORMER_PIPELINE:
                    DataUtils.print_header("Verifying Transformer Model Availability")
                    from huggingface_hub import model_info
                    for model_cfg in self.base_config.TRANSFORMER_MODELS_TO_RUN:
                        model_id = model_cfg['hf_id']
                        local_path = self.base_config.DATA_MODELS_DIR / model_id
                        if not (local_path.exists() and (local_path / "tf_model.h5").exists()):
                            try:
                                info = model_info(model_id)
                                is_pytorch_only = "tensorflow" not in info.tags and "tf" not in info.tags
                                if is_pytorch_only:
                                    print(f"\n--- ACTION: Model '{model_id}' requires conversion. ---")
                                    module_path = "source.utils.models.model_converter"
                                    subprocess.run([sys.executable, "-m", module_path, model_id], check=True)
                            except Exception as e:
                                print(f"  Warning: Could not verify or convert model '{model_id}': {e}")

                for fasta_file_path in files_to_process:
                    try:
                        # Use a shallow copy to avoid re-running Config.__init__ and creating new directories
                        config = copy.copy(self.base_config)
                        dataset_name = fasta_file_path.stem
                        DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")

                        # Ensure ProtGram settings (including PROTGRAM_NGRAM_MAX_N) are restored from YAML
                        # to avoid leaked overrides from earlier steps/tests.
                        if hasattr(config, "_setup_gcn_params"):
                            config._setup_gcn_params()

                        # BUG FIX: Ensure the correct (downsampled) file path is used.
                        # Overwrite both variables to be safe in case protgram.py uses the wrong one.
                        config.SEQUENCE_FILE_PATHS = [fasta_file_path]
                        config.ORIGINAL_SEQUENCE_FILE_PATHS = [fasta_file_path]

                        paths_to_specialize = [
                            'RESULTS_GRAPH_OBJECTS_DIR', 'RESULTS_GCN_EMBEDDINGS_DIR',
                            'RESULTS_W2V_EMBEDDINGS_DIR', 'RESULTS_LSTM_EMBEDDINGS_DIR',
                            'RESULTS_TRANSFORMER_EMBEDDINGS_DIR', 'RESULTS_EVALUATION_DIR'
                        ]
                        for path_attr in paths_to_specialize:
                            if hasattr(config, path_attr):
                                original_path = getattr(self.base_config, path_attr)
                                setattr(config, path_attr, original_path / dataset_name)

                        checkpoint_dir_uri = os.path.join(str(config.BASE_OUTPUT_DIR), "checkpoints", dataset_name)
                        checkpoint_manager = CheckpointManager(checkpoint_dir_uri)

                        # Graph Building Stage
                        if config.RUN_PROTGRAM_PIPELINE:
                            if not checkpoint_manager.get_checkpoint("GraphBuilding"):
                                # Offer reuse/copy option before rebuilding
                                expected_graphs_dir = config.RESULTS_GRAPH_OBJECTS_DIR
                                # If interactive prompts are enabled, ask the user
                                user_decision = self.ui_manager.prompt_graph_objects_reuse_or_rebuild(
                                    config=config,
                                    dataset_name=dataset_name,
                                    expected_output_dir=expected_graphs_dir,
                                    n_max=config.PROTGRAM_NGRAM_MAX_N
                                )
                                proceed_to_build = True
                                if user_decision.get('choice') == 'manual':
                                    # Verify existence after manual copy
                                    all_present = all((expected_graphs_dir / f"ngram_graph_n{i}").exists()
                                                      for i in range(1, config.PROTGRAM_NGRAM_MAX_N + 1))
                                    if all_present:
                                        print("  ✅ Detected manually copied graph objects. Skipping rebuild.")
                                        checkpoint_manager.save_checkpoint("GraphBuilding", {"status": "reused"})
                                        proceed_to_build = False
                                    else:
                                        print("  ❌ Expected graph objects not found after manual copy. Proceeding to rebuild.")
                                elif user_decision.get('choice') == 'auto_copy' and user_decision.get('source'):
                                    src_path = user_decision['source']
                                    print(f"  Attempting to copy graph objects from: {src_path}")
                                    success = self.ui_manager.copy_graph_objects_from_source(
                                        source_root=Path(src_path),
                                        dest_dataset_dir=expected_graphs_dir,
                                        n_max=config.PROTGRAM_NGRAM_MAX_N
                                    )
                                    if success:
                                        print("  ✅ Graph objects copied successfully. Skipping rebuild.")
                                        checkpoint_manager.save_checkpoint("GraphBuilding", {"status": "copied"})
                                        proceed_to_build = False
                                    else:
                                        print("  ❌ Copy failed or incomplete. Proceeding to rebuild.")
                                # Rebuild if needed
                                if proceed_to_build:
                                    try:
                                        DataUtils.print_header("Building all n-gram graphs for the main pipeline")
                                        ProtGramDataBuilder(config).run()
                                        checkpoint_manager.save_checkpoint("GraphBuilding", {"status": "completed"})
                                    except KeyboardInterrupt:
                                        print("\n--- Received interrupt during graph build. Cleanup has run. Exiting gracefully. ---")
                                        return
                            if not self.ui_manager.prompt_to_continue("Graph Building", config): continue
                        else:
                            print("  INFO: Skipping graph building as RUN_PROTGRAM_PIPELINE is false.")
                            if not self.ui_manager.prompt_to_continue("Graph Building (skipped)", config): continue

                        # Pre-analysis/benchmarking step
                        if not checkpoint_manager.get_checkpoint("PreAnalysis"):
                            if not self._run_pre_analysis_and_prompt(config, fasta_file_path):
                                continue
                            checkpoint_manager.save_checkpoint("PreAnalysis", {"status": "completed"})

                        # Optional: Hyperparameter tuning for ProtGram-XGCN before training
                        if config.RUN_HPO and config.RUN_PROTGRAM_XGCN_PIPELINE:
                            DataUtils.print_header("Running HPO for ProtGram-XGCN")
                            optimizer = HyperparameterOptimizer(config)
                            best_xgcn_params = optimizer.optimize_protgram_xgcn()
                            if isinstance(best_xgcn_params, dict) and best_xgcn_params:
                                # Apply best params to runtime config
                                for k, v in best_xgcn_params.items():
                                    if hasattr(config, k):
                                        setattr(config, k, v)
                                print("  Applied best ProtGram-XGCN hyperparameters to config.")

                        # Main embedding generation
                        generated_files = self._run_main_embedding_pipelines(config, checkpoint_manager)
                        generated_files = generated_files if isinstance(generated_files, list) else []

                        # Discover existing embeddings
                        discovered_files = self._discover_existing_embeddings(config, generated_files)
                        generated_files.extend(discovered_files)
                        config.LP_EMBEDDING_FILES_TO_EVALUATE = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_files

                        # Hyperparameter Optimization
                        if config.RUN_HPO and config.RUN_PROTGRAM_XGCN_PIPELINE and config.RUN_MAIN_PPI_EVALUATION:
                            DataUtils.print_header("Running Hyperparameter Optimization")
                            optimizer = HyperparameterOptimizer(config)
                            target_model_name = config.HPO_TARGET_EMBEDDING_MODEL
                            target_embedding_path = None
                            for emb_file in config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                if target_model_name.lower() in emb_file['name'].lower():
                                    target_embedding_path = emb_file['path']
                                    break
                            if target_embedding_path:
                                best_params = optimizer.optimize_ppi_mlp(str(target_embedding_path), target_model_name)
                                # Apply best hyperparameters to the runtime Config so PPI uses them
                                if isinstance(best_params, dict) and best_params:
                                    mapping = {
                                        'MLP_LEARNING_RATE': 'EVAL_MLP_LEARNING_RATE',
                                        'DENSE1_UNITS': 'EVAL_MLP_DENSE1_UNITS',
                                        'DROPOUT1_RATE': 'EVAL_MLP_DROPOUT1_RATE',
                                        'DENSE2_UNITS': 'EVAL_MLP_DENSE2_UNITS',
                                        'DROPOUT2_RATE': 'EVAL_MLP_DROPOUT2_RATE',
                                        'L2_REG': 'EVAL_MLP_L2_REG',
                                        'BATCH_SIZE': 'EVAL_BATCH_SIZE',
                                    }
                                    for k, v in best_params.items():
                                        if k in mapping:
                                            setattr(config, mapping[k], v)
                                    print("  Applied best HPO hyperparameters to PPI config.")
                            else:
                                print(f"  Warning: HPO target embedding '{target_model_name}' not found in generated/configured files. Skipping HPO.")
                        else:
                            if config.RUN_HPO:
                                print("  Info: Skipping HPO. It requires both RUN_PROTGRAM_XGCN_PIPELINE and RUN_MAIN_PPI_EVALUATION to be enabled.")

                        # Main PPI Evaluation
                        if config.RUN_MAIN_PPI_EVALUATION:
                            DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
                            if not checkpoint_manager.get_checkpoint("PPI_Evaluation"):
                                if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                    ppi_evaluator = PPIPipeline(config)
                                    ppi_evaluator.run()
                                    checkpoint_manager.save_checkpoint("PPI_Evaluation", {"status": "completed"})
                    except Exception as e:
                        print(f"\n--- ❌ ERROR processing dataset: {dataset_name} ---")
                        print(f"  This dataset will be skipped. The pipeline will continue with the next one.")
                        print(f"  Error details: {e}")
                        import traceback
                        traceback.print_exc()

            DataUtils.print_header(f"Full Orchestration Finished in {time.monotonic() - script_start_time:.2f} seconds.")
            self.ui_manager.launch_mlflow_ui(self.base_config)

        except Exception as e:
            print(f"\n--- PIPELINE FAILED ---: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    config = Config()
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run()
