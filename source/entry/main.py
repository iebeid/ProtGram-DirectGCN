# ==============================================================================
# MODULE: source/entry/main.py
# PURPOSE: Main pipeline entry point and orchestrator.
# VERSION: 11.0 (Consolidated pre-analysis logic)
# AUTHOR: Islam Ebeid
# ==============================================================================

import copy
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import mlflow
import tensorflow as tf

# --- Local Application Imports ---
from configuration.config import Config
from source.benchmarkers.gnns import GNNBenchmarker
from configuration.manager import DataManager
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker
from source.experiments.ppi_1 import PPIPipeline
from source.testers.unit_tests import run_all_tests
from source.trainers.lstm import LSTMBasedEmbedder
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.trainers.hyperparameter_optimizer import HyperparameterOptimizer
from configuration.data_downloader import DataDownloader
from source.utils.data.data_utils import DataUtils
from source.utils.logging.file_logger import FileLogger
from .checkpoints import CheckpointManager
from .ui import UIManager


class PipelineOrchestrator:
    """Orchestrates the entire data processing and model training pipeline."""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.base_config = Config()
        self.ui_manager = UIManager()

        self._configure_gpu()
        DataUtils.set_seeds(self.base_config.RANDOM_STATE)

    def _configure_gpu(self):
        """Configures GPU settings for TensorFlow."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Warning: Could not set memory growth for GPUs: {e}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('WARNING')

    def _run_main_embedding_pipelines(self, config: Config, checkpoint_manager: CheckpointManager) -> List[Dict[str, str]]:
        """Runs the main embedding generation pipelines with checkpointing and user prompts."""
        pipelines = [
            {"name": "ProtGram-XGCN", "flag": "RUN_GCN_PIPELINE", "runner": lambda: ProtGramXGCNTrainer(config).run(),
             "formatter": lambda paths: [{"name": name, "path": path} for name, path in paths.items()]},
            {"name": "Word2Vec", "flag": "RUN_WORD2VEC_PIPELINE", "runner": lambda: Word2VecEmbedder(config).run(),
             "formatter": lambda path: [{"name": "Word2Vec-Generated", "path": str(path)}]},
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
                # --- NEW: Add robust error handling for each pipeline ---
                # This prevents a single failing pipeline (e.g., Word2Vec) from crashing the entire run.
                try:
                    result = p_config["runner"]()
                    if result:
                        formatted_result = p_config["formatter"](result)
                        checkpoint_manager.save_checkpoint(p_config['name'], formatted_result)
                        generated_files.extend(formatted_result)
                        if not self.ui_manager.prompt_to_continue(p_config['name']):
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

    def _run_pre_analysis_and_prompt(self, config: Config, fasta_file_path: Path) -> bool:
        """Runs preliminary benchmarks, displays a summary, and prompts the user to continue."""
        all_benchmark_results = []

        # --- ERROR HANDLING: Wrap each benchmark suite to allow the pipeline to continue if one fails. ---
        try:
            if config.RUN_BENCHMARKING_PIPELINE:
                with mlflow.start_run(run_name="GNN_Benchmark_Suite"):
                    gnn_results_df = GNNBenchmarker(config).run()
                    if gnn_results_df is not None and not gnn_results_df.empty:
                        all_benchmark_results.append(gnn_results_df)
        except Exception as e:
            print(f"\n--- GNN BENCHMARKING FAILED: {e} ---\n")
            import traceback
            traceback.print_exc()

        try:
            if config.RUN_NETWORK_EMBEDDING_BENCHMARKING:
                with mlflow.start_run(run_name="NE_Benchmark_Suite"):
                    ne_results_df = NetworkEmbeddingBenchmarker(config).run()
                    if ne_results_df is not None and not ne_results_df.empty:
                        all_benchmark_results.append(ne_results_df)
        except Exception as e:
            print(f"\n--- NETWORK EMBEDDING BENCHMARKING FAILED: {e} ---\n")
            import traceback
            traceback.print_exc()

        try:
            if config.RUN_SINGLETON_GCN_EVAL:
                DataUtils.print_header("Ensuring n=1 Graph is Built for Singleton Evaluation")
                singleton_config = copy.deepcopy(config)
                singleton_config.PROTGRAM_NGRAM_MAX_N = 1
                build_script_path = singleton_config.BASE_SOURCE_DIR / "data_builders" / "build_graphs.py"
                subprocess.run([sys.executable, str(build_script_path), "--fasta_path", str(fasta_file_path)], check=True)

                # --- FIX: Use the new directory-based loading for graphs ---
                # The graph is now saved as a directory, not a single .pkl file.
                n1_graph_dir = singleton_config.RESULTS_GRAPH_OBJECTS_DIR / "ngram_graph_n1"
                if n1_graph_dir.exists() and n1_graph_dir.is_dir():
                    print("  n=1 graph found. Proceeding with singleton evaluation...")
                    from source.data_structures.graph import DirectedNgramGraph
                    from source.trainers.singleton_xgcn import SingletonXGCNTrainer
                    # Use the correct class method to load from the directory
                    n1_graph: DirectedNgramGraph = DirectedNgramGraph.load_from_dir(n1_graph_dir)
                    if n1_graph:
                        singleton_results_df = SingletonXGCNTrainer(singleton_config, n1_graph).run()
                        if singleton_results_df is not None and not singleton_results_df.empty:
                            singleton_results_df = singleton_results_df.rename(columns={'Model': 'model'})
                            singleton_results_df['dataset'] = f"ProtGram_n1_Singleton_{fasta_file_path.stem}"
                            singleton_results_df['error'] = None
                            all_benchmark_results.append(singleton_results_df)
                else:
                    print(f"  Warning: n=1 graph directory not found at {n1_graph_dir}. Skipping singleton evaluation.")
        except Exception as e:
            print(f"\n--- SINGLETON GCN EVALUATION FAILED: {e} ---\n")
            import traceback
            traceback.print_exc()

        self.ui_manager.display_aggregated_benchmark_summary(all_benchmark_results)

        print("\n" + "#" * 80)
        print("### PRELIMINARY ANALYSIS COMPLETE ###")
        print("#" * 80)
        return self.ui_manager.prompt_to_continue("Preliminary Analysis")

    def run(self):
        """Executes the entire pipeline."""
        script_start_time = time.monotonic()
        logger = FileLogger(self.base_config.LOG_DIR, enabled=self.base_config.ENABLE_FILE_LOGGING)

        with logger:
            try:
                DataUtils.print_header("Starting Protein-Protein Interaction Meta-Pipeline")

                # --- NEW: Data Validation and Restoration Logic ---
                # This logic now lives at the start of the main application run.
                data_manager = DataManager(self.base_config)
                print("\n--- Verifying local data integrity ---")
                is_data_valid = data_manager.validate_data_from_manifest()

                if not is_data_valid:
                    print("\n--- Local data is invalid or missing. Attempting to restore from cache... ---")
                    restored_ok = data_manager.restore_data_from_cache()
                    if not restored_ok:
                        print("\n" + "!" * 80)
                        print("!!! FATAL: Data is missing or corrupt, and could not be restored from cache. !!!")
                        print("!!! Please run the one-time setup script to download and process all data: !!!")
                        print("!!!                                                                          !!!")
                        print("!!!   bash configuration/reset.sh                                            !!!")
                        print("!" * 80)
                        sys.exit(1)
                else:
                    print("--- Local data is valid. ---")

                # --- NEW: Add dynamic system resource checks at the start ---
                DataUtils.report_system_resources(self.base_config.BASE_OUTPUT_DIR)

                # --- NEW: Add data download step with network resilience ---
                DataDownloader(self.base_config).run()
                if self.base_config.DEBUG_VERBOSE:
                    print("--- Full Configuration Values ---")
                    for key, value in sorted(vars(self.base_config).items()):
                        if not key.startswith('_'): print(f"  {key:<40} | {value}")
                    print("--------------------------")

                if self.base_config.USE_MLFLOW:
                    mlflow.set_tracking_uri(self.base_config.MLFLOW_TRACKING_URI)

                if self.base_config.RUN_INTEGRATED_TESTS:
                    DataUtils.print_header("Running Integrated Test Suite")
                    gpu_is_ok = run_all_tests()
                    DataUtils.print_header("Integrated Test Suite Finished.")
                    if not gpu_is_ok:
                        print("\n" + "!" * 80)
                        print("!!! WARNING: GPU verification failed. Pipeline will run on CPU. !!!")
                        print("!" * 80 + "\n")
                        if sys.stdin.isatty():
                            response = input("Continue with CPU-only execution? (y/n): ").lower().strip()
                            if response not in ['y', 'yes']: sys.exit(1)
                    if not self.ui_manager.prompt_to_continue("Integrated Tests"): sys.exit(0)

                with tempfile.TemporaryDirectory() as temp_dir:
                    files_to_process = self.ui_manager.get_fasta_files_to_process(self.base_config, Path(temp_dir))
                    if not files_to_process:
                        print("\nERROR: No sequence files defined. Cannot run experiments.")
                        return

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
                                        conversion_script_path = self.project_root / "source" / "utils" / "models" / "model_converter.py"
                                        # --- FIX: Remove capture_output to make the conversion process visible to the user ---
                                        # The conversion script prints its own progress, which is useful for debugging.
                                        subprocess.run([sys.executable, str(conversion_script_path), model_id], check=True)
                                except Exception as e:
                                    print(f"  Warning: Could not verify or convert model '{model_id}': {e}")

                    for fasta_file_path in files_to_process:
                        # --- NEW: Add robust error handling for each dataset ---
                        # This prevents a failure on one FASTA file from stopping the entire orchestration.
                        try:
                            config = copy.deepcopy(self.base_config)
                            dataset_name = fasta_file_path.stem
                            DataUtils.print_header(f"PROCESSING DATASET: {dataset_name.upper()}")
                            config.SEQUENCE_FILE_PATHS = [fasta_file_path]

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

                            if self._run_pre_analysis_and_prompt(config, fasta_file_path):
                                DataUtils.print_header("Building all n-gram graphs for the main pipeline")
                                build_script_path = config.BASE_SOURCE_DIR / "data_builders" / "build_graphs.py"
                                subprocess.run([sys.executable, str(build_script_path), "--fasta_path", str(fasta_file_path)], check=True)
                                if not self.ui_manager.prompt_to_continue("Graph Building"): sys.exit(0)

                                generated_files = self._run_main_embedding_pipelines(config, checkpoint_manager)
                                # --- FIX: Ensure generated_files is a list before extending ---
                                generated_files = generated_files if isinstance(generated_files, list) else []
                                config.LP_EMBEDDING_FILES_TO_EVALUATE = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_files # noqa

                                # --- NEW: Run Hyperparameter Optimization ---
                                if config.RUN_HPO:
                                    DataUtils.print_header("Running Hyperparameter Optimization")
                                    optimizer = HyperparameterOptimizer(config)
                                    target_model_name = config.HPO_TARGET_EMBEDDING_MODEL
                                    target_embedding_path = None
                                    for emb_file in config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                        if emb_file['name'] == target_model_name:
                                            target_embedding_path = emb_file['path']
                                            break

                                    if target_embedding_path:
                                        optimizer.optimize_ppi_mlp(str(target_embedding_path), target_model_name)
                                    else:
                                        print(f"  Warning: HPO target embedding '{target_model_name}' not found in generated/configured files. Skipping HPO.")

                                if config.RUN_MAIN_PPI_EVALUATION:
                                    DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
                                    if not checkpoint_manager.get_checkpoint("PPI_Evaluation"):
                                        if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                            ppi_evaluator = PPIPipeline(config)
                                            # --- FIX: The PPI pipeline should always run on the data configured
                                            # by the orchestrator. The `use_dummy_data` flag is for isolated
                                            # testing and should not be used here. The main `RUN_DUMMY_TEST`
                                            # flag already controls the input data for the entire pipeline. ---
                                            ppi_evaluator.run(use_dummy_data=config.RUN_DUMMY_TEST)
                                            checkpoint_manager.save_checkpoint("PPI_Evaluation", {"status": "completed"})
                        # --- FIX: Add the missing 'except' block for the per-dataset try block ---
                        # This ensures that if one dataset fails, the pipeline can continue to the next.
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
    orchestrator = PipelineOrchestrator()
    orchestrator.run()