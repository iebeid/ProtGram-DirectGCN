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
import time
import tempfile
from pathlib import Path

# --- Add project root to sys.path to allow for relative imports ---
# This must be done BEFORE any local modules (like 'configuration') are imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from typing import List, Dict
from source.utils.post.embedding_processor import EmbeddingProcessor
import mlflow
import tensorflow as tf
# --- Local Application Imports ---
from configuration.config import Config
from source.benchmarkers.gnns import GNNBenchmarker
from source.data_builders.protgram import ProtGramDataBuilder
from configuration.manager import DataManager
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker
from source.experiments.ppi_1 import PPIPipeline
from source.testers.unit_tests import run_all_tests
from source.trainers.lstm import LSTMBasedEmbedder
from source.trainers.protgram_xgcn import ProtGramXGCNTrainer
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.models.hyperparameter_optimizer import HyperparameterOptimizer
from source.utils.data.data_utils import DataUtils
from source.utils.logging.file_logger import FileLogger
from source.entry.checkpoints import CheckpointManager
from source.entry.ui import UIManager


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
            # --- DEFINITIVE FIX: Use the standalone tf_keras package for consistency ---
            # The setup script explicitly installs `tf-keras`. To avoid namespace conflicts
            # with the Keras bundled in TensorFlow, we will consistently use the `tf_keras` package.
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

    # Add this import with the other local application imports at the top of the file


    # Replace the existing function with this corrected version
    def _run_main_embedding_pipelines(self, config: Config, checkpoint_manager: CheckpointManager) -> List[Dict[str, str]]:
        """Runs the main embedding generation pipelines with checkpointing and user prompts."""
        pipelines = [
            {"name": "ProtGram-XGCN", "flag": "RUN_GCN_PIPELINE", "runner": lambda: ProtGramXGCNTrainer(config).run(),
             "formatter": lambda paths: [{"name": name, "path": path} for name, path in paths.items()]},
            # --- FIX: Update formatter to handle a dictionary of paths (main + optional PCA) ---
            {"name": "Word2Vec", "flag": "RUN_WORD2VEC_PIPELINE", "runner": lambda: Word2VecEmbedder(config).run(), # noqa
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

                        standardized_files = []
                        for file_info in formatted_raw_results:
                            # Standardize each file immediately after it's created
                            standardized_path = EmbeddingProcessor.standardize_embedding_file(file_info["path"])
                            standardized_files.append({"name": file_info["name"], "path": standardized_path})

                        checkpoint_manager.save_checkpoint(p_config['name'], standardized_files)
                        generated_files.extend(standardized_files)

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
                DataUtils.print_header("Running Singleton (n=1) GCN Evaluation")

                # --- FIX: Use the new directory-based loading for graphs ---
                # The graph is now saved as a directory, not a single .pkl file.
                n1_graph_dir = config.RESULTS_GRAPH_OBJECTS_DIR / "ngram_graph_n1"
                if n1_graph_dir.exists() and n1_graph_dir.is_dir():
                    print("  n=1 graph found. Proceeding...")
                    from source.data_structures.direct_ngram_graph import DirectedNgramGraph
                    from source.trainers.singleton_xgcn import SingletonXGCNTrainer
                    # Use the correct class method to load from the directory
                    n1_graph: DirectedNgramGraph = DirectedNgramGraph.load_from_dir(n1_graph_dir)
                    if n1_graph:
                        singleton_results_df = SingletonXGCNTrainer(config, n1_graph).run()
                        if singleton_results_df is not None and not singleton_results_df.empty:
                            singleton_results_df = singleton_results_df.rename(columns={'Model': 'model'})
                            singleton_results_df['dataset'] = f"ProtGram_n1_Singleton_{fasta_file_path.stem}"
                            singleton_results_df['error'] = None
                            all_benchmark_results.append(singleton_results_df)
                else:
                    print(f"  Warning: n=1 graph not found at {n1_graph_dir}. It may not have been built. Skipping singleton evaluation.")
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

                data_manager = DataManager(self.base_config)
                print("\n--- Verifying local data integrity ---")
                # --- DEFINITIVE FIX: Use the more robust setup check ---
                if not data_manager.is_setup_complete():
                    print("\n--- Local data is invalid or incomplete. Attempting to restore from cache... ---")
                    # Try to restore source files. This is just an optimization.
                    if data_manager.restore_data_from_cache():
                        print("  - ✅ Successfully restored source data from cache.")
                    else:
                        print("  - ℹ️ Could not restore from cache. Will proceed with full download.")

                    # Now, run the full setup. It's idempotent and will handle everything.
                    # If files were restored, it will skip downloading and just process.
                    # If restore failed, it will download and then process.
                    # This guarantees that the ID map cache is created.
                    print("\n--- Data setup is incomplete. Running full setup to generate derived files... ---")
                    data_manager.run_full_setup()

                    if not data_manager.is_setup_complete():
                        print("FATAL: Data validation failed even after a full setup. Please check logs.")
                        sys.exit(1)
                else:
                    print("  - ✅ Local data is valid.")

                # --- NEW: Add dynamic system resource checks at the start ---
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
                                        # --- DEFINITIVE FIX: Invoke the converter as a module (-m) ---
                                        # This is the standard, robust way to run a script from within a package,
                                        # as it correctly handles the Python path and avoids ModuleNotFoundError.
                                        module_path = "source.utils.models.model_converter"
                                        subprocess.run([sys.executable, "-m", module_path, model_id], check=True)
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

                            # --- DEFINITIVE FIX: Add checkpointing for major pipeline stages ---
                            # This prevents long-running steps from being re-executed unnecessarily.
                            if not checkpoint_manager.get_checkpoint("GraphBuilding"):
                                DataUtils.print_header("Building all n-gram graphs for the main pipeline")
                                ProtGramDataBuilder(config).run()
                                checkpoint_manager.save_checkpoint("GraphBuilding", {"status": "completed"})
                            if not self.ui_manager.prompt_to_continue("Graph Building"): continue

                            # --- DEFINITIVE FIX: Decouple main embedding generation from pre-analysis ---
                            # The main embedding pipelines have their own internal checkpointing and should
                            # always be run to ensure the list of files to evaluate is populated.
                            generated_files = self._run_main_embedding_pipelines(config, checkpoint_manager)
                            generated_files = generated_files if isinstance(generated_files, list) else []
                            config.LP_EMBEDDING_FILES_TO_EVALUATE = config.LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE + generated_files

                            # Now, run the optional pre-analysis/benchmarking step, gated by its own checkpoint.
                            if not checkpoint_manager.get_checkpoint("PreAnalysis"):
                                if not self._run_pre_analysis_and_prompt(config, fasta_file_path):
                                    continue  # User chose to stop after pre-analysis
                                checkpoint_manager.save_checkpoint("PreAnalysis", {"status": "completed"})

                            # --- Run Hyperparameter Optimization ---
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

                            # --- Run Main PPI Evaluation ---
                            if config.RUN_MAIN_PPI_EVALUATION:
                                DataUtils.print_header(f"Running Main Evaluation for Dataset: {dataset_name}")
                                if not checkpoint_manager.get_checkpoint("PPI_Evaluation"):
                                    if config.LP_EMBEDDING_FILES_TO_EVALUATE:
                                        ppi_evaluator = PPIPipeline(config)
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