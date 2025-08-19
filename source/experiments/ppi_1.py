# ==============================================================================
# MODULE: experiments/ppi_1.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# VERSION: 5.0 (Refactored for clarity and separation of concerns)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import random
import shutil
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

import h5py
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from configuration.config import Config
from source.models.fnn.mlp import MLP
# Refactored: FileUtils is now DataUtils and lives in manager.py
from source.utils.data.data_utils import DataUtils
from source.utils.data.ground_truth_loader import GroundTruthLoader
# Refactored: Dummy data creation is now in a dedicated helper file
from source.utils.post.embedding_loader import EmbeddingLoader
from source.utils.post.embedding_processor import EmbeddingProcessor
from source.utils.results.evaluation_reporter import EvaluationReporter
from source.utils.results.evaluation_summary_generator import EvaluationSummary

# Configure GPU memory growth at the start
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Warning: Could not set memory growth for GPUs: {e}")


class PPIPipeline:
    """
    A comprehensive pipeline for evaluating protein embeddings on a link prediction
    task (Protein-Protein Interaction prediction).

    The workflow includes:
    1. Pre-processing embeddings with PCA for dimensionality reduction.
    2. Loading positive and negative interaction pairs, filtered against available embeddings.
    3. Performing N-fold cross-validation.
    4. For each fold, training a simple MLP classifier on edge features derived from protein embeddings.
    5. Evaluating the model and aggregating performance metrics (AUC, F1, etc.).
    6. Generating summary reports, tables, and plots.
    """
    def __init__(self, config: Config):
        self.config = config
        print("PPIPipeline initialized.")
        DataUtils.print_header("PPI Evaluation Pipeline Initialized")
        # --- NEW: Set seeds for reproducibility of MLP initialization and training ---
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def _preprocess_embeddings_with_pca(self, emb_configs: List[Dict]) -> List[Dict]:
        """
        Applies PCA mandatorily to all embedding files before evaluation, saving
        the results to a new directory and returning updated configurations.
        """
        DataUtils.print_header("Pre-processing: Applying Mandatory PCA to All Embeddings")
        target_dim = self.config.PCA_TARGET_DIMENSION

        # FIX: Place processed embeddings inside the dataset-specific evaluation directory to prevent overwriting.
        processed_emb_dir = self.config.RESULTS_EVALUATION_DIR / "pca_processed_embeddings"
        processed_emb_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Target dimension set to: {target_dim}")
        print(f"  Processed files will be stored in: {processed_emb_dir}")

        processed_configs = []
        for config_item in emb_configs:
            original_path = Path(config_item['path'])
            new_config = config_item.copy()

            # --- FIX: Avoid re-running PCA on an already processed file ---
            if f".pca_{target_dim}" in original_path.name:
                print(f"  Skipping PCA for '{original_path.name}' as it appears to be already processed.")
                processed_configs.append(new_config)
                continue

            if not original_path.exists():
                print(f"  Skipping non-existent file: {original_path}")
                processed_configs.append(new_config)
                continue

            print(f"  Processing '{config_item['name']}' for mandatory PCA...")
            # --- FIX: Check if PCA returned the original path, indicating a failure/skip ---
            original_path_str = str(original_path)
            new_path = EmbeddingProcessor.apply_pca_to_h5(
                input_h5_path=original_path,
                output_dir=processed_emb_dir,
                target_dimension=target_dim,
                random_seed=self.config.RANDOM_STATE
            )
            new_config['path'] = str(new_path)
            if str(new_path) == original_path_str:
                print(f"    - Note: PCA was skipped or failed for {original_path.name}. The original file will be used in the evaluation.")
            processed_configs.append(new_config)

        return processed_configs

    def _train_and_evaluate_fold(
            self,
            train_pairs: List[Tuple[str, str, int]],
            val_pairs: List[Tuple[str, str, int]],
            protein_embeddings: Union[Dict[str, np.ndarray], 'EmbeddingLoader'],
            edge_feature_dim: int,
            embedding_dim: int,
            embedding_name: str,
            fold_num: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Handles the logic for a single fold of cross-validation: model building,
        training, and evaluation.
        """
        train_labels = np.array([p[2] for p in train_pairs])
        print(f"    Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
        print(f"    Train class distribution: Pos={np.sum(train_labels == 1)}, Neg={np.sum(train_labels == 0)}")

        # Create TensorFlow Dataset generators that produce batches of edge features on the fly.
        # --- Setup Data Generators ---
        num_train_batches = max(1, (len(train_pairs) + self.config.EVAL_BATCH_SIZE - 1) // self.config.EVAL_BATCH_SIZE)
        num_val_batches = max(1, (len(val_pairs) + self.config.EVAL_BATCH_SIZE - 1) // self.config.EVAL_BATCH_SIZE)

        output_signature = (tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float16), tf.TensorSpec(shape=(None,), dtype=tf.int32))
        train_gen_func = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=train_pairs, protein_embeddings=protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD,
                                 batch_size=self.config.EVAL_BATCH_SIZE, embedding_dim=embedding_dim)
        val_gen_func = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=val_pairs, protein_embeddings=protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD,
                               batch_size=self.config.EVAL_BATCH_SIZE, embedding_dim=embedding_dim)
        train_ds = tf.data.Dataset.from_generator(train_gen_func, output_signature=output_signature).shuffle(buffer_size=num_train_batches).repeat().prefetch(tf.data.AUTOTUNE)
        val_ds_for_fit = tf.data.Dataset.from_generator(val_gen_func, output_signature=output_signature).repeat().prefetch(tf.data.AUTOTUNE)

        # Build and compile the MLP model for this fold.
        # --- Build and Train Model ---
        # FIX: Pass config object to MLP builder
        mlp_params = {'dense1_units': self.config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': self.config.EVAL_MLP_DROPOUT1_RATE, 'dense2_units': self.config.EVAL_MLP_DENSE2_UNITS,
                      'dropout2_rate': self.config.EVAL_MLP_DROPOUT2_RATE, 'l2_reg': self.config.EVAL_MLP_L2_REG}
        model = MLP(edge_feature_dim, mlp_params, self.config.EVAL_LEARNING_RATE).build()
        print(f"    MLP model built with input shape: {edge_feature_dim}")

        # Calculate class weights to handle imbalanced datasets.
        neg_count, pos_count = np.sum(train_labels == 0), np.sum(train_labels == 1)
        class_weight = {0: (neg_count + pos_count) / (2.0 * neg_count), 1: (neg_count + pos_count) / (2.0 * pos_count)} if neg_count > 0 and pos_count > 0 else None

        # Train the model with optional early stopping.
        print(f"    Starting model training for {self.config.EVAL_EPOCHS} epochs...")
        history = model.fit(train_ds, epochs=self.config.EVAL_EPOCHS,
                            validation_data=val_ds_for_fit, steps_per_epoch=num_train_batches, validation_steps=num_val_batches,
                            verbose=1 if self.config.DEBUG_VERBOSE else 0, class_weight=class_weight,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)] if self.config.EARLY_STOPPING_PATIENCE > 0 else [])
        print("    Model training finished.")

        # --- ANTICIPATORY DEBUGGING: Make slow/heavy analysis optional and robust. ---
        # Generate SHAP summary plot only for the first fold of the main embedding, if configured.
        try:
            is_main_embedding = (embedding_name == self.config.EVAL_MAIN_EMBEDDING_FOR_STATS)
            if self.config.EVAL_GENERATE_SHAP_SUMMARY and fold_num == 0 and is_main_embedding:
                print(f"    Generating SHAP summary for main model '{embedding_name}' on fold {fold_num + 1}...")
                train_features_for_shap = np.vstack([x for x, y in train_ds.take(10)])
                reporter = EvaluationReporter(str(self.config.RESULTS_EVALUATION_DIR), self.config.EVAL_K_VALUES_FOR_TABLE)
                reporter.generate_shap_summary(
                    model=model, background_data=train_features_for_shap,
                    model_name=embedding_name, fold_num=fold_num + 1
                )
        except Exception as e_shap:
            print(f"    WARNING: SHAP summary generation failed with error: {e_shap}")

        # Predict on the validation set and calculate performance metrics.
        # --- Evaluate Model ---
        print("    Evaluating model on validation set...")
        y_true_list, y_pred_proba_list = [], []
        val_ds_eval = tf.data.Dataset.from_generator(val_gen_func, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)
        for x_batch, y_batch in val_ds_eval.take(num_val_batches):
            y_true_list.append(y_batch.numpy())
            y_pred_proba_list.append(model.predict_on_batch(x_batch).flatten())

        if not y_true_list:
            print("    Warning: No data yielded by validation generator. Skipping metrics.")
            return {'precision_sklearn': np.nan, 'recall_sklearn': np.nan, 'f1_sklearn': np.nan, 'auc_sklearn': np.nan}, history.history

        y_true = np.concatenate(y_true_list)
        y_pred_proba = np.concatenate(y_pred_proba_list)
        y_pred_class = (y_pred_proba > 0.5).astype(int)

        metrics = {'precision_sklearn': precision_score(y_true, y_pred_class, zero_division=0), 'recall_sklearn': recall_score(y_true, y_pred_class, zero_division=0),
                   'f1_sklearn': f1_score(y_true, y_pred_class, zero_division=0)}
        if len(np.unique(y_true)) > 1:
            metrics['auc_sklearn'] = roc_auc_score(y_true, y_pred_proba)
            metrics['roc_data'] = roc_curve(y_true, y_pred_proba)
        else:
            metrics['auc_sklearn'] = 0.5
            metrics['roc_data'] = (np.array([0, 1]), np.array([0, 1]), 0.5)

        metrics.update(EvaluationSummary._calculate_ranking_metrics(y_true=y_true, y_score=y_pred_proba, k_list=self.config.EVAL_K_VALUES_FOR_TABLE))

        del model, train_ds, val_ds_for_fit, val_ds_eval
        gc.collect()
        tf.keras.backend.clear_session()
        return metrics, history.history

    def _run_cv_workflow(self, embedding_name: str, all_pairs_for_cv: List[Tuple[str, str, int]], protein_embeddings: Union[Dict[str, np.ndarray], 'EmbeddingLoader']) -> Dict[str, Any]:
        """
        Manages the cross-validation process, including splitting data into folds,
        calling the training/evaluation logic for each fold, and aggregating results.
        """
        cv_start_time = time.monotonic()
        print(f"Starting CV workflow for {embedding_name}. Total pairs for CV: {len(all_pairs_for_cv)}")
        aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}, 'notes': ""}

        labels_array = np.array([p[2] for p in all_pairs_for_cv])
        if len(np.unique(labels_array)) < 2:
            note = "Single class in dataset for CV. Cannot perform meaningful stratified CV or calculate some metrics."
            print(f"  Warning: {note}")
            aggregated_results['notes'] = note
            return aggregated_results

        skf = StratifiedKFold(n_splits=self.config.EVAL_N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        # This list will store the metrics dictionary from each fold.
        fold_metrics_list: List[Dict[str, Any]] = []

        # --- NEW: Define expected metric keys for consistency across all folds ---
        expected_metric_keys = (
                ['precision_sklearn', 'recall_sklearn', 'f1_sklearn', 'auc_sklearn', 'roc_data'] +
                [f'hits_at_{k}' for k in self.config.EVAL_K_VALUES_FOR_TABLE] +
                [f'ndcg_at_{k}' for k in self.config.EVAL_K_VALUES_FOR_TABLE]
        )

        # Determine the embedding dimension safely from either a dict or a lazy loader.
        # --- MODIFIED: Get embedding dim safely from dict or loader ---
        if isinstance(protein_embeddings, dict):
            first_valid_emb = next((v for v in protein_embeddings.values() if v is not None and v.size > 0), None)
        else:  # Is an EmbeddingLoader
            first_key = next(iter(protein_embeddings.get_keys()), None)
            first_valid_emb = protein_embeddings[first_key] if first_key else None

        if first_valid_emb is None:
            aggregated_results['notes'] = "No valid embeddings found for CV."
            return aggregated_results
        embedding_dim = first_valid_emb.shape[0]

        feature_dim_map = {'concatenate': embedding_dim * 2, 'average': embedding_dim, 'hadamard': embedding_dim, 'l1_distance': embedding_dim, 'l2_distance': embedding_dim}
        edge_feature_dim = feature_dim_map.get(self.config.EVAL_EDGE_EMBEDDING_METHOD, embedding_dim * 2)

        # --- NEW: Proactive memory check before starting CV ---
        # Estimate memory needed for the full dataset's features (a rough upper bound).
        # float16 = 2 bytes per element.
        estimated_gb_needed = (len(all_pairs_for_cv) * edge_feature_dim * 2) / (1024**3)
        if not DataUtils.has_enough_memory(estimated_gb_needed, f"PPI CV for {embedding_name}"):
            aggregated_results['notes'] = "Skipped due to insufficient available memory."
            print(f"  Skipping CV for {embedding_name} due to memory constraints.")
            return aggregated_results

        # Loop through each cross-validation fold.
        for fold_num, (train_idx, val_idx) in enumerate(tqdm(skf.split(np.zeros(len(all_pairs_for_cv)), labels_array), total=self.config.EVAL_N_FOLDS, desc=f"  CV Folds for {embedding_name}", leave=False)):
            fold_start_time = time.monotonic()
            print(f"\n  --- Fold {fold_num + 1}/{self.config.EVAL_N_FOLDS} for {embedding_name} ---")

            # --- ANTICIPATORY DEBUGGING: Add exception handling for individual folds to make the pipeline more robust. ---
            try:
                train_pairs_fold = [all_pairs_for_cv[i] for i in train_idx]
                val_pairs_fold = [all_pairs_for_cv[i] for i in val_idx]

                # Train and evaluate the model for the current fold.
                fold_metrics, history = self._train_and_evaluate_fold(
                    train_pairs=train_pairs_fold, val_pairs=val_pairs_fold,
                    protein_embeddings=protein_embeddings, edge_feature_dim=edge_feature_dim,
                    embedding_dim=embedding_dim,
                    embedding_name=embedding_name, fold_num=fold_num
                )
                fold_metrics_list.append(fold_metrics)
                # Store history and a representative ROC curve from the first fold for plotting.
                if fold_num == 0:
                    aggregated_results['history_dict_fold1'] = history
                    if 'roc_data' in fold_metrics:
                        aggregated_results['roc_data_representative'] = fold_metrics['roc_data']

                print(f"    Fold {fold_num + 1} Metrics: {fold_metrics}")
                print(f"    Fold {fold_num + 1} completed in {time.monotonic() - fold_start_time:.2f}s.")
            except Exception as e:
                print(f"    ❌ ERROR in Fold {fold_num + 1} for {embedding_name}: {e}")
                import traceback
                traceback.print_exc()
                if self.config.USE_MLFLOW:
                    with mlflow.start_run(run_name=f"Fold_{fold_num + 1}_FAILED", nested=True):
                        mlflow.set_tag("status", "FAILED")
                        mlflow.log_param("error", str(e))
                # Use a predefined, consistent set of keys for failed folds to prevent silent failures from skewing the final average metrics.
                nan_metrics = {key: np.nan for key in expected_metric_keys}
                fold_metrics_list.append(nan_metrics)

        # After all folds are complete, calculate the mean and std dev of the metrics.
        if fold_metrics_list:
            metrics_keys = [k for k in expected_metric_keys if k != 'roc_data']
            for key in metrics_keys:
                values = [fm.get(key, np.nan) for fm in fold_metrics_list]
                aggregated_results[f'test_{key}'] = np.nanmean(values)
                aggregated_results[f'test_{key}_std'] = np.nanstd(values) if self.config.EVAL_N_FOLDS > 1 else 0.0

            aggregated_results['fold_f1_scores'] = [fm.get('f1_sklearn', np.nan) for fm in fold_metrics_list]
            aggregated_results['fold_auc_scores'] = [fm.get('auc_sklearn', np.nan) for fm in fold_metrics_list]

        print(f"CV workflow for {embedding_name} finished in {time.monotonic() - cv_start_time:.2f}s.")
        return aggregated_results

    def run(self, use_dummy_data: bool = False, parent_run_id: Optional[str] = None):
        """
        The main public entry point for the PPI evaluation pipeline.
        """
        pipeline_start_time = time.monotonic()
        run_type = "DUMMY EVALUATION" if use_dummy_data else "MAIN EVALUATION"
        DataUtils.print_header(f"PPI EVALUATION PIPELINE ({run_type})")

        # --- DEFINITIVE FIX: Wrap the main logic in a try...finally to guarantee cleanup ---
        dummy_data_dir = None
        try:
            if use_dummy_data:  # noqa
                num_proteins = 50
                embedding_dim = 16
                num_pos = 100
                num_neg = 100

                dummy_data_dir = self.config.BASE_OUTPUT_DIR / "dummy_data_temp"
                if dummy_data_dir.exists():
                    shutil.rmtree(dummy_data_dir)
                dummy_data_dir.mkdir(parents=True, exist_ok=True)
                print(f"Creating dummy data in: {dummy_data_dir} (Proteins: {num_proteins}, Dim: {embedding_dim}, Pos: {num_pos}, Neg: {num_neg})")

                protein_ids = [f"DUMMY_P{i:04d}" for i in range(num_proteins)]

                # Create dummy embeddings
                dummy_emb_file = dummy_data_dir / "dummy_embeddings.h5"
                with h5py.File(dummy_emb_file, 'w') as hf:
                    for pid in protein_ids:
                        hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float16))
                print(f"  Dummy embeddings saved to: {dummy_emb_file}")

                # Create dummy positive interactions
                pos_fp = dummy_data_dir / "dummy_pos.csv"
                pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)], columns=['p1', 'p2'])
                pos_pairs.to_csv(pos_fp, header=False, index=False)
                print(f"  Dummy positive interactions saved to: {pos_fp}")

                # Create dummy negative interactions
                neg_fp = dummy_data_dir / "dummy_neg.csv"  # noqa
                neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)], columns=['p1', 'p2'])
                neg_pairs.to_csv(neg_fp, header=False, index=False)
                print(f"  Dummy negative interactions saved to: {neg_fp}")

                dummy_emb_config = [{"path": str(dummy_emb_file), "name": "DummyEmb"}]

                # --- FIX: Assign the dummy config to the main variable used by the pipeline ---
                emb_configs = dummy_emb_config
            else:
                emb_configs = getattr(self.config, 'LP_EMBEDDING_FILES_TO_EVALUATE', [])
                pos_fp = self.config.POS_INTERACTIONS_PATH
                neg_fp = self.config.NEG_INTERACTIONS_PATH
                if not emb_configs:
                    print("Warning: 'LP_EMBEDDING_FILES_TO_EVALUATE' is empty in config. No evaluation will run.")
                    return

            # Refactored: PCA pre-processing is now a clean, single method call
            emb_configs = self._preprocess_embeddings_with_pca(emb_configs)

            reporter = EvaluationReporter(base_output_dir=str(self.config.RESULTS_EVALUATION_DIR), k_vals_table=self.config.EVAL_K_VALUES_FOR_TABLE)

            all_cv_results_list = []
            for emb_config_item in tqdm(emb_configs, desc="Evaluating Embedding Models"):
                emb_name = emb_config_item['name']
                emb_path = emb_config_item['path']
                mlflow_active = self.config.USE_MLFLOW
                run_context = mlflow.start_run(run_name=emb_name, nested=bool(parent_run_id)) if mlflow_active else nullcontext()

                with run_context as run:
                    DataUtils.print_header(f"Processing Embedding: {emb_name}")
                    print(f"  Path: {emb_path}")
                    if mlflow_active and run:
                        mlflow.log_params({"embedding_name": emb_name, "embedding_path": emb_path, "edge_embedding_method": self.config.EVAL_EDGE_EMBEDDING_METHOD, "n_folds": self.config.EVAL_N_FOLDS})

                    if not Path(emb_path).exists():
                        print(f"ERROR: Embedding file not found for {emb_name} at {emb_path}. Skipping.")
                        continue

                    try:
                        # --- FIX: Pass the config object to the EmbeddingLoader ---
                        # This allows the loader to dynamically choose its loading strategy (lazy vs. in-memory).
                        with EmbeddingLoader(emb_path, config=self.config) as protein_embeddings_loader:
                            # --- DEFINITIVE FIX for Scalability: Stream and filter interaction pairs ---
                            # Instead of loading all interaction pairs into memory, we first get the IDs
                            # available in the current embedding file. Then, we stream the interaction
                            # files and only load the pairs for which we have embeddings. This dramatically
                            # reduces memory usage.
                            available_ids = protein_embeddings_loader.get_keys()
                            if not available_ids:
                                print(f"  No embeddings found in H5 file for {emb_name}. Skipping CV.")
                                continue

                            print(f"  Found {len(available_ids)} embeddings. Filtering interaction files against these IDs...")
                            pos_pairs = GroundTruthLoader.load_interaction_pairs_filtered(
                                pos_fp, 1, available_ids, random_state=self.config.RANDOM_STATE
                            )
                            num_pos_for_sampling = len(pos_pairs)
                            neg_pairs = GroundTruthLoader.load_interaction_pairs_filtered(
                                neg_fp, 0, available_ids, sample_n=num_pos_for_sampling, random_state=self.config.RANDOM_STATE
                            )

                            pairs_for_cv = pos_pairs + neg_pairs
                            if not pairs_for_cv:
                                print(f"  No interaction pairs remain after filtering against available embeddings for {emb_name}. Skipping CV.")
                                continue

                            print(f"  Proceeding to CV with {len(pairs_for_cv)} filtered pairs for {emb_name}.")
                            random.shuffle(pairs_for_cv)

                            results = self._run_cv_workflow(emb_name, pairs_for_cv, protein_embeddings_loader)
                            all_cv_results_list.append(results)

                            if mlflow_active and run and results:
                                metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
                                mlflow.log_metrics(metrics_to_log)
                                # --- FIX: Plot the training history that was collected for the first fold ---
                                if self.config.PLOT_TRAINING_HISTORY and results.get('history_dict_fold1'):
                                    history_plot_path = reporter.plot_training_history(results['history_dict_fold1'], emb_name)
                                    if history_plot_path:
                                        mlflow.log_artifact(str(history_plot_path), "training_plots")
                    # --- NEW: Specific handling for Out-of-Memory errors ---
                    # This catches OOM errors from TensorFlow (ResourceExhaustedError) or NumPy/Python (MemoryError)
                    # that can occur during batch generation or model training, allowing the pipeline to continue.
                    except (MemoryError, tf.errors.ResourceExhaustedError) as mem_e:
                        print("\n" + "!" * 80)
                        print(f"!!! GRACEFUL SHUTDOWN for {emb_name} due to OUT-OF-MEMORY !!!")
                        print(f"  Caught a memory-related error: {type(mem_e).__name__}")
                        print(f"  This means the system ran out of RAM or GPU memory for this specific task.")
                        print(f"  The pipeline will now clean up and proceed to the next embedding file.")
                        print("!" * 80 + "\n")
                        if mlflow_active and run:
                            mlflow.set_tag("status", "FAILED_OOM")
                            mlflow.log_param("error", str(mem_e))
                    except Exception as e:
                        print(f"\n--- UNEXPECTED ERROR during processing for {emb_name}: {e}. ---")
                        print("--- Continuing to next embedding file. ---")
                        import traceback
                        traceback.print_exc()
                    finally:
                        DataUtils.report_memory_usage(f"After processing {emb_name}")
                        gc.collect()

            if all_cv_results_list:
                DataUtils.print_header("FINAL AGGREGATE RESULTS & REPORTING")
                # --- FIX: Instantiate and use the correct class for writing the summary file ---
                summary_generator = EvaluationSummary(base_output_dir=str(self.config.RESULTS_EVALUATION_DIR), k_vals_table=self.config.EVAL_K_VALUES_FOR_TABLE)
                summary_generator.write_summary_file(all_cv_results_list, self.config.EVAL_MAIN_EMBEDDING_FOR_STATS, 'test_auc_sklearn', self.config.EVAL_STATISTICAL_TEST_ALPHA)
                reporter.plot_roc_curves(all_cv_results_list)
                reporter.plot_comparison_charts(all_cv_results_list)

        finally:
            # This block is guaranteed to run, ensuring cleanup.
            if use_dummy_data and self.config.CLEANUP_DUMMY_DATA and dummy_data_dir and dummy_data_dir.exists():
                shutil.rmtree(dummy_data_dir)
                print(f"Cleaned up dummy data directory: {dummy_data_dir}")

        DataUtils.print_header(f"PPI Evaluation Pipeline ({run_type}) FINISHED in {time.monotonic() - pipeline_start_time:.2f}s")
