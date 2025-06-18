# ==============================================================================
# MODULE: pipeline/ppi_main.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# VERSION: 3.2 (Corrected EvaluationReporter usage)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import random
import shutil
import time
from contextlib import nullcontext
from typing import List, Optional, Dict, Any, Tuple

import h5py
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import Config
from src.models.mlp import MLP
from src.utils.data_utils import DataUtils, GroundTruthLoader
from src.utils.models_utils import EmbeddingLoader, EmbeddingProcessor
from src.utils.results_utils import EvaluationReporter


class PPIPipeline:
    def __init__(self, config: Config):
        self.config = config
        print("PPIPipeline initialized.")
        DataUtils.print_header("PPI Evaluation Pipeline Initialized")

    @staticmethod
    def _create_dummy_data(base_dir: str, num_proteins: int, embedding_dim: int, num_pos: int, num_neg: int) -> Tuple[str, str, List[Dict[str, Any]]]:
        dummy_data_dir = os.path.join(base_dir, "dummy_data_temp")
        if os.path.exists(dummy_data_dir): shutil.rmtree(dummy_data_dir)
        os.makedirs(dummy_data_dir, exist_ok=True)
        print(f"Creating dummy data in: {dummy_data_dir} (Proteins: {num_proteins}, Dim: {embedding_dim}, Pos: {num_pos}, Neg: {num_neg})")
        protein_ids = [f"DUMMY_P{i:04d}" for i in range(num_proteins)]

        dummy_emb_file = os.path.join(dummy_data_dir, "dummy_embeddings.h5")
        with h5py.File(dummy_emb_file, 'w') as hf:
            for pid in protein_ids:
                hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float32))
        print(f"  Dummy embeddings saved to: {dummy_emb_file}")

        dummy_pos_path = os.path.join(dummy_data_dir, "dummy_pos.csv")
        pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)], columns=['p1', 'p2'])
        pos_pairs.to_csv(dummy_pos_path, header=False, index=False)
        print(f"  Dummy positive interactions saved to: {dummy_pos_path}")

        dummy_neg_path = os.path.join(dummy_data_dir, "dummy_neg.csv")
        neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)], columns=['p1', 'p2'])
        neg_pairs.to_csv(dummy_neg_path, header=False, index=False)
        print(f"  Dummy negative interactions saved to: {dummy_neg_path}")

        dummy_emb_config = [{"path": dummy_emb_file, "name": "DummyEmb"}]
        return dummy_pos_path, dummy_neg_path, dummy_emb_config

    def _run_cv_workflow(self, embedding_name: str, all_pairs: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        cv_start_time = time.time()
        print(f"Starting CV workflow for {embedding_name}. Total pairs: {len(all_pairs)}")
        aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}, 'notes': ""}

        labels = np.array([p[2] for p in all_pairs])
        if len(np.unique(labels)) < 2:
            note = "Single class in dataset. Cannot perform meaningful stratified CV or calculate some metrics."
            print(f"  Warning: {note}")
            aggregated_results['notes'] = note
            aggregated_results.update({
                'test_precision_sklearn': 0.0, 'test_recall_sklearn': 0.0,
                'test_f1_sklearn': 0.0, 'test_auc_sklearn': 0.5,
                'test_f1_sklearn_std': 0.0, 'test_auc_sklearn_std': 0.0,
                'fold_f1_scores': [0.0] * self.config.EVAL_N_FOLDS,
                'fold_auc_scores': [0.5] * self.config.EVAL_N_FOLDS
            })
            return aggregated_results

        skf = StratifiedKFold(n_splits=self.config.EVAL_N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        fold_metrics_list: List[Dict[str, Any]] = []

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_pairs)), labels)):
            fold_start_time = time.time()
            print(f"\n  --- Fold {fold_num + 1}/{self.config.EVAL_N_FOLDS} for {embedding_name} ---")

            train_pairs = [all_pairs[i] for i in train_idx]
            val_pairs = [all_pairs[i] for i in val_idx]
            print(f"    Train pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")

            print("    Creating features for training set...")
            X_train, y_train = EmbeddingProcessor.create_edge_embeddings(train_pairs, protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD)
            print("    Creating features for validation set...")
            X_val, y_val = EmbeddingProcessor.create_edge_embeddings(val_pairs, protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD)

            if X_train is None or X_val is None or X_train.size == 0 or X_val.size == 0:
                print(f"    Skipping fold {fold_num + 1} due to feature creation failure or empty features.")
                fold_metrics_list.append({'precision_sklearn': np.nan, 'recall_sklearn': np.nan, 'f1_sklearn': np.nan, 'auc_sklearn': np.nan})
                continue

            print(f"    Training features shape: {X_train.shape}, Validation features shape: {X_val.shape}")
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(self.config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            mlp_params = {'dense1_units': self.config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': self.config.EVAL_MLP_DROPOUT1_RATE,
                          'dense2_units': self.config.EVAL_MLP_DENSE2_UNITS, 'dropout2_rate': self.config.EVAL_MLP_DROPOUT2_RATE,
                          'l2_reg': self.config.EVAL_MLP_L2_REG}
            model_builder = MLP(X_train.shape[1], mlp_params, self.config.EVAL_LEARNING_RATE)
            model = model_builder.build()
            print(f"    MLP model built with input shape: {X_train.shape[1]}")

            print(f"    Starting model training for {self.config.EVAL_EPOCHS} epochs...")
            history = model.fit(train_ds, epochs=self.config.EVAL_EPOCHS, validation_data=val_ds,
                                verbose=1 if self.config.DEBUG_VERBOSE else 0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)] if self.config.EARLY_STOPPING_PATIENCE > 0 else [])
            if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history
            print(f"    Model training finished for fold {fold_num + 1}.")

            print("    Evaluating model on validation set...")
            y_pred_proba = model.predict(val_ds, verbose=0).flatten()
            y_pred_class = (y_pred_proba > 0.5).astype(int)

            current_metrics = {
                'precision_sklearn': precision_score(y_val, y_pred_class, zero_division=0),
                'recall_sklearn': recall_score(y_val, y_pred_class, zero_division=0),
                'f1_sklearn': f1_score(y_val, y_pred_class, zero_division=0)
            }
            if len(np.unique(y_val)) > 1:
                current_metrics['auc_sklearn'] = roc_auc_score(y_val, y_pred_proba)
                if fold_num == 0:
                    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
                    aggregated_results['roc_data_representative'] = (fpr, tpr, current_metrics['auc_sklearn'])
            else:
                current_metrics['auc_sklearn'] = 0.5
                if fold_num == 0: aggregated_results['roc_data_representative'] = (np.array([0, 1]), np.array([0, 1]), 0.5)

            # Add Hits@k and NDCG@k metrics if needed for PPI
            # These typically require ranking predictions for a set of candidates per protein
            # This dummy run doesn't generate candidates, so we'll add placeholder metrics
            for k in self.config.EVAL_K_VALUES_FOR_TABLE:
                current_metrics[f'hits_at_{k}'] = 0.0  # Placeholder
                current_metrics[f'ndcg_at_{k}'] = 0.0  # Placeholder

            fold_metrics_list.append(current_metrics)
            print(f"    Fold {fold_num + 1} Metrics: {current_metrics}")
            del model, history, train_ds, val_ds, X_train, y_train, X_val, y_val
            gc.collect()
            tf.keras.backend.clear_session()
            print(f"    Fold {fold_num + 1} completed in {time.time() - fold_start_time:.2f}s.")

        if fold_metrics_list:
            # Calculate mean and std dev across folds
            metrics_keys = fold_metrics_list[0].keys() if fold_metrics_list else []
            for key in metrics_keys:
                values = [fm.get(key, np.nan) for fm in fold_metrics_list]
                aggregated_results[f'test_{key}'] = np.nanmean(values)
                # Calculate std dev only if there's more than one fold
                if self.config.EVAL_N_FOLDS > 1:
                    aggregated_results[f'test_{key}_std'] = np.nanstd(values)
                else:
                    aggregated_results[f'test_{key}_std'] = 0.0  # Std dev is 0 for 1 fold

            # Store fold scores explicitly for statistical tests later
            aggregated_results['fold_f1_scores'] = [fm.get('f1_sklearn', np.nan) for fm in fold_metrics_list]
            aggregated_results['fold_auc_scores'] = [fm.get('auc_sklearn', np.nan) for fm in fold_metrics_list]
            for k in self.config.EVAL_K_VALUES_FOR_TABLE:
                aggregated_results[f'fold_hits_at_{k}_scores'] = [fm.get(f'hits_at_{k}', np.nan) for fm in fold_metrics_list]
                aggregated_results[f'fold_ndcg_at_{k}_scores'] = [fm.get(f'ndcg_at_{k}', np.nan) for fm in fold_metrics_list]

        print(f"CV workflow for {embedding_name} finished in {time.time() - cv_start_time:.2f}s.")
        if self.config.DEBUG_VERBOSE: print(f"  Aggregated results for {embedding_name}: {aggregated_results}")
        return aggregated_results

    def run(self, use_dummy_data: bool = False, parent_run_id: Optional[str] = None):
        pipeline_start_time = time.time()
        run_type = "DUMMY EVALUATION" if use_dummy_data else "MAIN EVALUATION"
        DataUtils.print_header(f"PPI EVALUATION PIPELINE ({run_type})")

        if use_dummy_data:
            output_dir = os.path.join(str(self.config.EVALUATION_RESULTS_DIR), "dummy_run_output")
            pos_fp, neg_fp, emb_configs = PPIPipeline._create_dummy_data(
                base_dir=str(self.config.BASE_OUTPUT_DIR), num_proteins=50,
                embedding_dim=16, num_pos=100, num_neg=100
            )
        else:
            output_dir = str(self.config.EVALUATION_RESULTS_DIR)
            emb_configs = getattr(self.config, 'LP_EMBEDDING_FILES_TO_EVALUATE', [])
            pos_fp = str(self.config.INTERACTIONS_POSITIVE_PATH)
            neg_fp = str(self.config.INTERACTIONS_NEGATIVE_PATH)
            if not emb_configs:
                print("Warning: 'LP_EMBEDDING_FILES_TO_EVALUATE' is empty in config. No evaluation will run.")
                return

        # Create output directories
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")

        # Instantiate the EvaluationReporter *once* for this run
        reporter = EvaluationReporter(base_output_dir=output_dir, k_vals_table=self.config.EVAL_K_VALUES_FOR_TABLE)

        DataUtils.print_header("Loading Interaction Pairs")
        load_pairs_start_time = time.time()
        all_pairs: List[Tuple[str, str, int]] = []
        positive_stream = GroundTruthLoader.stream_interaction_pairs(pos_fp, 1, batch_size=self.config.EVAL_BATCH_SIZE * 10, random_state=self.config.RANDOM_STATE)
        for batch in positive_stream: all_pairs.extend(batch)
        print(f"  Loaded {len(all_pairs)} positive pairs.")

        current_pos_count = len(all_pairs)
        negative_stream = GroundTruthLoader.stream_interaction_pairs(neg_fp, 0, batch_size=self.config.EVAL_BATCH_SIZE * 10, sample_n=self.config.SAMPLE_NEGATIVE_PAIRS, random_state=self.config.RANDOM_STATE)
        for batch in negative_stream: all_pairs.extend(batch)
        print(f"  Loaded {len(all_pairs) - current_pos_count} negative pairs.")

        if not all_pairs:
            print("CRITICAL: No interaction pairs were loaded. Exiting evaluation.")
            return
        print(f"Total pairs loaded: {len(all_pairs)} in {time.time() - load_pairs_start_time:.2f}s.")
        random.shuffle(all_pairs)

        required_ids = GroundTruthLoader.get_required_ids_from_files([pos_fp, neg_fp])
        print(f"Found {len(required_ids)} unique protein IDs that need embeddings from source files.")

        all_cv_results_list = []
        for emb_config_item in emb_configs:
            emb_name = emb_config_item['name']
            emb_path = str(emb_config_item['path'])

            mlflow_active = self.config.USE_MLFLOW
            run_context = mlflow.start_run(run_name=emb_name, nested=True if parent_run_id else False) if mlflow_active else nullcontext()

            with run_context as run:
                DataUtils.print_header(f"Processing Embedding: {emb_name}")
                print(f"  Path: {emb_path}")
                if mlflow_active and run:
                    mlflow.log_params({
                        "embedding_name": emb_name, "embedding_path": emb_path,
                        "edge_embedding_method": self.config.EVAL_EDGE_EMBEDDING_METHOD,
                        "n_folds": self.config.EVAL_N_FOLDS, "epochs": self.config.EVAL_EPOCHS,
                        "batch_size": self.config.EVAL_BATCH_SIZE, "learning_rate": self.config.EVAL_LEARNING_RATE,
                        "is_dummy_run": use_dummy_data
                    })

                if self.config.PERFORM_H5_INTEGRITY_CHECK and os.path.exists(emb_path):
                    DataUtils.check_h5_embeddings_integrity(emb_path)
                elif not os.path.exists(emb_path):
                    print(f"ERROR: Embedding file not found for {emb_name} at {emb_path}. Skipping.")
                    if mlflow_active and run: mlflow.log_param("status", "file_not_found")
                    continue

                try:
                    with EmbeddingLoader(emb_path) as protein_embeddings_loader:
                        print("  Filtering interaction pairs based on available embeddings...")
                        filtering_start_time = time.time()
                        filtered_pairs = []
                        missing_ids_count = 0
                        for p1, p2, label in tqdm(all_pairs, desc="  Filtering pairs", leave=False, disable=not self.config.DEBUG_VERBOSE):
                            if p1 in protein_embeddings_loader and p2 in protein_embeddings_loader:
                                filtered_pairs.append((p1, p2, label))
                            else:
                                missing_ids_count += 1
                        print(f"  Pair filtering complete in {time.time() - filtering_start_time:.2f}s.")
                        print(f"  Filtered pairs: {len(filtered_pairs)} (Removed {missing_ids_count} pairs due to missing embeddings).")

                        if not filtered_pairs:
                            print(f"  No pairs remain after filtering for {emb_name}. Skipping CV.")
                            if mlflow_active and run: mlflow.log_param("status", "no_valid_pairs")
                            continue

                        print("  Loading required embeddings into memory for CV...")
                        load_mem_start_time = time.time()
                        current_protein_embeddings_dict = {pid: protein_embeddings_loader[pid] for pid in required_ids if pid in protein_embeddings_loader}
                        print(f"  Loaded {len(current_protein_embeddings_dict)} embeddings into memory in {time.time() - load_mem_start_time:.2f}s.")

                        if not current_protein_embeddings_dict:
                            print(f"  No embeddings loaded into memory for {emb_name}. Skipping CV.")
                            if mlflow_active and run: mlflow.log_param("status", "no_embeddings_loaded")
                            continue

                        results = self._run_cv_workflow(emb_name, filtered_pairs, current_protein_embeddings_dict)
                        all_cv_results_list.append(results)

                        if mlflow_active and run and results:
                            metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
                            mlflow.log_metrics(metrics_to_log)
                            if results.get('notes'): mlflow.log_param("notes", results['notes'])

                        # Call instance method, passing only history and model name
                        if self.config.PLOT_TRAINING_HISTORY and results and results.get('history_dict_fold1'):
                            history_plot_path = reporter.plot_training_history(results['history_dict_fold1'], results['embedding_name'])  # MODIFIED CALL
                            if mlflow_active and run and history_plot_path and os.path.exists(history_plot_path):
                                mlflow.log_artifact(history_plot_path, "plots")

                        del current_protein_embeddings_dict
                        gc.collect()

                except FileNotFoundError as e_fnf:
                    print(f"ERROR: Embedding file not found for {emb_name}. Reason: {e_fnf}")
                    if mlflow_active and run: mlflow.log_param("status", f"file_not_found_exception: {e_fnf}")
                except Exception as e_gen:
                    print(f"UNEXPECTED ERROR during processing for {emb_name}: {e_gen}")
                    import traceback
                    traceback.print_exc()
                    if mlflow_active and run: mlflow.log_param("status", f"unexpected_error: {e_gen}")
                finally:
                    if mlflow_active and run: mlflow.end_run()

        if all_cv_results_list:
            DataUtils.print_header("FINAL AGGREGATE RESULTS & REPORTING")
            summary_run_context = mlflow.start_run(run_id=parent_run_id,
                                                   experiment_id=mlflow.get_experiment_by_name(self.config.MLFLOW_EXPERIMENT_NAME).experiment_id if parent_run_id and self.config.USE_MLFLOW else None,
                                                   run_name="Evaluation_Summary_Report", nested=bool(parent_run_id)) if self.config.USE_MLFLOW else nullcontext()

            with summary_run_context as summary_run:
                # Call instance method, passing results list and config values
                summary_path = reporter.write_summary_file(  # MODIFIED CALL
                    all_cv_results_list, self.config.EVAL_MAIN_EMBEDDING_FOR_STATS,
                    'test_auc_sklearn', self.config.EVAL_STATISTICAL_TEST_ALPHA
                )
                # Call instance methods for other plots
                roc_plot_path = reporter.plot_roc_curves(all_cv_results_list)  # MODIFIED CALL
                comparison_chart_path = reporter.plot_comparison_charts(all_cv_results_list)  # MODIFIED CALL

                if self.config.USE_MLFLOW and summary_run:
                    print("Logging summary artifacts to MLflow...")
                    if summary_path and os.path.exists(summary_path): mlflow.log_artifact(summary_path, "summary_reports")
                    if roc_plot_path and os.path.exists(roc_plot_path): mlflow.log_artifact(roc_plot_path, "summary_plots")
                    if comparison_chart_path and os.path.exists(comparison_chart_path): mlflow.log_artifact(comparison_chart_path, "summary_plots")
                    main_emb_results = next((r for r in all_cv_results_list if r['embedding_name'] == self.config.EVAL_MAIN_EMBEDDING_FOR_STATS), None)
                    if main_emb_results:
                        mlflow.log_metric(f"summary_{self.config.EVAL_MAIN_EMBEDDING_FOR_STATS}_auc", main_emb_results.get('test_auc_sklearn', 0))
                        mlflow.log_metric(f"summary_{self.config.EVAL_MAIN_EMBEDDING_FOR_STATS}_f1", main_emb_results.get('test_f1_sklearn', 0))
                if self.config.USE_MLFLOW and summary_run and parent_run_id:
                    mlflow.end_run()
        else:
            print("\nNo CV results generated from any embedding configurations.")

        if use_dummy_data and self.config.CLEANUP_DUMMY_DATA:
            dummy_dir_to_clean = os.path.join(str(self.config.BASE_OUTPUT_DIR), "dummy_data_temp")
            if os.path.exists(dummy_dir_to_clean):
                try:
                    shutil.rmtree(dummy_dir_to_clean)
                    print(f"Cleaned up dummy data directory: {dummy_dir_to_clean}")
                except Exception as e:
                    print(f"Error cleaning up dummy data directory {dummy_dir_to_clean}: {e}")
            else:
                print(f"Dummy data directory {dummy_dir_to_clean} not found for cleanup.")

        DataUtils.print_header(f"PPI Evaluation Pipeline ({run_type}) FINISHED in {time.time() - pipeline_start_time:.2f}s")
