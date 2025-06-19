# ==============================================================================
# MODULE: pipeline/ppi_main.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# VERSION: 3.4 (Using time.monotonic, fixed TF dataset exhaustion)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import random
import shutil
import time  # Ensure time is imported
from contextlib import nullcontext
from typing import List, Optional, Dict, Any, Tuple
from functools import partial

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
                hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float16))
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

    def _run_cv_workflow(self, embedding_name: str, all_pairs_for_cv: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        cv_start_time = time.monotonic()  # MODIFICATION
        print(f"Starting CV workflow for {embedding_name}. Total pairs for CV: {len(all_pairs_for_cv)}")
        aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}, 'notes': ""}

        labels_array = np.array([p[2] for p in all_pairs_for_cv])
        if len(np.unique(labels_array)) < 2:
            note = "Single class in dataset for CV. Cannot perform meaningful stratified CV or calculate some metrics."
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

        first_valid_emb = next((v for v in protein_embeddings.values() if v is not None and v.size > 0), None)
        if first_valid_emb is None:
            print(f"  ERROR: No valid embeddings in protein_embeddings for {embedding_name}. Skipping CV.")
            aggregated_results['notes'] = "No valid embeddings found for CV."
            return aggregated_results
        embedding_dim = first_valid_emb.shape[0]

        feature_dim_map = {'concatenate': embedding_dim * 2, 'average': embedding_dim,
                           'hadamard': embedding_dim, 'l1_distance': embedding_dim, 'l2_distance': embedding_dim}
        edge_feature_dim = feature_dim_map.get(self.config.EVAL_EDGE_EMBEDDING_METHOD, embedding_dim * 2)

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_pairs_for_cv)), labels_array)):
            fold_start_time = time.monotonic()  # MODIFICATION
            print(f"\n  --- Fold {fold_num + 1}/{self.config.EVAL_N_FOLDS} for {embedding_name} ---")

            train_pairs_fold = [all_pairs_for_cv[i] for i in train_idx]
            val_pairs_fold = [all_pairs_for_cv[i] for i in val_idx]
            print(f"    Train pairs: {len(train_pairs_fold)}, Validation pairs: {len(val_pairs_fold)}")

            # Calculate steps per epoch
            num_train_batches = (len(train_pairs_fold) + self.config.EVAL_BATCH_SIZE - 1) // self.config.EVAL_BATCH_SIZE
            num_val_batches = (len(val_pairs_fold) + self.config.EVAL_BATCH_SIZE - 1) // self.config.EVAL_BATCH_SIZE

            if len(train_pairs_fold) > 0 and num_train_batches == 0: num_train_batches = 1
            if len(val_pairs_fold) > 0 and num_val_batches == 0: num_val_batches = 1

            train_generator_func = partial(EmbeddingProcessor.generate_edge_features_batched,
                                           interaction_pairs=train_pairs_fold,
                                           protein_embeddings=protein_embeddings,
                                           method=self.config.EVAL_EDGE_EMBEDDING_METHOD,
                                           batch_size=self.config.EVAL_BATCH_SIZE,
                                           embedding_dim=embedding_dim)

            val_generator_func_for_fit = partial(EmbeddingProcessor.generate_edge_features_batched,  # For Keras model.fit validation_data
                                                 interaction_pairs=val_pairs_fold,
                                                 protein_embeddings=protein_embeddings,
                                                 method=self.config.EVAL_EDGE_EMBEDDING_METHOD,
                                                 batch_size=self.config.EVAL_BATCH_SIZE,
                                                 embedding_dim=embedding_dim)

            val_generator_func_for_eval = partial(EmbeddingProcessor.generate_edge_features_batched,  # For final evaluation
                                                  interaction_pairs=val_pairs_fold,
                                                  protein_embeddings=protein_embeddings,
                                                  method=self.config.EVAL_EDGE_EMBEDDING_METHOD,
                                                  batch_size=self.config.EVAL_BATCH_SIZE,
                                                  embedding_dim=embedding_dim)

            output_signature = (
                tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float16),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )

            train_ds = tf.data.Dataset.from_generator(train_generator_func, output_signature=output_signature)
            train_ds = train_ds.shuffle(buffer_size=max(1, num_train_batches)).repeat().prefetch(tf.data.AUTOTUNE)

            val_ds_for_fit = tf.data.Dataset.from_generator(val_generator_func_for_fit, output_signature=output_signature)
            val_ds_for_fit = val_ds_for_fit.repeat().prefetch(tf.data.AUTOTUNE)

            mlp_params = {'dense1_units': self.config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': self.config.EVAL_MLP_DROPOUT1_RATE,
                          'dense2_units': self.config.EVAL_MLP_DENSE2_UNITS, 'dropout2_rate': self.config.EVAL_MLP_DROPOUT2_RATE,
                          'l2_reg': self.config.EVAL_MLP_L2_REG}
            model_builder = MLP(edge_feature_dim, mlp_params, self.config.EVAL_LEARNING_RATE)
            model = model_builder.build()
            print(f"    MLP model built with input shape: {edge_feature_dim}")

            print(f"    Starting model training for {self.config.EVAL_EPOCHS} epochs (train_steps: {num_train_batches}, val_steps: {num_val_batches})...")
            history = model.fit(train_ds, epochs=self.config.EVAL_EPOCHS,
                                validation_data=val_ds_for_fit,
                                steps_per_epoch=num_train_batches,
                                validation_steps=num_val_batches if num_val_batches > 0 else None,
                                verbose=1 if self.config.DEBUG_VERBOSE else 0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)] if self.config.EARLY_STOPPING_PATIENCE > 0 else [])
            if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history
            print(f"    Model training finished for fold {fold_num + 1}.")

            print("    Evaluating model on validation set...")
            y_val_fold_true_np_list = []
            y_pred_proba_list = []

            val_ds_eval = tf.data.Dataset.from_generator(val_generator_func_for_eval, output_signature=output_signature)
            val_ds_eval = val_ds_eval.prefetch(tf.data.AUTOTUNE)

            eval_iterations = num_val_batches if num_val_batches > 0 else (1 if len(val_pairs_fold) > 0 else 0)

            for x_batch_val, y_batch_val in val_ds_eval.take(eval_iterations):
                y_val_fold_true_np_list.append(y_batch_val.numpy())
                y_pred_proba_list.append(model.predict_on_batch(x_batch_val).flatten())

            if not y_val_fold_true_np_list:
                print(f"    Warning: No data yielded by validation generator for fold {fold_num + 1}. Skipping metrics.")
                current_metrics = {'precision_sklearn': np.nan, 'recall_sklearn': np.nan, 'f1_sklearn': np.nan, 'auc_sklearn': np.nan}
            else:
                y_val_fold_true_np = np.concatenate(y_val_fold_true_np_list)
                y_pred_proba = np.concatenate(y_pred_proba_list)
                y_pred_class = (y_pred_proba > 0.5).astype(int)

                current_metrics = {
                    'precision_sklearn': precision_score(y_val_fold_true_np, y_pred_class, zero_division=0),
                    'recall_sklearn': recall_score(y_val_fold_true_np, y_pred_class, zero_division=0),
                    'f1_sklearn': f1_score(y_val_fold_true_np, y_pred_class, zero_division=0)
                }
                if len(np.unique(y_val_fold_true_np)) > 1:
                    current_metrics['auc_sklearn'] = roc_auc_score(y_val_fold_true_np, y_pred_proba)
                    if fold_num == 0:
                        fpr, tpr, _ = roc_curve(y_val_fold_true_np, y_pred_proba)
                        aggregated_results['roc_data_representative'] = (fpr, tpr, current_metrics['auc_sklearn'])
                else:
                    current_metrics['auc_sklearn'] = 0.5
                    if fold_num == 0: aggregated_results['roc_data_representative'] = (np.array([0, 1]), np.array([0, 1]), 0.5)

            for k_val_table in self.config.EVAL_K_VALUES_FOR_TABLE:
                current_metrics[f'hits_at_{k_val_table}'] = 0.0  # Placeholder, implement if needed
                current_metrics[f'ndcg_at_{k_val_table}'] = 0.0  # Placeholder, implement if needed

            fold_metrics_list.append(current_metrics)
            print(f"    Fold {fold_num + 1} Metrics: {current_metrics}")
            del model, history, train_ds, val_ds_for_fit, val_ds_eval
            gc.collect()
            tf.keras.backend.clear_session()
            print(f"    Fold {fold_num + 1} completed in {time.monotonic() - fold_start_time:.2f}s.")  # MODIFICATION

        if fold_metrics_list:
            metrics_keys = fold_metrics_list[0].keys() if fold_metrics_list else []
            for key in metrics_keys:
                values = [fm.get(key, np.nan) for fm in fold_metrics_list]
                aggregated_results[f'test_{key}'] = np.nanmean(values)
                if self.config.EVAL_N_FOLDS > 1:
                    aggregated_results[f'test_{key}_std'] = np.nanstd(values)
                else:
                    aggregated_results[f'test_{key}_std'] = 0.0

            aggregated_results['fold_f1_scores'] = [fm.get('f1_sklearn', np.nan) for fm in fold_metrics_list]
            aggregated_results['fold_auc_scores'] = [fm.get('auc_sklearn', np.nan) for fm in fold_metrics_list]
            for k_val_table in self.config.EVAL_K_VALUES_FOR_TABLE:
                aggregated_results[f'fold_hits_at_{k_val_table}_scores'] = [fm.get(f'hits_at_{k_val_table}', np.nan) for fm in fold_metrics_list]
                aggregated_results[f'fold_ndcg_at_{k_val_table}_scores'] = [fm.get(f'ndcg_at_{k_val_table}', np.nan) for fm in fold_metrics_list]

        print(f"CV workflow for {embedding_name} finished in {time.monotonic() - cv_start_time:.2f}s.")  # MODIFICATION
        if self.config.DEBUG_VERBOSE: print(f"  Aggregated results for {embedding_name}: {aggregated_results}")
        return aggregated_results

    def run(self, use_dummy_data: bool = False, parent_run_id: Optional[str] = None):
        pipeline_start_time = time.monotonic()  # MODIFICATION
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

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
        reporter = EvaluationReporter(base_output_dir=output_dir, k_vals_table=self.config.EVAL_K_VALUES_FOR_TABLE)

        DataUtils.print_header("Loading Interaction Pairs")
        load_pairs_start_time = time.monotonic()  # MODIFICATION
        all_pairs_initial_load: List[Tuple[str, str, int]] = []
        streaming_batch_size = self.config.EVAL_BATCH_SIZE * 100
        positive_stream = GroundTruthLoader.stream_interaction_pairs(pos_fp, 1, batch_size=streaming_batch_size, random_state=self.config.RANDOM_STATE)
        for batch in positive_stream: all_pairs_initial_load.extend(batch)
        print(f"  Loaded {len(all_pairs_initial_load)} positive pairs.")
        current_pos_count = len(all_pairs_initial_load)

        negative_stream = GroundTruthLoader.stream_interaction_pairs(neg_fp, 0, batch_size=streaming_batch_size, sample_n=self.config.SAMPLE_NEGATIVE_PAIRS, random_state=self.config.RANDOM_STATE)
        for batch in negative_stream: all_pairs_initial_load.extend(batch)
        print(f"  Loaded {len(all_pairs_initial_load) - current_pos_count} negative pairs.")

        if not all_pairs_initial_load:
            print("CRITICAL: No interaction pairs were loaded. Exiting evaluation.")
            return
        print(f"Total pairs loaded: {len(all_pairs_initial_load)} in {time.monotonic() - load_pairs_start_time:.2f}s.")  # MODIFICATION
        random.shuffle(all_pairs_initial_load)

        all_required_protein_ids = GroundTruthLoader.get_required_ids_from_files([pos_fp, neg_fp])
        print(f"Found {len(all_required_protein_ids)} unique protein IDs across all interaction files.")

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
                        print("  Loading required embeddings into memory for CV...")
                        load_mem_start_time = time.monotonic()  # MODIFICATION
                        current_protein_embeddings_dict = {
                            pid: protein_embeddings_loader[pid]
                            for pid in all_required_protein_ids if pid in protein_embeddings_loader
                        }
                        loaded_emb_dtype = next(iter(current_protein_embeddings_dict.values())).dtype if current_protein_embeddings_dict else 'N/A'
                        print(f"  Loaded {len(current_protein_embeddings_dict)} embeddings (dtype: {loaded_emb_dtype}) into memory in {time.monotonic() - load_mem_start_time:.2f}s.")  # MODIFICATION

                        if not current_protein_embeddings_dict:
                            print(f"  No embeddings loaded into memory for {emb_name}. Skipping CV.")
                            if mlflow_active and run: mlflow.log_param("status", "no_embeddings_loaded_for_pairs")
                            continue

                        pairs_for_cv = []
                        missing_from_loaded_dict = 0
                        for p1, p2, label in all_pairs_initial_load:
                            if p1 in current_protein_embeddings_dict and p2 in current_protein_embeddings_dict:
                                pairs_for_cv.append((p1, p2, label))
                            else:
                                missing_from_loaded_dict += 1

                        if missing_from_loaded_dict > 0:
                            print(f"  Note: {missing_from_loaded_dict} pairs were further removed because one/both proteins were not in the loaded embedding dictionary.")

                        if not pairs_for_cv:
                            print(f"  No pairs remain after ensuring both proteins have loaded embeddings for {emb_name}. Skipping CV.")
                            if mlflow_active and run: mlflow.log_param("status", "no_valid_pairs_after_emb_load")
                            continue
                        print(f"  Proceeding with {len(pairs_for_cv)} pairs for CV for {emb_name}.")

                        results = self._run_cv_workflow(emb_name, pairs_for_cv, current_protein_embeddings_dict)
                        all_cv_results_list.append(results)

                        if mlflow_active and run and results:
                            metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
                            mlflow.log_metrics(metrics_to_log)
                            if results.get('notes'): mlflow.log_param("notes", results['notes'])
                        if self.config.PLOT_TRAINING_HISTORY and results and results.get('history_dict_fold1'):
                            history_plot_path = reporter.plot_training_history(results['history_dict_fold1'], results['embedding_name'])
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
                summary_path = reporter.write_summary_file(
                    all_cv_results_list, self.config.EVAL_MAIN_EMBEDDING_FOR_STATS,
                    'test_auc_sklearn', self.config.EVAL_STATISTICAL_TEST_ALPHA
                )
                roc_plot_path = reporter.plot_roc_curves(all_cv_results_list)
                comparison_chart_path = reporter.plot_comparison_charts(all_cv_results_list)

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
        DataUtils.print_header(f"PPI Evaluation Pipeline ({run_type}) FINISHED in {time.monotonic() - pipeline_start_time:.2f}s")  # MODIFICATION