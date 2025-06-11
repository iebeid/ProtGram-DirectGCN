# ==============================================================================
# MODULE: pipeline/ppi_main.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# VERSION: 3.0 (Refactored into PPIPipeline class)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import random
import shutil
from typing import List, Optional, Dict, Any, Tuple

import h5py
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# Import from our new project structure
from src.config import Config
from src.models.mlp import MLP
from src.utils.data_utils import DataUtils, GroundTruthLoader
from src.utils.models_utils import EmbeddingLoader, EmbeddingProcessor
from src.utils.results_utils import EvaluationReporter


class PPIPipeline:
    """
    Orchestrates the Protein-Protein Interaction (PPI) link prediction
    evaluation pipeline.
    """

    def __init__(self, config: Config):
        """
        Initializes the PPIPipeline.

        Args:
            config (Config): The configuration object for the pipeline.
        """
        self.config = config

    @staticmethod
    def _create_dummy_data(base_dir: str, num_proteins: int, embedding_dim: int, num_pos: int, num_neg: int) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Generates dummy data for a quick test run of the evaluation pipeline."""
        dummy_data_dir = os.path.join(base_dir, "dummy_data_temp")
        os.makedirs(dummy_data_dir, exist_ok=True)
        print(f"Creating dummy data in: {dummy_data_dir}")
        protein_ids = [f"P{i:04d}" for i in range(num_proteins)]

        dummy_emb_file = os.path.join(dummy_data_dir, "dummy_embeddings.h5")
        with h5py.File(dummy_emb_file, 'w') as hf:
            for pid in protein_ids:
                hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float32))

        dummy_pos_path = os.path.join(dummy_data_dir, "dummy_pos.csv")
        pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)])
        pos_pairs.to_csv(dummy_pos_path, header=False, index=False)

        dummy_neg_path = os.path.join(dummy_data_dir, "dummy_neg.csv")
        neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)])
        neg_pairs.to_csv(dummy_neg_path, header=False, index=False)

        dummy_emb_config = [{"path": dummy_emb_file, "name": "DummyEmb"}]
        return dummy_pos_path, dummy_neg_path, dummy_emb_config

    def _run_cv_workflow(self, embedding_name: str, all_pairs: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        The core CV worker function that trains and evaluates the MLP.
        This version is memory-efficient, creating feature matrices per fold.
        Uses self.config for parameters.
        """
        aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}, 'notes': ""}

        labels = np.array([p[2] for p in all_pairs])
        if len(np.unique(labels)) < 2:
            aggregated_results['notes'] = "Single class in dataset."
            return aggregated_results

        skf = StratifiedKFold(n_splits=self.config.EVAL_N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        fold_metrics_list: List[Dict[str, Any]] = []

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_pairs)), labels)):
            print(f"\n--- Fold {fold_num + 1}/{self.config.EVAL_N_FOLDS} for {embedding_name} ---")

            train_pairs = [all_pairs[i] for i in train_idx]
            val_pairs = [all_pairs[i] for i in val_idx]

            print("Creating features for training set...")
            # Note: DataLoader.create_edge_embeddings is static, so it's fine.
            # If it were an instance method of DataLoader, you'd need an instance.
            X_train, y_train = EmbeddingProcessor.create_edge_embeddings(train_pairs, protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD)
            print("Creating features for validation set...")
            X_val, y_val = EmbeddingProcessor.create_edge_embeddings(val_pairs, protein_embeddings, method=self.config.EVAL_EDGE_EMBEDDING_METHOD)

            if X_train is None or X_val is None or X_train.size == 0 or X_val.size == 0:
                print(f"Skipping fold {fold_num + 1} due to feature creation failure or empty features.")
                continue

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(self.config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            mlp_params = {'dense1_units': self.config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': self.config.EVAL_MLP_DROPOUT1_RATE, 'dense2_units': self.config.EVAL_MLP_DENSE2_UNITS,
                          'dropout2_rate': self.config.EVAL_MLP_DROPOUT2_RATE,  # Corrected typo from _rate to _RATE
                          'l2_reg': self.config.EVAL_MLP_L2_REG}
            model = MLP(X_train.shape[1], mlp_params, self.config.EVAL_LEARNING_RATE).build()

            history = model.fit(train_ds, epochs=self.config.EVAL_EPOCHS, validation_data=val_ds, verbose=1 if self.config.DEBUG_VERBOSE else 0)
            if fold_num == 0:
                aggregated_results['history_dict_fold1'] = history.history

            y_pred_proba = model.predict(val_ds).flatten()
            y_pred_class = (y_pred_proba > 0.5).astype(int)

            current_metrics = {'precision_sklearn': precision_score(y_val, y_pred_class, zero_division=0), 'recall_sklearn': recall_score(y_val, y_pred_class, zero_division=0),
                               'f1_sklearn': f1_score(y_val, y_pred_class, zero_division=0)}

            if len(np.unique(y_val)) > 1:
                current_metrics['auc_sklearn'] = roc_auc_score(y_val, y_pred_proba)
                if fold_num == 0:
                    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
                    aggregated_results['roc_data_representative'] = (fpr, tpr, current_metrics['auc_sklearn'])
            else:
                current_metrics['auc_sklearn'] = 0.5  # Default for single class

            fold_metrics_list.append(current_metrics)
            del model, history, train_ds, val_ds, X_train, y_train, X_val, y_val
            gc.collect()
            tf.keras.backend.clear_session()

        if fold_metrics_list:
            for key in fold_metrics_list[0].keys():  # Iterate over keys of the first fold's metrics
                mean_val = np.nanmean([fm.get(key, np.nan) for fm in fold_metrics_list])  # Use np.nan for missing keys
                aggregated_results[f'test_{key}'] = mean_val
            f1_scores = [fm.get('f1_sklearn', np.nan) for fm in fold_metrics_list]
            auc_scores = [fm.get('auc_sklearn', np.nan) for fm in fold_metrics_list]
            aggregated_results['test_f1_sklearn_std'] = np.nanstd(f1_scores)
            aggregated_results['test_auc_sklearn_std'] = np.nanstd(auc_scores)
            aggregated_results['fold_f1_scores'] = f1_scores
            aggregated_results['fold_auc_scores'] = auc_scores

        return aggregated_results

    def run(self, use_dummy_data: bool = False, parent_run_id: Optional[str] = None):
        """Main entry point for the evaluation pipeline step."""
        if use_dummy_data:
            print("\n" + "#" * 30 + " RUNNING DUMMY EVALUATION " + "#" * 30)
            # Use self._create_dummy_data (static method)
            output_dir = os.path.join(str(self.config.EVALUATION_RESULTS_DIR), "dummy_run_output")
            pos_fp, neg_fp, emb_configs = PPIPipeline._create_dummy_data(base_dir=str(self.config.BASE_OUTPUT_DIR), num_proteins=50, embedding_dim=16, num_pos=100, num_neg=100)
        else:
            print("\n" + "#" * 30 + " RUNNING MAIN EVALUATION " + "#" * 30)
            output_dir = str(self.config.EVALUATION_RESULTS_DIR)
            emb_configs = getattr(self.config, 'LP_EMBEDDING_FILES_TO_EVALUATE', [])
            pos_fp = str(self.config.INTERACTIONS_POSITIVE_PATH)
            neg_fp = str(self.config.INTERACTIONS_NEGATIVE_PATH)
            if not emb_configs:
                print("Warning: 'LP_EMBEDDING_FILES_TO_EVALUATE' is empty. No evaluation will run.")
                return  # Exit if no embeddings to evaluate

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        all_pairs = []
        # GroundTruthLoader methods are static
        positive_stream = GroundTruthLoader.stream_interaction_pairs(pos_fp, 1, batch_size=self.config.EVAL_BATCH_SIZE, random_state=self.config.RANDOM_STATE)
        for batch in positive_stream:
            all_pairs.extend(batch)

        negative_stream = GroundTruthLoader.stream_interaction_pairs(neg_fp, 0, batch_size=self.config.EVAL_BATCH_SIZE, sample_n=self.config.SAMPLE_NEGATIVE_PAIRS, random_state=self.config.RANDOM_STATE)
        for batch in negative_stream:
            all_pairs.extend(batch)

        if not all_pairs:
            print("CRITICAL: No interaction pairs were loaded from the streams. Exiting.")
            return

        random.shuffle(all_pairs)  # Shuffle once after collecting all pairs

        required_ids = GroundTruthLoader.get_required_ids_from_files([pos_fp, neg_fp])
        print(f"Found {len(required_ids)} unique protein IDs that need embeddings.")

        all_cv_results = []
        for emb_config in emb_configs:
            run_name = emb_config['name']
            # MLflow context management
            mlflow_active = self.config.USE_MLFLOW
            if mlflow_active:
                # Ensure parent_run_id is only used if it's a nested run scenario
                current_run_context = mlflow.start_run(run_name=run_name, nested=True if parent_run_id else False)
            else:
                # Create a dummy context manager if MLflow is not used
                from contextlib import nullcontext
                current_run_context = nullcontext()

            with current_run_context as run:  # run will be None if mlflow is not active or start_run fails
                print(f"\n{'=' * 25} Processing: {emb_config['name']} {'=' * 25}")

                if mlflow_active and run:  # Check if run object is valid
                    mlflow.log_params(
                        {"embedding_name": emb_config['name'], "embedding_path": emb_config.get('path', 'N/A'), "edge_embedding_method": self.config.EVAL_EDGE_EMBEDDING_METHOD, "n_folds": self.config.EVAL_N_FOLDS,
                         "epochs": self.config.EVAL_EPOCHS, "batch_size": self.config.EVAL_BATCH_SIZE, "learning_rate": self.config.EVAL_LEARNING_RATE, })

                if self.config.PERFORM_H5_INTEGRITY_CHECK:
                    # DataUtils methods are static
                    DataUtils.check_h5_embeddings_integrity(str(emb_config['path']))

                try:
                    # EmbeddingLoader is a context manager
                    with EmbeddingLoader(str(emb_config['path'])) as protein_embeddings:
                        # Call instance method _run_cv_workflow
                        results = self._run_cv_workflow(emb_config['name'], all_pairs, protein_embeddings)
                        all_cv_results.append(results)

                        if mlflow_active and run and results:  # Check if run and results are valid
                            metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
                            mlflow.log_metrics(metrics_to_log)

                        if self.config.PLOT_TRAINING_HISTORY and results and results.get('history_dict_fold1'):
                            # EvaluationReporter methods are static
                            history_plot_path = EvaluationReporter.plot_training_history(results['history_dict_fold1'], results['embedding_name'], plots_dir)
                            if mlflow_active and run and history_plot_path:
                                mlflow.log_artifact(history_plot_path, "plots")

                except FileNotFoundError as e:
                    print(f"ERROR: Could not process {emb_config['name']}. Reason: {e}")
                    continue  # Continue to the next embedding configuration
                except Exception as e_gen:
                    print(f"UNEXPECTED ERROR during processing for {emb_config['name']}: {e_gen}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Garbage collect to be safe, though context manager handles the file.
                gc.collect()

        if all_cv_results:
            # For the aggregate summary, use the parent_run_id if provided,
            # otherwise, it will create a new top-level run if MLflow is active.
            mlflow_active_summary = self.config.USE_MLFLOW
            if mlflow_active_summary:
                summary_run_context = mlflow.start_run(run_id=parent_run_id if parent_run_id else None, run_name="Evaluation_Summary" if not parent_run_id else None, nested=bool(parent_run_id))
            else:
                from contextlib import nullcontext
                summary_run_context = nullcontext()

            with summary_run_context as summary_run:
                print("\n" + "=" * 25 + " FINAL AGGREGATE RESULTS " + "=" * 25)
                summary_path = EvaluationReporter.write_summary_file(all_cv_results, output_dir, self.config.EVAL_MAIN_EMBEDDING_FOR_STATS, 'test_auc_sklearn', self.config.EVAL_STATISTICAL_TEST_ALPHA,
                                                                     self.config.EVAL_K_VALUES_FOR_TABLE)
                roc_plot_path = EvaluationReporter.plot_roc_curves(all_cv_results, plots_dir)
                comparison_chart_path = EvaluationReporter.plot_comparison_charts(all_cv_results, self.config.EVAL_K_VALUES_FOR_TABLE, plots_dir)

                if mlflow_active_summary and summary_run:
                    if summary_path: mlflow.log_artifact(summary_path, "summary")
                    if roc_plot_path: mlflow.log_artifact(roc_plot_path, "summary_plots")
                    if comparison_chart_path: mlflow.log_artifact(comparison_chart_path, "summary_plots")
        else:
            print("\nNo results generated from any configurations.")

        if use_dummy_data and self.config.CLEANUP_DUMMY_DATA:
            try:
                shutil.rmtree(os.path.join(str(self.config.BASE_OUTPUT_DIR), "dummy_data_temp"))
                print("Cleaned up dummy data directory.")
            except Exception as e:
                print(f"Error cleaning up dummy data: {e}")

        print(f"--- Evaluation Pipeline Finished. Results in: {output_dir} ---")
