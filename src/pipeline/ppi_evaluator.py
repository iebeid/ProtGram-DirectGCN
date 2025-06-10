# ==============================================================================
# MODULE: pipeline/5_evaluator.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# VERSION: 2.3 (Stream-Loading and Memory-Efficient CV)
# ==============================================================================

import os
import shutil
import numpy as np
import pandas as pd
import time
import gc
import h5py
import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from typing import List, Optional, Dict, Any, Set, Tuple
import mlflow

# Import from our new project structure
from src.config import Config
from src.utils.file_handler import check_h5_embeddings
from src.utils.data_loader import stream_interaction_pairs, H5EmbeddingLoader, get_required_ids_from_files
from src.utils.graph_processor import EdgeFeatureProcessor
from src.utils.reporter import plot_training_history, plot_roc_curves, plot_comparison_charts, write_summary_file
from src.models.mlp import build_mlp_model


def _create_dummy_data(base_dir: str, num_proteins: int, embedding_dim: int, num_pos: int, num_neg: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Generates dummy data for a quick test run of the evaluation pipeline."""
    dummy_data_dir = os.path.join(base_dir, "dummy_data_temp");
    os.makedirs(dummy_data_dir, exist_ok=True)
    print(f"Creating dummy data in: {dummy_data_dir}")
    protein_ids = [f"P{i:04d}" for i in range(num_proteins)]

    dummy_emb_file = os.path.join(dummy_data_dir, "dummy_embeddings.h5")
    with h5py.File(dummy_emb_file, 'w') as hf:
        for pid in protein_ids: hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float32))

    dummy_pos_path = os.path.join(dummy_data_dir, "dummy_pos.csv")
    pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)]);
    pos_pairs.to_csv(dummy_pos_path, header=False, index=False)

    dummy_neg_path = os.path.join(dummy_data_dir, "dummy_neg.csv")
    neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)]);
    neg_pairs.to_csv(dummy_neg_path, header=False, index=False)

    dummy_emb_config = [{"path": dummy_emb_file, "name": "DummyEmb"}]
    return dummy_pos_path, dummy_neg_path, dummy_emb_config


def _run_cv_workflow(embedding_name: str, all_pairs: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray], config: Config) -> Dict[str, Any]:
    """
    The core CV worker function that trains and evaluates the MLP.
    This version is memory-efficient, creating feature matrices per fold.
    """
    aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}, 'notes': ""}

    labels = np.array([p[2] for p in all_pairs])
    if len(np.unique(labels)) < 2:
        aggregated_results['notes'] = "Single class in dataset.";
        return aggregated_results

    skf = StratifiedKFold(n_splits=config.EVAL_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics_list: List[Dict[str, Any]] = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_pairs)), labels)):
        print(f"\n--- Fold {fold_num + 1}/{config.EVAL_N_FOLDS} for {embedding_name} ---")

        train_pairs = [all_pairs[i] for i in train_idx]
        val_pairs = [all_pairs[i] for i in val_idx]

        print("Creating features for training set...")
        X_train, y_train = EdgeFeatureProcessor.create_edge_embeddings(train_pairs, protein_embeddings, method=config.EVAL_EDGE_EMBEDDING_METHOD)
        print("Creating features for validation set...")
        X_val, y_val = EdgeFeatureProcessor.create_edge_embeddings(val_pairs, protein_embeddings, method=config.EVAL_EDGE_EMBEDDING_METHOD)

        if X_train is None or X_val is None:
            print(f"Skipping fold {fold_num + 1} due to feature creation failure.")
            continue

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        mlp_params = {'dense1_units': config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': config.EVAL_MLP_DROPOUT1_RATE, 'dense2_units': config.EVAL_MLP_DENSE2_UNITS, 'dropout2_rate': config.EVAL_MLP_DROPOUT2_rate,
                      'l2_reg': config.EVAL_MLP_L2_REG}
        model = build_mlp_model(X_train.shape[1], mlp_params, config.EVAL_LEARNING_RATE)

        history = model.fit(train_ds, epochs=config.EVAL_EPOCHS, validation_data=val_ds, verbose=1 if config.DEBUG_VERBOSE else 0)
        if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history

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
            current_metrics['auc_sklearn'] = 0.5

        fold_metrics_list.append(current_metrics)
        del model, history, train_ds, val_ds, X_train, y_train, X_val, y_val;
        gc.collect();
        tf.keras.backend.clear_session()

    if fold_metrics_list:
        for key in fold_metrics_list[0].keys():
            mean_val = np.nanmean([fm.get(key) for fm in fold_metrics_list])
            aggregated_results[f'test_{key}'] = mean_val
        f1_scores = [fm.get('f1_sklearn') for fm in fold_metrics_list]
        auc_scores = [fm.get('auc_sklearn') for fm in fold_metrics_list]
        aggregated_results['test_f1_sklearn_std'] = np.nanstd(f1_scores)
        aggregated_results['test_auc_sklearn_std'] = np.nanstd(auc_scores)
        aggregated_results['fold_f1_scores'] = f1_scores
        aggregated_results['fold_auc_scores'] = auc_scores

    return aggregated_results


# --- Main Orchestration Function for this Module ---
def run_evaluation(config: Config, use_dummy_data: bool = False, parent_run_id: Optional[str] = None):
    """Main entry point for the evaluation pipeline step."""
    if use_dummy_data:
        print("\n" + "#" * 30 + " RUNNING DUMMY EVALUATION " + "#" * 30)
        output_dir = os.path.join(config.EVALUATION_RESULTS_DIR, "dummy_run_output")
        pos_fp, neg_fp, emb_configs = _create_dummy_data(base_dir=config.BASE_OUTPUT_DIR, num_proteins=50, embedding_dim=16, num_pos=100, num_neg=100)
    else:
        print("\n" + "#" * 30 + " RUNNING MAIN EVALUATION " + "#" * 30)
        output_dir = config.EVALUATION_RESULTS_DIR
        emb_configs = getattr(config, 'LP_EMBEDDING_FILES_TO_EVALUATE', [])
        pos_fp, neg_fp = config.INTERACTIONS_POSITIVE_PATH, config.INTERACTIONS_NEGATIVE_PATH
        if not emb_configs:
            print("Warning: 'LP_EMBEDDING_FILES_TO_EVALUATE' is empty. No evaluation will run.")

    plots_dir = os.path.join(output_dir, "plots");
    os.makedirs(output_dir, exist_ok=True);
    os.makedirs(plots_dir, exist_ok=True)

    # Use the memory-efficient streaming loader to get all pairs
    # We collect the pairs into a list to be able to use StratifiedKFold, which needs all labels at once.
    # This is still far more memory-efficient than creating the full feature matrix.
    all_pairs = []
    positive_stream = stream_interaction_pairs(pos_fp, 1, batch_size=config.EVAL_BATCH_SIZE, random_state=config.RANDOM_STATE)
    for batch in positive_stream:
        all_pairs.extend(batch)

    negative_stream = stream_interaction_pairs(neg_fp, 0, batch_size=config.EVAL_BATCH_SIZE, sample_n=config.SAMPLE_NEGATIVE_PAIRS, random_state=config.RANDOM_STATE)
    for batch in negative_stream:
        all_pairs.extend(batch)

    if not all_pairs:
        print("CRITICAL: No interaction pairs were loaded from the streams. Exiting.")
        return

    random.shuffle(all_pairs)

    required_ids = get_required_ids_from_files([pos_fp, neg_fp])
    print(f"Found {len(required_ids)} unique protein IDs that need embeddings.")

    all_cv_results = []
    for emb_config in emb_configs:
        run_name = emb_config['name']
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            print(f"\n{'=' * 25} Processing: {emb_config['name']} {'=' * 25}")

            if config.USE_MLFLOW:
                mlflow.log_params({"embedding_name": emb_config['name'], "embedding_path": emb_config.get('path', 'N/A'), "edge_embedding_method": config.EVAL_EDGE_EMBEDDING_METHOD, "n_folds": config.EVAL_N_FOLDS,
                                   "epochs": config.EVAL_EPOCHS, "batch_size": config.EVAL_BATCH_SIZE, "learning_rate": config.EVAL_LEARNING_RATE, })

            if config.PERFORM_H5_INTEGRITY_CHECK:
                check_h5_embeddings(emb_config['path'])

            # protein_embeddings = load_h5_embeddings_selectively(emb_config['path'], required_ids)
            # if not protein_embeddings:
            #     print(f"Skipping {emb_config['name']}: No relevant embeddings loaded.")
            #     continue
            #
            # results = _run_cv_workflow(emb_config['name'], all_pairs, protein_embeddings, config)
            # all_cv_results.append(results)
            #
            # if config.USE_MLFLOW:
            #     metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
            #     mlflow.log_metrics(metrics_to_log)
            #
            # if config.PLOT_TRAINING_HISTORY and results.get('history_dict_fold1'):
            #     history_plot_path = plot_training_history(results['history_dict_fold1'], results['embedding_name'], plots_dir)
            #     if config.USE_MLFLOW and history_plot_path:
            #         mlflow.log_artifact(history_plot_path, "plots")
            #
            # del protein_embeddings;
            # gc.collect()

            try:
                # Use the H5EmbeddingLoader as a context manager
                with H5EmbeddingLoader(emb_config['path']) as protein_embeddings:
                    results = _run_cv_workflow(emb_config['name'], all_pairs, protein_embeddings, config)
                    all_cv_results.append(results)

                    if config.USE_MLFLOW:
                        metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float, np.number))}
                        mlflow.log_metrics(metrics_to_log)

                    if config.PLOT_TRAINING_HISTORY and results.get('history_dict_fold1'):
                        history_plot_path = plot_training_history(results['history_dict_fold1'], results['embedding_name'], plots_dir)
                        if config.USE_MLFLOW and history_plot_path:
                            mlflow.log_artifact(history_plot_path, "plots")

            except FileNotFoundError as e:
                print(f"ERROR: Could not process {emb_config['name']}. Reason: {e}")
                continue

            # Garbage collect to be safe, though context manager handles the file.
            gc.collect()

    if all_cv_results:
        with mlflow.start_run(run_id=parent_run_id):
            print("\n" + "=" * 25 + " FINAL AGGREGATE RESULTS " + "=" * 25)
            summary_path = write_summary_file(all_cv_results, output_dir, config.EVAL_MAIN_EMBEDDING_FOR_STATS, 'test_auc_sklearn', config.EVAL_STATISTICAL_TEST_ALPHA, config.EVAL_K_VALUES_FOR_TABLE)
            roc_plot_path = plot_roc_curves(all_cv_results, plots_dir)
            comparison_chart_path = plot_comparison_charts(all_cv_results, config.EVAL_K_VALUES_FOR_TABLE, plots_dir)

            if config.USE_MLFLOW:
                if summary_path: mlflow.log_artifact(summary_path, "summary")
                if roc_plot_path: mlflow.log_artifact(roc_plot_path, "summary_plots")
                if comparison_chart_path: mlflow.log_artifact(comparison_chart_path, "summary_plots")
    else:
        print("\nNo results generated from any configurations.")

    if use_dummy_data and config.CLEANUP_DUMMY_DATA:
        try:
            shutil.rmtree(os.path.join(config.BASE_OUTPUT_DIR, "dummy_data_temp"));
            print("Cleaned up dummy data directory.")
        except Exception as e:
            print(f"Error cleaning up dummy data: {e}")

    print(f"--- Evaluation Pipeline Finished. Results in: {output_dir} ---")
