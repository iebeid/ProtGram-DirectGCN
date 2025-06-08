# ==============================================================================
# MODULE: pipeline/evaluator.py
# PURPOSE: Contains the complete workflow for evaluating one or more sets of
#          protein embeddings on a link prediction task.
# ==============================================================================

import os
import shutil
import numpy as np
import pandas as pd
import time
import gc
import h5py
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from typing import List, Optional, Dict, Any, Set, Tuple

# Import from our new project structure
from config import Config
from utils.diagnostics import check_h5_embeddings
from utils.data_loader import load_interaction_pairs
from utils.graph_processing import EdgeFeatureProcessor
from utils.reporting import plot_training_history, plot_roc_curves, plot_comparison_charts, write_summary_file
from models.mlp import build_mlp_model  # We will create this models/mlp.py file next


def _create_dummy_data(base_dir: str, num_proteins: int, embedding_dim: int, num_pos: int, num_neg: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Generates dummy data for a quick test run of the evaluation pipeline."""
    dummy_data_dir = os.path.join(base_dir, "dummy_data_temp")
    os.makedirs(dummy_data_dir, exist_ok=True)
    print(f"Creating dummy data in: {dummy_data_dir}")
    protein_ids = [f"P{i:04d}" for i in range(num_proteins)]

    # Create dummy embedding file
    dummy_emb_file = os.path.join(dummy_data_dir, "dummy_embeddings.h5")
    with h5py.File(dummy_emb_file, 'w') as hf:
        for pid in protein_ids:
            hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float32))

    # Create dummy interaction files
    dummy_pos_path = os.path.join(dummy_data_dir, "dummy_pos.csv")
    pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)])
    pos_pairs.to_csv(dummy_pos_path, header=False, index=False)

    dummy_neg_path = os.path.join(dummy_data_dir, "dummy_neg.csv")
    neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)])
    neg_pairs.to_csv(dummy_neg_path, header=False, index=False)

    dummy_emb_config = [{"path": dummy_emb_file, "name": "DummyEmb"}]
    return dummy_pos_path, dummy_neg_path, dummy_emb_config


def _run_cv_workflow(embedding_name: str, X_full: np.ndarray, y_full: np.ndarray, config: Config) -> Dict[str, Any]:
    """The core CV worker function that trains and evaluates the MLP."""
    aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'history_dict_fold1': {}}
    if len(np.unique(y_full)) < 2:
        aggregated_results['notes'] = "Single class in dataset."
        return aggregated_results

    skf = StratifiedKFold(n_splits=config.EVAL_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics_list: List[Dict[str, Any]] = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold_num + 1}/{config.EVAL_N_FOLDS} for {embedding_name} ---")
        X_train, y_train, X_val, y_val = X_full[train_idx], y_full[train_idx], X_full[val_idx], y_full[val_idx]

        # Use from_tensor_slices as it's more direct when data is in memory
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config.EVAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        mlp_params = {
            'dense1_units': config.EVAL_MLP_DENSE1_UNITS, 'dropout1_rate': config.EVAL_MLP_DROPOUT1_RATE,
            'dense2_units': config.EVAL_MLP_DENSE2_UNITS, 'dropout2_rate': config.EVAL_MLP_DROPOUT2_RATE,
            'l2_reg': config.EVAL_MLP_L2_REG
        }
        model = build_mlp_model(X_train.shape[1], mlp_params, config.EVAL_LEARNING_RATE)

        history = model.fit(train_ds, epochs=config.EVAL_EPOCHS, validation_data=val_ds, verbose=1 if config.DEBUG_VERBOSE else 0)
        if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history

        # Calculate metrics using sklearn for consistency
        y_pred_proba = model.predict(val_ds).flatten()
        y_pred_class = (y_pred_proba > 0.5).astype(int)

        current_metrics = {
            'precision_sklearn': precision_score(y_val, y_pred_class, zero_division=0),
            'recall_sklearn': recall_score(y_val, y_pred_class, zero_division=0),
            'f1_sklearn': f1_score(y_val, y_pred_class, zero_division=0),
            'auc_sklearn': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        }

        if fold_num == 0 and current_metrics['auc_sklearn'] > 0.5:
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            aggregated_results['roc_data_representative'] = (fpr, tpr, current_metrics['auc_sklearn'])

        fold_metrics_list.append(current_metrics)
        del model, history, train_ds, val_ds;
        gc.collect();
        tf.keras.backend.clear_session()

    # Aggregate results
    if fold_metrics_list:
        for key in fold_metrics_list[0].keys():
            aggregated_results[f'test_{key}'] = np.mean([fm.get(key, np.nan) for fm in fold_metrics_list])

    aggregated_results['fold_f1_scores'] = [fm.get('f1_sklearn', np.nan) for fm in fold_metrics_list]
    aggregated_results['fold_auc_scores'] = [fm.get('auc_sklearn', np.nan) for fm in fold_metrics_list]

    return aggregated_results


# --- Main Orchestration Function for this Module ---
def run_evaluation(config: Config, use_dummy_data: bool = False):
    """
    The main entry point for the evaluation pipeline step.
    """
    if use_dummy_data:
        print("\n" + "#" * 30 + " RUNNING DUMMY TEST CASE " + "#" * 30)
        output_dir = os.path.join(config.EVALUATION_RESULTS_DIR, "dummy_run_output")
        pos_fp, neg_fp, emb_configs = _create_dummy_data(
            base_dir=config.BASE_OUTPUT_DIR, num_proteins=50, embedding_dim=16, num_pos=100, num_neg=100)
    else:
        print("\n" + "#" * 30 + " RUNNING NORMAL EVALUATION CASE " + "#" * 30)
        output_dir = config.EVALUATION_RESULTS_DIR
        pos_fp, neg_fp, emb_configs = config.INTERACTIONS_POSITIVE_PATH, config.INTERACTIONS_NEGATIVE_PATH, config.LP_EMBEDDING_FILES_TO_EVALUATE

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    positive_pairs = load_interaction_pairs(pos_fp, 1, random_state=config.RANDOM_STATE)
    negative_pairs = load_interaction_pairs(neg_fp, 0, sample_n=config.SAMPLE_NEGATIVE_PAIRS, random_state=config.RANDOM_STATE)
    if not positive_pairs: print("CRITICAL: No positive pairs loaded."); return

    required_ids = set(p for pair in positive_pairs + negative_pairs for p in pair[:2])
    print(f"Found {len(required_ids)} unique protein IDs that need embeddings.")

    all_cv_results = []
    for emb_config in emb_configs:
        print(f"\n{'=' * 25} Processing: {emb_config['name']} {'=' * 25}")
        if config.PERFORM_H5_INTEGRITY_CHECK:
            check_h5_embeddings(emb_config['path'])

        protein_embeddings = FileOps.load_h5_embeddings_selectively(emb_config['path'], required_ids)
        if not protein_embeddings: print(f"Skipping {emb_config['name']}: No relevant embeddings loaded."); continue

        X_full, y_full = EdgeFeatureProcessor.create_edge_embeddings(positive_pairs + negative_pairs, protein_embeddings, method=config.EVAL_EDGE_EMBEDDING_METHOD)
        if X_full is None: print(f"Skipping {emb_config['name']}: Failed to create edge features."); continue

        results = _run_cv_workflow(emb_config['name'], X_full, y_full, config)
        all_cv_results.append(results)

        if config.PLOT_TRAINING_HISTORY and results.get('history_dict_fold1'):
            plot_training_history(results['history_dict_fold1'], results['embedding_name'], plots_dir)
        del protein_embeddings, X_full, y_full, results;
        gc.collect()

    if all_cv_results:
        print("\n" + "=" * 25 + " FINAL AGGREGATE RESULTS " + "=" * 25)
        write_summary_file(all_cv_results, output_dir, config.EVAL_MAIN_EMBEDDING_FOR_STATS, 'test_auc_sklearn', config.EVAL_STATISTICAL_TEST_ALPHA, config.EVAL_K_VALUES_FOR_TABLE)
        plot_roc_curves(all_cv_results, plots_dir)
        plot_comparison_charts(all_cv_results, config.EVAL_K_VALUES_FOR_TABLE, plots_dir)
    else:
        print("\nNo results generated from any configurations.")

    if use_dummy_data and config.CLEANUP_DUMMY_DATA:
        try:
            shutil.rmtree(os.path.join(config.BASE_OUTPUT_DIR, "dummy_data_temp"))
            print("Cleaned up dummy data directory.")
        except Exception as e:
            print(f"Error cleaning up dummy data: {e}")

    print(f"--- Evaluation Pipeline Finished. Results in: {output_dir} ---")