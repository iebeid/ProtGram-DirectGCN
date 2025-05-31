# bioeval_cv_worker.py
import numpy as np
import pandas as pd
import time
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
import tensorflow as tf
from typing import List, Optional, Dict, Any, Set, Tuple

# Assumes bioeval_utils.py is in the same directory or Python path
from bioeval_utils import build_mlp_model, DEBUG_VERBOSE  # Import necessary functions/globals


# --- CV Workflow ---
def main_workflow_cv(embedding_name: str,
                     protein_embeddings: Dict[str, np.ndarray],
                     positive_pairs: List[Tuple[str, str, int]],
                     negative_pairs: List[Tuple[str, str, int]],
                     # required_protein_ids_for_interactions: Set[str], # Not strictly needed here as protein_embeddings should be pre-filtered
                     mlp_params: Dict[str, Any],
                     edge_embedding_method: str,
                     n_folds: int,
                     random_state: int,
                     max_train_samples_cv: Optional[int],
                     max_val_samples_cv: Optional[int],
                     max_shuffle_buffer_size: int,
                     batch_size: int,
                     epochs: int,
                     learning_rate: float,
                     k_values_for_ranking: List[int]
                     ) -> Dict[str, Any]:  # Return type changed to Dict

    # Default result structure, matching what's expected by plotting/table functions
    aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'training_time': 0.0,
                                          'history_dict_fold1': {},
                                          'roc_data_representative': (np.array([]), np.array([]), 0.0), 'notes': "",
                                          'fold_f1_scores': [], 'fold_auc_scores': [],
                                          **{k: 0.0 for k in ['test_loss', 'test_accuracy_keras', 'test_auc_keras',
                                                              'test_precision_keras', 'test_recall_keras',
                                                              'test_precision_sklearn', 'test_recall_sklearn',
                                                              'test_f1_sklearn', 'test_auc_sklearn']},
                                          **{f'test_hits_at_{k_val}': 0.0 for k_val in k_values_for_ranking},
                                          **{f'test_ndcg_at_{k_val}': 0.0 for k_val in
                                             k_values_for_ranking}}  # Use passed k_values

    if not protein_embeddings:
        aggregated_results['notes'] = "No embeddings provided to CV workflow."
        print(f"No embeddings for {embedding_name}. Skip CV.");
        return aggregated_results

    # protein_embeddings should already be filtered to relevant IDs by the orchestrator

    if not positive_pairs and not negative_pairs:
        aggregated_results['notes'] = "No interaction pairs.";
        print(f"No interaction pairs for {embedding_name}.");
        return aggregated_results

    all_interaction_pairs = positive_pairs + negative_pairs
    if not all_interaction_pairs:
        aggregated_results['notes'] = "No combined interactions.";
        print(f"No combined interactions for {embedding_name}.");
        return aggregated_results

    # Graph class is now in bioeval_utils, but create_edge_embeddings is part of it.
    # For this split, we'll assume Graph is available or pass its methods.
    # To simplify, let's assume create_edge_embeddings is also moved to utils or passed if Graph isn't.
    # For this example, I'll assume it's available via an imported Graph utility class.
    from bioeval_utils import Graph  # Assuming Graph is in utils now
    graph_processor = Graph()
    X_full, y_full = graph_processor.create_edge_embeddings(all_interaction_pairs, protein_embeddings,
                                                            method=edge_embedding_method)

    if X_full is None or y_full is None or len(X_full) == 0:
        aggregated_results['notes'] = "Dataset creation failed (no edge features)."
        print(f"Dataset creation failed for {embedding_name}.");
        return aggregated_results
    if DEBUG_VERBOSE: print(
        f"Total samples for {embedding_name} for CV: {len(y_full)} (+:{np.sum(y_full == 1)}, -:{np.sum(y_full == 0)})")
    if len(np.unique(y_full)) < 2:
        aggregated_results['notes'] = "Single class in dataset y_full for CV."
        print(f"Warning: Only one class for {embedding_name}. CV not meaningful.");
        return aggregated_results

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics_list: List[Dict[str, Any]] = [];
    total_training_time = 0.0

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold_num + 1}/{n_folds} for {embedding_name} ---")
        X_kfold_train, y_kfold_train = X_full[train_idx], y_full[train_idx];
        X_kfold_val, y_kfold_val = X_full[val_idx], y_full[val_idx]
        X_train_use, y_train_use = X_kfold_train, y_kfold_train
        if max_train_samples_cv is not None and X_kfold_train.shape[0] > max_train_samples_cv:
            if DEBUG_VERBOSE: print(f"Sampling train: {X_kfold_train.shape[0]}->{max_train_samples_cv}")
            idx = np.random.choice(X_kfold_train.shape[0], max_train_samples_cv, replace=False);
            X_train_use, y_train_use = X_kfold_train[idx], y_kfold_train[idx]
        X_val_use, y_val_use = X_kfold_val, y_kfold_val
        if max_val_samples_cv is not None and X_kfold_val.shape[0] > max_val_samples_cv:
            if DEBUG_VERBOSE: print(f"Sampling val: {X_kfold_val.shape[0]}->{max_val_samples_cv}")
            idx = np.random.choice(X_kfold_val.shape[0], max_val_samples_cv, replace=False);
            X_val_use, y_val_use = X_kfold_val[idx], y_kfold_val[idx]

        current_fold_metrics: Dict[str, Any] = {'fold': fold_num + 1}
        if X_train_use.shape[0] == 0:
            print(f"Fold {fold_num + 1}: Training data empty. Skipping.");
            [current_fold_metrics.update({k_metric: np.nan}) for k_metric in aggregated_results if
             'test_' in k_metric or 'hits_at' in k_metric or 'ndcg_at' in k_metric];
            fold_metrics_list.append(current_fold_metrics);
            continue

        shuffle_buffer = min(X_train_use.shape[0], max_shuffle_buffer_size)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_use, y_train_use)).shuffle(shuffle_buffer).batch(
            batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_use, y_val_use)).batch(batch_size).prefetch(
            tf.data.AUTOTUNE) if X_val_use.shape[0] > 0 else None

        edge_dim = X_train_use.shape[1]
        model = build_mlp_model(edge_dim, learning_rate, mlp_params=mlp_params)
        if fold_num == 0 and DEBUG_VERBOSE: model.summary(print_fn=print)
        print(f"Training Fold {fold_num + 1} ({X_train_use.shape[0]} samples)...")
        start_time = time.time();
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1 if DEBUG_VERBOSE else 0);
        fold_training_time = time.time() - start_time
        total_training_time += fold_training_time;
        current_fold_metrics['training_time'] = fold_training_time
        if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history

        y_val_eval_np = np.array(y_val_use).flatten()
        if X_val_use.shape[0] > 0 and val_ds:
            eval_res = model.evaluate(val_ds, verbose=0);
            keras_keys = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                          'test_recall_keras']
            for name, val in zip(keras_keys, eval_res): current_fold_metrics[name] = val
            y_pred_proba_val = model.predict(X_val_use, batch_size=batch_size).flatten();
            y_pred_class_val = (y_pred_proba_val > 0.5).astype(int)
            current_fold_metrics.update(
                {'test_precision_sklearn': precision_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                 'test_recall_sklearn': recall_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                 'test_f1_sklearn': f1_score(y_val_eval_np, y_pred_class_val, zero_division=0)})
            if len(np.unique(y_val_eval_np)) > 1:
                current_fold_metrics['test_auc_sklearn'] = roc_auc_score(y_val_eval_np,
                                                                         y_pred_proba_val);fpr, tpr, _ = roc_curve(
                    y_val_eval_np, y_pred_proba_val);aggregated_results['roc_data_representative'] = (fpr, tpr,
                                                                                                      current_fold_metrics[
                                                                                                          'test_auc_sklearn']) if fold_num == 0 else \
                    aggregated_results['roc_data_representative']
            else:
                current_fold_metrics['test_auc_sklearn'] = 0.0;aggregated_results['roc_data_representative'] = (
                    np.array([]), np.array([]), 0.0) if fold_num == 0 else aggregated_results['roc_data_representative']
            desc_indices = np.argsort(y_pred_proba_val)[::-1];
            sorted_y_val = y_val_eval_np[desc_indices]
            for k_rank in k_values_for_ranking:  # Use passed k_values_for_ranking
                eff_k = min(k_rank, len(sorted_y_val));
                current_fold_metrics[f'test_hits_at_{k_rank}'] = np.sum(sorted_y_val[:eff_k] == 1) if eff_k > 0 else 0;
                current_fold_metrics[f'test_ndcg_at_{k_rank}'] = ndcg_score(np.asarray([y_val_eval_np]),
                                                                            np.asarray([y_pred_proba_val]), k=eff_k,
                                                                            ignore_ties=True) if eff_k > 0 and len(
                    np.unique(y_val_eval_np)) > 1 else 0.0
        else:
            print(f"Fold {fold_num + 1}: Eval skipped. Metrics NaN.");
            metric_keys_to_nan = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                                  'test_recall_keras', 'test_precision_sklearn', 'test_recall_sklearn',
                                  'test_f1_sklearn', 'test_auc_sklearn'] + [f'test_hits_at_{k}' for k in
                                                                            k_values_for_ranking] + [f'test_ndcg_at_{k}'
                                                                                                     for k in
                                                                                                     k_values_for_ranking]
            for key_nan in metric_keys_to_nan: current_fold_metrics[key_nan] = np.nan
        fold_metrics_list.append(current_fold_metrics)
        del model, history, train_ds, val_ds;
        gc.collect();
        tf.keras.backend.clear_session()

    if not fold_metrics_list: aggregated_results['notes'] = "No folds completed.";print(
        f"No folds completed for {embedding_name}.");return aggregated_results

    for key_to_avg in aggregated_results.keys():
        if key_to_avg not in ['embedding_name', 'history_dict_fold1', 'roc_data_representative', 'notes',
                              'fold_f1_scores', 'fold_auc_scores', 'training_time', 'test_f1_sklearn_std',
                              'test_auc_sklearn_std']:
            valid_fold_values = [fm.get(key_to_avg) for fm in fold_metrics_list if
                                 fm.get(key_to_avg) is not None and not np.isnan(fm.get(key_to_avg))]
            aggregated_results[key_to_avg] = np.mean(valid_fold_values) if valid_fold_values else 0.0
    aggregated_results['training_time'] = total_training_time / len(fold_metrics_list) if fold_metrics_list else 0.0
    aggregated_results['fold_f1_scores'] = [fm.get('test_f1_sklearn', np.nan) for fm in fold_metrics_list]
    aggregated_results['fold_auc_scores'] = [fm.get('test_auc_sklearn', np.nan) for fm in fold_metrics_list]
    f1_valid = [s for s in aggregated_results['fold_f1_scores'] if not np.isnan(s)];
    aggregated_results['test_f1_sklearn_std'] = np.std(f1_valid) if len(f1_valid) > 1 else 0.0
    auc_valid = [s for s in aggregated_results['fold_auc_scores'] if not np.isnan(s)];
    aggregated_results['test_auc_sklearn_std'] = np.std(auc_valid) if len(auc_valid) > 1 else 0.0
    print(f"===== Finished CV for {embedding_name} =====");
    return aggregated_results
