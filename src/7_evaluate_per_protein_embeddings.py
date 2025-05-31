import os
import time
import gc
from typing import List, Optional, Dict, Any, Set, Tuple, Union  # Added all used typing imports
import math  # Added import math

import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt
import tensorflow as tf

# --- USER CONFIGURATION SECTION ---
POSITIVE_INTERACTIONS_PATH = os.path.normpath('C:/tmp/Models/ground_truth/positive_interactions.csv')
NEGATIVE_INTERACTIONS_PATH = os.path.normpath('C:/tmp/Models/ground_truth/negative_interactions.csv')
DEFAULT_EMBEDDING_LOADER = 'load_h5_embeddings_selectively'  # Default to the new selective loader
EMBEDDING_FILES_TO_COMPARE = [
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Directed_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "per-protein-10-digcn"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Undirected_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "GCN"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/prot-t5-uniprot-per-residue.h5",  # This is the large file
        "name": "ProtT5_Pooled_Mean"
    }
]
# --- END OF USER CONFIGURATION SECTION ---

# --- SCRIPT BEHAVIOR AND MODEL PARAMETERS ---
DEBUG_VERBOSE = True
RANDOM_STATE = 42
EDGE_EMBEDDING_METHOD = 'concatenate'
N_FOLDS = 5
MAX_TRAIN_SAMPLES_CV = None  # Set to an int to sample, None to use all
MAX_VAL_SAMPLES_CV = None  # Set to an int to sample, None to use all
MAX_SHUFFLE_BUFFER_SIZE = 200000

MLP_DENSE1_UNITS = 128;
MLP_DROPOUT1_RATE = 0.4
MLP_DENSE2_UNITS = 64;
MLP_DROPOUT2_RATE = 0.4
MLP_L2_REG = 0.001

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
K_VALUES_FOR_TABLE_DISPLAY = [50, 100]

MAIN_EMBEDDING_NAME = "ProtT5_Pooled_Mean"
STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'
STATISTICAL_TEST_ALPHA = 0.05


# --- Placeholder for classes/functions that might be in biohelper.py ---
# If these are not in your biohelper.py, you might need to define them here
# or ensure biohelper.py is in your Python path and contains these.

class ProteinFileOps:  # Placeholder
    @staticmethod
    def load_interaction_pairs(filepath: str, label: int) -> List[Tuple[str, str, int]]:
        filepath = os.path.normpath(filepath)
        print(f"Loading interaction pairs from: {filepath} (label: {label})...")
        if not os.path.exists(filepath):
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str)
            pairs = [(str(row.protein1), str(row.protein2), label) for _, row in df.iterrows()]
            print(f"Successfully loaded {len(pairs)} pairs from {filepath}.")
            return pairs
        except Exception as e:
            print(f"Error loading interaction file {filepath}: {e}")
            return []


class FileOps:
    @staticmethod
    def load_h5_embeddings_selectively(h5_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        h5_path = os.path.normpath(h5_path)
        load_mode = "selectively" if required_ids else "all keys"
        required_count_info = f"for {len(required_ids)} IDs" if required_ids else ""
        print(f"Loading embeddings from: {h5_path} ({load_mode} {required_count_info})...")

        if not os.path.exists(h5_path):
            print(f"Warning: Embedding file not found: {h5_path}")
            return {}

        protein_embeddings: Dict[str, np.ndarray] = {}
        loaded_count = 0
        try:
            with h5py.File(h5_path, 'r') as hf:
                keys_to_iterate = list(hf.keys())  # Get all keys first

                target_keys = []
                if required_ids:
                    for key in keys_to_iterate:
                        if key in required_ids:
                            target_keys.append(key)
                    if DEBUG_VERBOSE: print(
                        f"  Will attempt to load {len(target_keys)} matching required IDs from {os.path.basename(h5_path)}.")
                else:
                    target_keys = keys_to_iterate

                if not target_keys and required_ids:  # No overlap between H5 keys and required_ids
                    print(f"  No matching keys found in {os.path.basename(h5_path)} for the required IDs.")

                for key in tqdm(target_keys, desc=f"  Reading {os.path.basename(h5_path)}", leave=False,
                                unit="protein"):
                    if isinstance(hf[key], h5py.Dataset):
                        # Load only if key is in required_ids OR if no filter is applied
                        protein_embeddings[key] = hf[key][:].astype(np.float32)
                        loaded_count += 1
            print(f"Loaded {loaded_count} embeddings from {os.path.basename(h5_path)}.")
        except Exception as e:
            print(f"Error loading HDF5 file {h5_path}: {e}")
            return {}
        return protein_embeddings

    @staticmethod
    def load_custom_embeddings(p_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        print(f"Warning: 'load_custom_embeddings' is a placeholder. Path: {p_path}")
        return {}


class Graph:
    def __init__(self):
        self.embedding_dim: Optional[int] = None

    def get_embedding_dimension(self, protein_embeddings: Dict[str, np.ndarray]) -> int:
        if not protein_embeddings: self.embedding_dim = 0; return 0
        for emb_vec in protein_embeddings.values():
            if emb_vec is not None and hasattr(emb_vec, 'shape') and len(emb_vec.shape) > 0:
                self.embedding_dim = emb_vec.shape[-1];
                print(f"Inferred embedding dimension: {self.embedding_dim}");
                return self.embedding_dim
        self.embedding_dim = 0;
        print("Warning: Could not infer embedding dimension.");
        return 0

    def create_edge_embeddings(self, interaction_pairs: List[Tuple[str, str, int]],
                               protein_embeddings: Dict[str, np.ndarray],
                               method: str = 'concatenate') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"Creating edge embeddings using method: {method}...")
        if not protein_embeddings: print("Protein embeddings empty. Cannot create edge features."); return None, None
        if self.embedding_dim is None or self.embedding_dim == 0: self.get_embedding_dimension(protein_embeddings)
        if self.embedding_dim == 0: print("Embedding dimension is 0. Cannot create edge features."); return None, None
        edge_features = [];
        labels = [];
        skipped_pairs_count = 0
        for p1_id, p2_id, label in tqdm(interaction_pairs, desc="Creating Edge Features", leave=False):
            emb1, emb2 = protein_embeddings.get(p1_id), protein_embeddings.get(p2_id)
            if emb1 is not None and emb2 is not None:
                if emb1.ndim > 1: emb1 = emb1.flatten()
                if emb2.ndim > 1: emb2 = emb2.flatten()
                if emb1.shape[0] != self.embedding_dim or emb2.shape[
                    0] != self.embedding_dim: skipped_pairs_count += 1; continue
                if method == 'concatenate':
                    feature = np.concatenate((emb1, emb2))
                elif method == 'average':
                    feature = (emb1 + emb2) / 2.0
                elif method == 'hadamard':
                    feature = emb1 * emb2
                elif method == 'subtract':
                    feature = np.abs(emb1 - emb2)
                else:
                    feature = np.concatenate((emb1, emb2))  # Fallback
                edge_features.append(feature);
                labels.append(label)
            else:
                skipped_pairs_count += 1
        if skipped_pairs_count > 0: print(f"Skipped {skipped_pairs_count} pairs (embeddings not found/dim mismatch).")
        if not edge_features: print("No edge features created."); return None, None
        print(f"Created {len(edge_features)} edge features with dimension {edge_features[0].shape[0]}.")
        return np.array(edge_features, dtype=np.float32), np.array(labels, dtype=np.int32)


# --- End Placeholder ---

LOADER_FUNCTION_MAP = {
    'load_h5_embeddings_selectively': lambda p, req_ids=None: FileOps.load_h5_embeddings_selectively(p,
                                                                                                     required_ids=req_ids),
    'load_h5_embeddings': lambda p, req_ids=None: FileOps.load_h5_embeddings_selectively(p, required_ids=req_ids),
    'load_custom_embeddings': lambda p, req_ids=None: FileOps.load_custom_embeddings(p, required_ids=req_ids)
}


def build_mlp_model(input_shape: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(MLP_DENSE1_UNITS, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(MLP_L2_REG)),
        tf.keras.layers.Dropout(MLP_DROPOUT1_RATE),
        tf.keras.layers.Dense(MLP_DENSE2_UNITS, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(MLP_L2_REG)),
        tf.keras.layers.Dropout(MLP_DROPOUT2_RATE),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc_keras'),
                           tf.keras.metrics.Precision(name='precision_keras'),
                           tf.keras.metrics.Recall(name='recall_keras')])
    return model


def plot_training_history(history_dict: dict[str, Any], model_name: str, fold_num: Optional[int] = None):
    title_suffix = f" (Fold {fold_num})" if fold_num is not None else " (Representative Fold)"
    if not history_dict or not any(
            isinstance(val_list, list) and len(val_list) > 0 for val_list in history_dict.values()):
        if DEBUG_VERBOSE: print(f"Plotting: No history data for {model_name}{title_suffix}. Plot empty.")
        return
    plt.figure(figsize=(12, 5));
    plotted_loss_axes = False
    if 'loss' in history_dict and history_dict['loss']: plt.subplot(1, 2, 1); plt.plot(history_dict['loss'],
                                                                                       label='Training Loss',
                                                                                       marker='.' if len(history_dict[
                                                                                                             'loss']) < 5 else None); plotted_loss_axes = True
    if 'val_loss' in history_dict and history_dict['val_loss']:
        if not plotted_loss_axes: plt.subplot(1, 2, 1)
        plt.plot(history_dict['val_loss'], label='Validation Loss',
                 marker='.' if len(history_dict['val_loss']) < 5 else None);
        plotted_loss_axes = True
    if plotted_loss_axes: plt.title(f'Model Loss: {model_name}{title_suffix}');plt.ylabel('Loss');plt.xlabel(
        'Epoch');plt.legend()
    plotted_acc_axes = False
    if 'accuracy' in history_dict and history_dict['accuracy']: plt.subplot(1, 2, 2); plt.plot(history_dict['accuracy'],
                                                                                               label='Training Accuracy',
                                                                                               marker='.' if len(
                                                                                                   history_dict[
                                                                                                       'accuracy']) < 5 else None); plotted_acc_axes = True
    if 'val_accuracy' in history_dict and history_dict['val_accuracy']:
        if not plotted_acc_axes: plt.subplot(1, 2, 2)
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy',
                 marker='.' if len(history_dict['val_accuracy']) < 5 else None);
        plotted_acc_axes = True
    if plotted_acc_axes: plt.title(f'Model Accuracy: {model_name}{title_suffix}');plt.ylabel('Accuracy');plt.xlabel(
        'Epoch');plt.legend()
    if not (plotted_loss_axes or plotted_acc_axes): plt.close(); return
    plt.suptitle(f"Training History: {model_name}{title_suffix}", fontsize=16);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.show()


def plot_roc_curves(results_list: list[dict[str, Any]]):
    plt.figure(figsize=(10, 8));
    plotted_anything = False
    for result in results_list:
        roc_data_to_plot = result.get('roc_data_representative')
        if roc_data_to_plot:
            fpr, tpr, auc_val_representative = roc_data_to_plot
            avg_auc = result.get('test_auc_sklearn',
                                 auc_val_representative if auc_val_representative is not None else 0.0)
            if fpr is not None and tpr is not None and avg_auc is not None and len(fpr) > 0 and len(tpr) > 0: plt.plot(
                fpr, tpr, lw=2,
                label=f"{result.get('embedding_name', 'Unknown')} (Avg AUC = {avg_auc:.3f})");plotted_anything = True
    if not plotted_anything and DEBUG_VERBOSE: print("Plotting: No valid ROC data.")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate (FPR)');
    plt.ylabel('True Positive Rate (TPR)');
    plt.title('ROC Curves Comparison');
    plt.legend(loc="lower right");
    plt.grid(True);
    plt.show()


def plot_comparison_charts(results_list: list[dict[str, Any]]):
    if not results_list: if
    DEBUG_VERBOSE: print("Plotting: No results for comparison charts.");
    return
    metrics_to_plot = {'Accuracy': 'test_accuracy_keras', 'Precision': 'test_precision_sklearn',
                       'Recall': 'test_recall_sklearn', 'F1-Score': 'test_f1_sklearn', 'AUC': 'test_auc_sklearn',
                       f'Hits@{K_VALUES_FOR_TABLE_DISPLAY[0]}': f'test_hits_at_{K_VALUES_FOR_TABLE_DISPLAY[0]}',
                       f'NDCG@{K_VALUES_FOR_TABLE_DISPLAY[0]}': f'test_ndcg_at_{K_VALUES_FOR_TABLE_DISPLAY[0]}'}
    if len(K_VALUES_FOR_TABLE_DISPLAY) > 1: metrics_to_plot[
        f'Hits@{K_VALUES_FOR_TABLE_DISPLAY[1]}'] = f'test_hits_at_{K_VALUES_FOR_TABLE_DISPLAY[1]}';metrics_to_plot[
        f'NDCG@{K_VALUES_FOR_TABLE_DISPLAY[1]}'] = f'test_ndcg_at_{K_VALUES_FOR_TABLE_DISPLAY[1]}';
    embedding_names = [res.get('embedding_name', f'Run {i + 1}') for i, res in enumerate(results_list)];
    num_perf_metrics = len(metrics_to_plot)
    cols = min(4, num_perf_metrics + 1);
    rows = math.ceil((num_perf_metrics + 1) / cols)  # Used math.ceil
    plt.figure(figsize=(max(15, len(embedding_names) * cols * 0.8), rows * 5));
    plot_index = 1
    for metric_display_name, metric_key in metrics_to_plot.items():
        plt.subplot(rows, cols, plot_index);
        values = [res.get(metric_key) if res.get(metric_key) is not None else 0.0 for res in results_list];
        values = [v if isinstance(v, (int, float)) else 0.0 for v in values]
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(embedding_names))));
        plt.title(metric_display_name);
        plt.ylabel('Score');
        plt.xticks(rotation=45, ha="right")
        current_max_val = max(values) if values else 0.0;
        plot_upper_limit = 1.05
        if "Hits@" in metric_display_name:
            plot_upper_limit = max(current_max_val * 1.15 if current_max_val > 0 else 10, 10)
        elif values:
            plot_upper_limit = max(1.05, current_max_val * 1.15 if current_max_val > 0 else 0.1)
        plt.ylim(0, plot_upper_limit)
        for bar_idx, bar in enumerate(bars): yval = values[bar_idx];plt.text(bar.get_x() + bar.get_width() / 2.0,
                                                                             yval + 0.01 * plot_upper_limit,
                                                                             f'{yval:.3f}', ha='center', va='bottom')
        plot_index += 1
    if plot_index <= rows * cols:
        plt.subplot(rows, cols, plot_index);
        training_times = [res.get('training_time') if res.get('training_time') is not None else 0.0 for res in
                          results_list];
        training_times = [t if isinstance(t, (int, float)) else 0.0 for t in training_times]
        bars = plt.bar(embedding_names, training_times, color=plt.cm.plasma(np.linspace(0, 1, len(embedding_names))));
        plt.title('Training Time');
        plt.ylabel('Seconds');
        plt.xticks(rotation=45, ha="right")
        max_time_val = max(training_times) if training_times else 1.0;
        plot_upper_limit_time = max_time_val * 1.15 if max_time_val > 0 else 10
        plt.ylim(0, plot_upper_limit_time)
        for bar_idx, bar in enumerate(bars): yval = training_times[bar_idx];plt.text(
            bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plot_upper_limit_time, f'{yval:.2f}s', ha='center',
            va='bottom')
    plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.show()


def print_results_table(results_list: list[dict[str, Any]], is_cv: bool = False):
    if not results_list: if
    DEBUG_VERBOSE: print("\nNo results to display in table.");
    return
    suffix = " (Avg over Folds)" if is_cv else "";
    print(f"\n\n--- Overall Performance Comparison Table{suffix} ---")
    metric_keys_headers = [('embedding_name', "Embedding Name")];
    metric_keys_headers.append(('training_time', f"Train Time (s){suffix}"));
    metric_keys_headers.append(('test_loss', f"Val Loss{suffix}"));
    metric_keys_headers.append(('test_accuracy_keras', f"Accuracy{suffix}"));
    metric_keys_headers.append(('test_precision_sklearn', f"Precision{suffix}"));
    metric_keys_headers.append(('test_recall_sklearn', f"Recall{suffix}"));
    metric_keys_headers.append(('test_f1_sklearn', f"F1-Score{suffix}"));
    metric_keys_headers.append(('test_auc_sklearn', f"AUC{suffix}"))
    for k in K_VALUES_FOR_TABLE_DISPLAY: metric_keys_headers.append((f'test_hits_at_{k}', f"Hits@{k}{suffix}"))
    for k in K_VALUES_FOR_TABLE_DISPLAY: metric_keys_headers.append((f'test_ndcg_at_{k}', f"NDCG@{k}{suffix}"))
    if is_cv: metric_keys_headers.append(('test_f1_sklearn_std', "F1 StdDev"));metric_keys_headers.append(
        ('test_auc_sklearn_std', "AUC StdDev"))
    headers = [h for _, h in metric_keys_headers];
    metric_keys_to_extract = [k for k, _ in metric_keys_headers]
    table_data = [headers]
    for res_dict in results_list:
        row = []
        for key in metric_keys_to_extract:
            val = res_dict.get(key);
            is_placeholder = False
            if isinstance(val, (int, float)):
                if val == -1.0 and 'loss' not in key: is_placeholder = True
                if val == 0 and ('hits_at_' in key or 'ndcg_at_' in key) and res_dict.get('notes'):
                    if "Single class" in res_dict['notes'] or "Empty" in res_dict['notes']: is_placeholder = True
            if val is None or is_placeholder:
                row.append("N/A")
            elif isinstance(val, float):
                if key == 'training_time' or '_std' in key:
                    row.append(f"{val:.2f}")
                else:
                    row.append(f"{val:.4f}")
            elif isinstance(val, int) and "hits_at_" in key:
                row.append(str(val))
            else:
                row.append(str(val))
        table_data.append(row)
    if len(table_data) <= 1: if
    DEBUG_VERBOSE: print("No data rows for table.");
    return
    column_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)];
    fmt_str = " | ".join([f"{{:<{w}}}" for w in column_widths]);
    print(fmt_str.format(*headers));
    print("-+-".join(["-" * w for w in column_widths]))
    for i in range(1, len(table_data)): print(fmt_str.format(*table_data[i]))
    print("-------------------------------------------\n")


def perform_statistical_tests(results_list: list[dict[str, Any]], main_embedding_name_cfg: Optional[str],
                              metric_key_cfg: str, alpha_cfg: float):
    if not main_embedding_name_cfg: print("\nStat Tests: Main embedding name not specified. Skipping.");return
    if len(results_list) < 2: print("\nStat Tests: Need at least two methods to compare.");return
    print(
        f"\n\n--- Statistical Comparison vs '{main_embedding_name_cfg}' on '{metric_key_cfg}' (Alpha={alpha_cfg}) ---")
    key_for_fold_scores = None
    if metric_key_cfg == 'test_auc_sklearn':
        key_for_fold_scores = 'fold_auc_scores'
    elif metric_key_cfg == 'test_f1_sklearn':
        key_for_fold_scores = 'fold_f1_scores'
    else:
        print(f"Stat Tests: Metric key '{metric_key_cfg}' not supported for fold scores.");return
    main_model_res = next((res for res in results_list if res['embedding_name'] == main_embedding_name_cfg), None)
    if not main_model_res: print(f"Stat Tests: Main model '{main_embedding_name_cfg}' not found.");return
    main_model_scores_all_folds = main_model_res.get(key_for_fold_scores, []);
    main_model_scores_valid = [s for s in main_model_scores_all_folds if not np.isnan(s)]
    if len(main_model_scores_valid) < 2: print(
        f"Stat Tests: Not enough valid scores for main model '{main_embedding_name_cfg}'.");return
    other_model_results_list = [res for res in results_list if res['embedding_name'] != main_embedding_name_cfg]
    if not other_model_results_list: print("No other models to compare.");return
    header_parts = [f"{'Compared Embedding':<30}", f"{'Wilcoxon p-val':<15}", f"{'Signif. (p<{alpha_cfg})':<18}",
                    f"{'Pearson r':<10}", f"{'r-squared':<10}"];
    header = " | ".join(header_parts);
    print(header);
    print("-" * len(header))
    for other_model_res in other_model_results_list:
        other_model_name = other_model_res['embedding_name'];
        other_model_scores_all_folds = other_model_res.get(key_for_fold_scores, []);
        other_model_scores_valid = [s for s in other_model_scores_all_folds if not np.isnan(s)]
        if len(main_model_scores_valid) != len(other_model_scores_valid) or len(main_model_scores_valid) < 2: print(
            f"{other_model_name:<30} | {'N/A (score list mismatch/few scores)':<15} | {'N/A':<18} | {'N/A':<10} | {'N/A':<10}");continue
        current_main_scores = np.array(main_model_scores_valid);
        current_other_scores = np.array(other_model_scores_valid)
        p_value_wilcoxon, pearson_r_val, r_squared_val = 1.0, np.nan, np.nan;
        significance_diff = "No";
        correlation_note = ""
        try:
            if not np.allclose(current_main_scores, current_other_scores):
                stat, p_value_wilcoxon = wilcoxon(current_main_scores, current_other_scores, alternative='two-sided',
                                                  zero_method='pratt')
                if p_value_wilcoxon < alpha_cfg:
                    mean_main = np.mean(current_main_scores);mean_other = np.mean(
                        current_other_scores);significance_diff = f"Yes (Main Better)" if mean_main > mean_other else (
                        f"Yes (Main Worse)" if mean_main < mean_other else "Yes (Diff, Means Eq.)")
                else:
                    significance_diff = "No"
            else:
                p_value_wilcoxon = 1.0;significance_diff = "No (Identical Scores)"
        except ValueError as e:
            p_value_wilcoxon = 1.0;significance_diff = f"N/A (Wilcoxon Err)"; if
        DEBUG_VERBOSE: print(f"Wilcoxon error for {other_model_name}:{e}")
        try:
            if len(np.unique(current_main_scores)) > 1 and len(np.unique(current_other_scores)) > 1:
                pearson_r_val, p_val_corr = pearsonr(current_main_scores, current_other_scores);
                r_squared_val = pearson_r_val ** 2
                if p_val_corr < alpha_cfg: correlation_note = f"(Corr. p={p_val_corr:.2e})"
        except Exception as e_corr:
            if
        DEBUG_VERBOSE: print(f"Pearson r error for {other_model_name}:{e_corr}")
        print(
            f"{other_model_name:<30} | {p_value_wilcoxon:<15.4f} | {significance_diff:<18} | {pearson_r_val:<10.4f} | {r_squared_val:<10.4f} {correlation_note}")
    print("-" * len(header));
    print(f"Note: Wilcoxon tests difference in paired scores. Pearson r for linear correlation.")


def main_workflow_cv(embedding_name: str, protein_embeddings: dict[str, np.ndarray],
                     positive_pairs: List[Tuple[str, str, int]],
                     negative_pairs: List[Tuple[str, str, int]],
                     required_protein_ids_for_interactions: Set[str]) -> Optional[dict[str, Any]]:
    aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'training_time': 0.0,
                                          'history_dict_fold1': {},
                                          'roc_data_representative': (np.array([]), np.array([]), 0.0), 'notes': "",
                                          'fold_f1_scores': [], 'fold_auc_scores': [],
                                          **{k: 0.0 for k in ['test_loss', 'test_accuracy_keras', 'test_auc_keras',
                                                              'test_precision_keras', 'test_recall_keras',
                                                              'test_precision_sklearn', 'test_recall_sklearn',
                                                              'test_f1_sklearn', 'test_auc_sklearn']},
                                          **{f'test_hits_at_{k_val}': 0.0 for k_val in K_VALUES_FOR_RANKING_METRICS},
                                          **{f'test_ndcg_at_{k_val}': 0.0 for k_val in K_VALUES_FOR_RANKING_METRICS}}

    if not protein_embeddings:
        aggregated_results['notes'] = "No embeddings provided to CV workflow."
        print(f"No embeddings provided for {embedding_name} to CV workflow. Skipping.")
        return aggregated_results

    num_unique_proteins_with_embeddings = len(protein_embeddings)  # Already filtered by selective loader
    print(f"  Using {num_unique_proteins_with_embeddings} pre-filtered protein embeddings for CV for {embedding_name}.")

    if not positive_pairs and not negative_pairs: aggregated_results['notes'] = "No interaction pairs.";print(
        f"No interaction pairs for {embedding_name}.");return aggregated_results
    all_interaction_pairs = positive_pairs + negative_pairs
    if not all_interaction_pairs: aggregated_results['notes'] = "No combined interactions.";print(
        f"No combined interactions for {embedding_name}.");return aggregated_results

    graph_processor = Graph()
    X_full, y_full = graph_processor.create_edge_embeddings(all_interaction_pairs, protein_embeddings,
                                                            method=EDGE_EMBEDDING_METHOD)  # protein_embeddings is already filtered

    if X_full is None or y_full is None or len(X_full) == 0:
        aggregated_results['notes'] = "Dataset creation failed (no edge features)."
        print(
            f"Dataset creation failed for {embedding_name} (X_full or y_full is None or empty after create_edge_embeddings).")
        return aggregated_results
    if DEBUG_VERBOSE: print(
        f"Total samples for {embedding_name} for CV: {len(y_full)} (+:{np.sum(y_full == 1)}, -:{np.sum(y_full == 0)})")

    if len(np.unique(y_full)) < 2:
        aggregated_results['notes'] = "Single class in dataset y_full for CV."
        print(f"Warning: Only one class in y_full for {embedding_name}. CV not performed meaningfully.");
        return aggregated_results

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics_list: List[Dict[str, Any]] = [];
    total_training_time_for_embedding = 0.0

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold_num + 1}/{N_FOLDS} for {embedding_name} ---")
        X_kfold_train, X_kfold_val = X_full[train_idx], X_full[val_idx]
        y_kfold_train, y_kfold_val = y_full[train_idx], y_full[val_idx]
        X_train_to_use, y_train_to_use = X_kfold_train, y_kfold_train
        if MAX_TRAIN_SAMPLES_CV is not None and X_kfold_train.shape[0] > MAX_TRAIN_SAMPLES_CV:
            if DEBUG_VERBOSE: print(f"Sampling train set: {X_kfold_train.shape[0]} -> {MAX_TRAIN_SAMPLES_CV}")
            train_fold_indices = np.random.choice(X_kfold_train.shape[0], MAX_TRAIN_SAMPLES_CV, replace=False)
            X_train_to_use, y_train_to_use = X_kfold_train[train_fold_indices], y_kfold_train[train_fold_indices]
        X_val_to_use, y_val_to_use = X_kfold_val, y_kfold_val
        if MAX_VAL_SAMPLES_CV is not None and X_kfold_val.shape[0] > MAX_VAL_SAMPLES_CV:
            if DEBUG_VERBOSE: print(f"Sampling val set: {X_kfold_val.shape[0]} -> {MAX_VAL_SAMPLES_CV}")
            val_fold_indices = np.random.choice(X_kfold_val.shape[0], MAX_VAL_SAMPLES_CV, replace=False)
            X_val_to_use, y_val_to_use = X_kfold_val[val_fold_indices], y_kfold_val[val_fold_indices]

        current_fold_metrics: Dict[str, Any] = {'fold': fold_num + 1}
        if X_train_to_use.shape[0] == 0:
            print(f"Fold {fold_num + 1}: Training data empty. Skipping.");
            [current_fold_metrics.update({k_metric: np.nan}) for k_metric in aggregated_results if
             'test_' in k_metric or 'hits_at' in k_metric or 'ndcg_at' in k_metric];
            fold_metrics_list.append(current_fold_metrics);
            continue

        shuffle_buffer = min(X_train_to_use.shape[0], MAX_SHUFFLE_BUFFER_SIZE)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_to_use, y_train_to_use)).shuffle(shuffle_buffer).batch(
            BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_to_use, y_val_to_use)).batch(BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE) if X_val_to_use.shape[0] > 0 else None

        edge_dim = X_train_to_use.shape[1]
        model = build_mlp_model(edge_dim, LEARNING_RATE)
        if fold_num == 0 and DEBUG_VERBOSE: model.summary(print_fn=print)
        print(f"Training Fold {fold_num + 1} ({X_train_to_use.shape[0]} samples)...")
        start_time = time.time();
        history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1 if DEBUG_VERBOSE else 0);
        fold_training_time = time.time() - start_time
        total_training_time_for_embedding += fold_training_time;
        current_fold_metrics['training_time'] = fold_training_time
        if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history

        y_val_eval_np = np.array(y_val_to_use).flatten()
        if X_val_to_use.shape[0] > 0 and val_ds:
            eval_res = model.evaluate(val_ds, verbose=0);
            keras_keys = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                          'test_recall_keras']
            for name, val in zip(keras_keys, eval_res): current_fold_metrics[name] = val
            y_pred_proba_val = model.predict(X_val_to_use, batch_size=BATCH_SIZE).flatten();
            y_pred_class_val = (y_pred_proba_val > 0.5).astype(int)
            current_fold_metrics.update(
                {'test_precision_sklearn': precision_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                 'test_recall_sklearn': recall_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                 'test_f1_sklearn': f1_score(y_val_eval_np, y_pred_class_val, zero_division=0)})
            if len(np.unique(y_val_eval_np)) > 1:
                current_fold_metrics['test_auc_sklearn'] = roc_auc_score(y_val_eval_np,
                                                                         y_pred_proba_val); fpr, tpr, _ = roc_curve(
                    y_val_eval_np, y_pred_proba_val); aggregated_results['roc_data_representative'] = (fpr, tpr,
                                                                                                       current_fold_metrics[
                                                                                                           'test_auc_sklearn']) if fold_num == 0 else \
                aggregated_results['roc_data_representative']
            else:
                current_fold_metrics['test_auc_sklearn'] = 0.0; aggregated_results['roc_data_representative'] = (
                    np.array([]), np.array([]), 0.0) if fold_num == 0 else aggregated_results['roc_data_representative']
            desc_indices = np.argsort(y_pred_proba_val)[::-1];
            sorted_y_val = y_val_eval_np[desc_indices]
            for k in K_VALUES_FOR_RANKING_METRICS: eff_k = min(k, len(sorted_y_val)); current_fold_metrics[
                f'test_hits_at_{k}'] = np.sum(sorted_y_val[:eff_k] == 1) if eff_k > 0 else 0; current_fold_metrics[
                f'test_ndcg_at_{k}'] = ndcg_score(np.asarray([y_val_eval_np]), np.asarray([y_pred_proba_val]), k=eff_k,
                                                  ignore_ties=True) if eff_k > 0 and len(
                np.unique(y_val_eval_np)) > 1 else 0.0
        else:
            print(f"Fold {fold_num + 1}: Eval skipped (empty val set). Metrics NaN.");
            metric_keys_to_nan = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                                  'test_recall_keras', 'test_precision_sklearn', 'test_recall_sklearn',
                                  'test_f1_sklearn', 'test_auc_sklearn'] + [f'test_hits_at_{k}' for k in
                                                                            K_VALUES_FOR_RANKING_METRICS] + [
                                     f'test_ndcg_at_{k}' for k in K_VALUES_FOR_RANKING_METRICS]
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
    aggregated_results['training_time'] = total_training_time_for_embedding / len(
        fold_metrics_list) if fold_metrics_list else 0.0
    aggregated_results['fold_f1_scores'] = [fm.get('test_f1_sklearn', np.nan) for fm in fold_metrics_list]
    aggregated_results['fold_auc_scores'] = [fm.get('test_auc_sklearn', np.nan) for fm in fold_metrics_list]
    f1_valid = [s for s in aggregated_results['fold_f1_scores'] if not np.isnan(s)];
    aggregated_results['test_f1_sklearn_std'] = np.std(f1_valid) if len(f1_valid) > 1 else 0.0
    auc_valid = [s for s in aggregated_results['fold_auc_scores'] if not np.isnan(s)];
    aggregated_results['test_auc_sklearn_std'] = np.std(auc_valid) if len(auc_valid) > 1 else 0.0
    print(f"===== Finished CV for {embedding_name} =====");
    return aggregated_results


# --- Main Execution ---
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';
    tf.get_logger().setLevel('ERROR')
    print(f"Numpy Version:  {np.__version__}")
    print(f"Tensorflow Version:  {tf.__version__}")
    print(f"Keras Version:  {tf.keras.__version__}")
    print(tf.config.list_physical_devices('GPU'))

    if not os.path.exists(POSITIVE_INTERACTIONS_PATH) or not os.path.exists(NEGATIVE_INTERACTIONS_PATH):
        print("CRITICAL ERROR: Interaction file paths are invalid. Exiting.");
        exit()

    print("Loading all interaction pairs to determine required proteins...")
    positive_pairs_all = ProteinFileOps.load_interaction_pairs(POSITIVE_INTERACTIONS_PATH, 1)
    negative_pairs_all = ProteinFileOps.load_interaction_pairs(NEGATIVE_INTERACTIONS_PATH, 0)
    all_interaction_pairs_for_ids = positive_pairs_all + negative_pairs_all
    if not all_interaction_pairs_for_ids: print(
        "No interaction pairs loaded. Cannot determine required protein IDs. Exiting."); exit()

    required_protein_ids_for_interactions = set()
    for p1, p2, _ in all_interaction_pairs_for_ids: required_protein_ids_for_interactions.add(
        p1); required_protein_ids_for_interactions.add(p2)
    print(
        f"Found {len(required_protein_ids_for_interactions)} unique protein IDs in interaction files that need embeddings.")

    EMBEDDING_CONFIGURATIONS_PROCESSED: List[Dict[str, Any]] = []
    for item in EMBEDDING_FILES_TO_COMPARE:
        config = {};
        path, name, loader_name_str = None, None, DEFAULT_EMBEDDING_LOADER
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get('path'); name = item.get('name'); loader_name_str = item.get('loader',
                                                                                         DEFAULT_EMBEDDING_LOADER)
        else:
            print(f"Warning: Invalid item in EMBEDDING_FILES_TO_COMPARE: {item}. Skipping."); continue
        if not path: print(f"Warning: Path missing: {item}. Skipping."); continue
        norm_path = os.path.normpath(path)
        if not os.path.exists(norm_path): print(
            f"Warning: Embedding path does not exist: {norm_path}. Skipping."); continue
        config['path'] = norm_path
        config['name'] = name if name else os.path.splitext(os.path.basename(norm_path))[0]
        actual_loader_func = LOADER_FUNCTION_MAP.get(loader_name_str)
        if not actual_loader_func:
            print(f"Warning: Loader '{loader_name_str}' not found. Trying default.");
            actual_loader_func = LOADER_FUNCTION_MAP.get(DEFAULT_EMBEDDING_LOADER)
            if not actual_loader_func: print(
                f"CRITICAL: Default loader '{DEFAULT_EMBEDDING_LOADER}' not found. Skipping {config['name']}."); continue
        config['loader_func'] = actual_loader_func
        EMBEDDING_CONFIGURATIONS_PROCESSED.append(config)

    if not EMBEDDING_CONFIGURATIONS_PROCESSED: print("No valid embedding configurations. Exiting."); exit()

    all_cv_results: list[dict[str, Any]] = []
    for config_item in EMBEDDING_CONFIGURATIONS_PROCESSED:
        print(f"\n{'=' * 25} Processing CV for: {config_item['name']} (Path: {config_item['path']}) {'=' * 25}")
        protein_embeddings = config_item['loader_func'](config_item['path'],
                                                        req_ids=required_protein_ids_for_interactions)

        if protein_embeddings and len(protein_embeddings) > 0:
            actually_loaded_ids = set(protein_embeddings.keys())
            relevant_loaded_ids = actually_loaded_ids.intersection(required_protein_ids_for_interactions)
            if len(relevant_loaded_ids) < 2 and len(protein_embeddings) > 0:
                print(
                    f"Skipping {config_item['name']}: Insufficient relevant protein embeddings loaded ({len(relevant_loaded_ids)} found of {len(required_protein_ids_for_interactions)} needed for interactions). Check ID matching.")
                all_cv_results.append({'embedding_name': config_item['name'],
                                       'notes': f"Insufficient relevant embeddings: {len(relevant_loaded_ids)} of {len(required_protein_ids_for_interactions)} needed IDs found.",
                                       **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']},
                                       'fold_f1_scores': [], 'fold_auc_scores': []})
                continue
            elif not relevant_loaded_ids and len(protein_embeddings) > 0:
                print(
                    f"Warning: No embeddings loaded for {config_item['name']} that are present in the interaction dataset. Skipping CV.")
                all_cv_results.append({'embedding_name': config_item['name'],
                                       'notes': "Embeddings loaded, but none match interaction data.",
                                       **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']},
                                       'fold_f1_scores': [], 'fold_auc_scores': []})
                continue

            cv_run_result = main_workflow_cv(config_item['name'], protein_embeddings, positive_pairs_all,
                                             negative_pairs_all,
                                             required_protein_ids_for_interactions)  # Pass interaction pairs
            if cv_run_result:
                all_cv_results.append(cv_run_result)
                history_fold1 = cv_run_result.get('history_dict_fold1', {})
                if PLOT_TRAINING_HISTORY and history_fold1 and any(
                        isinstance(val_list, list) and len(val_list) > 0 for val_list in history_fold1.values()):
                    plot_training_history(history_fold1, cv_run_result['embedding_name'], fold_num=1)
                elif DEBUG_VERBOSE and PLOT_TRAINING_HISTORY:
                    print(
                        f"No/empty training history from fold 1 for {cv_run_result.get('embedding_name', 'Unknown Run')}.")
        else:
            print(
                f"Skipping CV for {config_item['name']}: Failed to load or embeddings are empty after selective loading.")
            all_cv_results.append(
                {'embedding_name': config_item['name'], 'notes': "Failed to load or no relevant embeddings found.",
                 **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']}, 'fold_f1_scores': [],
                 'fold_auc_scores': []})
        del protein_embeddings;
        gc.collect()

    if all_cv_results:
        if DEBUG_VERBOSE:
            print(f"\nDEBUG (__main__): Final all_cv_results before aggregate plots/table (summary):")
            for i, res_dict in enumerate(all_cv_results):
                print(
                    f"  Summary for CV run {i + 1} ({res_dict.get('embedding_name')}): F1_avg={res_dict.get('test_f1_sklearn'):.4f}, AUC_avg={res_dict.get('test_auc_sklearn'):.4f}, Notes: {res_dict.get('notes')}")
        print("\n\nGenerating aggregate comparison plots & table (based on CV averages)...")
        valid_roc_res = [r for r in all_cv_results if
                         r.get('roc_data_representative') and r['roc_data_representative'][0] is not None and len(
                             r['roc_data_representative'][0]) > 0]
        if valid_roc_res:
            plot_roc_curves(all_cv_results)
        else:
            print("No valid representative ROC data to plot across models.")
        plot_comparison_charts(all_cv_results)
        print_results_table(all_cv_results, is_cv=True)
        perform_statistical_tests(all_cv_results, MAIN_EMBEDDING_NAME, metric_key_cfg=STATISTICAL_TEST_METRIC_KEY,
                                  alpha_cfg=STATISTICAL_TEST_ALPHA)
    else:
        print("\nNo results generated from any configurations to plot or tabulate.")
    print("\nScript finished.")