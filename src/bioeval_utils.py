# bioeval_utils.py
import os
import h5py
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt
import tensorflow as tf
import math  # Ensures math is imported for math.ceil
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any, Set, Tuple, Union

# --- Configuration Snippets (Defaults that can be overridden by the main script) ---
DEBUG_VERBOSE = False
K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
K_VALUES_FOR_TABLE_DISPLAY = [50, 100]
RANDOM_STATE = 42
STATISTICAL_TEST_ALPHA = 0.05


class ProteinFileOps:
    @staticmethod
    def load_interaction_pairs(filepath: str,
                               label: int,
                               sample_n: Optional[int] = None,
                               random_state_for_sampling: Optional[int] = None
                               ) -> List[Tuple[str, str, int]]:  # Added sample_n and random_state_for_sampling
        filepath = os.path.normpath(filepath)
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Loading interaction pairs from: {filepath} (label: {label}){sampling_info}...")
        if not os.path.exists(filepath):
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            # This loads the entire file into a DataFrame first.
            # If the file is too large to fit in memory even for this step,
            # a more complex line-by-line sampling or chunked reading approach would be needed.
            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str)  #

            if sample_n is not None and sample_n > 0 and sample_n < len(df):
                print(
                    f"  Original pair count in {os.path.basename(filepath)}: {len(df)}. Sampling down to {sample_n} pairs.")
                df = df.sample(n=sample_n, random_state=random_state_for_sampling)
            elif sample_n is not None and sample_n <= 0:
                print(
                    f"  Warning: sample_n is {sample_n}. No pairs will be loaded from {os.path.basename(filepath)} due to sampling.")
                df = df.iloc[0:0]  # Empty dataframe

            # Ensure IDs are stripped of potential leading/trailing whitespace
            pairs = [(str(row.protein1).strip(), str(row.protein2).strip(), label) for _, row in df.iterrows()]  #
            print(f"Successfully loaded {len(pairs)} pairs from {os.path.basename(filepath)}.")
            return pairs
        except Exception as e:
            print(f"Error loading interaction file {filepath}: {e}")
            return []


# ... (rest of bioeval_utils.py)


class FileOps:
    @staticmethod
    def load_h5_embeddings_selectively(h5_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        h5_path = os.path.normpath(h5_path)
        load_mode = "selectively" if required_ids else "all keys"
        required_count_info = f"for up to {len(required_ids)} IDs" if required_ids else ""
        # print(f"Loading embeddings from: {h5_path} ({load_mode} {required_count_info})...") # Verbose, can be enabled

        if not os.path.exists(h5_path):
            print(f"Warning: Embedding file not found: {h5_path}")
            return {}

        protein_embeddings: Dict[str, np.ndarray] = {}
        loaded_count = 0
        try:
            with h5py.File(h5_path, 'r') as hf:
                keys_in_file = list(hf.keys())

                keys_to_load_final = []
                if required_ids:
                    for key in keys_in_file:
                        if key in required_ids:
                            keys_to_load_final.append(key)
                    # if DEBUG_VERBOSE: print(f"  Will attempt to load {len(keys_to_load_final)} matching required IDs from {os.path.basename(h5_path)}.")
                else:
                    keys_to_load_final = keys_in_file

                if not keys_to_load_final and required_ids and keys_in_file:
                    if DEBUG_VERBOSE: print(f"  No keys in {os.path.basename(h5_path)} match the required_ids set.")

                for key in tqdm(keys_to_load_final, desc=f"  Reading {os.path.basename(h5_path)}", leave=False,
                                unit="protein", disable=not DEBUG_VERBOSE):
                    if isinstance(hf[key], h5py.Dataset):
                        try:
                            protein_embeddings[key] = hf[key][:].astype(np.float32)
                            loaded_count += 1
                        except Exception as e_load:
                            if DEBUG_VERBOSE: print(f"    Could not load dataset for key '{key}': {e_load}")
            print(f"Loaded {loaded_count} embeddings from {os.path.basename(h5_path)}.")
        except Exception as e:
            print(f"Error opening or processing HDF5 file {h5_path}: {e}")
            return {}
        return protein_embeddings

    @staticmethod
    def load_custom_embeddings(p_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        print(f"Warning: 'load_custom_embeddings' is a placeholder. Path: {p_path}")
        return {}


# --- Graph Processing ---
class Graph:
    def __init__(self):
        self.embedding_dim: Optional[int] = None

    def get_embedding_dimension(self, protein_embeddings: Dict[str, np.ndarray]) -> int:
        if not protein_embeddings:
            self.embedding_dim = 0
            return 0
        for emb_vec in protein_embeddings.values():
            if emb_vec is not None and hasattr(emb_vec, 'shape') and len(emb_vec.shape) > 0:
                self.embedding_dim = emb_vec.shape[-1]
                if DEBUG_VERBOSE: print(f"Inferred embedding dimension: {self.embedding_dim}")
                return self.embedding_dim
        self.embedding_dim = 0
        print("Warning: Could not infer embedding dimension from provided embeddings.")
        return 0

    def create_edge_embeddings(self, interaction_pairs: List[Tuple[str, str, int]],
                               protein_embeddings: Dict[str, np.ndarray],
                               method: str = 'concatenate') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"Creating edge embeddings using method: {method}...")
        if not protein_embeddings:
            print("Protein embeddings dictionary is empty. Cannot create edge features.")
            return None, None

        if self.embedding_dim is None or self.embedding_dim == 0:
            self.get_embedding_dimension(protein_embeddings)
        if self.embedding_dim == 0:
            print("Embedding dimension is 0. Cannot create valid edge features.")
            return None, None

        edge_features = []
        labels = []
        skipped_pairs_count = 0

        for p1_id, p2_id, label in tqdm(interaction_pairs, desc="Creating Edge Features", leave=False,
                                        disable=not DEBUG_VERBOSE):
            emb1, emb2 = protein_embeddings.get(p1_id), protein_embeddings.get(p2_id)
            if emb1 is not None and emb2 is not None:
                if emb1.ndim > 1: emb1 = emb1.flatten()
                if emb2.ndim > 1: emb2 = emb2.flatten()
                if emb1.shape[0] != self.embedding_dim or emb2.shape[0] != self.embedding_dim:
                    skipped_pairs_count += 1
                    continue
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
                edge_features.append(feature)
                labels.append(label)
            else:
                skipped_pairs_count += 1

        if skipped_pairs_count > 0:
            print(f"Skipped {skipped_pairs_count} (out of {len(interaction_pairs)}) pairs due to missing embeddings or dimension mismatch.")
        if not edge_features:
            print("No edge features created. Check protein ID matching and embedding integrity.")
            return None, None

        print(f"Created {len(edge_features)} edge features with dimension {edge_features[0].shape[0]}.")
        return np.array(edge_features, dtype=np.float32), np.array(labels, dtype=np.int32)


# --- MLP Model ---
def build_mlp_model(input_shape: int, learning_rate: float, mlp_params: Dict[str, Any]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(mlp_params['dense1_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(mlp_params['l2_reg'])),
        tf.keras.layers.Dropout(mlp_params['dropout1_rate']),
        tf.keras.layers.Dense(mlp_params['dense2_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(mlp_params['l2_reg'])),
        tf.keras.layers.Dropout(mlp_params['dropout2_rate']),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc_keras'),
                           tf.keras.metrics.Precision(name='precision_keras'),
                           tf.keras.metrics.Recall(name='recall_keras')])
    return model


# --- Plotting Functions ---
def plot_training_history(history_dict: Dict[str, Any], model_name: str, plots_output_dir: str,
                          fold_num: Optional[int] = None):
    title_suffix = f" (Fold {fold_num})" if fold_num is not None else " (Representative Fold)"
    if not history_dict or not any(
            isinstance(val_list, list) and len(val_list) > 0 for val_list in history_dict.values()):
        if DEBUG_VERBOSE: print(f"Plotting: No history data for {model_name}{title_suffix}. Plot empty.")
        return

    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir,
                                 f"history_{model_name.replace(' / ', '_').replace(':', '-')}{'_F' + str(fold_num) if fold_num else ''}.png")

    plt.figure(figsize=(12, 5))
    plotted_loss_axes = False
    if 'loss' in history_dict and history_dict['loss']:
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['loss'], label='Training Loss', marker='.' if len(history_dict['loss']) < 15 else None)
        plotted_loss_axes = True
    if 'val_loss' in history_dict and history_dict['val_loss']:
        if not plotted_loss_axes:
            plt.subplot(1, 2, 1)
        plt.plot(history_dict['val_loss'], label='Validation Loss', marker='.' if len(history_dict['val_loss']) < 15 else None)
        plotted_loss_axes = True
    if plotted_loss_axes:
        plt.title(f'Model Loss: {model_name}{title_suffix}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

    plotted_acc_axes = False
    if 'accuracy' in history_dict and history_dict['accuracy']:
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['accuracy'], label='Training Accuracy', marker='.' if len(history_dict['accuracy']) < 15 else None)
        plotted_acc_axes = True
    if 'val_accuracy' in history_dict and history_dict['val_accuracy']:
        if not plotted_acc_axes:
            plt.subplot(1, 2, 2)
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='.' if len(history_dict['val_accuracy']) < 15 else None)
        plotted_acc_axes = True
    if plotted_acc_axes:
        plt.title(f'Model Accuracy: {model_name}{title_suffix}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

    if not (plotted_loss_axes or plotted_acc_axes):
        plt.close()
        return

    plt.suptitle(f"Training History: {model_name}{title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(plot_filename)
        print(f"  Saved training history plot to {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close()


def plot_roc_curves(results_list: List[Dict[str, Any]], plots_output_dir: str):
    plt.figure(figsize=(10, 8))
    plotted_anything = False
    for result in results_list:
        roc_data_to_plot = result.get('roc_data_representative')
        if roc_data_to_plot:
            fpr, tpr, auc_val_representative = roc_data_to_plot
            avg_auc = result.get('test_auc_sklearn',
                                 auc_val_representative if auc_val_representative is not None else 0.0)
            if fpr is not None and tpr is not None and avg_auc is not None and hasattr(fpr, '__len__') and len(
                fpr) > 0 and hasattr(tpr, '__len__') and len(tpr) > 0:
                plt.plot(fpr, tpr, lw=2, label=f"{result.get('embedding_name', 'Unknown')} (Avg AUC = {avg_auc:.3f})")
                plotted_anything = True
    if not plotted_anything:
        if DEBUG_VERBOSE: print("Plotting: No valid ROC data for any embedding.")
        plt.close()
        return

    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir, "comparison_roc_curves.png")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    try:
        plt.savefig(plot_filename)
        print(f"  Saved ROC comparison plot to {plot_filename}")
    except Exception as e:
        print(f"  Error saving ROC plot {plot_filename}: {e}")
    plt.close()


def plot_comparison_charts(results_list: List[Dict[str, Any]], plots_output_dir: str, k_values_for_table: List[int]):
    if not results_list:
        if DEBUG_VERBOSE:
            print("Plotting: No results for comparison charts.")
        return

    metrics_to_plot = {'Accuracy': 'test_accuracy_keras', 'Precision': 'test_precision_sklearn',
                       'Recall': 'test_recall_sklearn', 'F1-Score': 'test_f1_sklearn', 'AUC': 'test_auc_sklearn'}
    if k_values_for_table:  # Check if list is not empty
        if len(k_values_for_table) > 0:
            metrics_to_plot[f'Hits@{k_values_for_table[0]}'] = f'test_hits_at_{k_values_for_table[0]}'
            metrics_to_plot[f'NDCG@{k_values_for_table[0]}'] = f'test_ndcg_at_{k_values_for_table[0]}'
        if len(k_values_for_table) > 1:
            metrics_to_plot[f'Hits@{k_values_for_table[1]}'] = f'test_hits_at_{k_values_for_table[1]}'
            metrics_to_plot[f'NDCG@{k_values_for_table[1]}'] = f'test_ndcg_at_{k_values_for_table[1]}'

    embedding_names = [res.get('embedding_name', f'Run {i + 1}') for i, res in enumerate(results_list)]
    num_perf_metrics = len(metrics_to_plot)
    cols = min(3, num_perf_metrics + 1)
    rows = math.ceil((num_perf_metrics + 1) / cols)
    plt.figure(figsize=(max(15, cols * 5), rows * 4))
    plot_index = 1
    for metric_display_name, metric_key in metrics_to_plot.items():
        plt.subplot(rows, cols, plot_index)
        values = [res.get(metric_key) if res.get(metric_key) is not None else 0.0 for res in results_list]
        values = [v if isinstance(v, (int, float)) else 0.0 for v in values]
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(embedding_names))))
        plt.title(metric_display_name)
        plt.ylabel('Score')
        plt.xticks(rotation=30, ha="right")
        current_max_val = max(values) if values else 0.0
        plot_upper_limit = 1.05
        if "Hits@" in metric_display_name:
            plot_upper_limit = max(current_max_val * 1.15 if current_max_val > 0 else 10, 10)
        elif values and current_max_val > 0:
            plot_upper_limit = max(1.05 if current_max_val <= 1 else current_max_val * 1.15, 0.1)
        else:
            plot_upper_limit = 1.05 if "Hits@" not in metric_display_name else max(10, current_max_val * 1.15)
        plt.ylim(0, plot_upper_limit)
        for bar_idx, bar in enumerate(bars):
            yval = values[bar_idx]
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plot_upper_limit,
                     f'{yval:.3f}', ha='center', va='bottom')
        plot_index += 1

    if plot_index <= rows * cols:
        plt.subplot(rows, cols, plot_index)
        training_times = [res.get('training_time') if res.get('training_time') is not None else 0.0 for res in results_list]
        training_times = [t if isinstance(t, (int, float)) else 0.0 for t in training_times]
        bars = plt.bar(embedding_names, training_times, color=plt.cm.plasma(np.linspace(0, 1, len(embedding_names))))
        plt.title('Avg Training Time per Fold')
        plt.ylabel('Seconds')
        plt.xticks(rotation=30, ha="right")
        max_time_val = max(training_times) if training_times else 1.0
        plot_upper_limit_time = max_time_val * 1.15 if max_time_val > 0 else 10
        plt.ylim(0, plot_upper_limit_time)
        for bar_idx, bar in enumerate(bars):
            yval = training_times[bar_idx]
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plot_upper_limit_time,
                     f'{yval:.2f}s', ha='center', va='bottom')

    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir, "comparison_metrics_summary.png")
    plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(plot_filename, dpi=150)
        print(f"  Saved comparison chart to {plot_filename}")
    except Exception as e:
        print(f"  Error saving comparison chart {plot_filename}: {e}")
    plt.close()


# --- Results Table & Stats ---
def print_results_table(results_list: List[Dict[str, Any]], k_values_for_table: List[int], is_cv: bool = False,
                        output_dir: str = ".", filename: str = "results_summary_table.txt"):
    if not results_list:
        if DEBUG_VERBOSE:
            print("\nNo results to display in table.")
        return

    suffix = " (Avg over Folds)" if is_cv else ""
    header_text = f"--- Overall Performance Comparison Table{suffix} ---"
    print(f"\n\n{header_text}")
    metric_keys_headers = [('embedding_name', "Embedding Name")]
    metric_keys_headers.append(('training_time', f"Train Time(s){suffix}"))
    metric_keys_headers.append(('test_loss', f"Val Loss{suffix}"))
    metric_keys_headers.append(('test_accuracy_keras', f"Accuracy{suffix}"))
    metric_keys_headers.append(('test_precision_sklearn', f"Precision{suffix}"))
    metric_keys_headers.append(('test_recall_sklearn', f"Recall{suffix}"))
    metric_keys_headers.append(('test_f1_sklearn', f"F1-Score{suffix}"))
    metric_keys_headers.append(('test_auc_sklearn', f"AUC{suffix}"))
    for k in k_values_for_table: metric_keys_headers.append((f'test_hits_at_{k}', f"Hits@{k}{suffix}"))
    for k in k_values_for_table: metric_keys_headers.append((f'test_ndcg_at_{k}', f"NDCG@{k}{suffix}"))
    if is_cv:
        metric_keys_headers.append(('test_f1_sklearn_std', "F1 StdDev"))
        metric_keys_headers.append(('test_auc_sklearn_std', "AUC StdDev"))

    headers = [h for _, h in metric_keys_headers]
    metric_keys_to_extract = [k for k, _ in metric_keys_headers]
    table_data = [headers]
    for res_dict in results_list:
        row = []
        for key in metric_keys_to_extract:
            val = res_dict.get(key)
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

    if len(table_data) <= 1:
        if DEBUG_VERBOSE:
            print("No data rows for table.")
        return

    column_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
    fmt_str = " | ".join([f"{{:<{w}}}" for w in column_widths])
    table_string = fmt_str.format(*headers) + "\n"
    table_string += "-+-".join(["-" * w for w in column_widths]) + "\n"
    for i in range(1, len(table_data)):
        table_string += fmt_str.format(*table_data[i]) + "\n"
    print(table_string)

    if filename:
        full_path = os.path.normpath(os.path.join(output_dir, filename))
        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(full_path, 'w') as f:
                f.write(header_text + "\n\n" + table_string)
            print(f"Results table saved to: {full_path}")
        except Exception as e:
            print(f"Error saving results table to {full_path}: {e}")


def perform_statistical_tests(results_list: List[Dict[str, Any]], main_embedding_name_cfg: Optional[str],
                              metric_key_cfg: str, alpha_cfg: float):
    if not main_embedding_name_cfg:
        print("\nStat Tests: Main embedding name not specified. Skipping.")
        return
    if len(results_list) < 2:
        print("\nStat Tests: Need at least two methods to compare.")
        return

    print(f"\n\n--- Statistical Comparison vs '{main_embedding_name_cfg}' on '{metric_key_cfg}' (Alpha={alpha_cfg}) ---")
    key_for_fold_scores = None
    if metric_key_cfg == 'test_auc_sklearn':
        key_for_fold_scores = 'fold_auc_scores'
    elif metric_key_cfg == 'test_f1_sklearn':
        key_for_fold_scores = 'fold_f1_scores'
    else:
        print(f"Stat Tests: Metric key '{metric_key_cfg}' not supported for fold scores.")
        return

    main_model_res = next((res for res in results_list if res['embedding_name'] == main_embedding_name_cfg), None)
    if not main_model_res:
        print(f"Stat Tests: Main model '{main_embedding_name_cfg}' not found.")
        return

    main_model_scores_all_folds = main_model_res.get(key_for_fold_scores, [])
    main_model_scores_valid = [s for s in main_model_scores_all_folds if not np.isnan(s)]
    if len(main_model_scores_valid) < 2:
        print(f"Stat Tests: Not enough valid scores for main model '{main_embedding_name_cfg}'.")
        return

    other_model_results_list = [res for res in results_list if res['embedding_name'] != main_embedding_name_cfg]
    if not other_model_results_list:
        print("No other models to compare.")
        return

    header_parts = [f"{'Compared Embedding':<30}", f"{'Wilcoxon p-val':<15}", f"{'Signif. (p<{alpha_cfg})':<18}",
                    f"{'Pearson r':<10}", f"{'r-squared':<10}"]
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for other_model_res in other_model_results_list:
        other_model_name = other_model_res['embedding_name']
        other_model_scores_all_folds = other_model_res.get(key_for_fold_scores, [])
        other_model_scores_valid = [s for s in other_model_scores_all_folds if not np.isnan(s)]

        if len(main_model_scores_valid) != len(other_model_scores_valid) or len(main_model_scores_valid) < 2:
            print(f"{other_model_name:<30} | {'N/A (scores mismatch/few)':<15} | {'N/A':<18} | {'N/A':<10} | {'N/A':<10}")
            continue

        current_main_scores = np.array(main_model_scores_valid)
        current_other_scores = np.array(other_model_scores_valid)
        p_value_wilcoxon, pearson_r_val, r_squared_val = 1.0, np.nan, np.nan
        significance_diff = "No"
        correlation_note = ""

        try:
            if not np.allclose(current_main_scores, current_other_scores):
                stat, p_value_wilcoxon = wilcoxon(current_main_scores, current_other_scores, alternative='two-sided', zero_method='pratt')
                if p_value_wilcoxon < alpha_cfg:
                    mean_main = np.mean(current_main_scores)
                    mean_other = np.mean(current_other_scores)
                    significance_diff = f"Yes (Main Better)" if mean_main > mean_other else (f"Yes (Main Worse)" if mean_main < mean_other else "Yes (Diff, Means Eq.)")
            else:
                p_value_wilcoxon = 1.0
                significance_diff = "No (Identical Scores)"
        except ValueError as e:
            p_value_wilcoxon = 1.0
            significance_diff = "N/A (Wilcoxon Err)"
            if DEBUG_VERBOSE:
                print(f"Wilcoxon error for {other_model_name}: {e}")

        try:
            if len(np.unique(current_main_scores)) > 1 and len(np.unique(current_other_scores)) > 1:
                pearson_r_val, p_val_corr = pearsonr(current_main_scores, current_other_scores)
                r_squared_val = pearson_r_val ** 2
                if p_val_corr < alpha_cfg:
                    correlation_note = f"(Corr. p={p_val_corr:.2e})"
        except Exception as e_corr:
            if DEBUG_VERBOSE:
                print(f"Pearson r error for {other_model_name}: {e_corr}")

        print(f"{other_model_name:<30} | {p_value_wilcoxon:<15.4f} | {significance_diff:<18} | {pearson_r_val:<10.4f} | {r_squared_val:<10.4f} {correlation_note}")
    print("-" * len(header))
    print("Note: Wilcoxon tests difference. Pearson r for linear correlation.")

# --- CV Workflow (moved to bioeval_cv_worker.py) ---
# def main_workflow_cv(...)