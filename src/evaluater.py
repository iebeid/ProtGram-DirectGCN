# Combined Protein-Protein Interaction Evaluation Script
import os
import sys
import shutil
import numpy as np
import pandas as pd
import time
import gc
import h5py
import math
import tensorflow as tf
import sklearn  # <--- ADD THIS LINE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
# ... other imports ...
from scipy.stats import wilcoxon, pearsonr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any, Set, Tuple, Union, Callable

# --- Global Configuration & Script Behavior ---

# General Settings
DEBUG_VERBOSE = True  # Set to False for less console output
RANDOM_STATE = 42
RUN_DUMMY_TEST_FIRST = True  # Set to True to run a quick test with dummy data
CLEANUP_DUMMY_DATA = True  # Set to True to remove dummy data after the dummy run

# CV and Model Parameters (can be overridden for dummy run)
EDGE_EMBEDDING_METHOD = 'concatenate'
N_FOLDS = 5
MAX_TRAIN_SAMPLES_CV = 100000  # Increased from original for more realistic run
MAX_VAL_SAMPLES_CV = 20000  # Increased
MAX_SHUFFLE_BUFFER_SIZE = 200000  # Buffer size for tf.data.Dataset.shuffle
PLOT_TRAINING_HISTORY = True  # Whether to plot training history for the first fold of each CV

# MLP Architecture
MLP_DENSE1_UNITS = 128
MLP_DROPOUT1_RATE = 0.4
MLP_DENSE2_UNITS = 64
MLP_DROPOUT2_RATE = 0.4
MLP_L2_REG = 0.001

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10  # Number of epochs for training
LEARNING_RATE = 1e-3

# Metrics and Reporting
K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]  # @K values for Hits@K, NDCG@K during CV
K_VALUES_FOR_TABLE_DISPLAY = [50, 100]  # @K values for the summary table
MAIN_EMBEDDING_NAME_FOR_STATS = "ProtT5_Example_Data"  # Baseline embedding for statistical tests (update if ProtT5 is not run or has a different name)
STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'  # Metric to use for statistical comparison
STATISTICAL_TEST_ALPHA = 0.05  # Significance level for statistical tests

# Output Directories (can be overridden for dummy run)
BASE_OUTPUT_DIR = "C:/tmp/Models/ppi_evaluation_results_combined/"  # Main output directory for results

# --- TensorFlow GPU Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow informational messages
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow errors unless critical

# Check for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"TensorFlow: GPU Devices Detected: {gpu_devices}")
    # You can add tf.config.experimental.set_memory_growth(gpu, True) for each GPU if needed
else:
    print("TensorFlow: Warning: No GPU detected. Running on CPU.")


# --- Utility Classes and Functions (formerly bioeval_utils.py) ---

class ProteinFileOps:
    @staticmethod
    def load_interaction_pairs(filepath: str,
                               label: int,
                               sample_n: Optional[int] = None,
                               random_state_for_sampling: Optional[int] = None
                               ) -> List[Tuple[str, str, int]]:
        filepath = os.path.normpath(filepath)
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        if DEBUG_VERBOSE:
            print(f"Loading interaction pairs from: {filepath} (label: {label}){sampling_info}...")

        if not os.path.exists(filepath):
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str)
            if sample_n is not None and sample_n > 0 and sample_n < len(df):
                if DEBUG_VERBOSE:
                    print(
                        f"  Original pair count in {os.path.basename(filepath)}: {len(df)}. Sampling down to {sample_n} pairs.")
                df = df.sample(n=sample_n, random_state=random_state_for_sampling)
            elif sample_n is not None and sample_n <= 0:
                if DEBUG_VERBOSE:
                    print(
                        f"  Warning: sample_n is {sample_n}. No pairs will be loaded from {os.path.basename(filepath)} due to sampling.")
                df = df.iloc[0:0]

            pairs = [(str(row.protein1).strip(), str(row.protein2).strip(), label) for _, row in df.iterrows()]
            if DEBUG_VERBOSE:
                print(f"Successfully loaded {len(pairs)} pairs from {os.path.basename(filepath)}.")
            return pairs
        except Exception as e:
            print(f"Error loading interaction file {filepath}: {e}")
            return []


class FileOps:
    @staticmethod
    def load_h5_embeddings_selectively(h5_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        h5_path = os.path.normpath(h5_path)
        # load_mode = "selectively" if required_ids else "all keys"
        # required_count_info = f"for up to {len(required_ids)} IDs" if required_ids else ""
        # print(f"Loading embeddings from: {h5_path} ({load_mode} {required_count_info})...") # Verbose

        if not os.path.exists(h5_path):
            print(f"Warning: Embedding file not found: {h5_path}")
            return {}

        protein_embeddings: Dict[str, np.ndarray] = {}
        loaded_count = 0
        try:
            with h5py.File(h5_path, 'r') as hf:
                keys_in_file = list(hf.keys())
                keys_to_load_final = [key for key in keys_in_file if
                                      required_ids and key in required_ids] if required_ids else keys_in_file

                if not keys_to_load_final and required_ids and keys_in_file and DEBUG_VERBOSE:
                    print(f"  No keys in {os.path.basename(h5_path)} match the required_ids set.")

                for key in tqdm(keys_to_load_final, desc=f"  Reading {os.path.basename(h5_path)}", leave=False,
                                unit="protein", disable=not DEBUG_VERBOSE):
                    if isinstance(hf[key], h5py.Dataset):
                        try:
                            protein_embeddings[key] = hf[key][:].astype(np.float32)
                            loaded_count += 1
                        except Exception as e_load:
                            if DEBUG_VERBOSE: print(f"    Could not load dataset for key '{key}': {e_load}")
            if DEBUG_VERBOSE:
                print(f"Loaded {loaded_count} embeddings from {os.path.basename(h5_path)}.")
        except Exception as e:
            print(f"Error opening or processing HDF5 file {h5_path}: {e}")
            return {}
        return protein_embeddings

    @staticmethod
    def load_custom_embeddings(p_path: str, required_ids: Optional[Set[str]] = None) -> Dict[str, np.ndarray]:
        # This is a placeholder as in the original script
        print(f"Warning: 'load_custom_embeddings' is a placeholder. Path: {p_path}. No embeddings loaded.")
        return {}


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
        if DEBUG_VERBOSE:
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
                    if DEBUG_VERBOSE:
                        print(
                            f"Warning: Dimension mismatch for pair ({p1_id}, {p2_id}). Emb1: {emb1.shape}, Emb2: {emb2.shape}, Expected: {self.embedding_dim}. Skipping.")
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

        if skipped_pairs_count > 0 and DEBUG_VERBOSE:
            print(
                f"Skipped {skipped_pairs_count} (out of {len(interaction_pairs)}) pairs due to missing embeddings or dimension mismatch.")
        if not edge_features:
            print("No edge features created. Check protein ID matching and embedding integrity.")
            return None, None

        if DEBUG_VERBOSE:
            print(f"Created {len(edge_features)} edge features with dimension {edge_features[0].shape[0]}.")
        return np.array(edge_features, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_mlp_model(input_shape: int, learning_rate: float, mlp_params_dict: Dict[str, Any]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(mlp_params_dict['dense1_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(mlp_params_dict['l2_reg'])),
        tf.keras.layers.Dropout(mlp_params_dict['dropout1_rate']),
        tf.keras.layers.Dense(mlp_params_dict['dense2_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(mlp_params_dict['l2_reg'])),
        tf.keras.layers.Dropout(mlp_params_dict['dropout2_rate']),
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
    # (Content from bioeval_utils.plot_training_history, slightly adapted for verbosity)
    title_suffix = f" (Fold {fold_num})" if fold_num is not None else " (Representative Fold)"
    if not history_dict or not any(
            isinstance(val_list, list) and len(val_list) > 0 for val_list in history_dict.values()):
        if DEBUG_VERBOSE: print(f"Plotting: No history data for {model_name}{title_suffix}. Plot empty.")
        return

    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir,
                                 f"history_{model_name.replace(' / ', '_').replace(':', '-')}{'_F' + str(fold_num) if fold_num else ''}.png")

    plt.figure(figsize=(12, 5))
    # ... (rest of the plotting logic from bioeval_utils.plot_training_history) ...
    # For brevity, I'll assume the internal plotting logic is copied here.
    # Ensure it uses DEBUG_VERBOSE for its print statements.
    plotted_loss_axes = False
    if 'loss' in history_dict and history_dict['loss']:
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['loss'], label='Training Loss', marker='.' if len(history_dict['loss']) < 15 else None)
        plotted_loss_axes = True
    if 'val_loss' in history_dict and history_dict['val_loss']:
        if not plotted_loss_axes: plt.subplot(1, 2, 1)
        plt.plot(history_dict['val_loss'], label='Validation Loss',
                 marker='.' if len(history_dict['val_loss']) < 15 else None)
        plotted_loss_axes = True
    if plotted_loss_axes:
        plt.title(f'Model Loss: {model_name}{title_suffix}');
        plt.ylabel('Loss');
        plt.xlabel('Epoch');
        plt.legend();
        plt.grid(True)

    plotted_acc_axes = False
    if 'accuracy' in history_dict and history_dict['accuracy']:
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['accuracy'], label='Training Accuracy',
                 marker='.' if len(history_dict['accuracy']) < 15 else None)
        plotted_acc_axes = True
    if 'val_accuracy' in history_dict and history_dict['val_accuracy']:
        if not plotted_acc_axes: plt.subplot(1, 2, 2)
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy',
                 marker='.' if len(history_dict['val_accuracy']) < 15 else None)
        plotted_acc_axes = True
    if plotted_acc_axes:
        plt.title(f'Model Accuracy: {model_name}{title_suffix}');
        plt.ylabel('Accuracy');
        plt.xlabel('Epoch');
        plt.legend();
        plt.grid(True)

    if not (plotted_loss_axes or plotted_acc_axes): plt.close(); return
    plt.suptitle(f"Training History: {model_name}{title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(plot_filename);
        print(f"  Saved training history plot to {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close()


def plot_roc_curves(results_list: List[Dict[str, Any]], plots_output_dir: str):
    # (Content from bioeval_utils.plot_roc_curves)
    plt.figure(figsize=(10, 8));
    plotted_anything = False
    for result in results_list:
        roc_data_to_plot = result.get('roc_data_representative')
        if roc_data_to_plot:
            fpr, tpr, auc_val_representative = roc_data_to_plot
            avg_auc = result.get('test_auc_sklearn',
                                 auc_val_representative if auc_val_representative is not None else 0.0)
            if fpr is not None and tpr is not None and avg_auc is not None and hasattr(fpr, '__len__') and len(
                    fpr) > 0 and hasattr(tpr, '__len__') and len(tpr) > 0:
                plt.plot(fpr, tpr, lw=2, label=f"{result.get('embedding_name', 'Unknown')} (Avg AUC = {avg_auc:.3f})");
                plotted_anything = True
    if not plotted_anything:
        if DEBUG_VERBOSE: print("Plotting: No valid ROC data for any embedding."); plt.close(); return
    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir, "comparison_roc_curves.png")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)');
    plt.ylabel('True Positive Rate (TPR)');
    plt.title('ROC Curves Comparison');
    plt.legend(loc="lower right");
    plt.grid(True)
    try:
        plt.savefig(plot_filename);
        print(f"  Saved ROC comparison plot to {plot_filename}")
    except Exception as e:
        print(f"  Error saving ROC plot {plot_filename}: {e}")
    plt.close()


def plot_comparison_charts(results_list: List[Dict[str, Any]], plots_output_dir: str, k_vals_table: List[int]):
    # (Content from bioeval_utils.plot_comparison_charts, adapted for k_vals_table and DEBUG_VERBOSE)
    if not results_list:
        if DEBUG_VERBOSE: print("Plotting: No results for comparison charts."); return

    metrics_to_plot = {'Accuracy': 'test_accuracy_keras', 'Precision': 'test_precision_sklearn',
                       'Recall': 'test_recall_sklearn', 'F1-Score': 'test_f1_sklearn', 'AUC': 'test_auc_sklearn'}
    if k_vals_table:
        if len(k_vals_table) > 0:
            metrics_to_plot[f'Hits@{k_vals_table[0]}'] = f'test_hits_at_{k_vals_table[0]}'
            metrics_to_plot[f'NDCG@{k_vals_table[0]}'] = f'test_ndcg_at_{k_vals_table[0]}'
        if len(k_vals_table) > 1:  # Support for a second k-value if provided
            metrics_to_plot[f'Hits@{k_vals_table[1]}'] = f'test_hits_at_{k_vals_table[1]}'
            metrics_to_plot[f'NDCG@{k_vals_table[1]}'] = f'test_ndcg_at_{k_vals_table[1]}'

    embedding_names = [res.get('embedding_name', f'Run {i + 1}') for i, res in enumerate(results_list)]
    # ... (rest of the plotting logic from bioeval_utils.plot_comparison_charts) ...
    # For brevity, I'll assume the internal plotting logic is copied here.
    num_perf_metrics = len(metrics_to_plot)
    cols = min(3, num_perf_metrics + 1);
    rows = math.ceil((num_perf_metrics + 1) / cols)
    plt.figure(figsize=(max(15, cols * 5), rows * 4));
    plot_index = 1
    for metric_display_name, metric_key in metrics_to_plot.items():
        plt.subplot(rows, cols, plot_index)
        values = [res.get(metric_key, 0.0) for res in results_list]  # Default to 0.0 if missing
        values = [v if isinstance(v, (int, float)) and not np.isnan(v) else 0.0 for v in
                  values]  # Ensure numeric, handle NaN
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(embedding_names))))
        plt.title(metric_display_name);
        plt.ylabel('Score');
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
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plot_upper_limit, f'{yval:.3f}', ha='center',
                     va='bottom')
        plot_index += 1

    if plot_index <= rows * cols:  # Add training time plot
        plt.subplot(rows, cols, plot_index)
        training_times = [res.get('training_time', 0.0) for res in results_list]
        training_times = [t if isinstance(t, (int, float)) and not np.isnan(t) else 0.0 for t in training_times]
        bars = plt.bar(embedding_names, training_times, color=plt.cm.plasma(np.linspace(0, 1, len(embedding_names))))
        plt.title('Avg Training Time per Fold');
        plt.ylabel('Seconds');
        plt.xticks(rotation=30, ha="right")
        max_time_val = max(training_times) if training_times else 1.0
        plot_upper_limit_time = max_time_val * 1.15 if max_time_val > 0 else 10
        plt.ylim(0, plot_upper_limit_time)
        for bar_idx, bar in enumerate(bars):
            yval = training_times[bar_idx]
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plot_upper_limit_time, f'{yval:.2f}s',
                     ha='center', va='bottom')

    os.makedirs(plots_output_dir, exist_ok=True)
    plot_filename = os.path.join(plots_output_dir, "comparison_metrics_summary.png")
    plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(plot_filename, dpi=150);
        print(f"  Saved comparison chart to {plot_filename}")
    except Exception as e:
        print(f"  Error saving comparison chart {plot_filename}: {e}")
    plt.close()


def print_results_table(results_list: List[Dict[str, Any]], k_vals_table: List[int], is_cv: bool = False,
                        output_dir: str = ".", filename: str = "results_summary_table.txt"):
    # (Content from bioeval_utils.print_results_table, adapted for k_vals_table and DEBUG_VERBOSE)
    if not results_list:
        if DEBUG_VERBOSE: print("\nNo results to display in table."); return

    suffix = " (Avg over Folds)" if is_cv else ""
    header_text = f"--- Overall Performance Comparison Table{suffix} ---"
    print(f"\n\n{header_text}")
    # ... (rest of the table printing logic from bioeval_utils.print_results_table) ...
    metric_keys_headers = [('embedding_name', "Embedding Name")]
    metric_keys_headers.append(('training_time', f"Train Time(s){suffix}"))
    metric_keys_headers.append(('test_loss', f"Val Loss{suffix}"))
    metric_keys_headers.append(('test_accuracy_keras', f"Accuracy{suffix}"))
    metric_keys_headers.append(('test_precision_sklearn', f"Precision{suffix}"))
    metric_keys_headers.append(('test_recall_sklearn', f"Recall{suffix}"))
    metric_keys_headers.append(('test_f1_sklearn', f"F1-Score{suffix}"))
    metric_keys_headers.append(('test_auc_sklearn', f"AUC{suffix}"))
    for k in k_vals_table: metric_keys_headers.append((f'test_hits_at_{k}', f"Hits@{k}{suffix}"))
    for k in k_vals_table: metric_keys_headers.append((f'test_ndcg_at_{k}', f"NDCG@{k}{suffix}"))
    if is_cv:
        metric_keys_headers.append(('test_f1_sklearn_std', "F1 StdDev"))
        metric_keys_headers.append(('test_auc_sklearn_std', "AUC StdDev"))

    headers = [h for _, h in metric_keys_headers];
    metric_keys_to_extract = [k for k, _ in metric_keys_headers]
    table_data = [headers]
    for res_dict in results_list:
        row = []
        for key in metric_keys_to_extract:
            val = res_dict.get(key);
            is_placeholder = False
            if isinstance(val, (int, float)):
                if val == -1.0 and 'loss' not in key: is_placeholder = True  # Original placeholder logic
                if val == 0 and ('hits_at_' in key or 'ndcg_at_' in key) and res_dict.get('notes'):
                    if "Single class" in res_dict['notes'] or "Empty" in res_dict['notes']: is_placeholder = True
            if val is None or is_placeholder or (isinstance(val, float) and np.isnan(val)):
                row.append("N/A")  # Handle NaN as N/A
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
        if DEBUG_VERBOSE: print("No data rows for table."); return

    column_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
    fmt_str = " | ".join([f"{{:<{w}}}" for w in column_widths])
    table_string = fmt_str.format(*headers) + "\n" + "-+-".join(["-" * w for w in column_widths]) + "\n"
    for i in range(1, len(table_data)): table_string += fmt_str.format(*table_data[i]) + "\n"
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


def perform_statistical_tests(results_list: List[Dict[str, Any]], main_emb_name: str, metric_key: str,
                              alpha_val: float):
    # (Content from bioeval_utils.perform_statistical_tests)
    if not main_emb_name: print("\nStat Tests: Main embedding name not specified. Skipping."); return
    if len(results_list) < 2: print("\nStat Tests: Need at least two methods to compare."); return

    print(f"\n\n--- Statistical Comparison vs '{main_emb_name}' on '{metric_key}' (Alpha={alpha_val}) ---")
    key_for_fold_scores = 'fold_auc_scores' if metric_key == 'test_auc_sklearn' else (
        'fold_f1_scores' if metric_key == 'test_f1_sklearn' else None)
    if not key_for_fold_scores: print(f"Stat Tests: Metric key '{metric_key}' not supported for fold scores."); return

    main_model_res = next((res for res in results_list if res['embedding_name'] == main_emb_name), None)
    if not main_model_res: print(f"Stat Tests: Main model '{main_emb_name}' not found."); return
    # ... (rest of the statistical test logic from bioeval_utils.perform_statistical_tests) ...
    main_model_scores_all_folds = main_model_res.get(key_for_fold_scores, [])
    main_model_scores_valid = [s for s in main_model_scores_all_folds if not np.isnan(s)]
    if len(main_model_scores_valid) < 2:  # Need at least 2 scores for comparison
        print(
            f"Stat Tests: Not enough valid scores ({len(main_model_scores_valid)}) for main model '{main_emb_name}'. Min 2 required.");
        return

    other_model_results_list = [res for res in results_list if res['embedding_name'] != main_emb_name]
    if not other_model_results_list: print("No other models to compare."); return

    header_parts = [f"{'Compared Embedding':<30}", f"{'Wilcoxon p-val':<15}", f"{'Signif. (p<{alpha_val})':<20}",
                    f"{'Pearson r':<10}", f"{'r-squared':<10}"]
    header = " | ".join(header_parts);
    print(header);
    print("-" * len(header))

    for other_model_res in other_model_results_list:
        other_model_name = other_model_res['embedding_name']
        other_model_scores_all_folds = other_model_res.get(key_for_fold_scores, [])
        other_model_scores_valid = [s for s in other_model_scores_all_folds if not np.isnan(s)]

        if len(main_model_scores_valid) != len(other_model_scores_valid) or len(main_model_scores_valid) < 2:
            print(
                f"{other_model_name:<30} | {'N/A (scores mismatch/few)':<15} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10}")
            continue

        # Ensure scores are actual numbers for tests
        current_main_scores = np.array(main_model_scores_valid, dtype=float)
        current_other_scores = np.array(other_model_scores_valid, dtype=float)

        p_value_wilcoxon, pearson_r_val, r_squared_val = 1.0, np.nan, np.nan
        significance_diff = "No";
        correlation_note = ""

        try:
            if not np.allclose(current_main_scores, current_other_scores):  # Avoid Wilcoxon error for identical samples
                stat, p_value_wilcoxon = wilcoxon(current_main_scores, current_other_scores, alternative='two-sided',
                                                  zero_method='pratt')
                if p_value_wilcoxon < alpha_val:
                    mean_main = np.mean(current_main_scores);
                    mean_other = np.mean(current_other_scores)
                    significance_diff = f"Yes ({'Main Better' if mean_main > mean_other else ('Main Worse' if mean_main < mean_other else 'Diff, Means Eq.')})"
            else:
                p_value_wilcoxon = 1.0;
                significance_diff = "No (Identical Scores)"
        except ValueError as e:  # Catches issues like too few samples or all differences are zero
            p_value_wilcoxon = 1.0;
            significance_diff = "N/A (Wilcoxon Err)"
            if DEBUG_VERBOSE: print(f"Wilcoxon error for {other_model_name} vs {main_emb_name}: {e}")

        try:  # Pearson correlation
            if len(np.unique(current_main_scores)) > 1 and len(np.unique(current_other_scores)) > 1:  # Needs variance
                pearson_r_val, p_val_corr = pearsonr(current_main_scores, current_other_scores)
                r_squared_val = pearson_r_val ** 2
                if p_val_corr < alpha_val: correlation_note = f"(Corr. p={p_val_corr:.2e})"
        except Exception as e_corr:
            if DEBUG_VERBOSE: print(f"Pearson r error for {other_model_name}: {e_corr}")

        print(
            f"{other_model_name:<30} | {p_value_wilcoxon:<15.4f} | {significance_diff:<20} | {pearson_r_val if not np.isnan(pearson_r_val) else 'N/A':<10.4f} | {r_squared_val if not np.isnan(r_squared_val) else 'N/A':<10.4f} {correlation_note}")
    print("-" * len(header));
    print("Note: Wilcoxon tests difference. Pearson r for linear correlation.")


# --- CV Workflow Function (formerly bioeval_cv_worker.main_workflow_cv) ---
def main_workflow_cv(embedding_name: str,
                     protein_embeddings: Dict[str, np.ndarray],
                     positive_pairs: List[Tuple[str, str, int]],
                     negative_pairs: List[Tuple[str, str, int]],
                     mlp_params_dict: Dict[str, Any],
                     edge_emb_method: str,  # Renamed for clarity
                     num_folds: int,  # Renamed
                     rand_state: int,  # Renamed
                     max_train_samp_cv: Optional[int],  # Renamed
                     max_val_samp_cv: Optional[int],  # Renamed
                     max_shuff_buffer: int,  # Renamed
                     cv_batch_size: int,  # Renamed
                     cv_epochs: int,  # Renamed
                     cv_learning_rate: float,  # Renamed
                     k_vals_ranking: List[int]  # Renamed
                     ) -> Dict[str, Any]:
    # Default result structure
    aggregated_results: Dict[str, Any] = {'embedding_name': embedding_name, 'training_time': 0.0,
                                          'history_dict_fold1': {},
                                          'roc_data_representative': (np.array([]), np.array([]), 0.0),
                                          'notes': "", 'fold_f1_scores': [], 'fold_auc_scores': [],
                                          **{k: 0.0 for k in ['test_loss', 'test_accuracy_keras', 'test_auc_keras',
                                                              'test_precision_keras', 'test_recall_keras',
                                                              'test_precision_sklearn', 'test_recall_sklearn',
                                                              'test_f1_sklearn', 'test_auc_sklearn']},
                                          **{f'test_hits_at_{k_val}': 0.0 for k_val in k_vals_ranking},
                                          **{f'test_ndcg_at_{k_val}': 0.0 for k_val in k_vals_ranking}}

    if not protein_embeddings:
        aggregated_results['notes'] = "No embeddings provided to CV workflow."
        if DEBUG_VERBOSE: print(f"No embeddings for {embedding_name}. Skip CV."); return aggregated_results

    all_interaction_pairs = positive_pairs + negative_pairs
    if not all_interaction_pairs:
        aggregated_results['notes'] = "No combined interactions.";
        if DEBUG_VERBOSE: print(f"No combined interactions for {embedding_name}."); return aggregated_results

    graph_processor = Graph()  # Graph class is now defined in this script
    X_full, y_full = graph_processor.create_edge_embeddings(all_interaction_pairs, protein_embeddings,
                                                            method=edge_emb_method)

    if X_full is None or y_full is None or len(X_full) == 0:
        aggregated_results['notes'] = "Dataset creation failed (no edge features)."
        if DEBUG_VERBOSE: print(f"Dataset creation failed for {embedding_name}."); return aggregated_results
    if DEBUG_VERBOSE: print(
        f"Total samples for {embedding_name} for CV: {len(y_full)} (+:{np.sum(y_full == 1)}, -:{np.sum(y_full == 0)})")
    if len(np.unique(y_full)) < 2:
        aggregated_results['notes'] = "Single class in dataset y_full for CV."
        if DEBUG_VERBOSE: print(
            f"Warning: Only one class for {embedding_name}. CV not meaningful."); return aggregated_results

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rand_state)
    fold_metrics_list: List[Dict[str, Any]] = [];
    total_training_time = 0.0

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        if DEBUG_VERBOSE: print(f"\n--- Fold {fold_num + 1}/{num_folds} for {embedding_name} ---")
        X_kfold_train, y_kfold_train = X_full[train_idx], y_full[train_idx]
        X_kfold_val, y_kfold_val = X_full[val_idx], y_full[val_idx]

        X_train_use, y_train_use = X_kfold_train, y_kfold_train
        if max_train_samp_cv is not None and X_kfold_train.shape[0] > max_train_samp_cv:
            if DEBUG_VERBOSE: print(f"Sampling train: {X_kfold_train.shape[0]}->{max_train_samp_cv}")
            idx = np.random.choice(X_kfold_train.shape[0], max_train_samp_cv, replace=False);
            X_train_use, y_train_use = X_kfold_train[idx], y_kfold_train[idx]

        X_val_use, y_val_use = X_kfold_val, y_kfold_val
        if max_val_samp_cv is not None and X_kfold_val.shape[0] > max_val_samp_cv:
            if DEBUG_VERBOSE: print(f"Sampling val: {X_kfold_val.shape[0]}->{max_val_samp_cv}")
            idx = np.random.choice(X_kfold_val.shape[0], max_val_samp_cv, replace=False);
            X_val_use, y_val_use = X_kfold_val[idx], y_kfold_val[idx]

        current_fold_metrics: Dict[str, Any] = {'fold': fold_num + 1}
        if X_train_use.shape[0] == 0:
            if DEBUG_VERBOSE: print(f"Fold {fold_num + 1}: Training data empty. Skipping.");
            # Initialize all potential metrics to NaN
            for k_metric in aggregated_results:
                if 'test_' in k_metric or 'hits_at' in k_metric or 'ndcg_at' in k_metric:
                    current_fold_metrics[k_metric] = np.nan
            fold_metrics_list.append(current_fold_metrics);
            continue

        shuffle_buffer = min(X_train_use.shape[0], max_shuff_buffer)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_use, y_train_use)).shuffle(shuffle_buffer).batch(
            cv_batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_use, y_val_use)).batch(cv_batch_size).prefetch(
            tf.data.AUTOTUNE) if X_val_use.shape[0] > 0 else None

        edge_dim = X_train_use.shape[1]
        model = build_mlp_model(edge_dim, cv_learning_rate, mlp_params_dict=mlp_params_dict)
        if fold_num == 0 and DEBUG_VERBOSE: model.summary(
            print_fn=lambda x: print(x))  # Use lambda for print_fn for direct print
        if DEBUG_VERBOSE: print(f"Training Fold {fold_num + 1} ({X_train_use.shape[0]} samples)...")

        start_time = time.time()
        history = model.fit(train_ds, epochs=cv_epochs, validation_data=val_ds, verbose=1 if DEBUG_VERBOSE else 0)
        fold_training_time = time.time() - start_time
        total_training_time += fold_training_time;
        current_fold_metrics['training_time'] = fold_training_time
        if fold_num == 0: aggregated_results['history_dict_fold1'] = history.history

        y_val_eval_np = np.array(y_val_use).flatten()
        if X_val_use.shape[0] > 0 and val_ds:
            eval_res = model.evaluate(val_ds, verbose=0)
            keras_keys = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                          'test_recall_keras']
            for name, val in zip(keras_keys, eval_res): current_fold_metrics[name] = val

            y_pred_proba_val = model.predict(X_val_use, batch_size=cv_batch_size).flatten()
            y_pred_class_val = (y_pred_proba_val > 0.5).astype(int)

            current_fold_metrics.update({
                'test_precision_sklearn': precision_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                'test_recall_sklearn': recall_score(y_val_eval_np, y_pred_class_val, zero_division=0),
                'test_f1_sklearn': f1_score(y_val_eval_np, y_pred_class_val, zero_division=0)})

            if len(np.unique(y_val_eval_np)) > 1:  # AUC is only valid for multi-class
                current_fold_metrics['test_auc_sklearn'] = roc_auc_score(y_val_eval_np, y_pred_proba_val)
                if fold_num == 0:  # Store ROC data from first fold as representative
                    fpr, tpr, _ = roc_curve(y_val_eval_np, y_pred_proba_val)
                    aggregated_results['roc_data_representative'] = (fpr, tpr, current_fold_metrics['test_auc_sklearn'])
            else:
                current_fold_metrics['test_auc_sklearn'] = 0.0  # Or np.nan, but 0.0 was original
                if fold_num == 0: aggregated_results['roc_data_representative'] = (np.array([]), np.array([]), 0.0)

            # Ranking metrics
            desc_indices = np.argsort(y_pred_proba_val)[::-1];
            sorted_y_val = y_val_eval_np[desc_indices]
            for k_rank in k_vals_ranking:
                eff_k = min(k_rank, len(sorted_y_val))
                current_fold_metrics[f'test_hits_at_{k_rank}'] = np.sum(sorted_y_val[:eff_k] == 1) if eff_k > 0 else 0
                current_fold_metrics[f'test_ndcg_at_{k_rank}'] = ndcg_score(np.asarray([y_val_eval_np]),
                                                                            np.asarray([y_pred_proba_val]), k=eff_k,
                                                                            ignore_ties=True) if eff_k > 0 and len(
                    np.unique(y_val_eval_np)) > 1 else 0.0
        else:  # No validation data or val_ds is None
            if DEBUG_VERBOSE: print(f"Fold {fold_num + 1}: Eval skipped due to no validation data. Metrics set to NaN.")
            metric_keys_to_nan = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras',
                                  'test_recall_keras',
                                  'test_precision_sklearn', 'test_recall_sklearn', 'test_f1_sklearn',
                                  'test_auc_sklearn'] + \
                                 [f'test_hits_at_{k}' for k in k_vals_ranking] + [f'test_ndcg_at_{k}' for k in
                                                                                  k_vals_ranking]
            for key_nan in metric_keys_to_nan: current_fold_metrics[key_nan] = np.nan

        fold_metrics_list.append(current_fold_metrics)
        del model, history, train_ds, val_ds;
        gc.collect();
        tf.keras.backend.clear_session()

    if not fold_metrics_list:
        aggregated_results['notes'] = "No folds completed."
        if DEBUG_VERBOSE: print(f"No folds completed for {embedding_name}."); return aggregated_results

    # Aggregate metrics from all folds
    for key_to_avg in aggregated_results.keys():
        if key_to_avg not in ['embedding_name', 'history_dict_fold1', 'roc_data_representative', 'notes',
                              'fold_f1_scores', 'fold_auc_scores', 'training_time', 'test_f1_sklearn_std',
                              'test_auc_sklearn_std']:
            valid_fold_values = [fm.get(key_to_avg) for fm in fold_metrics_list if
                                 fm.get(key_to_avg) is not None and not np.isnan(fm.get(key_to_avg))]
            aggregated_results[key_to_avg] = np.mean(
                valid_fold_values) if valid_fold_values else 0.0  # Use 0.0 if all NaN or empty

    aggregated_results['training_time'] = total_training_time / len(fold_metrics_list) if fold_metrics_list else 0.0
    aggregated_results['fold_f1_scores'] = [fm.get('test_f1_sklearn', np.nan) for fm in fold_metrics_list]
    aggregated_results['fold_auc_scores'] = [fm.get('test_auc_sklearn', np.nan) for fm in fold_metrics_list]

    f1_valid = [s for s in aggregated_results['fold_f1_scores'] if not np.isnan(s)]
    aggregated_results['test_f1_sklearn_std'] = np.std(f1_valid) if len(f1_valid) > 1 else 0.0
    auc_valid = [s for s in aggregated_results['fold_auc_scores'] if not np.isnan(s)]
    aggregated_results['test_auc_sklearn_std'] = np.std(auc_valid) if len(auc_valid) > 1 else 0.0

    if DEBUG_VERBOSE: print(f"===== Finished CV for {embedding_name} =====")
    return aggregated_results


# --- Dummy Data Generation and Test Run ---
def create_dummy_data(base_dir: str, num_proteins: int = 50, embedding_dim: int = 10, num_pos_pairs: int = 100,
                      num_neg_pairs: int = 100) -> Tuple[str, str, List[Dict[str, Any]]]:
    dummy_data_dir = os.path.join(base_dir, "dummy_data_temp")
    os.makedirs(dummy_data_dir, exist_ok=True)
    if DEBUG_VERBOSE: print(f"Creating dummy data in: {dummy_data_dir}")

    protein_ids = [f"P{i:03d}" for i in range(num_proteins)]

    # Create dummy embedding file
    dummy_emb_file = os.path.join(dummy_data_dir, "dummy_embeddings.h5")
    with h5py.File(dummy_emb_file, 'w') as hf:
        for pid in protein_ids:
            hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float32))

    dummy_embedding_config = [{"path": dummy_emb_file, "name": "DummyEmb"}]

    # Create dummy interaction files
    dummy_pos_path = os.path.join(dummy_data_dir, "dummy_positive_interactions.csv")
    dummy_neg_path = os.path.join(dummy_data_dir, "dummy_negative_interactions.csv")

    # Ensure enough unique pairs can be generated
    max_possible_pairs = num_proteins * (num_proteins - 1) // 2
    if num_pos_pairs > max_possible_pairs or num_neg_pairs > max_possible_pairs:
        print(
            f"Warning: Requested dummy pairs ({num_pos_pairs} pos, {num_neg_pairs} neg) might exceed possible unique pairs ({max_possible_pairs}) for {num_proteins} proteins. Duplicates may occur or fewer pairs generated.")
        num_pos_pairs = min(num_pos_pairs, max_possible_pairs)
        # num_neg_pairs = min(num_neg_pairs, max_possible_pairs) # Negatives can overlap with positives conceptually before filtering

    used_pairs = set()

    def generate_pairs(filepath, num_pairs_to_gen, label):
        pairs_data = []
        attempts = 0
        max_attempts = num_pairs_to_gen * 5  # Allow some attempts to find unique pairs
        while len(pairs_data) < num_pairs_to_gen and attempts < max_attempts:
            p1, p2 = np.random.choice(protein_ids, 2, replace=False)
            pair_tuple = tuple(sorted((p1, p2)))  # Canonical representation
            if pair_tuple not in used_pairs or label == 0:  # Allow negatives to overlap with positive *conceptually* before explicit separation
                if label == 1: used_pairs.add(
                    pair_tuple)  # Only add to used_pairs if it's a positive pair to avoid negative self-collision
                pairs_data.append([p1, p2])
            attempts += 1
        if len(pairs_data) < num_pairs_to_gen and DEBUG_VERBOSE:
            print(
                f"Warning: Could only generate {len(pairs_data)} unique pairs for label {label} out of requested {num_pairs_to_gen}.")

        pd.DataFrame(pairs_data).to_csv(filepath, header=False, index=False)

    generate_pairs(dummy_pos_path, num_pos_pairs, 1)
    generate_pairs(dummy_neg_path, num_neg_pairs, 0)  # Negatives generated independently for simplicity here

    if DEBUG_VERBOSE: print("Dummy data created.")
    return dummy_pos_path, dummy_neg_path, dummy_embedding_config


# --- Main Orchestration Logic (Refactored from run_ppi_evaluation.py) ---
def run_evaluation_pipeline(
        positive_interactions_fp: str,  # fp for filepath
        negative_interactions_fp: str,
        embedding_configs: List[Dict[str, Any]],  # List of {"path": ..., "name": ..., "loader_func_key": ...}
        output_root_dir: str,  # Base directory for all outputs of this run

        # Global parameters that might be overridden for dummy vs. real runs
        current_random_state: int,
        current_sample_negative_pairs: Optional[int],
        current_edge_embedding_method: str,
        current_n_folds: int,
        current_max_train_samples_cv: Optional[int],
        current_max_val_samples_cv: Optional[int],
        current_max_shuffle_buffer_size: int,
        current_plot_training_history: bool,
        current_mlp_params: Dict[str, Any],
        current_batch_size: int,
        current_epochs: int,
        current_learning_rate: float,
        current_k_vals_ranking: List[int],
        current_k_vals_table: List[int],
        current_main_embedding_name_stats: str,
        current_stat_test_metric: str,
        current_stat_test_alpha: float
):
    print(f"\n--- Starting Evaluation Pipeline ---")
    print(f"Output will be saved in: {output_root_dir}")
    plots_dir = os.path.join(output_root_dir, "plots")
    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Define loader function map (now uses FileOps class defined above)
    loader_function_map: Dict[str, Callable[[str, Optional[Set[str]]], Dict[str, np.ndarray]]] = {
        'load_h5_embeddings_selectively': FileOps.load_h5_embeddings_selectively,
        'load_h5_embeddings': FileOps.load_h5_embeddings_selectively,  # Alias
        'load_custom_embeddings': FileOps.load_custom_embeddings
    }
    default_embedding_loader_key = 'load_h5_embeddings_selectively'

    print("Loading all interaction pairs ONCE to determine required proteins...")
    positive_pairs_all = ProteinFileOps.load_interaction_pairs(
        positive_interactions_fp, 1, random_state_for_sampling=current_random_state  # No sampling for positives usually
    )
    negative_pairs_all = ProteinFileOps.load_interaction_pairs(
        negative_interactions_fp, 0,
        sample_n=current_sample_negative_pairs,
        random_state_for_sampling=current_random_state
    )

    all_interaction_pairs_for_ids = positive_pairs_all + negative_pairs_all
    if not all_interaction_pairs_for_ids:
        print("CRITICAL: No interaction pairs loaded. Exiting pipeline.")
        return

    required_protein_ids_for_interactions = set()
    for p1, p2, _ in all_interaction_pairs_for_ids:
        required_protein_ids_for_interactions.add(p1)
        required_protein_ids_for_interactions.add(p2)
    if DEBUG_VERBOSE:
        print(
            f"Found {len(required_protein_ids_for_interactions)} unique protein IDs in interaction files that need embeddings.")

    # Process embedding configurations
    processed_embedding_configs: List[Dict[str, Any]] = []
    for item in embedding_configs:
        config = {}
        path = item.get('path')
        name = item.get('name')
        loader_key = item.get('loader_func_key', default_embedding_loader_key)

        if not path: print(f"Warning: Path missing in item: {item}. Skipping."); continue
        norm_path = os.path.normpath(path)
        if not os.path.exists(norm_path): print(
            f"Warning: Embedding path does not exist: {norm_path}. Skipping."); continue

        config['path'] = norm_path
        config['name'] = name if name else os.path.splitext(os.path.basename(norm_path))[0]

        actual_loader_func = loader_function_map.get(loader_key)
        if not actual_loader_func:
            print(f"Warning: Loader '{loader_key}' not found for {config['name']}. Using default.")
            actual_loader_func = loader_function_map.get(default_embedding_loader_key)
            if not actual_loader_func:  # Should not happen if default is in map
                print(
                    f"CRITICAL: Default loader '{default_embedding_loader_key}' also not found. Skipping {config['name']}.");
                continue
        config['loader_func'] = actual_loader_func
        processed_embedding_configs.append(config)

    if not processed_embedding_configs:
        print("No valid embedding configurations found after processing. Exiting pipeline.")
        return

    all_cv_results: List[Dict[str, Any]] = []
    for config_item in processed_embedding_configs:
        if DEBUG_VERBOSE:
            print(f"\n{'=' * 25} Processing CV for: {config_item['name']} (Path: {config_item['path']}) {'=' * 25}")

        protein_embeddings = config_item['loader_func'](config_item['path'],
                                                        required_protein_ids_for_interactions)

        if protein_embeddings and len(protein_embeddings) > 0:
            # Further check if loaded embeddings are relevant
            actually_loaded_ids = set(protein_embeddings.keys())
            relevant_loaded_ids = actually_loaded_ids.intersection(required_protein_ids_for_interactions)

            if not relevant_loaded_ids:
                print(
                    f"Skipping {config_item['name']}: No relevant protein embeddings loaded that are part of the interaction dataset.")
                # Add a placeholder result
                all_cv_results.append({'embedding_name': config_item['name'], 'notes': "No relevant embeddings found.",
                                       'fold_f1_scores': [],
                                       'fold_auc_scores': []})  # Add more default keys if needed by downstream
                continue
            # (Original script had a check for <2 relevant_loaded_ids, which might be too strict if one protein is highly connected.
            #  The create_edge_embeddings will handle pairs where one or both embeddings are missing.)

            cv_run_result = main_workflow_cv(  # Call the CV worker function
                embedding_name=config_item['name'],
                protein_embeddings=protein_embeddings,
                positive_pairs=positive_pairs_all,  # Pass the full lists
                negative_pairs=negative_pairs_all,
                mlp_params_dict=current_mlp_params,
                edge_emb_method=current_edge_embedding_method,
                num_folds=current_n_folds,
                rand_state=current_random_state,
                max_train_samp_cv=current_max_train_samples_cv,
                max_val_samp_cv=current_max_val_samples_cv,
                max_shuff_buffer=current_max_shuffle_buffer_size,
                cv_batch_size=current_batch_size,
                cv_epochs=current_epochs,
                cv_learning_rate=current_learning_rate,
                k_vals_ranking=current_k_vals_ranking
            )
            if cv_run_result:
                all_cv_results.append(cv_run_result)
                history_fold1 = cv_run_result.get('history_dict_fold1', {})
                if current_plot_training_history and history_fold1 and any(
                        isinstance(v, list) and len(v) > 0 for v in history_fold1.values()):
                    plot_training_history(history_fold1, cv_run_result['embedding_name'], plots_dir, fold_num=1)
        else:
            print(
                f"Skipping CV for {config_item['name']}: Failed to load or embeddings are empty/irrelevant after selective loading.")
            all_cv_results.append(
                {'embedding_name': config_item['name'], 'notes': "Failed to load or no relevant embeddings.",
                 'fold_f1_scores': [], 'fold_auc_scores': []})

        del protein_embeddings;
        gc.collect()  # Manual garbage collection

    # Post-processing: Plots, Tables, Stats
    if all_cv_results:
        if DEBUG_VERBOSE:
            print("\nDEBUG: Final all_cv_results before aggregate plots/table (summary):")
            for i, res_dict in enumerate(all_cv_results):
                print(
                    f"  Summary for CV run {i + 1} ({res_dict.get('embedding_name')}): F1_avg={res_dict.get('test_f1_sklearn', 0.0):.4f}, AUC_avg={res_dict.get('test_auc_sklearn', 0.0):.4f}, Notes: {res_dict.get('notes')}")

        print("\nGenerating aggregate comparison plots & table (based on CV averages)...")

        # Check for valid ROC data before attempting to plot
        valid_roc_data_exists = any(
            isinstance(res.get('roc_data_representative'), tuple) and len(res['roc_data_representative']) == 3 and
            res['roc_data_representative'][0] is not None and hasattr(res['roc_data_representative'][0],
                                                                      '__len__') and len(
                res['roc_data_representative'][0]) > 0
            for res in all_cv_results
        )
        if valid_roc_data_exists:
            plot_roc_curves(all_cv_results, plots_dir)
        else:
            print("No valid representative ROC data to plot across models.")

        plot_comparison_charts(all_cv_results, plots_dir, current_k_vals_table)
        print_results_table(all_cv_results, current_k_vals_table, is_cv=True, output_dir=output_root_dir,
                            filename="ppi_evaluation_summary_table.txt")
        perform_statistical_tests(all_cv_results, current_main_embedding_name_stats, current_stat_test_metric,
                                  current_stat_test_alpha)
    else:
        print("\nNo results generated from any configurations to plot or tabulate.")

    print(f"--- Evaluation Pipeline Finished. Results in: {output_root_dir} ---")


# --- Main Execution Block ---
if __name__ == '__main__':
    print(f"Running Protein-Protein Interaction Evaluation Script...")
    print(f"Numpy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")  # Added sklearn version

    # --- MLP Parameters Dictionary ---
    mlp_parameters_global = {
        'dense1_units': MLP_DENSE1_UNITS, 'dropout1_rate': MLP_DROPOUT1_RATE,
        'dense2_units': MLP_DENSE2_UNITS, 'dropout2_rate': MLP_DROPOUT2_RATE,
        'l2_reg': MLP_L2_REG
    }

    # --- DUMMY/TEST CASE RUN ---
    if RUN_DUMMY_TEST_FIRST:
        print("\n" + "=" * 30 + " RUNNING DUMMY TEST CASE " + "=" * 30)
        dummy_output_dir = os.path.join(BASE_OUTPUT_DIR, "dummy_run_output")

        dummy_pos_fp, dummy_neg_fp, dummy_emb_configs = create_dummy_data(
            base_dir=BASE_OUTPUT_DIR,  # Create dummy_data_temp inside this
            num_proteins=20,
            embedding_dim=8,
            num_pos_pairs=30,
            num_neg_pairs=30
        )

        # Use a distinct main embedding name for dummy stats if its name is "DummyEmb"
        dummy_main_emb_name = "DummyEmb" if any(e['name'] == "DummyEmb" for e in dummy_emb_configs) else ""

        run_evaluation_pipeline(
            positive_interactions_fp=dummy_pos_fp,
            negative_interactions_fp=dummy_neg_fp,
            embedding_configs=dummy_emb_configs,
            output_root_dir=dummy_output_dir,
            current_random_state=RANDOM_STATE,  # Use global random state
            current_sample_negative_pairs=50,  # Small sample for dummy
            current_edge_embedding_method=EDGE_EMBEDDING_METHOD,
            current_n_folds=2,  # Minimal folds for speed
            current_max_train_samples_cv=40,  # Small samples
            current_max_val_samples_cv=20,
            current_max_shuffle_buffer_size=100,  # Small buffer
            current_plot_training_history=True,  # Good to test plotting
            current_mlp_params=mlp_parameters_global,  # Use global MLP params
            current_batch_size=16,  # Smaller batch for dummy
            current_epochs=2,  # Minimal epochs
            current_learning_rate=LEARNING_RATE,
            current_k_vals_ranking=K_VALUES_FOR_RANKING_METRICS[:1],  # Test with one K
            current_k_vals_table=K_VALUES_FOR_TABLE_DISPLAY[:1],  # Test with one K
            current_main_embedding_name_stats=dummy_main_emb_name,  # Specific for dummy
            current_stat_test_metric=STATISTICAL_TEST_METRIC_KEY,
            current_stat_test_alpha=STATISTICAL_TEST_ALPHA
        )

        if CLEANUP_DUMMY_DATA:
            dummy_data_temp_dir = os.path.join(BASE_OUTPUT_DIR, "dummy_data_temp")
            if os.path.exists(dummy_data_temp_dir):
                try:
                    shutil.rmtree(dummy_data_temp_dir)
                    print(f"Successfully cleaned up dummy data directory: {dummy_data_temp_dir}")
                except Exception as e:
                    print(f"Error cleaning up dummy data directory {dummy_data_temp_dir}: {e}")
        print("=" * 30 + " DUMMY TEST CASE FINISHED " + "=" * 30 + "\n")
        # Decide if you want to exit after dummy run or continue to normal run
        # For now, it will continue to normal run if configured.

    # --- NORMAL (USER-CONFIGURED) CASE RUN ---
    print("\n" + "=" * 30 + " RUNNING NORMAL EVALUATION CASE " + "=" * 30)

    # User Configuration - Paths and Embedding Files for the Normal Run
    # Ensure these paths are correct for your system
    normal_positive_interactions_path = os.path.normpath('C:/tmp/Models/ground_truth/positive_interactions.csv')
    normal_negative_interactions_path = os.path.normpath('C:/tmp/Models/ground_truth/negative_interactions.csv')
    normal_sample_negative_pairs: Optional[int] = 500000  # As per original config

    # List of embedding files to compare in the normal run
    normal_embedding_files_to_compare = [
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Directed_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Directed_ProtDiGCN", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Directed_Tong_Library_DiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Directed_Tong", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Directed_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Directed_GCN", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Undirected_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Undirected_ProtDiGCN", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Undirected_Tong_Library_DiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Undirected_Tong", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Undirected_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "name": "Undirected_GCN", "loader_func_key": "load_h5_embeddings_selectively"},
        {
            "path": "C:/tmp/Models/embeddings_to_evaluate/per-protein.h5",
            "name": "ProtT5", "loader_func_key": "load_h5_embeddings_selectively"},
        # Add other embeddings from your original EMBEDDING_FILES_TO_COMPARE list here
    ]

    # Check if interaction files for normal run exist
    if not os.path.exists(normal_positive_interactions_path) or not os.path.exists(normal_negative_interactions_path):
        print(f"CRITICAL ERROR: Interaction file paths for normal run are invalid. Skipping normal evaluation.")
        print(
            f"Positive path: {normal_positive_interactions_path} (Exists: {os.path.exists(normal_positive_interactions_path)})")
        print(
            f"Negative path: {normal_negative_interactions_path} (Exists: {os.path.exists(normal_negative_interactions_path)})")
    else:
        normal_output_dir = os.path.join(BASE_OUTPUT_DIR, "normal_run_output")
        run_evaluation_pipeline(
            positive_interactions_fp=normal_positive_interactions_path,
            negative_interactions_fp=normal_negative_interactions_path,
            embedding_configs=normal_embedding_files_to_compare,
            output_root_dir=normal_output_dir,
            current_random_state=RANDOM_STATE,
            current_sample_negative_pairs=normal_sample_negative_pairs,
            current_edge_embedding_method=EDGE_EMBEDDING_METHOD,
            current_n_folds=N_FOLDS,
            current_max_train_samples_cv=MAX_TRAIN_SAMPLES_CV,
            current_max_val_samples_cv=MAX_VAL_SAMPLES_CV,
            current_max_shuffle_buffer_size=MAX_SHUFFLE_BUFFER_SIZE,
            current_plot_training_history=PLOT_TRAINING_HISTORY,
            current_mlp_params=mlp_parameters_global,
            current_batch_size=BATCH_SIZE,
            current_epochs=EPOCHS,
            current_learning_rate=LEARNING_RATE,
            current_k_vals_ranking=K_VALUES_FOR_RANKING_METRICS,
            current_k_vals_table=K_VALUES_FOR_TABLE_DISPLAY,
            current_main_embedding_name_stats=MAIN_EMBEDDING_NAME_FOR_STATS,  # Use global default
            current_stat_test_metric=STATISTICAL_TEST_METRIC_KEY,
            current_stat_test_alpha=STATISTICAL_TEST_ALPHA
        )

    print("\nCombined Script finished.")
