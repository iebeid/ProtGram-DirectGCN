# ==============================================================================
# SCRIPT 3: Link Prediction Evaluation and Reporting
# PURPOSE: Evaluates embeddings, generates plots, tables, and a summary file.
# VERSION: 2.0 (Complete, No Placeholders)
# ==============================================================================

import os
import sys
import h5py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Scikit-learn and SciPy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, ndcg_score
from scipy.stats import wilcoxon

# --- TensorFlow GPU Configuration ---
tf.get_logger().setLevel('ERROR')
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow: GPU Devices Detected and Memory Growth enabled.")
    except RuntimeError as e:
        print(f"TensorFlow: Error setting memory growth: {e}")
else:
    print("TensorFlow: Warning: No GPU detected. Running on CPU.")


# ==============================================================================
# --- CONFIGURATION CLASS ---
# ==============================================================================
class ScriptConfig:
    """
    Centralized configuration class for the evaluation script.
    """

    def __init__(self):
        # --- GENERAL SETTINGS ---
        self.DEBUG_VERBOSE = True
        self.RANDOM_STATE = 42

        # !!! IMPORTANT: SET YOUR BASE DIRECTORIES HERE !!!
        self.BASE_DATA_DIR = "C:/ProgramData/ProtDiGCN/"
        self.BASE_OUTPUT_DIR = os.path.join(self.BASE_DATA_DIR, "ppi_evaluation_results_final_dummy")

        # --- INPUT FILES & DIRECTORIES ---
        self.LP_POSITIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/positive_interactions.csv')
        self.LP_NEGATIVE_INTERACTIONS_PATH = os.path.join(self.BASE_DATA_DIR, 'ground_truth/negative_interactions.csv')

        # !!! IMPORTANT: SET THE PATH TO YOUR EMBEDDING FILES TO TEST !!!
        self.LP_EMBEDDING_FILES_TO_EVALUATE = [{"path": os.path.join(self.BASE_DATA_DIR, "models/per-protein.h5"), "name": "ProtT5-Precomputed"},
            {"path": os.path.join(self.BASE_OUTPUT_DIR, "ngram_gcn_generated_embeddings", "per_protein_embeddings_from_3gram.h5"), "name": "NgramGCN-Generated"}, ]

        # --- OUTPUT DIRECTORY ---
        self.RESULTS_OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, "final_evaluation_results")

        # --- Link Prediction Evaluation Configuration ---
        self.LP_EDGE_EMBEDDING_METHOD = 'concatenate'
        self.LP_N_FOLDS = 5
        self.LP_DESIRED_POS_TO_NEG_RATIO: float = 1.0
        self.LP_MLP_DENSE1_UNITS = 128
        self.LP_MLP_DROPOUT1_RATE = 0.4
        self.LP_MLP_DENSE2_UNITS = 64
        self.LP_MLP_DROPOUT2_RATE = 0.4
        self.LP_MLP_L2_REG = 0.001
        self.LP_BATCH_SIZE = 128
        self.LP_EPOCHS = 10
        self.LP_LEARNING_RATE = 1e-3

        # --- Reporting & Analysis Configuration ---
        self.LP_PLOT_TRAINING_HISTORY = True
        self.LP_K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
        self.LP_K_VALUES_FOR_TABLE_DISPLAY = [50, 100]
        self.LP_MAIN_EMBEDDING_NAME_FOR_STATS = "ProtT5-Precomputed"
        self.LP_STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'
        self.LP_STATISTICAL_TEST_ALPHA = 0.05


# ==============================================================================
# --- CORE EVALUATION FUNCTIONS ---
# ==============================================================================

def load_h5_embeddings_selectively(filepath: str, protein_ids: set) -> dict:
    embeddings = {}
    try:
        with h5py.File(filepath, 'r') as hf:
            keys_to_load = [pid for pid in protein_ids if pid in hf]
            for prot_id in keys_to_load:
                embeddings[prot_id] = hf[prot_id][:]
    except Exception as e:
        print(f"Error loading H5 file {filepath}: {e}")
    return embeddings


def create_edge_embedding_generator(pairs_df: pd.DataFrame, embeddings: dict, batch_size: int, embed_method: str):
    num_samples = len(pairs_df)

    def generator():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_pairs = pairs_df.iloc[start:end]
            batch_embeds1 = [embeddings.get(p1) for p1 in batch_pairs['protein1']]
            batch_embeds2 = [embeddings.get(p2) for p2 in batch_pairs['protein2']]
            labels = batch_pairs['label'].values
            valid_indices = [i for i, (e1, e2) in enumerate(zip(batch_embeds1, batch_embeds2)) if e1 is not None and e2 is not None]
            if not valid_indices: continue
            valid_embeds1 = np.array([batch_embeds1[i] for i in valid_indices])
            valid_embeds2 = np.array([batch_embeds2[i] for i in valid_indices])
            valid_labels = labels[valid_indices]
            if embed_method == 'concatenate':
                edge_features = np.concatenate([valid_embeds1, valid_embeds2], axis=1)
            elif embed_method == 'hadamard':
                edge_features = valid_embeds1 * valid_embeds2
            elif embed_method == 'average':
                edge_features = (valid_embeds1 + valid_embeds2) / 2
            else:  # Default to concatenate
                edge_features = np.concatenate([valid_embeds1, valid_embeds2], axis=1)
            yield edge_features, valid_labels

    return generator


def build_mlp_model_lp(input_dim: int, config: ScriptConfig) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(config.LP_MLP_DENSE1_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(inp)
    x = Dropout(config.LP_MLP_DROPOUT1_RATE)(x)
    x = Dense(config.LP_MLP_DENSE2_UNITS, activation='relu', kernel_regularizer=l2(config.LP_MLP_L2_REG))(x)
    x = Dropout(config.LP_MLP_DROPOUT2_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    optimizer = Adam(learning_rate=config.LP_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc_keras')])
    return model


def run_evaluation_for_one_embedding(config: ScriptConfig, eval_pairs_df: pd.DataFrame, emb_name: str, embeddings: dict):
    if eval_pairs_df.empty: return None
    aggregated_results = {'embedding_name': emb_name, 'history_dict_fold1': {}, 'roc_data_representative': (None, None, 0.0)}
    skf = StratifiedKFold(n_splits=config.LP_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(eval_pairs_df, eval_pairs_df['label'])):
        print(f"\n-- Fold {fold + 1}/{config.LP_N_FOLDS} for {emb_name} --")
        train_df, test_df = eval_pairs_df.iloc[train_idx], eval_pairs_df.iloc[test_idx]
        embed_dim = next(iter(embeddings.values())).shape[0]
        input_dim = embed_dim * 2 if config.LP_EDGE_EMBEDDING_METHOD == 'concatenate' else embed_dim
        model = build_mlp_model_lp(input_dim, config)

        train_gen_callable = create_edge_embedding_generator(train_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)
        test_gen_callable = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)
        train_dataset = tf.data.Dataset.from_generator(train_gen_callable, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape([None, input_dim]), tf.TensorShape([None, ])))
        test_dataset = tf.data.Dataset.from_generator(test_gen_callable, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape([None, input_dim]), tf.TensorShape([None, ])))

        history = model.fit(train_dataset, epochs=config.LP_EPOCHS, validation_data=test_dataset, verbose=1 if config.DEBUG_VERBOSE else 0)
        if fold == 0: aggregated_results['history_dict_fold1'] = history.history

        y_pred_probs = model.predict(test_dataset).flatten()
        y_true_gen = create_edge_embedding_generator(test_df, embeddings, config.LP_BATCH_SIZE, config.LP_EDGE_EMBEDDING_METHOD)()
        y_true_aligned = np.concatenate([labels for _, labels in y_true_gen])

        current_fold_metrics = {}
        y_pred_class = (y_pred_probs > 0.5).astype(int)
        current_fold_metrics['test_precision_sklearn'] = precision_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_recall_sklearn'] = recall_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_f1_sklearn'] = f1_score(y_true_aligned, y_pred_class, zero_division=0)
        current_fold_metrics['test_auc_sklearn'] = roc_auc_score(y_true_aligned, y_pred_probs) if len(np.unique(y_true_aligned)) > 1 else 0.5

        if fold == 0 and len(np.unique(y_true_aligned)) > 1:
            fpr, tpr, _ = roc_curve(y_true_aligned, y_pred_probs)
            aggregated_results['roc_data_representative'] = (fpr, tpr, current_fold_metrics['test_auc_sklearn'])

        desc_indices = np.argsort(y_pred_probs)[::-1]
        sorted_y_true = y_true_aligned[desc_indices]
        for k in config.LP_K_VALUES_FOR_RANKING_METRICS:
            eff_k = min(k, len(sorted_y_true))
            current_fold_metrics[f'test_hits_at_{k}'] = np.sum(sorted_y_true[:eff_k]) if eff_k > 0 else 0
            current_fold_metrics[f'test_ndcg_at_{k}'] = ndcg_score([y_true_aligned], [y_pred_probs], k=eff_k) if eff_k > 0 and len(np.unique(y_true_aligned)) > 1 else 0.0

        fold_metrics_list.append(current_fold_metrics)
        print(f"Fold {fold + 1} Results: AUC={current_fold_metrics['test_auc_sklearn']:.4f}, F1={current_fold_metrics['test_f1_sklearn']:.4f}")

    if not fold_metrics_list: return None
    for key in fold_metrics_list[0].keys():
        values = [d[key] for d in fold_metrics_list]
        aggregated_results[key] = np.mean(values)
        aggregated_results[f"{key}_std"] = np.std(values)
    aggregated_results['fold_auc_scores'] = [d.get('test_auc_sklearn', 0.0) for d in fold_metrics_list]
    aggregated_results['fold_f1_scores'] = [d.get('test_f1_sklearn', 0.0) for d in fold_metrics_list]
    return aggregated_results


# ==============================================================================
# --- MODIFIED: REPORTING AND PLOTTING FUNCTIONS (Save to File) ---
# ==============================================================================

def plot_training_history(history_dict: dict, model_name: str, fold_num: int, output_dir: str):
    if not history_dict: return
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict: plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss: {model_name} (Fold {fold_num})');
    plt.xlabel('Epoch');
    plt.legend();
    plt.grid(True)
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict: plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy: {model_name} (Fold {fold_num})');
    plt.xlabel('Epoch');
    plt.legend();
    plt.grid(True)

    plt.suptitle(f"Training History: {model_name}", fontsize=16);
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"training_history_{model_name.replace(' ', '_')}_fold{fold_num}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()  # Free memory
    print(f"Saved training history plot to {filename}")


def plot_roc_curves(results_list: list, output_dir: str):
    plt.figure(figsize=(10, 8))
    for result in results_list:
        if 'roc_data_representative' in result:
            fpr, tpr, auc_val = result['roc_data_representative']
            if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0:
                plt.plot(fpr, tpr, lw=2, label=f"{result['embedding_name']} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison (from First Fold)');
    plt.legend(loc="lower right");
    plt.grid(True)

    filename = "roc_curves_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved ROC curve plot to {filename}")


def plot_comparison_charts(results_list: list, config: ScriptConfig, output_dir: str):
    metrics_to_plot = {'AUC': 'test_auc_sklearn', 'F1-Score': 'test_f1_sklearn', 'Precision': 'test_precision_sklearn', 'Recall': 'test_recall_sklearn'}
    for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY:
        metrics_to_plot[f'Hits@{k}'] = f'test_hits_at_{k}';
        metrics_to_plot[f'NDCG@{k}'] = f'test_ndcg_at_{k}'

    embedding_names = [res['embedding_name'] for res in results_list]
    num_metrics = len(metrics_to_plot)
    cols = min(3, num_metrics);
    rows = math.ceil(num_metrics / cols)
    plt.figure(figsize=(cols * 6, rows * 5))

    for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        plt.subplot(rows, cols, i + 1)
        values = [res.get(metric_key, 0) for res in results_list]
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(embedding_names))))
        plt.ylabel('Score');
        plt.title(f'{metric_name} Comparison');
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, max(values) * 1.15 if any(v > 0 for v in values) else 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = "metrics_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved metrics comparison plot to {filename}")


def write_results_table(results_list: list, config: ScriptConfig, file_handle):
    if not results_list: return

    headers = ["Embedding Name", "AUC", "F1", "Precision", "Recall"]
    for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY: headers.extend([f"Hits@{k}", f"NDCG@{k}"])
    headers.extend(["AUC StdDev", "F1 StdDev"])

    table_data = []
    for res in results_list:
        row = [res['embedding_name'], f"{res.get('test_auc_sklearn', 0):.4f}", f"{res.get('test_f1_sklearn', 0):.4f}", f"{res.get('test_precision_sklearn', 0):.4f}", f"{res.get('test_recall_sklearn', 0):.4f}"]
        for k in config.LP_K_VALUES_FOR_TABLE_DISPLAY:
            row.append(f"{res.get(f'test_hits_at_{k}', 0)}")
            row.append(f"{res.get(f'test_ndcg_at_{k}', 0):.4f}")
        row.append(f"{res.get('test_auc_sklearn_std', 0):.4f}")
        row.append(f"{res.get('test_f1_sklearn_std', 0):.4f}")
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=headers)

    file_handle.write("--- Overall Performance Comparison Table (Averaged over Folds) ---\n")
    file_handle.write(df.to_string(index=False))
    file_handle.write("\n\n")


def write_statistical_tests(results_list: list, config: ScriptConfig, file_handle):
    if len(results_list) < 2: return

    main_emb_name = config.LP_MAIN_EMBEDDING_NAME_FOR_STATS
    metric_key = config.LP_STATISTICAL_TEST_METRIC_KEY
    alpha = config.LP_STATISTICAL_TEST_ALPHA
    fold_scores_key = 'fold_auc_scores' if 'auc' in metric_key else 'fold_f1_scores'

    main_res = next((r for r in results_list if r['embedding_name'] == main_emb_name), None)
    if not main_res or fold_scores_key not in main_res:
        file_handle.write(f"--- Statistical Comparison ---\n")
        file_handle.write(f"Main model '{main_emb_name}' or its fold scores not found. Skipping tests.\n")
        return

    file_handle.write(f"--- Statistical Comparison vs '{main_emb_name}' on '{metric_key}' (Alpha={alpha}) ---\n")
    main_scores = main_res[fold_scores_key]
    header = f"{'Compared Embedding':<30} | {'p-value':<12} | {'Significant?':<15}\n"
    file_handle.write(header)
    file_handle.write("-" * len(header) + "\n")

    for other_res in [r for r in results_list if r['embedding_name'] != main_emb_name]:
        other_scores = other_res.get(fold_scores_key, [])
        if len(main_scores) != len(other_scores) or len(main_scores) == 0:
            result_line = f"{other_res['embedding_name']:<30} | {'N/A (score mismatch)':<12} | {'-':<15}\n"
        else:
            stat, p_value = wilcoxon(main_scores, other_scores)
            conclusion = "Yes" if p_value < alpha else "No"
            result_line = f"{other_res['embedding_name']:<30} | {p_value:<12.4f} | {conclusion:<15}\n"
        file_handle.write(result_line)


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    print("--- SCRIPT 3: Link Prediction Evaluation and Reporting ---")
    config = ScriptConfig()
    os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)

    try:
        all_pos_df = pd.read_csv(config.LP_POSITIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()
        all_neg_df = pd.read_csv(config.LP_NEGATIVE_INTERACTIONS_PATH, dtype=str, header=None, names=['protein1', 'protein2']).dropna()
        all_pos_df['label'] = 1;
        all_neg_df['label'] = 0
        print(f"Loaded {len(all_pos_df)} total positive and {len(all_neg_df)} total negative candidate pairs.")
    except FileNotFoundError:
        print("Interaction CSV files not found. Aborting.");
        sys.exit(1)

    all_cv_results = []

    for emb_config in config.LP_EMBEDDING_FILES_TO_EVALUATE:
        path = emb_config.get("path")
        name = emb_config.get("name", os.path.basename(str(path)) if path else "Unknown")

        print(f"\n--- Evaluating file: {name} ---")
        if not path or not os.path.exists(path): print(f"File not found: {path}. Skipping."); continue

        all_proteins_in_network = set(all_pos_df['protein1']) | set(all_pos_df['protein2']) | set(all_neg_df['protein1']) | set(all_neg_df['protein2'])
        protein_embeddings = load_h5_embeddings_selectively(path, all_proteins_in_network)
        if not protein_embeddings: print(f"No relevant embeddings loaded. Skipping."); continue

        valid_pos_mask = all_pos_df['protein1'].isin(protein_embeddings.keys()) & all_pos_df['protein2'].isin(protein_embeddings.keys())
        valid_pos_df = all_pos_df[valid_pos_mask]
        valid_neg_mask = all_neg_df['protein1'].isin(protein_embeddings.keys()) & all_neg_df['protein2'].isin(protein_embeddings.keys())
        valid_neg_df = all_neg_df[valid_neg_mask]

        target_neg_count = int(len(valid_pos_df) / config.LP_DESIRED_POS_TO_NEG_RATIO)
        print(f"Found {len(valid_pos_df)} valid positive pairs. Sampling {target_neg_count} negative pairs from {len(valid_neg_df)} valid candidates.")

        if len(valid_neg_df) >= target_neg_count:
            sampled_neg_df = valid_neg_df.sample(n=target_neg_count, random_state=config.RANDOM_STATE)
        else:
            print(f"Warning: Not enough valid negative pairs. Using all {len(valid_neg_df)}.");
            sampled_neg_df = valid_neg_df

        eval_pairs_df = pd.concat([valid_pos_df, sampled_neg_df], ignore_index=True)
        eval_pairs_df = eval_pairs_df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print(f"Created and shuffled evaluation set with {len(eval_pairs_df)} pairs.")

        results_for_emb = run_evaluation_for_one_embedding(config, eval_pairs_df, name, protein_embeddings)
        if results_for_emb:
            all_cv_results.append(results_for_emb)
            if config.LP_PLOT_TRAINING_HISTORY:
                plot_training_history(results_for_emb.get('history_dict_fold1', {}), name, fold_num=1, output_dir=config.RESULTS_OUTPUT_DIR)

    if all_cv_results:
        print("\n\n" + "#" * 20 + " FINALIZING RESULTS " + "#" * 20)

        # Write text-based reports to summary file
        summary_path = os.path.join(config.RESULTS_OUTPUT_DIR, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            write_results_table(all_cv_results, config, f)
            write_statistical_tests(all_cv_results, config, f)
        print(f"Saved results table and statistical tests to: {summary_path}")

        # Generate and save plots
        plot_comparison_charts(all_cv_results, config, config.RESULTS_OUTPUT_DIR)
        plot_roc_curves(all_cv_results, config.RESULTS_OUTPUT_DIR)
    else:
        print("\nNo embeddings were successfully evaluated.")

    print("\n--- Script 3 Finished ---")
