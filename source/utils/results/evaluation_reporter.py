# ==============================================================================
# MODULE: utils/evaluation_reporter.py
# PURPOSE: Contains all functions for plotting results and writing summary
#          files for the PPI evaluation trainers.
# VERSION: 3.1 (Added error bars to comparison charts for better visualization)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import random
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import math
import h5py
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon, pearsonr

# Conditionally import SHAP to avoid making it a hard dependency
try:
    import shap
except ImportError:
    shap = None

from sklearn.manifold import TSNE
from source.utils.post.embedding_loader import EmbeddingLoader

# --- Configuration for t-SNE plotting ---
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_INIT_PCA = True
TSNE_LEARNING_RATE = 'auto'
SAMPLE_N_FOR_COMBINED_TSNE = 2000




class EvaluationReporter:
    """
    A class to handle plotting of results, writing summary files,
    and generating t-SNE visualizations.
    """

    def __init__(self, base_output_dir: str, k_vals_table: List[int]):
        """
        Initializes the reporter with a base directory for outputs.

        Args:
            base_output_dir (str): The root directory where all reports and plots will be saved.
            k_vals_table (List[int]): List of k values for Hits@k and NDCG@k metrics.
        """
        self.base_output_dir = Path(base_output_dir)
        self.plots_output_dir = self.base_output_dir / "plots"
        self.summary_file_output_dir = self.base_output_dir
        self.k_vals_table = k_vals_table
        self.plots_output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file_output_dir.mkdir(parents=True, exist_ok=True)



    def plot_training_history(self, history_dict: Dict[str, Any], model_name: str) -> Optional[Path]:
        """
        Plots the training and validation loss/accuracy from a Keras history object.
        """
        if not history_dict:
            print(f"Plotting: No history data for {model_name} to plot.")
            return None

        plot_filename = self.plots_output_dir / f"history_{model_name.replace(' ', '_')}.png"

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if 'loss' in history_dict and history_dict['loss']:
            plt.plot(history_dict['loss'], label='Training Loss')
        if 'val_loss' in history_dict and history_dict['val_loss']:
            plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss: {model_name}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # FIX: Robustly find the primary metric (e.g., 'accuracy', 'auc')
        metric_key = next((k for k in ['accuracy', 'auc'] if k in history_dict), None)
        val_metric_key = f'val_{metric_key}' if metric_key else None

        if metric_key and metric_key in history_dict and history_dict[metric_key]:
            plt.plot(history_dict[metric_key], label=f'Training {metric_key.capitalize()}')
        if val_metric_key and val_metric_key in history_dict and history_dict[val_metric_key]:
            plt.plot(history_dict[val_metric_key], label=f'Validation {metric_key.capitalize()}')

        metric_title = metric_key.capitalize() if metric_key else "Metric"
        plt.title(f'Model {metric_title}: {model_name}')
        plt.ylabel(metric_title)
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Training History: {model_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        try:
            plt.savefig(plot_filename)
            print(f"  Saved training history plot to {plot_filename}")
        except Exception as e:
            print(f"  Error saving plot {plot_filename}: {e}")
        plt.close()
        return plot_filename

    def plot_roc_curves(self, results_list: List[Dict[str, Any]]) -> Optional[Path]:
        """
        Plots a comparison of ROC curves from multiple model evaluation results.
        """
        plot_filename = self.plots_output_dir / "comparison_roc_curves.png"

        plt.figure(figsize=(10, 8))
        plotted_anything = False
        for result in results_list:
            if 'roc_data_representative' in result and result['roc_data_representative'][0].size > 0:
                fpr, tpr, _ = result['roc_data_representative']
                avg_auc = result.get('test_auc_sklearn', 0.0)
                plt.plot(fpr, tpr, lw=2, label=f"{result.get('embedding_name', 'Unknown')} (Avg AUC = {avg_auc:.4f})")
                plotted_anything = True

        if not plotted_anything:
            print("Plotting: No valid ROC data available for any model.")
            plt.close()
            return None

        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)

        try:
            plt.savefig(plot_filename)
            print(f"  Saved ROC comparison plot to {plot_filename}")
        except Exception as e:
            print(f"  Error saving ROC plot {plot_filename}: {e}")
        plt.close()
        return plot_filename

    def plot_comparison_charts(self, results_list: List[Dict[str, Any]]) -> Optional[Path]:
        """
        Generates a set of bar charts comparing key performance metrics across all models,
        including error bars for standard deviation.
        """
        if not results_list:
            print("Plotting: No results data provided for comparison charts.")
            return None

        plot_filename = self.plots_output_dir / "comparison_metrics_barchart.png"

        metrics = {'AUC': 'test_auc_sklearn', 'F1-Score': 'test_f1_sklearn', 'Precision': 'test_precision_sklearn', 'Recall': 'test_recall_sklearn'}
        for k in self.k_vals_table:
            metrics[f'Hits@{k}'] = f'test_hits_at_{k}'
            metrics[f'NDCG@{k}'] = f'test_ndcg_at_{k}'

        names = [res.get('embedding_name', 'Unknown') for res in results_list]
        num_metrics = len(metrics)
        cols = min(3, num_metrics)
        rows = math.ceil(num_metrics / cols)

        plt.figure(figsize=(cols * 6, rows * 5))
        for i, (name, key) in enumerate(metrics.items()):
            plt.subplot(rows, cols, i + 1)
            values = [res.get(key, 0) for res in results_list]
            # FIX: Add error bars using the standard deviation from cross-validation
            std_dev_key = f"{key}_std"
            errors = [res.get(std_dev_key, 0) for res in results_list]

            bars = plt.bar(names, values, yerr=errors, capsize=5, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(names))), alpha=0.8)
            plt.ylabel('Score')
            plt.title(name)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(bottom=0, top=max(1.0, max(values) * 1.1))
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle("Model Performance Comparison (Averaged over Folds)", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        try:
            plt.savefig(plot_filename)
            print(f"  Saved metrics comparison barchart to {plot_filename}")
        except Exception as e:
            print(f"  Error saving comparison chart {plot_filename}: {e}")
        plt.close()
        return plot_filename

    def generate_shap_summary(self, model, background_data: np.ndarray, model_name: str, fold_num: int) -> Optional[Path]:
        """
        Generates and saves a SHAP summary plot to explain model predictions.

        Note: This interprets the importance of the *input features to the MLP*,
        which are the dimensions of the concatenated protein embeddings, not the
        n-grams themselves.
        """
        # --- NEW: Check if SHAP is installed before proceeding ---
        if shap is None:
            print("  SHAP library not installed. Skipping SHAP summary generation.")
            print("  To enable, please install it: pip install shap")
            return None

        plot_filename = self.plots_output_dir / f"shap_summary_{model_name.replace(' ', '_')}_Fold{fold_num}.png"
        print(f"  Generating SHAP summary plot for {model_name}...")

        try:
            # The background data is now pre-summarized (e.g., by k-means)
            # before being passed to this function, so we can use it directly.

            # --- REFACTOR: Use DeepExplainer for TensorFlow models ---
            # DeepExplainer is significantly faster and optimized for deep learning models.
            explainer = shap.DeepExplainer(model, background_data)
            shap_values = explainer.shap_values(background_data)  # Explain the background data

            # For a single-output model, shap_values is a list with one array.
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            plt.figure()
            # --- NEW: Limit the number of features displayed for clarity ---
            shap.summary_plot(shap_values, background_data, show=False, plot_type="bar", max_display=20)
            plt.title(f"SHAP Feature Importance\n({model_name} - Fold {fold_num})")
            plt.tight_layout()
            plt.savefig(plot_filename)
            print(f"  Saved SHAP summary plot to {plot_filename}")
        except Exception as e:
            print(f"  Error generating SHAP plot for {model_name}: {e}")
            import traceback
            traceback.print_exc()
        plt.close()
        return plot_filename

    def generate_pooling_attention_plot(self, attention_json_path: Path, model_name: str, num_top_ngrams: int = 20) -> Optional[Path]:
        """
        Generates a bar chart showing the n-grams with the highest attention
        weights for a sample protein from the pooling strategy.
        """
        if not attention_json_path.exists():
            print(f"  Pooling attention file not found: {attention_json_path}. Skipping plot.")
            return None

        print(f"  Generating pooling attention plot for {model_name} from {attention_json_path.name}...")

        try:
            # --- DEFINITIVE FIX for OOM Crash: Read only the first line of the JSONL file ---
            # This avoids loading the entire (potentially massive) attention log into memory.
            with open(attention_json_path, 'r') as f:
                first_line = f.readline()
                if not first_line:
                    print("  No attention data found in file.")
                    return None
                # The first line contains the entire JSON object for the first protein.
                attention_data = json.loads(first_line)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error reading attention JSON file: {e}")
            return None

        # Select a sample protein to visualize (e.g., the first one)
        sample_protein_id = next(iter(attention_data))
        protein_attention = attention_data[sample_protein_id]

        # Sort by weight and take the top N
        sorted_ngrams = sorted(protein_attention.items(), key=lambda item: item[1], reverse=True)
        top_ngrams = dict(sorted_ngrams[:num_top_ngrams])

        plt.figure(figsize=(12, 8))
        plt.bar(top_ngrams.keys(), top_ngrams.values(), color='skyblue')
        plt.xlabel("N-Grams")
        plt.ylabel("Attention Weight")
        plt.title(f"Top {num_top_ngrams} N-Gram Attention Weights for Protein '{sample_protein_id}'\n(Model: {model_name})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_filename = self.plots_output_dir / f"pooling_attention_{model_name.replace(' ', '_')}.png"
        try:
            plt.savefig(plot_filename)
            print(f"  Saved attention summary plot to {plot_filename}")
        except Exception as e:
            print(f"  Error saving attention plot {plot_filename}: {e}")
        plt.close()
        return plot_filename

    def generate_hierarchical_attention_plot(self, attention_json_path: Path, model_name: str, num_samples_per_level: int = 5) -> Optional[Path]:
        """
        Generates a set of bar charts showing the attention weights for a sample
        of n-grams from each level of the hierarchy.
        """
        if not attention_json_path.exists():
            print(f"  Hierarchical attention file not found: {attention_json_path}. Skipping plot.")
            return None

        print(f"  Generating hierarchical attention plot for {model_name} from {attention_json_path.name}...")

        try:
            # --- DEFINITIVE FIX for OOM Crash: Read the JSONL file line-by-line ---
            attention_data = {}
            with open(attention_json_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Each line is a JSON object like {"2": {...}}
                        line_data = json.loads(line)
                        # The key is the n-gram level (as a string)
                        level_key_str = next(iter(line_data))
                        level_data = line_data[level_key_str]
                        attention_data[int(level_key_str)] = level_data
        except (json.JSONDecodeError, IOError, ValueError, StopIteration) as e:
            print(f"  Error reading or parsing attention JSONL file: {e}")
            return None

        if not attention_data:
            print("  No hierarchical attention data found in file.")
            return None

        # The keys are n-gram levels (e.g., 2, 3). Sort them numerically.
        levels = sorted(attention_data.keys())
        num_levels = len(levels)
        if num_levels == 0:
            print("  Attention data is empty, no levels to plot.")
            return None

        fig, axes = plt.subplots(num_levels, 1, figsize=(12, 6 * num_levels), squeeze=False)
        fig.suptitle(f'Hierarchical Attention Weights\n(Model: {model_name})', fontsize=16)

        for i, level in enumerate(levels):
            ax = axes[i, 0]
            level_data = attention_data[level]

            if not level_data:
                ax.text(0.5, 0.5, f'No attention data for n={level}', ha='center', va='center')
                ax.set_title(f'N-Gram Level: {level}')
                continue

            # Take a random sample of n-grams to visualize
            sample_keys = random.sample(list(level_data.keys()), min(len(level_data), num_samples_per_level))
            sample_data = {k: level_data[k] for k in sample_keys}

            bar_labels, parent1_weights, parent2_weights = [], [], []
            for ngram, parents in sample_data.items():
                bar_labels.append(ngram)
                parent1_weights.append(parents.get(ngram[:-1], 0))
                parent2_weights.append(parents.get(ngram[1:], 0))

            x = np.arange(len(bar_labels))
            width = 0.35
            ax.bar(x - width / 2, parent1_weights, width, label=f'Parent 1 ({bar_labels[0][:-1][:4]}...)')
            ax.bar(x + width / 2, parent2_weights, width, label=f'Parent 2 (...{bar_labels[0][1:][-4:]})')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'N-Gram Level: {level} (Sample of {len(bar_labels)} n-grams)')
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = self.plots_output_dir / f"hierarchical_attention_{model_name.replace(' ', '_')}.png"
        try:
            plt.savefig(plot_filename)
            print(f"  Saved hierarchical attention summary plot to {plot_filename}")
        except Exception as e:
            print(f"  Error saving hierarchical attention plot {plot_filename}: {e}")
        plt.close()
        return plot_filename



    def plot_tsne_from_embedding_file(self, h5_path: str, embedding_type: str = 'per_protein') -> Optional[Path]:
        """
        Loads an H5 embedding file and generates a t-SNE visualization plot.
        """
        print(f"\n--- Generating t-SNE plot for {os.path.basename(h5_path)} ---")
        if not os.path.exists(h5_path):
            print(f"  Error: H5 file not found at {h5_path}")
            return None

        try:
            # --- DEFINITIVE FIX for OOM Error: Use EmbeddingLoader to sample keys before loading ---
            # The previous implementation loaded the entire H5 file into memory, which
            # would crash on large embedding files. This new approach samples the keys
            # first and then loads only the required embeddings.
            with EmbeddingLoader(h5_path) as loader:
                all_keys = list(loader.get_keys())
                if not all_keys:
                    print("  Error: No embeddings found in the H5 file.")
                    return None

                num_embeddings = len(all_keys)
                print(f"  Found {num_embeddings} items. Processing as '{embedding_type}'.")

                if num_embeddings > SAMPLE_N_FOR_COMBINED_TSNE:
                    print(f"  Sampling {SAMPLE_N_FOR_COMBINED_TSNE} points from {num_embeddings} for performance.")
                    random.seed(TSNE_RANDOM_STATE)
                    keys_to_load = random.sample(all_keys, SAMPLE_N_FOR_COMBINED_TSNE)
                else:
                    keys_to_load = all_keys

                embeddings_array = np.array([loader[key] for key in keys_to_load])

                base_filename = os.path.splitext(os.path.basename(h5_path))[0]
                title = f"t-SNE of Per-Protein Embeddings\n(Source: {base_filename})"

                num_samples = embeddings_array.shape[0]
                if num_samples <= 1:
                    raise ValueError(f"Not enough samples ({num_samples}) for t-SNE.")

                effective_perplexity = float(min(TSNE_PERPLEXITY, max(1.0, num_samples - 1.0)))
                tsne_init_method = 'pca' if TSNE_INIT_PCA and embeddings_array.shape[1] > 2 else 'random'

                print(f"  Fitting t-SNE (perplexity: {effective_perplexity:.1f}, init: {tsne_init_method})...")
                tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=effective_perplexity,
                            max_iter=TSNE_N_ITER, init=tsne_init_method, learning_rate=TSNE_LEARNING_RATE, n_jobs=-1)

                tsne_results = tsne.fit_transform(embeddings_array)

                df_tsne = pd.DataFrame({'tsne_1': tsne_results[:, 0], 'tsne_2': tsne_results[:, 1]})

                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                sns.scatterplot(x="tsne_1", y="tsne_2", data=df_tsne, legend=False, s=50, alpha=0.7, ax=ax)
                ax.set_title(title, fontsize=16)
                ax.set_xlabel('t-SNE Component 1', fontsize=12)
                ax.set_ylabel('t-SNE Component 2', fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.96])

                plot_filename = self.plots_output_dir / f"tsne_{base_filename}.png"
                plt.savefig(plot_filename)
                plt.close(fig)
                print(f"  Successfully saved t-SNE plot to: {plot_filename}")
                return plot_filename

        except Exception as e:
            print(f"  An error occurred during t-SNE processing: {e}")
            return None