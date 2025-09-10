# ==============================================================================
# MODULE: utils/results/evaluation_reporter.py
# PURPOSE: A consolidated class for all evaluation reporting, including plots,
#          summary files, statistical tests, and interpretability visualizations.
# VERSION: 4.0 (Merged summary generation, improved stats, added heatmaps)
# AUTHOR: Islam Ebeid
# ==============================================================================

import json
import math
import random
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.metrics import auc

# Conditionally import SHAP to avoid making it a hard dependency
try:
    import shap
except ImportError:
    shap = None

from source.utils.post.embedding_loader import EmbeddingLoader
from sklearn.manifold import TSNE

# --- Configuration for t-SNE plotting ---
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_INIT_PCA = True
TSNE_LEARNING_RATE = 'auto'
SAMPLE_N_FOR_COMBINED_TSNE = 2000


class EvaluationReporter:
    """
    A consolidated class to handle all aspects of reporting for the PPI evaluation pipeline.
    This includes plotting results, writing summary files with statistical tests,
    and generating interpretability visualizations like SHAP plots and attention heatmaps.
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

    # --- Summary File Generation ---

    def write_summary_file(self, results_list: List[Dict[str, Any]], main_model_name: str, test_metric: str, alpha: float) -> Optional[Path]:
        """
        Writes a formatted summary table and statistical test results to a text file.
        Also saves a clean, machine-readable CSV version of the performance metrics.
        """
        if not results_list:
            print("Reporting: No results data provided for summary file.")
            return None

        csv_filepath = self.summary_file_output_dir / "evaluation_summary.csv"
        txt_filepath = self.summary_file_output_dir / "evaluation_summary.txt"
        performance_df = self._create_performance_dataframe(results_list)

        try:
            performance_df.to_csv(csv_filepath, index=False, float_format='%.4f')
            print(f"Machine-readable results summary saved to {csv_filepath}")

            formatted_df = performance_df.copy()
            for col in formatted_df.select_dtypes(include=['float']).columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f'{x:.4f}')

            with open(txt_filepath, 'w') as f:
                f.write("--- Performance Summary ---\n")
                f.write(formatted_df.to_string(index=False))
                f.write("\n\n")
                self._write_statistical_comparison(f, results_list, main_model_name, test_metric, alpha)
            print(f"Human-readable report with stats saved to {txt_filepath}")
        except IOError as e:
            print(f"  ERROR: Could not write summary files: {e}")
            return None

        return txt_filepath

    @staticmethod
    def _calculate_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, k_list: List[int]) -> Dict[str, float]:
        """
        Calculates ranking metrics:
          - hits_at_k: Recall@K (positives in top-K / total positives)
          - precision_at_k: Precision@K (positives in top-K / K)
          - hits_count_at_k: raw count of positives in top-K
          - ndcg_at_k: DCG@K normalized by IDCG@K (binary relevance)
        Notes:
          - Computations are done in float64 for numerical stability.
        """
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length.")

        # Ensure stable dtypes
        y_true = np.asarray(y_true).astype(np.float64)
        y_score = np.asarray(y_score).astype(np.float64)

        # Sort by predicted scores descending
        order = np.argsort(y_score)[::-1]
        sorted_true_labels = y_true[order]

        metrics: Dict[str, float] = {}
        total_positives = float(np.sum(y_true))

        # If no positives, all metrics are zero
        if total_positives == 0.0:
            for k in k_list:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'precision_at_{k}'] = 0.0
                metrics[f'hits_count_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
            return metrics

        # Ideal ordering by true labels for IDCG computation (binary relevance)
        ideal_ranking = np.sort(y_true)[::-1]

        for k in k_list:
            actual_k = int(min(k, len(sorted_true_labels)))
            if actual_k <= 0:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'precision_at_{k}'] = 0.0
                metrics[f'hits_count_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
                continue

            topk_labels = sorted_true_labels[:actual_k]
            hits_in_top_k = float(np.sum(topk_labels))

            # Recall@K (keep legacy name hits_at_k)
            metrics[f'hits_at_{k}'] = hits_in_top_k / total_positives
            # Precision@K and raw count
            metrics[f'precision_at_{k}'] = hits_in_top_k / float(actual_k)
            metrics[f'hits_count_at_{k}'] = hits_in_top_k

            # DCG/IDCG with binary gains; use (2^rel - 1) / log2(1+rank)
            ranks = np.arange(1, actual_k + 1, dtype=np.float64)
            discounts = np.log2(ranks + 1.0)

            gains = (2.0 ** topk_labels - 1.0)
            dcg = float(np.sum(gains / discounts))

            ideal_topk = ideal_ranking[:actual_k]
            ideal_gains = (2.0 ** ideal_topk - 1.0)
            idcg = float(np.sum(ideal_gains / discounts))

            metrics[f'ndcg_at_{k}'] = (dcg / idcg) if idcg > 0.0 else 0.0

        return metrics

    def _create_performance_dataframe(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Helper to create the main performance summary DataFrame with raw numeric data."""
        rows_data = []
        for res in results_list:
            row = {
                "Embedding Name": res.get('embedding_name', 'N/A'),
                "AUC": res.get('test_auc_sklearn', 0.0),
                "F1": res.get('test_f1_sklearn', 0.0),
                "Precision": res.get('test_precision_sklearn', 0.0),
                "Recall": res.get('test_recall_sklearn', 0.0),
                "AUC StdDev": res.get('test_auc_sklearn_std', 0.0),
                "F1 StdDev": res.get('test_f1_sklearn_std', 0.0)
            }
            for k_val in self.k_vals_table:
                # Existing: recall@K named Hits@K for backward compatibility
                row[f"Hits@{k_val}"] = res.get(f'test_hits_at_{k_val}', 0.0)
                row[f"NDCG@{k_val}"] = res.get(f'test_ndcg_at_{k_val}', 0.0)
                # New (optional): Precision@K and Hit Count@K if available
                if f'test_precision_at_{k_val}' in res:
                    row[f"Precision@{k_val}"] = res.get(f'test_precision_at_{k_val}', 0.0)
                if f'test_hits_count_at_{k_val}' in res:
                    row[f"HitsCount@{k_val}"] = res.get(f'test_hits_count_at_{k_val}', 0.0)
            rows_data.append(row)

        return pd.DataFrame(rows_data)

    def _write_statistical_comparison(self, f, results_list: List[Dict[str, Any]], main_model_name: str, metric: str, alpha: float):
        """Helper to write the statistical comparison section to the file using a paired t-test with Bonferroni correction."""
        main_model_results = next((res for res in results_list if res.get('embedding_name') == main_model_name), None)
        scores_key = 'fold_auc_scores' if 'auc' in metric else 'fold_f1_scores'
        main_model_scores = main_model_results.get(scores_key) if main_model_results else None

        if main_model_scores is None:
            f.write(f"\n--- Statistical Comparison Skipped: Baseline model '{main_model_name}' or its fold scores not found. ---\n")
            return

        other_models = [res for res in results_list if res['embedding_name'] != main_model_name]
        num_comparisons = len(other_models)
        if num_comparisons == 0:
            f.write("\n--- Statistical Comparison Skipped: No other models to compare against. ---\n")
            return

        corrected_alpha = alpha / num_comparisons

        f.write(f"\n--- Statistical Significance Tests (Paired t-test) ---\n")
        f.write(f"Comparing all models against '{main_model_name}' using metric '{metric}' (alpha={alpha}, Bonferroni corrected alpha={corrected_alpha:.4f})\n")
        f.write("-" * 50 + "\n")

        for other_model_results in other_models:
            other_model_scores = other_model_results.get(scores_key)
            f.write(f"\n  Comparison: '{main_model_name}' vs '{other_model_results['embedding_name']}'\n")

            if other_model_scores is None or len(main_model_scores) != len(other_model_scores):
                f.write("    - ❌ Result: Cannot perform test. Score lists have different lengths or are missing.\n")
                continue

            try:
                t_stat, p_val = ttest_rel(main_model_scores, other_model_scores, nan_policy='omit')
                f.write(f"    - P-value: {p_val:.4f}\n")
                mean_main, mean_other = np.nanmean(main_model_scores), np.nanmean(other_model_scores)
                if p_val < corrected_alpha:
                    if mean_main > mean_other:
                        f.write(f"    - ✅ Result: Statistically significant improvement. ({mean_main:.4f} > {mean_other:.4f})\n")
                    elif mean_other > mean_main:
                        f.write(f"    - ❌ Result: Statistically significant decline. ({mean_main:.4f} < {mean_other:.4f})\n")
                    else:
                        f.write("    - ➖ Result: Statistically significant, but means are equal.\n")
                else:
                    f.write(f"    - ➖ Result: No statistically significant difference. ({mean_main:.4f} vs {mean_other:.4f})\n")
            except Exception as e:
                f.write(f"    - ❌ Result: Statistical test failed with error: {e}\n")

    # --- Plotting Functions ---

    def plot_training_history(self, history_dict: Dict[str, Any], model_name: str) -> Optional[Path]:
        """Plots the training and validation loss/accuracy from a Keras history object."""
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
        """Plots a comparison of ROC curves from multiple model evaluation results."""
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
        """Generates a set of bar charts comparing key performance metrics across all models."""
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
            std_dev_key = f"{key}_std"
            errors = [res.get(std_dev_key, 0) for res in results_list]

            bars = plt.bar(names, values, yerr=errors, capsize=5, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(names))), alpha=0.8)
            plt.ylabel('Score')
            plt.title(name)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(bottom=0, top=max(1.0, max(values) * 1.1 if values else 1.0))
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

    # --- Interpretability Plots ---

    def generate_shap_summary(self, model, background_data: np.ndarray, model_name: str, fold_num: int) -> Optional[Path]:
        """Generates and saves a SHAP summary plot to explain model predictions."""
        if shap is None:
            print("  SHAP library not installed. Skipping SHAP summary generation.")
            return None

        plot_filename = self.plots_output_dir / f"shap_summary_{model_name.replace(' ', '_')}_Fold{fold_num}.png"
        print(f"  Generating SHAP summary plot for {model_name}...")

        try:
            explainer = shap.DeepExplainer(model, background_data)
            shap_values = explainer.shap_values(background_data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            plt.figure()
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

    def plot_attention_heatmap(self, attention_data: Dict[str, Dict[str, float]], model_name: str, num_top_proteins: int = 20, num_top_ngrams: int = 25) -> Optional[Path]:
        """
        Generates and saves a heatmap of n-gram attention weights for the proteins
        with the highest overall attention variance.
        """
        print(f"  Generating attention heatmap for '{model_name}'...")
        try:
            df = pd.DataFrame.from_dict(attention_data, orient='index').fillna(0)
            if df.empty:
                print("    - WARNING: Attention data is empty. Cannot generate heatmap.")
                return None

            top_proteins = df.var(axis=1).nlargest(num_top_proteins).index
            top_ngrams = df.loc[top_proteins].mean(axis=0).nlargest(num_top_ngrams).index
            heatmap_data = df.loc[top_proteins, top_ngrams]

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(18, 12))
            sns.heatmap(heatmap_data, ax=ax, cmap="viridis", annot=False)
            ax.set_title(f'N-Gram Attention Heatmap for Top {num_top_proteins} Proteins ({model_name})', fontsize=16)
            ax.set_xlabel('N-Grams', fontsize=12)
            ax.set_ylabel('Protein IDs', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            output_path = self.plots_output_dir / f"attention_heatmap_{model_name}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    - Attention heatmap saved to: {output_path.name}")
            return output_path
        except Exception as e:
            print(f"    - ❌ ERROR: Could not generate attention heatmap: {e}")
            return None

    def plot_tsne_from_embedding_file(self, h5_path: str, embedding_type: str = 'per_protein') -> Optional[Path]:
        """Loads an H5 embedding file and generates a t-SNE visualization plot."""
        print(f"\n--- Generating t-SNE plot for {os.path.basename(h5_path)} ---")
        if not os.path.exists(h5_path):
            print(f"  Error: H5 file not found at {h5_path}")
            return None

        try:
            with EmbeddingLoader(h5_path) as loader:
                all_keys = list(loader.get_keys())
                if not all_keys:
                    print("  Error: No embeddings found in the H5 file.")
                    return None

                num_embeddings = len(all_keys)
                print(f"  Found {num_embeddings} items. Processing as '{embedding_type}'.")

                if num_embeddings > SAMPLE_N_FOR_COMBINED_TSNE:
                    print(f"  Sampling {SAMPLE_N_FOR_COMBINED_TSNE} points from {num_embeddings} for performance.")
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