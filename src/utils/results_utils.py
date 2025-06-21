# ==============================================================================
# MODULE: utils/results_utils.py
# PURPOSE: Contains all functions for plotting results and writing summary
#          files for the PPI evaluation pipeline.
# VERSION: 2.1 (Implemented Hits@k and NDCG@k calculation)
# AUTHOR: Islam Ebeid
# ==============================================================================

from pathlib import Path
from typing import List, Dict, Any, Optional

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr


class EvaluationReporter:
    """
    A class to handle plotting of results and writing summary files
    for the PPI evaluation pipeline.
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

    @staticmethod
    def _calculate_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, k_list: List[int]) -> Dict[str, float]:
        """
        Calculates ranking metrics like Hits@k (as Recall@k) and NDCG@k.

        Args:
            y_true (np.ndarray): The true binary labels.
            y_score (np.ndarray): The predicted probabilities or scores.
            k_list (List[int]): A list of k values to calculate metrics for.

        Returns:
            Dict[str, float]: A dictionary with the calculated metrics.
        """
        # Combine scores and true labels, then sort by score descending
        combined = np.stack([y_score, y_true], axis=1)
        sorted_combined = combined[np.argsort(combined[:, 0])[::-1]]
        sorted_true_labels = sorted_combined[:, 1]

        metrics = {}
        total_positives = np.sum(y_true)

        if total_positives == 0:
            # If there are no positive samples, all ranking metrics are trivially 0
            for k in k_list:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
            return metrics

        # Create the ideal ranking (all positives at the top) for IDCG calculation
        ideal_ranking = np.sort(y_true)[::-1]

        for k in k_list:
            # Ensure k is not larger than the number of items
            actual_k = min(k, len(sorted_true_labels))
            if actual_k == 0:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
                continue

            # --- Hits@k (defined as Recall@k) ---
            # How many of the true positives are in the top k predictions?
            hits_in_top_k = np.sum(sorted_true_labels[:actual_k])
            metrics[f'hits_at_{k}'] = hits_in_top_k / total_positives

            # --- NDCG@k ---
            # Calculate DCG for the actual ranking
            ranks = np.arange(1, actual_k + 1)
            discounts = np.log2(ranks + 1)
            dcg = np.sum(sorted_true_labels[:actual_k] / discounts)

            # Calculate IDCG for the ideal ranking
            idcg = np.sum(ideal_ranking[:actual_k] / discounts)

            metrics[f'ndcg_at_{k}'] = dcg / idcg if idcg > 0 else 0.0

        return metrics

    def plot_training_history(self, history_dict: Dict[str, Any], model_name: str) -> Optional[Path]:
        """
        Plots the training and validation loss/accuracy from a Keras history object.
        """
        if not history_dict:
            print(f"Plotting: No history data for {model_name} to plot.")
            return None

        plot_filename = self.plots_output_dir / f"history_{model_name.replace(' ', '_')}.png"

        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        if 'loss' in history_dict and history_dict['loss']:
            plt.plot(history_dict['loss'], label='Training Loss')
        if 'val_loss' in history_dict and history_dict['val_loss']:
            plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss: {model_name} (Fold 1)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        if 'accuracy' in history_dict and history_dict['accuracy']:
            plt.plot(history_dict['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history_dict and history_dict['val_accuracy']:
            plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy: {model_name} (Fold 1)')
        plt.ylabel('Accuracy')
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
        plt.title('ROC Curves Comparison (from First Fold)')
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
        Generates a set of bar charts comparing key performance metrics across all models.
        Uses self.k_vals_table for Hits@k and NDCG@k metrics.
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
            bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(names))))
            plt.ylabel('Score')
            plt.title(name)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(bottom=0)
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

    def write_summary_file(self, results_list: List[Dict[str, Any]], main_emb_name: str, test_metric: str, alpha: float) -> Optional[Path]:
        """
        Writes a formatted summary table and statistical test results to a text file.
        Uses self.k_vals_table for table headers.
        """
        if not results_list:
            print("Reporting: No results data provided for summary file.")
            return None

        filepath = self.summary_file_output_dir / "evaluation_summary.txt"

        with open(filepath, 'w') as f:
            # Part 1: Performance Table
            f.write("--- Overall Performance Comparison Table (Averaged over Folds) ---\n")
            headers = ["Embedding Name", "AUC", "F1", "Precision", "Recall"]
            for k in self.k_vals_table:
                headers.extend([f"Hits@{k}", f"NDCG@{k}"])
            headers.extend(["AUC StdDev", "F1 StdDev"])

            rows_data = []
            for res in results_list:
                row = [res.get('embedding_name', 'N/A'), f"{res.get('test_auc_sklearn', 0):.4f}", f"{res.get('test_f1_sklearn', 0):.4f}", f"{res.get('test_precision_sklearn', 0):.4f}",
                       f"{res.get('test_recall_sklearn', 0):.4f}"]
                for k_val in self.k_vals_table:
                    row.append(f"{res.get(f'test_hits_at_{k_val}', 0):.4f}") # Changed to .4f for recall@k
                    row.append(f"{res.get(f'test_ndcg_at_{k_val}', 0):.4f}")
                row.append(f"{res.get('test_auc_sklearn_std', 0):.4f}")
                row.append(f"{res.get('test_f1_sklearn_std', 0):.4f}")
                rows_data.append(row)

            df = pd.DataFrame(rows_data, columns=headers)
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Part 2: Statistical Tests
            f.write(f"--- Statistical Comparison vs '{main_emb_name}' on '{test_metric}' (alpha={alpha}) ---\n")
            main_res = next((r for r in results_list if r.get('embedding_name') == main_emb_name), None)
            scores_key = 'fold_auc_scores' if test_metric == 'test_auc_sklearn' else 'fold_f1_scores'  # Simplified

            if main_res and scores_key in main_res and main_res[scores_key]:
                main_scores = [s for s in main_res[scores_key] if not np.isnan(s)]
                f.write(f"{'Compared Embedding':<30} | {'p-value (Wilcoxon)':<20} | {'Significantly Different?':<25} | {'Pearson r':<10}\n")
                f.write("-" * 95 + "\n")

                for other_res in [r for r in results_list if r.get('embedding_name') != main_emb_name]:
                    other_scores = [s for s in other_res.get(scores_key, []) if not np.isnan(s)]
                    if len(main_scores) == len(other_scores) and len(main_scores) > 1:
                        try:
                            # Wilcoxon requires non-identical samples for p-value calculation if differences are all zero
                            if np.allclose(main_scores, other_scores):
                                p_val_wilcoxon = 1.0  # Or handle as a special case
                                conclusion = "Identical scores"
                            else:
                                _, p_val_wilcoxon = wilcoxon(main_scores, other_scores)
                                conclusion = f"Yes (p < {alpha})" if p_val_wilcoxon < alpha else "No"

                            # Pearson correlation
                            p_corr, _ = pearsonr(main_scores, other_scores) if len(np.unique(main_scores)) > 1 and len(np.unique(other_scores)) > 1 else (np.nan, 0)
                            f.write(f"{other_res.get('embedding_name', 'Unknown'):<30} | {p_val_wilcoxon:<20.4e} | {conclusion:<25} | {p_corr:<10.4f}\n")
                        except ValueError as e_stat:  # Catch specific errors from stats functions
                            f.write(f"{other_res.get('embedding_name', 'Unknown'):<30} | N/A (stat error: {e_stat})\n")
                    else:
                        f.write(f"{other_res.get('embedding_name', 'Unknown'):<30} | N/A (score mismatch or too few/invalid folds)\n")
            else:
                f.write(f"Could not perform stats: Baseline model '{main_emb_name}' or its fold scores ('{scores_key}') not found or empty.\n")

        print(f"Results summary saved to {filepath}")
        return filepath