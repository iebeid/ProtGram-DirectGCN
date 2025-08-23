
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

# --- Configuration for t-SNE plotting ---
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_INIT_PCA = True
TSNE_LEARNING_RATE = 'auto'
SAMPLE_N_FOR_COMBINED_TSNE = 2000

class EvaluationSummary:
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
        """
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length.")

        combined = np.stack([y_score, y_true], axis=1)
        sorted_combined = combined[np.argsort(combined[:, 0])[::-1]]
        sorted_true_labels = sorted_combined[:, 1]

        metrics = {}
        total_positives = np.sum(y_true)

        if total_positives == 0:
            for k in k_list:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
            return metrics

        ideal_ranking = np.sort(y_true)[::-1]

        for k in k_list:
            actual_k = min(k, len(sorted_true_labels))
            if actual_k == 0:
                metrics[f'hits_at_{k}'] = 0.0
                metrics[f'ndcg_at_{k}'] = 0.0
                continue

            hits_in_top_k = np.sum(sorted_true_labels[:actual_k])
            metrics[f'hits_at_{k}'] = hits_in_top_k / total_positives

            ranks = np.arange(1, actual_k + 1)
            discounts = np.log2(ranks + 1)
            dcg = np.sum(sorted_true_labels[:actual_k] / discounts)
            idcg = np.sum(ideal_ranking[:actual_k] / discounts)

            metrics[f'ndcg_at_{k}'] = dcg / idcg if idcg > 0 else 0.0

        return metrics

    def _create_performance_dataframe(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Helper to create the main performance summary DataFrame."""
        headers = ["Embedding Name", "AUC", "F1", "Precision", "Recall"]
        for k in self.k_vals_table:
            headers.extend([f"Hits@{k}", f"NDCG@{k}"])
        headers.extend(["AUC StdDev", "F1 StdDev"])

        rows_data = []
        for res in results_list:
            row = [res.get('embedding_name', 'N/A'), f"{res.get('test_auc_sklearn', 0):.4f}",
                   f"{res.get('test_f1_sklearn', 0):.4f}", f"{res.get('test_precision_sklearn', 0):.4f}",
                   f"{res.get('test_recall_sklearn', 0):.4f}"]
            for k_val in self.k_vals_table:
                row.append(f"{res.get(f'test_hits_at_{k_val}', 0):.4f}")
                row.append(f"{res.get(f'test_ndcg_at_{k_val}', 0):.4f}")
            row.append(f"{res.get('test_auc_sklearn_std', 0):.4f}")
            row.append(f"{res.get('test_f1_sklearn_std', 0):.4f}")
            rows_data.append(row)

        return pd.DataFrame(rows_data, columns=headers)

    def _write_statistical_comparison(self, f, results_list: List[Dict[str, Any]], main_emb_name: str, test_metric: str,
                                      alpha: float):
        """Helper to write the statistical comparison section to the file."""
        f.write(f"--- Statistical Comparison vs '{main_emb_name}' on '{test_metric}' (alpha={alpha}) ---\n")
        main_res = next((r for r in results_list if r.get('embedding_name') == main_emb_name), None)
        scores_key = 'fold_auc_scores' if 'auc' in test_metric else 'fold_f1_scores'

        if main_res and scores_key in main_res:
            main_scores = [s for s in main_res[scores_key] if not np.isnan(s)]
            f.write(
                f"{'Compared Embedding':<30} | {'p-value (Wilcoxon)':<20} | {'Significantly Different?':<25} | {'Pearson r':<10}\n")
            f.write("-" * 95 + "\n")

            for other_res in [r for r in results_list if r.get('embedding_name') != main_emb_name]:
                other_scores = [s for s in other_res.get(scores_key, []) if not np.isnan(s)]
                if len(main_scores) > 1 and len(other_scores) == len(main_scores):
                    try:
                        if np.allclose(main_scores, other_scores):
                            p_val_wilcoxon = 1.0
                            conclusion = "Identical scores"
                        else:
                            _, p_val_wilcoxon = wilcoxon(main_scores, other_scores)
                            conclusion = f"Yes (p < {alpha:.2f})" if p_val_wilcoxon < alpha else "No"

                        p_corr, _ = pearsonr(main_scores, other_scores) if len(
                            np.unique(main_scores)) > 1 and len(np.unique(other_scores)) > 1 else (np.nan, 0)
                        f.write(
                            f"{other_res.get('embedding_name', 'Unknown'):<30} | {p_val_wilcoxon:<20.4e} | {conclusion:<25} | {p_corr:<10.4f}\n")
                    except ValueError as e_stat:
                        f.write(f"{other_res.get('embedding_name', 'Unknown'):<30} | N/A (stat error: {e_stat})\n")
                else:
                    f.write(
                        f"{other_res.get('embedding_name', 'Unknown'):<30} | N/A (score mismatch or too few/invalid folds)\n")
        else:
            f.write(
                f"Could not perform stats: Baseline model '{main_emb_name}' or its fold scores ('{scores_key}') not found or empty.\n")

    def write_summary_file(self, results_list: List[Dict[str, Any]], main_emb_name: str, test_metric: str, alpha: float) -> Optional[Path]:
        """
        Writes a formatted summary table and statistical test results to a text file.
        """
        if not results_list:
            print("Reporting: No results data provided for summary file.")
            return None

        # --- REFACTOR: Save as a more useful CSV file instead of TXT ---
        filepath = self.summary_file_output_dir / "evaluation_summary.csv"
        performance_df = self._create_performance_dataframe(results_list)

        # Save the main performance table as a CSV
        performance_df.to_csv(filepath, index=False)

        with open(filepath, 'w') as f:
            f.write(performance_df.to_string(index=False))
            f.write("\n\n")
            self._write_statistical_comparison(f, results_list, main_emb_name, test_metric, alpha)

        print(f"Results summary saved to {filepath}")
        return filepath