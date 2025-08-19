# ==============================================================================
# MODULE: testers/reporting.py
# PURPOSE: Contains tests for the EvaluationReporter class.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import os
import shutil
import numpy as np
from source.utils.data.data_utils import DataUtils
from source.utils.results.evaluation_reporter import EvaluationReporter


class ReportingTests(unittest.TestCase):
    """A class for testing the EvaluationReporter."""

    def setUp(self):
        self.test_output_dir = "./temp_test_evaluation_reporter_output"
        if os.path.exists(self.test_output_dir): shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_output_dir): shutil.rmtree(self.test_output_dir)

    def test_reporter(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("EvaluationReporter Test")
        print("=" * 80)
        sample_k_vals = [10, 20]

        reporter = EvaluationReporter(base_output_dir=self.test_output_dir, k_vals_table=sample_k_vals)

        history1 = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.55, 0.42, 0.33], 'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.68, 0.78, 0.88]}
        reporter.plot_training_history(history1, "Model_A_Fold1")

        results_data = [
            {'embedding_name': 'Model_A', 'test_auc_sklearn': 0.92, 'test_f1_sklearn': 0.85, 'test_precision_sklearn': 0.88, 'test_recall_sklearn': 0.82, 'test_hits_at_10': 0.5, 'test_ndcg_at_10': 0.75,
             'test_hits_at_20': 0.8, 'test_ndcg_at_20': 0.78, 'test_auc_sklearn_std': 0.01, 'test_f1_sklearn_std': 0.02, 'roc_data_representative': (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]), 0.92),
             'fold_auc_scores': [0.91, 0.93], 'fold_f1_scores': [0.84, 0.86]},
            {'embedding_name': 'Model_B', 'test_auc_sklearn': 0.88, 'test_f1_sklearn': 0.80, 'test_precision_sklearn': 0.82, 'test_recall_sklearn': 0.78, 'test_hits_at_10': 0.4, 'test_ndcg_at_10': 0.65,
             'test_hits_at_20': 0.7, 'test_ndcg_at_20': 0.68, 'test_auc_sklearn_std': 0.015, 'test_f1_sklearn_std': 0.022, 'roc_data_representative': (np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), 0.88),
             'fold_auc_scores': [0.87, 0.89], 'fold_f1_scores': [0.79, 0.81]}
        ]
        reporter.plot_roc_curves(results_data)
        reporter.plot_comparison_charts(results_data)
        reporter.write_summary_file(results_data, main_emb_name='Model_A', test_metric='test_auc_sklearn', alpha=0.05)

        print(f"  Example reporting complete. Check '{self.test_output_dir}' directory.")
        print("--- EvaluationReporter Test Complete ---")