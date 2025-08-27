# ==============================================================================
# MODULE: testers/gnns.py
# PURPOSE: Contains smoke tests for the GNN benchmarker.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import mlflow
from configuration.config import Config
from source.benchmarkers.gnns import GNNBenchmarker
from source.utils.data.data_utils import DataUtils


class GNNBenchmarkerTests(unittest.TestCase):
    """A class for smoke testing the GNN benchmarker."""

    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.original_datasets = self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS
        self.original_epochs = self.config.BENCHMARK_GNN_EPOCHS # --- DEFINITIVE FIX: Isolate the persistent cache and project root for this test ---
        # The previous implementation was using the real user cache, which could
        # cause side effects and non-reproducible test runs.
        self.original_persistent_cache = self.config.PERSISTENT_DATA_CACHE
        self.original_project_root = self.config.PROJECT_ROOT

        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config.PERSISTENT_DATA_CACHE = self.base_test_dir / ".cache"
        self.config.PROJECT_ROOT = self.base_test_dir
        self.config._setup_paths()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = self.original_datasets
        self.config.BENCHMARK_GNN_EPOCHS = self.original_epochs
        self.config.PERSISTENT_DATA_CACHE = self.original_persistent_cache
        self.config.PROJECT_ROOT = self.original_project_root
        self.config._setup_paths()
        shutil.rmtree(self.base_test_dir)

    def test_gnn_benchmarker_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("GNN Benchmarker Smoke Test")
        print("=" * 80)

        self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = ["KarateClub"]
        self.config.BENCHMARK_GNN_MODELS_TO_RUN = ["GCN", "GAT"]
        self.config.BENCHMARK_GNN_EPOCHS = 2
        self.config.BENCHMARK_SAVE_EMBEDDINGS = False

        with mlflow.start_run(run_name="GNN_Benchmark_SMOKE_TEST"):
            benchmarker = GNNBenchmarker(self.config)
            results = benchmarker.run()
            self.assertIsInstance(results, pd.DataFrame, "Benchmarker did not return a pandas DataFrame.")
            self.assertFalse(results.empty, "Benchmarker returned an empty DataFrame.")
            # --- DEFINITIVE FIX: Make assertion aware of the number of graph variants ---
            # The test runs on both original and undirected graphs if configured.
            num_variants = 2 if self.config.BENCHMARK_TEST_ON_UNDIRECTED else 1
            expected_len = len(self.config.BENCHMARK_GNN_MODELS_TO_RUN) * num_variants
            self.assertEqual(len(results), expected_len, f"Expected {expected_len} results, but got {len(results)}.")
            self.assertIn("model", results.columns, "Results DataFrame is missing 'model' column.")
            self.assertIn("Accuracy", results.columns, "Results DataFrame is missing 'Accuracy' column.")

        print("\n  GNNBenchmarker smoke test ran successfully.")
        print(results[['model', 'dataset', 'Accuracy']].to_string(index=False))
        print("--- GNN Benchmarker Smoke Test Complete ---")