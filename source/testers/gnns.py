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
        self.original_epochs = self.config.BENCHMARK_GNN_EPOCHS
        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config._setup_paths()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = self.original_datasets
        self.config.BENCHMARK_GNN_EPOCHS = self.original_epochs
        self.config._setup_paths()
        shutil.rmtree(self.base_test_dir)

    def test_gnn_benchmarker_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("GNN Benchmarker Smoke Test")
        print("=" * 80)

        self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = ["KarateClub"]
        self.config.BENCHMARK_GNN_EPOCHS = 2
        self.config.BENCHMARK_SAVE_EMBEDDINGS = False

        with mlflow.start_run(run_name="GNN_Benchmark_SMOKE_TEST"):
            benchmarker = GNNBenchmarker(self.config)
            results = benchmarker.run()
            self.assertFalse(results.empty)

        print("\n  GNNBenchmarker smoke test ran successfully.")
        print("--- GNN Benchmarker Smoke Test Complete ---")