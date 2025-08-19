# ==============================================================================
# MODULE: testers/ppi.py
# PURPOSE: Contains smoke tests for the PPI pipeline.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import shutil
import tempfile
from pathlib import Path
import mlflow
from configuration.config import Config
from source.experiments.ppi_1 import PPIPipeline
from source.utils.data.data_utils import DataUtils


class PPIPipelineTests(unittest.TestCase):
    """A class for smoke testing the PPI pipeline."""

    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.original_epochs = self.config.EVAL_EPOCHS
        self.original_folds = self.config.EVAL_N_FOLDS
        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config._setup_paths()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.EVAL_EPOCHS = self.original_epochs
        self.config.EVAL_N_FOLDS = self.original_folds
        self.config._setup_paths()
        shutil.rmtree(self.base_test_dir)

    def test_ppi_pipeline_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("PPI Pipeline (Dummy Run) Smoke Test")
        print("=" * 80)

        self.config.EVAL_EPOCHS = 1
        self.config.EVAL_N_FOLDS = 2

        with mlflow.start_run(run_name="PPI_Pipeline_SMOKE_TEST") as parent_run:
            evaluator = PPIPipeline(self.config)
            evaluator.run(use_dummy_data=True, parent_run_id=parent_run.info.run_id)

        print("\n  PPIPipeline (dummy run) smoke test ran successfully.")
        print("--- PPI Pipeline (Dummy Run) Smoke Test Complete ---")