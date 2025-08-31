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
from source.testers.dummy import DummyDataFactory


class PPIPipelineTests(unittest.TestCase):
    """A class for smoke testing the PPI pipeline."""

    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.original_epochs = self.config.EVAL_EPOCHS
        self.original_folds = self.config.EVAL_N_FOLDS
        # --- NEW: Isolate project root and cache for true test isolation ---
        self.original_project_root = self.config.PROJECT_ROOT
        self.original_persistent_cache = self.config.PERSISTENT_DATA_CACHE

        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config.PROJECT_ROOT = self.base_test_dir
        self.config.PERSISTENT_DATA_CACHE = self.base_test_dir / ".cache"
        # --- DEFINITIVE FIX: Fully re-initialize all path-dependent configs ---
        self.config._setup_paths()
        self.config._setup_data_sources()
        self.config._link_data_sources_to_attributes()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.EVAL_EPOCHS = self.original_epochs
        self.config.EVAL_N_FOLDS = self.original_folds
        self.config.PROJECT_ROOT = self.original_project_root
        self.config.PERSISTENT_DATA_CACHE = self.original_persistent_cache
        self.config._setup_paths()
        self.config._setup_data_sources()
        self.config._link_data_sources_to_attributes()
        shutil.rmtree(self.base_test_dir)

    def test_ppi_pipeline_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("PPI Pipeline (Dummy Run) Smoke Test")
        print("=" * 80)

        self.config.EVAL_EPOCHS = 1
        self.config.EVAL_N_FOLDS = 2

        protein_ids = [f"DUMMY_P{i:04d}" for i in range(50)]
        dummy_emb_file = DummyDataFactory.create_h5_embeddings(
            str(self.base_test_dir), "dummy_embeddings.h5", protein_ids=protein_ids, dim=16
        )
        pos_fp, neg_fp = DummyDataFactory.create_interaction_files(
            str(self.base_test_dir), num_pairs=100, num_proteins=len(protein_ids)
        )
        emb_configs = [{"path": str(dummy_emb_file), "name": "DummyEmb"}]

        self.config.LP_EMBEDDING_FILES_TO_EVALUATE = emb_configs
        self.config.POS_INTERACTIONS_PATH = pos_fp
        self.config.NEG_INTERACTIONS_PATH = neg_fp

        with mlflow.start_run(run_name="PPI_Pipeline_SMOKE_TEST"):
            evaluator = PPIPipeline(self.config)
            evaluator.run()

        print("\n  PPIPipeline (dummy run) smoke test ran successfully.")
        print("--- PPI Pipeline (Dummy Run) Smoke Test Complete ---")
