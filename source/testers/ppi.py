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
from source.utils.data.ground_truth_loader import GroundTruthLoader
from source.utils.post.embedding_loader import EmbeddingLoader
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

    def test_ppi_sanity_check_logic(self):
        """
        Tests the sanity check logic on dummy data. This test ensures that
        the pipeline can correctly filter a ground truth dataset against a small
        set of available embeddings and run a minimal evaluation.
        """
        print("\n" + "=" * 80)
        DataUtils.print_header("PPI Sanity Check Logic Smoke Test")
        print("=" * 80)

        # 1. Create a small universe of dummy data
        protein_ids = [f"DUMMY_P{i:04d}" for i in range(100)]
        dummy_emb_file = DummyDataFactory.create_h5_embeddings(
            str(self.base_test_dir), "sanity_check_embeddings.h5", protein_ids=protein_ids[:50], dim=16 # Only 50 have embeddings
        )
        pos_fp, neg_fp = DummyDataFactory.create_interaction_files(
            str(self.base_test_dir), num_pairs=200, num_proteins=len(protein_ids)
        )

        # 2. Mimic the sanity check logic
        with EmbeddingLoader(dummy_emb_file, config=self.config) as loader:
            available_ids = loader.get_keys()

        self.assertEqual(len(available_ids), 50)

        pos_pairs = GroundTruthLoader.load_interaction_pairs_filtered(pos_fp, 1, available_ids)
        neg_pairs = GroundTruthLoader.load_interaction_pairs_filtered(neg_fp, 0, available_ids, sample_n=len(pos_pairs))

        self.assertGreater(len(pos_pairs), 0, "Filtered positive pairs should not be empty.")
        self.assertGreater(len(neg_pairs), 0, "Filtered negative pairs should not be empty.")

        # 3. Run a minimal pipeline on the filtered data
        sanity_config = self.config
        sanity_config.EVAL_EPOCHS = 1
        sanity_config.EVAL_N_FOLDS = 2
        sanity_config.LP_EMBEDDING_FILES_TO_EVALUATE = [{"name": "SanityCheckEmb", "path": str(dummy_emb_file)}]
        sanity_config.POS_INTERACTIONS_PATH = pos_pairs
        sanity_config.NEG_INTERACTIONS_PATH = neg_pairs

        evaluator = PPIPipeline(sanity_config)
        evaluator.run()

        print("\n  PPI Sanity Check logic test ran successfully.")
        print("--- PPI Sanity Check Logic Smoke Test Complete ---")
