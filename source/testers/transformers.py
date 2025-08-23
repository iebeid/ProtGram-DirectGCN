# ==============================================================================
# MODULE: testers/transformers.py
# PURPOSE: Contains smoke tests for the Transformer embedder pipeline.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import shutil
import tempfile
from pathlib import Path
import mlflow
from configuration.config import Config
from source.trainers.transformers import TransformerEmbedder
from source.utils.data.data_utils import DataUtils
from source.utils.models.model_converter import ModelConverter
from .dummy import DummyDataFactory


class TransformerPipelineTests(unittest.TestCase):
    """A class for smoke testing the Transformer embedder pipeline."""

    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config._setup_paths()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config._setup_paths()
        shutil.rmtree(self.base_test_dir)

    def test_transformer_embedder_pipeline_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("Transformer Embedder Pipeline Smoke Test")
        print("=" * 80)

        dummy_fasta_path = DummyDataFactory.create_fasta(str(self.base_test_dir / "input"), "transformer_test.fasta", num_seqs=2)

        temp_model_dir = self.base_test_dir / "models"
        self.config.DATA_MODELS_DIR = temp_model_dir
        for model_cfg in self.config.TRANSFORMER_MODELS_TO_RUN:
            ModelConverter.convert_and_save_model(model_cfg['hf_id'], temp_model_dir)

        self.config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        self.config.TRANSFORMER_BASE_BATCH_SIZE = 1

        with mlflow.start_run(run_name="Transformer_Embedder_SMOKE_TEST"):
            embedder = TransformerEmbedder(self.config)
            generated_paths = embedder.run()
            # --- REFACTOR: Make assertions dynamic and more robust ---
            self.assertIsInstance(generated_paths, dict, "The run method should return a dictionary.")
            self.assertGreater(len(generated_paths), 0, "The run method returned an empty dictionary.")
            for model_name, output_path in generated_paths.items():
                print(f"  - Verifying output for {model_name}...")
                self.assertTrue(output_path.exists(), f"Output file for {model_name} was not created at {output_path}")

        print("\n  TransformerEmbedder smoke test ran successfully.")
        print("--- Transformer Embedder Pipeline Smoke Test Complete ---")