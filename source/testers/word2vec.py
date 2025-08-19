# ==============================================================================
# MODULE: testers/word2vec.py
# PURPOSE: Contains smoke tests for the Word2Vec pipeline.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import os
import shutil
import tempfile
from pathlib import Path
from configuration.config import Config
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.data.data_utils import DataUtils
from .dummy import DummyDataFactory


class Word2VecPipelineTests(unittest.TestCase):
    """A class for smoke testing the Word2Vec pipeline."""

    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.original_epochs = self.config.W2V_EPOCHS
        self.config.BASE_OUTPUT_DIR = self.base_test_dir
        self.config._setup_paths()

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.W2V_EPOCHS = self.original_epochs
        self.config._setup_paths()
        shutil.rmtree(self.base_test_dir)

    def test_word2vec_pipeline_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("Word2Vec Pipeline Smoke Test")
        print("=" * 80)

        dummy_fasta_path = DummyDataFactory.create_fasta(str(self.base_test_dir / "input"), "w2v_test.fasta")
        self.config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        self.config.W2V_EPOCHS = 1

        embedder = Word2VecEmbedder(self.config)
        result_path = embedder.run()

        self.assertIsNotNone(result_path)
        self.assertTrue(os.path.exists(result_path))
        print("\n  Word2VecEmbedder smoke test ran successfully.")
        print("--- Word2Vec Pipeline Smoke Test Complete ---")