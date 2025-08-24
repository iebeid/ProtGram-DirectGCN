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
import mlflow
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
        # --- DEFINITIVE FIX: Override min sequence length and isolate all paths ---
        self.original_min_len = self.config.PROTGRAM_FASTA_MIN_LEN
        self.original_w2v_dir = self.config.RESULTS_W2V_EMBEDDINGS_DIR
        self.original_id_mapping_path = self.config.ID_MAPPING_PATH

        self.config.PROTGRAM_FASTA_MIN_LEN = 1
        self.config.RESULTS_W2V_EMBEDDINGS_DIR = self.base_test_dir / "word2vec_embeddings"

    def tearDown(self):
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.W2V_EPOCHS = self.original_epochs
        self.config.PROTGRAM_FASTA_MIN_LEN = self.original_min_len
        self.config.RESULTS_W2V_EMBEDDINGS_DIR = self.original_w2v_dir
        self.config.ID_MAPPING_PATH = self.original_id_mapping_path
        shutil.rmtree(self.base_test_dir)

    def test_word2vec_pipeline_run(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("Word2Vec Pipeline Smoke Test")
        print("=" * 80)

        # --- Create dummy data ---
        num_seqs = 5
        input_dir = self.base_test_dir / "input"
        dummy_fasta_path = DummyDataFactory.create_fasta(str(input_dir), "w2v_test.fasta", num_seqs=num_seqs)
        # This prevents the test from trying to load the massive production ID mapping file.
        dummy_mapping_path = DummyDataFactory.create_dummy_id_mapping_parquet(
            str(input_dir), "dummy_id_mapping.parquet", num_ids=num_seqs
        )
        self.config.ID_MAPPING_PATH = Path(dummy_mapping_path)
        self.config.ID_MAPPING_MODE = 'file'

        self.config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        self.config.W2V_EPOCHS = 1

        with mlflow.start_run(run_name="Word2Vec_Embedder_SMOKE_TEST"):
            embedder = Word2VecEmbedder(self.config)
            result_path = embedder.run()

        self.assertIsNotNone(result_path)
        self.assertIsInstance(result_path, str, "The run method should return a string path.")
        self.assertTrue(os.path.exists(result_path))
        print("\n  Word2VecEmbedder smoke test ran successfully.")
        print("--- Word2Vec Pipeline Smoke Test Complete ---")