import unittest
import os
import shutil
import pickle
from pathlib import Path
import h5py
import numpy as np
from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.id_mapper import IDMapper
from source.utils.post.embedding_loader import EmbeddingLoader
from source.data_builders.protgram import ProtGramDataBuilder
from source.testers.dummy import DummyDataFactory
import tempfile
import time


class DataUtilityTests(unittest.TestCase):
    """A class for testing data utilities like loaders and mappers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        # Store original values to restore them in tearDown
        self.original_base_output_dir = self.config.BASE_OUTPUT_DIR
        self.original_graph_objects_dir = self.config.RESULTS_GRAPH_OBJECTS_DIR
        self.original_downsample = self.config.SEQUENCE_DOWNSAMPLE_FRACTION
        self.original_min_len = self.config.PROTGRAM_FASTA_MIN_LEN
        # --- DEFINITIVE FIX: Isolate the test's "project root" to its temp directory ---
        # This prevents the test from writing cache files to the actual project root
        # and interfering with the main application run.
        self.original_project_root = self.config.PROJECT_ROOT

        # --- DEFINITIVE FIX: Manually override all relevant paths for true isolation ---
        # This avoids calling _setup_paths() and its side effects.
        self.config.BASE_OUTPUT_DIR = Path(self.temp_dir)
        self.config.RESULTS_GRAPH_OBJECTS_DIR = Path(self.temp_dir) / "graph_objects"

        # Now, apply other test-specific overrides
        self.config.SEQUENCE_DOWNSAMPLE_FRACTION = None
        self.config.PROTGRAM_FASTA_MIN_LEN = 1
        self.config.PROJECT_ROOT = Path(self.temp_dir)

    def tearDown(self):
        # Restore original config values
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.RESULTS_GRAPH_OBJECTS_DIR = self.original_graph_objects_dir
        self.config.SEQUENCE_DOWNSAMPLE_FRACTION = self.original_downsample
        self.config.PROTGRAM_FASTA_MIN_LEN = self.original_min_len
        self.config.PROJECT_ROOT = self.original_project_root
        shutil.rmtree(self.temp_dir)

    def test_embedding_loader(self):
        """Tests the basic functionality of the EmbeddingLoader."""
        print("\n" + "=" * 80)
        DataUtils.print_header("Testing: EmbeddingLoader")
        print("=" * 80)

        dummy_h5_path = os.path.join(self.temp_dir, "temp_dummy_embeddings.h5")
        with h5py.File(dummy_h5_path, 'w') as hf:
            hf.create_dataset("protein_X", data=np.random.rand(10))
        with EmbeddingLoader(dummy_h5_path) as loader:
            self.assertIn("protein_X", loader)
            embedding = loader["protein_X"]
            print(f"  Successfully loaded dummy embedding for protein_X, shape: {embedding.shape}")
        print("--- EmbeddingLoader Test Complete ---")

    def test_id_mapper_pregeneration(self):
        """
        Tests the IDMapper's pre-generation of cache files from a parquet source.
        """
        print("\n" + "=" * 80)
        DataUtils.print_header("Testing: IDMapper Pregeneration")
        print("=" * 80)

        # 1. Setup: Create a dummy source parquet file, which is the input for pre-generation.
        input_dir = Path(self.temp_dir) / "input"
        dummy_parquet_path = DummyDataFactory.create_dummy_id_mapping_parquet(
            str(input_dir), "dummy_id_mapping.parquet", num_ids=5
        )

        # 2. Configure the test
        self.config.ID_MAPPING_MODE = 'file'
        self.config.ID_MAPPING_PATH = Path(dummy_parquet_path)

        # 3. Run the pre-generation
        mapper = IDMapper(self.config)
        mapper.pregenerate_caches()

        # 4. Assertions
        expected_cache_file = self.config.PROJECT_ROOT / "file_map_cache.pkl"
        try:
            self.assertTrue(expected_cache_file.exists(), "The ID map pickle cache was not created.")

            with open(expected_cache_file, 'rb') as f:
                loaded_map = pickle.load(f)
            self.assertIsInstance(loaded_map, dict)
            # Based on the dummy data created by the factory
            self.assertEqual(loaded_map.get("DUMMY0001"), "DUMMY0001")
            print("--- IDMapper Pregeneration Test Complete ---")
        finally:
            # Clean up the generated cache file to not interfere with other tests
            if expected_cache_file.exists():
                expected_cache_file.unlink()

    def test_protgram_data_builder_smoke_test(self):
        """Smoke test for the ProtGramDataBuilder to ensure it runs without crashing."""
        print("\n" + "=" * 80)
        DataUtils.print_header("Smoke Test: ProtGramDataBuilder")
        print("=" * 80)
        dummy_fasta_path = DummyDataFactory.create_fasta(self.temp_dir, "graph_builder_test.fasta")
        self.config.SEQUENCE_FILE_PATHS = [dummy_fasta_path]
        self.config.PROTGRAM_NGRAM_MAX_N = 2 # Keep it small for a fast test

        builder = ProtGramDataBuilder(self.config)
        builder.run()

        # Verify that the output directories were created
        graph_n1_dir = self.config.RESULTS_GRAPH_OBJECTS_DIR / "ngram_graph_n1"
        self.assertTrue(graph_n1_dir.exists() and graph_n1_dir.is_dir(), "Graph for n=1 was not created.")
        print(f"  Successfully created graph directory: {graph_n1_dir}")
        print("--- ProtGramDataBuilder Smoke Test Complete ---")