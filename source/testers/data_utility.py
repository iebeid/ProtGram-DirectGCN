import unittest
import os
import shutil
from pathlib import Path
import h5py
import numpy as np
from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.id_mapper import IDMapGenerator
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
        self.original_downsample = self.config.SEQUENCE_DOWNSAMPLE_FRACTION
        self.original_min_len = self.config.PROTGRAM_FASTA_MIN_LEN

        # --- DEFINITIVE FIX: Set the temporary output directory BEFORE setting up other paths ---
        # This ensures all output paths derived from BASE_OUTPUT_DIR point to the temp directory.
        self.config.BASE_OUTPUT_DIR = Path(self.temp_dir)
        self.config._setup_paths()

        # Now, apply other test-specific overrides
        self.config.SEQUENCE_DOWNSAMPLE_FRACTION = None
        self.config.PROTGRAM_FASTA_MIN_LEN = 1

    def tearDown(self):
        # Restore original config values
        self.config.BASE_OUTPUT_DIR = self.original_base_output_dir
        self.config.SEQUENCE_DOWNSAMPLE_FRACTION = self.original_downsample
        self.config.PROTGRAM_FASTA_MIN_LEN = self.original_min_len
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

    def test_id_map_generator(self):
        """Tests the IDMapGenerator in regex mode."""
        print("\n" + "=" * 80)
        DataUtils.print_header("Testing: IDMapGenerator")
        print("=" * 80)
        # --- REFACTOR: Use the DummyDataFactory for consistency ---
        dummy_fasta_path = DummyDataFactory.create_fasta(self.temp_dir, "id_map_test.fasta")
        self.config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        self.config.ID_MAPPING_PATH = Path(os.path.join(self.temp_dir, "dummy_id_map.tsv"))
        self.config.ID_MAPPING_MODE = 'regex'

        parser_mapper = IDMapGenerator(config=self.config)
        id_map_dictionary = parser_mapper.generate_id_maps()
        print(f"  IDMapGenerator created {len(id_map_dictionary)} mappings.")
        self.assertGreater(len(id_map_dictionary), 0)
        print("--- IDMapGenerator Test Complete ---")

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