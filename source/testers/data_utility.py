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
import tempfile
import time


class DataUtilityTests(unittest.TestCase):
    """A class for testing data utilities like loaders and mappers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
        # Isolate paths for this test class
        self.config.BASE_OUTPUT_DIR = Path(self.temp_dir)
        self.config._setup_paths()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_data_utilities(self):
        """Tests the EmbeddingLoader and IDMapGenerator."""
        print("\n" + "=" * 80)
        DataUtils.print_header("Data Utilities Test")
        print("=" * 80)

        # Test EmbeddingLoader
        print("\nTesting EmbeddingLoader:")
        dummy_h5_path = os.path.join(self.temp_dir, "temp_dummy_embeddings.h5")
        with h5py.File(dummy_h5_path, 'w') as hf:
            hf.create_dataset("protein_X", data=np.random.rand(10))
        with EmbeddingLoader(dummy_h5_path) as loader:
            self.assertIn("protein_X", loader)
            embedding = loader["protein_X"]
            print(f"  Successfully loaded dummy embedding for protein_X, shape: {embedding.shape}")

        # Test IDMapGenerator
        print("\nTesting IDMapGenerator (regex mode):")
        dummy_fasta_path = os.path.join(self.temp_dir, "dummy_id_map.fasta")
        self.config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        self.config.ID_MAPPING_PATH = Path(os.path.join(self.temp_dir, "dummy_id_map.tsv"))
        self.config.ID_MAPPING_MODE = 'regex'
        with open(dummy_fasta_path, 'w') as f:
            f.write(">sp|P12345|TEST_HUMAN Test protein\nACGT\n")
            f.write(">tr|A0A0A0|ANOTHER_TEST Another test\nGTCA\n")

        parser_mapper = IDMapGenerator(config=self.config)
        id_map_dictionary = parser_mapper.generate_id_maps()
        print(f"  IDMapGenerator created {len(id_map_dictionary)} mappings.")
        self.assertGreater(len(id_map_dictionary), 0)

        print("--- Data Utilities Test Complete ---")