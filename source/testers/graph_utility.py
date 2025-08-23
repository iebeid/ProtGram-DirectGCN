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

class GraphBuilderTests(unittest.TestCase):
    """Tests for the GraphBuilder pipeline."""

    def setUp(self):
        """Set up a temporary directory and a dummy FASTA file."""
        self.base_test_dir = tempfile.mkdtemp()
        self.temp_input_dir = os.path.join(self.base_test_dir, "input")
        os.makedirs(self.temp_input_dir, exist_ok=True)
        self.fasta_path = os.path.join(self.temp_input_dir, "test_sequences.fasta")
        with open(self.fasta_path, "w") as f:
            f.write(">seq1\nACGT\n>seq2\nTTAC\n>seq3\nAGA\n")
        self.temp_output_dir = os.path.join(self.base_test_dir, "output")
        os.makedirs(self.temp_output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        shutil.rmtree(self.base_test_dir)

    def test_graph_builder_smoke(self):
        """Smoke test for GraphBuilder.run() with minimal config."""
        print("\n" + "=" * 80)
        DataUtils.print_header("GraphBuilder Smoke Test (unittest)")
        print("=" * 80)
        config = Config()
        config.BASE_OUTPUT_DIR = Path(self.temp_output_dir)
        config.RESULTS_GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "graph_objects"
        config.SEQUENCE_FILE_PATHS = [Path(self.fasta_path)]
        config.PROTGRAM_NGRAM_MAX_N = 1
        config.GRAPH_BUILDER_WORKERS = 1

        graph_builder = ProtGramDataBuilder(config)
        graph_builder.run()

        expected_graph_dir = config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{config.PROTGRAM_NGRAM_MAX_N}"
        self.assertTrue(expected_graph_dir.is_dir(), f"Expected graph directory not found: {expected_graph_dir}")
        self.assertTrue((expected_graph_dir / "metadata.json").exists(), "Graph metadata.json is missing.")
        print("\n  GraphBuilder smoke test completed successfully.")
        print("--- GraphBuilder Smoke Test Complete ---")

    def test_graph_builder_full_run(self):
        """Runs a full, synchronous test of the GraphBuilder pipeline."""
        script_start_time = time.time()
        print("\n" + "=" * 80)
        DataUtils.print_header("Starting GraphBuilder Full Run Test (Synchronous Dask)")
        print("=" * 80)

        config = Config()
        config.BASE_OUTPUT_DIR = Path(self.temp_output_dir)
        config.RESULTS_GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "graph_objects"
        config.SEQUENCE_FILE_PATHS = [Path(self.fasta_path)]
        config.PROTGRAM_NGRAM_MAX_N = 3
        config.GRAPH_BUILDER_WORKERS = 1

        graph_builder = ProtGramDataBuilder(config)
        graph_builder.run()

        for n in range(1, config.PROTGRAM_NGRAM_MAX_N + 1):
            expected_dir = config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}"
            self.assertTrue(expected_dir.is_dir(), f"Expected graph directory for n={n} not found.")

        print(f"Total time for script: {time.time() - script_start_time:.2f}s")
        print("--- GraphBuilder Full Run Test Complete ---")