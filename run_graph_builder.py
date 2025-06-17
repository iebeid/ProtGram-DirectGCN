# minimal_graph_builder_test.py
from config import Config
from src.pipeline.data_builder import GraphBuilder
from pathlib import Path
import os
import tempfile
import shutil

if __name__ == "__main__":
    base_test_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(base_test_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)
    fasta_content = ">seq1\nACGT\n>seq2\nTTAC\n>seq3\nAGA\n"
    fasta_path = os.path.join(temp_input_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    temp_output_dir = os.path.join(base_test_dir, "output")
    os.makedirs(temp_output_dir, exist_ok=True)

    config = Config()
    config.GCN_INPUT_FASTA_PATH = Path(fasta_path)
    config.BASE_OUTPUT_DIR = Path(temp_output_dir)
    config.GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "1_graph_objects"
    config.GCN_NGRAM_MAX_N = 1
    config.GRAPH_BUILDER_WORKERS = 1  # Crucial for this test
    config.DEBUG_VERBOSE = True  # Keep verbose logging

    print("--- Starting Minimal GraphBuilder Test ---")
    try:
        graph_builder = GraphBuilder(config)
        graph_builder.run()
        print("--- Minimal GraphBuilder Test COMPLETED ---")
    except Exception as e:
        print(f"--- Minimal GraphBuilder Test FAILED: {e} ---")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
