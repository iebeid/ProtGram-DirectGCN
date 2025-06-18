# run_graph_builder.py
import time
import os
import shutil
import tempfile
from pathlib import Path
import dask  # Import dask for config setting
# pandas, pyarrow, networkx, community are used by GraphBuilder internally

from config import Config
from src.pipeline.data_builder import GraphBuilder
from src.utils.data_utils import DataUtils
# DirectedNgramGraph is used internally by GraphBuilder

if __name__ == "__main__":
    script_start_time = time.time()
    DataUtils.print_header("Starting GraphBuilder Full Run Test (Synchronous Dask)")

    base_test_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(base_test_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)

    # Using a slightly more complex FASTA to ensure some edges are likely
    fasta_content = (
        ">seq1\nACGTACT\n"  # n=1: A,C,G,T ; n=2: AC,CG,GT,TA,AC,CT ; n=3: ACG, CGT, GTA, TAC, ACT
        ">seq2\nTTACGTT\n" # n=1: T,A,C,G ; n=2: TT,TA,AC,CG,GT,TT ; n=3: TTA,TAC,ACG,CGT,GTT
        ">seq3\nAGATAGA\n" # n=1: A,G,T   ; n=2: AG,GA,AT,TA,AG,GA ; n=3: AGA,GAT,ATA,TAG,AGA
    )
    fasta_path = os.path.join(temp_input_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    # GraphBuilder will create its own temp_dir based on config.BASE_OUTPUT_DIR
    # and its own final graph objects dir based on config.GRAPH_OBJECTS_DIR

    config = Config()
    config.GCN_INPUT_FASTA_PATH = Path(fasta_path)
    # Set BASE_OUTPUT_DIR for the test to avoid polluting main results
    config.BASE_OUTPUT_DIR = Path(base_test_dir) / "test_pipeline_output"
    config.GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "1_graph_objects" # GraphBuilder uses this
    config.DEBUG_VERBOSE = True  # Ensure debug prints are active
    config.GCN_NGRAM_MAX_N = 3   # Test for n=1, 2, 3 to see edge variations
    config.GRAPH_BUILDER_WORKERS = 1 # Force synchronous for this test

    print(f"--- Running GraphBuilder instance for n_max={config.GCN_NGRAM_MAX_N} ---")
    print(f"  Input FASTA: {config.GCN_INPUT_FASTA_PATH}")
    print(f"  GraphBuilder output will be within: {config.BASE_OUTPUT_DIR}")

    # Dask config is handled within GraphBuilder's synchronous fallback logic
    # original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Also handled within GraphBuilder
    # print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for the main process.")

    try:
        graph_builder_instance = GraphBuilder(config)
        graph_builder_instance.run() # This will execute Phase 1, 2 (with stats), and 3
        print(f"--- GraphBuilder run() method completed ---")

        # Check if graph files were created (optional, as GraphBuilder prints this)
        for n_val_check in range(1, config.GCN_NGRAM_MAX_N + 1):
            expected_graph_file = config.GRAPH_OBJECTS_DIR / f"ngram_graph_n{n_val_check}.pkl"
            if expected_graph_file.exists():
                print(f"  OK: Final graph object file found: {expected_graph_file}")
            else:
                print(f"  WARN: Final graph object file NOT found: {expected_graph_file}")

    except Exception as e:
        print(f"--- GraphBuilder run() method FAILED: {e} ---")
        import traceback
        traceback.print_exc()

    # if original_cuda_visible is None:
    #     if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    # print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")
    # CUDA device handling is now internal to GraphBuilder's run method

    if os.path.exists(base_test_dir):
        print(f"Cleaning up base temporary directory: {base_test_dir}")
        shutil.rmtree(base_test_dir)

    print(f"Total time for script: {time.time() - script_start_time:.2f}s")
