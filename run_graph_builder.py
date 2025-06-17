# run_graph_builder.py
import time
import os
import shutil
import tempfile
from pathlib import Path
import dask  # Import dask for config setting

from config import Config
from src.pipeline.data_builder import GraphBuilder  # GraphBuilder._create_intermediate_files
from src.utils.data_utils import DataUtils  # For print_header

if __name__ == "__main__":
    script_start_time = time.time()
    DataUtils.print_header("Starting Minimal GraphBuilder Test (Synchronous Dask)")

    base_test_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(base_test_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)

    # Using the same dummy FASTA content as in your unit test
    fasta_content = ">seq1\nACGT\n>seq2\nTTAC\n>seq3\nAGA\n"
    fasta_path = os.path.join(temp_input_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    # Temporary directory for intermediate files (used by _create_intermediate_files)
    intermediate_temp_dir = os.path.join(base_test_dir, "intermediate_output")
    os.makedirs(intermediate_temp_dir, exist_ok=True)

    # Final graph objects directory (though we might not get this far if it crashes)
    final_graph_objects_dir = os.path.join(base_test_dir, "final_graphs")
    os.makedirs(final_graph_objects_dir, exist_ok=True)

    config = Config()  # Load default config
    # Override paths for the test
    config.GCN_INPUT_FASTA_PATH = Path(fasta_path)
    config.GRAPH_OBJECTS_DIR = Path(final_graph_objects_dir)  # For completeness
    config.BASE_OUTPUT_DIR = Path(base_test_dir)  # For temp_dir calculation if GraphBuilder instance was used

    n_value_to_test = 1
    num_dask_partitions_for_test = 1  # For a small file, 1 partition is fine

    print(f"  Testing _create_intermediate_files for n={n_value_to_test}")
    print(f"  Input FASTA: {fasta_path}")
    print(f"  Intermediate output to: {intermediate_temp_dir}")

    # --- Configure Dask to use the synchronous scheduler ---
    # This runs all Dask computations in the main thread, no separate workers.
    dask.config.set(scheduler='synchronous')
    print("  Dask scheduler set to 'synchronous'.")

    # --- Temporarily disable GPU for this specific test if issues persist ---
    # This is an extra precaution. The one inside _create_intermediate_files should also work.
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for the main process.")

    try:
        # Call the static method directly
        GraphBuilder._create_intermediate_files(
            n_value_to_test,
            intermediate_temp_dir,  # This is the 'temp_dir' argument for the method
            str(config.GCN_INPUT_FASTA_PATH),
            num_dask_partitions_for_test
        )
        print(f"--- Minimal GraphBuilder Test (_create_intermediate_files for n={n_value_to_test}) COMPLETED (Synchronous Dask) ---")

        # Check if files were created (basic check)
        expected_map_file = os.path.join(intermediate_temp_dir, f'ngram_map_n{n_value_to_test}.parquet')
        expected_edge_file = os.path.join(intermediate_temp_dir, f'edge_list_n{n_value_to_test}.txt')
        if os.path.exists(expected_map_file):
            print(f"  OK: Ngram map file created: {expected_map_file}")
        else:
            print(f"  FAIL: Ngram map file NOT created: {expected_map_file}")
        if os.path.exists(expected_edge_file):
            print(f"  OK: Edge list file created: {expected_edge_file}")
        else:
            print(f"  FAIL: Edge list file NOT created: {expected_edge_file}")

    except Exception as e:
        print(f"--- Minimal GraphBuilder Test (_create_intermediate_files) FAILED (Synchronous Dask): {e} ---")
        import traceback

        traceback.print_exc()
    finally:
        # Restore CUDA_VISIBLE_DEVICES
        if original_cuda_visible is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
        print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")

        if os.path.exists(base_test_dir):
            print(f"Cleaning up temporary directory: {base_test_dir}")
            shutil.rmtree(base_test_dir)

        print(f"Total time for minimal script: {time.time() - script_start_time:.2f}s")
