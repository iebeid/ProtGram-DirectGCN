# ==============================================================================
# MODULE: testers/unit_tests.py
# PURPOSE: A unified script for all environment checks, unit testers,
#          and pipeline smoke testers for the ProtGram-DirectGCN project.
# VERSION: 1.2 (Corrects test isolation and data type issues)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, List

# --- Dependencies from unit_tests.py ---
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import mlflow

# --- Local Application Imports ---
from configuration.config import Config
from source.benchmarkers.gnns import GNNBenchmarker
from source.data_builders.protgram import ProtGramBuilder
from source.experiments.ppi_1 import PPIPipeline
from source.models.fnn.mlp import MLP
from source.trainers.transformers import TransformerEmbedder
from source.trainers.word2vec import Word2VecEmbedder
from source.utils.data import DataUtils, IDMapGenerator
from source.utils.models import EmbeddingLoader
from source.utils.results import EvaluationReporter


# ==============================================================================
# --- NEW SECTION: Comprehensive GPU Environment Verification ---
# ==============================================================================

def verify_full_gpu_environment() -> bool:
    """
    A comprehensive test to verify that CUDA, cuDNN, PyTorch, and TensorFlow
    are all correctly configured and can access the GPU within the same script execution.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("Comprehensive GPU Environment Verification")
    print("=" * 80)
    all_ok = True

    # --- Part 1: PyTorch Verification ---
    print("\n--- Verifying PyTorch ---")
    try:
        print(f"PyTorch Version: {torch.__version__}")
        pt_cuda_available = torch.cuda.is_available()
        print(f"Is CUDA available for PyTorch? -> {pt_cuda_available}")
        if not pt_cuda_available:
            print("  ❌ [Error] PyTorch cannot find a CUDA-enabled GPU.")
            all_ok = False
        else:
            print(f"  CUDA Version PyTorch built with: {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"  Number of GPUs found: {device_count}")
            for i in range(device_count):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            # Perform a test operation
            device = torch.device("cuda")
            cpu_tensor = torch.randn(2, 2)
            gpu_tensor = cpu_tensor.to(device)
            gpu_result = gpu_tensor * gpu_tensor
            print(f"  ✅ [Success] PyTorch GPU tensor operation completed.")
    except Exception as e:
        print(f"  ❌ [Error] An unexpected error occurred during PyTorch verification: {e}")
        all_ok = False

    # --- Part 2: TensorFlow Verification ---
    print("\n--- Verifying TensorFlow ---")
    try:
        print(f"TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Is CUDA available for TensorFlow? -> {len(gpus) > 0}")
        if not gpus:
            print("  ❌ [Error] TensorFlow cannot find a CUDA-enabled GPU.")
            all_ok = False
        else:
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
                c = tf.matmul(a, b)
            # Accessing .numpy() forces computation
            _ = c.numpy()
            print(f"  ✅ [Success] TensorFlow GPU tensor operation completed.")
    except Exception as e:
        print(f"  ❌ [Error] An unexpected error occurred during TensorFlow verification: {e}")
        all_ok = False

    # --- Part 3: cuDNN Library Check ---
    print("\n--- Verifying cuDNN Library ---")
    try:
        # The most reliable check is whether the PyTorch backend can access it,
        # as we intentionally remove the nvidia-cudnn pip package.
        cudnn_available = torch.backends.cudnn.is_available()
        if cudnn_available:
            cudnn_version = torch.backends.cudnn.version()
            print(f"  ✅ [Success] PyTorch backend reports cuDNN is available.")
            print(f"  cuDNN Version reported by PyTorch: {cudnn_version}")
        else:
            print("  ❌ [Error] PyTorch backend reports cuDNN is NOT available.")
            all_ok = False
    except Exception as e:
        print(f"  ❌ [Error] An unexpected error occurred during cuDNN verification: {e}")
        all_ok = False

    # --- Part 4: Final Summary ---
    print("\n" + "-" * 40)
    if all_ok:
        print("✅ GPU Environment Verification Passed for both PyTorch and TensorFlow.")
    else:
        print("❌ GPU Environment Verification FAILED. Please check the errors above.")
    print("-" * 40)
    return all_ok


# ==============================================================================
# --- SECTION 1: CUDA and cuDNN Verification (from verify_cuda_cudnn.py) ---
# ==============================================================================

def verify_cuda_with_pycuda():
    """
    Verifies basic CUDA functionality by performing a matrix multiplication on the GPU using PyCUDA.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("CUDA Verification with PyCUDA")
    print("=" * 80)
    try:
        # 1. Import PyCUDA and initialize it
        import pycuda.autoinit
        import pycuda.driver as drv
        import pycuda.gpuarray as gpuarray
        import numpy as np

        # 2. Clear caches now that we know the module is imported and initialized.
        # This resolves the UnboundLocalError and ensures we start fresh.
        import pycuda.tools
        pycuda.tools.clear_context_caches()

        # 3. Get information about the current GPU device
        device = drv.Device(0)
        print(f"Successfully selected GPU 0: {device.name()}")

    except Exception as e:
        print("Error: Could not find or initialize a CUDA-enabled GPU with PyCUDA.")
        print("Please ensure PyCUDA is installed correctly and your CUDA drivers are working.")
        print(f"Details: {e}")
        return  # Exit this function if PyCUDA fails

    # 4. Create two random matrices of the same shape on the CPU using NumPy
    print("\nCreating two random matrices on the CPU (NumPy)...")
    matrix_a_cpu = np.random.randn(512, 1024).astype(np.float32)
    matrix_b_cpu = np.random.randn(512, 1024).astype(np.float32)  # Shape must match for element-wise op
    print(f"Matrix A shape: {matrix_a_cpu.shape} (on CPU)")
    print(f"Matrix B shape: {matrix_b_cpu.shape} (on CPU)")

    # 5. Transfer the matrices from the CPU to the GPU
    print("\nTransferring matrices from CPU to GPU...")
    matrix_a_gpu = gpuarray.to_gpu(matrix_a_cpu)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b_cpu)
    print("Transfer complete.")

    # 6. Perform a simple element-wise operation on the GPU.
    # This is more robust for a basic verification test than a dot product, which can have complex dependencies.
    print("\nPerforming element-wise addition on the GPU...")
    result_gpu = matrix_a_gpu + matrix_b_gpu

    # PyCUDA operations are synchronous by default in this context, so the next line executes after the dot product is complete.
    print("GPU operation complete.")
    print(f"Result matrix shape: {result_gpu.shape} (on GPU)")

    # 7. Transfer the result back to the CPU (as a NumPy array) to print it
    result_cpu = result_gpu.get()

    print(f"\nVerification successful! A small subset of the result tensor:\n{result_cpu[:2, :2]}")
    print("\nThis confirms that CUDA is working correctly for direct computation via PyCUDA.")


def verify_cudnn_with_pycuda_lib():
    """
    Verifies that the nvidia-cudnn-python library is installed and accessible.
    The functional verification of cuDNN is implicitly handled by the PyTorch and TensorFlow testers.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("cuDNN Library Verification (nvidia-cudnn-python)")
    print("=" * 80)

    try:
        # 1. Import necessary library
        import nvidia.cudnn
        # 2. Get and print the version using the stable PyTorch backend
        version = torch.backends.cudnn.version()
        print(f"✅ [Success] Found and imported 'nvidia.cudnn' library.")
        print(f"  cuDNN Version reported by PyTorch backend: {version}")
        print("  Functional test is deferred to PyTorch/TensorFlow diagnostics.")
    except ImportError:
        print("❌ [Error] Could not import the 'nvidia.cudnn' library.")
        print("  Please ensure 'nvidia-cudnn-cu12' (or similar) is installed via pip.")
    except Exception as e:
        print(f"❌ [Error] An unexpected error occurred while checking nvidia-cudnn-python: {e}")


# ==============================================================================
# --- SECTION 2: Framework GPU Diagnostics (from unit_tests.py) ---
# ==============================================================================

def test_pytorch_gpu():
    """
    Checks the status of PyTorch's CUDA availability and prints diagnostic information.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("PyTorch GPU Diagnostic")
    print("=" * 80)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    is_cuda_available = torch.cuda.is_available()
    print(f"\nIs CUDA available? -> {is_cuda_available}")

    if not is_cuda_available:
        print("\n[Error] PyTorch cannot find a CUDA-enabled GPU.")
        print("  This may be due to a driver issue or an incorrect PyTorch installation.")
    else:
        print("\n[Success] PyTorch has detected a CUDA-enabled GPU.")
        print(f"  CUDA Version PyTorch was built with: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"  Number of GPUs found: {device_count}")
        for i in range(device_count):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")

        cudnn_available = torch.backends.cudnn.is_available()
        print(f"\nIs cuDNN available? -> {cudnn_available}")
        if cudnn_available:
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            torch.backends.cudnn.benchmark = True
            print(f"  torch.backends.cudnn.benchmark set to {torch.backends.cudnn.benchmark}")
        else:
            print("  [Warning] cuDNN is not available or not enabled for PyTorch.")

        device = torch.device("cuda")
        print("\nAttempting a simple tensor operation on GPU...")
        try:
            cpu_tensor = torch.randn(3, 3)
            print(f"  Tensor on CPU: (Device: {cpu_tensor.device})\n{cpu_tensor}")
            gpu_tensor = cpu_tensor.to(device)
            print(f"  Tensor on GPU: (Device: {gpu_tensor.device})\n{gpu_tensor}")
            gpu_result = gpu_tensor * gpu_tensor
            print(f"  Result of computation on GPU:\n{gpu_result}")
            print("\n[Success] PyTorch GPU tensor operations seem to be working.")
        except Exception as e:
            print(f"\n[Error] Failed PyTorch GPU tensor operation: {e}")


def test_tensorflow_gpu():
    """
    Checks for GPU availability in TensorFlow and performs a test operation.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("TensorFlow GPU Diagnostic")
    print("=" * 80)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU(s) found! Total devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
        print("  TensorFlow typically uses cuDNN if a GPU is detected and CUDA is set up correctly.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("  Set GPU memory growth to True.")
            with tf.device('/GPU:0'):
                print("\n--- Performing a test matrix multiplication on GPU:0 ---")
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
                c = tf.matmul(a, b)
            print("Matrix A:\n", a.numpy())
            print("Matrix B:\n", b.numpy())
            print("Result of A * B on GPU:\n", c.numpy())
            print("\n🎉 TensorFlow GPU is set up correctly and operational!")
        except RuntimeError as e:
            print(f"\n❌ An error occurred while trying to use the TensorFlow GPU: {e}")
        except Exception as e_gen:
            print(f"\n❌ An unexpected error occurred with TensorFlow GPU: {e_gen}")
    else:
        print("\n❌ No GPU detected by TensorFlow.")


# ==============================================================================
# --- SECTION 3: Helper Functions for Dummy Pipeline Tests ---
# ==============================================================================

def _create_dummy_fasta_for_testing(directory: str, filename: str = "dummy_test.fasta", num_seqs: int = 5):
    """Creates a small dummy FASTA file for testing purposes."""
    os.makedirs(directory, exist_ok=True)
    fasta_path = os.path.join(directory, filename)
    with open(fasta_path, "w") as f:
        for i in range(num_seqs):
            seq_id = f"dummy_prot_{i + 1}"
            sequence = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=np.random.randint(20, 50)))
            f.write(f">{seq_id}\n{sequence}\n")
    return fasta_path


def _create_dummy_interaction_files_for_testing(directory: str, num_pairs: int = 10, num_proteins: int = 20):
    """Creates dummy positive and negative interaction files."""
    os.makedirs(directory, exist_ok=True)
    protein_ids = [f"P{i:03d}" for i in range(num_proteins)]
    pos_path = os.path.join(directory, "dummy_positive_interactions.csv")
    neg_path = os.path.join(directory, "dummy_negative_interactions.csv")

    def generate_pairs(filepath, count):
        pairs = set()
        max_possible_unique_pairs = num_proteins * (num_proteins - 1) // 2 if num_proteins >= 2 else 0
        actual_count = min(count, max_possible_unique_pairs)
        if count > max_possible_unique_pairs and actual_count > 0:
            print(f"  Warning: Requested {count} pairs, but only {actual_count} unique pairs possible. Generating {actual_count}.")
        elif actual_count == 0 and count > 0:
            print(f"  Warning: Cannot generate {count} pairs with {num_proteins} proteins. Generating 0 pairs.")
        attempts = 0
        max_attempts_multiplier = 5
        while len(pairs) < actual_count and attempts < actual_count * max_attempts_multiplier:
            if num_proteins < 2: break
            p1, p2 = np.random.choice(protein_ids, 2, replace=False)
            pairs.add(tuple(sorted((p1, p2))))
            attempts += 1
        df = pd.DataFrame(list(pairs), columns=['p1', 'p2'])
        df.to_csv(filepath, header=False, index=False)

    generate_pairs(pos_path, num_pairs)
    generate_pairs(neg_path, num_pairs)
    return pos_path, neg_path


def _create_dummy_h5_embeddings_for_testing(directory: str, filename: str = "dummy_embeddings.h5", protein_ids: Optional[List[str]] = None, num_proteins: int = 20, dim: int = 10):
    """Creates a dummy H5 embedding file."""
    os.makedirs(directory, exist_ok=True)
    h5_path = os.path.join(directory, filename)
    if protein_ids is None:
        protein_ids = [f"DUMMY_P{i:05d}" for i in range(num_proteins)]
    with h5py.File(h5_path, 'w') as hf:
        for pid in protein_ids:
            # FIX: Use float32 to prevent potential TF/cuDNN issues with float16
            hf.create_dataset(pid, data=np.random.rand(dim).astype(np.float32))
    return h5_path


# ==============================================================================
# --- SECTION 4: Standalone Unit/Smoke Tests (from unit_tests.py) ---
# ==============================================================================

def test_reporter():
    print("\n" + "=" * 80)
    DataUtils.print_header("EvaluationReporter Test")
    print("=" * 80)
    sample_k_vals = [10, 20]
    test_output_dir = "./temp_test_evaluation_reporter_output"
    if os.path.exists(test_output_dir): shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    try:
        reporter = EvaluationReporter(base_output_dir=test_output_dir, k_vals_table=sample_k_vals)
        history1 = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.55, 0.42, 0.33], 'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.68, 0.78, 0.88]}
        reporter.plot_training_history(history1, "Model_A_Fold1")
        results_data = [{'embedding_name': 'Model_A', 'test_auc_sklearn': 0.92, 'test_f1_sklearn': 0.85, 'test_precision_sklearn': 0.88, 'test_recall_sklearn': 0.82, 'test_hits_at_10': 50, 'test_ndcg_at_10': 0.75,
                         'test_hits_at_20': 80, 'test_ndcg_at_20': 0.78, 'test_auc_sklearn_std': 0.01, 'test_f1_sklearn_std': 0.02, 'roc_data_representative': (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]), 0.92),
                         'fold_auc_scores': [0.91, 0.93], 'fold_f1_scores': [0.84, 0.86]},
                        {'embedding_name': 'Model_B', 'test_auc_sklearn': 0.88, 'test_f1_sklearn': 0.80, 'test_precision_sklearn': 0.82, 'test_recall_sklearn': 0.78, 'test_hits_at_10': 40, 'test_ndcg_at_10': 0.65,
                         'test_hits_at_20': 70, 'test_ndcg_at_20': 0.68, 'test_auc_sklearn_std': 0.015, 'test_f1_sklearn_std': 0.022, 'roc_data_representative': (np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), 0.88),
                         'fold_auc_scores': [0.87, 0.89], 'fold_f1_scores': [0.79, 0.81]}]
        reporter.plot_roc_curves(results_data)
        reporter.plot_comparison_charts(results_data)
        reporter.write_summary_file(results_data, main_emb_name='Model_A', test_metric='test_auc_sklearn', alpha=0.05)
        print(f"  Example reporting complete. Check './{os.path.basename(test_output_dir)}' directory.")
    finally:
        if os.path.exists(test_output_dir): shutil.rmtree(test_output_dir)
    print(f"--- EvaluationReporter Test Complete ---")


def test_data_utilities():
    print("\n" + "=" * 80)
    DataUtils.print_header("Data Utilities Test")
    print("=" * 80)
    config_instance = Config()
    temp_test_dir_base = "./temp_test_data_utilities_output"
    if os.path.exists(temp_test_dir_base): shutil.rmtree(temp_test_dir_base)
    os.makedirs(temp_test_dir_base, exist_ok=True)
    try:
        print("\nTesting EmbeddingLoader:")
        dummy_h5_path = os.path.join(temp_test_dir_base, "temp_dummy_embeddings.h5")
        with h5py.File(dummy_h5_path, 'w') as hf:
            hf.create_dataset("protein_X", data=np.random.rand(10))
        with EmbeddingLoader(dummy_h5_path) as loader:
            if "protein_X" in loader:
                embedding = loader["protein_X"]
                print(f"  Successfully loaded dummy embedding for protein_X, shape: {embedding.shape}")
        print(f"  EmbeddingLoader test passed.")

        print("\nTesting DataLoader ID mapping:")
        dummy_fasta_path = os.path.join(temp_test_dir_base, "dummy_id_map.fasta")
        # Override the sequence file paths to point to our dummy file
        config_instance.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
        # Use a temporary file for the mapping output to ensure isolation
        config_instance.ID_MAPPING_PATH = Path(os.path.join(temp_test_dir_base, "dummy_id_map.tsv"))
        config_instance.ID_MAPPING_MODE = 'regex'
        with open(dummy_fasta_path, 'w') as f:
            f.write(">sp|P12345|TEST_HUMAN Test protein\nACDEFGHIKLMNPQRSTVWY\n")
            f.write(">tr|A0A0A0|ANOTHER_TEST Another test\nWYTSRQPONMLKIHGFEDCA\n")
        parser_mapper = IDMapGenerator(config=config_instance)
        id_map_dictionary = parser_mapper.generate_id_maps()
        print(f"  DataLoader generate_id_maps called. Number of mappings: {len(id_map_dictionary)}")
        assert len(id_map_dictionary) > 0, "ID mapping should produce some results."
        print(f"  DataLoader ID mapping test passed.")
    except Exception as e:
        print(f"  A Data Utility test failed: {e}")
        raise
    finally:
        if os.path.exists(temp_test_dir_base): shutil.rmtree(temp_test_dir_base)
    print(f"--- Data Utilities Test Complete ---")


def test_mlp_model_build():
    print("\n" + "=" * 80)
    DataUtils.print_header("MLP Model Build Test")
    print("=" * 80)
    config_instance = Config()
    dummy_mlp_params = {'dense1_units': 32, 'dropout1_rate': 0.1, 'dense2_units': 16, 'dropout2_rate': 0.1, 'l2_reg': 0.001}
    input_dim = 128
    try:
        mlp_builder = MLP(input_shape=input_dim, mlp_params=dummy_mlp_params, learning_rate=config_instance.EVAL_LEARNING_RATE)
        model = mlp_builder.build()
        assert model is not None, "MLP model build failed, model is None."
        assert model.input_shape == (None, input_dim), f"MLP input shape mismatch."
        model.summary(print_fn=lambda x: print(f"  {x}"))
        print(f"\n  MLPModelBuilder build test passed.")
    except Exception as e:
        print(f"  MLPModelBuilder build test FAILED: {e}")
        raise
    print(f"--- MLPModelBuilder Build Test Complete ---")


# ==============================================================================
# --- SECTION 5: GraphBuilder Full Run Test (from run_graph_builder.py) ---
# ==============================================================================

def run_graph_builder_full_test():
    """
    Runs a full, synchronous test of the GraphBuilder pipeline.
    """
    script_start_time = time.time()
    print("\n" + "=" * 80)
    DataUtils.print_header("Starting GraphBuilder Full Run Test (Synchronous Dask)")
    print("=" * 80)

    base_test_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(base_test_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)

    fasta_content = (
        ">seq1\nACGTACT\n"
        ">seq2\nTTACGTT\n"
        ">seq3\nAGATAGA\n"
    )
    fasta_path = os.path.join(temp_input_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    config = Config()
    # FIX: Isolate ALL paths by overriding them *after* initialization to avoid being reset.
    config.BASE_OUTPUT_DIR = Path(base_test_dir) / "test_pipeline_output"
    config.RESULTS_GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "graph_objects"
    config.SEQUENCE_FILE_PATHS = [Path(fasta_path)]

    config.DEBUG_VERBOSE = True
    config.GCN_NGRAM_MAX_N = 3
    config.GRAPH_BUILDER_WORKERS = 1  # Force synchronous

    print(f"--- Running GraphBuilder instance for n_max={config.GCN_NGRAM_MAX_N} ---")
    print(f"  Input FASTA from list: {config.SEQUENCE_FILE_PATHS[0]}")
    print(f"  GraphBuilder output will be within: {config.BASE_OUTPUT_DIR.name}")

    try:
        graph_builder_instance = ProtGramBuilder(config)
        graph_builder_instance.run()
        print(f"--- GraphBuilder run() method completed ---")
        for n_val_check in range(1, config.GCN_NGRAM_MAX_N + 1):
            expected_graph_file = config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n_val_check}.pkl"
            if expected_graph_file.exists():
                print(f"  OK: Final graph object file found: {expected_graph_file}")
            else:
                print(f"  WARN: Final graph object file NOT found: {expected_graph_file}")
    except Exception as e:
        print(f"--- GraphBuilder run() method FAILED: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(base_test_dir):
            print(f"Cleaning up base temporary directory: {base_test_dir}")
            shutil.rmtree(base_test_dir)

    print(f"Total time for script: {time.time() - script_start_time:.2f}s")
    print("--- GraphBuilder Full Run Test Complete ---")


# ==============================================================================
# --- SECTION 6: Unittest Class for GraphBuilder Smoke Test ---
# ==============================================================================

class TestGraphBuilderSmoke(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory and a dummy FASTA file."""
        self.base_test_dir = tempfile.mkdtemp()
        self.temp_input_dir = os.path.join(self.base_test_dir, "input")
        os.makedirs(self.temp_input_dir, exist_ok=True)
        self.fasta_content = ">seq1\nACGT\n>seq2\nTTAC\n>seq3\nAGA\n"
        self.fasta_path = os.path.join(self.temp_input_dir, "test_sequences.fasta")
        with open(self.fasta_path, "w") as f:
            f.write(self.fasta_content)
        self.temp_output_dir = os.path.join(self.base_test_dir, "output")
        os.makedirs(self.temp_output_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        if os.path.exists(self.base_test_dir):
            shutil.rmtree(self.base_test_dir)

    def test_graph_builder_smoke(self):
        """Smoke test for GraphBuilder.run() with minimal config."""
        print("\n" + "=" * 80)
        DataUtils.print_header("GraphBuilder Smoke Test (unittest)")
        print("=" * 80)
        try:
            config = Config()
            # FIX: Properly isolate all paths used by GraphBuilder by overriding them after init
            config.BASE_OUTPUT_DIR = Path(self.temp_output_dir)
            config.RESULTS_GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "graph_objects"
            config.SEQUENCE_FILE_PATHS = [Path(self.fasta_path)]

            config.GCN_NGRAM_MAX_N = 1
            config.GRAPH_BUILDER_WORKERS = 1

            print(f"  Running GraphBuilder with FASTA: {config.SEQUENCE_FILE_PATHS[0]}")
            print(f"  Outputting to: {config.BASE_OUTPUT_DIR.name}")
            print(f"  N_max: {config.GCN_NGRAM_MAX_N}, Workers: {config.GRAPH_BUILDER_WORKERS}")

            graph_builder = ProtGramBuilder(config)
            graph_builder.run()

            expected_graph_file = config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{config.GCN_NGRAM_MAX_N}.pkl"
            self.assertTrue(expected_graph_file.exists(), f"Expected graph file not found: {expected_graph_file}")
            print(f"\n  GraphBuilder smoke test completed successfully. Output file found: {expected_graph_file}")

        except Exception as e:
            print(f"\n  GraphBuilder smoke test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"GraphBuilder.run() raised an exception: {e}")
        print("--- GraphBuilder Smoke Test Complete ---")


# ==============================================================================
# --- SECTION 7: Pipeline Smoke Tests (from unit_tests.py) ---
# ==============================================================================

def test_word2vec_pipeline_run():
    print("\n" + "=" * 80)
    DataUtils.print_header("Word2Vec Pipeline Smoke Test")
    print("=" * 80)
    config = Config()
    base_test_dir = tempfile.mkdtemp()
    dummy_fasta_path = _create_dummy_fasta_for_testing(os.path.join(base_test_dir, "input"), "w2v_test.fasta")

    # Store original paths and settings
    original_fasta_paths = config.SEQUENCE_FILE_PATHS
    original_w2v_output_dir = config.RESULTS_W2V_EMBEDDINGS_DIR
    original_epochs = config.W2V_EPOCHS

    # Override with temporary test settings
    config.SEQUENCE_FILE_PATHS = [Path(dummy_fasta_path)]
    config.RESULTS_W2V_EMBEDDINGS_DIR = Path(base_test_dir) / "test_w2v_embeddings"
    config.W2V_EPOCHS = 1
    config.APPLY_PCA_TO_W2V = False  # Keep it fast for a smoke test

    try:
        embedder = Word2VecEmbedder(config)
        embedder.run()
        print("\n  Word2VecEmbedder smoke test ran successfully.")
    except Exception as e:
        print(f"\n  Word2VecEmbedder smoke test FAILED: {e}")
        raise
    finally:
        # Restore original settings
        config.SEQUENCE_FILE_PATHS = original_fasta_paths
        config.RESULTS_W2V_EMBEDDINGS_DIR = original_w2v_output_dir
        config.W2V_EPOCHS = original_epochs
        # Clean up temporary files
        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
    print("--- Word2Vec Pipeline Smoke Test Complete ---")


def test_transformer_embedder_pipeline_run():
    print("\n" + "=" * 80)
    DataUtils.print_header("Transformer Embedder Pipeline Smoke Test")
    print("=" * 80)
    config = Config()
    base_test_dir = tempfile.mkdtemp()
    # FIX: The input directory for the test should be where the dummy FASTA is, not the output dir.
    dummy_input_dir = Path(base_test_dir) / "input"
    dummy_input_dir.mkdir()
    _create_dummy_fasta_for_testing(str(dummy_input_dir), "transformer_test.fasta", num_seqs=2)

    # Store original paths and settings
    original_sequence_paths = config.SEQUENCE_FILE_PATHS
    original_transformer_output_dir = config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR

    # Override with temporary test settings
    config.SEQUENCE_FILE_PATHS = [dummy_input_dir / "transformer_test.fasta"]
    config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR = Path(base_test_dir) / "test_transformer_embeddings"
    config.APPLY_PCA_TO_TRANSFORMER = False
    config.TRANSFORMER_BASE_BATCH_SIZE = 1

    try:
        # FIX: Add meaningful logging to the MLflow run context.
        with mlflow.start_run(run_name="Transformer_Embedder_SMOKE_TEST") as run:
            mlflow.set_tag("test_type", "smoke_test")
            # Log key parameters used in the test
            model_config_to_test = config.TRANSFORMER_MODELS_TO_RUN[0]
            mlflow.log_params({
                "model_name": model_config_to_test['name'],
                "hf_id": model_config_to_test['hf_id'],
                "pooling_strategy": config.TRANSFORMER_POOLING_STRATEGY
            })

            embedder = TransformerEmbedder(config)
            generated_paths = embedder.run()
            assert isinstance(generated_paths, dict), "TransformerEmbedder.run() should return a dictionary."
            assert "ProtBERT" in generated_paths, "Expected 'ProtBERT' key in the returned paths."
            assert generated_paths["ProtBERT"].exists(), "The embedding file for ProtBERT was not created."
            # Log the output file as an artifact for inspection
            mlflow.log_artifact(str(generated_paths["ProtBERT"]), "generated_embeddings")
            print("\n  TransformerEmbedder smoke test ran successfully.")
    except Exception as e:
        print(f"\n  TransformerEmbedder smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original settings
        config.SEQUENCE_FILE_PATHS = original_sequence_paths
        config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR = original_transformer_output_dir
        # Clean up temporary files
        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)
    print("--- Transformer Embedder Pipeline Smoke Test Complete ---")


def test_gnn_benchmarker_run():
    print("\n" + "=" * 80)
    DataUtils.print_header("GNN Benchmarker Smoke Test")
    print("=" * 80)
    config = Config()
    original_datasets = config.BENCHMARK_NODE_CLASSIFICATION_DATASETS
    config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = ["KarateClub"]
    original_epochs = config.EVAL_EPOCHS
    config.EVAL_EPOCHS = 1
    config.BENCHMARK_SAVE_EMBEDDINGS = False
    config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = False

    test_benchmark_output_dir = Path(config.BASE_OUTPUT_DIR) / "test_gnn_benchmark_results"
    if os.path.exists(test_benchmark_output_dir): shutil.rmtree(test_benchmark_output_dir)
    # --- FIX: The directory must be created before it can be used ---
    test_benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    # FIX: Use and restore the correct config variable name
    original_benchmark_output_dir = config.RESULTS_BENCHMARKING_DIR
    config.RESULTS_BENCHMARKING_DIR = test_benchmark_output_dir

    pyg_dataset_root = Path(config.BASE_DATA_DIR) / "standard_datasets_pyg"
    karate_specific_path = pyg_dataset_root / "KarateClub"

    try:
        # FIX: Wrap the test in its own MLflow run to manage context properly.
        with mlflow.start_run(run_name="GNN_Benchmark_SMOKE_TEST"):
            benchmarker = GNNBenchmarker(config)
            benchmarker.run()
            print("\n  GNNBenchmarker smoke test ran successfully.")
    except Exception as e:
        print(f"\n  GNNBenchmarker smoke test FAILED: {e}")
        raise
    finally:
        config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = original_datasets
        config.EVAL_EPOCHS = original_epochs
        # FIX: Restore the correct config variable
        config.RESULTS_BENCHMARKING_DIR = original_benchmark_output_dir
        if os.path.exists(test_benchmark_output_dir): shutil.rmtree(test_benchmark_output_dir)
        if os.path.exists(karate_specific_path): shutil.rmtree(karate_specific_path)
    print("--- GNN Benchmarker Smoke Test Complete ---")


def test_ppi_pipeline_run():
    print("\n" + "=" * 80)
    DataUtils.print_header("PPI Pipeline (Dummy Run) Smoke Test")
    print("=" * 80)
    config = Config()
    original_dummy_flag = config.RUN_DUMMY_TEST
    config.RUN_DUMMY_TEST = True
    original_epochs = config.EVAL_EPOCHS
    config.EVAL_EPOCHS = 1
    original_folds = config.EVAL_N_FOLDS
    config.EVAL_N_FOLDS = 2

    test_ppi_output_dir = Path(config.BASE_OUTPUT_DIR) / "test_ppi_eval_results"
    if os.path.exists(test_ppi_output_dir): shutil.rmtree(test_ppi_output_dir)
    # FIX: Use and restore the correct config variable name
    original_eval_results_dir = config.RESULTS_EVALUATION_DIR
    config.RESULTS_EVALUATION_DIR = test_ppi_output_dir

    try:
        # FIX: Wrap the test in its own MLflow run and pass the parent_run_id
        # to ensure the inner runs are correctly nested and closed.
        with mlflow.start_run(run_name="PPI_Pipeline_SMOKE_TEST") as parent_run:
            evaluator = PPIPipeline(config)
            evaluator.run(use_dummy_data=True, parent_run_id=parent_run.info.run_id)
            print("\n  PPIPipeline (dummy run) smoke test ran successfully.")
    except Exception as e:
        print(f"\n  PPIPipeline (dummy run) smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        config.RUN_DUMMY_TEST = original_dummy_flag
        config.EVAL_EPOCHS = original_epochs
        config.EVAL_N_FOLDS = original_folds
        # FIX: Restore the correct config variable
        config.RESULTS_EVALUATION_DIR = original_eval_results_dir
        # --- FIX: The dummy data is created inside the temporary test directory, so we clean that. ---
        if os.path.exists(test_ppi_output_dir) and config.CLEANUP_DUMMY_DATA:
            print(f"  Cleaning up temporary PPI test directory: {test_ppi_output_dir}")
            shutil.rmtree(test_ppi_output_dir)
    print("--- PPI Pipeline (Dummy Run) Smoke Test Complete ---")


# ==============================================================================
# --- SECTION 8: Main Execution Block ---
# ==============================================================================

def run_all_tests() -> bool:
    """
    Runs the full suite of verification, unit, and smoke testers.
    Returns True if the GPU environment is OK, False otherwise.
    """
    print("\n" + "#" * 100)
    print("### Starting All Integrated Tests... ###")
    print("#" * 100)

    # Phase 1: Comprehensive GPU Environment Verification
    gpu_ok = verify_full_gpu_environment()
    verify_cuda_with_pycuda()

    # Phase 2: Utility and Model Build Tests
    test_reporter()
    test_data_utilities()
    test_mlp_model_build()

    # Phase 3: Graph Builder Tests
    # Run the unittest version of the smoke test
    # This automatically handles setUp and tearDown
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGraphBuilderSmoke))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if result.failures or result.errors:
        print("FATAL: GraphBuilder smoke test (unittest) failed. Stopping test suite.")
        # sys.exit(1) # Optional: uncomment to halt execution entirely on failure

    # Run the full, synchronous GraphBuilder test only if the smoke test passed
    if not (result.failures or result.errors):
        run_graph_builder_full_test()

    # Phase 4: Full Pipeline Smoke Tests
    # Note: These are designed to be quick. You can comment out any that you don't need to run every time.
    test_word2vec_pipeline_run()
    test_transformer_embedder_pipeline_run()
    test_gnn_benchmarker_run()
    test_ppi_pipeline_run()

    print("\n" + "#" * 100)
    print("### All Integrated Tests Finished. ###")
    print("#" * 100)
    return gpu_ok
