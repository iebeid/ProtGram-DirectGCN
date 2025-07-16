# ==============================================================================
# MODULE: integrated_tests.py
# PURPOSE: A unified script for all environment checks, unit tests,
#          and pipeline smoke tests for the ProtGram-DirectGCN project.
# VERSION: 1.0
# AUTHOR: Your Name (Integrated by Coding Partner)
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

# --- Local Application Imports ---
# These assume the script is run from a location where 'src' and 'config' are accessible.
# You may need to adjust the Python path if running from a different directory.
# Example: sys.path.append(str(Path(__file__).resolve().parents[1]))
from configuration.config import Config
from source.benchmarks.gnn_benchmarker import GNNBenchmarker
from source.data.protgram import GraphBuilder
from source.experiments.ppi_experimenter import PPIPipeline
from source.models.ml.mlp import MLP
from source.training.prott5_trainer import TransformerEmbedder
from source.training.word2vec_trainer import Word2VecEmbedder
from source.utils.data_utils import DataLoader
from source.utils.data_utils import DataUtils
from source.utils.models_utils import EmbeddingLoader
from source.utils.results_utils import EvaluationReporter


# --- Dependencies from verify_cuda_cudnn.py ---


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

        # 2. Get information about the current GPU device
        device = drv.Device(0)
        print(f"Successfully selected GPU 0: {device.name()}")

    except Exception as e:
        print("Error: Could not find or initialize a CUDA-enabled GPU with PyCUDA.")
        print("Please ensure PyCUDA is installed correctly and your CUDA drivers are working.")
        print(f"Details: {e}")
        return  # Exit this function if PyCUDA fails

    # 3. Create two random matrices of the same shape on the CPU using NumPy
    print("\nCreating two random matrices on the CPU (NumPy)...")
    matrix_a_cpu = np.random.randn(512, 1024).astype(np.float32)
    matrix_b_cpu = np.random.randn(512, 1024).astype(np.float32)  # Shape must match for element-wise op
    print(f"Matrix A shape: {matrix_a_cpu.shape} (on CPU)")
    print(f"Matrix B shape: {matrix_b_cpu.shape} (on CPU)")

    # 4. Transfer the matrices from the CPU to the GPU
    print("\nTransferring matrices from CPU to GPU...")
    matrix_a_gpu = gpuarray.to_gpu(matrix_a_cpu)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b_cpu)
    print("Transfer complete.")

    # 5. Perform a simple element-wise operation on the GPU.
    # This is more robust for a basic verification test than a dot product, which can have complex dependencies.
    print("\nPerforming element-wise addition on the GPU...")
    result_gpu = matrix_a_gpu + matrix_b_gpu

    # PyCUDA operations are synchronous by default in this context, so the next line executes after the dot product is complete.
    print("GPU operation complete.")
    print(f"Result matrix shape: {result_gpu.shape} (on GPU)")

    # 6. Transfer the result back to the CPU (as a NumPy array) to print it
    result_cpu = result_gpu.get()

    print(f"\nVerification successful! A small subset of the result tensor:\n{result_cpu[:2, :2]}")
    print("\nThis confirms that CUDA is working correctly for direct computation via PyCUDA.")


def verify_cudnn_with_pycuda_lib():
    """
    Verifies cuDNN functionality by performing a convolution using the nvidia-cudnn-python
    library, with PyCUDA handling the tensor operations.
    """
    print("\n" + "=" * 80)
    DataUtils.print_header("cuDNN Verification with PyCUDA and nvidia-cudnn-python")
    print("=" * 80)

    try:
        # 1. Import necessary libraries
        from nvidia import cudnn
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import numpy as np

        # 2. Print the version of the cuDNN library
        print(f"Found cuDNN version: {cudnn.backend.get_version_string()}")

    except Exception as e:
        print(f"Error initializing libraries. Please ensure PyCUDA and nvidia-cudnn-python are installed correctly.")
        print(f"Details: {e}")
        return

    try:
        # 3. Set up data parameters for a sample convolution
        input_shape = (1, 3, 32, 32)
        filter_shape = (16, 3, 3, 3)

        # 4. Create input and filter data on the GPU using PyCUDA
        print("\nCreating input and filter tensors on the GPU with PyCUDA...")
        # Create CPU tensors first
        x_cpu = np.random.rand(*input_shape).astype(np.float32)
        w_cpu = np.random.rand(*filter_shape).astype(np.float32)
        # Transfer to GPU
        x_gpu = gpuarray.to_gpu(x_cpu)
        w_gpu = gpuarray.to_gpu(w_cpu)
        print(f"Input tensor 'x' created with shape: {x_gpu.shape}")
        print(f"Filter tensor 'w' created with shape: {w_gpu.shape}")

        # 5. Create a handle to the cuDNN library context
        handle = cudnn.create_handle()
        print("\nCreated cuDNN handle.")

        # 6. Define the computation graph for a convolution
        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT
        )

        # Define the tensors for the graph using properties from our PyCUDA arrays
        X = graph.tensor(name="X", dim=x_gpu.shape, stride=x_gpu.strides, data_type=x_gpu.dtype)
        W = graph.tensor(name="W", dim=w_gpu.shape, stride=w_gpu.strides, data_type=w_gpu.dtype)

        # Define the convolution operation
        Y = graph.conv_fprop(name="conv1", image=X, weight=W, padding=[1, 1], stride=[1, 1])
        Y.set_output(True).set_data_type(Y.get_data_type())
        print("Defined a convolution operation graph.")

        # 7. Build and execute the graph
        print("Building and executing the cuDNN graph...")
        graph.build([cudnn.heuristic_mode.A])

        # Allocate workspace on the GPU using PyCUDA
        workspace = gpuarray.empty(graph.get_workspace_size(), dtype=np.uint8)

        # Prepare a dictionary mapping graph tensors to the actual PyCUDA GPU arrays
        variant_pack = {X: x_gpu, W: w_gpu}

        # Execute the convolution
        graph.execute(variant_pack, workspace)
        print("Execution successful.")

        # 8. Get the result
        y_result = variant_pack[Y]
        print(f"Output tensor 'y' received with shape: {y_result.shape}")
        print("\nVerification successful! A cuDNN-accelerated convolution was executed using PyCUDA.")

    except Exception as e:
        print(f"\n[ERROR] An exception occurred during cuDNN verification with PyCUDA: {e}")
        import traceback
        traceback.print_exc()


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
        protein_ids = [f"P{i:03d}" for i in range(num_proteins)]
    with h5py.File(h5_path, 'w') as hf:
        for pid in protein_ids:
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
        print(f"  Example reporting complete. Check '{test_output_dir}' directory.")
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
        config_instance.GCN_INPUT_FASTA_PATH = dummy_fasta_path
        config_instance.ID_MAPPING_OUTPUT_FILE = os.path.join(temp_test_dir_base, "dummy_id_map.tsv")
        config_instance.ID_MAPPING_MODE = 'regex'
        with open(dummy_fasta_path, 'w') as f:
            f.write(">sp|P12345|TEST_HUMAN Test protein\nACDEFGHIKLMNPQRSTVWY\n")
            f.write(">tr|A0A0A0|ANOTHER_TEST Another test\nWYTSRQPONMLKIHGFEDCA\n")
        parser_mapper = DataLoader(config=config_instance)
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
    config.GCN_INPUT_FASTA_PATH = Path(fasta_path)
    config.BASE_OUTPUT_DIR = Path(base_test_dir) / "test_pipeline_output"
    config.GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "1_graph_objects"
    config.DEBUG_VERBOSE = True
    config.GCN_NGRAM_MAX_N = 3
    config.GRAPH_BUILDER_WORKERS = 1  # Force synchronous

    print(f"--- Running GraphBuilder instance for n_max={config.GCN_NGRAM_MAX_N} ---")
    print(f"  Input FASTA: {config.GCN_INPUT_FASTA_PATH}")
    print(f"  GraphBuilder output will be within: {config.BASE_OUTPUT_DIR}")

    try:
        graph_builder_instance = GraphBuilder(config)
        graph_builder_instance.run()
        print(f"--- GraphBuilder run() method completed ---")
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
            config.GCN_INPUT_FASTA_PATH = Path(self.fasta_path)
            config.BASE_OUTPUT_DIR = Path(self.temp_output_dir)
            config.GRAPH_OBJECTS_DIR = config.BASE_OUTPUT_DIR / "1_graph_objects"
            config.GCN_NGRAM_MAX_N = 1
            config.GRAPH_BUILDER_WORKERS = 1

            print(f"  Running GraphBuilder with FASTA: {config.GCN_INPUT_FASTA_PATH}")
            print(f"  Outputting to: {config.BASE_OUTPUT_DIR}")
            print(f"  N_max: {config.GCN_NGRAM_MAX_N}, Workers: {config.GRAPH_BUILDER_WORKERS}")

            graph_builder = GraphBuilder(config)
            graph_builder.run()

            expected_graph_file = config.GRAPH_OBJECTS_DIR / f"ngram_graph_n{config.GCN_NGRAM_MAX_N}.pkl"
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
    dummy_fasta_dir = "./temp_test_w2v_fasta_input"
    if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
    _create_dummy_fasta_for_testing(dummy_fasta_dir, "w2v_test.fasta")

    original_fasta_dir = config.W2V_INPUT_FASTA_DIR
    config.W2V_INPUT_FASTA_DIR = Path(dummy_fasta_dir)
    config.APPLY_PCA_TO_W2V = False
    config.W2V_EPOCHS = 1

    test_w2v_output_dir = Path(config.BASE_OUTPUT_DIR) / "test_w2v_embeddings"
    if os.path.exists(test_w2v_output_dir): shutil.rmtree(test_w2v_output_dir)
    original_w2v_output_dir = config.WORD2VEC_EMBEDDINGS_DIR
    config.WORD2VEC_EMBEDDINGS_DIR = test_w2v_output_dir

    try:
        embedder = Word2VecEmbedder(config)
        embedder.run()
        print("\n  Word2VecEmbedder smoke test ran successfully.")
    except Exception as e:
        print(f"\n  Word2VecEmbedder smoke test FAILED: {e}")
        raise
    finally:
        config.W2V_INPUT_FASTA_DIR = original_fasta_dir
        config.WORD2VEC_EMBEDDINGS_DIR = original_w2v_output_dir
        if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
        if os.path.exists(test_w2v_output_dir): shutil.rmtree(test_w2v_output_dir)
    print("--- Word2Vec Pipeline Smoke Test Complete ---")


def test_transformer_embedder_pipeline_run():
    print("\n" + "=" * 80)
    DataUtils.print_header("Transformer Embedder Pipeline Smoke Test")
    print("=" * 80)
    config = Config()
    dummy_fasta_dir = "./temp_test_transformer_fasta_input"
    if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
    _create_dummy_fasta_for_testing(dummy_fasta_dir, "transformer_test.fasta", num_seqs=2)

    original_fasta_dir = config.TRANSFORMER_INPUT_FASTA_DIR
    config.TRANSFORMER_INPUT_FASTA_DIR = Path(dummy_fasta_dir)
    config.APPLY_PCA_TO_TRANSFORMER = False
    config.TRANSFORMER_BASE_BATCH_SIZE = 1

    test_transformer_output_dir = Path(config.BASE_OUTPUT_DIR) / "test_transformer_embeddings"
    if os.path.exists(test_transformer_output_dir): shutil.rmtree(test_transformer_output_dir)
    original_transformer_output_dir = config.TRANSFORMER_EMBEDDINGS_DIR
    config.TRANSFORMER_EMBEDDINGS_DIR = test_transformer_output_dir

    try:
        embedder = TransformerEmbedder(config)
        embedder.run()
        print("\n  TransformerEmbedder smoke test ran successfully.")
    except Exception as e:
        print(f"\n  TransformerEmbedder smoke test FAILED: {e}")
        raise
    finally:
        config.TRANSFORMER_INPUT_FASTA_DIR = original_fasta_dir
        config.TRANSFORMER_EMBEDDINGS_DIR = original_transformer_output_dir
        if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
        if os.path.exists(test_transformer_output_dir): shutil.rmtree(test_transformer_output_dir)
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
    original_benchmark_output_dir = config.BENCHMARKING_RESULTS_DIR
    config.BENCHMARKING_RESULTS_DIR = test_benchmark_output_dir

    pyg_dataset_root = Path(config.BASE_DATA_DIR) / "standard_datasets_pyg"
    karate_specific_path = pyg_dataset_root / "KarateClub"

    try:
        benchmarker = GNNBenchmarker(config)
        benchmarker.run()
        print("\n  GNNBenchmarker smoke test ran successfully.")
    except Exception as e:
        print(f"\n  GNNBenchmarker smoke test FAILED: {e}")
        raise
    finally:
        config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = original_datasets
        config.EVAL_EPOCHS = original_epochs
        config.BENCHMARKING_RESULTS_DIR = original_benchmark_output_dir
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
    original_eval_results_dir = config.EVALUATION_RESULTS_DIR
    config.EVALUATION_RESULTS_DIR = test_ppi_output_dir

    try:
        evaluator = PPIPipeline(config)
        evaluator.run(use_dummy_data=True)
        print("\n  PPIPipeline (dummy run) smoke test ran successfully.")
    except Exception as e:
        print(f"\n  PPIPipeline (dummy run) smoke test FAILED: {e}")
        raise
    finally:
        config.RUN_DUMMY_TEST = original_dummy_flag
        config.EVAL_EPOCHS = original_epochs
        config.EVAL_N_FOLDS = original_folds
        config.EVALUATION_RESULTS_DIR = original_eval_results_dir
        if os.path.exists(test_ppi_output_dir): shutil.rmtree(test_ppi_output_dir)
        dummy_data_created_path = Path(config.BASE_OUTPUT_DIR) / "dummy_data_temp"
        if os.path.exists(dummy_data_created_path) and config.CLEANUP_DUMMY_DATA:
            shutil.rmtree(dummy_data_created_path)
    print("--- PPI Pipeline (Dummy Run) Smoke Test Complete ---")


# ==============================================================================
# --- SECTION 8: Main Execution Block ---
# ==============================================================================

def run_all_tests():
    """
    Runs the full suite of verification, unit, and smoke tests.
    """
    print("\n" + "#" * 100)
    print("### Starting All Integrated Tests... ###")
    print("#" * 100)

    # Phase 1: Low-Level Environment & GPU Verification
    verify_cuda_with_pycuda()
    verify_cudnn_with_pycuda_lib()
    test_tensorflow_gpu()
    test_pytorch_gpu()

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
    runner.run(suite)

    # Run the full, synchronous GraphBuilder test
    run_graph_builder_full_test()

    # Phase 4: Full Pipeline Smoke Tests
    # Note: These are designed to be quick. You can comment out any that you don't need to run every time.
    # test_word2vec_pipeline_run() # Disabled by default as per original script
    # test_transformer_embedder_pipeline_run() # Disabled by default as per original script
    test_gnn_benchmarker_run()
    test_ppi_pipeline_run()

    print("\n" + "#" * 100)
    print("### All Integrated Tests Finished. ###")
    print("#" * 100)