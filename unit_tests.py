# ==============================================================================
# MODULE: unit_tests.py
# PURPOSE: Unit and smoke tests
# VERSION: 1.2 (Pipeline tests uncommented, added dummy test functions)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import sys
import shutil # For cleaning up temp directories

import h5py
import numpy as np
import tensorflow as tf
import torch

# Local application imports
from src.config import Config
from src.utils.data_utils import DataUtils # For print_header in dummy tests

# --- Helper Functions for Dummy Pipeline Tests ---
def _create_dummy_fasta_for_testing(directory: str, filename: str = "dummy_test.fasta", num_seqs: int = 5):
    """Creates a small dummy FASTA file for testing purposes."""
    os.makedirs(directory, exist_ok=True)
    fasta_path = os.path.join(directory, filename)
    with open(fasta_path, "w") as f:
        for i in range(num_seqs):
            seq_id = f"dummy_prot_{i+1}"
            # Generate a short random-like sequence
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
        while len(pairs) < count:
            p1, p2 = np.random.choice(protein_ids, 2, replace=False)
            pairs.add(tuple(sorted((p1, p2))))
        df = pd.DataFrame(list(pairs), columns=['p1', 'p2'])
        df.to_csv(filepath, header=False, index=False)

    generate_pairs(pos_path, num_pairs)
    generate_pairs(neg_path, num_pairs) # For simplicity, negatives might overlap if not carefully managed
    return pos_path, neg_path

def _create_dummy_h5_embeddings_for_testing(directory: str, filename: str = "dummy_embeddings.h5",
                                           protein_ids: Optional[List[str]] = None, num_proteins: int = 20, dim: int = 10):
    """Creates a dummy H5 embedding file."""
    os.makedirs(directory, exist_ok=True)
    h5_path = os.path.join(directory, filename)
    if protein_ids is None:
        protein_ids = [f"P{i:03d}" for i in range(num_proteins)]
    with h5py.File(h5_path, 'w') as hf:
        for pid in protein_ids:
            hf.create_dataset(pid, data=np.random.rand(dim).astype(np.float32))
    return h5_path


# --- Standard Unit/Smoke Tests ---

def test_pytorch_gpu():
    """
    Checks the status of PyTorch's CUDA availability and prints diagnostic information,
    including cuDNN status.
    """
    print("--- PyTorch GPU Diagnostic ---")
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
            print("  [Success] PyTorch GPU tensor operations seem to be working.")
        except Exception as e:
            print(f"  [Error] Failed PyTorch GPU tensor operation: {e}")
    print("\n--- End of PyTorch GPU Diagnostic ---")


def test_tensorflow_gpu():
    """
    Checks for GPU availability in TensorFlow, performs a test operation,
    and mentions cuDNN.
    """
    print(f"\n--- TensorFlow GPU Test ---")
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
    print("\n--- TensorFlow GPU Test Complete ---")


def test_reporter():
    from src.utils.results_utils import EvaluationReporter
    print(f"\n--- EvaluationReporter Test ---")
    sample_k_vals = [10, 20]
    test_output_dir = "./temp_test_evaluation_reporter_output" # Unique name
    if os.path.exists(test_output_dir): shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    try:
        reporter = EvaluationReporter(base_output_dir=test_output_dir, k_vals_table=sample_k_vals)
        history1 = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.55, 0.42, 0.33],
                    'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.68, 0.78, 0.88]}
        reporter.plot_training_history(history1, "Model_A_Fold1")
        results_data = [
            {'embedding_name': 'Model_A', 'test_auc_sklearn': 0.92, 'test_f1_sklearn': 0.85,
             'test_precision_sklearn': 0.88, 'test_recall_sklearn': 0.82,
             'test_hits_at_10': 50, 'test_ndcg_at_10': 0.75, 'test_hits_at_20': 80, 'test_ndcg_at_20': 0.78,
             'test_auc_sklearn_std': 0.01, 'test_f1_sklearn_std': 0.02,
             'roc_data_representative': (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]), 0.92),
             'fold_auc_scores': [0.91, 0.93], 'fold_f1_scores': [0.84, 0.86]},
            {'embedding_name': 'Model_B', 'test_auc_sklearn': 0.88, 'test_f1_sklearn': 0.80,
             'test_precision_sklearn': 0.82, 'test_recall_sklearn': 0.78,
             'test_hits_at_10': 40, 'test_ndcg_at_10': 0.65, 'test_hits_at_20': 70, 'test_ndcg_at_20': 0.68,
             'test_auc_sklearn_std': 0.015, 'test_f1_sklearn_std': 0.022,
             'roc_data_representative': (np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), 0.88),
             'fold_auc_scores': [0.87, 0.89], 'fold_f1_scores': [0.79, 0.81]}]
        reporter.plot_roc_curves(results_data)
        reporter.plot_comparison_charts(results_data)
        reporter.write_summary_file(results_data, main_emb_name='Model_A', test_metric='test_auc_sklearn', alpha=0.05)
        print(f"  Example reporting complete. Check '{test_output_dir}' directory.")
    finally:
        if os.path.exists(test_output_dir): shutil.rmtree(test_output_dir)
    print(f"--- EvaluationReporter Test Complete ---")


def test_data_utilities():
    from src.utils.data_utils import DataLoader, DataUtils
    from src.utils.models_utils import EmbeddingLoader
    print(f"\n--- Data Utilities Test ---")
    config_instance = Config()
    temp_test_dir_base = "./temp_test_data_utilities_output" # Unique name
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
    from src.models.mlp import MLP
    print(f"\n--- MLPModelBuilder Build Test ---")
    config_instance = Config()
    dummy_mlp_params = {'dense1_units': 32, 'dropout1_rate': 0.1, 'dense2_units': 16, 'dropout2_rate': 0.1, 'l2_reg': 0.001}
    input_dim = 128
    try:
        mlp_builder = MLP(input_shape=input_dim, mlp_params=dummy_mlp_params, learning_rate=config_instance.EVAL_LEARNING_RATE)
        model = mlp_builder.build()
        assert model is not None, "MLP model build failed, model is None."
        assert model.input_shape == (None, input_dim), f"MLP input shape mismatch."
        model.summary(print_fn=lambda x: print(f"  {x}"))
        print(f"  MLPModelBuilder build test passed.")
    except Exception as e:
        print(f"  MLPModelBuilder build test FAILED: {e}")
        raise
    print(f"--- MLPModelBuilder Build Test Complete ---")

# --- Integration/Pipeline Smoke Tests ---
# These tests are more involved and might take longer.
# They are valuable for checking if major pipeline components run end-to-end.

def test_word2vec_pipeline_run():
    DataUtils.print_header("Word2Vec Pipeline Smoke Test")
    from src.pipeline.word2vec_embedder import Word2VecEmbedder
    config = Config()
    # Setup dummy FASTA for this test
    dummy_fasta_dir = "./temp_test_w2v_fasta_input"
    _create_dummy_fasta_for_testing(dummy_fasta_dir, "w2v_test.fasta")
    original_fasta_dir = config.W2V_INPUT_FASTA_DIR
    config.W2V_INPUT_FASTA_DIR = dummy_fasta_dir # Override
    config.APPLY_PCA_TO_W2V = False # Faster test
    config.W2V_EPOCHS = 1
    try:
        embedder = Word2VecEmbedder(config)
        embedder.run()
        print("  Word2VecEmbedder smoke test ran successfully (check output files).")
    except Exception as e:
        print(f"  Word2VecEmbedder smoke test FAILED: {e}")
        raise
    finally:
        config.W2V_INPUT_FASTA_DIR = original_fasta_dir # Restore
        if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
        # Clean up output files created by the embedder if necessary
        if os.path.exists(config.WORD2VEC_EMBEDDINGS_DIR):
            for f in os.listdir(config.WORD2VEC_EMBEDDINGS_DIR):
                if "w2v_test" in f or "dummy" in f: # Be more specific if needed
                    os.remove(os.path.join(config.WORD2VEC_EMBEDDINGS_DIR, f))
    print("--- Word2Vec Pipeline Smoke Test Complete ---")


def test_transformer_embedder_pipeline_run():
    DataUtils.print_header("Transformer Embedder Pipeline Smoke Test")
    from src.pipeline.transformer_embedder import TransformerEmbedder
    config = Config()
    dummy_fasta_dir = "./temp_test_transformer_fasta_input"
    _create_dummy_fasta_for_testing(dummy_fasta_dir, "transformer_test.fasta", num_seqs=2) # Small number for speed
    original_fasta_dir = config.TRANSFORMER_INPUT_FASTA_DIR
    config.TRANSFORMER_INPUT_FASTA_DIR = dummy_fasta_dir
    config.APPLY_PCA_TO_TRANSFORMER = False
    config.TRANSFORMER_BASE_BATCH_SIZE = 1 # Small batch for test
    original_models_to_run = config.TRANSFORMER_MODELS_TO_RUN
    # Use a very small, fast model if available, or just run ProtBERT for 1-2 seqs
    # For now, we'll use the configured ProtBERT but it might be slow.
    # config.TRANSFORMER_MODELS_TO_RUN = [{"name": "DummyTransformer", "hf_id": "prajjwal1/bert-tiny", "is_t5": False, "batch_size_multiplier": 1}]


    try:
        embedder = TransformerEmbedder(config)
        embedder.run()
        print("  TransformerEmbedder smoke test ran successfully (check output files).")
    except Exception as e:
        print(f"  TransformerEmbedder smoke test FAILED: {e}")
        raise
    finally:
        config.TRANSFORMER_INPUT_FASTA_DIR = original_fasta_dir
        config.TRANSFORMER_MODELS_TO_RUN = original_models_to_run
        if os.path.exists(dummy_fasta_dir): shutil.rmtree(dummy_fasta_dir)
        if os.path.exists(config.TRANSFORMER_EMBEDDINGS_DIR):
             for f in os.listdir(config.TRANSFORMER_EMBEDDINGS_DIR):
                if "transformer_test" in f or "DummyTransformer" in f:
                    os.remove(os.path.join(config.TRANSFORMER_EMBEDDINGS_DIR, f))
    print("--- Transformer Embedder Pipeline Smoke Test Complete ---")


def test_gnn_benchmarker_run():
    DataUtils.print_header("GNN Benchmarker Smoke Test")
    from src.benchmarks.gnn_benchmarker import GNNBenchmarker
    config = Config()
    # Override datasets to run only KarateClub for a quick test
    original_datasets = config.BENCHMARK_NODE_CLASSIFICATION_DATASETS
    config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = ["KarateClub"]
    config.EVAL_EPOCHS = 2 # Very few epochs for test
    config.BENCHMARK_SAVE_EMBEDDINGS = False # Don't save for quick test
    config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS = False
    try:
        benchmarker = GNNBenchmarker(config)
        benchmarker.run()
        print("  GNNBenchmarker smoke test ran successfully (check output files).")
    except Exception as e:
        print(f"  GNNBenchmarker smoke test FAILED: {e}")
        raise
    finally:
        config.BENCHMARK_NODE_CLASSIFICATION_DATASETS = original_datasets # Restore
        # Clean up benchmark results for KarateClub if needed
        karate_summary = os.path.join(config.BENCHMARKING_RESULTS_DIR, "benchmark_summary_KarateClub.csv")
        if os.path.exists(karate_summary): os.remove(karate_summary)
        karate_dir = os.path.join(config.BENCHMARKING_RESULTS_DIR, "KarateClub")
        if os.path.exists(karate_dir): shutil.rmtree(karate_dir)
        # Clean downloaded dataset if it's in a known temp location (PyG usually handles this well)
        pyg_dataset_path = os.path.join(config.BASE_DATA_DIR, "standard_datasets_pyg", "KarateClub")
        if os.path.exists(pyg_dataset_path): shutil.rmtree(pyg_dataset_path)

    print("--- GNN Benchmarker Smoke Test Complete ---")


def test_ppi_pipeline_run():
    DataUtils.print_header("PPI Pipeline (Dummy Run) Smoke Test")
    from src.pipeline.ppi_main import PPIPipeline
    config = Config()
    original_dummy_flag = config.RUN_DUMMY_TEST
    config.RUN_DUMMY_TEST = True # Force dummy for this test
    config.EVAL_EPOCHS = 1 # Minimal epochs for dummy
    config.EVAL_N_FOLDS = 1 # Minimal folds
    try:
        evaluator = PPIPipeline(config)
        evaluator.run(use_dummy_data=True) # Explicitly use dummy data
        print("  PPIPipeline (dummy run) smoke test ran successfully (check dummy_run_output).")
    except Exception as e:
        print(f"  PPIPipeline (dummy run) smoke test FAILED: {e}")
        raise
    finally:
        config.RUN_DUMMY_TEST = original_dummy_flag # Restore
        if config.CLEANUP_DUMMY_DATA:
            dummy_output = os.path.join(config.EVALUATION_RESULTS_DIR, "dummy_run_output")
            if os.path.exists(dummy_output): shutil.rmtree(dummy_output)
            dummy_data_temp = os.path.join(config.BASE_OUTPUT_DIR, "dummy_data_temp")
            if os.path.exists(dummy_data_temp): shutil.rmtree(dummy_data_temp)
    print("--- PPI Pipeline (Dummy Run) Smoke Test Complete ---")


if __name__ == "__main__":
    print("Starting All Unit Tests / Smoke Tests...\n")
    # Environment Checks
    test_tensorflow_gpu()
    test_pytorch_gpu()

    # Utility Tests
    test_reporter()
    test_data_utilities()

    # Model Build Tests
    test_mlp_model_build()

    # --- Integration/Pipeline Smoke Tests ---
    # These are uncommented to be part of the standard test run.
    # They are designed to be relatively quick by using dummy data or minimal configs.
    print("\n--- Starting Integration/Pipeline Smoke Tests ---")
    # Note: These tests might create temporary files/directories.
    # Ensure cleanup logic within each test function is robust.

    test_word2vec_pipeline_run()
    test_transformer_embedder_pipeline_run()
    test_gnn_benchmarker_run()
    test_ppi_pipeline_run()

    print("\nAll Unit Tests / Smoke Tests Finished.")
