import os  # Added for path joining in test_reporter
import sys

import h5py
import numpy as np
import tensorflow as tf
import torch

# Local application imports
from src.config import Config  # Assuming this is the correct path


# It's good practice to group imports
# Standard library imports first, then third-party, then local application.

# It's better to import specific classes/functions needed for each test
# within the test function or at the top if widely used and clearly named.
# For this file, since it's a collection of tests for different parts,
# some imports will be specific to test functions.

def test_pytorch_gpu():
    """
    Checks the status of PyTorch's CUDA availability and prints diagnostic information.
    """
    # print("--- PyTorch GPU Diagnostic ---")
    # print(f"Python Version: {sys.version}")
    # print(f"PyTorch Version: {torch.__version__}")

    # # The main check for CUDA availability
    # is_available = torch.cuda.is_available()
    # print(f"\nIs CUDA available? -> {is_available}")
    #
    # if not is_available:
    #     print("\n[Error] PyTorch cannot find a CUDA-enabled GPU.")
    #     print("This may be due to a driver issue or an incorrect PyTorch installation.")
    # else:
    #     print("\n[Success] PyTorch has detected a CUDA-enabled GPU.")
    #     # Print details about the CUDA and GPU setup
    #     print(f"CUDA Version PyTorch was built with: {torch.version.cuda}")
    #     device_count = torch.cuda.device_count()
    #     print(f"Number of GPUs found: {device_count}")
    #     for i in range(device_count):
    #         print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. Check if the GPU is available (we know it is, but this is best practice)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available! Using the GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using the CPU.")

    # 2. Create a tensor on the CPU by default
    cpu_tensor = torch.randn(3, 3)
    print("\nTensor on CPU:")
    print(cpu_tensor)
    print(f"Device: {cpu_tensor.device}")

    # 3. Move the tensor to the GPU using .to(device)
    gpu_tensor = cpu_tensor.to(device)
    print("\nTensor on GPU:")
    print(gpu_tensor)
    print(f"Device: {gpu_tensor.device}")  # This will now say 'cuda:0'

    # Now any operations on gpu_tensor will happen on the GPU
    gpu_result = gpu_tensor * gpu_tensor
    print("\nResult of computation on GPU:")
    print(gpu_result)

    print("\n--- End of Diagnostic ---")

# def test_pytorch_gpu():
#     """
#     Checks for GPU availability in PyTorch and performs a test operation.
#     """
#     print(f"--- PyTorch GPU Test ---")
#     print(f"PyTorch Version: {torch.__version__}")
#     print(f"Python Version: {sys.version}")
#
#     is_cuda_available = torch.cuda.is_available()
#     if is_cuda_available:
#         gpu_count = torch.cuda.device_count()
#         current_device_id = torch.cuda.current_device()
#         current_device_name = torch.cuda.get_device_name(current_device_id)
#
#         print(f"\n✅ GPU(s) found! Total devices: {gpu_count}")
#         print(f"  - Current Device ID: {current_device_id}")
#         print(f"  - Current Device Name: {current_device_name}")
#
#         try:
#             device = torch.device("cuda:0")
#             print(f"\n--- Performing a test matrix multiplication on {device} ---")
#             a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
#             b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
#             c = torch.matmul(a, b)
#             print("Matrix A on GPU:\n", a)
#             print("Matrix B on GPU:\n", b)
#             print("Result of A * B on GPU:\n", c.cpu().numpy())
#             print("\n🎉 GPU is set up correctly and operational!")
#         except Exception as e:
#             print(f"\n❌ An error occurred while trying to use the GPU: {e}")
#             print("   Please check your CUDA installation and driver compatibility.")
#     else:
#         print("\n❌ No CUDA-enabled GPU detected by PyTorch.")
#         print("   PyTorch will run on the CPU. If you have a GPU, please check:")
#         print("   1. NVIDIA drivers are correctly installed.")
#         print("   2. You have installed a PyTorch version with CUDA support.")
#         print("      (e.g., 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')")
#         print("   3. Your installed CUDA Toolkit version is compatible with your PyTorch build.")
#     print("\n--- PyTorch GPU Test Complete ---")


def test_tensorflow_gpu():
    """
    Checks for GPU availability in TensorFlow and performs a test operation.
    """
    print(f"--- TensorFlow GPU Test ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU(s) found! Total devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
        try:
            with tf.device('/GPU:0'):
                print("\n--- Performing a test matrix multiplication on GPU:0 ---")
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
                c = tf.matmul(a, b)
            print("Matrix A:\n", a.numpy())
            print("Matrix B:\n", b.numpy())
            print("Result of A * B on GPU:\n", c.numpy())
            print("\n🎉 GPU is set up correctly and operational!")
        except Exception as e:
            print(f"\n❌ An error occurred while trying to use the GPU: {e}")
            print("   Please check your CUDA and cuDNN installation.")
    else:
        print("\n❌ No GPU detected by TensorFlow.")
        print("   TensorFlow will run on the CPU. If you have a GPU, please check:")
        print("   1. NVIDIA drivers are correctly installed.")
        print("   2. CUDA Toolkit version is compatible with your TensorFlow version.")
        print("   3. cuDNN libraries are correctly installed and in the system's PATH.")
    print("\n--- TensorFlow GPU Test Complete ---")


def test_reporter():
    from src.utils.results_utils import EvaluationReporter  # Moved import here
    print(f"--- EvaluationReporter Test ---")
    # Dummy data for demonstration
    sample_k_vals = [10, 20]
    # Create a temporary directory for test outputs
    test_output_dir = "./temp_evaluation_output_example"
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    reporter = EvaluationReporter(base_output_dir=test_output_dir, k_vals_table=sample_k_vals)

    history1 = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.55, 0.42, 0.33],
                'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.68, 0.78, 0.88]}
    reporter.plot_training_history(history1, "Model_A_Fold1")

    results_data = [
        {'embedding_name': 'Model_A', 'test_auc_sklearn': 0.92, 'test_f1_sklearn': 0.85,
         'test_precision_sklearn': 0.88, 'test_recall_sklearn': 0.82,
         'test_hits_at_10': 50, 'test_ndcg_at_10': 0.75,
         'test_hits_at_20': 80, 'test_ndcg_at_20': 0.78,
         'test_auc_sklearn_std': 0.01, 'test_f1_sklearn_std': 0.02,
         'roc_data_representative': (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]), 0.92),  # Corrected: Added AUC score
         'fold_auc_scores': [0.91, 0.93], 'fold_f1_scores': [0.84, 0.86]},
        {'embedding_name': 'Model_B', 'test_auc_sklearn': 0.88, 'test_f1_sklearn': 0.80,
         'test_precision_sklearn': 0.82, 'test_recall_sklearn': 0.78,
         'test_hits_at_10': 40, 'test_ndcg_at_10': 0.65,
         'test_hits_at_20': 70, 'test_ndcg_at_20': 0.68,
         'test_auc_sklearn_std': 0.015, 'test_f1_sklearn_std': 0.022,
         'roc_data_representative': (np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), 0.88),  # Corrected: Added AUC score
         'fold_auc_scores': [0.87, 0.89], 'fold_f1_scores': [0.79, 0.81]}]
    reporter.plot_roc_curves(results_data)
    reporter.plot_comparison_charts(results_data)
    reporter.write_summary_file(results_data, main_emb_name='Model_A', test_metric='test_auc_sklearn', alpha=0.05)

    print(f"Example reporting complete. Check '{test_output_dir}' directory.")
    print(f"--- EvaluationReporter Test Complete ---")
    # Consider cleaning up test_output_dir after test if not needed.


def test_data_utilities():
    # Assuming DataLoader, GroundTruthLoader, DataUtils are in data_loader.py
    # and EmbeddingLoader is in models_utils.py
    from src.utils.data_utils import DataLoader, DataUtils
    from src.utils.models_utils import EmbeddingLoader

    print(f"--- Data Utilities Test ---")
    config_instance = Config()

    # Note: The following examples use placeholder paths.
    # In a real unit test, you would mock file system interactions or use temporary files.
    # These will likely raise FileNotFoundError if run as is without actual files.

    # Example: Loading H5 embeddings
    print("\nTesting EmbeddingLoader (will likely fail without a real H5 file at 'dummy_path/embeddings.h5'):")
    try:
        # Create a dummy H5 file for the test to pass this part
        dummy_h5_path = "./temp_dummy_embeddings.h5"
        with h5py.File(dummy_h5_path, 'w') as hf:
            hf.create_dataset("protein_X", data=np.random.rand(10))
        with EmbeddingLoader(dummy_h5_path) as loader:
            if "protein_X" in loader:
                embedding = loader["protein_X"]
                print(f"  Successfully loaded dummy embedding for protein_X, shape: {embedding.shape}")
            # print(f"All keys: {loader.get_keys()}")
        os.remove(dummy_h5_path)  # Clean up
    except Exception as e:
        print(f"  EmbeddingLoader test part failed as expected or due to error: {e}")

    # Example: Loading interaction pairs (GroundTruthLoader)
    print("\nTesting GroundTruthLoader (will fail without real CSV/TSV files):")
    # positive_pairs = GroundTruthLoader.load_interaction_pairs("dummy_path/positive.csv", label=1, sample_n=10)
    # ids = GroundTruthLoader.get_required_ids_from_files(["dummy_path/positive.csv", "dummy_path/negative.csv"])
    # for batch in GroundTruthLoader.stream_interaction_pairs("dummy_path/interactions.tsv", label=1, batch_size=5):
    #     pass # process batch
    print("  (Skipping actual file operations for GroundTruthLoader due to placeholder paths)")

    # Example: Parsing a FASTA file (DataLoader)
    print("\nTesting DataLoader.parse_sequences (will fail without a real FASTA file):")
    # for prot_id, sequence in DataLoader.parse_sequences("dummy_path/sequences.fasta"):
    #     pass # process sequence
    print("  (Skipping actual file operation for DataLoader.parse_sequences due to placeholder path)")

    # Example: Using DataUtils
    print("\nTesting DataUtils:")
    DataUtils.print_header("Starting DataUtils Test Section")
    temp_data_dir = "./temp_data_utils_output"
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)
    my_data_path = os.path.join(temp_data_dir, "my_data.pkl")

    my_data = {"a": 1, "b": [1, 2, 3]}
    DataUtils.save_object(my_data, my_data_path)
    loaded_data = DataUtils.load_object(my_data_path)
    assert my_data == loaded_data, "DataUtils save/load failed."
    print(f"  DataUtils save_object and load_object test passed.")
    # DataUtils.check_h5_embeddings_integrity("dummy_path/some_embeddings.h5") # Needs a real H5
    print("  (Skipping H5 integrity check due to placeholder path)")

    # Test DataLoader ID mapping
    print("\nTesting DataLoader ID mapping:")
    # This part assumes config_instance.GCN_INPUT_FASTA_PATH points to a readable (even if small/dummy) FASTA
    # and config_instance.ID_MAPPING_MODE is set (e.g., 'regex').
    # The actual mapping might not produce results if the FASTA is too simple or mode is 'none'.
    try:
        # Ensure the FASTA file directory exists for the dummy FASTA
        dummy_fasta_dir = os.path.dirname(config_instance.GCN_INPUT_FASTA_PATH)
        if not os.path.exists(dummy_fasta_dir):
            os.makedirs(dummy_fasta_dir, exist_ok=True)
        # Create a minimal dummy FASTA if it doesn't exist, to prevent FileNotFoundError
        if not os.path.exists(config_instance.GCN_INPUT_FASTA_PATH):
            with open(config_instance.GCN_INPUT_FASTA_PATH, 'w') as f:
                f.write(">sp|P12345|TEST_HUMAN Test protein\n")
                f.write("ACDEFGHIKLMNPQRSTVWY\n")
            print(f"  Created dummy FASTA at {config_instance.GCN_INPUT_FASTA_PATH} for ID mapping test.")

        parser_mapper = DataLoader(config=config_instance)
        id_map_dictionary = parser_mapper.generate_id_maps()
        print(f"  DataLoader generate_id_maps called. Number of mappings: {len(id_map_dictionary)}")
        # Add assertions here if you expect specific mappings from your dummy FASTA
    except Exception as e:
        print(f"  DataLoader ID mapping test encountered an error: {e}")

    # Clean up temporary directory
    if os.path.exists(temp_data_dir):
        import shutil
        shutil.rmtree(temp_data_dir)

    print(f"--- Data Utilities Test Complete ---")


def test_word2vec_pipeline_run():  # Renamed for clarity
    from src.pipeline.word2vec_embedder import Word2VecEmbedder  # Updated class name
    # from src.utils.data_loader import DataUtils # DataUtils is used internally by the pipeline

    print(f"--- Word2VecEmbedder Pipeline Run Test ---")
    config_instance = Config()
    # This is an integration test. It will run the full Word2Vec pipeline.
    # Ensure W2V_INPUT_FASTA_DIR in config points to a directory with a small dummy FASTA for testing.
    # Example: Create a dummy FASTA if it doesn't exist
    dummy_w2v_fasta_dir = config_instance.W2V_INPUT_FASTA_DIR
    if not os.path.exists(dummy_w2v_fasta_dir):
        os.makedirs(dummy_w2v_fasta_dir, exist_ok=True)
    dummy_fasta_file = os.path.join(dummy_w2v_fasta_dir, "dummy_w2v_test.fasta")
    if not os.path.exists(dummy_fasta_file):
        with open(dummy_fasta_file, "w") as f:
            f.write(">protein1\nACGT\n>protein2\nGTCA\n")
        print(f"  Created dummy FASTA for Word2Vec test: {dummy_fasta_file}")

    word2vec_pipeline = Word2VecEmbedder(config_instance)
    word2vec_pipeline.run()  # Assuming the method is now .run()
    print(f"--- Word2VecEmbedder Pipeline Run Test Complete ---")


def test_transformer_embedder_pipeline_run():  # Renamed for clarity
    from src.pipeline.transformer_embedder import TransformerEmbedder  # Updated class name
    # from src.utils.data_loader import DataUtils

    print(f"--- TransformerEmbedder Pipeline Run Test ---")
    config_instance = Config()
    # This is an integration test.
    # Ensure TRANSFORMER_INPUT_FASTA_DIR in config points to a small dummy FASTA.
    dummy_transformer_fasta_dir = config_instance.TRANSFORMER_INPUT_FASTA_DIR
    if not os.path.exists(dummy_transformer_fasta_dir):
        os.makedirs(dummy_transformer_fasta_dir, exist_ok=True)
    dummy_fasta_file = os.path.join(dummy_transformer_fasta_dir, "dummy_transformer_test.fasta")
    if not os.path.exists(dummy_fasta_file):
        with open(dummy_fasta_file, "w") as f:
            f.write(">protein1\nACGT\n>protein2\nGTCA\n")
        print(f"  Created dummy FASTA for Transformer test: {dummy_fasta_file}")

    transformer_pipeline = TransformerEmbedder(config_instance)
    transformer_pipeline.run()
    print(f"--- TransformerEmbedder Pipeline Run Test Complete ---")


def test_gnn_benchmarker_run():  # Renamed for clarity
    from src.benchmarks.gnn_benchmarker import GNNBenchmarker

    print(f"--- GNNBenchmarker Run Test ---")
    config_instance = Config()
    # This is an integration test. It will download PPI dataset if not present.
    # Ensure config.BASE_DATA_DIR is writable.
    benchmarker = GNNBenchmarker(config_instance)
    benchmarker.run()
    print(f"--- GNNBenchmarker Run Test Complete ---")


def test_ppi_pipeline_run():  # Renamed for clarity
    from src.pipeline.ppi_main import PPIPipeline

    print(f"--- PPIPipeline Run Test ---")
    config_instance = Config()
    # This is an integration test.
    # It will use dummy data if config.RUN_DUMMY_TEST is True.
    evaluator = PPIPipeline(config_instance)
    evaluator.run(use_dummy_data=config_instance.RUN_DUMMY_TEST)
    print(f"--- PPIPipeline Run Test Complete ---")


def test_mlp_model_build():  # Renamed for clarity
    from src.models.mlp import MLP  # Updated class name

    print(f"--- MLPModelBuilder Build Test ---")
    config_instance = Config()  # Needed for EVAL_LEARNING_RATE
    dummy_mlp_params = {
        'dense1_units': 32, 'dropout1_rate': 0.1,
        'dense2_units': 16, 'dropout2_rate': 0.1,
        'l2_reg': 0.001
    }
    input_dim = 128

    mlp_builder = MLP(
        input_shape=input_dim,
        mlp_params=dummy_mlp_params,
        learning_rate=config_instance.EVAL_LEARNING_RATE  # Get from config
    )
    model = mlp_builder.build()
    assert model is not None, "MLP model build failed, model is None."
    assert model.input_shape == (None, input_dim), "MLP model input shape mismatch."
    print(model.summary())
    print(f"  MLPModelBuilder build test passed.")
    print(f"--- MLPModelBuilder Build Test Complete ---")


if __name__ == "__main__":
    print("Starting All Unit Tests / Smoke Tests...\n")
    # Environment Checks
    # test_tensorflow_gpu()
    test_pytorch_gpu()

    # # Utility Tests
    # test_reporter()
    # test_data_utilities()  # This test has parts that might fail without real files or further mocking
    #
    # # Model Build Tests
    # test_mlp_model_build()

    # Pipeline Run Tests (Integration/Smoke Tests)
    # These can be time-consuming and require proper configuration (e.g., dummy data paths)
    # test_word2vec_pipeline_run()
    # test_transformer_embedder_pipeline_run()
    # test_gnn_benchmarker_run() # This will download PPI dataset
    # test_ppi_pipeline_run()

    print("\nAll Unit Tests / Smoke Tests Finished.")
