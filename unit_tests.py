import torch
import tensorflow as tf
import sys
from src.utils.results_utils import EvaluationReporter
import numpy as np


def test_pytorch_gpu():
    """
    Checks for GPU availability in PyTorch and performs a test operation.
    """
    print(f"--- PyTorch GPU Test ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")

    # Step 1: Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()

    # Step 2: Check if any GPUs were detected
    if is_cuda_available:
        gpu_count = torch.cuda.device_count()
        current_device_id = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device_id)

        print(f"\n✅ GPU(s) found! Total devices: {gpu_count}")
        print(f"  - Current Device ID: {current_device_id}")
        print(f"  - Current Device Name: {current_device_name}")

        # Run a simple operation on the GPU to confirm it's working
        try:
            # Set the device to the first GPU
            device = torch.device("cuda:0")
            print(f"\n--- Performing a test matrix multiplication on {device} ---")

            # Create two random tensors and move them to the GPU
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

            # Perform matrix multiplication
            c = torch.matmul(a, b)

            # Move the result back to the CPU for printing if needed
            print("Matrix A on GPU:\n", a)
            print("Matrix B on GPU:\n", b)
            print("Result of A * B on GPU:\n", c.cpu().numpy())
            print("\n🎉 GPU is set up correctly and operational!")

        except Exception as e:
            print(f"\n❌ An error occurred while trying to use the GPU: {e}")
            print("   Please check your CUDA installation and driver compatibility.")

    else:
        print("\n❌ No CUDA-enabled GPU detected by PyTorch.")
        print("   PyTorch will run on the CPU. If you have a GPU, please check:")
        print("   1. NVIDIA drivers are correctly installed.")
        print("   2. You have installed a PyTorch version with CUDA support.")
        print("      (e.g., 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')")
        print("   3. Your installed CUDA Toolkit version is compatible with your PyTorch build.")

    print("\n--- Test Complete ---")


def test_tensorflow_gpu():
    """
    Checks for GPU availability in TensorFlow and performs a test operation.
    """
    print(f"--- TensorFlow GPU Test ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")

    # Step 1: List physical devices and filter for GPUs
    gpus = tf.config.list_physical_devices('GPU')

    # Step 2: Check if any GPUs were detected
    if gpus:
        print(f"\n✅ GPU(s) found! Total devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")

        # Run a simple operation on the first GPU to confirm it's working
        try:
            # Explicitly place the operations on the first GPU
            with tf.device('/GPU:0'):
                print("\n--- Performing a test matrix multiplication on GPU:0 ---")
                # Create two random matrices
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)

                # Perform matrix multiplication
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

    print("\n--- Test Complete ---")


def test_reporter():
    # Dummy data for demonstration
    sample_k_vals = [10, 20]
    reporter = EvaluationReporter(base_output_dir="./evaluation_output_example", k_vals_table=sample_k_vals)

    # Dummy history for one model
    history1 = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.55, 0.42, 0.33], 'accuracy': [0.7, 0.8, 0.9], 'val_accuracy': [0.68, 0.78, 0.88]}
    reporter.plot_training_history(history1, "Model_A_Fold1")

    # Dummy results list for other plots/summary
    results_data = [
        {'embedding_name': 'Model_A', 'test_auc_sklearn': 0.92, 'test_f1_sklearn': 0.85, 'test_precision_sklearn': 0.88, 'test_recall_sklearn': 0.82, 'test_hits_at_10': 50, 'test_ndcg_at_10': 0.75, 'test_hits_at_20': 80,
         'test_ndcg_at_20': 0.78, 'test_auc_sklearn_std': 0.01, 'test_f1_sklearn_std': 0.02, 'roc_data_representative': (np.array([0, 0.1, 1]), np.array([0, 0.8, 1]), []), 'fold_auc_scores': [0.91, 0.93],
         'fold_f1_scores': [0.84, 0.86]},
        {'embedding_name': 'Model_B', 'test_auc_sklearn': 0.88, 'test_f1_sklearn': 0.80, 'test_precision_sklearn': 0.82, 'test_recall_sklearn': 0.78, 'test_hits_at_10': 40, 'test_ndcg_at_10': 0.65, 'test_hits_at_20': 70,
         'test_ndcg_at_20': 0.68, 'test_auc_sklearn_std': 0.015, 'test_f1_sklearn_std': 0.022, 'roc_data_representative': (np.array([0, 0.2, 1]), np.array([0, 0.7, 1]), []), 'fold_auc_scores': [0.87, 0.89],
         'fold_f1_scores': [0.79, 0.81]}]
    reporter.plot_roc_curves(results_data)
    reporter.plot_comparison_charts(results_data)
    reporter.write_summary_file(results_data, main_emb_name='Model_A', test_metric='test_auc_sklearn', alpha=0.05)

    print("Example reporting complete. Check './evaluation_output_example' directory.")

def test_data_loaders():
    from src.utils.data_utils import H5EmbeddingLoader, InteractionLoader, FastaParser, DataUtils

    # Example: Loading H5 embeddings
    with H5EmbeddingLoader("path/to/embeddings.h5") as loader:
        if "protein_X" in loader:
            embedding = loader["protein_X"]
            # print(f"All keys: {loader.get_keys()}")

    # Example: Loading interaction pairs
    positive_pairs = InteractionLoader.load_interaction_pairs("path/to/positive.csv", label=1, sample_n=1000)
    ids = InteractionLoader.get_required_ids_from_files(["path/to/positive.csv", "path/to/negative.csv"])

    # Example: Streaming interaction pairs
    for batch in InteractionLoader.stream_interaction_pairs("path/to/interactions.tsv", label=1, batch_size=128):
        # process batch
        pass

    # Example: Parsing a FASTA file
    for prot_id, sequence in FastaParser.parse_sequences("path/to/sequences.fasta"):
        # process sequence
        pass

    # Example: Using DataUtils
    DataUtils.print_header("Starting Analysis")
    my_data = {"a": 1}
    DataUtils.save_object(my_data, "output/my_data.pkl")
    loaded_data = DataUtils.load_object("output/my_data.pkl")
    DataUtils.check_h5_embeddings_integrity("path/to/some_embeddings.h5")


    from src.utils.data_utils import FastaParser
    from src.config import Config  # Your Config class

    config_instance = Config()  # Initialize your configuration

    # Create an instance of FastaParser with the configuration
    parser_mapper = FastaParser(config=config_instance)

    # Generate ID maps
    id_map_dictionary = parser_mapper.generate_id_maps()

    # If you also need to parse sequences using the same instance (though parse_sequences is static)
    # for prot_id, sequence in parser_mapper.parse_sequences(config_instance.GCN_INPUT_FASTA_PATH):
    #     # ... process ...

def test_word2vec_model():
    from src.config import Config
    from src.pipeline.word2vec_embedder import Word2VecEmbedderPipeline

    config_instance = Config()
    # Need to ensure DataUtils is accessible if used as below, or import it.
    # For simplicity, if DataUtils is part of DataLoader, you might not call it directly here
    # or ensure DataLoader is instantiated if DataUtils methods are not static.
    # However, DataUtils.print_header is static.
    from src.utils.data_utils import DataUtils # Assuming DataUtils is directly importable or part of DataLoader

    word2vec_pipeline = Word2VecEmbedderPipeline(config_instance)
    word2vec_pipeline.run_pipeline()

def test_transformer_model():
    from src.config import Config
    from src.utils.data_utils import DataUtils # Ensure DataUtils is imported
    from src.pipeline.transformer_embedder import TransformerEmbedderPipeline

    config_instance = Config()
    transformer_pipeline = TransformerEmbedderPipeline(config_instance)
    transformer_pipeline.run_pipeline()


def test_gnn_benchmarker():
    from src.config import Config
    from src.benchmarks.gnn_benchmarker import GNNBenchmarker
    config_instance = Config()
    benchmarker = GNNBenchmarker(config_instance)
    benchmarker.run()


if __name__ == "__main__":
    test_tensorflow_gpu()
    test_pytorch_gpu()
    test_reporter()
