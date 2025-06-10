import torch
import tensorflow as tf
import sys


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


if __name__ == "__main__":
    test_tensorflow_gpu()
    test_pytorch_gpu()