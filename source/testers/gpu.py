# ==============================================================================
# MODULE: testers/gpu.py
# PURPOSE: Contains consolidated tests for GPU and CUDA environment verification.
# VERSION: 2.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import torch
import tensorflow as tf
import numpy as np
from source.utils.data.data_utils import DataUtils


class GPUTests:
    """A class to encapsulate all GPU, CUDA, and framework verification tests."""

    @staticmethod
    def _test_pytorch_gpu() -> bool:
        """
        Checks the status of PyTorch's CUDA availability and performs a test operation.
        Returns True on success, False on failure.
        """
        print("\n--- Verifying PyTorch ---")
        try:
            print(f"PyTorch Version: {torch.__version__}")
            if not torch.cuda.is_available():
                print("  ❌ [Error] PyTorch cannot find a CUDA-enabled GPU.")
                return False

            print(f"  ✅ [Success] PyTorch has detected a CUDA-enabled GPU.")
            print(f"  CUDA Version PyTorch was built with: {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"  Number of GPUs found: {device_count}")
            for i in range(device_count):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")

            device = torch.device("cuda")
            tensor = torch.randn(3, 3).to(device)
            _ = tensor * tensor
            print("  ✅ [Success] PyTorch GPU tensor operation completed.")
            return True
        except Exception as e:
            print(f"  ❌ [Error] An unexpected error occurred during PyTorch verification: {e}")
            return False

    @staticmethod
    def _test_tensorflow_gpu() -> bool:
        """
        Checks for GPU availability in TensorFlow and performs a test operation.
        Returns True on success, False on failure.
        """
        print("\n--- Verifying TensorFlow ---")
        try:
            print(f"TensorFlow Version: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("  ❌ [Error] TensorFlow cannot find a CUDA-enabled GPU.")
                return False

            print(f"  ✅ GPU(s) found! Total devices: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
                c = tf.matmul(a, b)
            _ = c.numpy()
            print("  ✅ [Success] TensorFlow GPU tensor operation completed.")
            return True
        except Exception as e:
            print(f"  ❌ [Error] An unexpected error occurred during TensorFlow verification: {e}")
            return False

    @staticmethod
    def _verify_cudnn_library() -> bool:
        """
        Verifies that the cuDNN library is accessible to PyTorch.
        Returns True on success, False on failure.
        """
        print("\n--- Verifying cuDNN Library ---")
        try:
            if not torch.backends.cudnn.is_available():
                print("  ❌ [Error] PyTorch backend reports cuDNN is NOT available.")
                return False

            version = torch.backends.cudnn.version()
            print(f"  ✅ [Success] PyTorch backend reports cuDNN is available.")
            print(f"  cuDNN Version reported by PyTorch: {version}")
            return True
        except Exception as e:
            print(f"  ❌ [Error] An unexpected error occurred during cuDNN verification: {e}")
            return False

    @staticmethod
    def verify_cuda_with_pycuda():
        """
        Verifies basic CUDA functionality by performing a matrix multiplication on the GPU using PyCUDA.
        """
        print("\n" + "=" * 80)
        DataUtils.print_header("CUDA Verification with PyCUDA")
        print("=" * 80)
        try:
            import pycuda.autoinit
            import pycuda.driver as drv
            import pycuda.gpuarray as gpuarray
            import pycuda.tools

            pycuda.tools.clear_context_caches()
            device = drv.Device(0)
            print(f"Successfully selected GPU 0: {device.name()}")

            matrix_a_cpu = np.random.randn(512, 1024).astype(np.float32)
            matrix_b_cpu = np.random.randn(512, 1024).astype(np.float32)
            matrix_a_gpu = gpuarray.to_gpu(matrix_a_cpu)
            matrix_b_gpu = gpuarray.to_gpu(matrix_b_cpu)
            result_gpu = matrix_a_gpu + matrix_b_gpu
            result_cpu = result_gpu.get()

            print(f"\nVerification successful! A small subset of the result tensor:\n{result_cpu[:2, :2]}")
        except Exception as e:
            print("Error: Could not find or initialize a CUDA-enabled GPU with PyCUDA.")
            print(f"Details: {e}")

    @staticmethod
    def verify_full_gpu_environment() -> bool:
        """
        A comprehensive test that calls all individual framework verification methods.
        """
        print("\n" + "=" * 80)
        DataUtils.print_header("Comprehensive GPU Environment Verification")
        print("=" * 80)

        # Call the individual, consolidated verification functions
        pytorch_ok = GPUTests._test_pytorch_gpu()
        tensorflow_ok = GPUTests._test_tensorflow_gpu()
        cudnn_ok = GPUTests._verify_cudnn_library()

        # Also run the PyCUDA specific test for completeness
        GPUTests.verify_cuda_with_pycuda()

        all_ok = pytorch_ok and tensorflow_ok and cudnn_ok

        # --- Final Summary ---
        print("\n" + "-" * 40)
        if all_ok:
            print("✅ GPU Environment Verification Passed for all frameworks.")
        else:
            print("❌ GPU Environment Verification FAILED. Please check the errors above.")
        print("-" * 40)
        return all_ok