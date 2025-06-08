# ==============================================================================
# MODULE: utils/diagnostics.py
# PURPOSE: Contains functions for checking data integrity and environment setup,
#          such as verifying HDF5 files and checking for GPU availability.
# ==============================================================================

import os
import h5py
import numpy as np
import random
import tensorflow as tf
from typing import List


def check_h5_embeddings(h5_filepath: str, num_samples_to_check: int = 5):
    """
    Reads an HDF5 embedding file to check its integrity and properties.
    (Adapted from check_h5.py)
    """
    print(f"\n--- Checking HDF5 file: {os.path.basename(h5_filepath)} ---")
    if not os.path.exists(h5_filepath):
        print(f"Error: File not found at '{h5_filepath}'")
        return
    if not h5py.is_hdf5(h5_filepath):
        print(f"Error: File at '{h5_filepath}' is not a valid HDF5 file.")
        return

    try:
        with h5py.File(h5_filepath, 'r') as hf:
            keys = list(hf.keys())
            if not keys:
                print("HDF5 check: File is empty or contains no embeddings.")
                return

            print(f"Found {len(keys)} total embeddings. Inspecting up to {num_samples_to_check} samples:")
            sample_keys = random.sample(keys, min(len(keys), num_samples_to_check))

            for i, key in enumerate(sample_keys):
                dataset = hf.get(key)
                if not isinstance(dataset, h5py.Dataset):
                    print(f"  - Key '{key}' is not a valid HDF5 Dataset.")
                    continue

                emb = dataset[:]
                print(f"  - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")

                if emb.ndim != 1:
                    print(f"    - Note: Embedding is not a 1D vector (ndim={emb.ndim}).")
                if np.isnan(emb).any():
                    print("    - WARNING: Embedding contains NaN values.")
                if np.isinf(emb).any():
                    print("    - WARNING: Embedding contains Inf values.")

    except Exception as e:
        print(f"An error occurred while checking HDF5 file '{h5_filepath}': {e}")


def check_gpu_environment():
    """
    Checks for and configures TensorFlow to use available GPUs.
    (Adapted from multiple scripts)
    """
    print("--- Checking TensorFlow GPU Environment ---")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info messages
    tf.get_logger().setLevel('ERROR')

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"TensorFlow: GPU Devices Detected: {gpu_devices}")
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Successfully enabled memory growth for all GPUs.")
        except RuntimeError as e:
            print(f"TensorFlow: Error setting memory growth: {e}")
    else:
        print("TensorFlow: Warning: No GPU detected. Running on CPU.")
