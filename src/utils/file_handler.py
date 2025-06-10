import os
import pickle
import os
import h5py
import numpy as np
import random
import tensorflow as tf
from typing import List


def print_header(title):
    """Prints a formatted header to the console."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}\n### {title} ###\n{border}\n")

def save_object(obj, filename):
    """Saves a Python object to a file using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def save_pandas_dataframe_to_csv(results_df, output_dir):
    """Saves the benchmark results dataframe to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'gnn_benchmark_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nBenchmark summary saved to: {results_path}")



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