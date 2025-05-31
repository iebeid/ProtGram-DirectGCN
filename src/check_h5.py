import h5py
import numpy as np
import os
import sys
import random

def check_h5_embeddings(h5_filepath: str, num_samples_to_check: int = 5):
    """
    Reads an HDF5 embedding file, lists some keys, and checks if embeddings exist
    and have expected properties.

    Args:
        h5_filepath (str): Path to the HDF5 file.
        num_samples_to_check (int): How many sample embeddings to inspect from the file.
    """
    print(f"\n--- Checking HDF5 file: {h5_filepath} ---")

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
                print("File is empty or contains no top-level datasets (embeddings).")
                return

            print(f"Found {len(keys)} total embeddings (keys) in the file.")

            print(f"\nInspecting up to {num_samples_to_check} sample embeddings:")
            sample_keys = random.sample(keys,num_samples_to_check)

            for i, key in enumerate(sample_keys):
                print(f"  {i + 1}. Key: '{key}'")

                try:
                    dataset = hf[key]
                    if not isinstance(dataset, h5py.Dataset):
                        print(f"    Error: Entry for key '{key}' is not an HDF5 Dataset.")
                        continue

                    # Load the embedding vector
                    embedding_vector = dataset[:]

                    print(f"    Successfully loaded embedding for key '{key}'.")
                    print(f"      Shape: {embedding_vector.shape}")
                    print(f"      Data type: {embedding_vector.dtype}")

                    # Basic checks for a valid embedding
                    if not isinstance(embedding_vector, np.ndarray):
                        print("      Warning: Embedding is not a NumPy array (this should not happen with h5py).")
                    elif embedding_vector.size == 0:
                        print("      Warning: Embedding array is empty.")
                    elif not np.issubdtype(embedding_vector.dtype, np.number):
                        print(f"      Warning: Embedding data type '{embedding_vector.dtype}' is not numeric.")
                    # Additional checks for floating point embeddings (common for neural embeddings)
                    elif np.issubdtype(embedding_vector.dtype, np.floating):
                        if np.isnan(embedding_vector).any():
                            print("      Warning: Embedding contains NaN (Not a Number) values.")
                        if np.isinf(embedding_vector).any():
                            print("      Warning: Embedding contains Inf (Infinity) values.")

                    # Example of checking if it's a 1D vector as expected from your pooling script
                    # The pooling script generates 1D vectors for each protein.
                    if embedding_vector.ndim != 1:
                        print(
                            f"      Note: Embedding is not a 1D vector (ndim={embedding_vector.ndim}). Your pooling script typically produces 1D vectors.")


                except Exception as e_dataset:
                    print(f"    Error processing key '{key}': {e_dataset}")

            if len(keys) > num_samples_to_check:
                print(f"\n... and {len(keys) - len(sample_keys)} more keys exist in the file.")

    except Exception as e:
        print(f"An error occurred while reading the HDF5 file '{h5_filepath}': {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepaths_to_check = sys.argv[1:]
    else:
        print("Usage: python your_script_name.py <path_to_h5_file1.h5> [path_to_h5_file2.h5 ...]")
        print("\nNo HDF5 file paths provided via command line.")
        print("You can modify the 'example_filepaths' list in the script with paths to your files for testing.")

        # Example: Provide a list of HDF5 file paths to check if no command-line arguments are given
        # Please modify this list with actual paths to your files.
        example_filepaths = [
            # Example based on your previous script's output structure:
            "C:/tmp/Models/embeddings_to_evaluate/GlobalCharGraph_Directed_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2_proteins_pca_dim32.h5",
            "C:/tmp/Models/embeddings_to_evaluate/per-protein.h5" # If you have this file
        ]

        # Filter out non-existent example paths if they are just placeholders
        filepaths_to_check = [fp for fp in example_filepaths if os.path.exists(fp)]

        if not filepaths_to_check and not example_filepaths:  # Only print if examples were empty too
            print(
                "No example file paths were hardcoded or found. Please edit the script or provide file paths as arguments.")
        elif not filepaths_to_check and example_filepaths:
            print(
                f"\nCould not find the example file(s): {', '.join(example_filepaths)}. Please check paths or provide arguments.")

    if filepaths_to_check:
        print(f"Checking {len(filepaths_to_check)} file(s) specified in the script...")
        for h5_file in filepaths_to_check:
            check_h5_embeddings(h5_file)
    elif len(sys.argv) <= 1:  # Message already printed if no hardcoded files found and no CLI args
        pass