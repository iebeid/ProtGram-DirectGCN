import h5py
import numpy as np
from sklearn.decomposition import PCA
import os

def process_embeddings_with_pca(input_h5_path, output_h5_path, target_dimension):
    """
    Reads feature embeddings from an HDF5 file, applies PCA to reduce their
    dimensionality, and saves the transformed embeddings to a new HDF5 file.

    Args:
        input_h5_path (str): Path to the input HDF5 file containing feature embeddings.
        output_h5_path (str): Path to the output HDF5 file for PCA-transformed embeddings.
        target_dimension (int): The desired target dimension after PCA.
    """

    if not os.path.exists(input_h5_path):
        print(f"Error: Input file '{input_h5_path}' not found.")
        return

    print(f"Processing embeddings from '{input_h5_path}'...")
    print(f"Target dimension for PCA: {target_dimension}")

    try:
        with h5py.File(input_h5_path, 'r') as infile:
            with h5py.File(output_h5_path, 'w') as outfile:
                for key in infile.keys():
                    embeddings = infile[key][()] # Load embeddings for the current key

                    # Ensure embeddings are at least 2D for PCA
                    if embeddings.ndim == 1:
                        embeddings = embeddings.reshape(1, -1)

                    original_dimension = embeddings.shape[-1]
                    print(f"  Processing key: '{key}' with original dimension: {original_dimension}")

                    if target_dimension >= original_dimension:
                        print(f"    Warning: Target dimension ({target_dimension}) is not smaller than or equal to original dimension ({original_dimension}). Skipping PCA for this key and copying data directly.")
                        outfile.create_dataset(key, data=embeddings)
                        continue

                    # Initialize PCA
                    # n_components can be an int (desired dimension) or a float (variance explained)
                    pca = PCA(n_components=target_dimension)

                    # Fit PCA and transform the embeddings
                    # If embeddings is 2D, PCA will work as expected.
                    # If embeddings is 3D (e.g., [num_samples, seq_len, embed_dim]), you might need to flatten
                    # or apply PCA to the last dimension. This script assumes the last dimension is the embedding.
                    # For simplicity, if it's >2D, we assume it's (..., feature_dimension) and reshape to (num_samples, feature_dimension)
                    # perform PCA, then reshape back.
                    original_shape = embeddings.shape
                    if embeddings.ndim > 2:
                        # Flatten all but the last dimension for PCA
                        num_samples_flat = int(np.prod(original_shape[:-1]))
                        flat_embeddings = embeddings.reshape(num_samples_flat, original_dimension)
                        pca_embeddings_flat = pca.fit_transform(flat_embeddings)
                        pca_embeddings = pca_embeddings_flat.reshape(*original_shape[:-1], target_dimension)
                    else:
                        pca_embeddings = pca.fit_transform(embeddings)

                    # Save the transformed embeddings to the new HDF5 file
                    outfile.create_dataset(key, data=pca_embeddings)
                    print(f"    Transformed to dimension: {pca_embeddings.shape[-1]}")

        print(f"Successfully processed embeddings and saved to '{output_h5_path}'")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

# --- How to use the script ---

# Define your input and output file paths
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
print("PROJECT ROOT IS: " + str(PROJECT_ROOT))
input_file = PROJECT_ROOT / "data" / "models" / "per-protein.h5" # <--- IMPORTANT: Change this to your input H5 file path
output_file = PROJECT_ROOT / "data" / "models" / 'pca_transformed_embeddings.h5'
desired_dimension = 64 # <--- IMPORTANT: Change this to your desired output dimension

# Example of creating a dummy H5 file for testing if you don't have one
# You can comment out this section if you already have your input_file
if not os.path.exists(input_file):
    print(f"Creating a dummy H5 file '{input_file}' for demonstration purposes...")
    with h5py.File(input_file, 'w') as f:
        f.create_dataset('embedding_set_1', data=np.random.rand(100, 512)) # 100 samples, 512 original dim
        f.create_dataset('embedding_set_2', data=np.random.rand(50, 256))  # 50 samples, 256 original dim
        f.create_dataset('embedding_set_3', data=np.random.rand(200, 128)) # 200 samples, 128 original dim
        f.create_dataset('embedding_set_4_small', data=np.random.rand(10, 30)) # For testing when target_dimension > original_dimension
        print("Dummy H5 file created.")


# Run the function
process_embeddings_with_pca(input_file, output_file, desired_dimension)

# You can optionally verify the output file
if os.path.exists(output_file):
    print(f"\nVerifying output file: '{output_file}'")
    with h5py.File(output_file, 'r') as f:
        for key in f.keys():
            print(f"  Key: '{key}', Shape: {f[key].shape}")