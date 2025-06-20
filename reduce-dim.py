import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer # Import for imputation
import os

def process_protein_embeddings_with_global_pca(input_h5_path, output_h5_path, target_dimension):
    """
    Reads protein embeddings (one vector per protein/key) from an HDF5 file,
    collects all embeddings, handles NaN values, applies PCA globally to reduce their dimensionality,
    and saves the transformed embeddings to a new HDF5 file, maintaining keys.

    Args:
        input_h5_path (str): Path to the input HDF5 file containing protein embeddings.
        output_h5_path (str): Path to the output HDF5 file for PCA-transformed embeddings.
        target_dimension (int): The desired target dimension for each protein embedding after PCA.
    """

    if not os.path.exists(input_h5_path):
        print(f"Error: Input file '{input_h5_path}' not found.")
        return

    print(f"Processing embeddings from '{input_h5_path}' using global PCA...")
    print(f"Target dimension for PCA: {target_dimension}")

    all_embeddings_list = []
    protein_keys = []
    original_embedding_dimension = -1 # To store the common dimension

    try:
        # Step 1: Collect all embeddings and keys
        print("Step 1/4: Collecting all protein embeddings...")
        with h5py.File(input_h5_path, 'r') as infile:
            for key in infile.keys():
                embedding = infile[key][()] # Load the embedding for the current key
                protein_keys.append(key)

                # Ensure the embedding is 2D (1, original_dimension) for consistency
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                elif embedding.ndim > 2:
                    print(f"  Warning: Embedding for key '{key}' has shape {embedding.shape}. Expected (features,) or (1, features). Flattening to (1, features).")
                    embedding = embedding.reshape(1, -1)

                if original_embedding_dimension == -1:
                    original_embedding_dimension = embedding.shape[1]
                elif embedding.shape[1] != original_embedding_dimension:
                    print(f"  Warning: Embedding for key '{key}' has dimension {embedding.shape[1]}, "
                          f"which is different from the first embedding's dimension ({original_embedding_dimension}). "
                          f"PCA requires consistent input dimensions. This might cause issues.")
                all_embeddings_list.append(embedding)

        all_embeddings_np = np.vstack(all_embeddings_list)
        num_proteins = all_embeddings_np.shape[0]
        actual_original_dimension = all_embeddings_np.shape[1]

        print(f"  Collected {num_proteins} protein embeddings, each with original dimension: {actual_original_dimension}")

        # Step 2: Handle NaN values
        print("Step 2/4: Checking for and handling NaN values...")
        nan_count = np.sum(np.isnan(all_embeddings_np))
        if nan_count > 0:
            print(f"  Found {nan_count} NaN values in the dataset. Imputing with mean strategy.")
            imputer = SimpleImputer(strategy='mean') # You can also try 'median'
            all_embeddings_imputed = imputer.fit_transform(all_embeddings_np)
            print("  NaN values imputed.")
        else:
            print("  No NaN values found. Skipping imputation.")
            all_embeddings_imputed = all_embeddings_np # Use original if no NaNs

        # Basic check for target_dimension validity
        if target_dimension >= actual_original_dimension:
            print(f"Error: Target dimension ({target_dimension}) must be smaller than the original embedding dimension ({actual_original_dimension}).")
            return
        if target_dimension <= 0:
            print(f"Error: Target dimension must be a positive integer.")
            return
        # PCA's n_components cannot exceed the number of samples.
        # This handles cases where target_dimension might be larger than num_proteins,
        # which can happen with small datasets and large target_dimension.
        if target_dimension > num_proteins:
             print(f"Warning: Target dimension ({target_dimension}) is greater than the total number of proteins ({num_proteins}). "
                   f"PCA's n_components cannot exceed the number of samples. Setting n_components to {num_proteins}.")
             target_dimension = num_proteins # Adjust n_components to max possible

        # Step 3: Apply PCA globally
        print(f"Step 3/4: Applying PCA to all embeddings...")
        pca = PCA(n_components=target_dimension)
        pca_transformed_embeddings = pca.fit_transform(all_embeddings_imputed) # Use the imputed data
        print(f"  PCA applied. Transformed data shape: {pca_transformed_embeddings.shape}")
        print(f"  Explained variance ratio (first {target_dimension} components): {np.sum(pca.explained_variance_ratio_):.4f}")

        # Step 4: Save the transformed embeddings to a new HDF5 file
        print(f"Step 4/4: Saving transformed embeddings to '{output_h5_path}'...")
        with h5py.File(output_h5_path, 'w') as outfile:
            for i, key in enumerate(protein_keys):
                outfile.create_dataset(key, data=pca_transformed_embeddings[i])

        print(f"Successfully processed all protein embeddings and saved to '{output_h5_path}'")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

# --- How to use the script (unchanged part) ---

# Define your input and output file paths
input_file = '/home/beid/documents/projects/ProtGram-DirectGCN/data/models/per-protein.h5'
output_file = '/home/beid/documents/projects/ProtGram-DirectGCN/data/models/pca_transformed_proteins.h5'
desired_dimension = 64

# Example of creating a dummy H5 file for testing
# This dummy will now include some NaNs to simulate your error
if not os.path.exists(input_file):
    print(f"Creating a dummy H5 file '{input_file}' for demonstration purposes...")
    with h5py.File(input_file, 'w') as f:
        # Good proteins
        f.create_dataset('ProteinA', data=np.random.rand(1024))
        f.create_dataset('ProteinB', data=np.random.rand(1024))

        # Protein with some NaNs
        nan_embedding = np.random.rand(1024)
        nan_embedding[10] = np.nan # Introduce a NaN
        nan_embedding[200] = np.nan
        f.create_dataset('ProteinWithNaNS', data=nan_embedding)

        # Another protein with NaNs
        nan_embedding2 = np.random.rand(1024)
        nan_embedding2[50] = np.nan
        f.create_dataset('AnotherNaNProtein', data=nan_embedding2)

        # Add more proteins to ensure enough 'samples' for PCA
        for i in range(100):
            f.create_dataset(f'Protein_{i}', data=np.random.rand(1024))
        print("Dummy H5 file created.")


# Run the function
process_protein_embeddings_with_global_pca(input_file, output_file, desired_dimension)

# # You can optionally verify the output file
# if os.path.exists(output_file):
#     print(f"\nVerifying output file: '{output_file}'")
#     with h5py.File(output_file, 'r') as f:
#         for key in f.keys():
#             print(f"  Key: '{key}', Shape: {f[key].shape}")