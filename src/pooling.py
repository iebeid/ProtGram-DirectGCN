import os
import h5py
import numpy as np
from sklearn.decomposition import PCA # For optional PCA
from tqdm.auto import tqdm
import gc

# --- User Configuration ---
DATASET_TAG = "uniref50_w2v_per_residue" # Should match the tag of the input H5 file
INPUT_PER_RESIDUE_H5_FILENAME = f"word2vec_per_residue_embeddings_{DATASET_TAG}.h5" # From Script A
# INPUT_PER_RESIDUE_H5_FILENAME = f"transformer_X_per_residue_embeddings_{DATASET_TAG}.h5" # Or from Script C

OUTPUT_BASE_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/word2vec_outputs" # Or transformer_outputs
# OUTPUT_BASE_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/transformer_outputs"

# Output per-protein embedding file
# The name indicates the source and that it's pooled
POOLED_EMBEDDING_FILENAME_CORE = "pooled_protein_embeddings" 

POOLING_STRATEGY = 'mean' # Options: 'mean', 'sum', 'max' (max might need care for padding)

# Optional PCA to a common dimension after pooling
APPLY_PCA = True # Set to False if no PCA is needed or if already at target dimension
COMMON_EMBEDDING_DIM_PCA = 64 # Target dimension if PCA is applied

# --- End User Configuration ---

def main():
    print(f"--- Script B: Pooling Per-Residue Embeddings to Per-Protein (Source: {INPUT_PER_RESIDUE_H5_FILENAME}) ---")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    input_h5_path = os.path.join(OUTPUT_BASE_DIR, INPUT_PER_RESIDUE_H5_FILENAME)
    if not os.path.exists(input_h5_path):
        print(f"Error: Input per-residue HDF5 file not found at {input_h5_path}. Exiting.")
        return

    output_filename_tag = DATASET_TAG
    if APPLY_PCA:
        output_filename_tag += f"_pca{COMMON_EMBEDDING_DIM_PCA}"
    
    output_per_protein_h5_path = os.path.join(OUTPUT_BASE_DIR, f"{POOLED_EMBEDDING_FILENAME_CORE}_{output_filename_tag}.h5")

    print(f"Loading per-residue embeddings from: {input_h5_path}")
    print(f"Pooling strategy: {POOLING_STRATEGY}")
    if APPLY_PCA:
        print(f"PCA will be applied to target dimension: {COMMON_EMBEDDING_DIM_PCA}")

    pooled_embeddings_dict = {}
    original_embedding_dim = None

    try:
        with h5py.File(input_h5_path, 'r') as hf_in:
            protein_ids = list(hf_in.keys())
            if not protein_ids:
                print("No protein IDs found in the input HDF5 file.")
                return

            for prot_id in tqdm(protein_ids, desc="Pooling per-residue embeddings"):
                per_residue_embs = hf_in[prot_id][:] # Load into numpy array
                if per_residue_embs.ndim == 2 and per_residue_embs.shape[0] > 0: # Ensure it's a non-empty 2D array
                    if original_embedding_dim is None:
                        original_embedding_dim = per_residue_embs.shape[1]
                    elif original_embedding_dim != per_residue_embs.shape[1]:
                        print(f"Warning: Inconsistent embedding dimension for {prot_id}. Expected {original_embedding_dim}, got {per_residue_embs.shape[1]}. Skipping.")
                        continue
                    
                    if POOLING_STRATEGY == 'mean':
                        pooled_embeddings_dict[prot_id] = np.mean(per_residue_embs, axis=0)
                    elif POOLING_STRATEGY == 'sum':
                        pooled_embeddings_dict[prot_id] = np.sum(per_residue_embs, axis=0)
                    elif POOLING_STRATEGY == 'max':
                        pooled_embeddings_dict[prot_id] = np.max(per_residue_embs, axis=0)
                    else:
                        print(f"Warning: Unknown pooling strategy '{POOLING_STRATEGY}'. Defaulting to mean for {prot_id}.")
                        pooled_embeddings_dict[prot_id] = np.mean(per_residue_embs, axis=0)
                else:
                    print(f"Warning: Empty or invalid per-residue embeddings for {prot_id}. Skipping.")
                    # Store a zero vector if you want to keep the ID
                    if original_embedding_dim is not None and original_embedding_dim > 0 :
                         pooled_embeddings_dict[prot_id] = np.zeros(original_embedding_dim, dtype=np.float32)


    except Exception as e:
        print(f"Error reading or processing HDF5 file {input_h5_path}: {e}")
        return

    if not pooled_embeddings_dict:
        print("No embeddings were pooled. Exiting.")
        return
        
    print(f"Successfully pooled embeddings for {len(pooled_embeddings_dict)} proteins.")
    current_embedding_dim = original_embedding_dim if original_embedding_dim is not None else 0


    # Optional PCA
    if APPLY_PCA and current_embedding_dim > 0:
        print(f"\nApplying PCA to reduce dimension from {current_embedding_dim} to {COMMON_EMBEDDING_DIM_PCA}...")
        ids_list = list(pooled_embeddings_dict.keys())
        embeddings_array = np.array([pooled_embeddings_dict[id_val] for id_val in ids_list])

        if embeddings_array.shape[0] > 0 and embeddings_array.shape[1] > 0:
            target_dim_pca = COMMON_EMBEDDING_DIM_PCA
            if embeddings_array.shape[1] == target_dim_pca:
                print(f"Embeddings already at target PCA dimension {target_dim_pca}. No PCA needed.")
            elif embeddings_array.shape[1] < target_dim_pca:
                print(f"Warning: Original dimension ({embeddings_array.shape[1]}) is less than PCA target dimension ({target_dim_pca}). Skipping PCA.")
            else:
                max_pca_components = min(embeddings_array.shape[0], embeddings_array.shape[1])
                pca_n_components = min(target_dim_pca, max_pca_components)
                
                if pca_n_components < 1:
                    print(f"Skipping PCA: Not enough samples or features for >= 1 component (Samples: {embeddings_array.shape[0]}, Features: {embeddings_array.shape[1]}).")
                else:
                    print(f"PCA: Applying PCA with n_components={pca_n_components}.")
                    pca = PCA(n_components=pca_n_components, random_state=42) # Your original random state
                    try:
                        transformed_embeddings = pca.fit_transform(embeddings_array)
                        print(f"PCA applied. New embedding dimension: {transformed_embeddings.shape[1]}")
                        current_embedding_dim = transformed_embeddings.shape[1]
                        pooled_embeddings_dict = {id_val: transformed_embeddings[i] for i, id_val in enumerate(ids_list)}
                    except Exception as e_pca:
                        print(f"Error during PCA: {e_pca}. Using original pooled embeddings.")
        else:
            print("PCA: Embeddings array is empty or has zero features. Skipping PCA.")


    print(f"\nSaving pooled per-protein embeddings (dim: {current_embedding_dim}) to {output_per_protein_h5_path}...")
    try:
        with h5py.File(output_per_protein_h5_path, 'w') as hf_out:
            for prot_id, embedding_vector in pooled_embeddings_dict.items():
                hf_out.create_dataset(prot_id, data=embedding_vector)
            hf_out.attrs['embedding_type'] = f'pooled_{POOLING_STRATEGY}_from_{os.path.basename(input_h5_path)}'
            hf_out.attrs['original_source_tag'] = DATASET_TAG
            hf_out.attrs['final_vector_size'] = current_embedding_dim
            if APPLY_PCA and current_embedding_dim == COMMON_EMBEDDING_DIM_PCA:
                hf_out.attrs['pca_applied_target_dim'] = COMMON_EMBEDDING_DIM_PCA

        print(f"Successfully saved pooled embeddings to {output_per_protein_h5_path}")
    except Exception as e:
        print(f"Error saving pooled embeddings HDF5 file: {e}")
    
    gc.collect()
    print("Script B finished.")

if __name__ == "__main__":
    main()