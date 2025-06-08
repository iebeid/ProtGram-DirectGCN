# ==============================================================================
# MODULE: utils/embedding_tools.py
# PURPOSE: Contains tools for post-processing embeddings, such as PCA.
# ==============================================================================

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


def apply_pca(embeddings_dict: Dict[str, np.ndarray], target_dim: int, random_seed: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Applies PCA to a dictionary of embeddings to reduce their dimensionality.
    (Adapted from protein_embedder.py)
    """
    if not embeddings_dict:
        print("PCA Error: No embeddings provided to transform.")
        return None

    print(f"\nApplying PCA to reduce dimensions to {target_dim}...")
    ids, vectors = zip(*embeddings_dict.items())
    embedding_matrix = np.array(vectors, dtype=np.float32)

    original_dimension = embedding_matrix.shape[1]
    n_samples = embedding_matrix.shape[0]

    # PCA n_components must be <= n_samples and <= n_features
    actual_target_dim = min(target_dim, original_dimension, n_samples)
    if actual_target_dim < target_dim:
        print(f"PCA Warning: Adjusted target dimension from {target_dim} to {actual_target_dim} due to data constraints (n_samples={n_samples}, original_dim={original_dimension}).")

    if actual_target_dim <= 0:
        print(f"PCA Error: Cannot perform PCA with target dimension {actual_target_dim}. Skipping.")
        return None

    try:
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embedding_matrix)

        pca = PCA(n_components=actual_target_dim, random_state=random_seed)
        transformed_embeddings = pca.fit_transform(scaled_embeddings)

        print(f"PCA Applied: Original shape: {embedding_matrix.shape}, Transformed shape: {transformed_embeddings.shape}")
        print(f"Explained variance by {actual_target_dim} components: {np.sum(pca.explained_variance_ratio_):.4f}")

        return {pid: transformed_vec.astype(np.float32) for pid, transformed_vec in zip(ids, transformed_embeddings)}
    except Exception as e:
        print(f"PCA Error: {e}. Skipping PCA.")
        return None
