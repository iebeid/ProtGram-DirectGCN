# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/utils/models_utils.py
# ==============================================================================
# MODULE: utils/models_utils.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, GCN node extraction, and edge feature creation.
# VERSION: 3.4 (Moved H5EmbeddingLoader here as EmbeddingLoader)
# ==============================================================================

import os  # Added for EmbeddingLoader
import h5py  # Added for EmbeddingLoader
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from typing import Dict, Optional, List, Tuple, Any, Set  # Added Set for EmbeddingLoader

# PyTorch Geometric import needed for extract_gcn_node_embeddings
from torch_geometric.data import Data
# Model import needed for extract_gcn_node_embeddings
# This creates a dependency from this utility to a specific model.
from src.models.protgram_directgcn import ProtNgramGCN


# Forward declaration for type hinting if Word2Vec is not directly imported here
# from gensim.models import Word2Vec


class EmbeddingLoader:  # Renamed from H5EmbeddingLoader
    """
    A lazy loader for HDF5 embeddings that acts like a dictionary.
    It keeps the H5 file open and retrieves embeddings on-the-fly as needed,
    which is highly memory-efficient. It should be used as a context manager.
    """

    def __init__(self, h5_path: str):
        self.h5_path = os.path.normpath(h5_path)
        self._h5_file: Optional[h5py.File] = None
        self._keys: Optional[Set[str]] = None

    def __enter__(self):
        """Open the HDF5 file and read the keys when entering the context."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Embedding file not found: {self.h5_path}")
        self._h5_file = h5py.File(self.h5_path, 'r')
        self._keys = set(self._h5_file.keys())
        print(f"Opened H5 file: {os.path.basename(self.h5_path)}, found {len(self._keys)} keys.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the HDF5 file is closed when exiting the context."""
        if self._h5_file:
            self._h5_file.close()
            print(f"Closed H5 file: {os.path.basename(self.h5_path)}")
            self._h5_file = None
            self._keys = None

    def __contains__(self, key: str) -> bool:
        """Check if a protein ID (key) exists in the HDF5 file."""
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return key in self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        """Retrieve a single embedding for a given protein ID (key)."""
        if self._h5_file is None or self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        if key in self._keys:
            return self._h5_file[key][:].astype(np.float32)
        raise KeyError(f"Key '{key}' not found in {self.h5_path}")

    def __len__(self) -> int:
        """Return the total number of embeddings in the file."""
        if self._keys is None:
            return 0
        return len(self._keys)

    def get_keys(self) -> Set[str]:
        """Returns a set of all keys (protein IDs) in the HDF5 file."""
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return set(self._keys)


class EmbeddingProcessor:
    """
    A class for handling common processing tasks for embeddings,
    such as PCA, normalization, extraction, pooling, and edge feature creation.
    """

    @staticmethod
    def apply_pca(embeddings_dict: Dict[str, np.ndarray], target_dim: int, random_seed: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Applies PCA to a dictionary of embeddings to reduce their dimensionality.
        """
        if not embeddings_dict:
            print("PCA Error: No embeddings provided to transform.")
            return None

        valid_vectors = [v for v in embeddings_dict.values() if v is not None and v.size > 0]
        if not valid_vectors:
            print("PCA Error: All embedding vectors are None or empty. Skipping PCA.")
            return None

        filtered_embeddings_dict = {k: v for k, v in embeddings_dict.items() if v is not None and v.size > 0}
        if len(filtered_embeddings_dict) < len(embeddings_dict):
            print(f"PCA Warning: {len(embeddings_dict) - len(filtered_embeddings_dict)} embeddings were None or empty and have been excluded from PCA.")

        if not filtered_embeddings_dict:
            print("PCA Error: No valid embeddings left after filtering. Skipping PCA.")
            return None

        print(f"\nApplying PCA to reduce dimensions to {target_dim} for {len(filtered_embeddings_dict)} embeddings...")
        ids, vectors = zip(*filtered_embeddings_dict.items())

        try:
            embedding_matrix = np.array(vectors, dtype=np.float32)
        except ValueError as e:
            print(f"PCA Error: Could not form a valid matrix from embedding vectors after filtering. Error: {e}. Skipping PCA.")
            return None

        if embedding_matrix.ndim == 1:
            print("PCA Warning: Only one valid sample available. PCA cannot be meaningfully applied for dimensionality reduction. Skipping.")
            return None

        original_dimension = embedding_matrix.shape[1]
        n_samples = embedding_matrix.shape[0]

        if original_dimension == 0:
            print("PCA Error: Original dimension of embeddings is 0. Skipping PCA.")
            return None

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
            print(f"PCA Error during transformation: {e}. Skipping PCA.")
            return None

    @staticmethod
    def l2_normalize_torch(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Performs L2 normalization on a batch of PyTorch embedding tensors.
        """
        if embeddings.ndim == 1:  # Single embedding vector
            norm = torch.norm(embeddings, p=2, keepdim=True)
            return embeddings / (norm + eps)
        elif embeddings.ndim == 2:  # Batch of embeddings
            norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            return embeddings / (norm + eps)
        else:
            raise ValueError(f"Unsupported tensor ndim for L2 normalization: {embeddings.ndim}. Expected 1 or 2.")

    @staticmethod
    def extract_transformer_residue_embeddings(raw_model_output: np.ndarray, original_sequence_length: int, is_t5_model: bool) -> np.ndarray:
        """
        Extracts per-residue vectors from a raw Transformer model output.
        """
        if original_sequence_length == 0:
            return np.array([], dtype=raw_model_output.dtype)
        if is_t5_model:
            residue_vectors = raw_model_output[:original_sequence_length, :]
        else:
            residue_vectors = raw_model_output[1:original_sequence_length + 1, :]
        return residue_vectors

    @staticmethod
    def get_word2vec_residue_embeddings(sequence: str, w2v_model: 'Word2Vec',  # Using string literal for Word2Vec type hint
            embedding_dim: int) -> Optional[np.ndarray]:
        """
        Retrieves Word2Vec vectors for each valid residue in a sequence.
        """
        if not sequence:
            return None
        # Ensure w2v_model.wv exists and is a KeyedVectors instance (or similar)
        if not hasattr(w2v_model, 'wv') or not hasattr(w2v_model.wv, 'key_to_index'):
            print(f"Warning: Word2Vec model does not have a valid 'wv' attribute. Sequence: {sequence[:30]}...")
            return np.zeros((0, embedding_dim), dtype=np.float32)

        residue_vectors_list = [w2v_model.wv[residue] for residue in sequence if residue in w2v_model.wv]
        if not residue_vectors_list:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        return np.array(residue_vectors_list, dtype=np.float32)

    @staticmethod
    def pool_residue_embeddings(residue_embeddings: np.ndarray, strategy: str, embedding_dim_if_empty: Optional[int] = None) -> np.ndarray:
        """
        Pools per-residue embeddings into a single per-protein vector.
        """
        if residue_embeddings.shape[0] == 0:
            if embedding_dim_if_empty is not None:
                return np.zeros(embedding_dim_if_empty, dtype=residue_embeddings.dtype if residue_embeddings.size > 0 else np.float32)
            return np.array([], dtype=residue_embeddings.dtype if residue_embeddings.size > 0 else np.float32)

        if strategy == 'mean':
            return np.mean(residue_embeddings, axis=0)
        elif strategy == 'sum':
            return np.sum(residue_embeddings, axis=0)
        elif strategy == 'max':
            return np.max(residue_embeddings, axis=0)
        else:
            print(f"Warning: Unknown pooling strategy '{strategy}'. Defaulting to 'mean'.")
            return np.mean(residue_embeddings, axis=0)

    @staticmethod
    def pool_ngram_embeddings_for_protein(protein_data: Tuple[str, str], n_val: int, ngram_map: Dict[str, int], ngram_embeddings: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
        """
        Pools n-gram embeddings for a single protein sequence to get a protein-level embedding.
        """
        original_id, seq = protein_data
        indices = [ngram_map.get("".join(seq[i:i + n_val])) for i in range(len(seq) - n_val + 1)]
        valid_indices = [idx for idx in indices if idx is not None]

        if valid_indices:
            return original_id, np.mean(ngram_embeddings[valid_indices], axis=0)
        return original_id, None

    @staticmethod
    def extract_gcn_node_embeddings(model: ProtNgramGCN,  # Specific model type
            data: Data,  # PyG Data object
            device: torch.device) -> np.ndarray:
        """
        Extracts the final node embeddings from a trained ProtNgramGCN model.
        """
        model.eval()
        model.to(device)
        data = data.to(device)
        with torch.no_grad():
            # ProtNgramGCN's forward returns (log_probs, normalized_embeddings)
            _, embeddings = model(data=data)
        return embeddings.cpu().numpy()

    @staticmethod
    def create_edge_embeddings(interaction_pairs: List[Tuple[str, str, int]], protein_embeddings: Dict[str, np.ndarray], method: str = 'concatenate') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates edge features for link prediction from per-protein embeddings.
        """
        print(f"Creating edge embeddings using method: '{method}'...")
        if not protein_embeddings:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: Protein embeddings dictionary is empty.")
            return None

        try:
            first_valid_embedding = next((v for v in protein_embeddings.values() if v is not None and v.size > 0), None)
            if first_valid_embedding is None:
                print("EmbeddingProcessor.create_edge_embeddings ERROR: No valid protein embeddings found to determine dimension.")
                return None
            embedding_dim = first_valid_embedding.shape[0]
        except (StopIteration, AttributeError, IndexError):
            print("EmbeddingProcessor.create_edge_embeddings ERROR: Could not determine embedding dimension.")
            return None

        if embedding_dim == 0:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: Embedding dimension is 0.")
            return None

        feature_dim_map = {'concatenate': embedding_dim * 2, 'average': embedding_dim, 'hadamard': embedding_dim, 'l1': embedding_dim, 'l2': embedding_dim}
        if method not in feature_dim_map:
            print(f"EmbeddingProcessor.create_edge_embeddings Warning: Unknown method '{method}', defaulting to 'concatenate'.")
            method = 'concatenate'
        feature_dim = feature_dim_map[method]

        valid_pairs_data = []
        skipped_mismatch_dim = 0
        skipped_missing_emb = 0

        for p1_id, p2_id, label in interaction_pairs:
            emb1 = protein_embeddings.get(p1_id)
            emb2 = protein_embeddings.get(p2_id)

            if emb1 is not None and emb2 is not None:
                if emb1.size > 0 and emb2.size > 0:
                    if emb1.shape[0] == embedding_dim and emb2.shape[0] == embedding_dim:
                        valid_pairs_data.append((emb1, emb2, label))
                    else:
                        skipped_mismatch_dim += 1
                else:
                    skipped_missing_emb += 1
            else:
                skipped_missing_emb += 1

        if skipped_mismatch_dim > 0:
            print(f"EmbeddingProcessor.create_edge_embeddings Warning: Skipped {skipped_mismatch_dim} pairs due to mismatched embedding dimensions.")
        if skipped_missing_emb > 0:
            print(f"EmbeddingProcessor.create_edge_embeddings Warning: Skipped {skipped_missing_emb} pairs due to missing/empty embeddings for one or both proteins.")

        if not valid_pairs_data:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: No valid pairs found with available and correctly dimensioned embeddings.")
            return None

        edge_features_list = []
        labels_list = []

        for emb1, emb2, label in tqdm(valid_pairs_data, desc="Creating Edge Features"):
            if method == 'concatenate':
                feature = np.concatenate((emb1, emb2))
            elif method == 'average':
                feature = (emb1 + emb2) / 2.0
            elif method == 'hadamard':
                feature = emb1 * emb2
            elif method == 'l1':
                feature = np.abs(emb1 - emb2)
            elif method == 'l2':
                feature = (emb1 - emb2) ** 2
            edge_features_list.append(feature)
            labels_list.append(label)

        edge_features = np.array(edge_features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)

        print(f"Created {len(edge_features)} edge features with dimension {feature_dim}.")
        return edge_features, labels
