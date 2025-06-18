# ==============================================================================
# MODULE: utils/models_utils.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, GCN node extraction, and edge feature creation.
# VERSION: 3.6 (Implement batch-wise edge feature generation and float16 loading)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from typing import Dict, Optional, List, Tuple, Set, Union, TYPE_CHECKING, Iterator

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm.auto import tqdm
from pathlib import Path

if TYPE_CHECKING:
    from src.models.protgram_directgcn import ProtGramDirectGCN
    from gensim.models import Word2Vec


class EmbeddingLoader:
    """
    A lazy loader for HDF5 embeddings that acts like a dictionary.
    It keeps the H5 file open and retrieves embeddings on-the-fly as needed,
    which is highly memory-efficient. It should be used as a context manager.
    """

    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = str(os.path.normpath(h5_path))
        self._h5_file: Optional[h5py.File] = None
        self._keys: Optional[Set[str]] = None

    def __enter__(self) -> 'EmbeddingLoader':
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Embedding file not found: {self.h5_path}")
        try:
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._keys = set(self._h5_file.keys())
            print(f"Opened H5 file: {os.path.basename(self.h5_path)}, found {len(self._keys)} keys.")
        except Exception as e:
            if self._h5_file:
                self._h5_file.close()
            raise IOError(f"Could not open or read HDF5 file {self.h5_path}: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5_file:
            self._h5_file.close()
            print(f"Closed H5 file: {os.path.basename(self.h5_path)}")
            self._h5_file = None
            self._keys = None

    def __contains__(self, key: str) -> bool:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return key in self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        if self._h5_file is None or self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        if key in self._keys:
            # Load as float16 to save memory
            return self._h5_file[key][:].astype(np.float16)
        raise KeyError(f"Key '{key}' not found in {self.h5_path}")

    def __len__(self) -> int:
        if self._keys is None:
            return 0
        return len(self._keys)

    def get_keys(self) -> Set[str]:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return set(self._keys)


class EmbeddingProcessor:
    """
    A class for handling common processing tasks for embeddings.
    """

    @staticmethod
    def apply_pca(embeddings_dict: Dict[str, np.ndarray], target_dim: int, random_seed: int, output_dtype: np.dtype = np.float16) -> Optional[Dict[str, np.ndarray]]:
        if not embeddings_dict:
            print("PCA Error: No embeddings provided to transform.")
            return None

        filtered_embeddings_list = []
        filtered_ids = []
        for pid, v_emb in embeddings_dict.items():
            if v_emb is not None and v_emb.size > 0:
                # Ensure input to PCA is float32 for stability
                filtered_embeddings_list.append(v_emb.astype(np.float32))
                filtered_ids.append(pid)

        if not filtered_embeddings_list:
            print("PCA Error: All embedding vectors are None or empty. Skipping PCA.")
            return None
        if len(filtered_ids) < len(embeddings_dict):
            print(f"PCA Warning: {len(embeddings_dict) - len(filtered_ids)} embeddings were None or empty and have been excluded from PCA.")

        print(f"\nApplying PCA to reduce dimensions to {target_dim} for {len(filtered_ids)} embeddings...")
        embedding_matrix = np.array(filtered_embeddings_list, dtype=np.float32)  # Already float32

        original_dimension = embedding_matrix.shape[1]
        n_samples = embedding_matrix.shape[0]

        if embedding_matrix.ndim == 1 or original_dimension == 0:
            print("PCA Warning/Error: Embedding matrix is 1D or original dimension is 0. Skipping PCA.")
            return {pid: emb.astype(output_dtype) for pid, emb in zip(filtered_ids, filtered_embeddings_list)}

        actual_target_dim = min(target_dim, original_dimension, n_samples)

        if actual_target_dim < target_dim:
            print(f"PCA Warning: Adjusted target dimension from {target_dim} to {actual_target_dim} due to data constraints (n_samples={n_samples}, original_dim={original_dimension}).")
        if actual_target_dim <= 0:
            print(f"PCA Error: Cannot perform PCA with target dimension {actual_target_dim}. Skipping.")
            return {pid: emb.astype(output_dtype) for pid, emb in zip(filtered_ids, filtered_embeddings_list)}

        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embedding_matrix)
            pca = PCA(n_components=actual_target_dim, random_state=random_seed)
            transformed_embeddings = pca.fit_transform(scaled_embeddings)
            print(f"PCA Applied: Original shape: {embedding_matrix.shape}, Transformed shape: {transformed_embeddings.shape}")
            if pca.explained_variance_ratio_ is not None:
                print(f"Explained variance by {actual_target_dim} components: {np.sum(pca.explained_variance_ratio_):.4f}")
            return {pid: transformed_vec.astype(output_dtype) for pid, transformed_vec in zip(filtered_ids, transformed_embeddings)}
        except Exception as e:
            print(f"PCA Error during transformation: {e}. Skipping PCA.")
            return {pid: emb.astype(output_dtype) for pid, emb in zip(filtered_ids, filtered_embeddings_list)}

    @staticmethod
    def l2_normalize_torch(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if embeddings.ndim == 1:
            norm = torch.norm(embeddings, p=2, keepdim=True)
            return embeddings / (norm + eps)
        elif embeddings.ndim == 2:
            norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            return embeddings / (norm + eps)
        else:
            raise ValueError(f"Unsupported tensor ndim for L2 normalization: {embeddings.ndim}. Expected 1 or 2.")

    @staticmethod
    def extract_transformer_residue_embeddings(raw_model_output: np.ndarray, original_sequence_length: int, is_t5_model: bool) -> np.ndarray:
        if original_sequence_length <= 0:
            return np.array([], dtype=raw_model_output.dtype).reshape(0, raw_model_output.shape[-1]) if raw_model_output.ndim > 1 else np.array([])
        if raw_model_output.shape[0] == 0:
            return np.array([], dtype=raw_model_output.dtype).reshape(0, raw_model_output.shape[-1]) if raw_model_output.ndim > 1 else np.array([])
        if is_t5_model:
            actual_len_to_extract = min(raw_model_output.shape[0], original_sequence_length)
            residue_vectors = raw_model_output[:actual_len_to_extract, :]
        else:
            if raw_model_output.shape[0] <= 1:
                return np.array([], dtype=raw_model_output.dtype).reshape(0, raw_model_output.shape[-1])
            end_index = min(raw_model_output.shape[0], original_sequence_length + 1)
            residue_vectors = raw_model_output[1:end_index, :]
        return residue_vectors

    @staticmethod
    def get_word2vec_residue_embeddings(sequence: str, w2v_model: 'Word2Vec',
                                        embedding_dim: int) -> Optional[np.ndarray]:
        if not sequence:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        if not hasattr(w2v_model, 'wv') or not hasattr(w2v_model.wv, 'key_to_index'):
            print(f"Warning: Word2Vec model does not have a valid 'wv' attribute. Sequence: {sequence[:30]}...")
            return np.zeros((0, embedding_dim), dtype=np.float32)
        residue_vectors_list = []
        for residue in sequence:
            if residue in w2v_model.wv:
                residue_vectors_list.append(w2v_model.wv[residue])
        if not residue_vectors_list:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        return np.array(residue_vectors_list, dtype=np.float32)

    @staticmethod
    def pool_residue_embeddings(residue_embeddings: np.ndarray, strategy: str, embedding_dim_if_empty: Optional[int] = None) -> np.ndarray:
        if residue_embeddings is None or residue_embeddings.shape[0] == 0:
            if embedding_dim_if_empty is not None:
                return np.zeros(embedding_dim_if_empty, dtype=np.float32)
            return np.array([], dtype=np.float32)
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
        original_id, seq = protein_data
        if not seq or len(seq) < n_val:
            return original_id, None
        indices = [ngram_map.get("".join(seq[i:i + n_val])) for i in range(len(seq) - n_val + 1)]
        valid_indices = [idx for idx in indices if idx is not None and idx < len(ngram_embeddings)]
        if valid_indices:
            pooled_embedding = np.mean(ngram_embeddings[valid_indices].astype(np.float32), axis=0)
            return original_id, pooled_embedding.astype(ngram_embeddings.dtype)
        return original_id, None

    @staticmethod
    def extract_gcn_node_embeddings(model: 'ProtGramDirectGCN',
                                    data: Data,
                                    device: torch.device) -> np.ndarray:
        model.eval()
        model.to(device)
        data = data.to(device)
        with torch.no_grad():
            _, embeddings = model(data=data)
        return embeddings.cpu().numpy()

    @staticmethod
    def generate_edge_features_batched(
            interaction_pairs: List[Tuple[str, str, int]],
            protein_embeddings: Dict[str, np.ndarray],  # Embeddings are np.float16
            method: str,
            batch_size: int,
            embedding_dim: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates edge features in batches for link prediction.
        Yields (features_batch, labels_batch). Features are float16.
        """
        if not protein_embeddings:
            print("EmbeddingProcessor.generate_edge_features_batched ERROR: Protein embeddings dictionary is empty.")
            return
        if embedding_dim <= 0:
            print("EmbeddingProcessor.generate_edge_features_batched ERROR: Invalid embedding_dim.")
            return

        current_batch_features = []
        current_batch_labels = []

        for p1_id, p2_id, label in interaction_pairs:
            emb1 = protein_embeddings.get(p1_id)
            emb2 = protein_embeddings.get(p2_id)

            if emb1 is not None and emb2 is not None and emb1.size > 0 and emb2.size > 0:
                if emb1.shape[0] == embedding_dim and emb2.shape[0] == embedding_dim:
                    if method == 'concatenate':
                        feature = np.concatenate((emb1, emb2))  # Result will be float16
                    elif method == 'average':
                        feature = ((emb1.astype(np.float32) + emb2.astype(np.float32)) / 2.0).astype(np.float16)
                    elif method == 'hadamard':
                        feature = emb1 * emb2  # Result float16
                    elif method == 'l1_distance':
                        feature = np.abs(emb1 - emb2)  # Result float16
                    elif method == 'l2_distance':
                        feature = (emb1 - emb2) ** 2  # Result float16
                    else:  # Default to concatenate
                        feature = np.concatenate((emb1, emb2))

                    current_batch_features.append(feature)
                    current_batch_labels.append(label)

                    if len(current_batch_features) == batch_size:
                        yield np.array(current_batch_features, dtype=np.float16), np.array(current_batch_labels, dtype=np.int32)
                        current_batch_features = []
                        current_batch_labels = []

        if current_batch_features:  # Yield any remaining items
            yield np.array(current_batch_features, dtype=np.float16), np.array(current_batch_labels, dtype=np.int32)

    @staticmethod
    def create_edge_embeddings(interaction_pairs: List[Tuple[str, str, int]],
                               protein_embeddings: Dict[str, np.ndarray],  # Embeddings are np.float16
                               method: str = 'concatenate') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        DEPRECATED in favor of generate_edge_features_batched for large datasets.
        Creates edge features for link prediction from per-protein embeddings.
        """
        print("EmbeddingProcessor.create_edge_embeddings: WARNING - This method loads all features into memory and is deprecated for large datasets. Use generate_edge_features_batched instead.")
        if not protein_embeddings:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: Protein embeddings dictionary is empty.")
            return None
        first_valid_embedding = next((v for v in protein_embeddings.values() if v is not None and v.size > 0), None)
        if first_valid_embedding is None:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: No valid protein embeddings found to determine dimension.")
            return None
        embedding_dim = first_valid_embedding.shape[0]

        edge_features_list = []
        labels_list = []
        skipped_mismatch_dim = 0
        skipped_missing_emb = 0

        for p1_id, p2_id, label in interaction_pairs:
            emb1 = protein_embeddings.get(p1_id)
            emb2 = protein_embeddings.get(p2_id)

            if emb1 is not None and emb2 is not None and emb1.size > 0 and emb2.size > 0:
                if emb1.shape[0] == embedding_dim and emb2.shape[0] == embedding_dim:
                    if method == 'concatenate':
                        feature = np.concatenate((emb1, emb2))
                    elif method == 'average':
                        feature = ((emb1.astype(np.float32) + emb2.astype(np.float32)) / 2.0).astype(np.float16)
                    elif method == 'hadamard':
                        feature = emb1 * emb2
                    elif method == 'l1_distance':
                        feature = np.abs(emb1 - emb2)
                    elif method == 'l2_distance':
                        feature = (emb1 - emb2) ** 2
                    else:
                        feature = np.concatenate((emb1, emb2))
                    edge_features_list.append(feature)
                    labels_list.append(label)
                else:
                    skipped_mismatch_dim += 1
            else:
                skipped_missing_emb += 1

        if skipped_mismatch_dim > 0:
            print(f"EmbeddingProcessor.create_edge_embeddings Warning: Skipped {skipped_mismatch_dim} pairs due to mismatched embedding dimensions (expected {embedding_dim}).")
        if skipped_missing_emb > 0:
            print(f"EmbeddingProcessor.create_edge_embeddings Warning: Skipped {skipped_missing_emb} pairs due to missing/empty embeddings for one or both proteins.")

        if not edge_features_list:
            print("EmbeddingProcessor.create_edge_embeddings ERROR: No edge features were generated.")
            return None
        return np.array(edge_features_list, dtype=np.float16), np.array(labels_list, dtype=np.int32)
