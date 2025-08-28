# ==============================================================================
# MODULE: utils/post/embedding_processor.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, and edge feature creation.
# VERSION: 7.0 (Aligned DirectGCN embedding extraction with Parallel Views architecture)
# AUTHOR: Islam Ebeid
# ==============================================================================

from pathlib import Path
from typing import Dict
from typing import Optional, List, Tuple, Union, TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm.auto import tqdm

from source.utils.data.id_mapper import IDMapper
from source.utils.fs.file_utils import FileUtils
from source.utils.post.embedding_loader import EmbeddingLoader

if TYPE_CHECKING:
    from configuration.config import Config


class EmbeddingProcessor:
    """
    A class for handling common processing tasks for embeddings.
    """

    @staticmethod
    def standardize_embedding_file(raw_embedding_path: str) -> str:
        """
        Creates a new, standardized HDF5 embedding file by applying an ID map.

        This function is the single, centralized point for ID mapping. It takes
        a path to an H5 file with raw IDs, applies the mapping, and saves a new
        file with a '_standardized' suffix.

        Args:
            raw_embedding_path: Path to the input H5 file with raw protein IDs.

        Returns:
            The path to the newly created standardized HDF5 file.
        """
        config = Config()
        if config.ID_MAPPING_MODE == 'none':
            return raw_embedding_path

        # --- DEFINITIVE FIX for OOM Error: Use on-demand mapping for 'file' mode ---
        if config.ID_MAPPING_MODE == 'file':
            print(f"  Standardizing embedding file using on-demand mapping: {Path(raw_embedding_path).name}")
            id_mapping_parquet_path = config.ID_MAPPING_PATH
            if not id_mapping_parquet_path.exists():
                print(f"  - ❌ ERROR: ID mapping Parquet file not found at {id_mapping_parquet_path}. Cannot standardize. Returning original file.")
                return raw_embedding_path

            # 1. Get the keys from the embedding file we need to translate
            with EmbeddingLoader(raw_embedding_path, config=config) as loader:
                keys_to_map = loader.get_keys()

            if not keys_to_map:
                print("  - WARNING: No embeddings found in the source file. Nothing to standardize.")
                return raw_embedding_path

            # 2. Load the mapping parquet and filter it to only the keys we need
            print(f"  Loading relevant mappings for {len(keys_to_map)} IDs from Parquet file...")
            try:
                import dask.dataframe as dd
                map_ddf = dd.read_parquet(id_mapping_parquet_path)

                # --- DEFINITIVE FIX for Performance: Replace slow 'isin' with a fast Dask 'merge' (join) ---
                # The previous implementation used `isin(list(keys_to_map))`, which is a known
                # performance anti-pattern in Dask. It requires serializing the entire (potentially huge)
                # list of keys and sending it to every worker, causing the process to hang.
                # This new approach converts the keys into a Dask DataFrame and performs a highly
                # optimized merge operation, which is the standard and scalable way to
                # perform this kind of filtering on large datasets.
                keys_df = pd.DataFrame(list(keys_to_map), columns=['db_id'])
                keys_ddf = dd.from_pandas(keys_df, npartitions=config.DASK_N_PARTITIONS)
                filtered_map_ddf = dd.merge(map_ddf, keys_ddf, on='db_id', how='inner')

                filtered_map_df = filtered_map_ddf.compute()
                id_map = dict(zip(filtered_map_df['db_id'], filtered_map_df['uniprot_id']))
                print(f"  Successfully created a specific ID map with {len(id_map)} entries.")
            except Exception as e:
                print(f"  - ❌ ERROR: Failed to load or filter the ID mapping Parquet file: {e}")
                return raw_embedding_path
        else: # Handle other modes like 'regex' which are small enough for memory
            mapper = IDMapper(config)
            id_map = mapper.get_map()
            if not id_map:
                print(f"  - WARNING: No ID map found for mode '{config.ID_MAPPING_MODE}'. Skipping standardization for {Path(raw_embedding_path).name}.")
                return raw_embedding_path

        raw_path = Path(raw_embedding_path)
        standardized_path = raw_path.with_name(f"{raw_path.stem}_standardized.h5")

        # 3. Now proceed with the original standardization logic using the (now much smaller) id_map
        try:
            standardized_embeddings_dict = {}
            # Use the EmbeddingLoader to handle both HDF5 formats transparently
            with EmbeddingLoader(raw_path, config=config) as loader:
                for original_id in loader.get_keys():
                    standardized_id = id_map.get(original_id, original_id)
                    standardized_embeddings_dict[standardized_id] = loader[original_id]

            FileUtils.write_h5(standardized_embeddings_dict, standardized_path, "Writing Standardized H5")

            print(f"    - Saved standardized embeddings to: {standardized_path.name}")
            return str(standardized_path)
        except Exception as e:
            print(f"  - ❌ ERROR during standardization of {raw_path.name}: {e}")
            return raw_embedding_path  # Return original path on failure

    @staticmethod
    def apply_pca(
            embeddings_dict: Dict[str, np.ndarray], target_dim: int, random_seed: int,
            output_dtype: np.dtype = np.float16) -> Optional[Dict[str, np.ndarray]]:
        """
        Applies PCA to an in-memory dictionary of embeddings.
        """
        if not embeddings_dict:
            print("PCA Error: No embeddings provided to transform.")
            return None

        # FIX: Explicitly filter out embeddings containing NaN values to prevent PCA from failing.
        filtered_embeddings_list = []
        filtered_ids = []
        for k, v in embeddings_dict.items():
            if v is not None and v.size > 0 and not np.isnan(v).any():
                filtered_embeddings_list.append(v.astype(np.float32))
                filtered_ids.append(k)
            elif v is not None and np.isnan(v).any():
                print(f"    PCA Warning: Skipping protein '{k}' because its embedding contains NaN values.")

        if not filtered_embeddings_list:
            print("PCA Error: All embedding vectors are None or empty. Skipping PCA.")
            return None

        print(f"\nApplying PCA to reduce dimensions to {target_dim} for {len(filtered_ids)} embeddings...")
        embedding_matrix = np.array(filtered_embeddings_list, dtype=np.float32)
        original_dimension = embedding_matrix.shape[1]
        n_samples = embedding_matrix.shape[0]

        actual_target_dim = min(target_dim, original_dimension, n_samples)
        if actual_target_dim < target_dim:
            print(f"PCA Warning: Adjusted target dimension from {target_dim} to {actual_target_dim} due to data constraints.")
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
    def apply_pca_to_h5(input_h5_path: Path, output_dir: Path, target_dimension: int, random_seed: int) -> Path:
        """
        Reads embeddings from an HDF5 file, applies PCA, and saves the transformed
        embeddings to a new HDF5 file.
        --- ANTICIPATORY DEBUGGING (SCALABILITY) ---
        This version uses IncrementalPCA to process the file in chunks, ensuring
        it can handle datasets that are too large to fit into memory.

        Args:
            input_h5_path (Path): Path to the input HDF5 file.
            output_dir (Path): Directory to save the new HDF5 file.
            target_dimension (int): The desired dimension after PCA.
            random_seed (int): The random seed for PCA reproducibility.
        Returns:
            Path: The path to the newly created HDF5 file, or the original path on failure.
        """
        import os
        import traceback
        from sklearn.decomposition import IncrementalPCA

        print(f"Applying IncrementalPCA to file: '{input_h5_path.name}'. Target dimension: {target_dimension}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_h5_path = output_dir / f"{input_h5_path.stem}.pca_{target_dimension}.h5"

        if output_h5_path.exists():
            print(f"  PCA-processed file already exists: {output_h5_path.name}. Skipping.")
            return output_h5_path

        try:
            # Use a reasonable batch size for memory efficiency
            batch_size = 5000
            ipca = IncrementalPCA(n_components=target_dimension, batch_size=batch_size)
            scaler = StandardScaler()

            # --- Phase 1: Fit the scaler and IPCA model in chunks ---
            print("  Phase 1: Fitting IncrementalPCA model on data chunks...")
            with EmbeddingLoader(input_h5_path) as loader:
                keys = list(loader.get_keys())
                if not keys:
                    print(f"  Warning: No embeddings found in '{input_h5_path.name}'. Cannot apply PCA.")
                    return input_h5_path

                # Check if target_dimension is valid
                first_emb = loader[keys[0]]
                original_dimension = first_emb.shape[0]
                if target_dimension >= original_dimension:
                    print(f"  PCA Skipped: Target dimension ({target_dimension}) is >= original dimension ({original_dimension}).")
                    return input_h5_path
                # --- FIX: Also check if the number of samples is sufficient for PCA ---
                if target_dimension >= len(keys):
                    print(f"  PCA Skipped: Target dimension ({target_dimension}) is >= number of samples ({len(keys)}).")
                    return input_h5_path

                for i in tqdm(range(0, len(keys), batch_size), desc="  - Fitting PCA"):
                    batch_keys = keys[i:i + batch_size]
                    # --- DEFINITIVE FIX: Check for both NaN and Inf values ---
                    # StandardScaler can produce NaNs if the input contains infinity, which
                    # then corrupts the PCA model. np.isfinite() checks for both.
                    batch_embeddings = np.array([loader[k] for k in batch_keys if loader[k].size > 0 and np.all(np.isfinite(loader[k]))], dtype=np.float32)
                    if batch_embeddings.shape[0] > 0:
                        # IncrementalPCA requires scaling, so we fit the scaler incrementally as well.
                        scaler.partial_fit(batch_embeddings)
                        ipca.partial_fit(scaler.transform(batch_embeddings))

            print(f"  PCA model fitted. Explained variance by {ipca.n_components_} components: {np.sum(ipca.explained_variance_ratio_):.4f}")

            # --- Phase 2: Transform the data in chunks and write to new H5 file ---
            print("  Phase 2: Transforming data and writing to new HDF5 file...")
            # --- DEFINITIVE FIX: Accumulate results and use the efficient FileUtils.write_h5 ---
            # This ensures the output format is consistent with the rest of the pipeline.
            transformed_embeddings_dict = {}
            with EmbeddingLoader(input_h5_path) as loader:
                keys = list(loader.get_keys())
                for i in tqdm(range(0, len(keys), batch_size), desc="  - Transforming & Writing"):
                    batch_keys = keys[i:i + batch_size]
                    valid_keys = []
                    valid_embeddings_list = []
                    # --- DEFINITIVE FIX: Also check for NaN/Inf in the transform loop for consistency ---
                    for k in batch_keys:
                        emb = loader[k]
                        if emb.size > 0 and np.all(np.isfinite(emb)):
                            valid_keys.append(k)
                            valid_embeddings_list.append(emb)
                    if valid_embeddings_list:
                        batch_embeddings = np.array(valid_embeddings_list, dtype=np.float32)
                        transformed_batch = ipca.transform(scaler.transform(batch_embeddings))
                        for key, transformed_vec in zip(valid_keys, transformed_batch):
                            transformed_embeddings_dict[key] = transformed_vec.astype(np.float16)

            FileUtils.write_h5(transformed_embeddings_dict, output_h5_path, "Writing PCA-Transformed H5")
            print(f"  Successfully saved PCA-transformed embeddings to '{output_h5_path.name}'")
            return output_h5_path

        except Exception as e:
            print(f"  ERROR during file-based PCA processing for '{input_h5_path.name}': {e}")
            traceback.print_exc()
            if Path(output_h5_path).exists():
                os.remove(output_h5_path)
            return input_h5_path  # Return original path on failure

    @staticmethod
    def l2_normalize_torch(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings / (norm + eps)

    @staticmethod
    def extract_transformer_residue_embeddings(raw_model_output: np.ndarray, original_sequence_length: int,
                                               is_t5_model: bool) -> np.ndarray:
        if original_sequence_length <= 0 or raw_model_output.shape[0] == 0:
            return np.array([], dtype=raw_model_output.dtype).reshape(0, raw_model_output.shape[-1])
        if is_t5_model:
            return raw_model_output[:min(raw_model_output.shape[0], original_sequence_length), :]
        else:
            # For BERT-like models, skip the [CLS] token at the beginning
            return raw_model_output[1:min(raw_model_output.shape[0], original_sequence_length + 1), :]

    @staticmethod
    def get_word2vec_residue_embeddings(sequence: str, w2v_model: Word2Vec,
                                        embedding_dim: int) -> Optional[np.ndarray]:
        if not sequence: return np.zeros((0, embedding_dim), dtype=np.float32)
        if not hasattr(w2v_model, 'wv'): return np.zeros((0, embedding_dim), dtype=np.float32)
        # --- FIX: More memory-efficient implementation ---
        # Pre-allocate the array and fill it, avoiding a large intermediate list of arrays.
        valid_residues = [res for res in sequence if res in w2v_model.wv]
        if not valid_residues:
            return np.zeros((0, embedding_dim), dtype=np.float32)

        residue_vectors = np.zeros((len(valid_residues), embedding_dim), dtype=np.float32)
        for i, residue in enumerate(valid_residues):
            residue_vectors[i] = w2v_model.wv[residue]
        return residue_vectors

    @staticmethod
    def pool_residue_embeddings(residue_embeddings: np.ndarray, strategy: str,
                                embedding_dim_if_empty: Optional[int] = None) -> np.ndarray:
        if residue_embeddings is None or residue_embeddings.shape[0] == 0:
            return np.zeros(embedding_dim_if_empty, dtype=np.float32) if embedding_dim_if_empty else np.array([], dtype=np.float32)
        if strategy == 'mean': return np.mean(residue_embeddings, axis=0)
        if strategy == 'sum': return np.sum(residue_embeddings, axis=0)
        if strategy == 'max': return np.max(residue_embeddings, axis=0)
        if strategy == 'attention':
            # A simple, non-parametric attention mechanism.
            # The score for each residue is its dot product with the mean embedding.
            if residue_embeddings.shape[0] > 1:
                mean_vec = np.mean(residue_embeddings, axis=0, keepdims=True)
                attention_scores = np.dot(residue_embeddings, mean_vec.T).flatten()
                # --- FIX: Use numerically stable softmax (subtract max score) ---
                # This prevents overflow and is consistent with other attention implementations in this file.
                exp_scores = np.exp(attention_scores - np.max(attention_scores))
                attention_weights = exp_scores / np.sum(exp_scores)
                return np.dot(attention_weights, residue_embeddings)
            return np.mean(residue_embeddings, axis=0)  # Fallback for single-residue sequences
        return np.mean(residue_embeddings, axis=0)

    @staticmethod
    def pool_ngram_embeddings_for_protein_fast(protein_sequences: List[Tuple[str, str]], n_val: int,
                                               ngram_map: Dict[str, int], ngram_embeddings: np.ndarray,
                                               strategy: str = 'mean') -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """
        A memory-efficient method for pooling n-gram embeddings to the protein level.
        This implementation processes proteins one by one to keep memory usage low and constant,
        regardless of the pooling strategy.

        Returns:
            A tuple containing (pooled_embeddings, attention_weights).
            The attention_weights dict is {protein_id: {ngram_str: weight}} and is
            only populated if strategy is 'attention'.
        """
        print(f"  Starting protein-level pooling (Strategy: {strategy})...")
        if not protein_sequences:
            return {}, {}

        embedding_dim = ngram_embeddings.shape[1]
        pooled_embeddings = {}
        attention_weights_log = {}

        for prot_id, seq in tqdm(protein_sequences, desc=f"    Pooling n-grams per protein ({strategy})"):
            ngram_indices_for_this_protein = []
            ngrams_for_this_protein = []  # For attention logging
            if len(seq) >= n_val:
                for i in range(len(seq) - n_val + 1):
                    ngram_str = "".join(seq[i:i + n_val])
                    ngram_idx = ngram_map.get(ngram_str)
                    if ngram_idx is not None:
                        ngram_indices_for_this_protein.append(ngram_idx)
                        if strategy == 'attention':
                            ngrams_for_this_protein.append(ngram_str)

            if not ngram_indices_for_this_protein:
                pooled_embeddings[prot_id] = np.zeros(embedding_dim, dtype=ngram_embeddings.dtype)
                continue

            # This is now memory-efficient as it's only for one protein at a time
            protein_ngrams_arr = ngram_embeddings[ngram_indices_for_this_protein].astype(np.float32)

            if protein_ngrams_arr.shape[0] == 0:
                pooled_embeddings[prot_id] = np.zeros(embedding_dim, dtype=ngram_embeddings.dtype)
                continue

            # Apply the pooling strategy to the n-grams of this single protein
            if strategy == 'mean':
                pooled_emb = np.mean(protein_ngrams_arr, axis=0)
            elif strategy == 'sum':
                pooled_emb = np.sum(protein_ngrams_arr, axis=0)
            elif strategy == 'max':
                pooled_emb = np.max(protein_ngrams_arr, axis=0)
            elif strategy == 'attention':
                if protein_ngrams_arr.shape[0] > 1:
                    mean_vec = np.mean(protein_ngrams_arr, axis=0, keepdims=True)
                    attention_scores = np.dot(protein_ngrams_arr, mean_vec.T).flatten()
                    exp_scores = np.exp(attention_scores - np.max(attention_scores))  # stable softmax
                    attention_weights = exp_scores / np.sum(exp_scores)
                    pooled_emb = np.dot(attention_weights, protein_ngrams_arr)

                    # Log the weights by their n-gram STRING
                    attention_weights_log[prot_id] = {
                        ngram_str: float(weight)
                        for ngram_str, weight in zip(ngrams_for_this_protein, attention_weights)
                    }
                else:  # Only one n-gram
                    pooled_emb = protein_ngrams_arr[0]
                    if ngrams_for_this_protein:
                        attention_weights_log[prot_id] = {ngrams_for_this_protein[0]: 1.0}
            else:
                raise ValueError(f"Unknown pooling strategy: '{strategy}'")

            pooled_embeddings[prot_id] = pooled_emb.astype(ngram_embeddings.dtype)

        return pooled_embeddings, attention_weights_log

    @staticmethod
    def extract_gcn_node_embeddings(model: nn.Module,
                                    full_data: Data, graph_obj: 'DirectedNgramGraph',
                                    config: 'Config', device: torch.device,
                                    prepare_data_func: Callable,
                                    create_clustered_subgraphs_func: callable) -> np.ndarray:
        """Extracts node embeddings from a GNN, handling both full-batch and clustered inference."""
        model.eval()
        model.to(device)

        if config.PROTGRAM_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > config.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES:
            print(f"  Extracting embeddings for {graph_obj.number_of_nodes} nodes using clustered inference...")
            partitions = create_clustered_subgraphs_func(graph_obj)
            if not partitions:
                return np.array([])

            results = []
            for node_idx_batch in tqdm(partitions, desc="  Inference on subgraphs", leave=False):
                nodes_tensor = torch.tensor(node_idx_batch, dtype=torch.long)
                subgraph_data = graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=nodes_tensor,
                    # --- DEFINITIVE FIX: Pass the homophily/heterophily matrices to the subgraph creator ---
                    A_homo_norm=getattr(full_data, 'A_homo_norm', None),
                    A_hetero_norm=getattr(full_data, 'A_hetero_norm', None)
                ).to(device)

                with torch.no_grad():
                    _, subgraph_embeddings = model(data=subgraph_data)
                results.append((subgraph_data.original_indices.cpu(), subgraph_embeddings.cpu()))

            if not results: return np.array([])
            first_emb = results[0][1]
            all_node_embeddings = torch.zeros(graph_obj.number_of_nodes, first_emb.shape[1], dtype=first_emb.dtype)

            for indices, embeddings in results:
                all_node_embeddings[indices] = embeddings
            return all_node_embeddings.numpy()
        else:
            print(f"  Extracting embeddings for {graph_obj.number_of_nodes} nodes using full-batch inference...")
            # --- DEFINITIVE FIX: Use the provided data preparation function ---
            # This ensures that data for inference is prepared identically to how it was for training.
            prepared_data = prepare_data_func(
                model_type=model.__class__.__name__.lower(),
                graph=graph_obj, features=full_data.x, labels=full_data.y
            ).to(device)

            with torch.no_grad():
                _, embeddings = model(data=prepared_data)
            return embeddings.cpu().numpy()

    @staticmethod
    def pool_lower_level_embeddings_for_init(
            graph_obj: 'DirectedNgramGraph',
            prev_level_embeddings: np.ndarray,
            prev_level_map: Dict[str, int],
            strategy: str = 'mean'
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Dict[str, float]]]]:
        """
        Generates initial features for an n-gram graph by pooling the embeddings
        of its constituent (n-1)-grams from the previous level. Supports 'mean'
        and 'attention' pooling strategies.

        Returns: A tuple of (features_tensor, attention_weights_dict).
        """
        n_val = graph_obj.n_value
        if n_val <= 1:
            return None  # This function is only for n > 1

        print(f"  Pooling (n-1)-gram embeddings to initialize features for n={n_val} graph (Strategy: {strategy})...")
        num_nodes = graph_obj.number_of_nodes
        embedding_dim = prev_level_embeddings.shape[1]
        # Initialize with zeros; n-grams with no valid parents will have a zero-vector feature
        hierarchical_attention_log: Dict[str, Dict[str, float]] = {}
        initial_features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

        for node_idx, ngram_str in tqdm(graph_obj.idx_to_node.items(), desc=f"  Initializing n={n_val} features", leave=False):
            # An n-gram has two (n-1)-gram parents
            parent1_str = ngram_str[:-1]
            parent2_str = ngram_str[1:]

            parent1_idx = prev_level_map.get(parent1_str)
            parent2_idx = prev_level_map.get(parent2_str)

            valid_parents = []
            if parent1_idx is not None:
                valid_parents.append(prev_level_embeddings[parent1_idx].astype(np.float32))
            if parent2_idx is not None:
                valid_parents.append(prev_level_embeddings[parent2_idx].astype(np.float32))

            if len(valid_parents) == 2:
                p1, p2 = valid_parents[0], valid_parents[1]
                if strategy == 'attention':
                    # Use a simple attention mechanism based on the dot product with the mean
                    context_vec = (p1 + p2) / 2.0
                    score1 = np.dot(p1, context_vec)
                    score2 = np.dot(p2, context_vec)
                    # Softmax for weights
                    scores = np.array([score1, score2])
                    exp_scores = np.exp(scores - np.max(scores))  # Stabilized softmax
                    weights = exp_scores / np.sum(exp_scores)
                    initial_features[node_idx] = (weights[0] * p1) + (weights[1] * p2)
                    # Log the attention weights
                    hierarchical_attention_log[ngram_str] = {
                        parent1_str: float(weights[0]),
                        parent2_str: float(weights[1])
                    }
                else:  # Default to mean pooling
                    initial_features[node_idx] = (p1 + p2) / 2.0
            elif len(valid_parents) == 1:
                # --- FIX: If only one parent is valid, use its embedding directly ---
                initial_features[node_idx] = valid_parents[0]

        return torch.from_numpy(initial_features), hierarchical_attention_log

    @staticmethod
    def create_edge_features(
            interaction_pairs: List[Tuple[str, str, int]],
            protein_embeddings: Union[Dict[str, np.ndarray], 'EmbeddingLoader'],
            method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a full feature matrix (X) and label vector (y) from interaction pairs.
        This is a high-memory operation intended to be run once before CV.
        """
        if not interaction_pairs:
            return np.array([]), np.array([])

        # Get embedding dimension from the first valid embedding
        first_key = next(iter(protein_embeddings.get_keys()), None)
        if not first_key:
            return np.array([]), np.array([])
        embedding_dim = protein_embeddings[first_key].shape[0]

        feature_dim_map = {
            'concatenate': embedding_dim * 2, 'average': embedding_dim,
            'hadamard': embedding_dim, 'l1_distance': embedding_dim, 'l2_distance': embedding_dim
        }
        edge_feature_dim = feature_dim_map.get(method, embedding_dim * 2)

        # Pre-allocate arrays for performance
        num_pairs = len(interaction_pairs)
        X = np.zeros((num_pairs, edge_feature_dim), dtype=np.float16)
        y = np.zeros(num_pairs, dtype=np.int32)

        valid_pair_count = 0
        for i, (p1_id, p2_id, label) in enumerate(tqdm(interaction_pairs, desc="  Creating Edge Features")):
            if p1_id in protein_embeddings and p2_id in protein_embeddings:
                emb1, emb2 = protein_embeddings[p1_id], protein_embeddings[p2_id]

                if emb1.shape[0] != embedding_dim or emb2.shape[0] != embedding_dim:
                    continue  # Skip pairs with mismatched embedding dimensions

                if method == 'concatenate': feature = np.concatenate((emb1, emb2))
                elif method == 'average': feature = (emb1.astype(np.float32) + emb2.astype(np.float32)) / 2.0
                elif method == 'hadamard': feature = emb1 * emb2
                elif method == 'l1_distance': feature = np.abs(emb1 - emb2)
                elif method == 'l2_distance': feature = (emb1 - emb2) ** 2
                else: feature = np.concatenate((emb1, emb2))

                X[valid_pair_count] = feature.astype(np.float16)
                y[valid_pair_count] = label
                valid_pair_count += 1

        # Trim arrays to the number of valid pairs found
        return X[:valid_pair_count], y[:valid_pair_count]