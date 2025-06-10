# ==============================================================================
# MODULE: pipeline/2_gcn_trainer.py
# PURPOSE: Trains the ProtNgramGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 2.0 (Dynamic Model Integration)
# ==============================================================================

import os
import gc
import pickle
import numpy as np
import h5py
import torch
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from functools import partial
from multiprocessing import Pool

# Import from our project structure
from src.config import Config
from src.utils.graph_processor import DirectedNgramGraphForGCN  # Use the correct graph object
from src.utils.data_loader import fast_fasta_parser
from src.utils import id_mapper
from src.utils.math_helper import apply_pca
from src.models.prot_ngram_gcn import ProtNgramGCN


def _train_ngram_model(model: ProtNgramGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device):
    """
    The main training loop for the ProtNgramGCN model.
    Note: The original 'task_mode' is simplified as the primary goal is embedding generation.
    The model's internal decoder is used for the node classification training task.
    """
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = torch.nn.NLLLoss()

    # The model is trained to predict community labels generated from the graph structure.
    # A simple mask is used to train on all nodes.
    targets = data.y
    mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)

    if mask.sum() == 0:
        print("Warning: No valid training samples found based on the mask.")
        return

    print("Starting model training...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        log_probs, _ = model(data.x)  # The model now handles adjacencies internally
        loss = criterion(log_probs[mask], targets[mask])

        if loss is not None:
            loss.backward()
            optimizer.step()

        if epoch % (max(1, epochs // 10)) == 0:
            print(f"  Epoch: {epoch:03d}, Loss: {loss.item():.4f}")
    print("Model training finished.")


def _extract_node_embeddings(model: ProtNgramGCN, data: Data, device: torch.device) -> np.ndarray:
    """Extracts the final node embeddings from the trained model."""
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        _, embeddings = model(data.x)
    return embeddings.cpu().numpy()


def _pool_single_protein(protein_data: Tuple[str, str], n_val: int, ngram_map: Dict[str, int], ngram_embeddings: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """Pools n-gram embeddings for a single protein sequence."""
    original_id, seq = protein_data

    # Get the integer indices for each n-gram in the sequence
    indices = [ngram_map.get("".join(seq[i:i + n_val])) for i in range(len(seq) - n_val + 1)]
    valid_indices = [idx for idx in indices if idx is not None]

    # If we found any valid n-grams, average their embeddings
    if valid_indices:
        return original_id, np.mean(ngram_embeddings[valid_indices], axis=0)

    return original_id, None


def _get_community_labels(graph: DirectedNgramGraphForGCN, random_state: int) -> Tuple[torch.Tensor, int]:
    """Generates node labels using the Louvain community detection algorithm."""
    import networkx as nx
    import community as community_louvain

    if graph.number_of_nodes == 0 or len(graph.weighted_edge_list) == 0:
        return torch.arange(graph.number_of_nodes, dtype=torch.long), graph.number_of_nodes

    # Create an undirected graph for Louvain algorithm
    edges = graph.weighted_edge_list[['source', 'target']].values.tolist()
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.number_of_nodes))
    nx_graph.add_edges_from(edges)

    partition = community_louvain.best_partition(nx_graph, random_state=random_state)
    labels = torch.tensor([partition.get(i, -1) for i in range(graph.number_of_nodes)], dtype=torch.long)
    num_classes = len(torch.unique(labels))

    print(f"Detected {num_classes} communities using Louvain algorithm.")
    return labels, num_classes


# --- Main Orchestration Function for this Module ---
def run_gcn_training(config: Config):
    print("\n" + "=" * 80)
    print("### PIPELINE STEP 2: Training GCN Model and Generating Embeddings ###")
    print("=" * 80)
    os.makedirs(config.GCN_EMBEDDINGS_DIR, exist_ok=True)

    # 1. Perform ID Mapping if necessary (can be time-consuming)
    print("\n--- Step 1: Generating Protein ID Mapping ---")
    id_map = id_mapper.generate_id_mapping(config)

    # 2. Train GCN for each n-gram level
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    level_embeddings: Dict[int, np.ndarray] = {}
    level_ngram_to_idx: Dict[int, Dict[str, int]] = {}

    for n_val in range(1, config.GCN_NGRAM_MAX_N + 1):
        print(f"\n--- Training N-gram Level: n = {n_val} ---")
        graph_path = os.path.join(config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")

        try:
            with open(graph_path, 'rb') as f:
                graph_obj: DirectedNgramGraphForGCN = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Graph object not found at {graph_path}. Please run the graph_builder first.")
            continue

        level_ngram_to_idx[n_val] = graph_obj.node_to_idx

        # Create initial node features
        if n_val == 1:
            # For 1-grams, initialize features randomly
            num_initial_features = config.GCN_1GRAM_INIT_DIM
            x = torch.randn(graph_obj.number_of_nodes, num_initial_features)
        else:
            # For n>1, use the embeddings from the (n-1) level as features
            prev_embeds = level_embeddings[n_val - 1]
            prev_map = level_ngram_to_idx[n_val - 1]
            num_initial_features = prev_embeds.shape[1]

            x = torch.zeros(graph_obj.number_of_nodes, num_initial_features)
            # Map previous embeddings to current n-gram nodes
            for ngram, idx in graph_obj.node_to_idx.items():
                # The first n-1 characters of the current n-gram form a previous-level n-gram
                prev_ngram = ngram[:-1]
                prev_idx = prev_map.get(prev_ngram)
                if prev_idx is not None:
                    x[idx] = torch.from_numpy(prev_embeds[prev_idx])

        # Get community labels for the training task
        labels, num_classes = _get_community_labels(graph_obj, config.RANDOM_STATE)

        # Prepare the PyG Data object
        data = Data(x=x, y=labels)

        # --- DYNAMIC MODEL CONFIGURATION ---
        # Construct the full layer dimensions list
        full_layer_dims = [num_initial_features] + config.GCN_HIDDEN_LAYER_DIMS
        print(f"Instantiating ProtNgramGCN with layer dimensions: {full_layer_dims}")

        model = ProtNgramGCN(layer_dims=full_layer_dims, num_graph_nodes=graph_obj.number_of_nodes, # The model internally processes the raw adjacency matrices
            raw_A_in=graph_obj.A_in_w, raw_A_out=graph_obj.A_out_w, task_num_output_classes=num_classes, n_gram_len=n_val, one_gram_dim=(config.GCN_1GRAM_INIT_DIM if n_val == 1 else 0), max_pe_len=config.GCN_MAX_PE_LEN,
            dropout=config.GCN_DROPOUT_RATE, use_vector_coeffs=config.GCN_USE_VECTOR_COEFFS)

        optimizer = optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=config.GCN_WEIGHT_DECAY)

        _train_ngram_model(model, data, optimizer, config.GCN_EPOCHS_PER_LEVEL, device)
        level_embeddings[n_val] = _extract_node_embeddings(model, data, device)

        # Clean up memory
        del model, data, graph_obj, optimizer
        gc.collect()

    # 3. Pool embeddings in parallel using the final n-gram level
    print("\n--- Step 3: Pooling N-gram Embeddings to Protein Level ---")
    final_n_val = config.GCN_NGRAM_MAX_N
    final_ngram_embeds = level_embeddings[final_n_val]
    final_ngram_map = level_ngram_to_idx[final_n_val]

    protein_sequences = list(fast_fasta_parser(config.GCN_INPUT_FASTA_PATH))

    # Use a partial function to pass fixed arguments to the pooling worker
    pool_func = partial(_pool_single_protein, n_val=final_n_val, ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds)

    pooled_embeddings = {}
    with Pool(processes=config.POOLING_WORKERS) as pool:
        for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="Pooling Protein Embeddings"):
            if vec is not None:
                final_key = id_map.get(original_id, original_id)
                pooled_embeddings[final_key] = vec

    # 4. Save primary embeddings to HDF5 file
    print("\n--- Step 4: Saving Generated Embeddings ---")
    output_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
    with h5py.File(output_h5_path, 'w') as hf:
        for key, vector in tqdm(pooled_embeddings.items(), desc="Writing H5 File"):
            hf.create_dataset(key, data=vector)
    print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

    # 5. Optionally apply and save PCA-reduced embeddings
    if config.APPLY_PCA_TO_GCN:
        print("\n--- Step 5: Applying PCA for Dimensionality Reduction ---")
        pca_embeds = apply_pca(pooled_embeddings, config.PCA_TARGET_DIMENSION)
        if pca_embeds:
            pca_dim = next(iter(pca_embeds.values())).shape[0]
            pca_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
            with h5py.File(pca_h5_path, 'w') as hf:
                for key, vector in tqdm(pca_embeds.items(), desc="Writing PCA H5 File"):
                    hf.create_dataset(key, data=vector)
            print(f"SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")

    print("\n### PIPELINE STEP 2 FINISHED ###")
