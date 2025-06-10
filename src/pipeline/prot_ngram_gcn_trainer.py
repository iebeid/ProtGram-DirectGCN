# ==============================================================================
# MODULE: pipeline/2_gcn_trainer.py
# PURPOSE: Trains the ProtNgramGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 2.1 (Explicit L2 Reg, Corrected Data Handling & Model Call)
# ==============================================================================

import os
import gc
import pickle
import numpy as np
import h5py
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix  # Added for sparse matrix conversion
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from functools import partial
from multiprocessing import Pool

# Import from our project structure
from src.config import Config  # Ensure Config class has GCN_L2_REG_LAMBDA if used
from src.utils.graph_processor import DirectedNgramGraphForGCN
from src.utils.data_loader import fast_fasta_parser
from src.utils import id_mapper
from src.utils.math_helper import apply_pca
from src.models.prot_ngram_gcn import ProtNgramGCN


def _train_ngram_model(model: ProtNgramGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device, l2_lambda: float = 0.0):  # Added l2_lambda for explicit L2 regularization
    """
    The main training loop for the ProtNgramGCN model.
    Applies explicit L2 regularization to the loss if l2_lambda > 0.
    """
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = torch.nn.NLLLoss()

    targets = data.y
    # Ensure the mask correctly identifies nodes to be used in loss calculation.
    # For PPI, this might be all nodes if it's node property prediction.
    # If data.train_mask exists and is relevant, use it. Otherwise, this default is fine.
    mask = getattr(data, 'train_mask', torch.ones(data.num_nodes, dtype=torch.bool, device=device))

    if mask.sum() == 0:
        print("Warning: No valid training samples found based on the mask.")
        return

    print("Starting model training...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        # ProtNgramGCN.forward returns (log_probs, normalized_embeddings)
        # log_probs are derived from unnormalized embeddings, suitable for loss.
        # The second output (normalized_embeddings) is ignored here.
        log_probs, _ = model(data=data)  # Pass the full PyG Data object

        primary_loss = criterion(log_probs[mask], targets[mask])

        # Explicit L2 regularization
        l2_reg_term = torch.tensor(0., device=device)
        if l2_lambda > 0:
            for param in model.parameters():
                # Sum of squares of all model parameters
                l2_reg_term += torch.norm(param, p=2).pow(2)

        loss = primary_loss + l2_lambda * l2_reg_term

        loss.backward()
        optimizer.step()

        if epoch % (max(1, epochs // 10)) == 0:
            print(f"  Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, "
                  f"Primary Loss: {primary_loss.item():.4f}, L2 Reg Term: {(l2_lambda * l2_reg_term).item():.4f}")
    print("Model training finished.")


def _extract_node_embeddings(model: ProtNgramGCN, data: Data, device: torch.device) -> np.ndarray:
    """
    Extracts the final node embeddings from the trained model.
    The model's forward pass returns (log_probs, normalized_embeddings).
    This function uses the second element (normalized_embeddings).
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
        # Pass the full PyG Data object
        # The second element returned by model.forward is final_normalized_embeddings
        _, embeddings = model(data=data)
    return embeddings.cpu().numpy()


def _pool_single_protein(protein_data: Tuple[str, str], n_val: int, ngram_map: Dict[str, int], ngram_embeddings: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    """Pools n-gram embeddings for a single protein sequence."""
    original_id, seq = protein_data

    indices = [ngram_map.get("".join(seq[i:i + n_val])) for i in range(len(seq) - n_val + 1)]
    valid_indices = [idx for idx in indices if idx is not None]

    if valid_indices:
        return original_id, np.mean(ngram_embeddings[valid_indices], axis=0)

    return original_id, None


def _get_community_labels(graph: DirectedNgramGraphForGCN, random_state: int) -> Tuple[torch.Tensor, int]:
    """Generates node labels using the Louvain community detection algorithm."""
    import networkx as nx
    # Ensure you have 'python-louvain' installed (community module)
    import community as community_louvain

    if graph.number_of_nodes == 0 or graph.A_in_w.nnz == 0 and graph.A_out_w.nnz == 0:  # Check if graph has edges
        print(f"Warning: Graph for n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'} has no nodes or edges. Assigning sequential labels.")
        labels = torch.arange(graph.number_of_nodes, dtype=torch.long)
        return labels, graph.number_of_nodes if graph.number_of_nodes > 0 else 1

    # Create an undirected graph for Louvain algorithm from both A_in and A_out
    # This combines both incoming and outgoing relationships for community detection
    combined_adj = graph.A_in_w + graph.A_out_w  # Element-wise sum for SciPy sparse
    # Symmetrize if not already symmetric for Louvain (Louvain works on undirected)
    # combined_adj = (combined_adj + combined_adj.T) / 2 # Optional: ensure symmetry

    nx_graph = nx.from_scipy_sparse_array(combined_adj)  # Use from_scipy_sparse_array

    if nx_graph.number_of_nodes() == 0:  # Check after conversion
        print(f"Warning: Converted NetworkX graph for n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'} has no nodes. Assigning sequential labels.")
        labels = torch.arange(graph.number_of_nodes, dtype=torch.long)
        return labels, graph.number_of_nodes if graph.number_of_nodes > 0 else 1

    partition = community_louvain.best_partition(nx_graph, random_state=random_state)

    # Ensure all nodes get a label, even isolated ones not in partition keys
    labels_list = [partition.get(i, -1) for i in range(graph.number_of_nodes)]

    # Remap labels to be contiguous from 0
    unique_labels = sorted(list(set(labels_list)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    remapped_labels_list = [label_map[lbl] for lbl in labels_list]

    labels = torch.tensor(remapped_labels_list, dtype=torch.long)
    num_classes = len(unique_labels)

    if num_classes == 0 and graph.number_of_nodes > 0:  # Handle case where no communities found but nodes exist
        print("Warning: Louvain found 0 communities. Assigning a single community label to all nodes.")
        labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
        num_classes = 1
    elif num_classes == 0 and graph.number_of_nodes == 0:
        num_classes = 1  # Avoid division by zero if no nodes

    print(f"Detected {num_classes} communities using Louvain algorithm.")
    return labels, num_classes


# --- Main Orchestration Function for this Module ---
def run_gcn_training(config: Config):
    print("\n" + "=" * 80)
    print("### PIPELINE STEP 2: Training GCN Model and Generating Embeddings ###")
    print("=" * 80)
    os.makedirs(config.GCN_EMBEDDINGS_DIR, exist_ok=True)

    print("\n--- Step 1: Generating Protein ID Mapping ---")
    id_map = id_mapper.generate_id_mapping(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    level_embeddings: Dict[int, np.ndarray] = {}
    level_ngram_to_idx: Dict[int, Dict[str, int]] = {}

    # Get L2 regularization lambda from config, default to 0.0 if not specified
    # Ensure GCN_L2_REG_LAMBDA is defined in your Config class if you want to use it.
    l2_lambda_val = getattr(config, 'GCN_L2_REG_LAMBDA', 0.0)

    for n_val in range(1, config.GCN_NGRAM_MAX_N + 1):
        print(f"\n--- Training N-gram Level: n = {n_val} ---")
        graph_path = os.path.join(config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")

        try:
            with open(graph_path, 'rb') as f:
                graph_obj: DirectedNgramGraphForGCN = pickle.load(f)
                if not hasattr(graph_obj, 'A_in_w') or not hasattr(graph_obj, 'A_out_w'):
                    print(f"ERROR: Graph object for n={n_val} is missing A_in_w or A_out_w attributes.")
                    continue
                if not hasattr(graph_obj, 'node_to_idx'):
                    print(f"ERROR: Graph object for n={n_val} is missing node_to_idx attribute.")
                    continue
                if not hasattr(graph_obj, 'number_of_nodes'):
                    print(f"ERROR: Graph object for n={n_val} is missing number_of_nodes attribute.")
                    continue
                graph_obj.n_value = n_val  # For logging in _get_community_labels

        except FileNotFoundError:
            print(f"ERROR: Graph object not found at {graph_path}. Please run the graph_builder first.")
            continue
        except Exception as e:
            print(f"ERROR: Could not load or validate graph object for n={n_val} from {graph_path}: {e}")
            continue

        level_ngram_to_idx[n_val] = graph_obj.node_to_idx

        if graph_obj.number_of_nodes == 0:
            print(f"Skipping n={n_val} as the graph has 0 nodes.")
            level_embeddings[n_val] = np.array([])  # Store empty to avoid key errors if accessed
            continue

        if n_val == 1:
            num_initial_features = config.GCN_1GRAM_INIT_DIM
            x = torch.randn(graph_obj.number_of_nodes, num_initial_features)
        else:
            if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                print(f"ERROR: Previous level n={n_val - 1} embeddings not found or empty. Cannot initialize features for n={n_val}.")
                continue
            prev_embeds = level_embeddings[n_val - 1]
            prev_map = level_ngram_to_idx[n_val - 1]
            num_initial_features = prev_embeds.shape[1]

            x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float)
            for ngram, idx in graph_obj.node_to_idx.items():
                prev_ngram = ngram[:-1]
                prev_idx = prev_map.get(prev_ngram)
                if prev_idx is not None and prev_idx < len(prev_embeds):
                    x[idx] = torch.from_numpy(prev_embeds[prev_idx])  # else:  #     print(f"Warning: n-gram '{ngram}' could not find predecessor '{prev_ngram}' in previous level map for n={n_val}")

        labels, num_classes = _get_community_labels(graph_obj, config.RANDOM_STATE)

        # --- CRUCIAL FIX: Populate Data object with edge information ---
        # Convert SciPy sparse matrices (A_in_w, A_out_w) to PyG edge_index and edge_weight
        edge_index_in, edge_weight_in = from_scipy_sparse_matrix(graph_obj.A_in_w)
        edge_index_out, edge_weight_out = from_scipy_sparse_matrix(graph_obj.A_out_w)

        data = Data(x=x, y=labels, edge_index_in=edge_index_in, edge_weight_in=edge_weight_in.float(), edge_index_out=edge_index_out, edge_weight_out=edge_weight_out.float())
        # ----------------------------------------------------------------

        full_layer_dims = [num_initial_features] + config.GCN_HIDDEN_LAYER_DIMS
        print(f"Instantiating ProtNgramGCN with layer dimensions: {full_layer_dims}")

        model = ProtNgramGCN(layer_dims=full_layer_dims, num_graph_nodes=graph_obj.number_of_nodes, task_num_output_classes=num_classes, n_gram_len=n_val,
            one_gram_dim=(config.GCN_1GRAM_INIT_DIM if n_val == 1 and config.GCN_1GRAM_INIT_DIM > 0 and config.GCN_MAX_PE_LEN > 0 else 0),  # Ensure PE layer conditions met
            max_pe_len=config.GCN_MAX_PE_LEN, dropout=config.GCN_DROPOUT_RATE, use_vector_coeffs=config.GCN_USE_VECTOR_COEFFS)

        # Adjust optimizer's weight_decay if explicit L2 regularization is used
        current_optimizer_weight_decay = config.GCN_WEIGHT_DECAY
        if l2_lambda_val > 0:
            print(f"Explicit L2 regularization (lambda={l2_lambda_val}) will be added to the loss.")
            if config.GCN_WEIGHT_DECAY > 0:
                print(f"Optimizer's original weight_decay was {config.GCN_WEIGHT_DECAY}, setting to 0.0 to avoid double L2 penalty.")
            current_optimizer_weight_decay = 0.0

        optimizer = optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=current_optimizer_weight_decay)

        _train_ngram_model(model, data, optimizer, config.GCN_EPOCHS_PER_LEVEL, device, l2_lambda=l2_lambda_val)

        # Extract embeddings (already uses normalized ones from the model)
        current_level_embeddings = _extract_node_embeddings(model, data, device)
        if current_level_embeddings.size == 0 and graph_obj.number_of_nodes > 0:
            print(f"Warning: Extracted embeddings for n={n_val} are empty, but graph had nodes. Check model output.")
            # Fallback or error handling might be needed here
            level_embeddings[n_val] = np.zeros((graph_obj.number_of_nodes, full_layer_dims[-1]))  # Placeholder
        else:
            level_embeddings[n_val] = current_level_embeddings

        del model, data, graph_obj, optimizer, x, labels, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n--- Step 3: Pooling N-gram Embeddings to Protein Level ---")
    final_n_val = config.GCN_NGRAM_MAX_N
    if final_n_val not in level_embeddings or level_embeddings[final_n_val].size == 0:
        print(f"ERROR: Final n-gram level (n={final_n_val}) embeddings are missing or empty. Cannot proceed with pooling.")
        # Potentially exit or handle this error gracefully
        return

    final_ngram_embeds = level_embeddings[final_n_val]
    final_ngram_map = level_ngram_to_idx[final_n_val]

    protein_sequences = list(fast_fasta_parser(config.GCN_INPUT_FASTA_PATH))
    pool_func = partial(_pool_single_protein, n_val=final_n_val, ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds)

    pooled_embeddings = {}
    with Pool(processes=config.POOLING_WORKERS) as pool:
        for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="Pooling Protein Embeddings"):
            if vec is not None:
                final_key = id_map.get(original_id, original_id)
                pooled_embeddings[final_key] = vec

    if not pooled_embeddings:
        print("Warning: No protein embeddings were generated after pooling. Check n-gram mappings and sequences.")  # Decide how to handle this: maybe save an empty file or skip saving.

    print("\n--- Step 4: Saving Generated Embeddings ---")
    output_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
    with h5py.File(output_h5_path, 'w') as hf:
        for key, vector in tqdm(pooled_embeddings.items(), desc="Writing H5 File"):
            hf.create_dataset(key, data=vector)
    print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

    if config.APPLY_PCA_TO_GCN and pooled_embeddings:
        print("\n--- Step 5: Applying PCA for Dimensionality Reduction ---")
        pca_embeds = apply_pca(pooled_embeddings, config.PCA_TARGET_DIMENSION)
        if pca_embeds:
            pca_dim = next(iter(pca_embeds.values())).shape[0]
            pca_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
            with h5py.File(pca_h5_path, 'w') as hf:
                for key, vector in tqdm(pca_embeds.items(), desc="Writing PCA H5 File"):
                    hf.create_dataset(key, data=vector)
            print(f"SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
        elif pooled_embeddings:
            print("Warning: PCA was requested but resulted in no embeddings. Check PCA input or parameters.")

    print("\n### PIPELINE STEP 2 FINISHED ###")
