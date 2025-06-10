# ==============================================================================
# MODULE: pipeline/2_gcn_trainer.py
# PURPOSE: Trains the ProtNgramGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 2.3 (Added Sequence Validity Task)
# ==============================================================================

import os
import gc
import pickle
import random # Already imported, good
import copy   # For deep copying walks
import numpy as np
import h5py
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List # Already imported
from functools import partial
from multiprocessing import Pool

# Import from our project structure
from src.config import Config
from src.utils.graph_processor import DirectedNgramGraphForGCN
from src.utils.data_loader import fast_fasta_parser
from src.utils import id_mapper
from src.utils.math_helper import apply_pca
from src.models.prot_ngram_gcn import ProtNgramGCN


def _train_ngram_model(model: ProtNgramGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device, l2_lambda: float = 0.0):
    """
    The main training loop for the ProtNgramGCN model.
    Applies explicit L2 regularization to the loss if l2_lambda > 0.
    """
    model.train()
    model.to(device)
    data = data.to(device)
    criterion = torch.nn.NLLLoss() # Suitable for all tasks as model outputs log_softmax

    targets = data.y
    mask = getattr(data, 'train_mask', torch.ones(data.num_nodes, dtype=torch.bool, device=device))

    if mask.sum() == 0:
        print("Warning: No valid training samples found based on the mask.")
        return

    print("Starting model training...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        log_probs, _ = model(data=data)

        # Ensure targets are on the same device as log_probs and are long type
        # Also, ensure log_probs[mask] and targets[mask] are not empty
        if log_probs[mask].size(0) == 0:
            if epoch == 1: # Print warning only once
                 print(f"Warning: Mask resulted in 0 training samples for loss calculation in epoch {epoch}. Skipping loss computation for this epoch.")
            continue # Skip if no samples to train on

        primary_loss = criterion(log_probs[mask], targets[mask].to(log_probs.device).long())


        l2_reg_term = torch.tensor(0., device=device)
        if l2_lambda > 0:
            for param in model.parameters():
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
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    with torch.no_grad():
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


def _generate_community_labels(graph: DirectedNgramGraphForGCN, random_state: int) -> Tuple[torch.Tensor, int]:
    """Generates node labels using the Louvain community detection algorithm."""
    import networkx as nx
    import community as community_louvain # Ensure 'python-louvain' is installed

    graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
    print(f"Generating 'community' labels for graph {graph_n_value_str}...")

    if graph.number_of_nodes == 0:
        print(f"Warning: Graph for {graph_n_value_str} has no nodes. Assigning 0 labels, 1 class.")
        return torch.empty(0, dtype=torch.long), 1

    if graph.A_in_w.nnz == 0 and graph.A_out_w.nnz == 0:
        print(f"Warning: Graph for {graph_n_value_str} has no edges. Assigning all nodes to a single community (0).")
        labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
        return labels, 1

    combined_adj = graph.A_in_w + graph.A_out_w
    nx_graph = nx.from_scipy_sparse_array(combined_adj)

    if nx_graph.number_of_nodes() == 0:
        print(f"Warning: Converted NetworkX graph for {graph_n_value_str} has no nodes. Assigning 0 labels, 1 class.")
        return torch.empty(0, dtype=torch.long), 1

    partition = community_louvain.best_partition(nx_graph, random_state=random_state)
    labels_list = [partition.get(i, -1) for i in range(graph.number_of_nodes)]
    unique_labels_from_partition = sorted(list(set(labels_list)))

    if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
        print(f"Warning: Louvain partition for {graph_n_value_str} resulted in no valid communities or only unassigned nodes. Assigning all nodes to a single community (0).")
        labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
        num_classes = 1
    else:
        label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
        remapped_labels_list = [label_map[lbl] for lbl in labels_list]
        labels = torch.tensor(remapped_labels_list, dtype=torch.long)
        num_classes = len(unique_labels_from_partition)

    print(f"Detected {num_classes} communities for {graph_n_value_str} using Louvain algorithm.")
    return labels, num_classes


def _generate_next_node_labels(graph: DirectedNgramGraphForGCN) -> Tuple[torch.Tensor, int]:
    """
    Generates labels for a 'next node' prediction task.
    For each node, the target is chosen from its successors based on the highest transition probability.
    If multiple successors share the highest probability, one is chosen randomly from that group.
    If a node has no successors, its target is itself (self-loop).
    The number of classes is the total number of nodes in the graph.
    """
    num_nodes = graph.number_of_nodes
    graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
    print(f"Generating 'next_node' labels for graph {graph_n_value_str} with {num_nodes} nodes (prioritizing highest probability)...")

    if num_nodes == 0:
        print(f"Warning: Graph for {graph_n_value_str} has no nodes. Assigning 0 labels, 1 class for 'next_node' task.")
        return torch.empty(0, dtype=torch.long), 1

    adj_out = graph.A_out_w  # SciPy CSR matrix (outgoing edges with weights)
    labels_list = [-1] * num_nodes

    nodes_with_no_successors = 0
    nodes_with_ties_in_max_prob = 0

    for i in range(num_nodes):
        row_slice = adj_out[i]
        successors = row_slice.indices  # Indices of successor nodes
        weights = row_slice.data  # Transition probabilities (weights) to these successors

        if len(successors) > 0:
            max_weight = np.max(weights)
            # Find all successors that have this maximum weight
            highest_prob_successors = [
                succ for succ, weight in zip(successors, weights) if weight == max_weight
            ]

            if len(highest_prob_successors) > 1:
                nodes_with_ties_in_max_prob += 1

            # Randomly choose from the successors with the highest probability
            labels_list[i] = random.choice(highest_prob_successors)
        else:
            # If no outgoing edges, predict self-loop as target
            labels_list[i] = i
            nodes_with_no_successors += 1

    if nodes_with_no_successors > 0:
        print(f"  {nodes_with_no_successors}/{num_nodes} nodes in graph {graph_n_value_str} had no outgoing edges; assigned self-loop as target.")
    if nodes_with_ties_in_max_prob > 0:
        print(f"  {nodes_with_ties_in_max_prob}/{num_nodes - nodes_with_no_successors} nodes with successors had ties for the highest transition probability.")

    final_labels = torch.tensor(labels_list, dtype=torch.long)
    # For 'next_node' prediction, each node can be a target, so num_classes is num_nodes.
    num_output_classes = num_nodes
    print(f"Finished 'next_node' label generation for {graph_n_value_str}. Task output classes: {num_output_classes}.")
    return final_labels, num_output_classes


def _generate_sequence_validity_labels(graph: DirectedNgramGraphForGCN, num_walks_k: int, walk_length_l: int) -> Tuple[torch.Tensor, int]:
    """
    Generates labels for a 'sequence validity' prediction task.
    For each node, K random walks of length L are generated. K-1 are corrupted.
    The model predicts which of the K walks is the uncorrupted one.
    The number of classes is K.
    """
    num_nodes = graph.number_of_nodes
    adj_out = graph.A_out_w # For random walks using outgoing edges
    graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
    print(f"Generating 'sequence_validity' labels for graph {graph_n_value_str} with {num_nodes} nodes (K={num_walks_k}, L={walk_length_l})...")

    if num_nodes == 0:
        print(f"Warning: Graph for {graph_n_value_str} has no nodes. Assigning 0 labels, K={num_walks_k} classes for 'sequence_validity' task.")
        return torch.empty(0, dtype=torch.long), num_walks_k

    if num_walks_k <= 1:
        raise ValueError("GCN_SEQ_VALIDITY_NUM_WALKS_K must be greater than 1 for corruption.")

    labels_for_nodes = torch.full((num_nodes,), -1, dtype=torch.long) # Index of the correct walk (0 to K-1)

    for start_node_idx in tqdm(range(num_nodes), desc=f"Generating walks for n={graph.n_value}"):
        generated_walks = []
        for _ in range(num_walks_k):
            current_walk = [start_node_idx]
            current_node = start_node_idx
            for _ in range(walk_length_l - 1):
                successors = adj_out[current_node].indices
                if len(successors) == 0:
                    break # Walk ends early
                next_node = random.choice(successors)
                current_walk.append(next_node)
                current_node = next_node
            generated_walks.append(current_walk)

        # The task is to identify the correct walk from K candidates.
        # The label for 'start_node_idx' will be the index of the uncorrupted walk.
        # The actual walks and their corruptions are conceptual for label generation;
        # the model itself doesn't directly process these K sequences as input features
        # in its GCN layers. It learns a representation for start_node_idx, and the
        # decoder predicts which of K "slots" is the correct one.

        correct_walk_index = random.randint(0, num_walks_k - 1)
        labels_for_nodes[start_node_idx] = correct_walk_index

        # (Optional: The corruption logic itself isn't strictly needed for label generation
        # if the model doesn't see the corrupted walks, but it's good for understanding the task setup)
        # Example of how one might conceptualize the K choices:
        # candidate_sequences_for_node = []
        # for i, walk_to_process in enumerate(generated_walks):
        #     if i == correct_walk_index:
        #         candidate_sequences_for_node.append(walk_to_process) # Uncorrupted
        #     else:
        #         corrupted_w = list(walk_to_process) # Make a copy
        #         if len(corrupted_w) > 1: # Ensure there's something to corrupt
        #             idx_to_corrupt = random.randint(1, len(corrupted_w) - 1) # Don't corrupt start node
        #             original_node = corrupted_w[idx_to_corrupt]
        #             # Replace with a random node that is not the original node
        #             possible_replacements = [n for n in range(num_nodes) if n != original_node]
        #             if possible_replacements:
        #                 corrupted_w[idx_to_corrupt] = random.choice(possible_replacements)
        #             # else: it's a 2-node graph, corruption is tricky, or single node graph.
        #         candidate_sequences_for_node.append(corrupted_w)
        # Here, candidate_sequences_for_node[correct_walk_index] would be the true one.

    num_output_classes = num_walks_k
    print(f"Finished 'sequence_validity' label generation for {graph_n_value_str}. Task output classes: {num_output_classes}.")
    return labels_for_nodes, num_output_classes


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

    l2_lambda_val = getattr(config, 'GCN_L2_REG_LAMBDA', 0.0)

    for n_val in range(1, config.GCN_NGRAM_MAX_N + 1):
        print(f"\n--- Training N-gram Level: n = {n_val} ---")
        graph_path = os.path.join(config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")

        try:
            with open(graph_path, 'rb') as f:
                graph_obj: DirectedNgramGraphForGCN = pickle.load(f)
                required_attrs = ['A_in_w', 'A_out_w', 'node_to_idx', 'number_of_nodes']
                if not all(hasattr(graph_obj, attr) for attr in required_attrs):
                    print(f"ERROR: Graph object for n={n_val} is missing one or more required attributes.")
                    continue
                graph_obj.n_value = n_val
        except FileNotFoundError:
            print(f"ERROR: Graph object not found at {graph_path}. Please run the graph_builder first.")
            continue
        except Exception as e:
            print(f"ERROR: Could not load or validate graph object for n={n_val} from {graph_path}: {e}")
            continue

        level_ngram_to_idx[n_val] = graph_obj.node_to_idx

        if graph_obj.number_of_nodes == 0:
            print(f"Skipping n={n_val} as the graph has 0 nodes.")
            level_embeddings[n_val] = np.array([])
            continue

        current_task_type = config.GCN_TASK_TYPES_PER_LEVEL.get(n_val, config.GCN_DEFAULT_TASK_TYPE)
        print(f"Selected training task for n={n_val}: '{current_task_type}'")

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
                    x[idx] = torch.from_numpy(prev_embeds[prev_idx])

        labels, num_classes = None, None # Initialize
        if current_task_type == "community":
            labels, num_classes = _generate_community_labels(graph_obj, config.RANDOM_STATE)
        elif current_task_type == "next_node":
            labels, num_classes = _generate_next_node_labels(graph_obj)
        elif current_task_type == "sequence_validity":
            labels, num_classes = _generate_sequence_validity_labels(
                graph_obj,
                config.GCN_SEQ_VALIDITY_NUM_WALKS_K,
                config.GCN_SEQ_VALIDITY_WALK_LENGTH_L
            )
        else:
            raise ValueError(f"Unsupported GCN_TASK_TYPE '{current_task_type}' for n-gram level {n_val}. Supported types: 'community', 'next_node', 'sequence_validity'.")

        if graph_obj.number_of_nodes > 0 and (labels is None or labels.numel() == 0):
             print(f"Warning: No labels generated for n={n_val} with task '{current_task_type}' despite having {graph_obj.number_of_nodes} nodes. Model training might fail.")
             if num_classes is None or num_classes == 0: num_classes = 1
             if labels is None or labels.numel() == 0 and graph_obj.number_of_nodes > 0:
                 labels = torch.zeros(graph_obj.number_of_nodes, dtype=torch.long)

        edge_index_in, edge_weight_in = from_scipy_sparse_matrix(graph_obj.A_in_w)
        edge_index_out, edge_weight_out = from_scipy_sparse_matrix(graph_obj.A_out_w)
        data = Data(x=x, y=labels,
                    edge_index_in=edge_index_in, edge_weight_in=edge_weight_in.float(),
                    edge_index_out=edge_index_out, edge_weight_out=edge_weight_out.float())

        full_layer_dims = [num_initial_features] + config.GCN_HIDDEN_LAYER_DIMS
        print(f"Instantiating ProtNgramGCN for n={n_val} (task: {current_task_type}) with layer dimensions: {full_layer_dims}, output classes: {num_classes}")

        model = ProtNgramGCN(
            layer_dims=full_layer_dims,
            num_graph_nodes=graph_obj.number_of_nodes,
            task_num_output_classes=num_classes,
            n_gram_len=n_val,
            one_gram_dim=(config.GCN_1GRAM_INIT_DIM if n_val == 1 and config.GCN_1GRAM_INIT_DIM > 0 and config.GCN_MAX_PE_LEN > 0 else 0),
            max_pe_len=config.GCN_MAX_PE_LEN,
            dropout=config.GCN_DROPOUT_RATE,
            use_vector_coeffs=config.GCN_USE_VECTOR_COEFFS
        )

        current_optimizer_weight_decay = config.GCN_WEIGHT_DECAY
        if l2_lambda_val > 0:
            print(f"Explicit L2 regularization (lambda={l2_lambda_val}) will be added to the loss.")
            if config.GCN_WEIGHT_DECAY > 0:
                print(f"Optimizer's original weight_decay was {config.GCN_WEIGHT_DECAY}, setting to 0.0 to avoid double L2 penalty.")
            current_optimizer_weight_decay = 0.0

        optimizer = optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=current_optimizer_weight_decay)

        _train_ngram_model(model, data, optimizer, config.GCN_EPOCHS_PER_LEVEL, device, l2_lambda=l2_lambda_val)

        current_level_embeddings = _extract_node_embeddings(model, data, device)
        if current_level_embeddings.size == 0 and graph_obj.number_of_nodes > 0:
            print(f"Warning: Extracted embeddings for n={n_val} are empty, but graph had nodes. Using zero placeholder.")
            level_embeddings[n_val] = np.zeros((graph_obj.number_of_nodes, full_layer_dims[-1]))
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
        print("Warning: No protein embeddings were generated after pooling. Check n-gram mappings and sequences.")

    print("\n--- Step 4: Saving Generated Embeddings ---")
    output_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
    with h5py.File(output_h5_path, 'w') as hf:
        for key, vector in tqdm(pooled_embeddings.items(), desc="Writing H5 File"):
            if vector is not None and vector.size > 0 :
                 hf.create_dataset(key, data=vector)
            else:
                 print(f"Warning: Skipping empty/None vector for key {key} during H5 save.")
    print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

    if config.APPLY_PCA_TO_GCN and pooled_embeddings:
        print("\n--- Step 5: Applying PCA for Dimensionality Reduction ---")
        valid_pooled_embeddings = {k: v for k, v in pooled_embeddings.items() if v is not None and v.size > 0}
        if not valid_pooled_embeddings:
            print("Warning: No valid embeddings to apply PCA.")
        else:
            pca_embeds = apply_pca(valid_pooled_embeddings, config.PCA_TARGET_DIMENSION)
            if pca_embeds:
                pca_dim = next(iter(pca_embeds.values())).shape[0]
                pca_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
                with h5py.File(pca_h5_path, 'w') as hf:
                    for key, vector in tqdm(pca_embeds.items(), desc="Writing PCA H5 File"):
                        hf.create_dataset(key, data=vector)
                print(f"SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
            elif pooled_embeddings :
                print("Warning: PCA was requested but resulted in no embeddings. Check PCA input or parameters.")

    print("\n### PIPELINE STEP 2 FINISHED ###")
