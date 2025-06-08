# ==============================================================================
# MODULE: pipeline/2_gcn_trainer.py
# PURPOSE: Trains the ProtNgramGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# ==============================================================================

import os
import sys
import gc
import pickle
import numpy as np
import pandas as pd
import h5py
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

# Import from our new project structure
from config import Config
from utils import graph_processing, id_mapping
from utils.math_helper import apply_pca
from models.prot_ngram_gcn import ProtNgramGCN


# Helper functions specific to this training module
def _detect_communities_louvain(edge_index, num_nodes, random_state):
    import networkx as nx
    import community as community_louvain
    if num_nodes == 0 or edge_index.numel() == 0: return torch.arange(num_nodes, dtype=torch.long), num_nodes
    nx_graph = nx.Graph();
    nx_graph.add_nodes_from(range(num_nodes));
    nx_graph.add_edges_from(edge_index.cpu().numpy().T)
    partition = community_louvain.best_partition(nx_graph, random_state=random_state)
    labels = torch.tensor([partition[i] for i in range(num_nodes)], dtype=torch.long)
    return labels, len(torch.unique(labels))


def _train_ngram_model(model, data, optimizer, epochs, device, task_mode):
    # ... (Full training loop logic)
    model.train();
    model.to(device);
    data = data.to(device);
    criterion = torch.nn.NLLLoss()
    targets = data.y_task_labels if task_mode == 'community_label' else data.y_next_node
    mask = torch.ones(data.num_nodes, dtype=torch.bool) if task_mode == 'community_label' else (targets != -1)
    if mask.sum() == 0: print("No valid training samples."); return
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad();
        log_probs, _ = model(data);
        loss = criterion(log_probs[mask], targets[mask])
        if loss is not None: loss.backward(); optimizer.step()
        if epoch % (max(1, epochs // 10)) == 0: print(f"  Epoch: {epoch:03d}, Loss: {loss.item():.4f}")


def _extract_node_embeddings(model, data, device):
    model.eval();
    model.to(device);
    data = data.to(device)
    with torch.no_grad(): _, embeddings = model(data); return embeddings.cpu().numpy()


def _pool_single_protein(protein_data, n_val, ngram_map, ngram_embeddings):
    original_id, seq = protein_data
    indices = [ngram_map.get("".join(seq[i:i + n_val])) for i in range(len(seq) - n_val + 1)]
    valid_indices = [idx for idx in indices if idx is not None]
    if valid_indices: return original_id, np.mean(ngram_embeddings[valid_indices], axis=0)
    return original_id, None





# --- Main Orchestration Function for this Module ---
def run_gcn_training(config: Config):
    print("\n" + "=" * 80);
    print("### PIPELINE STEP 2: Training GCN Model and Generating Embeddings ###");
    print("=" * 80)
    os.makedirs(config.GCN_EMBEDDINGS_DIR, exist_ok=True)

    # 1. Perform ID Mapping
    id_map = id_mapping.generate_id_mapping(config)

    # 2. Train GCN for each n-gram level
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    level_embeddings, level_ngram_to_idx = {}, {}
    for n_val in range(1, config.GCN_NGRAM_MAX_N + 1):
        print(f"\n--- Training N-gram Level: n = {n_val} ---")
        graph_path = os.path.join(config.GRAPH_OBJECTS_DIR, f"graph_n{n_val}.pkl")
        with open(graph_path, 'rb') as f:
            graph_obj = pickle.load(f)

        data = Data(num_nodes=graph_obj.number_of_nodes)
        level_ngram_to_idx[n_val] = graph_obj.node_to_idx
        s, t, w = zip(*[(e[0], e[1], e[2]) for e in graph_obj.edges]);
        data.edge_index_out = torch.tensor([s, t], dtype=torch.long);
        data.edge_weight_out = torch.tensor(w, dtype=torch.float)
        fs, ft, fw = zip(*[(e[0], e[1], e[2]) for e in graph_processing._flip_list(graph_obj.edges)]);
        data.edge_index_in = torch.tensor([fs, ft], dtype=torch.long);
        data.edge_weight_in = torch.tensor(fw, dtype=torch.float)

        task_mode = config.GCN_TASK_PER_LEVEL.get(n_val, 'next_node')
        # ... (task determination logic from previous version)

        if n_val == 1:
            data.x = torch.randn(data.num_nodes, config.GCN_1GRAM_INIT_DIM)
        else:
            # ... (feature generation logic from previous version)
            pass  # Placeholder for brevity, but full logic would be here

        model = ProtNgramGCN(num_initial_features=data.x.shape[1], hidden_dim1=config.GCN_HIDDEN_DIM_1, hidden_dim2=config.GCN_HIDDEN_DIM_2, num_graph_nodes=data.num_nodes, task_num_output_classes=data.num_nodes,
                             n_gram_len=n_val, one_gram_dim=(config.GCN_1GRAM_INIT_DIM if n_val == 1 else 0), max_pe_len=10, dropout=config.GCN_DROPOUT, use_vector_coeffs=True)
        optimizer = optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=config.GCN_WEIGHT_DECAY)
        _train_ngram_model(model, data, optimizer, config.GCN_EPOCHS_PER_LEVEL, device, task_mode)
        level_embeddings[n_val] = _extract_node_embeddings(model, data, device)

    # 3. Pool embeddings in parallel
    final_ngram_embeds = level_embeddings[config.GCN_NGRAM_MAX_N]
    final_ngram_map = level_ngram_to_idx[config.GCN_NGRAM_MAX_N]

    from utils.data_loader import fast_fasta_parser
    protein_sequences = list(fast_fasta_parser(config.GCN_INPUT_FASTA_PATH))
    pool_func = partial(_pool_single_protein, n_val=config.GCN_NGRAM_MAX_N, ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds)

    pooled_embeddings = {}
    with Pool(processes=config.POOLING_WORKERS) as pool:
        for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="Pooling Protein Embeddings"):
            if vec is not None:
                final_key = id_map.get(original_id, original_id)
                pooled_embeddings[final_key] = vec

    # 4. Save primary embeddings
    output_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{config.GCN_NGRAM_MAX_N}_embeddings.h5")
    with h5py.File(output_h5_path, 'w') as hf:
        for key, vector in pooled_embeddings.items(): hf.create_dataset(key, data=vector)
    print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

    # 5. Optionally apply and save PCA-reduced embeddings
    if config.APPLY_PCA_TO_GCN:
        pca_embeds = apply_pca(pooled_embeddings, config.PCA_TARGET_DIMENSION)
        if pca_embeds:
            pca_dim = next(iter(pca_embeds.values())).shape[0]
            pca_h5_path = os.path.join(config.GCN_EMBEDDINGS_DIR, f"gcn_n{config.GCN_NGRAM_MAX_N}_embeddings_pca{pca_dim}.h5")
            with h5py.File(pca_h5_path, 'w') as hf:
                for key, vector in pca_embeds.items(): hf.create_dataset(key, data=vector)
            print(f"SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")

    print("\n### PIPELINE STEP 2 FINISHED ###")