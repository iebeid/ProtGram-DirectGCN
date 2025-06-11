# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/pipeline/protgram_directgcn_embedder.py
# ==============================================================================
# MODULE: pipeline/2_gcn_trainer.py
# PURPOSE: Trains the ProtNgramGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 3.3 (Using preprocessed propagation matrices mathcal_A)
# ==============================================================================

import collections
import gc
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix  # If we want to convert mathcal_A to scipy sparse first
from torch_geometric.data import Data
# from torch_geometric.utils import from_scipy_sparse_matrix # We might not need this if passing dense and converting
from torch_geometric.utils import dense_to_sparse  # For converting dense mathcal_A to sparse
from tqdm import tqdm

# Import from our project structure
from src.config import Config
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataLoader
from src.utils.graph_utils import DirectedNgramGraph
from src.utils.models_utils import EmbeddingProcessor

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class ProtGramDirectGCNEmbedder:
    # ... ( __init__ and other methods like _train_ngram_model, _generate_..._labels remain largely the same) ...
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id_map: Dict[str, str] = {}
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)

    # ... _train_ngram_model, _generate_community_labels, etc. ...
    # Ensure these methods are correctly defined as in your previous version.
    # For brevity, I'm omitting them here if they don't directly change due to this specific request.
    # The key change is in run_pipeline where the Data object is created.

    def _train_model(self, model: ProtGramDirectGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, l2_lambda: float = 0.0):
        model.train()
        model.to(self.device)
        data = data.to(self.device)
        criterion = torch.nn.NLLLoss()
        targets = data.y
        mask = getattr(data, 'train_mask', torch.ones(data.num_nodes, dtype=torch.bool, device=self.device))

        if mask.sum() == 0:
            print("Warning: No valid training samples found based on the mask.")
            return

        print("Starting model training...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            log_probs, _ = model(data=data)  # Model uses edge_index and edge_weight from data
            if log_probs[mask].size(0) == 0:
                if epoch == 1: print(f"Warning: Mask resulted in 0 training samples for loss calculation in epoch {epoch}.")
                continue
            primary_loss = criterion(log_probs[mask], targets[mask].to(log_probs.device).long())
            l2_reg_term = torch.tensor(0., device=self.device)
            if l2_lambda > 0:
                for param in model.parameters():
                    l2_reg_term += torch.norm(param, p=2).pow(2)
            loss = primary_loss + l2_lambda * l2_reg_term
            loss.backward()
            optimizer.step()
            if epoch % (max(1, epochs // 10)) == 0:
                print(f"  Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * l2_reg_term).item():.4f}")
        print("Model training finished.")

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        import networkx as nx
        import community as community_louvain
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        print(f"Generating 'community' labels for graph {graph_n_value_str}...")
        if graph.number_of_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        # For Louvain, we use the sum of A_in_w and A_out_w to get an undirected sense
        # If A_in_w and A_out_w are dense, convert to sparse for nx.from_scipy_sparse_array
        # Or use the binary matrices if appropriate for community detection
        # Using A_in_w and A_out_w as they represent actual connection strengths
        A_in_w_sparse = csr_matrix(graph.A_in_w)
        A_out_w_sparse = csr_matrix(graph.A_out_w)

        if A_in_w_sparse.nnz == 0 and A_out_w_sparse.nnz == 0:
            print(f"Warning: Graph for {graph_n_value_str} has no weighted edges. Assigning all nodes to a single community (0).")
            return torch.zeros(graph.number_of_nodes, dtype=torch.long), 1

        combined_adj_sparse = A_in_w_sparse + A_out_w_sparse  # This creates an undirected representation
        nx_graph = nx.from_scipy_sparse_array(combined_adj_sparse)
        if nx_graph.number_of_nodes() == 0: return torch.empty(0, dtype=torch.long), 1

        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE)
        labels_list = [partition.get(i, -1) for i in range(graph.number_of_nodes)]
        unique_labels_from_partition = sorted(list(set(labels_list)))
        if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
            labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
            num_classes = 1
        else:
            label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
            labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
            num_classes = len(unique_labels_from_partition)
        print(f"Detected {num_classes} communities for {graph_n_value_str} using Louvain algorithm.")
        return labels, num_classes

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        print(f"Generating 'next_node' labels for graph {graph_n_value_str} (using A_out_w)...")
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1

        adj_out_weighted = graph.A_out_w  # Use the raw weighted for this task logic
        labels_list = [-1] * num_nodes
        for i in range(num_nodes):
            successors = np.where(adj_out_weighted[i, :] > 0)[0]
            weights = adj_out_weighted[i, successors]
            if len(successors) > 0:
                max_weight = np.max(weights)
                highest_prob_successors = successors[weights == max_weight]
                labels_list[i] = random.choice(highest_prob_successors)
            else:
                labels_list[i] = i  # Self-loop if no successors
        final_labels = torch.tensor(labels_list, dtype=torch.long)
        print(f"Finished 'next_node' label generation. Task output classes: {num_nodes}.")
        return final_labels, num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        # For BFS, we typically use unweighted connections.
        # We can use the binary out-adjacency matrix or derive from A_out_w.
        # Let's use A_out_w to find neighbors (any non-zero weight means an edge)
        adj_out_for_bfs = graph.A_out_w

        if not hasattr(graph, 'node_sequences'):
            # ... (logic to derive node_sequences if missing, same as before) ...
            if hasattr(graph, 'idx_to_node'):
                graph.node_sequences = [graph.idx_to_node[i] for i in range(num_nodes)] if isinstance(graph.idx_to_node, dict) else list(graph.idx_to_node)
            else:
                raise AttributeError("Graph needs 'node_sequences' or 'idx_to_node'")
        node_sequences = graph.node_sequences
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        print(f"Generating 'closest_amino_acid' labels for graph {graph_n_value_str} (k_hops={k_hops})...")
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        if k_hops < 0: raise ValueError("k_hops must be non-negative.")

        labels_for_nodes = torch.full((num_nodes,), k_hops, dtype=torch.long)  # Default to max_hops if not found earlier

        for start_node_idx in tqdm(range(num_nodes), desc=f"Generating closest AA labels for n={graph.n_value}"):
            target_aa = random.choice(AMINO_ACID_ALPHABET)
            if target_aa in node_sequences[start_node_idx]:
                labels_for_nodes[start_node_idx] = 0
                continue

            if k_hops > 0:
                q = collections.deque([(start_node_idx, 0)])
                visited = {start_node_idx}
                found_at_hop = -1

                while q:
                    curr_node, hop_level = q.popleft()
                    if hop_level >= k_hops: continue  # Max search depth reached for this path

                    # Get neighbors from adj_out_for_bfs
                    # For dense A_out_w, neighbors are where A_out_w[curr_node, :] > 0
                    neighbors = np.where(adj_out_for_bfs[curr_node, :] > 0)[0]

                    for neighbor_node in neighbors:
                        if neighbor_node not in visited:
                            visited.add(neighbor_node)
                            if target_aa in node_sequences[neighbor_node]:
                                labels_for_nodes[start_node_idx] = hop_level + 1
                                found_at_hop = hop_level + 1
                                break
                            q.append((neighbor_node, hop_level + 1))
                    if found_at_hop != -1: break

        num_output_classes = k_hops + 1
        print(f"Finished 'closest_amino_acid' label generation. Task output classes: {num_output_classes}.")
        return labels_for_nodes, num_output_classes

    def run(self):
        print("\n" + "=" * 80)
        print("### PIPELINE STEP 2: Training GCN Model and Generating Embeddings ###")
        print("=" * 80)
        os.makedirs(self.config.GCN_EMBEDDINGS_DIR, exist_ok=True)

        print("\n--- Step 1: Generating Protein ID Mapping ---")
        # Assuming FastaParser is now in data_loader and handles ID mapping
        id_mapper_instance = DataLoader(config=self.config)
        self.id_map = id_mapper_instance.generate_id_maps()
        print(f"Using device: {self.device}")

        level_embeddings: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)

        for n_val in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            print(f"\n--- Training N-gram Level: n = {n_val} ---")
            graph_path = os.path.join(self.config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")

            try:
                with open(graph_path, 'rb') as f:
                    # When loading, DirectedNgramGraphForGCN will be instantiated.
                    # We need to ensure its __init__ can receive epsilon.
                    # For now, assuming it's loaded and then we can access its attributes.
                    # If graph_obj is saved without epsilon, we might need to re-process or pass it.
                    # Let's assume the loaded graph_obj is an instance of the *new* DirectedNgramGraphForGCN
                    # or we re-create it with the loaded nodes/edges and the new epsilon.

                    # Simplest for now: load nodes/edges, then create new graph object
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, tuple) and len(loaded_data) == 2:  # Assuming (nodes, edges)
                        nodes_data, edges_data = loaded_data
                        graph_obj = DirectedNgramGraph(nodes_data, edges_data, epsilon_propagation=self.gcn_propagation_epsilon)
                    elif hasattr(loaded_data, 'nodes_map') and hasattr(loaded_data, 'original_edges'):  # If it's an old graph object
                        graph_obj = DirectedNgramGraph(loaded_data.nodes_map, loaded_data.original_edges, epsilon_propagation=self.gcn_propagation_epsilon)
                    elif isinstance(loaded_data, DirectedNgramGraph):  # If it's already the new type (less likely if re-running)
                        graph_obj = loaded_data
                        graph_obj.epsilon_propagation = self.gcn_propagation_epsilon  # Ensure epsilon is set
                        # Re-create propagation matrices if epsilon changed or they weren't saved
                        if graph_obj.number_of_nodes > 0:
                            graph_obj._create_propagation_matrices()
                    else:
                        raise TypeError("Loaded graph object is not in expected format.")

                required_attrs = ['mathcal_A_out', 'mathcal_A_in', 'node_to_idx', 'number_of_nodes']
                if not all(hasattr(graph_obj, attr) for attr in required_attrs):
                    print(f"ERROR: Graph object for n={n_val} is missing required attributes after processing.")
                    continue
                graph_obj.n_value = n_val  # Add n_value for logging in label generation
            except FileNotFoundError:
                print(f"ERROR: Graph object not found at {graph_path}.")
                continue
            except Exception as e:
                print(f"ERROR: Could not load or process graph for n={n_val}: {e}")
                import traceback
                traceback.print_exc()
                continue

            level_ngram_to_idx[n_val] = graph_obj.node_to_idx
            if graph_obj.number_of_nodes == 0:
                print(f"Skipping n={n_val} (0 nodes).")
                level_embeddings[n_val] = np.array([])
                continue

            current_task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n_val, self.config.GCN_DEFAULT_TASK_TYPE)
            print(f"Selected training task for n={n_val}: '{current_task_type}'")

            if n_val == 1:
                num_initial_features = self.config.GCN_1GRAM_INIT_DIM
                x = torch.randn(graph_obj.number_of_nodes, num_initial_features, device=self.device)
            else:
                if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                    print(f"ERROR: Prev level n={n_val - 1} embeddings missing/empty for n={n_val}.")
                    continue
                prev_embeds = level_embeddings[n_val - 1]
                prev_map = level_ngram_to_idx[n_val - 1]
                num_initial_features = prev_embeds.shape[1]
                x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float, device=self.device)
                for ngram, idx in graph_obj.node_to_idx.items():
                    prev_ngram = ngram[:-1]
                    prev_idx = prev_map.get(prev_ngram)
                    if prev_idx is not None and prev_idx < len(prev_embeds):
                        x[idx] = torch.from_numpy(prev_embeds[prev_idx]).to(self.device)

            labels, num_classes = None, None
            if current_task_type == "community":
                labels, num_classes = self._generate_community_labels(graph_obj)
            elif current_task_type == "next_node":
                labels, num_classes = self._generate_next_node_labels(graph_obj)
            elif current_task_type == "closest_aa":
                k_hops_for_task = getattr(self.config, 'GCN_CLOSEST_AA_K_HOPS', 3)
                labels, num_classes = self._generate_closest_amino_acid_labels(graph_obj, k_hops_for_task)
            else:
                raise ValueError(f"Unsupported GCN_TASK_TYPE '{current_task_type}' for n={n_val}.")

            if graph_obj.number_of_nodes > 0 and (labels is None or labels.numel() == 0):
                print(f"Warning: No labels generated for n={n_val}, task '{current_task_type}'. Using zeros.")
                if num_classes is None or num_classes == 0: num_classes = 1  # Default to 1 class
                labels = torch.zeros(graph_obj.number_of_nodes, dtype=torch.long)

            labels = labels.to(self.device)

            # Convert dense mathcal_A matrices to sparse edge_index and edge_weight
            edge_index_in, edge_weight_in = dense_to_sparse(torch.from_numpy(graph_obj.mathcal_A_in))
            edge_index_out, edge_weight_out = dense_to_sparse(torch.from_numpy(graph_obj.mathcal_A_out))

            data = Data(x=x.to(self.device), y=labels.to(self.device), edge_index_in=edge_index_in.to(self.device), edge_weight_in=edge_weight_in.float().to(self.device), edge_index_out=edge_index_out.to(self.device),
                        edge_weight_out=edge_weight_out.float().to(self.device))

            full_layer_dims = [num_initial_features] + self.config.GCN_HIDDEN_LAYER_DIMS
            model = ProtGramDirectGCN(  # Assuming ProtNgramGCN uses the edge_weights as is
                layer_dims=full_layer_dims, num_graph_nodes=graph_obj.number_of_nodes, task_num_output_classes=num_classes, n_gram_len=n_val,
                one_gram_dim=(self.config.GCN_1GRAM_INIT_DIM if n_val == 1 and self.config.GCN_1GRAM_INIT_DIM > 0 and self.config.GCN_MAX_PE_LEN > 0 else 0), max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE, use_vector_coeffs=getattr(self.config, 'GCN_USE_VECTOR_COEFFS', True)  # Add default if not in config
            )

            current_optimizer_weight_decay = self.config.GCN_WEIGHT_DECAY
            if l2_lambda_val > 0:
                print(f"Explicit L2 (lambda={l2_lambda_val}) added to loss; optimizer weight_decay {self.config.GCN_WEIGHT_DECAY} -> 0.0.")
                current_optimizer_weight_decay = 0.0
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=current_optimizer_weight_decay)

            self._train_model(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, l2_lambda_val)
            current_level_embeddings = EmbeddingProcessor.extract_gcn_node_embeddings(model, data, self.device)
            level_embeddings[n_val] = current_level_embeddings

            del model, data, graph_obj, optimizer, x, labels, edge_index_in, edge_weight_in, edge_index_out, edge_weight_out
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        # ... (Pooling and Saving steps remain the same) ...
        print("\n--- Step 3: Pooling N-gram Embeddings to Protein Level ---")
        final_n_val = self.config.GCN_NGRAM_MAX_N
        if final_n_val not in level_embeddings or level_embeddings[final_n_val].size == 0:
            print(f"ERROR: Final n={final_n_val} embeddings missing/empty. Cannot pool.")
            return

        final_ngram_embeds = level_embeddings[final_n_val]
        final_ngram_map = level_ngram_to_idx[final_n_val]

        # Ensure GCN_INPUT_FASTA_PATH is correctly accessed from config
        protein_sequences_path = str(self.config.GCN_INPUT_FASTA_PATH)
        protein_sequences = list(DataLoader.parse_sequences(protein_sequences_path))

        pool_func = partial(EmbeddingProcessor.pool_ngram_embeddings_for_protein, n_val=final_n_val, ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds)

        pooled_embeddings = {}
        num_pooling_workers = self.config.POOLING_WORKERS if self.config.POOLING_WORKERS is not None else os.cpu_count()
        if num_pooling_workers is None or num_pooling_workers < 1: num_pooling_workers = 1

        with Pool(processes=num_pooling_workers) as pool:
            for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="Pooling Protein Embeddings"):
                if vec is not None:
                    final_key = self.id_map.get(original_id, original_id)
                    pooled_embeddings[final_key] = vec
        if not pooled_embeddings: print("Warning: No protein embeddings after pooling.")

        print("\n--- Step 4: Saving Generated Embeddings ---")
        output_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            for key, vector in tqdm(pooled_embeddings.items(), desc="Writing H5 File"):
                if vector is not None and vector.size > 0:
                    hf.create_dataset(key, data=vector)
        print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

        if self.config.APPLY_PCA_TO_GCN and pooled_embeddings:
            print("\n--- Step 5: Applying PCA for Dimensionality Reduction ---")
            valid_pooled_embeddings = {k: v for k, v in pooled_embeddings.items() if v is not None and v.size > 0}
            if not valid_pooled_embeddings:
                print("Warning: No valid embeddings to apply PCA.")
            else:
                pca_embeds = EmbeddingProcessor.apply_pca(valid_pooled_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if pca_embeds:
                    first_valid_pca_emb = next((v for v in pca_embeds.values() if v is not None and v.size > 0), None)
                    if first_valid_pca_emb is not None:
                        pca_dim = first_valid_pca_emb.shape[0]
                        pca_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
                        with h5py.File(pca_h5_path, 'w') as hf:
                            for key, vector in tqdm(pca_embeds.items(), desc="Writing PCA H5 File"):
                                if vector is not None and vector.size > 0: hf.create_dataset(key, data=vector)
                        print(f"SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
                    else:
                        print("PCA Warning: No valid PCA embeddings to determine dimension for saving.")
                elif pooled_embeddings:
                    print("Warning: PCA was requested but resulted in no embeddings.")
        print("\n### PIPELINE STEP 2 FINISHED ###")
