# src/pipeline/protgram_directgcn_trainer.py
# ==============================================================================
# MODULE: pipeline/protgram_directgcn_trainer.py
# PURPOSE: Trains the ProtGramDirectGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 4.10 (Corrected clustering to use combined mathcal_A matrices)
# AUTHOR: Islam Ebeid
# ==============================================================================

import collections
import gc
import os
import random
import math  # Import math for ceil function
from typing import Dict, Tuple, Optional, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
from tqdm import tqdm

from config import Config
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataLoader, DataUtils
from src.utils.graph_utils import DirectedNgramGraph
from src.utils.models_utils import EmbeddingProcessor

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class ProtGramDirectGCNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id_map: Dict[str, str] = {}
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        DataUtils.print_header("ProtGramDirectGCNEmbedder Initialized")

    def _train_model_full_batch(self, model: ProtGramDirectGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                task_type: str, l2_lambda: float = 0.0):
        """Original training method for smaller graphs, now with Mixed Precision."""
        model.train()
        model.to(self.device)
        data = data.to(self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        criterion = F.nll_loss

        print(f"  Starting full-batch training for {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                task_output, _ = model(data=data)
                primary_loss = criterion(task_output, data.y)
                l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                loss = primary_loss + l2_lambda * l2_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch % (max(1, epochs // 10)) == 0 or epoch == epochs:
                if self.config.DEBUG_VERBOSE:
                    print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * l2_reg).item():.4f}")

    def _train_model_clustered(self, model: ProtGramDirectGCN, subgraphs: List[Data], optimizer: torch.optim.Optimizer,
                               epochs: int, task_type: str, l2_lambda: float = 0.0):
        """Cluster-GCN style training for large graphs with Mixed Precision."""
        model.train()
        model.to(self.device)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.nll_loss

        print(f"  Starting Cluster-GCN style training for {epochs} epochs on {len(subgraphs)} subgraphs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            random.shuffle(subgraphs)
            epoch_loss = 0.0
            for batch_data in tqdm(subgraphs, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    task_output, _ = model(data=batch_data)
                    primary_loss = criterion(task_output, batch_data.y)
                    l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                    loss = primary_loss + l2_lambda * l2_reg

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(subgraphs)
            if epoch % (max(1, epochs // 10)) == 0 or epoch == epochs:
                if self.config.DEBUG_VERBOSE:
                    print(f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}")

    def _create_clustered_subgraphs(self, graph: DirectedNgramGraph, full_data: Data) -> List[Data]:
        """Partitions the graph into subgraphs using METIS or Louvain."""

        # Dynamically determine num_clusters based on graph size and target nodes per cluster
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS)

        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters (target nodes/cluster: {self.config.GCN_TARGET_NODES_PER_CLUSTER})...")

        # --- MODIFIED: Use combined mathcal_A matrices for clustering ---
        # mathcal_A_out and mathcal_A_in are already sparse tensors on the device.
        # Sum them to get a combined adjacency for clustering.
        # Move to CPU for to_networkx, and ensure it's undirected.
        mathcal_A_combined_sparse_cpu = (graph.mathcal_A_out + graph.mathcal_A_in).coalesce().cpu()

        # Create a PyG Data object from the combined sparse tensor for to_networkx
        g_nx = to_networkx(Data(edge_index=mathcal_A_combined_sparse_cpu.indices(),
                                edge_attr=mathcal_A_combined_sparse_cpu.values(),
                                num_nodes=graph.number_of_nodes),
                           to_undirected=True,
                           edge_attrs=['edge_attr']) # Pass edge_attr to preserve weights in NetworkX
        # --- END MODIFIED ---

        try:
            import metis
            print("  Using METIS for graph partitioning...")
            # metis.part_graph returns (edge_cut, parts)
            _, parts = metis.part_graph(g_nx, num_clusters, seed=self.config.RANDOM_STATE)
            partition = {node_idx: part_id for node_idx, part_id in enumerate(parts)}
        except (ImportError, ModuleNotFoundError):
            import community as community_louvain
            print("  METIS not found. Falling back to Louvain for clustering (slower)...")
            partition = community_louvain.best_partition(g_nx, random_state=self.config.RANDOM_STATE)

        clusters = collections.defaultdict(list)
        for node, cluster_id in partition.items():
            clusters[cluster_id].append(node)

        cluster_list = list(clusters.values())
        print(f"  Graph partitioned into {len(cluster_list)} clusters.")

        subgraphs = []
        for cluster_nodes in tqdm(cluster_list, desc="  Creating subgraphs", leave=False):
            # FIX: Move cluster_nodes_tensor to the same device as graph.mathcal_A_in.indices()
            cluster_nodes_tensor = torch.tensor(cluster_nodes, dtype=torch.long).to(self.device)

            # Create subgraph for both IN and OUT propagation matrices
            # These still use the separate mathcal_A_in and mathcal_A_out from the original graph_obj
            sub_edge_index_in, sub_edge_weight_in = subgraph(cluster_nodes_tensor, graph.mathcal_A_in.indices(), graph.mathcal_A_in.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
            sub_edge_index_out, sub_edge_weight_out = subgraph(cluster_nodes_tensor, graph.mathcal_A_out.indices(), graph.mathcal_A_out.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)

            subgraph_data = Data(
                x=full_data.x[cluster_nodes_tensor],
                y=full_data.y[cluster_nodes_tensor] if full_data.y.numel() > 0 else torch.empty(0),
                edge_index_in=sub_edge_index_in,
                edge_weight_in=sub_edge_weight_in,
                edge_index_out=sub_edge_index_out,
                edge_weight_out=sub_edge_weight_out,
                original_indices=cluster_nodes_tensor
            )
            subgraphs.append(subgraph_data)

        return subgraphs

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        import networkx as nx
        import community as community_louvain
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating community labels for all {num_nodes} nodes.")
        A_in_w_cpu_sparse = graph.A_in_w.cpu()
        A_out_w_cpu_sparse = graph.A_out_w.cpu()
        combined_adj_sparse_torch = (A_in_w_cpu_sparse + A_out_w_cpu_sparse).coalesce()
        if combined_adj_sparse_torch._nnz() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1
        # Use to_networkx from torch_geometric.utils to handle sparse tensor conversion
        nx_graph = to_networkx(Data(edge_index=combined_adj_sparse_torch.indices(), edge_attr=combined_adj_sparse_torch.values(), num_nodes=num_nodes), to_undirected=True, edge_attrs=['edge_attr'])
        if nx_graph.number_of_edges() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1
        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE, weight='edge_attr')
        labels_list = [partition.get(i, -1) for i in range(num_nodes)]
        unique_labels_from_partition = sorted(list(set(labels_list)))
        if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
            return torch.zeros(num_nodes, dtype=torch.long), 1
        else:
            label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
            labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
            return labels, len(unique_labels_from_partition)

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating next_node labels for all {num_nodes} nodes.")
        adj_out_weighted_sparse = graph.A_out_w
        labels_list = [-1] * num_nodes
        for i in tqdm(range(num_nodes), desc="  Generating next_node labels", disable=not self.config.DEBUG_VERBOSE):
            row_mask = (adj_out_weighted_sparse.indices()[0] == i)
            if not torch.any(row_mask):
                labels_list[i] = i
            else:
                successors = adj_out_weighted_sparse.indices()[1][row_mask]
                weights = adj_out_weighted_sparse.values()[row_mask]
                max_weight_successors = successors[weights == weights.max()]
                labels_list[i] = random.choice(max_weight_successors.cpu().tolist())
        return torch.tensor(labels_list, dtype=torch.long), num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        labels_for_all_nodes = torch.full((num_nodes,), k_hops, dtype=torch.long)
        print(f"  Generating closest_aa labels for all {num_nodes} nodes.")
        adj_out_sparse = graph.A_out_w
        node_sequences = graph.node_sequences
        for start_node_idx in tqdm(range(num_nodes), desc="  Generating closest_aa labels", disable=not self.config.DEBUG_VERBOSE):
            target_aa = random.choice(AMINO_ACID_ALPHABET)
            if target_aa in str(node_sequences[start_node_idx]):
                labels_for_all_nodes[start_node_idx] = 0
            elif k_hops > 0:
                q = collections.deque([(start_node_idx, 0)])
                visited = {start_node_idx}
                found = False
                while q:
                    curr, hop = q.popleft()
                    if hop >= k_hops: continue
                    row_mask = (adj_out_sparse.indices()[0] == curr)
                    if not torch.any(row_mask): continue
                    for neighbor in adj_out_sparse.indices()[1][row_mask]:
                        n_idx = neighbor.item()
                        if n_idx not in visited:
                            visited.add(n_idx)
                            if target_aa in str(node_sequences[n_idx]):
                                labels_for_all_nodes[start_node_idx] = hop + 1
                                found = True
                                break
                            if hop + 1 < k_hops: q.append((n_idx, hop + 1))
                    if found: break
        return labels_for_all_nodes, k_hops + 1

    def run(self):
        DataUtils.print_header("PIPELINE STEP 2: Training ProtGramDirectGCN & Generating Embeddings")
        os.makedirs(self.config.GCN_EMBEDDINGS_DIR, exist_ok=True)
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if self.config.ID_MAPPING_MODE != 'none':
            id_mapper_instance = DataLoader(config=self.config)
            self.id_map = id_mapper_instance.generate_id_maps()
            print(f"  Loaded {len(self.id_map)} ID mappings.")
        else:
            self.id_map = {}
        print(f"Using device: {self.device}")
        level_embeddings: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)

        for n_val in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n_val}")
            graph_path = os.path.join(self.config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")
            try:
                graph_obj = DataUtils.load_object(graph_path)
                if not isinstance(graph_obj, DirectedNgramGraph): raise TypeError("Loaded object is not a DirectedNgramGraph")
                graph_obj.n_value = n_val
                if graph_obj.number_of_nodes > 0:
                    graph_obj.A_out_w = graph_obj.A_out_w.to(self.device)
                    graph_obj.A_in_w = graph_obj.A_in_w.to(self.device)
                    print("    Re-creating propagation matrices for loaded graph object...")
                    graph_obj._create_propagation_matrices_for_gcn()
            except Exception as e:
                print(f"  ERROR: Could not load or process graph for n={n_val}: {e}")
                continue
            if not graph_obj or graph_obj.number_of_nodes == 0:
                print(f"  Skipping n={n_val} (no nodes or failed to load).")
                level_embeddings[n_val] = np.array([])
                continue
            level_ngram_to_idx[n_val] = graph_obj.node_to_idx
            print(f"  Graph for n={n_val} loaded. Nodes: {graph_obj.number_of_nodes}")
            current_task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n_val, self.config.GCN_DEFAULT_TASK_TYPE)
            print(f"  Selected training task for n={n_val}: '{current_task_type}'")

            num_initial_features: int
            if n_val == 1:
                num_initial_features = self.config.GCN_1GRAM_INIT_DIM
                x = torch.randn(graph_obj.number_of_nodes, num_initial_features, device=self.device)
            else:
                if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                    print(f"  ERROR: Prev level n={n_val - 1} embeddings missing/empty. Skipping.")
                    continue
                prev_level_embeds_np = level_embeddings[n_val - 1]
                prev_level_ngram_to_idx_map = level_ngram_to_idx[n_val - 1]
                num_initial_features = prev_level_embeds_np.shape[1]
                x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float, device=self.device)
                print(f"  Initializing features for n={n_val} by pooling (n-1)-gram constituent embeddings...")
                for ngram_str, idx in tqdm(graph_obj.node_to_idx.items(), desc=f"  Initializing n={n_val} features", leave=False, disable=not self.config.DEBUG_VERBOSE):
                    prefix, suffix = ngram_str[:-1], ngram_str[1:]
                    p_idx, s_idx = prev_level_ngram_to_idx_map.get(prefix), prev_level_ngram_to_idx_map.get(suffix)
                    embeds_to_pool = [prev_level_embeds_np[i] for i in [p_idx, s_idx] if i is not None]
                    if embeds_to_pool:
                        x[idx] = torch.from_numpy(np.mean(np.array(embeds_to_pool, dtype=np.float32), axis=0)).to(self.device)
            print(f"  Initial node feature dimension for n={n_val}: {num_initial_features}")

            labels, num_classes = None, 0
            if current_task_type == "community":
                labels, num_classes = self._generate_community_labels(graph_obj)
            elif current_task_type == "next_node":
                labels, num_classes = self._generate_next_node_labels(graph_obj)
            elif current_task_type == "closest_aa":
                labels, num_classes = self._generate_closest_amino_acid_labels(graph_obj, self.config.GCN_CLOSEST_AA_K_HOPS)
            else:
                raise ValueError(f"Unsupported GCN_TASK_TYPE '{current_task_type}' for n={n_val}.")

            full_data = Data(x=x, y=labels.to(self.device))
            full_data.num_nodes = graph_obj.number_of_nodes

            full_layer_dims = [num_initial_features] + self.config.GCN_HIDDEN_LAYER_DIMS
            model = ProtGramDirectGCN(
                layer_dims=full_layer_dims, num_graph_nodes=graph_obj.number_of_nodes,
                task_num_output_classes=num_classes, n_gram_len=n_val,
                one_gram_dim=(self.config.GCN_1GRAM_INIT_DIM if n_val == 1 else 0),
                max_pe_len=self.config.GCN_MAX_PE_LEN, dropout=self.config.GCN_DROPOUT_RATE,
                use_vector_coeffs=self.config.GCN_USE_VECTOR_COEFFS
            )
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY if l2_lambda_val <= 0 else 0.0)

            if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
                subgraphs = self._create_clustered_subgraphs(graph_obj, full_data)
                self._train_model_clustered(model, subgraphs, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, current_task_type, l2_lambda_val)
            else:
                full_data.edge_index_in = graph_obj.mathcal_A_in.indices().to(self.device)
                full_data.edge_weight_in = graph_obj.mathcal_A_in.values().to(self.device)
                full_data.edge_index_out = graph_obj.mathcal_A_out.indices().to(self.device)
                full_data.edge_weight_out = graph_obj.mathcal_A_out.values().to(self.device)
                self._train_model_full_batch(model, full_data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, current_task_type, l2_lambda_val)

            if not hasattr(full_data, 'edge_index_in'):
                full_data.edge_index_in = graph_obj.mathcal_A_in.indices().to(self.device)
                full_data.edge_weight_in = graph_obj.mathcal_A_in.values().to(self.device)
                full_data.edge_index_out = graph_obj.mathcal_A_out.indices().to(self.device)
                full_data.edge_weight_out = graph_obj.mathcal_A_out.values().to(self.device)
            current_level_embeddings = EmbeddingProcessor.extract_gcn_node_embeddings(model, full_data, self.device)
            level_embeddings[n_val] = current_level_embeddings

            del model, full_data, graph_obj, optimizer, x, labels
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        DataUtils.print_header("Step 3: Pooling N-gram Embeddings to Protein Level")
        final_n_val = self.config.GCN_NGRAM_MAX_N
        if final_n_val not in level_embeddings or level_embeddings[final_n_val].size == 0:
            print(f"ERROR: Final n={final_n_val} embeddings missing/empty. Cannot pool.")
            return
        final_ngram_embeds = level_embeddings[final_n_val]
        final_ngram_map = level_ngram_to_idx[final_n_val]
        protein_sequences = list(DataLoader.parse_sequences(str(self.config.GCN_INPUT_FASTA_PATH)))
        pooled_embeddings = EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences, n_val=final_n_val,
            ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds
        )
        if self.id_map:
            pooled_embeddings = {self.id_map.get(k, k): v for k, v in pooled_embeddings.items()}
        DataUtils.print_header("Step 4: Saving Generated Embeddings")
        output_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            for key, vector in tqdm(pooled_embeddings.items(), desc="  Writing H5 File"):
                if vector is not None: hf.create_dataset(key, data=vector)
        print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")
        if self.config.APPLY_PCA_TO_GCN and pooled_embeddings:
            DataUtils.print_header("Step 5: Applying PCA for Dimensionality Reduction")
            pca_embeds = EmbeddingProcessor.apply_pca(pooled_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
            if pca_embeds:
                pca_dim = next(iter(pca_embeds.values())).shape[0]
                pca_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
                with h5py.File(pca_h5_path, 'w') as hf:
                    for key, vector in tqdm(pca_embeds.items(), desc="  Writing PCA H5 File"):
                        if vector is not None: hf.create_dataset(key, data=vector)
                print(f"  SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
        DataUtils.print_header("ProtGramDirectGCN Embedding PIPELINE STEP FINISHED")
