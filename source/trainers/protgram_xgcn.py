# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs.
# VERSION: 7.1 (Corrected data flow for DirectGCN and integrated homophily paths)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import collections
import copy
import gc
import json
import math
import random
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Tuple, Any

import community as community_louvain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import homophily
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataset
from source.experiments.ppi_1 import PPIPipeline
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.tongidigcn import TongDiGCN
from source.utils.data import DataUtils, IDMapGenerator, FastaUtils
from source.utils.models import EmbeddingProcessor, EarlyStopper
from source.utils.results import EvaluationReporter


class ProtGramXGCNTrainer:
    """
    Orchestrates the training of different GNN models on the pre-built
    ProtGram n-gram graphs and generates final protein-level embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataset(config)
        print(f"Using device: {self.device}")
        self._loaded_graphs: Dict[int, DirectedNgramGraph] = {}

    def run(self) -> Dict[str, str]:
        """
        Main execution function. Loops through n-gram levels, trains models,
        and generates final protein-level embeddings.
        """
        DataUtils.print_header("PIPELINE STEP: Training ProtGram Models & Generating Embeddings")

        final_protein_embeddings_per_model = {}
        all_attention_data_per_model: Dict[str, Dict[str, Any]] = {}
        id_map = self._load_id_map()

        context = id_map if isinstance(id_map, IDMapGenerator) else nullcontext(id_map)

        with context as mapper:
            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                DataUtils.print_header(f"Processing Model Type: {model_type.upper()}")
                ngram_embeddings_per_level, hierarchical_attention = self._train_gnns_hierarchically(model_type)

                final_protein_embeddings, protein_pooling_attention = self._pool_to_protein_level(
                    ngram_embeddings_per_level,
                    level_ngram_to_idx=self._get_level_ngram_maps()
                )

                if mapper and final_protein_embeddings:
                    print("  Applying ID mapping to final protein embeddings...")
                    final_protein_embeddings = {mapper.get(k, k): v for k, v in final_protein_embeddings.items()}

                final_protein_embeddings_per_model[model_type] = final_protein_embeddings
                all_attention_data_per_model[model_type] = {
                    "hierarchical": hierarchical_attention,
                    "protein_pooling": protein_pooling_attention
                }

        output_paths = self._save_final_embeddings(final_protein_embeddings_per_model)

        # Save and visualize all collected attention data
        self._save_and_visualize_attention(all_attention_data_per_model)

        if self.config.GCN_RUN_SANITY_CHECK_PPI:
            main_model_name_raw = self.config.PROTGRAM_MODELS_TO_TRAIN[0]
            main_model_key = f"ProtGram{main_model_name_raw.capitalize()}"
            # --- FIX: Prioritize the original, non-PCA'd file for the sanity check ---
            # The evaluation pipeline will handle its own PCA, so we avoid double-processing.
            embedding_path_for_check = output_paths.get(main_model_key, output_paths.get(f"{main_model_key}_pca"))
            if embedding_path_for_check:
                self._run_sanity_check_ppi(embedding_path_for_check)

        DataUtils.print_header("ProtGram Embedding PIPELINE STEP FINISHED")
        return output_paths

    def _load_graph_for_level(self, n: int) -> Optional[DirectedNgramGraph]:
        """Loads the graph object for a specific n-gram level, using a cache."""
        if n in self._loaded_graphs:
            return self._loaded_graphs[n]

        graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}.pkl"
        if not graph_obj_path.exists():
            print(f"  Graph object not found for n={n}. Skipping.")
            return None

        graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
        if graph_obj is None or graph_obj.number_of_nodes == 0:
            print(f"  Failed to load graph object or graph is empty for n={n}. Skipping.")
            return None

        self._loaded_graphs[n] = graph_obj
        print(f"  Graph for n={n} loaded. Nodes: {graph_obj.number_of_nodes}")
        # Ensure matrices are on CPU for potential multiprocessing in label generation
        graph_obj.A_out_w = graph_obj.A_out_w.cpu()
        graph_obj.A_in_w = graph_obj.A_in_w.cpu()
        graph_obj.A_undirected_norm_sparse = graph_obj.A_undirected_norm_sparse.cpu()
        # This is for other models, DirectGCN does not use it.
        graph_obj._create_propagation_matrices_for_gcn()
        return graph_obj

    def _get_initial_features_for_level(self, n: int, graph_obj: DirectedNgramGraph,
                                        prev_level_embeddings: Optional[np.ndarray],
                                        prev_level_map: Optional[Dict[str, int]]) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Generates the initial node features for the current n-gram level."""
        if n == 1:
            features = torch.randn((graph_obj.number_of_nodes, self.config.GCN_1GRAM_INIT_DIM))
            return features, {}
        else:
            if prev_level_embeddings is None or prev_level_embeddings.size == 0 or prev_level_map is None:
                print(f"  Cannot proceed for n={n}, previous level embeddings not found or empty.")
                return None
            result = EmbeddingProcessor.pool_lower_level_embeddings_for_init(
                graph_obj, prev_level_embeddings, prev_level_map,
                strategy=self.config.GCN_HIERARCHICAL_POOLING_STRATEGY)
            if result is None:
                return None
            features, attention_log = result
            return features, attention_log

    def _train_gnns_hierarchically(self, model_type: str) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """The main hierarchical training loop."""
        ngram_embeddings_per_level: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        hierarchical_attention_per_level: Dict[int, Dict] = {}

        for n in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n} for model '{model_type}'")

            graph_obj = self._load_graph_for_level(n)
            if not graph_obj: continue

            level_ngram_to_idx[n] = graph_obj.get_node_to_idx_map()
            prev_embeds = ngram_embeddings_per_level.get(n - 1)
            prev_map = level_ngram_to_idx.get(n - 1)

            feature_result = self._get_initial_features_for_level(n, graph_obj, prev_embeds, prev_map)
            if feature_result is None: continue
            initial_features, hierarchical_attention = feature_result

            if hierarchical_attention:
                hierarchical_attention_per_level[n] = hierarchical_attention

            task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n, self.config.GCN_DEFAULT_TASK_TYPE)
            labels, num_classes_for_task = self.label_generator.generate_task_labels(graph_obj, task_type)

            # --- NEW: Dynamic Architecture Selection for DirectGCN ---
            # Calculate homophily to decide if specialized paths should be used for this level.
            use_homo_hetero_paths_for_level = False
            if model_type == 'directgcn':
                homophily_ratio = homophily(graph_obj.A_undirected_norm_sparse.indices(), labels, method='edge')
                is_heterophilic = homophily_ratio < 0.6  # Standard threshold
                print(f"  Graph n={n} Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
                if is_heterophilic:
                    print(f"  -> Enabling specialized homophily/heterophily paths for DirectGCN at n={n}.")
                    use_homo_hetero_paths_for_level = True

            if use_homo_hetero_paths_for_level:
                graph_obj.split_edges_by_homophily(labels)

            model = self._build_model(model_type, n, initial_features.shape[1], num_classes_for_task, graph_obj, use_homo_hetero_paths_for_level)
            if model is None: continue

            data = Data(x=initial_features, y=labels, graph_obj=graph_obj)
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY)

            self._train_single_level(model, graph_obj, data, optimizer, use_homo_hetero_paths_for_level)

            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(
                model, data, graph_obj, self.config, self.device, use_homo_hetero_paths_for_level,
                lambda g: self._partition_graph(g)
            )

            print(f"  Generated {ngram_embeddings_per_level[n].shape[0]} embeddings of dim {ngram_embeddings_per_level[n].shape[1]} for n={n}.")
            del model, data, graph_obj, initial_features, labels, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ngram_embeddings_per_level, hierarchical_attention_per_level

    def _build_model(self, model_type: str, n_val: int, in_channels: int, num_classes: int, graph_obj: DirectedNgramGraph, use_homo_hetero_paths: bool) -> Optional[nn.Module]:
        """Model factory for creating different GNN architectures."""
        layer_dims = [in_channels] + self.config.GCN_HIDDEN_LAYER_DIMS
        if num_classes <= 0: num_classes = 1

        if model_type == 'directgcn':
            return DirectGCN(
                layer_dims=layer_dims, num_graph_nodes=graph_obj.number_of_nodes,
                task_num_output_classes=num_classes,
                n_gram_len=n_val,
                use_homo_hetero_paths=use_homo_hetero_paths,
                one_gram_dim=self.config.GCN_1GRAM_INIT_DIM, max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE, gating_mode=self.config.GCN_GATING_COEFF_MODE
            )
        elif model_type == 'rgcn':
            return RGCN(in_channels, self.config.GCN_HIDDEN_LAYER_DIMS[-1], num_classes, num_relations=2)
        elif model_type == 'tongdigcn':
            return TongDiGCN(in_channels, self.config.GCN_HIDDEN_LAYER_DIMS[0], num_classes)
        else:
            print(f"  ERROR: Unknown model type '{model_type}' for ProtGram training.")
            return None

    def _train_single_level(self, model: nn.Module, graph_obj: DirectedNgramGraph, data: Data, optimizer: torch.optim.Optimizer, use_homo_hetero_paths: bool):
        """Orchestrates the training for a single level, choosing between full-batch and clustered training."""
        task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(graph_obj.n_value, self.config.GCN_DEFAULT_TASK_TYPE)

        if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
            node_partitions = self._partition_graph(graph_obj)
            self._train_single_level_clustered(model, data, node_partitions, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, use_homo_hetero_paths)
        else:
            self._train_single_level_full_batch(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, use_homo_hetero_paths)

    def _train_single_level_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                       task_type: str, use_homo_hetero_paths: bool):
        """Full-batch training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        full_data_gpu = self._prepare_data_for_model(model.__class__.__name__.lower(), data.graph_obj, data.x, data.y, use_homo_hetero_paths).to(self.device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, Weight Decay: {optimizer.param_groups[0]['weight_decay']})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output, _ = model(data=full_data_gpu)
                loss = criterion(output, full_data_gpu.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step(loss)
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Loss: {loss.item():.4f}")
            if early_stopper and early_stopper.early_stop(loss.item()):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _train_single_level_clustered(self, model: nn.Module, full_data: Data, node_partitions: List[List[int]],
                                      optimizer: torch.optim.Optimizer, epochs: int, task_type: str, use_homo_hetero_paths: bool):
        """Clustered training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(node_partitions)} subgraphs (Task: {task_type})...")

        for epoch in range(1, epochs + 1):
            random.shuffle(node_partitions)
            epoch_loss = 0.0
            for node_idx_batch in tqdm(node_partitions, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                subgraph_data = full_data.graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=torch.tensor(node_idx_batch, dtype=torch.long),
                    use_homo_hetero_paths=use_homo_hetero_paths
                ).to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    output, _ = model(data=subgraph_data)
                    loss = criterion(output, subgraph_data.y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(node_partitions) if node_partitions else 0
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}")
            if scheduler: scheduler.step(avg_epoch_loss)
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _partition_graph(self, graph: DirectedNgramGraph) -> List[List[int]]:
        """
        Partitions the graph into clusters of nodes for batch training.
        This version is now model-agnostic and only returns the node indices for each partition.
        """
        if graph.number_of_nodes == 0: return []
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS, graph.number_of_nodes)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters...")

        A_combined_cpu = (graph.A_in_w.cpu() + graph.A_out_w.cpu()).coalesce()
        g_nx = to_networkx(Data(edge_index=A_combined_cpu.indices(), edge_attr=A_combined_cpu.values(), num_nodes=graph.number_of_nodes), to_undirected=True, edge_attrs=['edge_attr'])

        try:
            import metis
            print("  Using METIS for graph partitioning...")
            _, parts = metis.part_graph(g_nx, num_clusters, seed=self.config.RANDOM_STATE)
            partition = {node_idx: part_id for node_idx, part_id in enumerate(parts)}
        except (ImportError, ModuleNotFoundError):
            print("  METIS not found. Falling back to Louvain for clustering (slower)...")
            partition = community_louvain.best_partition(g_nx, random_state=self.config.RANDOM_STATE, weight='edge_attr')

        clusters = collections.defaultdict(list)
        for node, cluster_id in partition.items(): clusters[cluster_id].append(node)
        cluster_list = list(clusters.values())
        print(f"  Graph partitioned into {len(cluster_list)} clusters.")
        return cluster_list

    def _prepare_data_for_model(self, model_type: str, graph: DirectedNgramGraph, features: torch.Tensor, labels: torch.Tensor, use_homo_hetero_paths: bool) -> Data:
        """Prepares a PyG Data object tailored to the specific model's needs for full-batch training."""
        data_dict = {'x': features, 'y': labels}

        if model_type == 'directgcn':
            # --- DESIGN NOTE on DirectGCN Data ---
            # The DirectGCN model is designed to work with the raw, weighted adjacency
            # matrices. For the "Parallel Views" architecture, we provide 5 distinct
            # views when homophily/heterophily paths are enabled.

            # Always include the base structural and directional paths
            data_dict.update({
                'edge_index_in': graph.A_in_w.indices(), 'edge_weight_in': graph.A_in_w.values(),
                'edge_index_out': graph.A_out_w.indices(), 'edge_weight_out': graph.A_out_w.values(),
                'edge_index_undirected_norm': graph.A_undirected_norm_sparse.indices(),
                'edge_weight_undirected_norm': graph.A_undirected_norm_sparse.values()
            })

            # Conditionally add the new top-level homophily/heterophily paths
            if use_homo_hetero_paths and graph.A_homo_w is not None and graph.A_hetero_w is not None:
                # --- FIX: Corrected log message and added the missing logic to update the data object ---
                print("  Preparing data with parallel homophily/heterophily paths...")
                data_dict.update({
                    'edge_index_homo': graph.A_homo_w.indices(), 'edge_weight_homo': graph.A_homo_w.values(),
                    'edge_index_hetero': graph.A_hetero_w.indices(), 'edge_weight_hetero': graph.A_hetero_w.values()
                })
        elif model_type == 'rgcn':
            edge_index_out = graph.A_out_w.indices()
            edge_index_in = graph.A_in_w.indices()
            edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long)
            edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long)
            data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in], dim=0)
        elif model_type == 'tongdigcn':
            data_dict['edge_index'] = graph.A_out_w.indices()
            data_dict['edge_index_backward'] = graph.A_in_w.indices()
        else:
            print(f"  Note: Using standard undirected graph representation for model type '{model_type}'.")
            data_dict['edge_index'] = graph.A_undirected_norm_sparse.indices()
            data_dict['edge_attr'] = graph.A_undirected_norm_sparse.values()
        return Data.from_dict(data_dict)

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return None

    def _get_level_ngram_maps(self) -> Dict[int, Dict[str, int]]:
        """Returns the node_to_idx maps for all loaded graph levels."""
        return {n: graph.get_node_to_idx_map() for n, graph in self._loaded_graphs.items()}

    def _pool_to_protein_level(self, ngram_embeddings_per_level: Dict[int, np.ndarray],
                               level_ngram_to_idx: Dict[int, Dict[str, int]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, float]]]:
        """Pools the final n-gram embeddings to the protein level."""
        final_n = self.config.GCN_NGRAM_MAX_N
        final_level_embeddings = ngram_embeddings_per_level.get(final_n)
        final_level_map = level_ngram_to_idx.get(final_n)

        if final_level_embeddings is None or final_level_map is None:
            print("  Final level embeddings or map not found. Cannot perform protein-level pooling.")
            return {}, {}

        protein_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))

        return EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences,
            n_val=final_n,
            ngram_map=final_level_map,
            ngram_embeddings=final_level_embeddings,
            strategy=self.config.GCN_PROTEIN_POOLING_STRATEGY
        )

    def _save_final_embeddings(self, embeddings_per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, str]:
        """Saves final protein embeddings and their PCA versions to H5 files."""
        output_paths = {}
        for model_type, embeddings in embeddings_per_model.items():
            if not embeddings:
                print(f"  No embeddings generated for model '{model_type}'. Skipping save.")
                continue

            model_name = f"ProtGram{model_type.capitalize()}"
            output_dir = self.config.RESULTS_GCN_EMBEDDINGS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model_name}.h5"

            DataUtils.write_h5(embeddings, str(output_path), f"Writing H5 for {model_name}")
            output_paths[model_name] = str(output_path)

            # Apply PCA and save
            pca_path = EmbeddingProcessor.apply_pca_to_h5(
                input_h5_path=output_path,
                output_dir=output_dir,
                target_dimension=self.config.PCA_TARGET_DIMENSION,
                random_seed=self.config.RANDOM_STATE
            )
            if str(pca_path) != str(output_path):
                output_paths[f"{model_name}_pca"] = str(pca_path)

        return output_paths

    def _save_and_visualize_attention(self, attention_data: Dict[str, Dict[str, Any]]):
        """Saves attention weights to JSON and generates plots."""
        DataUtils.print_header("Saving and Visualizing Attention Weights")
        reporter = EvaluationReporter(str(self.config.RESULTS_EVALUATION_DIR), self.config.EVAL_K_VALUES_FOR_TABLE)

        for model_type, data in attention_data.items():
            model_name = f"ProtGram{model_type.capitalize()}"
            print(f"  Processing attention for {model_name}...")

            if data.get("hierarchical"):
                hier_path = self.config.RESULTS_EVALUATION_DIR / f"attention_hierarchical_{model_name}.json"
                DataUtils.save_json(data["hierarchical"], str(hier_path))
                reporter.generate_hierarchical_attention_plot(hier_path, model_name)

            if data.get("protein_pooling"):
                pool_path = self.config.RESULTS_EVALUATION_DIR / f"attention_pooling_{model_name}.json"
                DataUtils.save_json(data["protein_pooling"], str(pool_path))
                reporter.generate_pooling_attention_plot(pool_path, model_name)

    def _run_sanity_check_ppi(self, embedding_path: str):
        """Runs a quick, small-scale PPI evaluation as a sanity check."""
        DataUtils.print_header("Running Sanity Check PPI Evaluation")
        print(f"  Using embeddings from: {Path(embedding_path).name}")
        sanity_config = copy.deepcopy(self.config)

        # Override config for a quick run
        sanity_config.EVAL_EPOCHS = self.config.GCN_SANITY_CHECK_EPOCHS
        sanity_config.EVAL_N_FOLDS = 2  # A minimal number of folds for a quick check
        sanity_config.EVAL_GENERATE_SHAP_SUMMARY = False  # Disable for speed
        sanity_config.PLOT_TRAINING_HISTORY = False  # Disable for speed

        # Set the specific embedding file to evaluate
        model_name_for_eval = Path(embedding_path).stem.replace('_pca', '').replace(str(self.config.PCA_TARGET_DIMENSION), '')
        sanity_config.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": model_name_for_eval, "path": embedding_path}
        ]

        try:
            # Instantiate and run the pipeline with the modified config
            ppi_evaluator = PPIPipeline(sanity_config)
            ppi_evaluator.run(use_dummy_data=False)
        except Exception as e:
            print(f"  ❌ Sanity check PPI evaluation failed with an error: {e}")
            traceback.print_exc()

        print("--- Sanity Check PPI Evaluation Finished ---")