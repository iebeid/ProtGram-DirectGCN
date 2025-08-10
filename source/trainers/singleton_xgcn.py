# ==============================================================================
# MODULE: trainers/singleton_xgcn.py
# PURPOSE: A lightweight trainer for rapid evaluation of various GNNs on the n=1 graph.
# VERSION: 3.0 (Integrated dynamic homophily-based architecture selection)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Dict, List, Any, Optional

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import homophily
from torch_geometric.data import Data
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataset
from source.utils.data import DataUtils
from source.models.gnn.spatial.gat import GAT
from source.models.gnn.spatial.gin import GIN
from source.models.gnn.spatial.graphsage import GraphSAGE
from source.models.gnn.spectral.chebnet import ChebNet
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spectral.gcn import GCN
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.tongidigcn import TongDiGCN


class SingletonXGCNTrainer:
    """
    A specialized trainer for a fast, standalone evaluation of various GNN models
    on the n=1 n-gram graph. This is intended for rapid prototyping.
    """

    def __init__(self, config: Config, graph_obj: DirectedNgramGraph):
        self.config = config
        self.graph = graph_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataset(config)
        # --- NEW: Set seeds for reproducibility ---
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def run(self) -> pd.DataFrame:
        """
        Executes the entire training and evaluation workflow for the n=1 graph.

        Returns:
            A pandas DataFrame of performance metrics.
        """
        if not self.graph or self.graph.number_of_nodes == 0:
            print("  Singleton Trainer: Graph is empty or invalid. Cannot proceed.")
            return pd.DataFrame()

        # 1. Generate self-supervised labels (community detection for n=1)
        labels, num_classes = self.label_generator.generate_task_labels(self.graph, 'community')
        if num_classes <= 1:
            print("  Singleton Trainer: Only one community found. Cannot perform meaningful classification.")
            return pd.DataFrame()

        # --- NEW: Dynamic Architecture Selection for DirectGCN ---
        # Calculate homophily to decide if specialized paths should be used.
        homophily_ratio = homophily(self.graph.A_undirected_norm_sparse.indices(), labels, method='edge')
        is_heterophilic = homophily_ratio < 0.6  # Standard threshold
        print(f"  Singleton Graph (n=1) Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")

        # 2. Create initial random features
        initial_features = torch.randn((self.graph.number_of_nodes, self.config.GCN_1GRAM_INIT_DIM))

        # 3. Create train/test splits for nodes
        node_indices = np.arange(self.graph.number_of_nodes)
        try:
            train_idx, test_idx = train_test_split(
                node_indices,
                test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE,
                stratify=labels.numpy()
            )
        except ValueError:
            # Fallback for very small classes that can't be stratified
            train_idx, test_idx = train_test_split(
                node_indices,
                test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE
            )

        train_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(train_idx), 1)
        test_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(test_idx), 1)

        all_results = []
        for model_name in self.config.SINGLETON_EVAL_MODELS_TO_RUN:
            print(f"\n--- Evaluating Singleton Model: {model_name} ---")

            # Determine if this model run should use the specialized paths
            use_homo_hetero_for_this_model = is_heterophilic if model_name.lower() == 'directgcn' else False
            if use_homo_hetero_for_this_model:
                print("  -> Enabling specialized homophily/heterophily paths for DirectGCN.")
                self.graph.split_edges_by_homophily(labels)

            model = self._get_model(model_name, initial_features.shape[1], num_classes, use_homo_hetero_for_this_model)
            if model is None:
                continue

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.SINGLETON_EVAL_LR)
            data_for_model = self._prepare_data_for_model(model_name, initial_features, labels, train_mask, test_mask, use_homo_hetero_for_this_model).to(self.device)

            # Training Loop
            for epoch in tqdm(range(self.config.SINGLETON_EVAL_EPOCHS), desc=f"  Training {model_name}", leave=False):
                model.train()
                optimizer.zero_grad()
                logits, _ = model(data_for_model)
                if data_for_model.train_mask.sum() > 0:
                    loss = F.cross_entropy(logits[data_for_model.train_mask], data_for_model.y[data_for_model.train_mask].long())
                    loss.backward()
                    # --- FIX: Add Gradient Clipping to stabilize training on small/volatile graphs ---
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                logits, _ = model(data_for_model)
                preds = logits.argmax(dim=-1)
                y_true = data_for_model.y[data_for_model.test_mask].cpu().numpy()
                y_pred = preds[data_for_model.test_mask].cpu().numpy()

                metrics = {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "F1-Score (Macro)": f1_score(y_true, y_pred, average='macro', zero_division=0),
                    "Precision (Macro)": precision_score(y_true, y_pred, average='macro', zero_division=0),
                    "Recall (Macro)": recall_score(y_true, y_pred, average='macro', zero_division=0)
                }
            all_results.append(metrics)
        return pd.DataFrame(all_results)

    def _get_model(self, name: str, in_channels: int, num_classes: int, use_homo_hetero_paths: bool) -> torch.nn.Module:
        """Model factory for instantiating GNNs."""
        # --- FIX: Use dedicated SINGLETON parameters from config for consistency and independent control ---
        model_params = {
            'in_channels': in_channels,
            'hidden_channels': self.config.SINGLETON_GNN_HIDDEN_CHANNELS,
            'out_channels': num_classes,
            'num_layers': self.config.SINGLETON_GNN_NUM_LAYERS,
            'dropout_rate': self.config.SINGLETON_GNN_DROPOUT_RATE
        }
        if name == "GCN":
            return GCN(**model_params)
        if name == "GAT":
            gat_params = model_params.copy()
            gat_params.update({
                'heads': self.config.SINGLETON_GAT_HEADS,
                'dropout_rate': self.config.SINGLETON_GAT_DROPOUT_RATE
            })
            return GAT(**gat_params)
        if name == "GraphSAGE":
            return GraphSAGE(**model_params)
        if name == "GIN":
            return GIN(**model_params)
        if name == "ChebNet":
            return ChebNet(**model_params, K=self.config.SINGLETON_CHEBNET_K)
        if name == "RGCN":
            return RGCN(**model_params, num_relations=self.config.SINGLETON_RGCN_NUM_RELATIONS)
        if name == "TongDiGCN": return TongDiGCN(**model_params)
        if name == "DirectGCN":
            layer_dims = [in_channels] + self.config.SINGLETON_DIRECTGCN_HIDDEN_LAYER_DIMS
            return DirectGCN(
                layer_dims=layer_dims, num_graph_nodes=self.graph.number_of_nodes,
                task_num_output_classes=num_classes,
                n_gram_len=1, # This is the singleton trainer, so n is always 1
                use_homo_hetero_paths=use_homo_hetero_paths,
                one_gram_dim=self.config.GCN_1GRAM_INIT_DIM, max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE, gating_mode=self.config.GCN_GATING_COEFF_MODE
            )
        raise ValueError(f"Unknown model name '{name}' for singleton evaluation.")

    def _prepare_data_for_model(self, model_name: str, features: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor, test_mask: torch.Tensor, use_homo_hetero_paths: bool) -> Data:
        """Prepares a PyG Data object tailored to the specific model's needs."""
        data_dict = {'x': features, 'y': labels, 'train_mask': train_mask, 'test_mask': test_mask}
        model_name_lower = model_name.lower()

        if model_name_lower == 'directgcn':
            # Always include the base structural and directional paths
            data_dict.update({
                'edge_index_in': self.graph.A_in_w.indices(), 'edge_weight_in': self.graph.A_in_w.values(),
                'edge_index_out': self.graph.A_out_w.indices(), 'edge_weight_out': self.graph.A_out_w.values(),
                'edge_index_undirected_norm': self.graph.A_undirected_norm_sparse.indices(),
                'edge_weight_undirected_norm': self.graph.A_undirected_norm_sparse.values()
            })

            # Conditionally add the new top-level homophily/heterophily paths
            if use_homo_hetero_paths and self.graph.A_homo_w is not None and self.graph.A_hetero_w is not None:
                print("  Preparing data with parallel homophily/heterophily paths for singleton evaluation.")
                data_dict.update({
                    'edge_index_homo': self.graph.A_homo_w.indices(), 'edge_weight_homo': self.graph.A_homo_w.values(),
                    'edge_index_hetero': self.graph.A_hetero_w.indices(), 'edge_weight_hetero': self.graph.A_hetero_w.values()
                })
        elif model_name_lower == 'rgcn':
            edge_index_out = self.graph.A_out_w.indices()
            edge_index_in = self.graph.A_in_w.indices()
            data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([torch.zeros(edge_index_out.size(1)), torch.ones(edge_index_in.size(1))]).long()
        elif model_name_lower == 'tongdigcn':
            data_dict['edge_index'] = self.graph.A_out_w.indices() # Forward pass uses outgoing edges
            data_dict['edge_index_backward'] = self.graph.A_in_w.indices() # Backward pass uses incoming edges
        else:  # GCN, GAT, etc.
            # --- FIX: Align with the main GNN benchmarker for consistency. ---
            # Use the symmetrically normalized undirected graph for standard GNNs.
            data_dict['edge_index'] = self.graph.A_undirected_norm_sparse.indices()
            data_dict['edge_attr'] = self.graph.A_undirected_norm_sparse.values()

        return Data.from_dict(data_dict)
