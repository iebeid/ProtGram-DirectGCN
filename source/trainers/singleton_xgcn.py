# ==============================================================================
# MODULE: trainers/singleton_xgcn.py
# PURPOSE: A lightweight trainer for rapid evaluation of various GNNs on the n=1 graph.
# VERSION: 2.1 (Corrected return type to DataFrame)
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import Dict, List, Any

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataset
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

        # 4. Prepare a base data object
        base_data = Data(
            x=initial_features, y=labels,
            edge_index_in=self.graph.mathcal_A_in.indices(), edge_weight_in=self.graph.mathcal_A_in.values(),
            edge_index_out=self.graph.mathcal_A_out.indices(), edge_weight_out=self.graph.mathcal_A_out.values(),
            edge_index_undirected_norm=self.graph.A_undirected_norm_sparse.indices(),
            edge_weight_undirected_norm=self.graph.A_undirected_norm_sparse.values(),
            train_mask=train_mask, test_mask=test_mask
        )

        all_results = []
        for model_name in self.config.SINGLETON_EVAL_MODELS_TO_RUN:
            print(f"\n--- Evaluating Singleton Model: {model_name} ---")
            model = self._get_model(model_name, base_data, num_classes)
            if model is None:
                continue

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.SINGLETON_EVAL_LR)
            data_for_model = self._prepare_data_for_model(model_name, base_data).to(self.device)

            # Training Loop
            for epoch in tqdm(range(self.config.SINGLETON_EVAL_EPOCHS), desc=f"  Training {model_name}", leave=False):
                model.train()
                optimizer.zero_grad()
                logits, _ = model(data_for_model)
                loss = F.cross_entropy(logits[data_for_model.train_mask], data_for_model.y[data_for_model.train_mask])
                loss.backward()
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

    def _get_model(self, name: str, data: Data, num_classes: int) -> torch.nn.Module:
        """Model factory for instantiating GNNs."""
        # --- FIX: Use benchmark parameters from config for consistency ---
        model_params = {
            'in_channels': data.num_features,
            'hidden_channels': self.config.BENCHMARK_GNN_HIDDEN_CHANNELS,
            'out_channels': num_classes,
            'num_layers': self.config.BENCHMARK_GNN_NUM_LAYERS,
            'dropout_rate': self.config.BENCHMARK_GNN_DROPOUT_RATE
        }
        if name == "GCN":
            return GCN(**model_params)
        if name == "GAT":
            gat_params = model_params.copy()
            gat_params.update({
                'heads': self.config.BENCHMARK_GAT_HEADS,
                'dropout_rate': self.config.BENCHMARK_GAT_DROPOUT_RATE
            })
            return GAT(**gat_params)
        if name == "GraphSAGE":
            return GraphSAGE(**model_params)
        if name == "GIN":
            return GIN(**model_params)
        if name == "ChebNet":
            return ChebNet(**model_params, K=self.config.BENCHMARK_CHEBNET_K)
        if name == "RGCN":
            return RGCN(**model_params, num_relations=self.config.BENCHMARK_RGCN_NUM_RELATIONS)
        if name == "TongDiGCN": return TongDiGCN(**model_params)
        if name == "DirectGCN":
            # Add a fallback in case the hidden layer list is empty in the config
            hidden_dim = self.config.GCN_HIDDEN_LAYER_DIMS[0] if self.config.GCN_HIDDEN_LAYER_DIMS else 256
            layer_dims = [data.num_features, hidden_dim, num_classes]
            return DirectGCN(
                layer_dims=layer_dims, num_graph_nodes=data.num_nodes,
                task_num_output_classes=num_classes,
                n_gram_len=1,
                use_homo_hetero_paths=self.config.GCN_USE_HOMOPHILY_HETEROPHILY_PATHS,
                one_gram_dim=self.config.GCN_1GRAM_INIT_DIM, max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE, gating_mode=self.config.GCN_GATING_COEFF_MODE
            )
        raise ValueError(f"Unknown model name '{name}' for singleton evaluation.")

    def _prepare_data_for_model(self, model_name: str, data: Data) -> Data:
        """Prepares the Data object with the correct edge indices for the specified model."""
        # The base data object already contains everything DirectGCN needs.
        if model_name == 'DirectGCN':
            return data

        # For other models, we need to select the appropriate edge index.
        # Most standard GNNs work best with the undirected, normalized adjacency matrix.
        data_clone = data.clone()
        data_clone.edge_index = data.edge_index_undirected_norm
        data_clone.edge_attr = data.edge_weight_undirected_norm

        if model_name == 'RGCN':
            # RGCN needs a combined edge_index and an edge_type tensor
            edge_index_out = data.edge_index_out
            edge_index_in = data.edge_index_in
            data_clone.edge_index = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_clone.edge_type = torch.cat([
                torch.zeros(edge_index_out.size(1), dtype=torch.long),
                torch.ones(edge_index_in.size(1), dtype=torch.long)
            ])
        elif model_name == 'TongDiGCN':
            # TongDiGCN needs separate forward and backward edge indices
            data_clone.edge_index = data.edge_index_out
            data_clone.edge_index_backward = data.edge_index_in

        return data_clone
