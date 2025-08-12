# ==============================================================================
# MODULE: trainers/singleton_xgcn.py
# PURPOSE: A lightweight trainer for rapid evaluation of various GNNs on the n=1 graph.
# VERSION: 4.0 (Corrected data preparation logic for baseline GNNs)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import homophily
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_structures.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataBuilder
from source.utils.data import DataUtils, prepare_pyg_data_from_protgram_graph
from source.models.factory import ModelFactory


class SingletonXGCNTrainer:
    """
    A specialized trainer for a fast, standalone evaluation of various GNN models
    on the n=1 n-gram graph. This is intended for rapid prototyping.
    """

    def __init__(self, config: Config, graph_obj: DirectedNgramGraph):
        self.config = config
        self.graph = graph_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataBuilder(config)
        self.model_factory = ModelFactory(config, context='singleton')
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def run(self) -> pd.DataFrame:
        """
        Executes the entire training and evaluation workflow for the n=1 graph.
        """
        if not self.graph or self.graph.number_of_nodes == 0:
            print("  Singleton Trainer: Graph is empty or invalid. Cannot proceed.")
            return pd.DataFrame()

        task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(1, self.config.PROTGRAM_DEFAULT_TASK_TYPE)
        labels, num_classes = self.label_generator.generate_task_labels(self.graph, task_type)

        if task_type != 'masked_node' and num_classes <= 1:
            print("  Singleton Trainer: Only one community found. Cannot perform meaningful classification.")
            return pd.DataFrame()

        is_heterophilic = False
        y_for_stratify = torch.zeros(self.graph.number_of_nodes, dtype=torch.long)
        if labels is not None:
            y_for_stratify = labels
            # Use the raw, unnormalized graph for a more accurate homophily calculation
            homophily_ratio = homophily(self.graph.A_out_w.indices(), y_for_stratify, method='edge')
            is_heterophilic = homophily_ratio < self.config.GCN_HETEROPHILY_THRESHOLD
            print(f"  Singleton Graph (n=1) Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
        else:
            print("  Homophily calculation skipped for non-classification task (e.g., masked_node).")

        initial_features = torch.randn((self.graph.number_of_nodes, self.config.PROTGRAM_1GRAM_INIT_DIM))
        node_indices = np.arange(self.graph.number_of_nodes)
        try:
            train_idx, test_idx = train_test_split(
                node_indices, test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE, stratify=y_for_stratify.numpy()
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                node_indices, test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE
            )

        train_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(train_idx), 1)
        test_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(test_idx), 1)

        all_results = []
        for model_name in self.config.SINGLETON_EVAL_MODELS_TO_RUN:
            print(f"\n--- Evaluating Singleton Model: {model_name} ---")

            A_homo_norm, A_hetero_norm = None, None
            use_homo_hetero_for_this_model = is_heterophilic if model_name.lower() == 'directgcn' else False
            if use_homo_hetero_for_this_model and labels is not None:
                print("  -> Enabling specialized homophily/heterophily paths for DirectGCN.")
                split_result = self.graph.split_edges_by_homophily(labels)
                if split_result: A_homo_norm, A_hetero_norm = split_result

            model = self.model_factory.create_model(
                model_name=model_name, in_channels=initial_features.shape[1], num_classes=num_classes,
                graph_obj=self.graph, use_homo_hetero_paths=use_homo_hetero_for_this_model
            )
            if model is None: continue
            print(model)

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.SINGLETON_EVAL_LR)

            # --- DEFINITIVE FIX: Prepare the correct data object for each model type ---
            # This ensures that standard GNNs get a representation they can work with,
            # even on this difficult graph, preventing the zero-metric issue.
            data_for_model = prepare_pyg_data_from_protgram_graph(
                model_type=model_name, graph=self.graph, features=initial_features, labels=y_for_stratify,
                use_homo_hetero_paths=use_homo_hetero_for_this_model,
                A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm
            )
            data_for_model.train_mask = train_mask
            data_for_model.test_mask = test_mask

            # Training Loop
            for epoch in tqdm(range(self.config.SINGLETON_EVAL_EPOCHS), desc=f"  Training {model_name}", leave=False):
                model.train()
                optimizer.zero_grad()
                data_for_model = data_for_model.to(self.device)
                logits, _ = model(data_for_model)
                if data_for_model.train_mask.sum() > 0:
                    loss = F.cross_entropy(logits[data_for_model.train_mask], data_for_model.y[data_for_model.train_mask].long())
                else:
                    loss = torch.tensor(0.0, device=self.device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                logits, _ = model(data_for_model)
                preds = logits.argmax(dim=-1)
                y_true = data_for_model.y[data_for_model.test_mask].cpu().numpy()
                y_pred = preds[data_for_model.test_mask].cpu().numpy()

            if len(y_true) > 0:
                metrics = {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "F1-Score (Macro)": f1_score(y_true, y_pred, average='macro', zero_division=0),
                    "Precision (Macro)": precision_score(y_true, y_pred, average='macro', zero_division=0),
                    "Recall (Macro)": recall_score(y_true, y_pred, average='macro', zero_division=0)
                }
                all_results.append(metrics)
            else:
                print(f"  Skipping metrics for {model_name} as there was no data in the test set to evaluate.")

        return pd.DataFrame(all_results)