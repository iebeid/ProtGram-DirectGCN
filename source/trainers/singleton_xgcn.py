# ==============================================================================
# MODULE: trainers/singleton_xgcn.py
# PURPOSE: A lightweight trainer for rapid evaluation of various GNNs on the n=1 graph.
# VERSION: 3.1 (Fixed device placement issue for training masks)
# AUTHOR: Islam Ebeid
# ==============================================================================

import traceback
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import homophily
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataBuilder
from torch_geometric.data import Data
from source.utils.data.data_utils import DataUtils
from source.utils.data.protgram_helper import ProtgramDaskHelpers
from source.models.factory import ModelFactory


class SingletonXGCNTrainer:
    """
    A specialized trainer for a fast, standalone evaluation of various GNN models
    on the n-gram graphs. This is intended for rapid prototyping.
    """

    def __init__(self, config: Config, graph_obj: DirectedNgramGraph):
        self.config = config
        self.graph = graph_obj
        self.n_val = graph_obj.n_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataBuilder(config)
        self.model_factory = ModelFactory(config, context='singleton')
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def run(self) -> pd.DataFrame:
        """
        Executes the entire training and evaluation workflow for the n-gram graph.

        Returns:
            A pandas DataFrame of performance metrics.
        """
        if not self.graph or self.graph.number_of_nodes == 0:
            print("  Singleton Trainer: Graph is empty or invalid. Cannot proceed.")
            return pd.DataFrame()

        task_type = 'next_node'

        # --- DEFINITIVE FIX for Singleton Evaluation: Use component-aware community detection ---
        # The original label generator might only find communities in the largest connected component.
        # This new logic ensures that for the 'community' task, we generate labels for all nodes.
        if task_type == 'community': # --- DEFINITIVE FIX: Use the centralized, robust community detection utility ---
            community_graph_data = Data(
                edge_index=self.graph.A_undirected_w.indices(),
                edge_attr=self.graph.A_undirected_w.values(), num_nodes=self.graph.number_of_nodes
            )
            labels, num_classes = DataUtils.generate_community_labels(community_graph_data)
        else:
            labels, num_classes = self.label_generator.generate_task_labels(self.graph, task_type)

        # --- DEFINITIVE FIX for "Zero Results" on Small/Simple Graphs ---
        # This guard is the reason the subsequent code may appear "unreachable".
        # It correctly prevents a pointless evaluation when no meaningful classification
        # task can be defined (e.g., only one community was found).
        if task_type != 'masked_node' and num_classes <= 1:
            print(f"\n  SKIPPING SINGLETON EVALUATION: The task '{task_type}' resulted in only {num_classes} class(es).")
            print("  This is expected for very small or simple graphs. Returning empty results.")
            return pd.DataFrame()

        is_heterophilic = False
        y_for_stratify = torch.zeros(self.graph.number_of_nodes, dtype=torch.long)
        if labels is not None:
            y_for_stratify = labels
            homophily_ratio = homophily(self.graph.A_undirected_norm_sparse.indices(), y_for_stratify, method='edge')
            is_heterophilic = homophily_ratio < self.config.GCN_HETEROPHILY_THRESHOLD
            print(f"  Singleton Graph (n={self.n_val}) Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
        else:
            print("  Homophily calculation skipped for non-classification task (e.g., masked_node).")

        print(f"  Initializing features with identity matrix.")
        initial_features = torch.eye(self.graph.number_of_nodes)
        # --- NEW: Apply BatchNorm for consistency with the main pipeline's feature handling ---
        # This ensures that the model is always tested under similar input conditions,
        # even though the random features for n=1 are already somewhat normalized.
        bn = torch.nn.BatchNorm1d(initial_features.shape[1]).to(self.device)
        # Move features to device for normalization, then detach them from the computation graph.
        # This prevents the "trying to backward through the graph a second time" error.
        initial_features = bn(initial_features.to(self.device)).detach()

        import numpy as np
        node_indices = np.arange(self.graph.number_of_nodes)
        try:
            train_idx, test_idx = train_test_split(
                node_indices, test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE, stratify=y_for_stratify.cpu().numpy()
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                node_indices, test_size=self.config.SINGLETON_EVAL_TEST_SPLIT, random_state=self.config.RANDOM_STATE
            )

        # --- DEFINITIVE FIX for "Zero Results" on Small Graphs ---
        # If the graph is too small, train_test_split can produce an empty test set.
        # This guard prevents running the entire training loop only to find there's
        # nothing to evaluate, which previously resulted in an empty metrics table.
        if len(test_idx) == 0:
            print(f"  Singleton Trainer: The train/test split resulted in an empty test set (graph size: {self.graph.number_of_nodes}, test_split: {self.config.SINGLETON_EVAL_TEST_SPLIT}).")
            print("  This is expected for very small graphs. Cannot perform meaningful evaluation.")
            return pd.DataFrame()

        train_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(train_idx), 1)
        test_mask = torch.zeros(self.graph.number_of_nodes, dtype=torch.bool).scatter_(0, torch.from_numpy(test_idx), 1)

        all_results = []
        for model_name in tqdm(self.config.SINGLETON_EVAL_MODELS_TO_RUN, desc="Evaluating Singleton Models"):
            print(f"\n--- Evaluating Singleton Model: {model_name} ---")
            try:
                A_homo_norm, A_hetero_norm = None, None
                use_homo_hetero_for_this_model = is_heterophilic if model_name.lower() == 'directgcn' else False
                if use_homo_hetero_for_this_model and labels is not None:
                    print("  -> Enabling specialized homophily/heterophily paths for DirectGCN.")
                    split_result = DirectedNgramGraph.split_edges_by_homophily(
                        self.graph.A_out_w, self.graph.number_of_nodes, labels
                    )
                    if split_result: A_homo_norm, A_hetero_norm = split_result

                model = self.model_factory.create_model(
                    model_name=model_name, in_channels=initial_features.shape[1], num_classes=num_classes,
                    graph_obj=self.graph, use_homo_hetero_paths=use_homo_hetero_for_this_model, n_val=self.n_val
                )

                print(model)
                model.to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config.SINGLETON_EVAL_LR)

                # --- REFACTOR: Use the centralized data preparation utility ---
                # This replaces the large if/elif/else block and ensures consistency
                # with the main ProtGram trainer.
                data_for_model = ProtgramDaskHelpers.prepare_pyg_data_from_protgram_graph(
                    model_type=model_name, graph=self.graph, features=initial_features, labels=y_for_stratify,
                    A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm,
                    train_mask=train_mask, test_mask=test_mask
                ).to(self.device)

                optimizer.zero_grad()
                for epoch in tqdm(range(self.config.SINGLETON_EVAL_EPOCHS), desc=f"  Training {model_name}", leave=False):
                    model.train()

                    if task_type == 'masked_node':
                        masked_features, masked_indices, original_node_ids = self.label_generator.generate_masked_node_task(self.graph, initial_features, masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION, exclude_mask=test_mask)
                        epoch_data = data_for_model.clone()
                        epoch_data.x = masked_features
                        logits, _ = model(epoch_data)
                        loss = F.cross_entropy(logits[masked_indices], original_node_ids.to(self.device))
                    else:
                        logits, _ = model(data_for_model)
                        if data_for_model.train_mask.sum() > 0:
                            loss = F.cross_entropy(logits[data_for_model.train_mask], data_for_model.y[data_for_model.train_mask].long())
                        else:
                            loss = torch.tensor(0.0, device=self.device)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                model.eval()
                if task_type == 'masked_node':
                    with torch.no_grad():
                        masked_features, masked_indices, original_node_ids = self.label_generator.generate_masked_node_task(
                            self.graph, initial_features, masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION, exclude_mask=train_mask
                        )
                        eval_data = data_for_model.clone()
                        eval_data.x = masked_features
                        logits, _ = model(eval_data)
                        preds = logits[masked_indices].argmax(dim=-1)
                        y_true = original_node_ids.cpu().numpy()
                        y_pred = preds.cpu().numpy()
                else:
                    with torch.no_grad():
                        logits, _ = model(data_for_model)
                        preds = logits.argmax(dim=-1)
                        y_true = data_for_model.y[data_for_model.test_mask].cpu().numpy()
                        y_pred = preds[data_for_model.test_mask].cpu().numpy()

                if len(y_true) > 0:
                    from sklearn.metrics import confusion_matrix
                    import numpy as np
                    print(f"y_true shape: {y_true.shape}")
                    print(f"y_pred shape: {y_pred.shape}")
                    print(f"y_true unique values: {np.unique(y_true, return_counts=True)}")
                    print(f"y_pred unique values: {np.unique(y_pred, return_counts=True)}")
                    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
                    metrics = {
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_true, y_pred),
                        "F1-Score (Macro)": f1_score(y_true, y_pred, average='macro', zero_division=0)
                    }
                    all_results.append(metrics)
                else:
                    print(f"  Skipping metrics for {model_name} as there was no data in the test set to evaluate.")

            except Exception as e:
                print(f"  ❌ ERROR during evaluation of {model_name}: {e}")
                traceback.print_exc()
                all_results.append({
                    "Model": model_name, "Accuracy": 0.0, "F1-Score (Macro)": 0.0
                })

        return pd.DataFrame(all_results)
