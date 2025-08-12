# ==============================================================================
# MODULE: trainers/singleton_xgcn.py
# PURPOSE: A lightweight trainer for rapid evaluation of various GNNs on the n=1 graph.
# VERSION: 3.0 (Integrated dynamic homophily-based architecture selection)
# AUTHOR: Islam Ebeid
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
        # --- NEW: Centralized model creation ---
        self.model_factory = ModelFactory(config, context='singleton')
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

        # 1. Determine the task and generate labels if necessary
        # --- FIX: Use correct PROTGRAM_ prefixed config variables ---
        task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(1, self.config.PROTGRAM_DEFAULT_TASK_TYPE)
        labels, num_classes = self.label_generator.generate_task_labels(self.graph, task_type)

        if task_type != 'masked_node' and num_classes <= 1:
            print("  Singleton Trainer: Only one community found. Cannot perform meaningful classification.")
            return pd.DataFrame()

        # --- NEW: Dynamic Architecture Selection for DirectGCN ---
        # Calculate homophily to decide if specialized paths should be used.
        # --- DEFINITIVE FIX: Only calculate homophily if we have valid labels for a classification task ---
        is_heterophilic = False
        y_for_stratify = torch.zeros(self.graph.number_of_nodes, dtype=torch.long)
        if labels is not None:
            y_for_stratify = labels
            homophily_ratio = homophily(self.graph.A_undirected_norm_sparse.indices(), y_for_stratify, method='edge')
            is_heterophilic = homophily_ratio < 0.6  # Standard threshold
            print(f"  Singleton Graph (n=1) Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
        else:
            print("  Homophily calculation skipped for non-classification task (e.g., masked_node).")

        # 2. Create initial random features, consistent with the main ProtGram pipeline.
        print(f"  Using random features for initial features (dim={self.config.PROTGRAM_1GRAM_INIT_DIM}).")
        initial_features = torch.randn((self.graph.number_of_nodes, self.config.PROTGRAM_1GRAM_INIT_DIM))

        # 3. Create train/test splits for nodes
        node_indices = np.arange(self.graph.number_of_nodes)
        try:
            train_idx, test_idx = train_test_split(
                node_indices,
                test_size=self.config.SINGLETON_EVAL_TEST_SPLIT,
                random_state=self.config.RANDOM_STATE,
                stratify=y_for_stratify.numpy()
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
            A_homo_norm, A_hetero_norm = None, None
            use_homo_hetero_for_this_model = is_heterophilic if model_name.lower() == 'directgcn' else False
            if use_homo_hetero_for_this_model and labels is not None:
                print("  -> Enabling specialized homophily/heterophily paths for DirectGCN.")
                split_result = self.graph.split_edges_by_homophily(labels)
                if split_result:
                    A_homo_norm, A_hetero_norm = split_result

            model = self.model_factory.create_model(
                model_name=model_name, in_channels=initial_features.shape[1], num_classes=num_classes,
                graph_obj=self.graph, use_homo_hetero_paths=use_homo_hetero_for_this_model
            )
            if model is None:
                continue
            print(model)

            model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.SINGLETON_EVAL_LR)
            # Use the centralized, correct data preparation utility
            data_for_model = prepare_pyg_data_from_protgram_graph(
                model_type=model_name, graph=self.graph, features=initial_features, labels=y_for_stratify,
                use_homo_hetero_paths=use_homo_hetero_for_this_model,
                A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm
            )
            # --- DEFINITIVE FIX: Attach the masks to the data object after creation ---
            data_for_model.train_mask = train_mask
            data_for_model.test_mask = test_mask

            # Training Loop
            for epoch in tqdm(range(self.config.SINGLETON_EVAL_EPOCHS), desc=f"  Training {model_name}", leave=False):
                model.train()
                optimizer.zero_grad()

                # --- FIX: Handle dynamic label generation for masked_node task ---
                if task_type == 'masked_node':
                    masked_features, masked_indices, original_node_ids = self.label_generator.generate_masked_node_task(
                        self.graph, initial_features, masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION, exclude_mask=test_mask
                    )
                    if epoch == 0:  # Print only on the first epoch
                        print(f"  Created Masked Node Prediction task for training: Masked {len(masked_indices)} nodes.")
                    epoch_data = prepare_pyg_data_from_protgram_graph(
                        model_type=model_name, graph=self.graph, features=masked_features, labels=y_for_stratify,
                        use_homo_hetero_paths=use_homo_hetero_for_this_model,
                        A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm
                    ).to(self.device)
                    logits, _ = model(epoch_data)
                    loss = F.cross_entropy(logits[masked_indices], original_node_ids.to(self.device))
                else:
                    data_for_model.train_mask = train_mask
                    data_for_model = data_for_model.to(self.device)
                    logits, _ = model(data_for_model)
                    if data_for_model.train_mask.sum() > 0:
                        loss = F.cross_entropy(logits[data_for_model.train_mask], data_for_model.y[data_for_model.train_mask].long())
                    else:
                        loss = torch.tensor(0.0, device=self.device) # No training nodes

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # --- DEFINITIVE FIX: Use task-appropriate evaluation logic ---
            model.eval()
            if task_type == 'masked_node':
                # For masked_node, we evaluate on a fresh mask from the test set
                # This measures how well the model learned the general context.
                with torch.no_grad():
                    masked_features, masked_indices, original_node_ids = self.label_generator.generate_masked_node_task(
                        self.graph, initial_features, masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION, exclude_mask=train_mask # Mask only from test set
                    )
                    # --- NEW: Add log message for evaluation mask ---
                    print(f"  Created Masked Node Prediction task for evaluation: Masked {len(masked_indices)} test nodes.")
                    eval_data = prepare_pyg_data_from_protgram_graph(
                        model_type=model_name, graph=self.graph, features=masked_features, labels=y_for_stratify,
                        use_homo_hetero_paths=use_homo_hetero_for_this_model,
                        A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm
                    ).to(self.device)
                    logits, _ = model(eval_data)
                    preds = logits[masked_indices].argmax(dim=-1)
                    y_true = original_node_ids.cpu().numpy() # These are the true node indices
                    y_pred = preds.cpu().numpy()
            else: # Standard classification evaluation
                with torch.no_grad():
                    data_for_model.test_mask = test_mask
                    data_for_model = data_for_model.to(self.device)
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