# ==============================================================================
# MODULE: benchmarkers/nes.py
# PURPOSE: Handles benchmarking of Network Embedding models like Node2Vec.
# VERSION: 4.0 (Refactored to use BaseBenchmarker)
# AUTHOR: Islam Ebeid
# ==============================================================================

import traceback
from typing import Dict, Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

from configuration.config import Config
from source.benchmarkers.base import BaseBenchmarker
from source.utils.data.data_utils import DataUtils
from source.models.fnn.mlp import SimpleMLP
from source.utils.models.early_stopper import EarlyStopper

class NetworkEmbeddingBenchmarker(BaseBenchmarker):
    def __init__(self, config: Config):
        super().__init__(config, "Network Embedding Benchmarker")

    def _train_and_evaluate_mlp(self, embeddings: torch.Tensor, data: Data) -> Dict[str, float]:
        """Trains and evaluates a simple MLP on the generated embeddings."""
        num_classes = int(data.y.max().item()) + 1
        mlp = SimpleMLP(
            in_channels=embeddings.shape[1],
            hidden_channels=128,
            out_channels=num_classes,
            dropout=0.5
        ).to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

        # FIX: Use the helper to handle masks that might have multiple splits (e.g., WebKB)
        train_mask = self._get_1d_mask(data.train_mask)
        val_mask = self._get_1d_mask(data.val_mask)
        test_mask = self._get_1d_mask(data.test_mask)

        best_val_acc = -1
        # --- DEFINITIVE FIX for UnboundLocalError ---
        # Initialize metrics to a default value before the loop.
        test_acc_at_best_val = -1.0
        f1_at_best_val = -1.0
        precision_at_best_val = -1.0
        recall_at_best_val = -1.0

        # --- NEW: Add early stopping to prevent overfitting the simple classifier ---
        # --- ANTICIPATORY DEBUGGING: Use early stopping for more robust evaluation ---
        # Training for a fixed number of epochs can lead to overfitting the classifier.
        # Early stopping based on validation accuracy provides more reliable and comparable results.
        early_stopper = EarlyStopper(patience=self.config.PROTGRAM_EARLY_STOPPING_PATIENCE, min_delta=self.config.PROTGRAM_EARLY_STOPPING_MIN_DELTA)

        for epoch in range(1, 201):  # A fixed number of epochs for the MLP classifier
            mlp.train()
            optimizer.zero_grad()
            out = mlp(embeddings[train_mask].detach()) # Detach to be safe
            loss = F.cross_entropy(out, data.y[train_mask])
            loss.backward()
            optimizer.step()

            mlp.eval()
            with torch.no_grad():
                pred = mlp(embeddings).argmax(dim=1)
                val_correct = (pred[val_mask] == data.y[val_mask]).sum()
                val_acc = int(val_correct) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0
                test_correct = (pred[test_mask] == data.y[test_mask]).sum()
                test_acc = int(test_correct) / int(test_mask.sum()) if test_mask.sum() > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                # --- NEW: Capture all test metrics at the best validation epoch ---
                y_true_test = data.y[test_mask].cpu().numpy()
                y_pred_test = pred[test_mask].cpu().numpy()
                f1_at_best_val = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                precision_at_best_val = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                recall_at_best_val = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)

            # Early stopping is based on validation loss (or 1 - accuracy)
            if early_stopper.early_stop(1.0 - val_acc):
                print(f"    MLP training: Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
                break

        metrics = {
            'Accuracy': test_acc_at_best_val,
            'F1-Score (Macro)': f1_at_best_val,
            'Precision (Macro)': precision_at_best_val,
            'Recall (Macro)': recall_at_best_val,
            'best_val_accuracy': best_val_acc
        }
        return metrics

    def _run_on_dataset(self, dataset: Any, dataset_name: str, model_name: str) -> Dict:
        """Runs a single NE model on a single dataset."""
        data = dataset[0].to(self.device)
        print(f"--- Benchmarking Model: {model_name} on Dataset: {dataset_name} ---")

        if not all(hasattr(data, mask) and getattr(data, mask) is not None and getattr(data, mask).any() for mask in ['train_mask', 'val_mask', 'test_mask']):
            print(f"  - No predefined splits found for {dataset_name}. Creating random splits.")
            num_nodes = data.num_nodes
            rng = np.random.default_rng(self.config.RANDOM_STATE) # Use a seeded generator
            indices = rng.permutation(num_nodes)
            train_size = int(num_nodes * 0.1)
            val_size = int(num_nodes * 0.1)
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
            print(f"  Generated custom seeded split for {dataset_name}. Train: {train_size}, Val: {val_size}, Test: {int(data.test_mask.sum())}")

        # 1. Train Node2Vec to get embeddings
        node2vec_model = Node2Vec(
            data.edge_index,
            embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
            walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
            context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
            walks_per_node=10,
            num_negative_samples=1,
            p=1, q=1,
            sparse=True,
        ).to(self.device)

        # --- ANTICIPATORY DEBUGGING: Set num_workers=0 to prevent hangs. ---
        # Using multiple worker processes (num_workers > 0) can cause two issues:
        # 1. The main script hangs before an input() prompt, waiting for background processes.
        # 2. It introduces non-determinism unless a specific `worker_init_fn` is used.
        # Setting num_workers=0 forces data loading to happen in the main thread, resolving both.
        loader = node2vec_model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(node2vec_model.parameters()), lr=0.01)

        for _ in range(self.config.BENCHMARK_NE_EPOCHS):
            node2vec_model.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec_model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()

        # 2. Get the final embeddings
        with torch.no_grad():
            node2vec_model.eval()
            embeddings = node2vec_model()

        # 3. Train and evaluate an MLP on the embeddings
        metrics = self._train_and_evaluate_mlp(embeddings, data) # This call is now correct
        print(f"  ✅ Best Val Acc: {metrics.get('best_val_accuracy', -1):.4f}, Test Accuracy for {model_name} on {dataset_name}: {metrics.get('Accuracy', -1):.4f}")

        # --- ANTICIPATORY DEBUGGING: Explicitly delete large torch objects and collect garbage. ---
        # This helps prevent the loader from hanging in the background before the main script
        # prompts for user input, which can happen if resources are not released.
        del node2vec_model, loader, optimizer, embeddings
        # --- NEW: Add garbage collection to be thorough ---
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_row = {"dataset": dataset_name, "model": model_name, "error": None}
        result_row.update(metrics)

        return result_row

    def run(self) -> pd.DataFrame:
        """Main execution function for the benchmarker."""
        DataUtils.print_header("PIPELINE: Network Embedding BENCHMARKER")
        all_results = []

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset = self._get_dataset(dataset_name)
            if dataset is None: continue
            print(f"\nLoaded dataset: {dataset_name}. Nodes: {dataset[0].num_nodes}, Edges: {dataset[0].num_edges}")

            for model_name in self.config.BENCHMARK_NE_MODELS_TO_RUN:
                # --- ERROR HANDLING: Wrap each model's evaluation to prevent one failure from stopping the suite. ---
                try:
                    with mlflow.start_run(run_name=f"{model_name}_on_{dataset_name}", nested=True):
                        mlflow.set_tag("model_name", model_name)
                        mlflow.set_tag("dataset_name", dataset_name)
                        result = self._run_on_dataset(dataset, dataset_name, model_name)
                        all_results.append(result)
                        mlflow.log_metrics({
                            "best_val_accuracy": result.get('best_val_accuracy', 0.0),
                            "test_accuracy": result.get('Accuracy', 0.0)
                        })
                except Exception as e:
                    print(f"ERROR during benchmarking of {model_name} on {dataset_name}: {e}")
                    traceback.print_exc()
                    all_results.append({"dataset": dataset_name, "model": model_name, "error": str(e)})

        # --- FIX: Force garbage collection after the benchmark loop ---
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary_df = pd.DataFrame(all_results)
        # The full summary is now handled by main.py
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")
        return summary_df if all_results else pd.DataFrame()