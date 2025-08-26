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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch_geometric.data import Data
from tqdm.auto import tqdm
from torch_geometric.nn import Node2Vec

from configuration.config import Config
from source.benchmarkers.base import BaseBenchmarker
from source.utils.data.data_utils import DataUtils

class NetworkEmbeddingBenchmarker(BaseBenchmarker):
    def __init__(self, config: Config):
        super().__init__(config, "Network Embedding Benchmarker")

    def _train_and_evaluate_classifier(self, embeddings: torch.Tensor, data: Data) -> Dict[str, float]:
        """
        Trains and evaluates a Logistic Regression classifier on the generated embeddings.
        This helper method encapsulates the downstream task logic.
        """
        # FIX: Use the helper to handle masks that might have multiple splits (e.g., WebKB)
        train_mask = self._get_1d_mask(data.train_mask)
        test_mask = self._get_1d_mask(data.test_mask)

        X_train = embeddings[train_mask].detach().cpu().numpy()
        y_train = data.y[train_mask].cpu().numpy()
        X_test = embeddings[test_mask].detach().cpu().numpy()
        y_test = data.y[test_mask].cpu().numpy()

        classifier = LogisticRegression(
            C=self.config.BENCHMARK_NE_CLASSIFIER_C,
            max_iter=self.config.BENCHMARK_NE_CLASSIFIER_MAX_ITER,
            solver=self.config.BENCHMARK_NE_CLASSIFIER_SOLVER,
            random_state=self.config.RANDOM_STATE
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0)
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
            walks_per_node=self.config.BENCHMARK_NE_WALKS_PER_NODE,
            num_negative_samples=self.config.BENCHMARK_NE_NUM_NEGATIVE_SAMPLES,
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

        # --- REFACTOR: Add tqdm progress bar and loss logging for better visibility ---
        for epoch in range(self.config.BENCHMARK_NE_EPOCHS):
            node2vec_model.train()
            total_loss = 0
            for pos_rw, neg_rw in tqdm(loader, desc=f"  Epoch {epoch+1}/{self.config.BENCHMARK_NE_EPOCHS}", leave=False):
                optimizer.zero_grad()
                loss = node2vec_model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if self.config.DEBUG_VERBOSE:
                print(f"    Epoch {epoch+1}: Avg. Loss = {total_loss / len(loader):.4f}")

        # 2. Get the final embeddings
        with torch.no_grad():
            node2vec_model.eval()
            embeddings = node2vec_model()

        # 3. Train and evaluate an MLP on the embeddings
        metrics = self._train_and_evaluate_classifier(embeddings, data)
        print(f"  ✅ Test Accuracy for {model_name} on {dataset_name}: {metrics.get('Accuracy', -1):.4f}")

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
                        # --- FIX: Sanitize metric names to be MLflow compatible ---
                        # MLflow does not allow parentheses or other special characters in metric names.
                        sanitized_metrics = {
                            k.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'): v
                            for k, v in result.items() if isinstance(v, (int, float))
                        }
                        mlflow.log_metrics(sanitized_metrics)
                except Exception as e:
                    print(f"ERROR during benchmarking of {model_name} on {dataset_name}: {e}")
                    traceback.print_exc()
                    all_results.append({"dataset": dataset_name, "model": model_name, "error": str(e)})

        summary_df = pd.DataFrame(all_results)
        # The full summary is now handled by main.py
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")
        return summary_df if all_results else pd.DataFrame()