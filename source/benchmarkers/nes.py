# ==============================================================================
# MODULE: benchmarkers/nes.py
# PURPOSE: Handles benchmarking of traditional network embedding methods.
# VERSION: 1.0
# AUTHOR: Your Name (Assembled by Coding Partner)
# ==============================================================================

import os

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid, WebKB
# FIX: In recent PyG versions, Node2Vec is in the models submodule,
# but DeepWalk must be imported from its specific path.
from torch_geometric.nn.models import Node2Vec
from torch_geometric.nn.models.metapath2vec import MetaPath2Vec

from configuration.config import Config
from source.utils.data import DataUtils


class NetworkEmbeddingBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)
        self.models_to_run = config.BENCHMARK_NE_MODELS_TO_RUN
        os.makedirs(self.output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("### Network Embedding Benchmarker Initialized ###")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dataset root: {self.dataset_root}")
        print("=" * 80)

    def _get_dataset(self, name: str):
        path = self.dataset_root
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                return Planetoid(root=path, name=name)
            elif name in ['Cornell', 'Texas', 'Wisconsin']:
                return WebKB(root=path, name=name)
            elif name == 'KarateClub':
                from torch_geometric.datasets import KarateClub
                dataset = KarateClub()
                dataset.name = 'KarateClub'  # Manually add the name attribute
                return dataset
            else:
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _run_model_on_dataset(self, model_name, dataset):
        data = dataset[0]
        data = data.to(self.device)
        print(f"--- Benchmarking Model: {model_name} on Dataset: {data} ---")
        try:
            if model_name == 'Node2Vec':
                model = Node2Vec(
                    edge_index=data.edge_index,
                    embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
                    walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
                    context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
                    walks_per_node=10,
                    num_negative_samples=1,
                    p=1,
                    q=1,
                    sparse=True,
                ).to(self.device)
            elif model_name == 'MetaPath2Vec':
                model = MetaPath2Vec(edge_index_dict=data.edge_index,
                    embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
                    walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
                    context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
                    walks_per_node=10,
                    sparse=True
                ).to(self.device)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

            for _ in range(1, 3):  # Train for 2 epochs for a quick benchmark
                model.train()
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                z = model()

            # Simple node classification evaluation
            train_mask = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
            test_mask = data.test_mask[:, 0] if data.test_mask.dim() > 1 else data.test_mask

            clf = LogisticRegression(
                solver='lbfgs', multi_class='auto', random_state=self.config.RANDOM_STATE
            ).fit(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())

            test_acc = accuracy_score(data.y[test_mask].cpu().numpy(), clf.predict(z[test_mask].cpu().numpy()))

            print(f"  ✅ Test Accuracy for {model_name} on {dataset.name}: {test_acc:.4f}")
            return {
                "dataset": dataset.name,
                "model": model_name,
                "test_accuracy": test_acc,
                "error": None
            }
        except Exception as e:
            print(f"  ❌ FAILED to benchmark on {dataset.name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "dataset": dataset.name,
                "model": model_name,
                "test_accuracy": None,
                "error": str(e)
            }

    def run(self):
        DataUtils.print_header("PIPELINE: Network Embedding BENCHMARKER")
        all_results = []
        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset = self._get_dataset(dataset_name)
            if not dataset:
                print(f"Skipping NE benchmark for {dataset_name} as it could not be loaded.")
                all_results.append({
                    "dataset": dataset_name, "model": "N/A", "test_accuracy": None,
                    "error": f"Dataset {dataset_name} could not be loaded."
                })
                continue

            print(f"\nLoaded dataset: {dataset.name}. Nodes: {dataset[0].num_nodes}, Edges: {dataset[0].num_edges}")

            for model_name in self.models_to_run:
                result = self._run_model_on_dataset(model_name, dataset)
                all_results.append(result)

        summary_df = pd.DataFrame(all_results)
        summary_path = self.output_dir / "ne_benchmark_summary.csv"
        DataUtils.save_dataframe_to_csv(summary_df, str(summary_path))
        print(f"\n==================================================")
        print(f"Network Embedding Benchmark Summary saved to: {summary_path}")
        print(summary_df.to_string())
        print(f"==============================================")
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")
