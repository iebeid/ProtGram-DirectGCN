# ==============================================================================
# MODULE: benchmarkers/gnns.py
# PURPOSE: Handles benchmarking of various GNN models on standard datasets.
# VERSION: 4.2 (Fixed CUDA assert by deriving num_classes from data)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import traceback
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, Actor, KarateClub
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected

from configuration.config import Config
from source.models.gnn.spectral.chebnet import ChebNet
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spatial.gat import GAT
from source.models.gnn.spectral.gcn import GCN
from source.models.gnn.spatial.gin import GIN
from source.models.gnn.spatial.graphsage import GraphSAGE
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.tongidigcn import TongDiGCN
from source.utils.data import DataUtils
from source.utils.models import EmbeddingProcessor


class GNNBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.embedding_dir = config.RESULTS_BENCHMARK_EMBEDDINGS_DIR
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

        print("GNNBenchmarker initialized. Using device: {}".format(self.device))
        print(f"Benchmark embeddings will be saved to: {self.embedding_dir}")
        torch.manual_seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_STATE)
        print(f"  Seeds set to {config.RANDOM_STATE} for reproducibility.")

    def _get_dataset(self, name: str, undirected: bool):
        """Loads a standard PyG dataset."""
        transform = ToUndirected() if undirected else None
        path = self.dataset_root
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                return Planetoid(root=path, name=name, transform=transform)
            elif name in ['Cornell', 'Texas', 'Wisconsin']:
                return WebKB(root=path, name=name, transform=transform)
            elif name == 'Actor':
                return Actor(root=path, transform=transform)
            elif name == 'KarateClub':
                return KarateClub(transform=transform)
            else:
                print(f"  Dataset '{name}' not recognized by this loader.")
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _get_model(self, name: str, data: Any, num_classes: int) -> torch.nn.Module:
        """Model factory that correctly handles parameters for all models."""
        # Standardized parameters for most models
        model_params = {
            'in_channels': data.num_features,
            'hidden_channels': 256,
            'out_channels': num_classes,
            'num_layers': 2,
            'dropout_rate': 0.5
        }

        if name == "GCN":
            return GCN(**model_params)
        elif name == "GAT":
            gat_params = model_params.copy()
            gat_params.update({'heads': 8, 'dropout_rate': 0.6})
            return GAT(**gat_params)
        elif name == "GraphSAGE":
            return GraphSAGE(**model_params)
        elif name == "GIN":
            return GIN(**model_params)
        elif name == "ChebNet":
            return ChebNet(**model_params, K=3)
        elif name == "RGCN":
            # For benchmarks, we assume 2 relations: forward and backward (implicit)
            return RGCN(**model_params, num_relations=2)
        elif name == "TongDiGCN":
            return TongDiGCN(**model_params)
        elif name == "DirectGCN":
            # The DirectGCN model has a complex signature that must be
            # adapted for standard benchmark datasets.
            layer_dims = [data.num_features, 128, num_classes]
            return DirectGCN(
                layer_dims=layer_dims,
                num_graph_nodes=data.num_nodes,
                task_num_output_classes=num_classes,
                n_gram_len=0,  # Not applicable for standard benchmarks
                one_gram_dim=0,  # Not applicable for standard benchmarks
                max_pe_len=0,  # Not applicable for standard benchmarks
                dropout=0.5,
                use_vector_coeffs=False  # Use scalar coeffs for general benchmarks
            )
        else:
            raise ValueError(f"Model '{name}' not found in GNNBenchmarker.")

    def _preprocess_for_custom_models(self, data: Data) -> Data:
        """Prepares a data object with all necessary edge indices for custom models."""
        # For TongDiGCN, which needs a backward edge index
        data.edge_index_backward = data.edge_index.flip(0)

        # For DirectGCN, which needs separate in, out, and undirected matrices
        edge_index_undir = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = edge_index_undir
        deg = torch.bincount(col, minlength=data.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight_undir = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        data.edge_index_undirected_norm = edge_index_undir
        data.edge_weight_undirected_norm = edge_weight_undir
        data.edge_index_out = data.edge_index
        data.edge_weight_out = None  # No explicit weights for standard benchmarks
        data.edge_index_in = data.edge_index.flip(0)
        data.edge_weight_in = None  # No explicit weights for standard benchmarks
        return data

    def _get_1d_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to handle masks from datasets that may have multiple splits (e.g., WebKB)."""
        if mask_tensor.dim() > 1:
            return mask_tensor[:, 0].bool()
        return mask_tensor.bool()

    def train_and_evaluate(self, model: torch.nn.Module, data: Data) -> Tuple[float, float, pd.DataFrame]:
        """Handles the training and evaluation loop for a given model and data."""
        model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_acc = -1
        test_acc_at_best_val = -1
        history = {'epoch': [], 'loss': [], 'val_acc': [], 'test_acc': []}

        train_mask = self._get_1d_mask(data.train_mask)
        val_mask = self._get_1d_mask(data.val_mask)
        test_mask = self._get_1d_mask(data.test_mask)

        for epoch in range(1, self.config.EVAL_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_eval, _ = model(data)
                pred = logits_eval.argmax(dim=1)
                val_correct = (pred[val_mask] == data.y[val_mask]).sum()
                val_acc = int(val_correct) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0
                test_correct = (pred[test_mask] == data.y[test_mask]).sum()
                test_acc = int(test_correct) / int(test_mask.sum()) if test_mask.sum() > 0 else 0.0

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc

            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == self.config.EVAL_EPOCHS):
                print(f"    Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

        print(f"  Finished training. Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: {test_acc_at_best_val:.4f}")

        if self.config.BENCHMARK_SAVE_EMBEDDINGS:
            self._save_embeddings(model, data)

        return best_val_acc, test_acc_at_best_val, pd.DataFrame(history)

    def _save_embeddings(self, model: torch.nn.Module, data: Data):
        """Extracts, processes (with PCA), and saves embeddings."""
        print(f"    Extracting embeddings for {model.__class__.__name__}...")
        with torch.no_grad():
            model.eval()
            _, embeddings = model(data.to(self.device))

        if embeddings is None:
            print("    Warning: Could not extract embeddings.")
            return

        embeddings_np = embeddings.cpu().numpy()
        final_embedding_dim = embeddings_np.shape[1]
        output_suffix = f"_dim{final_embedding_dim}"

        if self.config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and embeddings_np.shape[0] > self.config.BENCHMARK_PCA_TARGET_DIM:
            print(f"      Applying PCA (target dim: {self.config.BENCHMARK_PCA_TARGET_DIM})...")
            embeddings_for_pca = {i: emb for i, emb in enumerate(embeddings_np)}
            pca_embed_dict = EmbeddingProcessor.apply_pca(embeddings_for_pca, self.config.BENCHMARK_PCA_TARGET_DIM, self.config.RANDOM_STATE)
            if pca_embed_dict:
                embeddings_np = np.array(list(pca_embed_dict.values()))
                final_embedding_dim = embeddings_np.shape[1]
                output_suffix = f"_pca{final_embedding_dim}"

        emb_dict = {str(i): embeddings_np[i] for i in range(embeddings_np.shape[0])}
        save_path_emb_dir = self.embedding_dir / data.name
        save_path_emb_dir.mkdir(parents=True, exist_ok=True)
        h5_path = save_path_emb_dir / f"{model.__class__.__name__}_embeddings{output_suffix}.h5"
        DataUtils.write_h5(emb_dict, h5_path, f"Writing H5 for {model.__class__.__name__}")
        print(f"      Saved embeddings to {h5_path}")

    def _run_on_dataset_variant(self, dataset: Any, variant_name: str) -> List[Dict]:
        """Runs all configured models on a single dataset variant."""
        print(f"\n" + "=" * 50)
        print(f"### Benchmarking on Dataset: {variant_name} ###")
        print("=" * 50 + "\n")

        data = dataset[0]
        data.name = variant_name

        # --- FIX for CUDA device-side assert ---
        # Instead of trusting dataset.num_classes, derive it directly from the labels.
        # This prevents errors if labels are e.g., [1, 2, 3, 4] but num_classes is reported as 4.
        num_classes = int(data.y.max().item()) + 1
        # --- END FIX ---

        # Handle data splits
        if not all(hasattr(data, mask) and getattr(data, mask) is not None and getattr(data, mask).any() for mask in ['train_mask', 'val_mask', 'test_mask']):
            print(f"  Generating custom seeded split for {variant_name}.")
            num_nodes = data.num_nodes
            indices = np.random.permutation(num_nodes)
            train_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['train'])
            val_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['val'])
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
        else:
            print(f"  Using existing standard masks for {variant_name}.")

        print(f"  {variant_name.split('_')[0]} loaded: Nodes={data.num_nodes}, Edges={data.num_edges}, Features={data.num_features}, Classes={num_classes}")

        # Preprocess data once for all custom models
        data = self._preprocess_for_custom_models(data)

        results = []
        for model_name in self.config.BENCHMARK_GNN_MODELS_TO_RUN:
            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {variant_name} ---")
            try:
                model = self._get_model(model_name, data, num_classes)
                print("  Model Architecture:")
                print(model)

                val_acc, test_acc, history_df = self.train_and_evaluate(model, data)
                results.append({"dataset": variant_name, "model": model_name, "best_val_accuracy": val_acc, "test_accuracy": test_acc, "error": None})

                history_path = self.output_dir / variant_name
                history_path.mkdir(parents=True, exist_ok=True)
                history_df.to_csv(history_path / f"benchmark_{model_name}_history.csv", index=False)
                print(f"  Saved training history to {history_path / f'benchmark_{model_name}_history.csv'}")

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {variant_name}: {e}")
                traceback.print_exc()
                results.append({"dataset": variant_name, "model": model_name, "best_val_accuracy": None, "test_accuracy": None, "error": str(e)})
        return results

    def run(self):
        """Main execution function for the benchmarker."""
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        all_results = []
        print(f"Standard PyG datasets will be stored in/loaded from: {self.dataset_root}")

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset_results = []

            # --- Run on Original (potentially directed) Graph ---
            dataset_original = self._get_dataset(dataset_name, undirected=False)
            if dataset_original:
                dataset_results.extend(self._run_on_dataset_variant(dataset_original, f"{dataset_name}_Original"))

            # --- Run on Undirected Graph (if configured) ---
            if self.config.BENCHMARK_TEST_ON_UNDIRECTED:
                dataset_undirected = self._get_dataset(dataset_name, undirected=True)
                if dataset_undirected:
                    dataset_results.extend(self._run_on_dataset_variant(dataset_undirected, f"{dataset_name}_Undirected"))

            # --- Save summary for the current dataset ---
            if dataset_results:
                summary_df = pd.DataFrame(dataset_results)
                summary_path = self.output_dir / f"benchmark_summary_{dataset_name}.csv"
                DataUtils.save_dataframe_to_csv(summary_df, str(summary_path))
                print(f"\nSummary for {dataset_name} saved to {summary_path}")
                print(summary_df.to_string())
                all_results.extend(dataset_results)

        # --- Save a final, grand summary of all results ---
        if all_results:
            full_summary_df = pd.DataFrame(all_results)
            full_summary_path = self.output_dir / "gnn_benchmark_FULL_SUMMARY.csv"
            DataUtils.save_dataframe_to_csv(full_summary_df, str(full_summary_path))
            print(f"\nFull GNN benchmarking summary saved to {full_summary_path}")
            print("\nFull Summary Table:")
            print(full_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")