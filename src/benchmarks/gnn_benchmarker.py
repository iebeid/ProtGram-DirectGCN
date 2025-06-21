# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 3.4.4 (Reduced per-epoch print verbosity)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.datasets import (KarateClub, Planetoid)
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from src.models.gnn_zoo import *
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataUtils
from src.utils.models_utils import EmbeddingProcessor


class GNNBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(str(self.config.BENCHMARK_EMBEDDINGS_DIR), exist_ok=True)
        print(f"GNNBenchmarker initialized. Using device: {self.device}")
        print(f"Benchmark embeddings will be saved to: {self.config.BENCHMARK_EMBEDDINGS_DIR}")

    def _get_dataset(self, name: str, root_path: str, make_undirected: bool = False):
        transform_list = [T.ToDevice(self.device)]
        transform_compose = T.Compose(transform_list)

        print(f"  Attempting to load dataset: {name} (root: {root_path}, undirected_requested: {make_undirected})...")

        dataset_obj = None
        data_obj = None

        if name.lower() == 'karateclub':
            dataset_obj = KarateClub(transform=None)
        elif name.lower() in ['cora', 'citeseer', 'pubmed']:
            dataset_obj = Planetoid(root=root_path, name=name, transform=None)
        elif name.lower() in ['cornell', 'texas', 'wisconsin']:
            from torch_geometric.datasets import WebKB
            try:
                dataset_obj = WebKB(root=root_path, name=name, transform=None)
            except Exception as e:
                print(f"Warning: Could not load {name} using WebKB loader: {e}")
                return None, None, None, None
        else:
            print(f"ERROR: Dataset '{name}' loader not implemented or dataset is excluded from benchmark list.")
            return None, None, None, None

        if dataset_obj is None or not hasattr(dataset_obj, 'data'):
            print(f"ERROR: Could not load data for dataset '{name}'.")
            return None, None, None, None

        data_obj = dataset_obj[0]

        if make_undirected:
            print(f"  Converting {name} to undirected graph.")
            if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
                edge_index_undir, edge_attr_undir = to_undirected(data_obj.edge_index, data_obj.edge_attr, num_nodes=data_obj.num_nodes, reduce="mean")
                data_obj.edge_index = edge_index_undir
                data_obj.edge_attr = edge_attr_undir
            else:
                data_obj.edge_index = to_undirected(data_obj.edge_index, num_nodes=data_obj.num_nodes)
            data_obj.is_undirected_explicit = True

        data_obj = transform_compose(data_obj)

        print(f"  {name} loaded: Nodes: {data_obj.num_nodes}, Edges: {data_obj.num_edges}, Features: {dataset_obj.num_features}, Classes: {dataset_obj.num_classes}")

        has_standard_masks = hasattr(data_obj, 'train_mask') and hasattr(data_obj, 'val_mask') and hasattr(data_obj, 'test_mask')
        if has_standard_masks and data_obj.train_mask.ndim > 1:
            print(f"  Dataset {name} has multiple standard mask splits. Using the first one (index 0).")
            data_obj.train_mask = data_obj.train_mask[:, 0]
            data_obj.val_mask = data_obj.val_mask[:, 0]
            data_obj.test_mask = data_obj.test_mask[:, 0]
            has_standard_masks = True

        if not has_standard_masks:
            print(f"  Dataset {name} is missing standard train/val/test masks. Applying RandomNodeSplit.")
            num_train = int(self.config.BENCHMARK_SPLIT_RATIOS['train'] * data_obj.num_nodes)
            num_val = int(self.config.BENCHMARK_SPLIT_RATIOS['val'] * data_obj.num_nodes)
            num_test = data_obj.num_nodes - num_train - num_val

            if num_train <= 0 or num_val <= 0 or num_test <= 0:
                print(f"ERROR: Calculated split sizes for {name} are invalid (train:{num_train}, val:{num_val}, test:{num_test}). Min 1 node per split needed. Skipping dataset.")
                return None, None, None, None

            split_transform = T.RandomNodeSplit(split='train_rest', num_val=num_val, num_test=num_test)
            data_on_cpu_for_split = data_obj.to('cpu')
            try:
                data_on_cpu_for_split = split_transform(data_on_cpu_for_split)
                data_obj = data_on_cpu_for_split.to(self.device)
                print(f"  Applied RandomNodeSplit to {name}. Train: {data_obj.train_mask.sum()}, Val: {data_obj.val_mask.sum()}, Test: {data_obj.test_mask.sum()}")
            except Exception as e_split:
                print(f"ERROR applying RandomNodeSplit to {name}: {e_split}. Skipping dataset.")
                return None, None, None, None

        return data_obj, data_obj, data_obj, 'single-label-node'

    def _save_node_embeddings(self, model: nn.Module, data_for_emb, model_name: str, dataset_name: str, graph_variant_suffix: str):
        if not self.config.BENCHMARK_SAVE_EMBEDDINGS:
            return

        print(f"    Extracting embeddings for {model_name} on {dataset_name}{graph_variant_suffix}...")
        model.eval()
        with torch.no_grad():
            node_embeddings_tensor = None
            if isinstance(data_for_emb, Data):
                # Special handling for ProtGramDirectGCN's forward method
                if model_name == "ProtGramDirectGCN":
                    # ProtGramDirectGCN's forward returns (logits, embeddings)
                    # It expects edge_index_in/out, so we need to ensure data_for_emb has them
                    # If data_for_emb is the original PyG Data object, we need to adapt it here too
                    pg_data_for_emb = data_for_emb.clone()
                    pg_data_for_emb.edge_index_out = pg_data_for_emb.edge_index
                    # --- MODIFIED: Use modern syntax for transpose ---
                    pg_data_for_emb.edge_index_in = pg_data_for_emb.edge_index[[1, 0]]
                    # --- END MODIFIED ---
                    pg_data_for_emb.edge_weight_out = getattr(pg_data_for_emb, 'edge_attr', None)
                    pg_data_for_emb.edge_weight_in = getattr(pg_data_for_emb, 'edge_attr', None)  # Assuming symmetric weights for transpose

                    _, node_embeddings_tensor = model(data=pg_data_for_emb.to(self.device))

                elif hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
                    node_embeddings_tensor = model.get_embeddings(data_for_emb.to(self.device))
                elif hasattr(model, 'embedding_output') and model.embedding_output is not None:
                    node_embeddings_tensor = model.embedding_output
                else:
                    output = model(data_for_emb.to(self.device))
                    if isinstance(output, tuple):
                        node_embeddings_tensor = output[1] if len(output) > 1 and output[1] is not None else output[0]
                    else:
                        node_embeddings_tensor = output
                    if node_embeddings_tensor.shape[-1] == getattr(data_for_emb, 'num_classes', -1) or \
                            node_embeddings_tensor.shape[-1] == getattr(data_for_emb, 'y', torch.tensor([-1])).max().item() + 1:
                        print(f"      Using final layer output (likely logits) as embeddings for {model_name}.")

                if node_embeddings_tensor is None:
                    print(f"      Could not extract embeddings for {model_name}. Skipping save.")
                    return
                node_embeddings_np = node_embeddings_tensor.cpu().numpy()
            else:
                print(f"      Unsupported data type for embedding extraction: {type(data_for_emb)}. Skipping save.")
                return

        if node_embeddings_np.size == 0:
            print(f"      Embeddings are empty for {model_name}. Skipping save.")
            return

        emb_dim = node_embeddings_np.shape[1]
        filename_suffix = f"_dim{emb_dim}"

        if self.config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and \
                emb_dim > self.config.BENCHMARK_PCA_TARGET_DIM and \
                node_embeddings_np.shape[0] > self.config.BENCHMARK_PCA_TARGET_DIM:
            print(f"      Applying PCA to {model_name} embeddings (target dim: {self.config.BENCHMARK_PCA_TARGET_DIM})...")
            temp_emb_dict = {str(i): node_embeddings_np[i] for i in range(node_embeddings_np.shape[0])}
            pca_embeds_dict = EmbeddingProcessor.apply_pca(temp_emb_dict, self.config.BENCHMARK_PCA_TARGET_DIM, self.config.RANDOM_STATE, output_dtype=np.float32)
            if pca_embeds_dict:
                pca_node_embeddings_np = np.array([pca_embeds_dict[str(i)] for i in range(len(pca_embeds_dict))])
                if pca_node_embeddings_np.size > 0:
                    node_embeddings_np = pca_node_embeddings_np
                    emb_dim = node_embeddings_np.shape[1]
                    filename_suffix = f"_pca{emb_dim}"
                    print(f"      PCA applied. New embedding dim: {emb_dim}")
                else:
                    print(f"      PCA resulted in empty embeddings for {model_name}, using full dimension embeddings.")
            else:
                print(f"      PCA failed or was skipped for {model_name}, using full dimension embeddings.")

        emb_dir = os.path.join(str(self.config.BENCHMARK_EMBEDDINGS_DIR), f"{dataset_name}{graph_variant_suffix}")
        os.makedirs(emb_dir, exist_ok=True)
        emb_filename = f"{model_name}_embeddings{filename_suffix}.h5"
        emb_path = os.path.join(emb_dir, emb_filename)

        try:
            with h5py.File(emb_path, 'w') as hf:
                hf.create_dataset('embeddings', data=node_embeddings_np)
                hf.attrs['model_name'] = model_name
                hf.attrs['dataset_name'] = dataset_name
                hf.attrs['graph_variant'] = graph_variant_suffix
                hf.attrs['shape'] = node_embeddings_np.shape
            print(f"      Saved {model_name} embeddings for {dataset_name}{graph_variant_suffix} to {emb_path}")
        except Exception as e:
            print(f"      ERROR saving embeddings to {emb_path}: {e}")

    def train_and_evaluate(
            self, model_name_str: str, model: nn.Module, dataset_name: str,
            train_data_or_loader, val_data_or_loader, test_data_or_loader,
            optimizer: torch.optim.Optimizer, num_epochs: int, task_type: str,
            num_classes: int):

        best_val_metric = 0.0
        best_test_metric_at_best_val = 0.0
        history = []
        final_trained_model_state = None

        model.to(self.device)
        print(f"  Training {model_name_str} on {dataset_name} using device: {self.device} for {num_epochs} epochs.")
        print(f"  Task type: {task_type}, Num classes: {num_classes}, Optimizer: {type(optimizer).__name__}, LR: {optimizer.defaults.get('lr')}")

        criterion = None
        metric_name = ""

        if task_type == 'multi-label-graph':
            criterion = F.binary_cross_entropy_with_logits
            metric_fn = lambda y_true, y_pred_logits: f1_score(y_true.cpu().numpy(), (y_pred_logits.detach() > 0).cpu().numpy(), average='micro', zero_division=0)
            metric_name = "F1 Micro"
        elif task_type == 'single-label-node':
            criterion = F.cross_entropy
            metric_fn = lambda y_true, y_pred_logits: accuracy_score(y_true.cpu().numpy(), y_pred_logits.detach().argmax(dim=-1).cpu().numpy())
            metric_name = "Accuracy"
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        print(f"  Using Loss: {criterion.__name__ if hasattr(criterion, '__name__') else str(criterion)}, Metric: {metric_name}")

        for epoch in range(num_epochs):
            epoch_start_time = time.monotonic()
            model.train()
            total_loss = 0
            num_batches_or_graphs = 0

            current_train_data = train_data_or_loader
            # Special handling for ProtGramDirectGCN's input format
            if model_name_str == "ProtGramDirectGCN":
                # Clone and adapt the data object for ProtGramDirectGCN's specific input
                # edge_index_out is the original edge_index (source -> target)
                # edge_index_in is the transpose of the original edge_index (target -> source)
                adapted_data = current_train_data.clone()
                adapted_data.edge_index_out = adapted_data.edge_index
                # --- MODIFIED: Use modern syntax for transpose ---
                adapted_data.edge_index_in = adapted_data.edge_index[[1, 0]]
                # --- END MODIFIED ---
                adapted_data.edge_weight_out = getattr(adapted_data, 'edge_attr', None)
                adapted_data.edge_weight_in = getattr(adapted_data, 'edge_attr', None)  # Assuming symmetric weights for transpose
                current_train_data = adapted_data.to(self.device)
            elif isinstance(current_train_data, Data):
                current_train_data = current_train_data.to(self.device)

            if isinstance(current_train_data, PyGDataLoader):
                for batch_data in current_train_data:
                    optimizer.zero_grad()
                    out = model(batch_data)
                    if isinstance(out, tuple): out = out[0]
                    loss = criterion(out, batch_data.y.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches_or_graphs += 1
            elif isinstance(current_train_data, Data):
                optimizer.zero_grad()
                # ProtGramDirectGCN's forward returns (logits, embeddings)
                if model_name_str == "ProtGramDirectGCN":
                    out, _ = model(data=current_train_data)
                else:
                    out = model(current_train_data)
                    if isinstance(out, tuple): out = out[0]

                if current_train_data.train_mask.sum() == 0:
                    print(f"Warning: Epoch {epoch:03d}, No training nodes in train_mask for {dataset_name}.")
                    loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                else:
                    loss = criterion(out[current_train_data.train_mask], current_train_data.y[current_train_data.train_mask].long())

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                num_batches_or_graphs = 1
            else:
                raise TypeError(f"train_data_or_loader type {type(current_train_data)} not supported.")

            avg_loss = total_loss / num_batches_or_graphs if num_batches_or_graphs > 0 else 0

            model.eval()
            with torch.no_grad():
                current_val_data = val_data_or_loader
                # Special handling for ProtGramDirectGCN's input format for validation
                if model_name_str == "ProtGramDirectGCN":
                    adapted_data_val = current_val_data.clone()
                    adapted_data_val.edge_index_out = adapted_data_val.edge_index
                    # --- MODIFIED: Use modern syntax for transpose ---
                    adapted_data_val.edge_index_in = adapted_data_val.edge_index[[1, 0]]
                    # --- END MODIFIED ---
                    adapted_data_val.edge_weight_out = getattr(adapted_data_val, 'edge_attr', None)
                    adapted_data_val.edge_weight_in = getattr(adapted_data_val, 'edge_attr', None)
                    current_val_data = adapted_data_val.to(self.device)
                elif isinstance(current_val_data, Data):
                    current_val_data = current_val_data.to(self.device)

                if isinstance(current_val_data, PyGDataLoader):
                    all_preds_val, all_labels_val = [], []
                    for batch_data_val in current_val_data:
                        preds_val = model(batch_data_val)
                        if isinstance(preds_val, tuple): preds_val = preds_val[0]
                        all_preds_val.append(preds_val)
                        all_labels_val.append(batch_data_val.y)
                    val_preds_cat = torch.cat(all_preds_val)
                    val_labels_cat = torch.cat(all_labels_val)
                    val_metric = metric_fn(val_labels_cat, val_preds_cat)
                elif isinstance(current_val_data, Data):
                    # ProtGramDirectGCN's forward returns (logits, embeddings)
                    if model_name_str == "ProtGramDirectGCN":
                        out_val, _ = model(data=current_val_data)
                    else:
                        out_val = model(current_val_data)
                        if isinstance(out_val, tuple): out_val = out_val[0]

                    if current_val_data.val_mask.sum() == 0:
                        val_metric = 0.0
                    else:
                        val_metric = metric_fn(current_val_data.y[current_val_data.val_mask], out_val[current_val_data.val_mask])
                else:
                    raise TypeError(f"val_data_or_loader type {type(current_val_data)} not supported.")

            epoch_duration = time.monotonic() - epoch_start_time
            # --- MODIFIED: Reduced print verbosity ---
            if epoch == num_epochs - 1: # Only print for the last epoch
                print(f"    Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val {metric_name}: {val_metric:.4f}, Time: {epoch_duration:.2f}s")
            # --- END MODIFIED ---
            history.append({'epoch': epoch, 'loss': avg_loss, f'val_{metric_name.lower().replace(" ", "_")}': val_metric})

            if val_metric >= best_val_metric:
                best_val_metric = val_metric
                final_trained_model_state = model.state_dict()

                current_test_data = test_data_or_loader
                # Special handling for ProtGramDirectGCN's input format for testing
                if model_name_str == "ProtGramDirectGCN":
                    adapted_data_test = current_test_data.clone()
                    adapted_data_test.edge_index_out = adapted_data_test.edge_index
                    # --- MODIFIED: Use modern syntax for transpose ---
                    adapted_data_test.edge_index_in = adapted_data_test.edge_index[[1, 0]]
                    # --- END MODIFIED ---
                    adapted_data_test.edge_weight_out = getattr(adapted_data_test, 'edge_attr', None)
                    adapted_data_test.edge_weight_in = getattr(adapted_data_test, 'edge_attr', None)
                    current_test_data = adapted_data_test.to(self.device)
                elif isinstance(current_test_data, Data):
                    current_test_data = current_test_data.to(self.device)

                if isinstance(current_test_data, PyGDataLoader):
                    all_preds_test, all_labels_test = [], []
                    for batch_data_test in current_test_data:
                        preds_test = model(batch_data_test)
                        if isinstance(preds_test, tuple): preds_test = preds_test[0]
                        all_preds_test.append(preds_test)
                        all_labels_test.append(batch_data_test.y)
                    test_preds_cat = torch.cat(all_preds_test)
                    test_labels_cat = torch.cat(all_labels_test)
                    best_test_metric_at_best_val = metric_fn(test_labels_cat, test_preds_cat)
                elif isinstance(current_test_data, Data):
                    # ProtGramDirectGCN's forward returns (logits, embeddings)
                    if model_name_str == "ProtGramDirectGCN":
                        out_test, _ = model(data=current_test_data)
                    else:
                        out_test = model(current_test_data)
                        if isinstance(out_test, tuple): out_test = out_test[0]
                    if current_test_data.test_mask.sum() == 0:
                        best_test_metric_at_best_val = 0.0
                    else:
                        best_test_metric_at_best_val = metric_fn(current_test_data.y[current_test_data.test_mask], out_test[current_test_data.test_mask])

        print(f"  Finished training for {model_name_str} on {dataset_name}.")
        print(f"  Best Val {metric_name}: {best_val_metric:.4f}, Corresponding Test {metric_name}: {best_test_metric_at_best_val:.4f}")

        if final_trained_model_state:
            model.load_state_dict(final_trained_model_state)

        return best_val_metric, best_test_metric_at_best_val, pd.DataFrame(history), metric_name

    def run_on_dataset_variant(self, dataset_name: str, model_zoo_config: Dict,
                               train_data, val_data, test_data, task_type: str,
                               num_features: int, num_classes: int, graph_variant_suffix: str = ""):
        dataset_results = []
        DataUtils.print_header(f"Benchmarking on Dataset: {dataset_name}{graph_variant_suffix}")

        # --- NEW: Handle ProtGramDirectGCN as a special case first ---
        print(f"\n--- Benchmarking Model: ProtGramDirectGCN on Dataset: {dataset_name}{graph_variant_suffix} ---")
        try:
            # ProtGramDirectGCN specific initialization
            # It uses GCN_HIDDEN_LAYER_DIMS from the main config
            layer_dims = [num_features] + self.config.GCN_HIDDEN_LAYER_DIMS + [num_classes]

            model_instance = ProtGramDirectGCN(
                layer_dims=layer_dims,
                num_graph_nodes=train_data.num_nodes,  # Pass actual number of nodes for vector coeffs
                task_num_output_classes=num_classes,
                n_gram_len=0,  # Not applicable for these datasets, set to 0
                one_gram_dim=0,  # Not applicable, set to 0
                max_pe_len=0,  # Not applicable, set to 0
                dropout=self.config.GCN_DROPOUT_RATE,
                use_vector_coeffs=self.config.GCN_USE_VECTOR_COEFFS
            ).to(self.device)

            if self.config.DEBUG_VERBOSE: print(f"  Model Architecture:\n{model_instance}")
            optimizer = torch.optim.Adam(model_instance.parameters(), lr=self.config.EVAL_LEARNING_RATE, weight_decay=5e-4)

            # train_and_evaluate will now handle adapting the Data object for ProtGramDirectGCN
            val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                "ProtGramDirectGCN", model_instance, f"{dataset_name}{graph_variant_suffix}",
                train_data, val_data, test_data,  # Pass the original Data objects
                optimizer, num_epochs=self.config.EVAL_EPOCHS, task_type=task_type,
                num_classes=num_classes
            )
            result_entry = {
                'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': "ProtGramDirectGCN",
                f'best_val_{metric_name_used.lower().replace(" ", "_")}': val_metric,
                f'test_{metric_name_used.lower().replace(" ", "_")}': test_metric
            }
            dataset_results.append(result_entry)

            # _save_node_embeddings also needs to handle ProtGramDirectGCN's specific input
            self._save_node_embeddings(model_instance, test_data, "ProtGramDirectGCN", dataset_name, graph_variant_suffix)

            reports_dir = str(self.config.BENCHMARKING_RESULTS_DIR / f"{dataset_name}{graph_variant_suffix}")
            os.makedirs(reports_dir, exist_ok=True)
            history_path = os.path.join(reports_dir, f'benchmark_ProtGramDirectGCN_history.csv')
            history_df.to_csv(history_path, index=False)
            print(f"  Saved ProtGramDirectGCN training history for {dataset_name}{graph_variant_suffix} to {history_path}")

        except Exception as e:
            print(f"ERROR during training/evaluation of ProtGramDirectGCN on {dataset_name}{graph_variant_suffix}: {e}")
            import traceback
            traceback.print_exc()
            dataset_results.append({'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': "ProtGramDirectGCN", 'error': str(e)})
        # --- End of special case for ProtGramDirectGCN ---

        # --- Loop for other standard models ---
        for model_name, m_config in model_zoo_config.items():
            # Skip ProtGramDirectGCN here as it's already handled above
            if model_name == "ProtGramDirectGCN":
                continue

            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {dataset_name}{graph_variant_suffix} ---")
            try:
                # Standard models receive the original Data object (train_data, val_data, test_data)
                model_instance = m_config["class"](in_channels=num_features, out_channels=num_classes, **m_config["params"]).to(self.device)
                if self.config.DEBUG_VERBOSE: print(f"  Model Architecture:\n{model_instance}")
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=self.config.EVAL_LEARNING_RATE, weight_decay=5e-4)

                val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                    model_name, model_instance, f"{dataset_name}{graph_variant_suffix}",
                    train_data, val_data, test_data,  # Pass the original data objects
                    optimizer, num_epochs=self.config.EVAL_EPOCHS, task_type=task_type,
                    num_classes=num_classes
                )
                result_entry = {
                    'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name,
                    f'best_val_{metric_name_used.lower().replace(" ", "_")}': val_metric,
                    f'test_{metric_name_used.lower().replace(" ", "_")}': test_metric
                }
                dataset_results.append(result_entry)

                data_for_emb_extraction = test_data
                self._save_node_embeddings(model_instance, data_for_emb_extraction, model_name, dataset_name, graph_variant_suffix)

                reports_dir = str(self.config.BENCHMARKING_RESULTS_DIR / f"{dataset_name}{graph_variant_suffix}")
                os.makedirs(reports_dir, exist_ok=True)
                history_path = os.path.join(reports_dir, f'benchmark_{model_name}_history.csv')
                history_df.to_csv(history_path, index=False)
                print(f"  Saved {model_name} training history for {dataset_name}{graph_variant_suffix} to {history_path}")

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {dataset_name}{graph_variant_suffix}: {e}")
                import traceback
                traceback.print_exc()
                dataset_results.append({'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name, 'error': str(e)})
        return dataset_results

    def run(self):
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        base_dataset_path = str(self.config.BASE_DATA_DIR / "standard_datasets_pyg")
        os.makedirs(base_dataset_path, exist_ok=True)
        print(f"Standard PyG datasets will be stored in/loaded from: {base_dataset_path}")

        all_results_summary = []

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            if dataset_name.lower() == 'ppi':
                print(f"Skipping dataset '{dataset_name}' as per requirement.")
                continue

            train_data_orig, val_data_orig, test_data_orig, task_type_orig = self._get_dataset(dataset_name, base_dataset_path, make_undirected=False)
            if train_data_orig is None:
                print(f"Skipping dataset {dataset_name} (original) due to loading error or missing masks.")
                continue

            num_features, num_classes = 0, 0
            if task_type_orig == 'single-label-node':
                num_features = train_data_orig.num_features
                num_classes = train_data_orig.y.max().item() + 1 if train_data_orig.y is not None and train_data_orig.y.numel() > 0 else 1
                if hasattr(train_data_orig, 'num_classes') and train_data_orig.num_classes is not None:
                    num_classes = train_data_orig.num_classes
                train_loader_orig, val_loader_orig, test_loader_orig = train_data_orig, val_data_orig, test_data_orig
            else:
                print(f"Unexpected task type '{task_type_orig}' for dataset {dataset_name}. Skipping.")
                continue

            # Define the standard model zoo configuration
            model_zoo_cfg = {
                "GCN": {"class": GCN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GAT": {"class": GAT, "params": {"hidden_channels": 32, "heads": 8, "num_layers": 2, "dropout_rate": 0.6}},
                "GraphSAGE": {"class": GraphSAGE, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GIN": {"class": GIN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "ChebNet": {"class": ChebNet, "params": {"hidden_channels": 256, "K": 3, "num_layers": 2, "dropout_rate": 0.5}},
                "RGCN_SR": {"class": RGCN, "params": {"hidden_channels": 256, "num_relations": 1, "num_layers": 2, "dropout_rate": 0.5}},
                "TongDiGCN": {"class": TongDiGCN, "params": {"hidden_channels": 128, "num_layers": 2, "dropout_rate": 0.5}},
            }

            # Run on the original (potentially directed) graph variant
            results_orig = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                       train_loader_orig, val_loader_orig, test_loader_orig,
                                                       task_type_orig, num_features, num_classes, "_Original")
            all_results_summary.extend(results_orig)

            # Run on the explicitly undirected graph variant if configured
            if self.config.BENCHMARK_TEST_ON_UNDIRECTED and task_type_orig == 'single-label-node':
                train_data_undir, val_data_undir, test_data_undir, task_type_undir = self._get_dataset(
                    dataset_name, base_dataset_path, make_undirected=True)
                if train_data_undir is not None:
                    results_undir = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                                train_data_undir, val_data_undir, test_data_undir,
                                                                task_type_undir, num_features, num_classes, "_Undirected")
                    all_results_summary.extend(results_undir)
                else:
                    print(f"Skipping undirected variant for {dataset_name} due to loading error.")

            # Save summary for the current dataset
            current_dataset_summary = [res for res in all_results_summary if res['dataset'].startswith(dataset_name)]
            if current_dataset_summary:
                dataset_summary_df = pd.DataFrame(current_dataset_summary)
                dataset_summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), f"benchmark_summary_{dataset_name}.csv")
                DataUtils.save_dataframe_to_csv(dataset_summary_df, dataset_summary_path)
                print(f"\nCombined Summary for dataset {dataset_name} (all variants) saved to {dataset_summary_path}")
                print(dataset_summary_df)

        # Save overall summary
        if all_results_summary:
            final_summary_df = pd.DataFrame(all_results_summary)
            final_summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), "gnn_benchmark_FULL_SUMMARY.csv")
            DataUtils.save_dataframe_to_csv(final_summary_df, final_summary_path)
            print(f"\nFull GNN benchmarking summary saved to {final_summary_path}")
            print("\nFull Summary Table:")
            print(final_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")