# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 3.1 (Save embeddings, PCA, undirected variants, RandomNodeSplit)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import time
import shutil  # For saving embeddings

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.datasets import (PPI, KarateClub, Planetoid, HeterophilousGraphDataset)
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected  # For creating undirected versions
import h5py  # For saving embeddings
import numpy as np  # For PCA

from src.config import Config
from src.models.gnn_zoo import *
from src.models.protgram_directgcn import ProtGramDirectGCN  # Assuming this is your custom model
from src.utils.data_utils import DataUtils
from src.utils.models_utils import EmbeddingProcessor  # For PCA


class GNNBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(str(self.config.BENCHMARK_EMBEDDINGS_DIR), exist_ok=True)
        print(f"GNNBenchmarker initialized. Using device: {self.device}")
        print(f"Benchmark embeddings will be saved to: {self.config.BENCHMARK_EMBEDDINGS_DIR}")

    def _get_dataset(self, name: str, root_path: str, make_undirected: bool = False):
        """Loads a specified dataset, optionally makes it undirected, and applies splits if needed."""
        transform_list = [T.ToDevice(self.device)]
        # if make_undirected: # Apply to_undirected before ToDevice if graph is modified
        #     transform_list.insert(0, T.ToUndirected()) # This transform works on Data object

        transform_compose = T.Compose(transform_list)

        print(f"  Attempting to load dataset: {name} (root: {root_path}, undirected_requested: {make_undirected})...")
        dataset_path = os.path.join(root_path, name)

        dataset_obj = None  # PyG Dataset object
        data_obj = None  # PyG Data object

        if name.lower() == 'ppi':
            # PPI is special: it's a collection of graphs.
            # ToUndirected would need to be applied per graph if make_undirected is True.
            # For now, we load PPI as is. If undirected PPI is needed, it requires custom loop.
            if make_undirected:
                print(f"  Note: For PPI, 'make_undirected={make_undirected}' will apply ToUndirected transform to each graph in the dataset.")
                ppi_transform = T.Compose([T.ToUndirected(), T.ToDevice(self.device)])
            else:
                ppi_transform = T.ToDevice(self.device)

            train_ds = PPI(dataset_path, split='train', transform=ppi_transform)
            val_ds = PPI(dataset_path, split='val', transform=ppi_transform)
            test_ds = PPI(dataset_path, split='test', transform=ppi_transform)
            print(f"  PPI loaded: Train graphs: {len(train_ds)}, Val graphs: {len(val_ds)}, Test graphs: {len(test_ds)}")
            return train_ds, val_ds, test_ds, 'multi-label-graph'
        else:
            # Single graph datasets
            if name.lower() == 'karateclub':
                dataset_obj = KarateClub(transform=None)  # Apply transforms after loading
            elif name.lower() in ['cora', 'citeseer', 'pubmed']:
                dataset_obj = Planetoid(root=root_path, name=name, transform=None)
            elif name.lower() in ['cornell', 'texas', 'wisconsin']:
                dataset_obj = HeterophilousGraphDataset(root=root_path, name=name, transform=None)
            else:
                print(f"ERROR: Dataset '{name}' loader not implemented in _get_dataset.")
                return None, None, None, None

            if dataset_obj is None or not hasattr(dataset_obj, 'data'):
                print(f"ERROR: Could not load data for dataset '{name}'.")
                return None, None, None, None

            data_obj = dataset_obj[0]  # Get the single Data object

            if make_undirected:
                print(f"  Converting {name} to undirected graph.")
                if hasattr(data_obj, 'edge_attr') and data_obj.edge_attr is not None:
                    edge_index_undir, edge_attr_undir = to_undirected(data_obj.edge_index, data_obj.edge_attr, num_nodes=data_obj.num_nodes, reduce="mean")
                    data_obj.edge_index = edge_index_undir
                    data_obj.edge_attr = edge_attr_undir
                else:
                    data_obj.edge_index = to_undirected(data_obj.edge_index, num_nodes=data_obj.num_nodes)
                data_obj.is_undirected_explicit = True  # Mark that we made it undirected

            data_obj = transform_compose(data_obj)  # Apply ToDevice and other transforms

            print(f"  {name} loaded: Nodes: {data_obj.num_nodes}, Edges: {data_obj.num_edges}, Features: {dataset_obj.num_features}, Classes: {dataset_obj.num_classes}")

            # Handle splits for single-graph datasets
            has_standard_masks = hasattr(data_obj, 'train_mask') and hasattr(data_obj, 'val_mask') and hasattr(data_obj, 'test_mask')

            if has_standard_masks and data_obj.train_mask.ndim > 1:  # Heterophilous datasets might have multiple splits
                print(f"  Dataset {name} has multiple standard mask splits. Using the first one (index 0).")
                data_obj.train_mask = data_obj.train_mask[:, 0]
                data_obj.val_mask = data_obj.val_mask[:, 0]
                data_obj.test_mask = data_obj.test_mask[:, 0]
                has_standard_masks = True  # Re-affirm after slicing

            if not has_standard_masks:
                print(f"  Dataset {name} is missing standard train/val/test masks. Applying RandomNodeSplit.")
                num_train = int(self.config.BENCHMARK_SPLIT_RATIOS['train'] * data_obj.num_nodes)
                num_val = int(self.config.BENCHMARK_SPLIT_RATIOS['val'] * data_obj.num_nodes)
                num_test = data_obj.num_nodes - num_train - num_val  # Remaining for test

                if num_train <= 0 or num_val <= 0 or num_test <= 0:
                    print(f"ERROR: Calculated split sizes for {name} are invalid (train:{num_train}, val:{num_val}, test:{num_test}). Min 1 node per split needed. Skipping dataset.")
                    return None, None, None, None

                split_transform = T.RandomNodeSplit(
                    split='train_rest',  # PyG <2.4.0 uses this, newer might use 'random'
                    num_train_per_class=None,  # Let it distribute based on overall num_train
                    num_val=num_val,
                    num_test=num_test
                )
                # For RandomNodeSplit to work correctly, data needs to be on CPU temporarily if not already
                data_on_cpu_for_split = data_obj.to('cpu')
                try:
                    data_on_cpu_for_split = split_transform(data_on_cpu_for_split)
                    data_obj = data_on_cpu_for_split.to(self.device)  # Move back to device
                    print(f"  Applied RandomNodeSplit to {name}. Train: {data_obj.train_mask.sum()}, Val: {data_obj.val_mask.sum()}, Test: {data_obj.test_mask.sum()}")
                except Exception as e_split:
                    print(f"ERROR applying RandomNodeSplit to {name}: {e_split}. Skipping dataset.")
                    return None, None, None, None

            return data_obj, data_obj, data_obj, 'single-label-node'

    def _save_node_embeddings(self, model: nn.Module, data_for_emb, model_name: str, dataset_name: str, graph_variant_suffix: str):
        """Extracts and saves node embeddings from a trained model."""
        if not self.config.BENCHMARK_SAVE_EMBEDDINGS:
            return

        print(f"    Extracting embeddings for {model_name} on {dataset_name}{graph_variant_suffix}...")
        model.eval()
        with torch.no_grad():
            if isinstance(data_for_emb, Data):  # Single graph
                # For models that store embedding_output
                if hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
                    node_embeddings_tensor = model.get_embeddings(data_for_emb.to(self.device))
                elif hasattr(model, 'embedding_output') and model.embedding_output is not None:
                    node_embeddings_tensor = model.embedding_output
                else:  # Fallback: run forward pass again (might be redundant but ensures we get something)
                    logits, node_embeddings_tensor_alt = model(data_for_emb.to(self.device))  # Assuming forward returns (logits, embeddings) or just logits
                    if node_embeddings_tensor_alt is not None:
                        node_embeddings_tensor = node_embeddings_tensor_alt
                    else:  # If model only returns logits, use them as embeddings
                        node_embeddings_tensor = logits
                        print(f"      Using final logits as embeddings for {model_name}.")

                if node_embeddings_tensor is None:
                    print(f"      Could not extract embeddings for {model_name}. Skipping save.")
                    return
                node_embeddings_np = node_embeddings_tensor.cpu().numpy()

            elif isinstance(data_for_emb, PyGDataLoader):  # Multi-graph (e.g., PPI test set)
                all_embs_list = []
                print(f"      Extracting embeddings for multi-graph dataset {dataset_name} (this might take a moment)...")
                for batch_data in tqdm(data_for_emb, desc="      Embedding Batches", leave=False):
                    batch_data = batch_data.to(self.device)
                    if hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
                        batch_embs = model.get_embeddings(batch_data)
                    elif hasattr(model, 'embedding_output') and model.embedding_output is not None:
                        batch_embs = model.embedding_output  # Assumes forward was just called
                    else:
                        logits, batch_embs_alt = model(batch_data)
                        batch_embs = batch_embs_alt if batch_embs_alt is not None else logits
                    if batch_embs is not None:
                        all_embs_list.append(batch_embs.cpu().numpy())
                if not all_embs_list:
                    print(f"      No embeddings extracted for multi-graph {model_name}. Skipping save.")
                    return
                node_embeddings_np = np.concatenate(all_embs_list, axis=0)
            else:
                print(f"      Unsupported data type for embedding extraction: {type(data_for_emb)}. Skipping save.")
                return

        if node_embeddings_np.size == 0:
            print(f"      Embeddings are empty for {model_name}. Skipping save.")
            return

        emb_dim = node_embeddings_np.shape[1]
        filename_suffix = f"_dim{emb_dim}"

        if self.config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and emb_dim > self.config.BENCHMARK_PCA_TARGET_DIM and node_embeddings_np.shape[0] > self.config.BENCHMARK_PCA_TARGET_DIM:
            print(f"      Applying PCA to {model_name} embeddings (target dim: {self.config.BENCHMARK_PCA_TARGET_DIM})...")
            # EmbeddingProcessor.apply_pca expects a dict {id: embedding}
            # For node classification, node indices are the "ids"
            temp_emb_dict = {str(i): node_embeddings_np[i] for i in range(node_embeddings_np.shape[0])}
            pca_embeds_dict = EmbeddingProcessor.apply_pca(temp_emb_dict, self.config.BENCHMARK_PCA_TARGET_DIM, self.config.RANDOM_STATE)
            if pca_embeds_dict:
                # Reconstruct numpy array from dict, ensuring order if possible (though order might not be critical for saving)
                # This part is tricky if node IDs are not simple range(N). For now, assume they are.
                node_embeddings_np = np.array([pca_embeds_dict[str(i)] for i in range(len(pca_embeds_dict))])
                emb_dim = node_embeddings_np.shape[1]
                filename_suffix = f"_pca{emb_dim}"
                print(f"      PCA applied. New embedding dim: {emb_dim}")
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
        # ... (this method remains largely the same as in the previous response,
        #      ensure it calls self._save_node_embeddings at the end if needed)
        # ... (ensure all paths for data are .to(self.device) if not already handled by loader/transform)

        best_val_metric = 0.0
        best_test_metric_at_best_val = 0.0
        history = []
        final_trained_model_state = None  # To store the state of the best model or last epoch model

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
            epoch_start_time = time.time()
            model.train()
            total_loss = 0
            num_batches_or_graphs = 0

            current_train_data = train_data_or_loader
            if isinstance(current_train_data, Data): current_train_data = current_train_data.to(self.device)  # Ensure on device

            if isinstance(current_train_data, PyGDataLoader):
                for batch_data in current_train_data:  # Already on device via transform
                    optimizer.zero_grad()
                    out = model(batch_data)
                    loss = criterion(out, batch_data.y.float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches_or_graphs += 1
            elif isinstance(current_train_data, Data):
                optimizer.zero_grad()
                out = model(current_train_data)
                if current_train_data.train_mask.sum() == 0:
                    print(f"Warning: Epoch {epoch:03d}, No training nodes in train_mask for {dataset_name}.")
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = criterion(out[current_train_data.train_mask], current_train_data.y[current_train_data.train_mask].long())
                if loss.requires_grad:  # Avoid backward on zero loss if no samples
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
                if isinstance(current_val_data, Data): current_val_data = current_val_data.to(self.device)

                if isinstance(current_val_data, PyGDataLoader):
                    all_preds_val, all_labels_val = [], []
                    for batch_data_val in current_val_data:
                        preds_val = model(batch_data_val)
                        all_preds_val.append(preds_val)
                        all_labels_val.append(batch_data_val.y)
                    val_preds_cat = torch.cat(all_preds_val)
                    val_labels_cat = torch.cat(all_labels_val)
                    val_metric = metric_fn(val_labels_cat, val_preds_cat)
                elif isinstance(current_val_data, Data):
                    out_val = model(current_val_data)
                    if current_val_data.val_mask.sum() == 0:
                        val_metric = 0.0
                    else:
                        val_metric = metric_fn(current_val_data.y[current_val_data.val_mask], out_val[current_val_data.val_mask])
                else:
                    raise TypeError(f"val_data_or_loader type {type(current_val_data)} not supported.")

            epoch_duration = time.time() - epoch_start_time
            if self.config.DEBUG_VERBOSE or epoch % (max(1, num_epochs // 10)) == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Val {metric_name}: {val_metric:.4f}, Time: {epoch_duration:.2f}s")
            history.append({'epoch': epoch, 'loss': avg_loss, f'val_{metric_name.lower().replace(" ", "_")}': val_metric})

            if val_metric >= best_val_metric:  # Use >= to save last best model
                best_val_metric = val_metric
                final_trained_model_state = model.state_dict()  # Save best model state

                current_test_data = test_data_or_loader
                if isinstance(current_test_data, Data): current_test_data = current_test_data.to(self.device)

                if isinstance(current_test_data, PyGDataLoader):
                    all_preds_test, all_labels_test = [], []
                    for batch_data_test in current_test_data:
                        preds_test = model(batch_data_test)
                        all_preds_test.append(preds_test)
                        all_labels_test.append(batch_data_test.y)
                    test_preds_cat = torch.cat(all_preds_test)
                    test_labels_cat = torch.cat(all_labels_test)
                    best_test_metric_at_best_val = metric_fn(test_labels_cat, test_preds_cat)
                elif isinstance(current_test_data, Data):
                    out_test = model(current_test_data)
                    if current_test_data.test_mask.sum() == 0:
                        best_test_metric_at_best_val = 0.0
                    else:
                        best_test_metric_at_best_val = metric_fn(current_test_data.y[current_test_data.test_mask], out_test[current_test_data.test_mask])

        print(f"  Finished training for {model_name_str} on {dataset_name}.")
        print(f"  Best Val {metric_name}: {best_val_metric:.4f}, Corresponding Test {metric_name}: {best_test_metric_at_best_val:.4f}")

        # Load the best model state for embedding extraction
        if final_trained_model_state:
            model.load_state_dict(final_trained_model_state)

        return best_val_metric, best_test_metric_at_best_val, pd.DataFrame(history), metric_name

    def run_on_dataset_variant(self, dataset_name: str, model_zoo_config: Dict,
                               train_data, val_data, test_data, task_type: str,
                               num_features: int, num_classes: int, graph_variant_suffix: str = ""):
        """Helper to run all models on a single dataset variant (e.g., original or undirected)."""
        dataset_results = []
        DataUtils.print_header(f"Benchmarking on Dataset: {dataset_name}{graph_variant_suffix}")

        for model_name, m_config in model_zoo_config.items():
            if dataset_name.lower() != 'ppi' and model_name == "ProtGramDirectGCN":
                print(f"Skipping ProtGramDirectGCN for non-PPI dataset {dataset_name}{graph_variant_suffix}.")
                continue

            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {dataset_name}{graph_variant_suffix} ---")
            try:
                model_instance = m_config["class"](in_channels=num_features, out_channels=num_classes, **m_config["params"]).to(self.device)
                if self.config.DEBUG_VERBOSE: print(f"  Model Architecture:\n{model_instance}")
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=self.config.EVAL_LEARNING_RATE, weight_decay=5e-4)

                val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                    model_name, model_instance, f"{dataset_name}{graph_variant_suffix}",
                    train_data, val_data, test_data,
                    optimizer, num_epochs=self.config.EVAL_EPOCHS, task_type=task_type,
                    num_classes=num_classes
                )
                dataset_results.append({
                    'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name,
                    f'best_val_{metric_name_used.lower().replace(" ", "_")}': val_metric,
                    f'test_{metric_name_used.lower().replace(" ", "_")}': test_metric
                })

                # Save embeddings using the model instance that has the best validation state loaded
                data_for_emb_extraction = test_data  # Or train_data if you want embeddings for all nodes
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
            # --- Original Graph Variant ---
            train_data_orig, val_data_orig, test_data_orig, task_type_orig = self._get_dataset(dataset_name, base_dataset_path, make_undirected=False)
            if train_data_orig is None:
                print(f"Skipping dataset {dataset_name} (original) due to loading error or missing masks.")
                continue

            num_features, num_classes = 0, 0
            if task_type_orig == 'multi-label-graph':
                num_features = train_data_orig.num_features
                num_classes = train_data_orig.num_classes
                train_loader_orig = PyGDataLoader(train_data_orig, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=True)
                val_loader_orig = PyGDataLoader(val_data_orig, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=False)
                test_loader_orig = PyGDataLoader(test_data_orig, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=False)
            elif task_type_orig == 'single-label-node':
                num_features = train_data_orig.num_features
                num_classes = train_data_orig.y.max().item() + 1 if train_data_orig.y is not None and train_data_orig.y.numel() > 0 else 1
                if hasattr(train_data_orig, 'num_classes') and train_data_orig.num_classes is not None:
                    num_classes = train_data_orig.num_classes
                train_loader_orig, val_loader_orig, test_loader_orig = train_data_orig, val_data_orig, test_data_orig
            else:
                continue  # Error already printed by _get_dataset

            model_zoo_cfg = {  # Define once per dataset based on its features/classes
                "GCN": {"class": GCN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GAT": {"class": GAT, "params": {"hidden_channels": 32, "heads": 8, "num_layers": 2, "dropout_rate": 0.6}},
                "GraphSAGE": {"class": GraphSAGE, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GIN": {"class": GIN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "ChebNet": {"class": ChebNet, "params": {"hidden_channels": 256, "K": 3, "num_layers": 2, "dropout_rate": 0.5}},
                "SignedNet": {"class": SignedNet, "params": {"hidden_channels": 128, "num_layers": 2, "dropout_rate": 0.5}},
                "RGCN_SR": {"class": RGCN, "params": {"hidden_channels": 256, "num_relations": 1, "num_layers": 2, "dropout_rate": 0.5}},
                "TongDiGCN": {"class": TongDiGCN, "params": {"hidden_channels": 128, "num_layers": 2, "dropout_rate": 0.5}},
            }
            if dataset_name.lower() == 'ppi':  # ProtGramDirectGCN might be suitable for PPI
                model_zoo_cfg["ProtGramDirectGCN"] = {"class": ProtGramDirectGCN, "params": {
                    "layer_dims": [num_features, self.config.GCN_HIDDEN_DIM_1, self.config.GCN_HIDDEN_DIM_2],
                    "num_graph_nodes": None,  # Not fixed for PPI
                    "task_num_output_classes": num_classes, "n_gram_len": 3, "one_gram_dim": 0, "max_pe_len": 512,
                    "dropout": self.config.GCN_DROPOUT_RATE, "use_vector_coeffs": True
                }}

            results_orig = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                       train_loader_orig, val_loader_orig, test_loader_orig,
                                                       task_type_orig, num_features, num_classes, "_Original")
            all_results_summary.extend(results_orig)

            # --- Undirected Graph Variant (Optional) ---
            if self.config.BENCHMARK_TEST_ON_UNDIRECTED and task_type_orig == 'single-label-node':  # PPI already multi-graph, to_undirected is per graph
                # For single-graph datasets, _get_dataset can handle making it undirected
                train_data_undir, val_data_undir, test_data_undir, task_type_undir = self._get_dataset(dataset_name, base_dataset_path, make_undirected=True)
                if train_data_undir is not None:
                    # num_features and num_classes should be the same
                    results_undir = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                                train_data_undir, val_data_undir, test_data_undir,
                                                                task_type_undir, num_features, num_classes, "_Undirected")
                    all_results_summary.extend(results_undir)
                else:
                    print(f"Skipping undirected variant for {dataset_name} due to loading error.")
            elif self.config.BENCHMARK_TEST_ON_UNDIRECTED and task_type_orig == 'multi-label-graph':
                print(f"Note: For {dataset_name} (multi-graph), undirected conversion is applied per graph via transform in _get_dataset if BENCHMARK_TEST_ON_UNDIRECTED is True.")
                # The original call to _get_dataset with make_undirected=True would handle this.
                # To avoid redundant loading, we can just run with the already loaded (and transformed) PPI.
                # Or, if we want a separate run explicitly labeled "_Undirected" for PPI:
                train_ppi_u, val_ppi_u, test_ppi_u, task_ppi_u = self._get_dataset(dataset_name, base_dataset_path, make_undirected=True)
                if train_ppi_u:
                    train_loader_ppi_u = PyGDataLoader(train_ppi_u, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=True)
                    val_loader_ppi_u = PyGDataLoader(val_ppi_u, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=False)
                    test_loader_ppi_u = PyGDataLoader(test_ppi_u, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=False)
                    results_ppi_u = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                                train_loader_ppi_u, val_loader_ppi_u, test_loader_ppi_u,
                                                                task_ppi_u, num_features, num_classes, "_Undirected")
                    all_results_summary.extend(results_ppi_u)

            # Save summary for the current dataset (both variants if run)
            current_dataset_summary = [res for res in all_results_summary if res['dataset'].startswith(dataset_name)]
            if current_dataset_summary:
                dataset_summary_df = pd.DataFrame(current_dataset_summary)
                dataset_summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), f"benchmark_summary_{dataset_name}.csv")
                DataUtils.save_dataframe_to_csv(dataset_summary_df, dataset_summary_path)
                print(f"\nCombined Summary for dataset {dataset_name} (all variants) saved to {dataset_summary_path}")
                print(dataset_summary_df)

        if all_results_summary:
            final_summary_df = pd.DataFrame(all_results_summary)
            final_summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), "gnn_benchmark_FULL_SUMMARY.csv")
            DataUtils.save_dataframe_to_csv(final_summary_df, final_summary_path)
            print(f"\nFull GNN benchmarking summary saved to {final_summary_path}")
            print("\nFull Summary Table:")
            print(final_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")
