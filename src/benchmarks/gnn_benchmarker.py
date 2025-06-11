# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 2.5 (Moved run_benchmarker into GNNBenchmarker class as run method)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os

import pandas as pd
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader as PyGDataLoader  # Aliased to avoid conflict

from src.config import Config
from src.models.gnn_zoo import *
# Assuming these are correctly located in your project structure
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataUtils


# ==============================================================================
# GNNBenchmarker Class
# ==============================================================================
class GNNBenchmarker:
    """
    Handles the training and evaluation of GNN models for benchmarking purposes.
    """

    def __init__(self, config: Config):
        """
        Initializes the GNNBenchmarker.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_and_evaluate(self, model, train_loader, val_loader, test_loader, optimizer, num_epochs=200):
        """Main training and evaluation loop for a given model."""
        best_f1 = 0
        history = []

        print(f"  Training on device: {self.device}")
        model.to(self.device)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.binary_cross_entropy_with_logits(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for data_batch in val_loader:
                    data_batch = data_batch.to(self.device)
                    preds = model(data_batch)
                    all_preds.append(preds)
                    all_labels.append(data_batch.y)
                val_f1 = f1_score(torch.cat(all_labels).cpu().numpy(), (torch.cat(all_preds) > 0).cpu().numpy(), average='micro')

            if val_f1 > best_f1:
                best_f1 = val_f1

            print(f'  Epoch {epoch:03d}, Loss: {total_loss / len(train_loader):.4f}, Val F1: {val_f1:.4f}')
            history.append({'epoch': epoch, 'loss': total_loss / len(train_loader), 'val_f1': val_f1})

        # Final Test Evaluation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for data_batch in test_loader:
                data_batch = data_batch.to(self.device)
                preds = model(data_batch)
                all_preds.append(preds)
                all_labels.append(data_batch.y)
            test_f1 = f1_score(torch.cat(all_labels).cpu().numpy(), (torch.cat(all_preds) > 0).cpu().numpy(), average='micro')

        return best_f1, test_f1, pd.DataFrame(history)

    def run(self):
        """Runs the GNN benchmarking pipeline."""
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        print(f"Using device: {self.device}")

        # Ensure BASE_DATA_DIR is a string for os.path.join
        base_input_path = str(self.config.BASE_DATA_DIR)
        path = os.path.join(base_input_path, 'PPI_dataset')

        # Create directory if it doesn't exist, as PPI dataset might download here
        os.makedirs(path, exist_ok=True)

        train_dataset = PPI(path, split='train')
        val_dataset = PPI(path, split='val')
        test_dataset = PPI(path, split='test')

        train_loader = PyGDataLoader(train_dataset, batch_size=self.config.EVAL_BATCH_SIZE, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = PyGDataLoader(test_dataset, batch_size=2, shuffle=False)

        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes

        model_names = ["GCN", "GAT", "GraphSAGE", "GIN", "ProtGramDirectGCN"]
        results = []

        for model_name in model_names:
            print(f"\n--- Benchmarking Model: {model_name} ---")

            if model_name == "ProtGramDirectGCN":
                model = ProtGramDirectGCN(num_initial_features=num_features, hidden_dim1=self.config.GCN_HIDDEN_DIM_1, hidden_dim2=self.config.GCN_HIDDEN_DIM_2, num_graph_nodes=None, task_num_output_classes=num_classes,
                                          n_gram_len=self.config.GCN_NGRAM_MAX_N, one_gram_dim=self.config.GCN_1GRAM_INIT_DIM, max_pe_len=self.config.GCN_MAX_PE_LEN, dropout=self.config.GCN_DROPOUT_RATE,
                                          use_vector_coeffs=getattr(self.config, 'GCN_USE_VECTOR_COEFFS', True))
            else:
                model_name_upper = model_name.upper()
                if model_name_upper == 'GCN':
                    return GCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
                elif model_name_upper == 'GAT':
                    return GAT(in_channels=num_features, hidden_channels=256, out_channels=num_classes, heads=4)
                elif model_name_upper == 'GRAPHSAGE':
                    return GraphSAGE(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
                elif model_name_upper == 'WLGCN':
                    return WLGCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
                elif model_name_upper == 'CHEBNET':
                    return ChebNet(in_channels=num_features, hidden_channels=256, out_channels=num_classes, K=3)
                elif model_name_upper == 'SIGNEDNET':
                    return SignedNet(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
                elif model_name_upper == 'RGCN_SR':  # SR for Single Relation
                    return RGCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes, num_relations=1)
                elif model_name_upper == 'GIN':
                    return GIN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
                else:
                    raise ValueError(f"Model '{model_name}' not supported in the local GNN Zoo.")

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.EVAL_LEARNING_RATE)

            val_f1, test_f1, history_df = self.train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, num_epochs=self.config.EVAL_EPOCHS)

            print(f"Final Results for {model_name}: Best Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")
            results.append({'model': model_name, 'best_val_f1': val_f1, 'test_f1': test_f1})

            reports_dir = str(self.config.BENCHMARKING_RESULTS_DIR)
            os.makedirs(reports_dir, exist_ok=True)

            history_path = os.path.join(reports_dir, f'benchmark_{model_name}_history.csv')
            history_df.to_csv(history_path, index=False)
            print(f"Saved {model_name} training history to {history_path}")

        results_df = pd.DataFrame(results)
        summary_file_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), "gnn_benchmark_summary.csv")
        DataUtils.save_dataframe_to_csv(results_df, summary_file_path)
        DataUtils.print_header("GNN Benchmarking FINISHED")
