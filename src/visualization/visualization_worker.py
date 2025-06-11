# ==============================================================================
# MODULE: visualization_worker.py
# PURPOSE: The thread spinner for loading embedding files.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from typing import Optional, Union, List

# Import PyQt6 components for the worker
from PyQt5.QtCore import QObject, pyqtSignal

# --- Default Configuration (can be overridden by GUI in the future) ---
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_INIT_PCA = True
TSNE_LEARNING_RATE = 'auto'
SAMPLE_N_FOR_COMBINED_TSNE = 2000
MAX_PROTEINS_FOR_PER_RESIDUE_PLOT = 3


# --- End Configuration ---

def _generate_tsne_figure(embeddings_array: np.ndarray, labels_for_coloring: Optional[np.ndarray], title: str, worker_signal: pyqtSignal):
    """
    Generates a t-SNE plot and returns the Matplotlib figure object.
    This is a modified version of the original plot_tsne_generic function.
    """
    num_samples = embeddings_array.shape[0]
    if num_samples == 0:
        raise ValueError("Cannot perform t-SNE: no data points.")

    # Adjust perplexity based on sample size, a requirement for t-SNE
    effective_perplexity = float(min(TSNE_PERPLEXITY, max(1.0, num_samples - 1.0)))
    if num_samples <= effective_perplexity or num_samples <= 1:
        raise ValueError(f"n_samples ({num_samples}) is too small for perplexity {effective_perplexity:.1f}.")

    worker_signal.emit(f"Adjusted perplexity to {effective_perplexity:.1f} for {num_samples} samples.")

    tsne_init_method = 'pca' if TSNE_INIT_PCA and embeddings_array.shape[1] > 2 else 'random'

    tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=effective_perplexity, max_iter=TSNE_N_ITER, init=tsne_init_method, learning_rate=TSNE_LEARNING_RATE, n_jobs=-1)

    worker_signal.emit(f"Fitting t-SNE ({tsne_init_method} init)...")
    tsne_results = tsne.fit_transform(embeddings_array)

    df_tsne = pd.DataFrame({'tsne_1': tsne_results[:, 0], 'tsne_2': tsne_results[:, 1]})
    hue_column = None
    if labels_for_coloring is not None and len(labels_for_coloring) == num_samples:
        df_tsne['label'] = labels_for_coloring
        hue_column = 'label'

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    unique_labels = np.unique(labels_for_coloring) if labels_for_coloring is not None else []

    palette = None
    if len(unique_labels) > 0:
        if 1 < len(unique_labels) <= 20:
            palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        else:
            palette = sns.color_palette("viridis", n_colors=len(unique_labels))

    sns.scatterplot(x="tsne_1", y="tsne_2", hue=hue_column, data=df_tsne, legend="full" if hue_column and len(unique_labels) <= 25 else False, palette=palette, s=50, alpha=0.7, ax=ax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)

    if hue_column and len(unique_labels) <= 25:
        ax.legend(title=str(hue_column).capitalize(), bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


class TSNEWorker(QObject):
    """
    Worker object to handle t-SNE processing in a separate thread.
    """
    # Signals to communicate with the main GUI thread
    finished = pyqtSignal(object)  # Emits the Matplotlib figure object when done
    progress = pyqtSignal(str)  # Emits status update strings
    error = pyqtSignal(str)  # Emits an error message string

    def __init__(self):
        super().__init__()

    def start_tsne_processing(self, h5_path: str, embedding_type: str):
        """
        Main entry point for the worker. Reads H5 and generates plots.
        """
        try:
            self.progress.emit(f"Loading embeddings from: {os.path.basename(h5_path)}")
            base_filename = os.path.splitext(os.path.basename(h5_path))[0]

            with h5py.File(h5_path, 'r') as hf:
                all_keys = list(hf.keys())
                if not all_keys:
                    raise FileNotFoundError("No datasets found in the HDF5 file.")
                self.progress.emit(f"Found {len(all_keys)} items. Processing as '{embedding_type}'.")

                if embedding_type == 'per_residue':
                    self._process_per_residue(hf, all_keys, base_filename)
                elif embedding_type in ['per_protein', 'generic_multiple_sets']:
                    self._process_combined(hf, all_keys, base_filename, embedding_type)
                else:
                    raise ValueError(f"Unknown embedding type specified: {embedding_type}")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")

    def _process_per_residue(self, hf, keys, base_filename):
        protein_ids = keys[:min(len(keys), MAX_PROTEINS_FOR_PER_RESIDUE_PLOT)]
        for prot_id in protein_ids:
            self.progress.emit(f"Processing protein: {prot_id}")
            embeddings = hf[prot_id][:]
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                self.progress.emit(f"Skipping {prot_id}: Invalid shape {embeddings.shape}")
                continue

            title = f"t-SNE of Per-Residue Embeddings for {prot_id}\n(Source: {base_filename})"
            fig = _generate_tsne_figure(embeddings, None, title, self.progress)
            self.finished.emit(fig)  # Emit a figure for each protein

    def _process_combined(self, hf, keys, base_filename, embedding_type):
        embeddings_list = []
        labels_list = []

        for key_id in keys:
            dataset_obj = hf[key_id]
            if isinstance(dataset_obj, h5py.Dataset):
                data = dataset_obj[:]
                if embedding_type == 'per_protein' and data.ndim == 1:
                    embeddings_list.append(data)
                elif embedding_type == 'generic_multiple_sets':
                    # Handles both 1D arrays (reshaped) and 2D arrays of embeddings
                    if data.ndim == 1:
                        embeddings_list.append(data.reshape(1, -1))
                        labels_list.append(key_id)
                    elif data.ndim == 2:
                        embeddings_list.append(data)
                        labels_list.extend([key_id] * data.shape[0])

        if not embeddings_list:
            raise ValueError("No valid embeddings found for combined plot.")

        self.progress.emit("Combining datasets...")
        embeddings_array = np.vstack(embeddings_list)
        labels_array = np.array(labels_list) if labels_list else None

        if embeddings_array.shape[0] > SAMPLE_N_FOR_COMBINED_TSNE:
            self.progress.emit(f"Sampling {SAMPLE_N_FOR_COMBINED_TSNE} points from {embeddings_array.shape[0]}...")
            indices = np.random.choice(embeddings_array.shape[0], SAMPLE_N_FOR_COMBINED_TSNE, replace=False)
            embeddings_array = embeddings_array[indices]
            if labels_array is not None:
                labels_array = labels_array[indices]

        title_prefix = "Per-Protein Embeddings" if embedding_type == 'per_protein' else "Generic Embedding Sets"
        title = f"t-SNE of {title_prefix}\n(Source: {base_filename})"
        fig = _generate_tsne_figure(embeddings_array, labels_array, title, self.progress)
        self.finished.emit(fig)
