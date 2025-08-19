# ==============================================================================
# MODULE: utils/model_converter.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, and edge feature creation.
# VERSION: 7.0 (Aligned DirectGCN embedding extraction with Parallel Views architecture)
# AUTHOR: Islam Ebeid
# ==============================================================================

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set, Union, TYPE_CHECKING, Iterator, Callable

import psutil
import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from configuration.config import Config


class EmbeddingLoader:
    """
    A lazy loader for HDF5 embeddings that acts like a dictionary.
    It keeps the H5 file open and retrieves embeddings on-the-fly, which is
    highly memory-efficient. It should be used as a context manager.
    """

    def __init__(self, h5_path: Union[str, Path], config: Optional['Config'] = None):
        self.h5_path = Path(h5_path)
        self.config = config
        self._h5_file: Optional[h5py.File] = None
        self._keys: Optional[Set[str]] = None
        self._in_memory_data: Optional[Dict[str, np.ndarray]] = None

    def __enter__(self) -> 'EmbeddingLoader':
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.h5_path}")

        # --- NEW: More sophisticated dynamic loading strategy ---
        strategy = self.config.MEMORY_USAGE_STRATEGY if self.config else 'low'
        should_load_to_memory = False

        if strategy == 'high':
            should_load_to_memory = True
            print(f"  Memory Strategy ('high'): Attempting to load '{self.h5_path.name}' into RAM.")
        elif strategy == 'dynamic':
            file_size_gb = self.h5_path.stat().st_size / (1024 ** 3)
            available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
            # Heuristic: Load if file is less than 25% of available RAM to leave room for other processes.
            if file_size_gb < (available_mem_gb * 0.25):
                should_load_to_memory = True
                print(f"  Memory Strategy ('dynamic'): Loading '{self.h5_path.name}' ({file_size_gb:.2f} GB) into RAM (Available: {available_mem_gb:.2f} GB).")
            else:
                print(f"  Memory Strategy ('dynamic'): File '{self.h5_path.name}' ({file_size_gb:.2f} GB) is too large for available RAM ({available_mem_gb:.2f} GB). Using lazy loading.")

        if should_load_to_memory:
            try:
                with h5py.File(self.h5_path, 'r') as hf:
                    self._in_memory_data = {key: hf[key][:].astype(np.float16) for key in hf.keys()}
                self._keys = set(self._in_memory_data.keys())
                self._h5_file = None  # Ensure we don't use the file handle
            except (MemoryError, OSError) as e:
                print(f"  WARNING: Failed to load '{self.h5_path.name}' into memory (Error: {e}). Falling back to lazy loading.")
                self._in_memory_data = None
                self._h5_file = h5py.File(self.h5_path, 'r')
                self._keys = set(self._h5_file.keys())
        else:
            # Default lazy loading
            if strategy == 'low':
                print(f"  Memory Strategy ('low'): Using lazy loading for '{self.h5_path.name}'.")
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._keys = set(self._h5_file.keys())
            self._in_memory_data = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5_file:
            self._h5_file.close()
            print(f"Closed H5 file: {self.h5_path.name}")
        # --- FIX: Clear all internal state on exit to release memory ---
        self._h5_file = None
        self._keys = None
        self._in_memory_data = None

    def __contains__(self, key: str) -> bool:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return key in self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")

        # --- FIX: Handle both in-memory and lazy-loading strategies ---
        if self._in_memory_data is not None:
            # High-memory mode: get from dict
            if key in self._in_memory_data:
                return self._in_memory_data[key]
        elif self._h5_file is not None:
            # Low-memory (lazy) mode: get from file
            return self._h5_file[key][:].astype(np.float16)
        raise KeyError(f"Key '{key}' not found in {self.h5_path}")

    def __len__(self) -> int:
        return len(self._keys) if self._keys is not None else 0

    def get_keys(self) -> Set[str]:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return set(self._keys)
