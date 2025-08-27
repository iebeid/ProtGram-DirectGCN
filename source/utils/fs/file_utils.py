# ==============================================================================
# MODULE: utils/data/file_utils.py
# PURPOSE: Contains all file I/O and related utilities.
# VERSION: 1.0 (Created by refactoring from data_utils.py)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import hashlib
import json
import tempfile
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Union, Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from source.utils.data.data_utils import DataUtils
from tqdm.auto import tqdm
from source.utils.fs.file_system_manager import fs_manager


# ==============================================================================
# 1. General Data Utilities (Moved from embedding_loader.py)
# ==============================================================================
class FileUtils:
    """A collection of static methods for file operations."""

    @staticmethod
    def save_object(obj: any, uri: Union[str, Path]):
        """Saves a Python object to a file in any fsspec-supported location."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        try:
            parent_dir = os.path.dirname(path)
            if parent_dir: fs.makedirs(parent_dir, exist_ok=True)
            with fs.open(path, 'wb') as f:
                pickle.dump(obj, f)
            print(f"  Object saved to {uri}")
        except Exception as e:
            print(f"  ERROR: Error saving object to {uri}: {e}")

    @staticmethod
    def load_object(uri: Union[str, Path]) -> Optional[any]:
        """Loads a pickled Python object from any fsspec-supported location."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        if not fs.exists(path):
            return None
        try:
            with fs.open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
            print(f"  ERROR: File '{os.path.basename(path)}' is corrupted or cannot be unpickled.")
            print(f"  Details: {e.__class__.__name__}: {e}")
            return None
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred while loading object from {uri}: {e}")
            return None

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, uri: Union[str, Path], index: bool = False):
        """Saves a Pandas DataFrame to a CSV file in any fsspec-supported location."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        try:
            parent_dir = os.path.dirname(path)
            if parent_dir: fs.makedirs(parent_dir, exist_ok=True)
            with fs.open(path, 'w', encoding='utf-8') as f:
                df.to_csv(f, index=index)
            print(f"DataFrame saved to: {uri}")
        except Exception as e:
            print(f"  ERROR: Error saving DataFrame to {uri}: {e}")

    @staticmethod
    def save_json(data: Dict, uri: Union[str, Path], json_lines: bool = False):
        """Saves a dictionary to a JSON file, with an option for JSONL format."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        try:
            parent_dir = os.path.dirname(path)
            if parent_dir: fs.makedirs(parent_dir, exist_ok=True)
            with fs.open(path, 'w', encoding='utf-8') as f:
                if json_lines:
                    for key, value in data.items():
                        json.dump({key: value}, f)
                        f.write('\n')
                else:
                    # Write a single, pretty-printed JSON object.
                    json.dump(data, f, indent=4, sort_keys=True)
            print(f"  JSON data saved to {uri}")
        except Exception as e:
            print(f"  ERROR: Could not save JSON to {uri}: {e}")

    @staticmethod
    def write_h5(embeddings_dict: Dict, uri: Union[str, Path], desc: str):
        """Helper function to write a dictionary of embeddings to an HDF5 file in any fsspec-supported location."""
        fs, final_path = fs_manager.get_fs_and_path(str(uri))
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_f:
            temp_local_path = tmp_f.name

        try:
            # --- DEFINITIVE FIX for Slow HDF5 I/O ---
            # Writing millions of small datasets is extremely inefficient.
            # Instead, we write two large, contiguous datasets: one for IDs and one for vectors.
            # This is orders of magnitude faster.
            with h5py.File(temp_local_path, 'w') as hf, tqdm(total=2, desc=f"  {desc}") as pbar:
                protein_ids = list(embeddings_dict.keys())
                embedding_vectors = np.array(list(embeddings_dict.values()), dtype=np.float16)

                # Store IDs as a variable-length UTF-8 string dataset
                hf.create_dataset('ids', data=np.array(protein_ids, dtype=h5py.string_dtype('utf-8')))
                pbar.update(1)

                # Store embeddings as a single, large numerical dataset
                hf.create_dataset('embeddings', data=embedding_vectors, chunks=True, compression="gzip")
                pbar.update(1)

            # Move the completed local file to the final destination URI
            final_dir = os.path.dirname(final_path)
            if final_dir:
                fs.makedirs(final_dir, exist_ok=True)
            fs.put(temp_local_path, final_path)
        except Exception as e:
            print(f"  ERROR: Could not write HDF5 file to {uri}: {e}")
        finally:
            if os.path.exists(temp_local_path):
                os.remove(temp_local_path)

    @staticmethod
    def check_h5_embeddings_integrity(uri: Union[str, Path], num_samples_to_check: int = 5):
        """Performs a basic integrity check on an HDF5 embedding file from any fsspec-supported location."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        DataUtils.print_header(f"--- Checking HDF5 file: {uri} ---")

        if not fs.exists(path):
            print(f"  ERROR: File at '{uri}' does not exist.")
            return

        try:
            with fs.open(path, 'rb') as f:
                with h5py.File(f, 'r') as hf:
                    # --- DEFINITIVE FIX: Handle both new and old HDF5 formats ---
                    if 'ids' in hf and 'embeddings' in hf:
                        # New, efficient format
                        ids = [s.decode('utf-8') for s in hf['ids'][:]]
                        embeddings_dataset = hf['embeddings']
                        num_embeddings = embeddings_dataset.shape[0]
                        print(f"  HDF5 check (new format): Found {num_embeddings} embeddings. Inspecting up to {num_samples_to_check} samples:")
                        if num_embeddings == 0:
                            return

                        sample_indices = random.sample(range(num_embeddings), min(num_embeddings, num_samples_to_check))
                        for i, idx in enumerate(sample_indices):
                            key = ids[idx]
                            emb = embeddings_dataset[idx]
                            print(f"    - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")
                            if np.isnan(emb).any(): print("      - WARNING: Embedding contains NaN values.")
                            if np.isinf(emb).any(): print("      - WARNING: Embedding contains Inf values.")
                    else:
                        # Old format (one dataset per protein)
                        keys = list(hf.keys())
                        if not keys:
                            print("  HDF5 check (old format): File is empty.")
                            return
                        print(f"  HDF5 check (old format): Found {len(keys)} total embeddings. Inspecting up to {num_samples_to_check} samples:")
                        sample_keys = random.sample(keys, min(len(keys), num_samples_to_check))
                        for i, key in enumerate(sample_keys):
                            dataset = hf.get(key)
                            if not isinstance(dataset, h5py.Dataset): continue
                            emb = dataset[:]
                            print(f"    - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")
                            if np.isnan(emb).any(): print("      - WARNING: Embedding contains NaN values.")
                            if np.isinf(emb).any(): print("      - WARNING: Embedding contains Inf values.")
        except Exception as e:
            print(f"  ERROR: An error occurred while checking HDF5 file '{uri}': {e}")

    @staticmethod
    def calculate_sha256(uri: Union[str, Path]) -> Optional[str]:
        """Calculates the SHA256 checksum of a file from any fsspec-supported location."""
        fs, path = fs_manager.get_fs_and_path(str(uri))
        sha256_hash = hashlib.sha256()
        try:
            if not fs.exists(path):
                return None
            # --- DEFINITIVE FIX: Explicitly handle directories to prevent silent failures ---
            if fs.isdir(path):
                print(f"  Warning: Cannot calculate checksum for a directory: {uri}")
                return None

            with fs.open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            print(f"  Warning: Could not calculate checksum for {uri}: {e}")
            return None