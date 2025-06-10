# ==============================================================================
# MODULE: utils/data_loader.py
# PURPOSE: Contains all data loading utilities for the PPI pipeline.
# ==============================================================================

import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Set, Tuple, Iterator


class H5EmbeddingLoader:
    """
    A lazy loader for HDF5 embeddings that acts like a dictionary.
    It keeps the H5 file open and retrieves embeddings on-the-fly as needed,
    which is highly memory-efficient. It should be used as a context manager.
    """

    def __init__(self, h5_path: str):
        self.h5_path = os.path.normpath(h5_path)
        self._h5_file = None
        self._keys = None

    def __enter__(self):
        """Open the HDF5 file and read the keys when entering the context."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Embedding file not found: {self.h5_path}")
        self._h5_file = h5py.File(self.h5_path, 'r')
        self._keys = set(self._h5_file.keys())
        print(f"Opened H5 file: {os.path.basename(self.h5_path)}, found {len(self._keys)} keys.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the HDF5 file is closed when exiting the context."""
        if self._h5_file:
            self._h5_file.close()
            print(f"Closed H5 file: {os.path.basename(self.h5_path)}")

    def __contains__(self, key: str) -> bool:
        """Check if a protein ID (key) exists in the HDF5 file."""
        return key in self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        """Retrieve a single embedding for a given protein ID (key)."""
        if key in self._keys:
            return self._h5_file[key][:].astype(np.float32)
        raise KeyError(f"Key '{key}' not found in {self.h5_path}")

    def __len__(self) -> int:
        """Return the total number of embeddings in the file."""
        return len(self._keys)


def get_required_ids_from_files(file_paths: List[str]) -> Set[str]:
    """
    Memory-efficiently reads interaction files to get the set of all unique protein IDs.
    Reads files line-by-line to avoid loading everything into memory.
    """
    print("Gathering all required protein IDs from interaction files...")
    required_ids = set()
    for filepath in file_paths:
        if not os.path.exists(filepath):
            print(f"Warning: File not found during ID gathering: {filepath}")
            continue
        with open(filepath, 'r') as f:
            for line in tqdm(f, desc=f"Scanning {os.path.basename(filepath)} for IDs", leave=False):
                parts = line.strip().replace('"', '').split(',')
                if len(parts) < 2:
                    parts = line.strip().replace('"', '').split('\t')
                if len(parts) >= 2:
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    if p1: required_ids.add(p1)
                    if p2: required_ids.add(p2)
    print(f"Found {len(required_ids)} unique protein IDs across all interaction files.")
    return required_ids


def load_interaction_pairs(filepath: str, label: int, sample_n: Optional[int] = None, random_state: Optional[int] = None) -> List[Tuple[str, str, int]]:
    """
    Loads interaction pairs from a CSV file. Includes option for sampling.
    """
    filepath = os.path.normpath(filepath)
    sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
    print(f"Loading pairs from: {os.path.basename(filepath)} (label: {label}){sampling_info}...")
    if not os.path.exists(filepath):
        print(f"Warning: Interaction file not found: {filepath}")
        return []
    try:
        df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn')
        df.dropna(inplace=True)
        if sample_n is not None and 0 < sample_n < len(df):
            df = df.sample(n=sample_n, random_state=random_state)
        pairs = [(str(row.protein1).strip(), str(row.protein2).strip(), label) for _, row in df.iterrows()]
        print(f"Successfully loaded {len(pairs)} pairs.")
        return pairs
    except Exception as e:
        print(f"Error loading interaction file {filepath}: {e}")
        return []


def stream_interaction_pairs(filepath: str, label: int, batch_size: int, sample_n: Optional[int] = None, random_state: Optional[int] = None) -> Iterator[List[Tuple[str, str, int]]]:
    """
    Reads interaction pairs from a CSV file line by line and yields them in batches.
    This is a memory-efficient generator alternative to load_interaction_pairs.
    """
    filepath = os.path.normpath(filepath)
    streaming_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
    print(f"Streaming pairs from: {os.path.basename(filepath)} (label: {label}, batch_size: {batch_size}){streaming_info}...")
    if not os.path.exists(filepath):
        print(f"Warning: Interaction file not found: {filepath}")
        return

    # If sampling, we need to read the whole file to know the total number of lines
    if sample_n is not None:
        with open(filepath, 'r') as f:
            total_lines = sum(1 for line in f)

        if sample_n < total_lines:
            # Create a random number generator
            rng = np.random.default_rng(random_state)
            # Generate unique random line indices to skip
            lines_to_skip = rng.choice(total_lines, total_lines - sample_n, replace=False)
            lines_to_skip = set(lines_to_skip)
        else:
            lines_to_skip = set()  # sample_n is larger than or equal to file size, so we read all lines

    batch = []
    try:
        with open(filepath, 'r') as f:
            # The original code used pandas which handles headers automatically.
            # We will assume no header for this implementation based on the original code's `header=None`.
            for i, line in enumerate(f):
                if sample_n is not None and i in lines_to_skip:
                    continue

                parts = line.strip().replace('"', '').split(',')
                if len(parts) < 2:
                    parts = line.strip().replace('"', '').split('\t')

                if len(parts) >= 2:
                    p1, p2 = parts[0].strip(), parts[1].strip()
                    if p1 and p2:
                        batch.append((p1, p2, label))
                        if len(batch) == batch_size:
                            yield batch
                            batch = []
    except Exception as e:
        print(f"Error streaming interaction file {filepath}: {e}")

    # Yield the last batch if it's not empty
    if batch:
        yield batch


# def load_h5_embeddings_selectively(h5_path: str, required_ids: Set[str]) -> Dict[str, np.ndarray]:
#     """
#     Loads embeddings from an HDF5 file, but only for the IDs in the required_ids set.
#     (Adapted from evaluater.py)
#     """
#     h5_path = os.path.normpath(h5_path)
#     print(f"Loading embeddings from: {os.path.basename(h5_path)}...")
#     if not os.path.exists(h5_path):
#         print(f"Warning: Embedding file not found: {h5_path}")
#         return {}
#     protein_embeddings: Dict[str, np.ndarray] = {}
#     try:
#         with h5py.File(h5_path, 'r') as hf:
#             keys_to_load = required_ids.intersection(hf.keys())
#             for key in tqdm(keys_to_load, desc=f"Reading {os.path.basename(h5_path)}", leave=False, unit="protein"):
#                 protein_embeddings[key] = hf[key][:].astype(np.float32)
#         print(f"Loaded {len(protein_embeddings)} relevant embeddings.")
#     except Exception as e:
#         print(f"Error processing HDF5 file {h5_path}: {e}")
#     return protein_embeddings


def fast_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
    """
    An efficient FASTA parser that reads one sequence at a time, yielding an ID and sequence.
    (Adapted from generate_transformer_per_residue_embeddings.py)
    """
    fasta_filepath = os.path.normpath(fasta_filepath)
    protein_id = None
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if protein_id and sequence_parts:
                        yield protein_id, "".join(sequence_parts)

                    header = line[1:]
                    # A robust way to get the ID, preferring UniProt format but falling back
                    parts = header.split('|')
                    if len(parts) > 1 and parts[1]:
                        protein_id = parts[1]
                    else:
                        protein_id = header.split()[0]
                    sequence_parts = []
                else:
                    sequence_parts.append(line.upper())
            # Yield the last sequence in the file
            if protein_id and sequence_parts:
                yield protein_id, "".join(sequence_parts)
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_filepath}")
