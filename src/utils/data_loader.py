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
    (Adapted from evaluater.py)
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


def load_h5_embeddings_selectively(h5_path: str, required_ids: Set[str]) -> Dict[str, np.ndarray]:
    """
    Loads embeddings from an HDF5 file, but only for the IDs in the required_ids set.
    (Adapted from evaluater.py)
    """
    h5_path = os.path.normpath(h5_path)
    print(f"Loading embeddings from: {os.path.basename(h5_path)}...")
    if not os.path.exists(h5_path):
        print(f"Warning: Embedding file not found: {h5_path}")
        return {}
    protein_embeddings: Dict[str, np.ndarray] = {}
    try:
        with h5py.File(h5_path, 'r') as hf:
            keys_to_load = required_ids.intersection(hf.keys())
            for key in tqdm(keys_to_load, desc=f"Reading {os.path.basename(h5_path)}", leave=False, unit="protein"):
                protein_embeddings[key] = hf[key][:].astype(np.float32)
        print(f"Loaded {len(protein_embeddings)} relevant embeddings.")
    except Exception as e:
        print(f"Error processing HDF5 file {h5_path}: {e}")
    return protein_embeddings


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
