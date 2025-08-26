from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple, Union
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from configuration.config import Config
# --- NEW: Import DataUtils for reservoir sampling ---
from source.utils.data.data_utils import DataUtils

# ==============================================================================
# 3. Interaction Data Loading
# ==============================================================================
class GroundTruthLoader:
    """
    Handles loading and processing of protein interaction data from files.
    """

    @staticmethod
    def get_required_ids_from_files(file_paths: List[Union[str, Path]]) -> Set[str]:
        """
        Memory-efficiently reads interaction files to get the set of all unique protein IDs.
        Reads files line-by-line to avoid loading everything into memory.
        """
        print("Gathering all required protein IDs from interaction files...")
        required_ids: Set[str] = set()
        for filepath in file_paths:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"Warning: File not found during ID gathering: {filepath}")
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in tqdm(f, desc=f"Scanning {filepath.name} for IDs", leave=False):
                        parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                        if len(parts) < 2:
                            parts = [p.strip() for p in line.strip().replace('"', '').split('\t')]
                        if len(parts) >= 2:
                            p1, p2 = parts[0], parts[1]
                            if p1: required_ids.add(p1)
                            if p2: required_ids.add(p2)
            except Exception as e:
                print(f"Error reading file {filepath} during ID gathering: {e}")
        print(f"Found {len(required_ids)} unique protein IDs across all interaction files.")
        return required_ids

    @staticmethod
    def load_interaction_pairs(filepath: Union[str, Path], label: int, sample_n: Optional[int] = None,
                               random_state: Optional[int] = None) -> List[Tuple[str, str, int]]:
        """
        Loads interaction pairs from a CSV/TSV file. Includes option for sampling.
        """
        filepath = Path(filepath)
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Loading pairs from: {filepath.name} (label: {label}){sampling_info}...")
        if not filepath.exists():
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            # Sniff the delimiter for robustness instead of nested try-except
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                sep = '\t' if '\t' in first_line else ','

            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn',
                             sep=sep)
            df.dropna(subset=['protein1', 'protein2'], inplace=True)
            df['protein1'] = df['protein1'].astype(str).str.strip()
            df['protein2'] = df['protein2'].astype(str).str.strip()
            df = df[(df['protein1'] != "") & (df['protein2'] != "")]

            if sample_n is not None and 0 < sample_n < len(df):
                df = df.sample(n=sample_n, random_state=random_state)

            pairs = [(row.protein1, row.protein2, label) for _, row in df.iterrows()]
            print(f"Successfully loaded {len(pairs)} pairs.")
            return pairs
        except Exception as e:
            print(f"  ERROR: Could not load or parse interaction file '{filepath.name}'.")
            print(f"  Please check the file format and integrity.")
            print(f"  Details: {e}")
            return []

    @staticmethod
    def stream_interaction_pairs(filepath: Union[str, Path], label: int, batch_size: int, sample_n: Optional[int] = None,
                                 random_state: Optional[int] = None) -> Iterator[List[Tuple[str, str, int]]]:
        """
        Reads interaction pairs from a CSV/TSV file line by line and yields them in batches.
        """
        filepath = Path(filepath)
        streaming_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Streaming pairs from: {filepath.name} (label: {label}, batch_size: {batch_size}){streaming_info}...")
        if not filepath.exists():
            print(f"Warning: Interaction file not found: {filepath}")
            return

        lines_to_read_indices: Optional[Set[int]] = None
        if sample_n is not None:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines = sum(1 for _ in f)
                if 0 < sample_n < total_lines:
                    rng = np.random.default_rng(random_state)
                    lines_to_read_indices = set(rng.choice(total_lines, sample_n, replace=False))
            except Exception as e:
                print(f"Error during pre-sampling count for {filepath}: {e}. Proceeding without sampling if possible.")

        batch: List[Tuple[str, str, int]] = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if lines_to_read_indices is not None and i not in lines_to_read_indices:
                        continue
                    parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                    if len(parts) < 2:
                        parts = [p.strip() for p in line.strip().replace('"', '').split('\t')]
                    if len(parts) >= 2:
                        p1, p2 = parts[0], parts[1]
                        if p1 and p2:
                            batch.append((p1, p2, label))
                            if len(batch) == batch_size:
                                yield batch
                                batch = []
        except Exception as e:
            print(f"Error streaming interaction file {filepath}: {e}")

        if batch:
            yield batch

    @staticmethod
    def _stream_filter_pairs(filepath: Path, label: int, available_ids: Set[str]) -> Iterator[Tuple[str, str, int]]:
        """The core streaming logic, refactored to be a true generator."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc=f"    Scanning {filepath.name}", leave=False):
                    parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                    if len(parts) < 2:
                        parts = [p.strip() for p in line.strip().replace('"', '').split('\t')]
                    if len(parts) >= 2:
                        p1, p2 = parts[0], parts[1]
                        if p1 in available_ids and p2 in available_ids:
                            yield (p1, p2, label) # Use yield to make it a generator
        except Exception as e:
            print(f"    ERROR: Could not load or filter interaction file '{filepath.name}'.")
            print(f"    Please check the file format and integrity.")
            print(f"    Details: {e}")
            return # Stop the generator on error

    @staticmethod
    def load_interaction_pairs_filtered(filepath: Union[str, Path], label: int, available_ids: Set[str],
                                        sample_n: Optional[int] = None,
                                        random_state: Optional[int] = None,
                                        config: Optional[Config] = None) -> List[Tuple[str, str, int]]:
        """
        Memory-efficiently loads interaction pairs where both proteins exist in the
        provided set of available IDs. Switches to a high-performance, in-memory
        method if the config allows.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"    Warning: Interaction file not found: {filepath}")
            return []

        # --- NEW: Use high-performance Pandas loading in high-memory mode ---
        if config and config.MEMORY_USAGE_STRATEGY == 'high':
            print(f"  High-performance filtering pairs from: {filepath.name} (label: {label})...")
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    sep = '\t' if '\t' in f.readline() else ','
                df = pd.read_csv(filepath, header=None, usecols=[0, 1], names=['p1', 'p2'], sep=sep, on_bad_lines='warn', dtype=str)
                df.dropna(inplace=True)
                # Vectorized filtering is much faster than iterating
                mask = df['p1'].isin(available_ids) & df['p2'].isin(available_ids)
                filtered_df = df[mask]
                filtered_pairs = [(row.p1, row.p2, label) for row in filtered_df.itertuples(index=False)]
            except Exception as e:
                print(f"    ERROR: Could not load or filter interaction file '{filepath.name}' with Pandas.")
                print(f"    Details: {e}")
                return []
        else:
            # Fallback to memory-efficient streaming for low-memory mode
            print(f"  Streaming-filtering pairs from: {filepath.name} (label: {label})...")
            # --- DEFINITIVE FIX: Consume the generator correctly ---
            # This makes the low-memory path truly memory-efficient.
            pair_iterator = GroundTruthLoader._stream_filter_pairs(filepath, label, available_ids)

            if sample_n is not None and sample_n > 0:
                print(f"    Applying reservoir sampling to keep up to {sample_n} pairs...")
                # Use reservoir sampling on the iterator to avoid loading all pairs into memory
                filtered_pairs = DataUtils.reservoir_sample(pair_iterator, sample_n, random_state)
            else:
                # If not sampling, consume the entire iterator into a list
                filtered_pairs = list(pair_iterator)

        num_found = len(filtered_pairs)
        print(f"    Found {num_found} pairs with available embeddings.")

        return filtered_pairs