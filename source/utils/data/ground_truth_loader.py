from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import dask.dataframe as dd
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
        Memory-efficiently reads interaction files to get the set of all unique protein IDs
        using Dask for scalability.
        """
        from dask.diagnostics import ProgressBar
        print("Gathering all required protein IDs from interaction files using Dask...")
        all_ids_series = []
        for filepath in file_paths:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"Warning: File not found during ID gathering: {filepath}")
                continue
            try:
                # --- REFACTOR: Use Dask for scalable and robust file reading ---
                if filepath.is_dir():  # Assume Parquet
                    ddf = dd.read_parquet(str(filepath), columns=['protein1', 'protein2'])
                else:  # Assume CSV/TSV
                    sep = '\t' if '.mitab' in filepath.name or '.tsv' in filepath.name else ','
                    ddf = dd.read_csv(str(filepath), sep=sep, header=None, usecols=[0, 1],
                                     names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn')

                ddf = ddf.dropna().astype(str)
                all_ids_series.append(ddf['protein1'])
                all_ids_series.append(ddf['protein2'])
            except Exception as e:
                print(f"Error reading file {filepath} during ID gathering: {e}")

        if not all_ids_series:
            print("No valid interaction files found to gather IDs.")
            return set()

        combined_ids = dd.concat(all_ids_series)
        with ProgressBar():
            unique_ids = combined_ids.unique().compute()

        required_ids = set(unique_ids)
        print(f"Found {len(required_ids):,} unique protein IDs across all interaction files.")
        return required_ids

    @staticmethod
    def _stream_filter_pairs(filepath: Path, label: int, available_ids: Set[str]) -> Iterator[Tuple[str, str, int]]:
        """The core streaming logic, refactored to be a true generator."""
        # --- DEFINITIVE FIX for IsADirectoryError ---
        # The previous implementation used `open()`, which cannot read a Parquet directory.
        # This now uses Dask to correctly stream from either CSV or Parquet formats.
        try:
            # --- REFACTOR: Use is_dir() for more robust Parquet detection ---
            if filepath.is_dir():
                ddf = dd.read_parquet(str(filepath), columns=['protein1', 'protein2'])
                ddf = ddf.rename(columns={'protein1': 'p1', 'protein2': 'p2'})
            else:
                sep = '\t' if '.mitab' in filepath.name or '.tsv' in filepath.name else ','
                ddf = dd.read_csv(str(filepath), sep=sep, header=None, names=['p1', 'p2'], usecols=[0, 1], dtype=str, on_bad_lines='warn')
            ddf = ddf.dropna().astype(str)
            filtered_ddf = ddf[ddf['p1'].isin(available_ids) & ddf['p2'].isin(available_ids)]

            for partition in tqdm(filtered_ddf.to_delayed(), desc=f"    Scanning {filepath.name}", leave=False):
                for row in partition.itertuples(index=False):
                    yield (row.p1, row.p2, label)
        except Exception as e:
            print(f"    ERROR: Could not load or filter interaction file '{filepath.name}'.")
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
                # --- DEFINITIVE FIX: Handle both CSV and Parquet files correctly ---
                # --- REFACTOR: Simplify logic to align with the robust Dask path ---
                # The previous check was brittle if a directory was not a parquet file.
                if filepath.is_dir():
                    # It's a directory, assume it's Parquet. Let pd.read_parquet handle errors.
                    df = pd.read_parquet(filepath, columns=['protein1', 'protein2'])
                    df = df.rename(columns={'protein1': 'p1', 'protein2': 'p2'}) # Align column names
                else:
                    # It's a file, assume it's CSV/TSV.
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        sep = '\t' if '\t' in f.readline() else ','
                    df = pd.read_csv(filepath, header=None, usecols=[0, 1], names=['p1', 'p2'], sep=sep, on_bad_lines='warn', dtype=str)
                df.dropna(inplace=True)
                # Vectorized filtering is much faster than iterating
                mask = df['p1'].isin(available_ids) & df['p2'].isin(available_ids)
                filtered_df = df[mask]
                # --- DEFINITIVE FIX: Add the missing sampling logic to the high-memory path ---
                if sample_n is not None and 0 < sample_n < len(filtered_df):
                    print(f"    Applying random sampling to keep up to {sample_n} pairs...")
                    filtered_df = filtered_df.sample(n=sample_n, random_state=random_state)
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