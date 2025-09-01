from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple, Union
import re
import os

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
    def process_raw_files(raw_file_paths: List[Path], output_path: Path, id_map_path: Path, is_positive: bool):
        """
        Reads raw interaction files, maps all IDs to UniProtKB, and saves to a clean Parquet file.
        This is the definitive method for creating the ground truth data.
        """
        # --- DEFINITIVE FIX for Unnecessary Reprocessing ---
        # This check makes the data setup process idempotent. If the final, processed
        # parquet file already exists, we skip the entire expensive mapping operation.
        if output_path.exists():
            print(f"  INFO: Standardized ground truth file '{output_path.name}' already exists. Skipping processing.")
            return

        if not raw_file_paths:
            print(f"  No raw files provided for {'positive' if is_positive else 'negative'} interactions. Skipping.")
            return

        if not id_map_path.exists():
            print(f"  ERROR: ID Mapping Parquet file not found at '{id_map_path}'. Cannot process ground truth.")
            return

        print(f"  Processing {'positive' if is_positive else 'negative'} raw interaction files...")

        # 1. Read all raw files into a single Dask DataFrame
        all_dfs = []
        for file_path in raw_file_paths:
            if not file_path.exists():
                print(f"    Warning: Raw interaction file not found: {file_path}")
                continue
            # Assume MITAB format for BioGrid and Russell Lab data
            ddf = dd.read_csv(
                str(file_path), sep='\t', header=None, usecols=[0, 1],
                names=['raw_id1', 'raw_id2'], dtype=str, on_bad_lines='warn'
            )
            all_dfs.append(ddf)

        if not all_dfs:
            print("  No valid raw interaction files could be read.")
            return

        combined_ddf = dd.concat(all_dfs).dropna().drop_duplicates()

        # 2. Extract all unique raw IDs that need to be mapped
        ids1 = combined_ddf['raw_id1'].unique()
        ids2 = combined_ddf['raw_id2'].unique()
        all_raw_ids = dd.concat([ids1, ids2]).unique()

        # 3. Load the ID map and perform an efficient, scalable merge to filter it
        id_map_ddf = dd.read_parquet(id_map_path)

        # --- DEFINITIVE FIX for Performance: Replace slow 'isin' with a fast Dask 'merge' (join) ---
        # The previous implementation used `isin(dask_series)`, which is a known
        # performance anti-pattern in Dask. It requires collecting one of the series
        # into memory and broadcasting it, causing the process to hang on large data.
        # This new approach converts the IDs to a DataFrame and performs a highly
        # optimized merge operation, which is the standard and scalable way to
        # perform this kind of filtering.
        all_raw_ids_ddf = all_raw_ids.to_frame(name="db_id") # noqa
        # --- DEFINITIVE FIX for "Zero Pairs" Error: Ensure consistent dtypes before merge ---
        # The merge was failing silently because the 'db_id' column from the raw files
        # was 'object' dtype, while the one from the mapping parquet was 'string'.
        # Explicitly casting to string ensures the distributed join works correctly.
        all_raw_ids_ddf['db_id'] = all_raw_ids_ddf['db_id'].astype(str)
        filtered_map_ddf = dd.merge(id_map_ddf, all_raw_ids_ddf, on='db_id', how='inner').compute()

        id_to_uniprot_map = dict(zip(filtered_map_ddf['db_id'], filtered_map_ddf['uniprot_id']))

        # 4. Map the raw IDs to UniProtKB IDs using the in-memory map
        def map_ids(partition: pd.DataFrame) -> pd.DataFrame:
            partition['protein1'] = partition['raw_id1'].map(id_to_uniprot_map)
            partition['protein2'] = partition['raw_id2'].map(id_to_uniprot_map)
            return partition.dropna(subset=['protein1', 'protein2'])

        mapped_ddf = combined_ddf.map_partitions(map_ids, meta={'raw_id1': 'str', 'raw_id2': 'str', 'protein1': 'str', 'protein2': 'str'})

        # 5. Select final columns and save to Parquet
        final_ddf = mapped_ddf[['protein1', 'protein2']].drop_duplicates()

        print(f"  Saving {len(final_ddf)} standardized interaction pairs to {output_path}...")
        final_ddf.to_parquet(
            str(output_path),
            engine='pyarrow',
            overwrite=True,
            write_index=False
        )
        print(f"  ✅ Successfully created standardized ground truth file: {output_path.name}")

    @staticmethod
    def process_raw_files_with_regex(raw_file_paths: List[Path], output_path: Path, is_positive: bool):
        """
        A faster processing method that uses regex to extract UniProtKB IDs directly
        from the raw interaction files, bypassing the need for the large mapping file.
        """
        if output_path.exists():
            print(f"  INFO: Standardized ground truth file '{output_path.name}' already exists. Skipping processing.")
            return

        if not raw_file_paths:
            print(f"  No raw files provided for {'positive' if is_positive else 'negative'} interactions. Skipping.")
            return

        print(f"  Processing {'positive' if is_positive else 'negative'} raw interaction files with REGEX...")

        # This regex is designed to capture UniProt IDs from various common formats.
        uniprot_regex = re.compile(r"(?:uniprotkb|uniprot/swiss-prot):([A-Z0-9]{6,10}(?:-\d+)?)")

        def _extract_uniprot_from_string(s: str) -> Optional[str]:
            """Helper to find the first UniProt ID in a given string."""
            match = uniprot_regex.search(s)
            return match.group(1) if match else None

        def map_ids_with_regex(partition: pd.DataFrame) -> pd.DataFrame:
            """A Dask partition function to apply the regex extraction."""
            partition['protein1'] = partition['raw_id1'].apply(_extract_uniprot_from_string)
            partition['protein2'] = partition['raw_id2'].apply(_extract_uniprot_from_string)
            return partition.dropna(subset=['protein1', 'protein2'])

        all_dfs = []
        for file_path in raw_file_paths:
            if not file_path.exists():
                print(f"    Warning: Raw interaction file not found: {file_path}")
                continue
            # For regex processing, we need more columns from BioGrid but not from Russell Lab
            usecols = [0, 1, 2, 3] if "BIOGRID" in file_path.name else [0, 1]
            names = ['raw_id1', 'raw_id2', 'full_id1', 'full_id2'] if "BIOGRID" in file_path.name else ['raw_id1', 'raw_id2']

            ddf = dd.read_csv(
                str(file_path), sep='\t', header=None, usecols=usecols,
                names=names, dtype=str, on_bad_lines='warn'
            )
            # For BioGrid, the UniProt IDs are in the full columns
            if "BIOGRID" in file_path.name:
                ddf['raw_id1'] = ddf['full_id1']
                ddf['raw_id2'] = ddf['full_id2']

            all_dfs.append(ddf[['raw_id1', 'raw_id2']])

        if not all_dfs:
            print("  No valid raw interaction files could be read for regex processing.")
            return

        combined_ddf = dd.concat(all_dfs).dropna().drop_duplicates()
        mapped_ddf = combined_ddf.map_partitions(map_ids_with_regex, meta={'raw_id1': 'str', 'raw_id2': 'str', 'protein1': 'str', 'protein2': 'str'})
        final_ddf = mapped_ddf[['protein1', 'protein2']].drop_duplicates()

        with ProgressBar():
            num_pairs = len(final_ddf)
            print(f"  Saving {num_pairs} standardized interaction pairs to {output_path}...")
            final_ddf.to_parquet(str(output_path), engine='pyarrow', overwrite=True, write_index=False)

        print(f"  ✅ Successfully created standardized ground truth file: {output_path.name}")

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
                import os
                print(f"Reading parquet directory: {filepath}")
                print(f"Contents: {os.listdir(filepath)}")
                ddf = dd.read_parquet(str(filepath), columns=['protein1', 'protein2'])
                ddf = ddf.rename(columns={'protein1': 'p1', 'protein2': 'p2'})
            else:
                sep = '\t' if '.mitab' in filepath.name or '.tsv' in filepath.name else ','
                ddf = dd.read_csv(str(filepath), sep=sep, header=None, names=['p1', 'p2'], usecols=[0, 1], dtype=str, on_bad_lines='warn')
            ddf = ddf.dropna().astype(str)
            filtered_ddf = ddf[ddf['p1'].isin(available_ids) & ddf['p2'].isin(available_ids)]

            # --- DEFINITIVE FIX: Make tqdm compatible with Dask's delayed objects ---
            # The previous implementation failed because tqdm tries to get the length of the
            # delayed object iterator, which is unknown. Providing the total number of
            # partitions explicitly solves this issue.
            for partition in tqdm(filtered_ddf.to_delayed(), total=filtered_ddf.npartitions, desc=f"    Scanning {filepath.name}", leave=False):
                for row in partition.compute().itertuples(index=False):
                    yield row.p1, row.p2, label
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
                # The previous check was brittle if a directory was not a parquet file.
                if filepath.is_dir():
                    # It's a directory, assume it's Parquet. Let pd.read_parquet handle errors.
                    ddf = dd.read_parquet(str(filepath))
                    df = ddf.compute()
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