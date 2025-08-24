import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union, Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils
from source.utils.fs.file_utils import FileUtils


# ==============================================================================
# 3. Protein ID Mapping Utilities
# ==============================================================================

# --- 3a. ID Map Generator ---
class IDMapGenerator:
    """
    Handles the complex process of generating protein ID mappings from various
    sources (API, regex, or large mapping files).
    """

    def __init__(self, config: Config):
        self.config = config
        self.fasta_files_for_mapping = config.SEQUENCE_FILE_PATHS
        self.mapping_output_file = str(config.ID_MAPPING_PATH) # This is now only used for regex mode
        self.random_seed_for_mapping = config.RANDOM_STATE
        self.mapping_mode = config.ID_MAPPING_MODE

    def generate_id_maps(self) -> Optional[Union[Mapping[str, str], dd.DataFrame]]:
        """
        Main entry point for generating ID mappings.
        Returns a dictionary for 'regex'/'api' modes or a Dask DataFrame for 'file' mode.
        """
        # --- DEFINITIVE FIX: The generate_id_maps method should always return a dictionary. ---
        # The logic for choosing the mode and caching the result is now centralized here.

        if self.mapping_mode == 'file':
            DataUtils.print_header("Loading Protein ID Mapping from File")
            # --- NEW CACHING LOGIC FOR FILE MODE ---
            cache_file_path = self.config.DATA_MAPPINGS_DIR / "file_map_cache.pkl"
            if cache_file_path.exists():
                print(f"  Found cached file-based ID map. Loading from: {cache_file_path.name}")
                cached_map = FileUtils.load_object(cache_file_path)
                if cached_map is not None:
                    return cached_map
                print("  Warning: Cached map file is corrupted. Regenerating...")

            # If cache doesn't exist, generate it from the Parquet file
            print("  No cache found. Generating ID map dictionary from Parquet file (one-time operation)...")
            mapping_ddf = self._get_mapping_dask_dataframe()
            if mapping_ddf is None:
                return {}

            with ProgressBar():
                # This is the expensive, one-time computation
                mapping_df = mapping_ddf.compute()

            # Convert to a dictionary for fast lookups
            id_map = dict(zip(mapping_df['original_id'], mapping_df['mapped_id']))

            if id_map:
                print(f"  Saving newly generated ID map to cache: {cache_file_path.name}")
                FileUtils.save_object(id_map, cache_file_path)

            print("--- Protein ID Mapping Finished ---")
            return id_map

        if self.mapping_mode == 'regex':
            DataUtils.print_header("Generating Protein ID Mapping (Regex Mode)")
            # --- DEFINITIVE FIX: Implement caching for regex-generated maps ---
            # Create a unique cache file name based on the primary input FASTA file.
            if not self.fasta_files_for_mapping:
                print("  Warning: No FASTA files provided for regex mapping. Cannot generate or load cache.")
                return {}
            input_fasta_stem = self.fasta_files_for_mapping[0].stem
            cache_file_path = self.config.DATA_MAPPINGS_DIR / f"regex_map_cache_{input_fasta_stem}.pkl"

            # Try to load from the persistent cache first.
            if cache_file_path.exists():
                print(f"  Found cached regex ID map. Loading from: {cache_file_path.name}")
                cached_map = FileUtils.load_object(cache_file_path)
                if cached_map is not None:
                    return cached_map
                print("  Warning: Cached map file is corrupted. Regenerating...")

            # If cache doesn't exist or is corrupt, generate the map.
            id_map = self._perform_regex_mapping()
            if id_map:
                print(f"  Saving newly generated regex ID map to cache: {cache_file_path.name}")
                FileUtils.save_object(id_map, cache_file_path)
            return id_map

        if self.mapping_mode == 'api':
            raise NotImplementedError("The 'api' mapping mode is configured but not yet implemented.")

        # Default case for 'none' or unknown modes
        return {}

    def _assess_fasta_header_compatibility(self) -> float:
        """
        Samples the FASTA file to determine how compatible its headers are with
        the fast regex parser.
        """
        print("  Assessing FASTA header compatibility for smart mapping...")
        # --- REFACTOR: Delegate to the centralized FastaUtils method ---
        return FastaUtils.check_uniprot_header_compatibility(
            fasta_paths=self.fasta_files_for_mapping,
            sample_size=self.config.REGEX_COMPATIBILITY_SAMPLE_SIZE
        )

    def _get_mapping_dask_dataframe(self) -> Optional[dd.DataFrame]:
        """
        Loads the large mapping file into a Dask DataFrame for scalable lookups.
        """
        source_parquet_path = self.config.ID_MAPPING_PATH
        if not source_parquet_path or not source_parquet_path.exists():
            print(f"ERROR: Processed mapping file not found at {source_parquet_path}.")
            print("Ensure the data setup process has been run successfully.")
            return None

        print(f"  Loading ID mapping from Parquet file: {source_parquet_path.name}")
        ddf = dd.read_parquet(source_parquet_path)
        # Rename for clarity in the merge operation
        ddf = ddf.rename(columns={'other_id': 'original_id', 'uniprot_id': 'mapped_id'})
        # Persist in memory for faster subsequent operations
        return ddf.persist()

    @staticmethod
    def apply_mapping(embeddings: Dict[str, np.ndarray], mapper: Any) -> Dict[str, np.ndarray]:
        """
        Applies an ID mapping to a dictionary of embeddings.
        This function now only expects an in-memory dictionary for mapping.
        """
        if mapper is None or not isinstance(mapper, Mapping) or not embeddings:
            return embeddings

        print("  Applying ID mapping to generated embeddings...")
        original_count = len(embeddings)

        # The mapper is now always a dictionary, so this is the only path needed.
        mapped_embeddings = {mapper.get(k, k): v for k, v in embeddings.items()}

        print(f"    Original count: {original_count}, Mapped count: {len(mapped_embeddings)}")
        return mapped_embeddings