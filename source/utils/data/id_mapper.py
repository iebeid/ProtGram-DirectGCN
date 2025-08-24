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
        if not self.mapping_output_file and self.mapping_mode not in ['file', 'none']:
            return None

        if self.mapping_mode == 'file':
            DataUtils.print_header("Loading Protein ID Mapping from File")
            return self._get_mapping_dask_dataframe()

        DataUtils.print_header("Generating Protein ID Mapping")
        output_dir = os.path.dirname(self.mapping_output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        id_map: Dict[str, str] = {}
        if self.mapping_mode == 'regex':
            id_map = self._perform_regex_mapping()
        elif self.mapping_mode == 'api':
            # --- FIX: Add explicit handling for the 'api' mode ---
            raise NotImplementedError("The 'api' mapping mode is configured but not yet implemented.")
        elif self.mapping_mode == 'none':
            return {}
        else:
            print(f"Warning: Unknown ID_MAPPING_MODE '{self.mapping_mode}'.")
            return {}

        if id_map:
            try:
                with open(self.mapping_output_file, 'w', encoding='utf-8') as f:
                    for original, mapped in id_map.items():
                        f.write(f"{original}\t{mapped}\n")
                print(f"ID mapping saved to {self.mapping_output_file}")
            except IOError as e:
                print(f"ERROR: Could not write ID mapping file: {e}")
        print("--- Protein ID Mapping Finished ---")
        return id_map

    def _perform_regex_mapping(self) -> Dict[str, str]:
        """Performs ID mapping by parsing FASTA headers with regular expressions."""
        if not self.fasta_files_for_mapping: return {}
        print(f"Starting Regex ID mapping for: {[p.name for p in self.fasta_files_for_mapping]}...")
        id_map = {}
        for fasta_file in self.fasta_files_for_mapping:
            try:
                # --- REFACTOR: Use the centralized FastaUtils header parser to avoid code duplication ---
                for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc=f"Parsing {fasta_file.name} with Regex", leave=False):
                    # The header might contain multiple IDs. We want to map them all to one canonical ID.
                    # The canonical ID is what we get from our robust regex.
                    canonical_id = FastaUtils.extract_id_from_header(record.description)
                    if canonical_id:
                        # Map the ID that BioPython parsed as the main ID
                        id_map[record.id] = canonical_id
                        # Also map the first word of the header, as it's often used as an ID
                        first_word = record.description.split()[0]
                        if first_word != record.id:
                            id_map[first_word] = canonical_id
            except Exception as e:
                print(f"An error during regex mapping on {fasta_file}: {e}")
        print(f"Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map

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
        This function handles both in-memory dictionaries and Dask DataFrames for mapping.
        """
        # --- DEFINITIVE FIX: Check for None explicitly to avoid ambiguous truth value error ---
        if mapper is None or not embeddings:
            return embeddings

        print("  Applying ID mapping to generated embeddings...")
        original_count = len(embeddings)

        if isinstance(mapper, dd.DataFrame):
            # Dask-based merge for large-scale mapping
            print("    Using Dask DataFrame for scalable mapping...")
            # Convert embeddings dict to a Pandas DataFrame, then to Dask
            emb_df = pd.DataFrame(embeddings.items(), columns=['original_id', 'embedding'])
            emb_ddf = dd.from_pandas(emb_df, npartitions=mapper.npartitions)

            # Perform the merge (join) operation
            with ProgressBar():
                merged_ddf = dd.merge(emb_ddf, mapper, on='original_id', how='left')
                # Use the mapped_id if available, otherwise keep the original
                merged_ddf['final_id'] = merged_ddf['mapped_id'].fillna(merged_ddf['original_id'])
                # Select final columns and compute the result
                final_df = merged_ddf[['final_id', 'embedding']].compute()

            # Convert back to a dictionary
            mapped_embeddings = dict(zip(final_df['final_id'], final_df['embedding']))

        elif isinstance(mapper, Mapping):
            # Standard dictionary-based mapping for smaller maps (regex, api)
            mapped_embeddings = {mapper.get(k, k): v for k, v in embeddings.items()}
        else:
            print(f"    Warning: Unknown mapper type '{type(mapper)}'. Skipping mapping.")
            return embeddings

        print(f"    Original count: {original_count}, Mapped count: {len(mapped_embeddings)}")
        return mapped_embeddings