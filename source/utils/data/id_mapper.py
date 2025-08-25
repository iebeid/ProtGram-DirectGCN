import gc
from pathlib import Path
from typing import Dict, Mapping, Optional, Any

import dask.dataframe as dd
import pandas as pd
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

class IDMapper:
    """
    A singleton class to manage loading and applying protein ID maps.

    This class ensures that the ID mapping dictionary is loaded from disk only
    once per application run, providing a consistent and efficient way for all
    pipelines to access it. It relies on the DataManager to have already
    generated the necessary cache files during the initial setup.
    """
    _instance = None
    _id_map: Optional[Dict[str, str]] = None
    _mode_used: Optional[str] = None

    def __new__(cls, config: Config):
        if cls._instance is None:
            cls._instance = super(IDMapper, cls).__new__(cls)
            # Store config on first creation, but don't re-initialize the map
            cls._instance.config = config
        return cls._instance

    def get_map(self) -> Optional[Mapping[str, str]]:
        """
        Gets the ID mapping dictionary based on the configuration.

        This method is idempotent. It loads the map from the appropriate
        pickle file on the first call and returns the cached-in-memory
        dictionary on subsequent calls.

        Returns:
            A mapping dictionary or None if mapping is disabled.
        """
        mode = self.config.ID_MAPPING_MODE

        if mode == 'none':
            return None

        # If map is already loaded and mode hasn't changed, return cached version
        if self._id_map is not None and self._mode_used == mode:
            return self._id_map

        # Load the map from the correct pickle file
        self._mode_used = mode
        cache_filename = f"{mode}_map_cache.pkl"
        # --- DEFINITIVE FIX: Check for the cache file in the project root, where DataManager places it. ---
        cache_path = self.config.PROJECT_ROOT / cache_filename

        if cache_path.exists():
            print(f"  INFO: Loading ID map for mode '{mode}' from cache: {cache_path.name}")
            self._id_map = FileUtils.load_object(cache_path)
            if self._id_map is None:
                print(f"  - ❌ ERROR: Failed to load or unpickle cached ID map '{cache_path.name}'.")
        else:
            print(f"  - ❌ ERROR: ID map cache file not found: '{cache_path.name}'. The data setup process may have failed.")
            self._id_map = None

        return self._id_map

    def pregenerate_caches(self):
        """
        Performs the slow, one-time generation of the ID map cache file.
        This method should ONLY be called by the DataManager during initial setup.
        """
        mode = self.config.ID_MAPPING_MODE
        if mode == 'none':
            return

        cache_filename = f"{mode}_map_cache.pkl"
        local_cache_path = self.config.PROJECT_ROOT / cache_filename

        if local_cache_path.exists():
            print(f"  INFO: ID map cache '{local_cache_path.name}' already exists. Skipping generation.")
            return

        id_map = {}
        if mode == 'file':
            id_map = self._generate_from_file()
        elif mode == 'regex':
            id_map = self._generate_from_regex()
        elif mode == 'api':
            raise NotImplementedError("The 'api' mapping mode is configured but not yet implemented.")

        if id_map:
            print(f"  Saving newly generated ID map to local cache: {local_cache_path.name}")
            FileUtils.save_object(id_map, local_cache_path)

    def _generate_from_file(self) -> Dict[str, str]:
        """Generates the ID map dictionary from the large Parquet file."""
        print("  Generating ID map dictionary from Parquet file (one-time operation)...")
        mapping_ddf = self._get_mapping_dask_dataframe()
        if mapping_ddf is None:
            return {}

        with ProgressBar():
            mapping_df = mapping_ddf.compute()

        id_map = dict(zip(mapping_df['original_id'], mapping_df['mapped_id']))
        del mapping_df, mapping_ddf
        gc.collect()
        return id_map

    def _generate_from_regex(self) -> Dict[str, str]:
        """Performs ID mapping by parsing FASTA headers with regular expressions."""
        fasta_files = self.config.SEQUENCE_FILE_PATHS
        if not fasta_files: return {}

        print(f"  Starting Regex ID mapping for: {[p.name for p in fasta_files]}...")
        id_map = {}
        for fasta_file in fasta_files:
            try:
                for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc=f"Parsing {fasta_file.name} with Regex", leave=False):
                    canonical_id = FastaUtils.extract_id_from_header(record.description)
                    if canonical_id:
                        id_map[record.id] = canonical_id
                        first_word = record.description.split()[0]
                        if first_word != record.id:
                            id_map[first_word] = canonical_id
            except Exception as e:
                print(f"An error during regex mapping on {fasta_file}: {e}")
        print(f"  Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map

    def _get_mapping_dask_dataframe(self) -> Optional[dd.DataFrame]:
        """Loads the large mapping file into a Dask DataFrame."""
        source_parquet_path = self.config.ID_MAPPING_PATH
        if not source_parquet_path or not source_parquet_path.exists():
            print(f"ERROR: Processed mapping file not found at {source_parquet_path}.")
            return None

        ddf = dd.read_parquet(source_parquet_path)
        ddf = ddf.rename(columns={'other_id': 'original_id', 'uniprot_id': 'mapped_id'})
        return ddf.persist()

    @staticmethod
    def apply_mapping(embeddings: Dict[str, Any], id_map: Optional[Mapping[str, str]]) -> Dict[str, Any]:
        """Applies the ID mapping to a dictionary of embeddings at the end of a pipeline."""
        if not id_map or not embeddings:
            return embeddings

        print("  INFO: Applying ID mapping to generated embeddings...")
        original_count = len(embeddings)
        mapped_embeddings = {id_map.get(k, k): v for k, v in embeddings.items()}
        print(f"    - Original IDs: {original_count}, Final Mapped IDs: {len(mapped_embeddings)}")
        return mapped_embeddings