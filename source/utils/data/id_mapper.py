# ==============================================================================
# MODULE: source/utils/data/id_mapper.py
# PURPOSE: Manages the creation and application of protein ID maps.
# VERSION: 3.0 (Streamlined to use direct DAT-to-Pickle conversion)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Mapping, Optional, Any

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from Bio import SeqIO
from tqdm.auto import tqdm


from configuration.config import Config
from source.utils.data.fasta_utils import FastaUtils
from source.utils.fs.file_utils import FileUtils


def _parse_fasta_for_regex_map(fasta_path: Path) -> Dict[str, str]:
    """
    Helper function to be run in a parallel process. Parses a single FASTA file
    and extracts ID mappings using regex.
    """
    local_id_map = {}
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            canonical_id = FastaUtils.extract_id_from_header(record.description)
            if canonical_id:
                local_id_map[record.id] = canonical_id
                first_word = record.description.split()[0]
                if first_word != record.id:
                    local_id_map[first_word] = canonical_id
    except Exception as e:
        print(f"An error occurred during parallel regex mapping on {fasta_path}: {e}")
    return local_id_map


class IDMapper:
    """
    A singleton class to manage loading and applying the protein ID map.

    This class handles the one-time, efficient creation of an ID mapping
    dictionary from the raw 'idmapping.dat' file. It ensures the map is
    loaded only once from its pickle cache for all subsequent uses.
    """
    _instance = None
    _id_map: Optional[Dict[str, str]] = None
    _mode_used: Optional[str] = None

    def __new__(cls, config: Config):
        if cls._instance is None:
            cls._instance = super(IDMapper, cls).__new__(cls)
            cls._instance.config = config
        return cls._instance

    @classmethod
    def reset(cls):
        """Resets the singleton instance for testing purposes."""
        cls._instance = None
        cls._id_map = None
        cls._mode_used = None

    def get_map(self) -> Optional[Mapping[str, str]]:
        """
        Gets the ID mapping dictionary. NOTE: For 'file' mode, this is now
        deprecated due to memory constraints and will return an empty dict.
        The mapping is now handled on-demand by the EmbeddingProcessor.
        """
        mode = self.config.ID_MAPPING_MODE

        if mode == 'file':
            print("  WARNING: IDMapper.get_map() was called in 'file' mode. This is deprecated.")
            print("  Returning an empty map to prevent OOM errors. On-demand mapping should be used instead.")
            return {}

        if mode == 'none':
            return None

        if self._id_map is not None and self._mode_used == mode:
            return self._id_map

        self._mode_used = mode
        cache_filename = f"{mode}_map_cache.pkl"
        local_cache_path = self.config.DATA_MAPPINGS_DIR / cache_filename
        persistent_cache_path = self.config.PERSISTENT_DATA_CACHE / cache_filename

        # 1. Try loading from local project root (if recently generated or restored)
        if local_cache_path.exists():
            print(f"  INFO: Loading ID map for mode '{mode}' from local cache: {local_cache_path.name}")
            self._id_map = FileUtils.load_object(local_cache_path)
            if self._id_map is None:
                print(f"  - ❌ ERROR: Failed to load or unpickle local cached ID map '{local_cache_path.name}'.")
                self._id_map = {} # Fallback
            return self._id_map

        # 2. If not in local project root, try loading from persistent cache
        if persistent_cache_path.exists():
            print(f"  INFO: Loading ID map for mode '{mode}' from persistent cache: {persistent_cache_path.name}")
            self._id_map = FileUtils.load_object(persistent_cache_path)
            if self._id_map is None:
                print(f"  - ❌ ERROR: Failed to load or unpickle persistent cached ID map '{persistent_cache_path.name}'.")
                self._id_map = {} # Fallback
            else:
                # Copy to local project root for faster access in subsequent calls
                print(f"  INFO: Copying ID map from persistent cache to local project root: {local_cache_path.name}")
                FileUtils.save_object(self._id_map, local_cache_path)
            return self._id_map

        # 3. If not found anywhere, log error and return empty dict
        print(f"  - ❌ ERROR: ID map cache file not found anywhere for mode '{mode}'. The data setup process may have failed.")
        self._id_map = {} # Return empty dict to prevent crashes

        return self._id_map

    def pregenerate_caches(self):
        """
        Performs the slow, one-time generation of the ID map cache file.
        This method should ONLY be called by the DataManager during initial setup.
        """
        mode = self.config.ID_MAPPING_MODE
        if mode == 'none':
            return

        if mode == 'file':
            # Check if the final parquet file exists.
            if self.config.ID_MAPPING_PATH.exists():
                print(f"  INFO: ID map Parquet file '{self.config.ID_MAPPING_PATH.name}' already exists. Skipping generation.")
                return
            self._generate_from_dat_file_direct()
        elif mode == 'regex':
            cache_filename = f"{mode}_map_cache.pkl"
            local_cache_path = self.config.DATA_MAPPINGS_DIR / cache_filename
            if local_cache_path.exists():
                print(f"  INFO: ID map cache '{local_cache_path.name}' already exists. Skipping generation.")
                return
            id_map = self._generate_from_regex()
            if id_map:
                print(f"  Saving newly generated ID map to local cache: {local_cache_path.name}")
                FileUtils.save_object(id_map, local_cache_path)
        elif mode == 'api':
            raise NotImplementedError("The 'api' mapping mode is configured but not yet implemented.")

    def _generate_from_dat_file_direct(self):
        """
        Directly creates the ID map from the raw `idmapping.dat` file using Dask.
        """
        print("  INFO: Starting direct conversion of 'idmapping.dat' to dictionary...")
        # --- DEFINITIVE FIX: Use the dynamic path from the config object ---
        # This removes the hardcoded 'idmapping.dat' filename.
        source_dat_path = self.config.ID_MAPPING_RAW_PATH
        # Normalize to Path in case a string or PathLike was provided
        if not isinstance(source_dat_path, Path):
            source_dat_path = Path(os.fspath(source_dat_path))

        if not source_dat_path.exists():
            logging.error(f"'idmapping.dat' not found at expected location: {source_dat_path}")
            print(f"  - ❌ ERROR: 'idmapping.dat' not found. Cannot generate map.")
            return {}

        try:
            with ProgressBar(dt=5.0):
                ddf = dd.read_csv(
                    source_dat_path,
                    sep='\t',
                    header=None,
                    names=['uniprot_id', 'db_type', 'db_id'],
                    usecols=['uniprot_id', 'db_type', 'db_id'],
                    dtype={'uniprot_id': 'string', 'db_type': 'string', 'db_id': 'string'},
                    on_bad_lines='warn',
                    blocksize='128MB'
                )

                # --- DEFINITIVE FIX for Ambiguous/Incorrect ID Mappings ---
                # The raw idmapping.dat file contains many-to-one mappings (e.g., one GeneID
                # can map to multiple UniProt isoforms). This was causing non-deterministic
                # mapping. The new logic creates a canonical, one-to-one map.
                print("  - Filtering mappings to trusted databases...")
                target_dbs = [
                    'BioGrid', 'GeneID', 'EMBL', 'RefSeq', 'PDB',
                    'UniProtKB-AC'  # Include self-mappings
                ]
                ddf = ddf[ddf['db_type'].isin(target_dbs)]

                print("  - De-duplicating mappings to create a canonical one-to-one map...")
                # For any db_id that still maps to multiple uniprot_ids, we sort and
                # take the first one. This is a deterministic way to choose the canonical entry.
                final_ddf = ddf[['db_id', 'uniprot_id']].drop_duplicates().sort_values(['db_id', 'uniprot_id'])
                final_ddf = final_ddf.drop_duplicates(subset=['db_id'], keep='first')

                print("  - Dask is now writing the final Parquet file. This may take a while...")
                output_path = self.config.ID_MAPPING_PATH
                output_path.parent.mkdir(parents=True, exist_ok=True)
                final_ddf.to_parquet(output_path, engine='pyarrow', overwrite=True)
                print(f"  - ✅ Success! Generated ID map Parquet file at {output_path}.")

        except Exception as e:
            logging.error(f"Failed during direct DAT-to-dictionary conversion: {e}", exc_info=True)
            # No return value needed as we write to disk

    def _generate_from_regex(self) -> Dict[str, str]:
        """Performs ID mapping by parsing FASTA headers with regular expressions."""
        fasta_paths = self.config.SEQUENCE_FILE_PATHS
        if not fasta_paths: return {}

        print(f"  Starting Regex ID mapping for: {[p.name for p in fasta_paths]}...")
        id_map = {}

        # --- DEFINITIVE FIX for Performance: Parallelize FASTA parsing for Regex mode ---
        # This avoids a bottleneck if the regex mode is ever used on very large files.
        num_workers = self.config.GRAPH_BUILDER_WORKERS
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_parse_fasta_for_regex_map, fasta_paths),
                                total=len(fasta_paths), desc="  Parsing FASTA for Regex Map (parallel)"))

        for partial_map in results:
            id_map.update(partial_map)

        print(f"  Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map