# ==============================================================================
# MODULE: source/utils/data/id_mapper.py
# PURPOSE: Manages the creation and application of protein ID maps.
# VERSION: 3.0 (Streamlined to use direct DAT-to-Pickle conversion)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Any

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from Bio import SeqIO
from tqdm.auto import tqdm


from configuration.config import Config
from source.utils.data.fasta_utils import FastaUtils
from source.utils.fs.file_utils import FileUtils


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
        Gets the ID mapping dictionary based on the configuration.

        This method is idempotent. It loads the map from the appropriate
        pickle file on the first call and returns the cached-in-memory
        dictionary on subsequent calls.
        """
        mode = self.config.ID_MAPPING_MODE

        if mode == 'none':
            return None

        if self._id_map is not None and self._mode_used == mode:
            return self._id_map

        self._mode_used = mode
        cache_filename = f"{mode}_map_cache.pkl"
        cache_path = self.config.PROJECT_ROOT / cache_filename

        if cache_path.exists():
            print(f"  INFO: Loading ID map for mode '{mode}' from cache: {cache_path.name}")
            self._id_map = FileUtils.load_object(cache_path)
            if self._id_map is None:
                print(f"  - ❌ ERROR: Failed to load or unpickle cached ID map '{cache_path.name}'.")
        else:
            print(f"  - ❌ ERROR: ID map cache file not found: '{cache_path.name}'. The data setup process may have failed.")
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

        cache_filename = f"{mode}_map_cache.pkl"
        local_cache_path = self.config.PROJECT_ROOT / cache_filename

        if local_cache_path.exists():
            print(f"  INFO: ID map cache '{local_cache_path.name}' already exists. Skipping generation.")
            return

        id_map = {}
        if mode == 'file':
            id_map = self._generate_from_dat_file_direct()
        elif mode == 'regex':
            id_map = self._generate_from_regex()
        elif mode == 'api':
            raise NotImplementedError("The 'api' mapping mode is configured but not yet implemented.")

        if id_map:
            print(f"  Saving newly generated ID map to local cache: {local_cache_path.name}")
            FileUtils.save_object(id_map, local_cache_path)

    def _generate_from_dat_file_direct(self) -> Dict[str, str]:
        """
        Directly creates the ID map from the raw `idmapping.dat` file using Dask.
        """
        print("  INFO: Starting direct conversion of 'idmapping.dat' to dictionary...")
        source_dat_path = Path(self.config.PERSISTENT_DATA_CACHE) / 'idmapping.dat'

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
                    blocksize='64MB'
                )

                # --- DEFINITIVE FIX: Remove the incorrect filter ---
                # The previous logic only mapped UniProt IDs to themselves, resulting in an
                # incomplete (and in this case, empty) map. The correct behavior is to
                # map ALL database IDs in the file to their canonical UniProt ID.
                print("  - Dask is now processing the file in parallel. This may take a while...")
                computed_df = ddf.compute()

            print("  - Aggregating computed results into the final dictionary...")
            id_map = dict(zip(computed_df['db_id'], computed_df['uniprot_id']))
            del ddf, computed_df
            gc.collect()

            print(f"  - ✅ Success! Generated ID map with {len(id_map)} entries.")
            return id_map
        except Exception as e:
            logging.error(f"Failed during direct DAT-to-dictionary conversion: {e}", exc_info=True)
            return {}

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