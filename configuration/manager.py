# ==============================================================================
# MODULE: configuration/manager.py
# PURPOSE: Handles the verification and acquisition of all external data files.
# VERSION: 4.0 (Integrated checksum validation, bundling, and restoration)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gzip
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
# --- NEW: Import PyG for benchmark dataset downloading ---
from torch_geometric.datasets import Planetoid, WebKB, Actor, KarateClub
from tqdm.auto import tqdm

from .processor import DataProcessor

# Conditionally import gdown to avoid making it a hard dependency
try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


# Local imports must be inside methods to avoid circular dependencies with Config
# from configuration.config import Config # Avoid top-level import


class DataManager:
    """
    Handles the download, processing, and validation of all project data.
    This class orchestrates the creation of a clean, analysis-ready set of
    Parquet files from various raw data sources.
    """

    def __init__(self, config):
        self.config = config
        self.files_to_cleanup: list[Path] = []

    def run_full_setup(self):
        """
        Executes the entire data pipeline: download, process, and bundle.
        This is a long-running, one-time operation.
        """
        print("\n--- Running Data Setup and Processing ---")
        print("  - Ensuring project data directory structure exists...")
        # Note: The Config object, passed during initialization, has already created
        # all necessary subdirectories (e.g., data/sequences, data/models).

        # 1. Download all raw source files
        self._download_all_sources()

        # 2. Process raw files into final Parquet format
        # --- FIX: Instantiate the processor to call instance methods ---
        processor = DataProcessor(self.config)
        processor._process_uniprot_mapping()
        processor._process_negative_interactions()
        processor._process_biogrid_interactions()

        # 3. Generate manifest and bundle the final data
        self._generate_manifest_and_bundle()

        # 4. Clean up intermediate files
        self._cleanup_intermediate_files()

        print("--- Data Setup and Processing Complete ---")

    def validate_data_from_manifest(self) -> bool:
        """Validates the current data directory against the cached manifest."""
        manifest_path = self.config.DATA_MANIFEST_PATH
        if not manifest_path.exists():
            return False

        print(f"--- Validating data against manifest: {manifest_path.name} ---")
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("  - ERROR: Manifest file is corrupt or unreadable.")
            return False

        all_valid = True
        for relative_path_str, properties in manifest.items():
            file_path = self.config.PROJECT_ROOT / relative_path_str
            if not file_path.exists():
                print(f"  - ❌ MISSING: {relative_path_str}")
                all_valid = False
                continue

            current_size = file_path.stat().st_size
            if current_size != properties['size']:
                print(f"  - ❌ INVALID SIZE: {relative_path_str} (Expected: {properties['size']}, Found: {current_size})")
                all_valid = False
                continue

            current_checksum = DataProcessor._calculate_sha256(file_path)
            if current_checksum != properties['sha256']:
                print(f"  - ❌ INVALID CHECKSUM: {relative_path_str}")
                all_valid = False
                continue

            print(f"  - ✅ VALID: {relative_path_str}")

        return all_valid

    def restore_data_from_bundle(self) -> bool:
        """Restores the data directory from the cached tar.gz bundle."""
        bundle_path = self.config.DATA_BUNDLE_PATH
        if not bundle_path.exists():
            print(f"  - ERROR: Data bundle not found at {bundle_path}. Cannot restore.")
            return False

        print(f"--- Restoring data from cached bundle: {bundle_path.name} ---")
        # Ensure the target data directory is clean before extraction
        if self.config.BASE_DATA_DIR.exists():
            shutil.rmtree(self.config.BASE_DATA_DIR)

        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                tar.extractall(path=self.config.PROJECT_ROOT)
            print("  - ✅ Successfully extracted data bundle.")
            # Re-validate after extraction to be certain
            return self.validate_data_from_manifest()
        except (tarfile.ReadError, IOError) as e:
            print(f"  - ❌ ERROR: Failed to extract data bundle: {e}")
            return False

    def _download_benchmark_datasets(self):
        """Uses PyG to download standard benchmark datasets."""
        print("\n--- Downloading PyG Benchmark Datasets ---")
        dataset_root = self.config.DATA_STANDARD_DATASETS_DIR
        dataset_root.mkdir(parents=True, exist_ok=True)

        for name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            print(f"  - Ensuring dataset '{name}' is downloaded...")
            try:
                if name in ['Cora', 'CiteSeer', 'PubMed']:
                    Planetoid(root=str(dataset_root), name=name)
                elif name in ['Cornell', 'Texas', 'Wisconsin']:
                    WebKB(root=str(dataset_root), name=name)
                elif name == 'Actor':
                    Actor(root=str(dataset_root))
                elif name == 'KarateClub':
                    KarateClub()
            except Exception as e:
                print(f"    - WARNING: Failed to download PyG dataset '{name}': {e}")

    def _download_all_sources(self):
        """Downloads and extracts all data sources defined in the config."""
        print("\n--- Step 1: Downloading and Extracting Raw Data ---")
        if hasattr(self.config, 'PERSISTENT_DATA_CACHE'):
            self.config.PERSISTENT_DATA_CACHE.mkdir(parents=True, exist_ok=True)

        for key, source_info in self.config.DATA_SOURCES.items():
            # --- NEW: Handle different source types ---
            if source_info.get("type") == "pyg_dataset":
                continue  # These are handled separately

            final_path = Path(source_info['path'])
            is_cacheable = source_info.get('cacheable', False)
            cache_path = self.config.PERSISTENT_DATA_CACHE / final_path.name if is_cacheable else None

            post_process_type = source_info.get('post_process')
            if post_process_type == 'ungzip':
                download_target_path = final_path.with_suffix(final_path.suffix + ".gz")
            elif post_process_type == 'unzip':
                download_target_path = final_path.with_suffix(".zip")
            else:
                download_target_path = final_path

            if download_target_path != final_path:
                self.files_to_cleanup.append(download_target_path)
            if post_process_type:
                self.files_to_cleanup.append(final_path)

            if DataProcessor._is_file_valid(final_path):
                print(f"☑ Found and verified raw file: {final_path.relative_to(self.config.PROJECT_ROOT)}")
                continue

            if cache_path and DataProcessor._is_file_valid(cache_path):
                print(f"☑ Found cached file: {cache_path}. Copying to project directory...")
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if final_path.exists() or final_path.is_symlink(): final_path.unlink()
                shutil.copy(cache_path, final_path)
                continue

            url = source_info.get('url')
            if not url: continue

            try:
                # The parent directory for the download target is guaranteed to exist
                # because the Config object creates the entire data structure on initialization.
                # --- FIX: Re-introduce gdown logic for Google Drive URLs ---
                if 'drive.google.com' in url:
                    if not GDOWN_AVAILABLE:
                        print(f"  ERROR: URL for '{key}' is a Google Drive link, but 'gdown' is not installed. Skipping.")
                        continue
                    print(f"Downloading '{download_target_path.name}' from Google Drive...")
                    gdown.download(url, str(download_target_path), quiet=False, fuzzy=True)
                else:
                    # --- FIX: Ensure the target directory exists before downloading ---
                    download_target_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Downloading from {url} to {download_target_path.name}...") # --- FIX: Add a standard User-Agent header to prevent being blocked by some servers (e.g., BioGRID) ---
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    # --- FIX: Add verify=False to handle potential SSL certificate issues in some environments ---
                    response = requests.get(url, stream=True, verify=False, headers=headers)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    with open(download_target_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True,
                                                                     desc=download_target_path.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                if post_process_type == 'ungzip':
                    print(f"Decompressing {download_target_path.name}...")
                    with gzip.open(download_target_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                elif post_process_type == 'unzip':
                    print(f"Decompressing {download_target_path.name}...")
                    with zipfile.ZipFile(download_target_path, 'r') as zip_ref:
                        file_to_extract = sorted(zip_ref.infolist(), key=lambda z: z.file_size, reverse=True)[0]
                        with zip_ref.open(file_to_extract) as zf, open(final_path, 'wb') as f_out:
                            shutil.copyfileobj(zf, f_out)

                # --- FIX: Corrected caching logic. Just copy to cache if it's not already there. ---
                if DataProcessor._is_file_valid(final_path) and cache_path and not cache_path.exists():
                    print(f"  Copying '{final_path.name}' to persistent cache for future use...")
                    shutil.copy(final_path, cache_path)

            except Exception as e:
                print(f"Error acquiring file for '{key}': {e}")

        # --- NEW: Trigger benchmark dataset download ---
        self._download_benchmark_datasets()

    def _generate_manifest_and_bundle(self):
        """Generates a checksum manifest and bundles the data directory."""
        print("\n--- Step 3: Generating Data Manifest and Bundling ---")
        manifest = {}
        # 1. Find all files to include in the manifest
        files_to_manifest = []
        for dirpath, _, filenames in os.walk(self.config.BASE_DATA_DIR):
            for f in filenames:
                files_to_manifest.append(Path(dirpath) / f)

        # 2. Generate checksums and sizes
        print(f"  - Generating checksums for {len(files_to_manifest)} files...")
        # --- DEFINITIVE FIX for Checksum Performance/OOM Kill ---
        # Define a set of huge files for which we will skip the expensive SHA256 calculation.
        # A simple size check is sufficient for these large, static source files.
        huge_files_to_skip_checksum = {
            "uniref50.fasta",
            "idmapping.dat"
        }
        for file_path in tqdm(files_to_manifest, desc="  Calculating Checksums"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.config.PROJECT_ROOT)
                checksum = "skipped_due_to_size" if file_path.name in huge_files_to_skip_checksum else DataProcessor._calculate_sha256(file_path)
                manifest[relative_path.as_posix()] = {
                    'size': file_path.stat().st_size,
                    'sha256': checksum
                }

        # 3. Save the manifest to the cache
        manifest_path = self.config.DATA_MANIFEST_PATH
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"  - ✅ Manifest saved to: {manifest_path}")
        except IOError as e:
            print(f"  - ❌ ERROR: Could not save manifest file: {e}")
            return

        # 4. Create the tar.gz bundle in the cache
        bundle_path = self.config.DATA_BUNDLE_PATH
        print(f"  - Creating data bundle at: {bundle_path}...")
        try:
            # This compresses the entire project's 'data' directory into a single
            # tar.gz file, which will be stored in the persistent cache.
            with tarfile.open(bundle_path, "w:gz") as tar:
                tar.add(self.config.BASE_DATA_DIR, arcname=self.config.BASE_DATA_DIR.name)
            print(f"  - ✅ Data successfully bundled.")
        except (tarfile.TarError, IOError) as e:
            print(f"  - ❌ ERROR: Could not create data bundle: {e}")

    def _cleanup_intermediate_files(self):
        """Removes all downloaded and intermediate raw files."""
        print("\n--- Step 3: Cleaning Up Intermediate Files ---")
        for f_path in set(self.files_to_cleanup):
            if f_path.exists():
                try:
                    if f_path.is_dir():
                        shutil.rmtree(f_path)
                    else:
                        f_path.unlink()
                    # --- FIX: Complete the print statement ---
                    print(f"  - Cleaned up: {f_path.name}")
                except OSError as e:
                    print(f"  - WARNING: Could not clean up file {f_path.name}. Error: {e}")


def setup_data(config):
    """
    Public entry point to instantiate and run the DataManager.
    """
    DataManager(config).run_full_setup()
