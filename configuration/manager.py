# ==============================================================================
# MODULE: configuration/manager.py
# PURPOSE: Handles the verification and acquisition of all external data files.
# VERSION: 7.1 (Corrected caching for benchmark datasets)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gzip
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from torch_geometric.datasets import Planetoid, WebKB, Actor, KarateClub
from tqdm.auto import tqdm

from .processor import DataProcessor

# Conditionally import gdown to avoid making it a hard dependency
try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


class DataManager:
    """
    Handles the download, processing, and validation of all project data.
    This class orchestrates the creation of a clean, analysis-ready set of
    Parquet files from various raw data sources.
    """

    def __init__(self, config):
        self.config = config
        self.files_to_cleanup: list[Path] = []

    def _copy_to_cache(self, source_path: Path):
        """
        Copies a file or directory to the persistent cache if it exists.
        This is now robust and handles both files and directories (like .parquet).
        """
        if not source_path.exists() or not self.config.PERSISTENT_DATA_CACHE:
            return
        cache_path = self.config.PERSISTENT_DATA_CACHE / source_path.name

        # If the cache destination exists, remove it to ensure a clean copy.
        if cache_path.is_dir():
            shutil.rmtree(cache_path)
        elif cache_path.exists():
            cache_path.unlink()

        print(f"  Caching '{source_path.name}' for future runs...")
        if source_path.is_dir():
            shutil.copytree(source_path, cache_path)
        elif source_path.is_file():
            shutil.copy(source_path, cache_path)
        else:
            print(f"  - WARNING: Source path '{source_path}' is not a file or directory. Cannot cache.")

    def run_full_setup(self):
        """
        Executes the entire data pipeline: download, process, and create manifest.
        This is a long-running, one-time operation.
        """
        print("\n--- Running Data Setup and Processing ---")
        print("  - Ensuring project data directory structure exists...")

        # 1. Download all raw source files
        self._download_all_sources()

        # 2. Process raw files into final Parquet format and cache them
        processor = DataProcessor(self.config)
        processor._process_uniprot_mapping()
        self._copy_to_cache(self.config.ID_MAPPING_PATH)
        processor._process_negative_interactions()
        self._copy_to_cache(self.config.NEG_INTERACTIONS_PATH)
        processor._process_biogrid_interactions()
        self._copy_to_cache(self.config.POS_INTERACTIONS_PATH)

        # 3. Generate manifest of all data files
        self._generate_manifest()

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

            # For directories, just check for existence.
            # For files, check size and checksum.
            is_dir = properties.get('type') == 'directory'
            if is_dir:
                # This is a directory, just confirm it exists.
                print(f"  - ✅ VALID: {relative_path_str}")
                continue

            current_size = file_path.stat().st_size
            if current_size != properties['size']:
                print(f"  - ❌ INVALID SIZE: {relative_path_str} (Expected: {properties['size']}, Found: {current_size})")
                all_valid = False
                continue

            if properties['sha256'] != "skipped_due_to_size":
                current_checksum = DataProcessor._calculate_sha256(file_path)
                if current_checksum != properties['sha256']:
                    print(f"  - ❌ INVALID CHECKSUM: {relative_path_str}")
                    all_valid = False
                    continue

            print(f"  - ✅ VALID: {relative_path_str}")

        return all_valid

    def restore_data_from_cache(self) -> bool:
        """Restores the data directory from individual files in the cache, guided by the manifest."""
        manifest_path = self.config.DATA_MANIFEST_PATH
        if not manifest_path.exists():
            print(f"  - ERROR: Cached manifest not found at {manifest_path}. Cannot restore.")
            return False

        print(f"--- Restoring data from individual files in cache based on manifest: {manifest_path.name} ---")
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            processed_files_in_cache = {
                Path(p).name for p in self.config.PROCESSED_FILE_DEPENDENCIES.keys()
                if (self.config.PERSISTENT_DATA_CACHE / Path(p).name).exists()
            }
            raw_files_to_skip = set()
            for processed_file, raw_dependencies in self.config.PROCESSED_FILE_DEPENDENCIES.items():
                if processed_file in processed_files_in_cache:
                    raw_files_to_skip.update(raw_dependencies)
            
            if raw_files_to_skip:
                print(f"  Smart Restore: Will skip restoring raw files: {raw_files_to_skip}")

            files_restored = 0
            for relative_path_str, properties in manifest.items():
                project_file_path = self.config.PROJECT_ROOT / relative_path_str
                
                # If a file or directory already exists, skip it.
                if project_file_path.exists():
                    continue

                cached_file_path = self.config.PERSISTENT_DATA_CACHE / Path(relative_path_str).name

                if cached_file_path.exists() and cached_file_path.name not in raw_files_to_skip:
                    print(f"  Restoring '{project_file_path.name}' from cache...")

                    project_file_path.parent.mkdir(parents=True, exist_ok=True)

                    if cached_file_path.is_dir():
                        shutil.copytree(cached_file_path, project_file_path)
                    else:
                        shutil.copy(cached_file_path, project_file_path)
                    files_restored += 1
                else:
                    if not project_file_path.exists():
                        print(f"  - WARNING: Manifest lists '{relative_path_str}' but it's not in the cache. It will need to be re-downloaded/processed.")

            print(f"  - ✅ Successfully restored {files_restored} file(s) from cache.")
            return self.validate_data_from_manifest()
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"  - ❌ ERROR: Failed to read manifest or restore from cache: {e}")
            return False

    def _download_benchmark_datasets(self):
        """Uses PyG to download standard benchmark datasets and caches them."""
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
                
                # After downloading, copy the dataset directory to the cache.
                dataset_dir = dataset_root / name
                if dataset_dir.exists() and dataset_dir.is_dir():
                    self._copy_to_cache(dataset_dir)

            except Exception as e:
                print(f"    - WARNING: Failed to download PyG dataset '{name}': {e}")

    def _download_all_sources(self):
        """Downloads and extracts all data sources defined in the config."""
        print("\n--- Step 1: Downloading and Extracting Raw Data ---")
        if hasattr(self.config, 'PERSISTENT_DATA_CACHE'):
            self.config.PERSISTENT_DATA_CACHE.mkdir(parents=True, exist_ok=True)

        for key, source_info in self.config.DATA_SOURCES.items():
            if source_info.get("type") == "pyg_dataset":
                continue

            final_path = Path(source_info['path'])
            is_cacheable = source_info.get('cacheable', False)
            cache_path = self.config.PERSISTENT_DATA_CACHE / final_path.name if is_cacheable else None

            post_process_type = source_info.get('post_process')
            download_target_path = final_path
            if post_process_type == 'ungzip':
                download_target_path = final_path.with_suffix(final_path.suffix + ".gz")
            elif post_process_type == 'unzip':
                download_target_path = final_path.with_suffix(".zip")

            if download_target_path != final_path:
                self.files_to_cleanup.append(download_target_path)
            if post_process_type:
                self.files_to_cleanup.append(final_path)

            if DataProcessor._is_file_valid(final_path):
                print(f"☑ Found and verified raw file: {final_path.relative_to(self.config.PROJECT_ROOT)}")
                self._copy_to_cache(final_path)  # Ensure it's cached for future runs
                continue

            if cache_path and DataProcessor._is_file_valid(cache_path):
                print(f"☑ Found cached file: {cache_path}. Copying to project directory...")
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if final_path.exists() or final_path.is_symlink():
                    final_path.unlink()
                shutil.copy(cache_path, final_path)
                continue

            url = source_info.get('url')
            if not url:
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if 'drive.google.com' in url:
                        if not GDOWN_AVAILABLE:
                            print(f"  ERROR: URL for '{key}' is a Google Drive link, but 'gdown' is not installed. Skipping.")
                            break
                        print(f"Downloading '{download_target_path.name}' from Google Drive...")
                        gdown.download(url, str(download_target_path), quiet=False, fuzzy=True)
                    else:
                        download_target_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f"Downloading from {url} to {download_target_path.name}...")
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                        response = requests.get(url, stream=True, verify=False, headers=headers)

                        content_type = response.headers.get('content-type', '')
                        if post_process_type == 'unzip' and 'application/zip' not in content_type:
                            raise IOError(f"Downloaded file is not a zip file. Content-Type: '{content_type}'. Check URL or User-Agent.")
                        response.raise_for_status()

                        total_size = int(response.headers.get('content-length', 0))
                        with open(download_target_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=download_target_path.name) as pbar:
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

                    self._copy_to_cache(final_path)
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"Error acquiring file for '{key}' on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt + 1 == max_retries:
                        print(f"  - ❌ FAILED to acquire file for '{key}' after {max_retries} attempts.")
                    else:
                        time.sleep(5)  # Wait before retrying

        self._download_benchmark_datasets()

    def _generate_manifest(self):
        """Generates a checksum manifest for all data files."""
        print("\n--- Step 3: Generating Data Manifest ---")
        manifest = {}
        files_to_manifest = []
        for dirpath, dirnames, filenames in os.walk(self.config.BASE_DATA_DIR):
            # Add directories (like .parquet) to the manifest
            for d in dirnames:
                files_to_manifest.append(Path(dirpath) / d)
            # Add files
            for f in filenames:
                files_to_manifest.append(Path(dirpath) / f)

        print(f"  - Generating checksums for {len(files_to_manifest)} items...")
        huge_files_to_skip_checksum = {
            "uniref50.fasta", "idmapping.dat", "id_mapping.parquet",
            "negative_interactions.parquet", "positive_interactions.parquet"
        }
        for file_path in tqdm(files_to_manifest, desc="  Calculating Checksums"):
            relative_path = file_path.relative_to(self.config.PROJECT_ROOT)
            if file_path.is_file():
                checksum = "skipped_due_to_size" if file_path.name in huge_files_to_skip_checksum else DataProcessor._calculate_sha256(file_path)
                manifest[relative_path.as_posix()] = {
                    'type': 'file',
                    'size': file_path.stat().st_size,
                    'sha256': checksum
                }
            elif file_path.is_dir():
                 manifest[relative_path.as_posix()] = {
                    'type': 'directory',
                    'size': sum(f.stat().st_size for f in file_path.glob('**/*') if f.is_file()),
                    'sha256': "skipped_for_directory"
                }

        manifest_path = self.config.DATA_MANIFEST_PATH
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"  - ✅ Manifest saved to: {manifest_path}")
        except IOError as e:
            print(f"  - ❌ ERROR: Could not save manifest file: {e}")
            return

    def _cleanup_intermediate_files(self):
        """Removes all downloaded and intermediate raw files."""
        print("\n--- Step 4: Cleaning Up Intermediate Files ---")
        for f_path in set(self.files_to_cleanup):
            if f_path.exists():
                try:
                    if f_path.is_dir():
                        shutil.rmtree(f_path)
                    else:
                        f_path.unlink()
                    print(f"  - Cleaned up: {f_path.name}")
                except OSError as e:
                    print(f"  - WARNING: Could not clean up file {f_path.name}. Error: {e}")


def setup_data(config):
    """
    Public entry point to instantiate and run the DataManager.
    """
    DataManager(config).run_full_setup()
