# ==============================================================================
# MODULE: data.py
# PURPOSE: Handles verification and automatic download of required data files.
# AUTHOR: Islam Ebeid
# ==============================================================================

import gzip
import hashlib
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from configuration.config import Config


class DataManager:
    """
    Verifies that all required data files defined in the config are present,
    and downloads them if they are missing or corrupt.
    """

    def __init__(self, config: Config):
        self.config = config
        self.base_data_dir = config.BASE_DATA_DIR

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculates the SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verifies the file's checksum."""
        if not expected_checksum or not expected_checksum.startswith("sha256:"):
            return True  # No checksum to verify or format is incorrect

        print(f"Verifying checksum for {file_path.name}...")
        expected_hash = expected_checksum.split(":", 1)[1]
        actual_hash = self._calculate_sha256(file_path)

        if actual_hash == expected_hash:
            print("Checksum OK.")
            return True
        else:
            print(f"Checksum mismatch for {file_path.name}!")
            print(f"  - Expected: {expected_hash}")
            print(f"  - Got:      {actual_hash}")
            return False

    def _download_file(self, url: str, dest_path: Path):
        """Downloads a file with a progress bar."""
        print(f"Downloading from {url} to {dest_path}...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, desc=dest_path.name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Clean up partial download
            raise

    def _post_process_file(self, downloaded_path: Path, final_path: Path, method: str):
        """Applies post-processing like unzipping."""
        if not method:
            return

        if method == 'ungzip':
            print(f"Decompressing {downloaded_path.name} to {final_path.name}...")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(downloaded_path, 'rb') as f_in:
                with open(final_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            downloaded_path.unlink()  # Remove the .gz file
        else:
            print(f"Warning: Unknown post_process method '{method}'")

    def run_check(self):
        """
        Main method to iterate through required files, check existence, and download if needed.
        """
        print("\n--- Running Data Verification and Download ---")
        if not hasattr(self.config, 'DATA_SOURCES'):
            print("DATA_SOURCES not defined in config. Skipping.")
            return

        for key, source_info in self.config.DATA_SOURCES.items():
            final_path = source_info.get('path')
            if not final_path:
                print(f"Warning: No 'path' defined for data source '{key}'. Skipping.")
                continue

            checksum = source_info.get('checksum')

            if final_path.exists() and self._verify_checksum(final_path, checksum):
                print(f"☑ Found and verified: {final_path.relative_to(self.config.PROJECT_ROOT)}")
                continue

            if final_path.exists():
                print(f"File {final_path.name} exists but is invalid. Re-downloading.")
                final_path.unlink()

            url = source_info.get('url')
            if not url:
                print(f"⚠ File missing and no URL provided for '{key}': {final_path}")
                continue

            post_process = source_info.get('post_process')
            download_target_path = final_path.with_suffix(f"{final_path.suffix}.gz") if post_process == 'ungzip' else final_path

            try:
                self._download_file(url, download_target_path)
                self._post_process_file(download_target_path, final_path, post_process)

                if not self._verify_checksum(final_path, checksum):
                    print(f"Error: Checksum failed for downloaded file: {final_path}. Please check URL/checksum in config.")
                else:
                    print(f"✔ Successfully acquired: {final_path.relative_to(self.config.PROJECT_ROOT)}")

            except Exception as e:
                print(f"Failed to acquire file for '{key}'. Error: {e}")

        print("--- Data Verification Complete ---\n")


def setup_data(config: Config):
    """Convenience function to instantiate and run the data manager."""
    manager = DataManager(config)
    manager.run_check()