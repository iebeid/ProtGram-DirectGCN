# ==============================================================================
# MODULE: source/utils/data/data_downloader.py
# PURPOSE: Handles downloading and post-processing of data files with network resilience.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

import requests
import time
import shutil
import gzip
import zipfile
import os
from pathlib import Path
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data.data_utils import DataUtils


class DataDownloader:
    """Handles downloading and post-processing of data files with network resilience."""

    def __init__(self, config: Config):
        self.config = config

    def _download_with_retries(self, url: str, destination: Path, max_retries: int = 5, initial_backoff: float = 1.0):
        """
        Downloads a file with retries and exponential backoff.
        Includes a progress bar and streams the download to handle large files.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_suffix(destination.suffix + '.part')
        backoff_time = initial_backoff

        for attempt in range(max_retries):
            try:
                print(f"    Attempt {attempt + 1}/{max_retries} to download from {url}...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    with open(temp_destination, 'wb') as f, tqdm(
                        total=total_size, unit='iB', unit_scale=True,
                        desc=f"      Downloading {destination.name}", leave=False
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                shutil.move(temp_destination, destination)
                print(f"    ✅ Download successful: {destination.name}")
                return True
            except (requests.exceptions.RequestException, IOError) as e:
                print(f"    - Download attempt failed: {e}")
                if attempt + 1 == max_retries:
                    print(f"    ❌ Download failed after {max_retries} attempts. Please check the URL and your network connection.")
                    if temp_destination.exists():
                        os.remove(temp_destination)
                    return False

                print(f"    Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
        return False

    def _post_process(self, filepath: Path, instruction: str):
        """Handles unzipping or ungzipping files."""
        print(f"    Post-processing '{filepath.name}' with instruction: '{instruction}'...")
        output_path = filepath.with_suffix('')

        try:
            if instruction == 'ungzip':
                with gzip.open(filepath, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"      Un-gzipped to: {output_path.name}")
                os.remove(filepath)
            elif instruction == 'unzip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    file_to_extract = max(zip_ref.infolist(), key=lambda z: z.file_size)
                    extracted_path = zip_ref.extract(file_to_extract, path=filepath.parent)
                    shutil.move(extracted_path, output_path)
                print(f"      Unzipped '{file_to_extract.filename}' to: {output_path.name}")
                os.remove(filepath)
        except Exception as e:
            print(f"    ❌ Error during post-processing of {filepath.name}: {e}")

    def run(self):
        """Iterates through data sources, downloading and processing them as needed."""
        DataUtils.print_header("Verifying and Downloading Required Data")
        for source_name, source_info in self.config.DATA_SOURCES.items():
            if source_info.get('type') == 'file' and 'url' in source_info and 'path' in source_info:
                filepath = Path(source_info['path'])
                final_path = filepath.with_suffix('') if source_info.get('post_process') else filepath

                if not final_path.exists():
                    print(f"\n  - File for '{source_name}' not found. Starting download process.")
                    if self._download_with_retries(source_info['url'], filepath):
                        if 'post_process' in source_info and source_info['post_process']:
                            self._post_process(filepath, source_info['post_process'])
                else:
                    print(f"  - File for '{source_name}' already exists at '{final_path}'. Skipping.")