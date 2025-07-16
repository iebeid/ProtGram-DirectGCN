# ==============================================================================
# MODULE: configuration/data.py
# PURPOSE: Handles the verification and acquisition of all external data files.
# VERSION: 1.0
# AUTHOR: Your Name (Integrated by Coding Partner)
# ==============================================================================

import gzip
import os
import shutil
import requests
from tqdm.auto import tqdm
from pathlib import Path

from configuration.config import Config


def setup_data(config: Config):
    """
    Checks for the existence of required data files, downloading them if necessary.
    Handles decompression and other post-processing steps.
    """
    print("\n--- Running Data Verification and Download ---")

    for key, source_info in config.DATA_SOURCES.items():
        final_path = Path(source_info['path'])

        # Determine the potential path of the downloaded (possibly compressed) file
        download_path = Path(str(final_path) + ".gz") if source_info.get('post_process') == 'ungzip' else final_path

        # 1. Check if the final, processed file already exists.
        if final_path.exists():
            print(f"☑ Found and verified: {final_path.relative_to(config.PROJECT_ROOT)}")
            continue

        # 2. Check if the compressed file exists but the final one doesn't.
        if not final_path.exists() and download_path.exists() and source_info.get('post_process') == 'ungzip':
            print(f"Decompressing {download_path.name} to {final_path.name}...")
            with gzip.open(download_path, 'rb') as f_in:
                with open(final_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✔ Successfully acquired: {final_path.relative_to(config.PROJECT_ROOT)}")
            continue

        # 3. If neither exists, attempt to download.
        url = source_info.get('url')
        if not url or 'example.com' in url:
            print(f"❓ Skipped '{key}': URL is a placeholder or not provided.")
            continue

        print(f"Downloading from {url} to {download_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(download_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=download_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Recursively call setup_data to handle the post-processing of the newly downloaded file
            setup_data({key: source_info})

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            print(f"Failed to acquire file for '{key}'. Error: {e}")

    print("--- Data Verification Complete ---")