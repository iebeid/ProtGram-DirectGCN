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

def _is_file_valid(file_path: Path) -> bool:
    """
    Performs a basic integrity check on a file beyond just existence to prevent
    using corrupt or incorrect files (e.g., HTML error pages).
    """
    # Check for existence and a minimal size to avoid empty files
    if not file_path.exists() or file_path.stat().st_size < 10:
        return False

    file_type = file_path.suffix.lower()

    # Check for HTML content, a common issue with bad downloads
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_chunk = f.read(1024)
            if first_chunk.strip().lower().startswith(('<!doctype html', '<html')):
                print(f"  - Validation failed for {file_path.name}: File appears to be an HTML document.")
                return False
    except Exception:
        # This is likely a binary file (like .h5 or .gz), which is fine.
        # The read will fail, so we can proceed with the assumption it's not a text-based error page.
        pass

    # FASTA-specific check
    if file_type == '.fasta':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:  # Find first non-empty line
                        if not stripped_line.startswith('>'):
                            print(f"  - Validation failed for {file_path.name}: Does not start with '>'.")
                            return False
                        break  # Found a header, it's probably fine
        except Exception:
            return False # Not a valid text-based FASTA

    # If all checks pass, the file is considered valid
    return True


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
        if _is_file_valid(final_path):
            print(f"☑ Found and verified: {final_path.relative_to(config.PROJECT_ROOT)}")
            continue
        elif final_path.exists():
            # File exists but is invalid (e.g., empty or HTML). Delete it to trigger re-download.
            print(f"⚠ Found invalid or corrupt file at '{final_path.name}'. Deleting and re-downloading.")
            try:
                final_path.unlink()
            except OSError as e:
                print(f"  Error deleting corrupt file: {e}. Skipping this file.")
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