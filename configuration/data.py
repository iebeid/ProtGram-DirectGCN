# ==============================================================================
# MODULE: configuration/data.py
# PURPOSE: Handles the verification and acquisition of all external data files.
# VERSION: 2.0 (Added gdown support for Google Drive URLs)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gzip
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

import shutil
from pathlib import Path

import requests
from tqdm.auto import tqdm

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
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle potential BOM
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
            with open(file_path, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle potential BOM
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:  # Find first non-empty line
                        if not stripped_line.startswith('>'):
                            print(f"  - Validation failed for {file_path.name}: Does not start with '>'.")
                            return False
                        break  # Found a header, it's probably fine
        except Exception:
            return False  # Not a valid text-based FASTA

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

        # 1. Check if the final, processed file already exists and is valid.
        if _is_file_valid(final_path):
            print(f"☑ Found and verified: {final_path.relative_to(config.PROJECT_ROOT)}")
            continue
        # If the file is not valid, we proceed. The logic below will handle
        # overwriting it via decompression or re-downloading.

        if not final_path.exists() and download_path.exists() and source_info.get('post_process') == 'ungzip':
            print(f"Found compressed file '{download_path.name}'. Attempting to decompress...")
            try:
                with gzip.open(download_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                # Re-validate after decompression to ensure integrity
                if _is_file_valid(final_path):
                    print(f"✔ Successfully acquired: {final_path.relative_to(config.PROJECT_ROOT)}")
                    continue  # Success, move to the next file in the loop
                else:
                    print(f"  Warning: Decompressed file '{final_path.name}' failed validation. Attempting re-download.")

            except (gzip.BadGzipFile, EOFError) as e:
                print(f"  Warning: Decompression failed for '{download_path.name}' (likely corrupt). Error: {e}. Attempting re-download.")
            # If we reach here, it means decompression failed or the result was invalid, so we fall through to the download logic.

        # 3. If neither exists, attempt to download.
        url = source_info.get('url')
        if not url or 'YOUR_FILE_ID' in url:
            print(f"❓ Skipped '{key}': URL is a placeholder or not provided.")
            continue

        try:
            # --- FIX: Use gdown for Google Drive URLs, requests for others ---
            if 'drive.google.com' in url:
                if not GDOWN_AVAILABLE:
                    print(f"  ERROR: URL for '{key}' is a Google Drive link, but 'gdown' is not installed. Please install it (`pip install gdown`). Skipping.")
                    continue
                print(f"Downloading '{download_path.name}' from Google Drive...")
                gdown.download(url, str(download_path), quiet=False)
            else:
                print(f"Downloading from {url} to {download_path}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(download_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True,
                                                          desc=download_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # After a successful download, handle any post-processing
            if source_info.get('post_process') == 'ungzip':
                print(f"Decompressing {download_path.name} to {final_path.name}...")
                with gzip.open(download_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                if download_path.exists():
                    download_path.unlink()

            # Final validation check
            if _is_file_valid(final_path):
                print(f"✔ Successfully acquired and verified: {final_path.relative_to(config.PROJECT_ROOT)}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            print(f"Failed to acquire file for '{key}'. Error: {e}")
    print("--- Data Verification Complete ---")
