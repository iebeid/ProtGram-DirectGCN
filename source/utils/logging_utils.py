# ==============================================================================
# MODULE: logging_utils.py
# PURPOSE: Provides a simple utility to redirect stdout/stderr to a log file.
# AUTHOR: Islam Ebeid
# ==============================================================================

import sys
from datetime import datetime
from pathlib import Path


class Tee:
    """
    A file-like object that redirects write calls to multiple streams.
    This allows printing to both the console and a file simultaneously.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if f:
                f.write(obj)
                f.flush()  # Ensure output is written immediately

    def flush(self):
        for f in self.files:
            if f:
                f.flush()

    def isatty(self):
        # tqdm checks this to decide whether to draw a progress bar.
        # We delegate this to the original stdout to preserve progress bars.
        return self.files[0].isatty() if self.files else False


# --- Global variables to manage logging state ---
log_file_handler = None
original_stdout = None
original_stderr = None


def start_logging(log_dir: Path):
    """
    Redirects stdout and stderr to both the console and a timestamped log file.
    """
    global log_file_handler, original_stdout, original_stderr

    if original_stdout is not None:
        print("Warning: Logging is already started.")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_dir / f"run_{timestamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handler = open(log_file_path, 'w', encoding='utf-8')

    sys.stdout = Tee(original_stdout, log_file_handler)
    sys.stderr = Tee(original_stderr, log_file_handler)

    print(f"--- Logging all console output to: {log_file_path} ---")


def stop_logging():
    """
    Restores stdout and stderr to their original configurations.
    """
    global original_stdout, original_stderr, log_file_handler
    if original_stdout:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file_handler:
            log_file_handler.close()
        original_stdout, original_stderr, log_file_handler = None, None, None
        print("--- File logging stopped. Console output is back to normal. ---")