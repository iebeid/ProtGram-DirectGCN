# ==============================================================================
# MODULE: utils/logging.py
# PURPOSE: Provides a class-based utility to redirect stdout/stderr to a log file.
# VERSION: 2.0 (Corrected path handling and context management)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, IO, Text


class FileLogger:
    """
    A logger that redirects stdout and stderr to a file and the console.
    Designed to be used as a context manager to ensure logging is
    always stopped correctly.

    Usage:
        logger = FileLogger(log_dir="/path/to/logs", enabled=True)
        with logger:
            print("This will be logged to console and file.")
        # Logging is automatically stopped here.
    """

    class _Tee:
        """
        A file-like object that redirects write calls to multiple streams.
        This allows printing to both the console and a file simultaneously.
        """

        def __init__(self, *files):
            self.files = files

        def write(self, obj: Text):
            # --- FIX for tqdm progress bar clutter in logs ---
            # tqdm uses carriage returns ('\r') to update a line in-place.
            # We detect these updates and only write them to TTY streams (the console),
            # not to the log file, which prevents repeated lines in the log.
            is_progress_bar_update = '\r' in obj

            for f in self.files:
                if f:
                    # If it's a progress bar update, only write it to TTYs.
                    # The log file handler's isatty() will be False.
                    if is_progress_bar_update and not f.isatty():
                        continue
                    f.write(obj)
                    f.flush()  # Ensure output is written immediately
            # --- END FIX ---

        def flush(self):
            for f in self.files:
                if f:
                    f.flush()

        def isatty(self) -> bool:
            # tqdm checks this to decide whether to draw a progress bar.
            # We delegate this to the original stdout to preserve progress bars.
            return self.files[0].isatty() if self.files else False

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        self.log_file_handler: Optional[IO[str]] = None
        self.original_stdout: Optional[IO[str]] = None
        self.original_stderr: Optional[IO[str]] = None

    def start(self):
        """
        Redirects stdout and stderr to both the console and a timestamped log file.
        """
        if not self.enabled:
            return

        if self.original_stdout is not None:
            print("Warning: Logging is already started.")
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = self.log_dir / f"run_{timestamp}.log"

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file_handler = open(log_file_path, 'w', encoding='utf-8')

        sys.stdout = self._Tee(self.original_stdout, self.log_file_handler)
        sys.stderr = self._Tee(self.original_stderr, self.log_file_handler)

        print(f"--- Logging all console output to: {log_file_path} ---")

    def stop(self):
        """
        Restores stdout and stderr to their original configurations.
        """
        if not self.enabled or self.original_stdout is None:
            return

        # Check if streams have already been restored to prevent errors
        if sys.stdout is self.original_stdout:
            return

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file_handler:
            self.log_file_handler.close()

        # Reset state
        self.original_stdout = None
        self.original_stderr = None
        self.log_file_handler = None
        print("--- File logging stopped. Console output is back to normal. ---")

    def __enter__(self):
        """Starts logging when entering a 'with' block."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops logging when exiting a 'with' block."""
        self.stop()