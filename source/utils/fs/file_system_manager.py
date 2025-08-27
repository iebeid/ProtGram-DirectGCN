# ==============================================================================
# MODULE: source/utils/fs/file_system_manager.py
# PURPOSE: Provides a unified interface for interacting with different file systems.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

import os
from typing import Tuple, Any
import fsspec


class FileSystemManager:
    """
    A singleton manager to provide a unified interface to different filesystems
    (local, S3, GCS, etc.) using fsspec. This allows the application to be
    storage-agnostic.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileSystemManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # fsspec can automatically use environment variables for credentials
        # (e.g., AWS_ACCESS_KEY_ID, GOOGLE_APPLICATION_CREDENTIALS).
        # You can also specify them here if needed.
        self.storage_options = {}
        self._initialized = True

    @staticmethod
    def get_protocol(uri: str) -> str:
        """Robustly determines the protocol from a given URI (e.g., 's3', 'gcs', 'file')."""
        # --- REFACTOR: Use fsspec's internal utility for more robust protocol detection ---
        # This correctly handles local Windows paths (e.g., "C:\...") unlike urlparse.
        return fsspec.utils.get_protocol(str(uri))

    def get_fs_and_path(self, uri: str) -> Tuple[Any, str]:
        """
        A convenience method to get both the filesystem object and the
        protocol-stripped path string from a URI.
        """
        # --- REFACTOR: Use fsspec.open() for idiomatic URI parsing ---
        # This is the most robust way to get the filesystem and path, as it
        # correctly handles all URI schemes and edge cases supported by fsspec.
        open_file = fsspec.open(str(uri), **self.storage_options)
        return open_file.fs, open_file.path

# Create a global instance for easy access throughout the application
fs_manager = FileSystemManager()