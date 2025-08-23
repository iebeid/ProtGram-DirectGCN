# ==============================================================================
# MODULE: source/utils/fs/file_system_manager.py
# PURPOSE: Provides a unified interface for interacting with different file systems.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

import os
from typing import Tuple, Any
import fsspec
from urllib.parse import urlparse


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
        protocol = self.get_protocol(uri)
        fs = fsspec.filesystem(protocol, **self.storage_options.get(protocol, {}))
        path = str(uri).split('://', 1)[-1] if '://' in uri else str(uri)
        return fs, path

# Create a global instance for easy access throughout the application
fs_manager = FileSystemManager()