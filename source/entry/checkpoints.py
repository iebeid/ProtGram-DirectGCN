# ==============================================================================
# MODULE: source/entry/checkpoints.py
# PURPOSE: Handles saving, loading, and validating pipeline step checkpoints with checksums.
# VERSION: 1.1 (Added checksum validation)
# AUTHOR: Islam Ebeid
# ==============================================================================
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from source.utils.fs.file_utils import FileUtils
from source.utils.fs.file_system_manager import fs_manager


class CheckpointManager:
    """Handles saving, loading, and validating pipeline step checkpoints."""

    def __init__(self, checkpoint_dir_uri: str):
        self.checkpoint_dir_uri = checkpoint_dir_uri
        # --- FIX: Use the filesystem manager to handle local or cloud paths ---
        self.fs, self.checkpoint_dir_path = fs_manager.get_fs_and_path(self.checkpoint_dir_uri)
        self.fs.makedirs(self.checkpoint_dir_path, exist_ok=True)
        self.checkpoint_file_path = os.path.join(self.checkpoint_dir_path, "pipeline_checkpoint.json")
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Loads the checkpoint JSON file from disk."""
        # --- FIX: Use the filesystem manager to check for existence and open the file ---
        if self.fs.exists(self.checkpoint_file_path):
            try:
                with self.fs.open(self.checkpoint_file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Checkpoint file is corrupted. Starting fresh.")
                return {}
        return {}

    def _make_serializable_paths_to_str(self, data: Any) -> Any:
        """Recursively converts Path objects in data structures to strings."""
        if isinstance(data, Path):
            return str(data)
        if isinstance(data, dict):
            return {k: self._make_serializable_paths_to_str(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._make_serializable_paths_to_str(i) for i in data]
        return data

    def save_checkpoint(self, step_name: str, data: Any):
        """Saves the output data for a specific pipeline step, including file checksums."""
        serializable_data = self._make_serializable_paths_to_str(data)

        # Add checksums for file-based checkpoints
        if isinstance(serializable_data, list):
            for item in serializable_data:
                if isinstance(item, dict) and "path" in item:
                    # FileUtils.calculate_sha256 is already cloud-aware and expects a URI
                    item["sha256"] = FileUtils.calculate_sha256(item["path"])

        self.data[step_name] = serializable_data
        # --- DEFINITIVE FIX: Implement atomic write to prevent corruption ---
        # Write to a temporary file first, then rename. This ensures the main
        # checkpoint file is never in a partially-written, corrupted state.
        temp_checkpoint_path = self.checkpoint_file_path + ".tmp"
        try:
            with self.fs.open(temp_checkpoint_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            # The rename operation is atomic on most filesystems
            self.fs.rename(temp_checkpoint_path, self.checkpoint_file_path)
        finally:
            # Ensure the temporary file is cleaned up on success or failure
            if self.fs.exists(temp_checkpoint_path):
                self.fs.rm(temp_checkpoint_path)
        print(f"  ✅ Checkpoint saved for step: '{step_name}'")

    def get_checkpoint(self, step_name: str) -> Optional[Any]:
        """Retrieves the data for a specific step if it exists and is valid."""
        checkpoint_data = self.data.get(step_name)
        if not checkpoint_data:
            print(f"  - No checkpoint found for step: '{step_name}'.")
            return None

        # --- NEW: Validate file-based checkpoints using checksums ---
        files_to_validate = []
        if isinstance(checkpoint_data, list):
            # This handles the format [{"name": "x", "path": "y", "sha256": "z"}, ...]
            for item in checkpoint_data:
                if isinstance(item, dict) and "path" in item and "sha256" in item:
                    files_to_validate.append(item)
        elif isinstance(checkpoint_data, dict) and "path" in checkpoint_data and "sha256" in checkpoint_data:
            # --- ANTICIPATORY DEBUGGING: Handle checkpoints that are a single file dictionary ---
            files_to_validate.append(checkpoint_data)
        elif isinstance(checkpoint_data, dict) and "status" in checkpoint_data:
            # This handles simple status checkpoints like {"status": "completed"}
            print(f"  ✅ Found valid status checkpoint for step: '{step_name}'. Skipping execution.")
            return checkpoint_data

        if not files_to_validate:
            # If it's not a file list and not a status dict, it's some other valid data.
            print(f"  ✅ Found valid, non-file checkpoint for step: '{step_name}'. Skipping execution.")
            return checkpoint_data

        for file_info in files_to_validate:
            file_uri = file_info["path"]
            expected_checksum = file_info["sha256"]
            # --- FIX: Use the filesystem manager to validate the file at its URI ---
            file_fs, file_path_str = fs_manager.get_fs_and_path(file_uri)

            if not file_fs.exists(file_path_str) or file_fs.info(file_path_str)['size'] == 0:
                print(f"  - Checkpoint for '{step_name}' is INVALID. File '{os.path.basename(file_path_str)}' is missing or empty. Re-running.")
                return None

            actual_checksum = FileUtils.calculate_sha256(file_uri)
            if actual_checksum != expected_checksum:
                print(f"  - Checkpoint for '{step_name}' is INVALID. Checksum mismatch for '{os.path.basename(file_path_str)}'. Re-running.")
                return None

        print(f"  ✅ Found valid file-based checkpoint for step: '{step_name}'. Skipping execution.")
        return checkpoint_data
