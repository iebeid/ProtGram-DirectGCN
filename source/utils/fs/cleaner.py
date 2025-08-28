# ==============================================================================
# MODULE: source/utils/fs/cleaner.py
# PURPOSE: Handles the programmatic cleaning of project directories.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

import shutil
from pathlib import Path


class ProjectCleaner:
    """A utility to programmatically clean project output directories."""

    @staticmethod
    def clean_project(project_root: Path):
        """
        Replicates the logic of clean.sh to ensure a fresh run state.
        This is called by the main entry point to make the app self-contained.
        """
        print("--- Cleaning previous run's outputs to ensure a fresh start... ---")

        results_dir = project_root / "results"
        mlruns_dir = project_root / "mlruns"
        mappings_dir = project_root / "data" / "mappings"
        ground_truth_dir = project_root / "data" / "ground_truth"

        # 1. Clean results directory, preserving logs
        if results_dir.exists():
            print(f"  - Removing previous run's output directories (excluding logs) from '{results_dir}'...")
            for path in results_dir.iterdir():
                if path.is_dir() and path.name != "logs":
                    shutil.rmtree(path)

        # 2. Clean MLflow directory
        if mlruns_dir.exists():
            print(f"  - Removing '{mlruns_dir}' directory...")
            shutil.rmtree(mlruns_dir)

        # 3. Clean Python bytecode cache
        print("  - Removing Python bytecode cache (__pycache__, *.pyc)...")
        for path in project_root.rglob("__pycache__"):
            if path.is_dir():
                shutil.rmtree(path)
        for path in project_root.rglob("*.pyc"):
            path.unlink(missing_ok=True)

        print("SUCCESS: Project directories cleaned.")