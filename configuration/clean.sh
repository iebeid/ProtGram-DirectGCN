#!/bin/bash

# ==============================================================================
# SCRIPT: clean.sh
# PURPOSE: Removes all generated output, results, logs, and cache files from
#          the project directory to ensure a completely clean run.
# WARNING: This is a destructive action for the 'results' directory.
# VERSION: 1.0
# ==============================================================================

# --- Get the directory of this script, then find the project root (one level up) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "--- Cleaning project directories and cache files ---"
echo "  - Removing '$PROJECT_ROOT/results' directory..."
rm -rf "$PROJECT_ROOT/results"
echo "  - Removing '$PROJECT_ROOT/mlruns' directory..."
rm -rf "$PROJECT_ROOT/mlruns"
echo "  - Removing local ID map cache files from project root..."
rm -f "$PROJECT_ROOT"/*_map_cache.pkl
echo "SUCCESS: Project directories cleaned."