#!/bin/bash

# ==============================================================================
# SCRIPT: clean.sh
# PURPOSE: Removes all generated output, results, logs, and cache files from
#          the project directory to ensure a completely clean run.
# WARNING: This is a destructive action for the 'results' directory.
# VERSION: 1.0
# ==============================================================================

echo "--- Cleaning project directories and cache files ---"
echo "  - Removing 'results' directory..."
rm -rf results
echo "  - Removing 'mlruns' directory..."
rm -rf mlruns
echo "  - Removing local ID map cache files (*_map_cache.pkl)..."
rm -f *_map_cache.pkl
echo "SUCCESS: Project directories cleaned."