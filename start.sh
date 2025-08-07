#!/bin/bash

# ==============================================================================
# SCRIPT: start.sh
# PURPOSE: Updates the existing project repository and runs the main application.
#          This script is non-destructive and assumes the environment has been
#          set up at least once by running 'reset.sh'.
# USAGE:   Run this script from WITHIN the project's root directory.
#          Example: cd /path/to/ProtGram-DirectGCN && bash start.sh
# VERSION: 1.1
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pre-flight Check: Ensure we are in the project root ---
if [ ! -f "run.py" ] || [ ! -d "configuration" ]; then
    echo "ERROR: This script must be run from the root of the 'ProtGram-DirectGCN' project directory."
    echo "Please 'cd' into the project directory and run it again."
    exit 1
fi
echo "INFO: Project root directory verified."

# --- Configuration ---
ENV_NAME="ppi-env"

# --- Step 1: Find and Activate Conda Environment ---
echo -e "\n--- STEP 1: Activating Conda Environment '$ENV_NAME' ---"
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not find Conda base directory. Is Conda installed?"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "ERROR: The Conda environment '$ENV_NAME' was not found."
    echo "Please run the full 'reset.sh' script from the parent directory to set up the environment first."
    exit 1
fi
conda activate "$ENV_NAME"
echo "SUCCESS: Environment '$ENV_NAME' activated."
python --version

# --- Step 2: Update the Repository ---
echo -e "\n--- STEP 2: Updating the project with the latest changes from Git ---"

# --- FIX: Use a safer pull method to prevent accidental deletion of local files ---
# Stashing any local changes, pulling, and then popping the stash is more robust.
echo "INFO: Stashing any local changes to prevent conflicts..."
git stash push -m "start.sh-autostash-$(date +%s)"
echo "INFO: Pulling latest changes from the remote repository..."
git pull --rebase
echo "INFO: Restoring any stashed local changes..."
# Pop the stash. If it fails (e.g., nothing to pop), it won't stop the script.
git stash pop || echo "INFO: No local changes to restore."

# --- NEW: Interactively handle git lfs pull ---
SKIP_LFS=false
if [ -d "data" ]; then
    echo -e "\n--------------------------------------------------"
    echo "The 'data' directory already exists. Current contents:"
    ls -lh data
    echo "--------------------------------------------------"
    read -r -p "Do you want to skip 'git lfs pull' to save time? (y/n): " response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        echo "INFO: Skipping 'git lfs pull' as requested."
        SKIP_LFS=true
    fi
fi

if [ "$SKIP_LFS" = false ]; then
    echo "INFO: Running 'git lfs pull' to update large files..."
    git lfs pull
fi
echo "SUCCESS: Project repository is up to date."

# --- Step 3: Run the Main Application ---
echo -e "\n--- STEP 3: Executing the main application via run.py ---"
echo "INFO: The 'run.py' script will automatically validate the environment."
echo "INFO: If validation fails, you may be prompted to run the setup again."

# --- CRITICAL: Export the Conda environment's library path. ---
# This ensures that TensorFlow and other programs can find the CUDA libraries (.so files)
# that were installed by Conda. This resolves the "Cannot dlopen" errors at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- CRITICAL for XLA/JIT: Point TensorFlow's XLA compiler to the Conda CUDA toolkit. ---
# This resolves the "libdevice not found" and "JIT compilation failed" errors when
# running Transformer models on the GPU.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0