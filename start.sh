#!/bin/bash

# ==============================================================================
# SCRIPT: start.sh
# PURPOSE: Updates the existing project repository and runs the main application.
#          This script is non-destructive and assumes the environment has been
#          set up at least once by running 'reset.sh'.
# USAGE:   Run this script from WITHIN the project's root directory.
#          Example: cd /path/to/ProtGram-DirectGCN && bash start.sh
# VERSION: 2.0 (Simplified to remove Git LFS and data backup logic)
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
# --- Stash local changes, pull, and restore stash ---
echo "INFO: Stashing any local changes to prevent conflicts..."
git stash push -m "start.sh-autostash-$(date +%s)" > /dev/null 2>&1 || true
echo "INFO: Pulling latest changes from the remote repository..."
git pull --rebase
echo "INFO: Restoring any stashed local changes..."
git stash pop > /dev/null 2>&1 || echo "INFO: No local changes to restore."
echo "SUCCESS: Project repository is up to date."

# --- Step 3: Run the Main Application ---
echo -e "\n--- STEP 3: Executing the main application via run.py ---"
echo "INFO: The 'run.py' script will automatically validate local data and restore from cache if needed."

# --- CRITICAL: Export the Conda environment's library path. ---
# This ensures that TensorFlow and other programs can find the CUDA libraries (.so files)
# that were installed by Conda. This resolves the "Cannot dlopen" errors at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- NEW FIX for HDF5 File Locking Issues ---
# On some network filesystems (like NFS), file locking can cause errors.
# This environment variable disables it for the HDF5 library.
export HDF5_USE_FILE_LOCKING=FALSE

# --- DEFINITIVE FIX for Reproducibility: Configure CUDA workspace ---
# This environment variable is required by `torch.use_deterministic_algorithms(True)`
# to ensure that operations like `index_add` (used by PyG's scatter_add) are deterministic.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- CRITICAL for XLA/JIT: Point TensorFlow's XLA compiler to the Conda CUDA toolkit. ---
# This resolves the "libdevice not found" and "JIT compilation failed" errors when
# running Transformer models on the GPU.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0