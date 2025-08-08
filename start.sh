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
# --- DEFINITIVE FIX: Temporarily move the 'data' directory to protect it from all git operations ---
DATA_DIR_EXISTS=false
TEMP_BACKUP_PATH="../$(basename "$PWD")_data_backup"
if [ -d "data" ]; then
    # --- FIX: Make backup non-destructive. Rename old backup instead of deleting. ---
    if [ -d "$TEMP_BACKUP_PATH" ]; then
        TIMESTAMP=$(date +%s)
        echo "WARN: Found an old temporary backup directory. Renaming it to '$TEMP_BACKUP_PATH-$TIMESTAMP' to prevent data loss."
        mv "$TEMP_BACKUP_PATH" "$TEMP_BACKUP_PATH-$TIMESTAMP"
    fi
    echo "INFO: Temporarily moving existing 'data' directory to a safe location..."
    mv data "$TEMP_BACKUP_PATH"
    DATA_DIR_EXISTS=true
    echo "INFO: 'data' directory backed up to '$TEMP_BACKUP_PATH'."
fi

# --- Stash local changes, pull, and restore stash ---
echo "INFO: Stashing any local changes to prevent conflicts..."
git stash push -m "start.sh-autostash-$(date +%s)" > /dev/null 2>&1 || true
echo "INFO: Pulling latest changes from the remote repository..."
git pull --rebase
echo "INFO: Restoring any stashed local changes..."
git stash pop > /dev/null 2>&1 || echo "INFO: No local changes to restore."

# --- NEW: Handle self-update ---
# Check if the start.sh script itself was updated by the pull.
# If so, re-execute it to ensure the latest logic is used.
# This will cause the script to restart from the top, which is safe.
if (git diff --name-only HEAD@{1} HEAD | grep -q "start.sh"); then
    echo "INFO: The start.sh script has been updated. Re-executing with the new version..."
    # Restore data before re-executing to avoid issues on the next run's backup step.
    if [ "$DATA_DIR_EXISTS" = true ]; then
        if [ -d "data" ]; then rm -rf data; fi
        mv "$TEMP_BACKUP_PATH" data
    fi
    exec bash "$0" "$@"
fi

# --- Interactively handle git lfs pull ---
SKIP_LFS=false
if [ "$DATA_DIR_EXISTS" = true ]; then
    echo -e "\n--------------------------------------------------"
    echo "A local 'data' directory was found and has been protected."
    echo "Current contents of your protected 'data' directory:"
    ls -lh "$TEMP_BACKUP_PATH"
    echo "--------------------------------------------------"
    # --- FIX: Clarify the purpose of the git lfs pull prompt ---
    echo "'git lfs pull' downloads large data files (e.g., ground truth interactions)."
    echo "If you are sure your local data files are up-to-date, you can skip this to save time."
    read -r -p "Skip 'git lfs pull'? (y/n): " response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        echo "INFO: Skipping 'git lfs pull' as requested."
        SKIP_LFS=true
    fi
fi

if [ "$SKIP_LFS" = false ]; then
    echo "INFO: Running 'git lfs pull' to update large files..."
    # --- FIX: Add robust error handling for git lfs pull ---
    # 'git lfs pull' can exit with code 0 even if errors occur (like 'Scanner error').
    # We capture the output to check for errors manually and attempt an automatic fix.
    if ! LFS_OUTPUT=$(git lfs pull 2>&1); then
        # This block catches non-zero exit codes, which are less common for this specific error.
        echo "$LFS_OUTPUT"
        echo "ERROR: 'git lfs pull' failed with a non-zero exit code. Aborting."
        exit 1
    fi
    echo "$LFS_OUTPUT" # Print the original output for the user
    if echo "$LFS_OUTPUT" | grep -q -i "error"; then
        echo "WARN: 'git lfs pull' reported errors. This can happen if the LFS cache is inconsistent."
        echo "INFO: Attempting a more forceful fetch with 'git lfs fetch --all' to try and fix this..."
        git lfs fetch --all && git lfs checkout
    fi
fi

# --- Restore the 'data' directory ---
if [ "$DATA_DIR_EXISTS" = true ]; then
    # If git lfs pull created a new data directory with pointers, remove it first.
    if [ -d "data" ]; then
        echo "INFO: A new 'data' directory with LFS pointers was created by the pull."
        echo "      This will be safely removed before restoring your local data."
        rm -rf data
    fi
    echo "INFO: Restoring local 'data' directory..."
    mv "$TEMP_BACKUP_PATH" data
    echo "INFO: Local 'data' directory restored."
fi
echo "SUCCESS: Project repository is up to date."

# --- NEW: Prompt for manual data upload ---
# This allows the user to add large files (e.g., from FTP) that shouldn't be in git.
# These files will be used by the current run and backed up by subsequent runs of this script.
echo -e "\n--------------------------------------------------"
echo "Your local 'data' directory is in place and the repository is updated."
echo "Do you need to pause to manually upload additional data files now?"
read -r -p "Pause for manual data upload? (y/n): " upload_response
if [[ "$upload_response" == "y" || "$upload_response" == "Y" ]]; then
    echo -e "\n--- USER ACTION REQUIRED ---"
    echo "The script is paused. Please upload your files to the following directory:"
    echo "  -> $(pwd)/data"
    read -p "Once you are finished, press [Enter] to continue..."
    echo "INFO: Resuming script."
fi

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