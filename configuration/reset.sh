#!/bin/bash

# ==============================================================================
# SCRIPT: reset_and_run.sh
# PURPOSE: Completely resets the project by removing the Conda environment
#          and the project directory, then re-clones, re-installs, and runs.
# WARNING: This is a DESTRUCTIVE script. It will delete your local
#          'ppi-env' Conda environment and the 'ProtGram-DirectGCN' folder.
# USAGE: Place this in the project root and run with 'bash reset_and_run.sh'
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_NAME="ppi-env"
REPO_URL="https://github.com/iebeid/ProtGram-DirectGCN.git"
PROJECT_DIR_NAME="ProtGram-DirectGCN"
PYTHON_VERSION="3.11"
GIT_BRANCH="v2"

# --- Step 0: Find Conda and the Project's Parent Directory ---
# This makes the script runnable from anywhere inside the project.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PARENT_DIR=$(dirname "$PROJECT_ROOT")

echo "INFO: Project Root detected as: $PROJECT_ROOT"
echo "INFO: Parent Directory is: $PARENT_DIR"

# Find the base conda directory to source the activation script
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not find Conda base directory. Is Conda installed?"
    exit 1
fi
echo "INFO: Conda base found at: $CONDA_BASE"

# Source the conda script to make 'conda activate' available
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Step 1: Deactivate and Remove Old Environment ---
echo -e "\n--- STEP 1: Deactivating and Removing Conda Environment '$ENV_NAME' ---"
# Deactivate in case we are currently in the environment
conda deactivate

# Check if the environment exists before trying to remove it
if conda env list | grep -q "$ENV_NAME"; then
    echo "INFO: Environment '$ENV_NAME' found. Removing..."
    conda env remove -n "$ENV_NAME" -y
    echo "SUCCESS: Environment '$ENV_NAME' removed."
else
    echo "INFO: Environment '$ENV_NAME' not found. Skipping removal."
fi
conda clean --all -y
echo "SUCCESS: Conda cache cleaned."

# --- Step 2: Re-create Environment and Activate ---
echo -e "\n--- STEP 2: Re-creating Conda Environment '$ENV_NAME' ---"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"
echo "SUCCESS: Environment '$ENV_NAME' created and activated."
python --version

# --- Step 3: Re-clone the Repository ---
echo -e "\n--- STEP 3: Removing Old Project Directory and Re-cloning ---"
# Navigate OUTSIDE the project directory to delete it
cd "$PARENT_DIR"
echo "INFO: Current directory: $(pwd)"

echo "INFO: Removing old project directory: $PROJECT_DIR_NAME/"
# Use 'rm -rf' instead of sudo, assuming you have permissions.
# If you created the folder with sudo, you will need sudo here.
rm -rf "$PROJECT_DIR_NAME"

echo "INFO: Cloning fresh repository from $REPO_URL..."
git clone "$REPO_URL"
cd "$PROJECT_DIR_NAME"
echo "SUCCESS: Repository cloned. Current directory: $(pwd)"

# --- Step 4: Checkout Branch and Pull Latest ---
echo -e "\n--- STEP 4: Checking out branch '$GIT_BRANCH' and pulling data ---"
git checkout "$GIT_BRANCH"
echo "INFO: Checked out branch '$GIT_BRANCH'."
git pull
echo "INFO: Pulled latest changes for the branch."
git lfs pull
echo "INFO: Pulled LFS data."

# --- Step 5: Run the Main Application ---
echo -e "\n--- STEP 5: Executing the main application via run.py ---"
# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0