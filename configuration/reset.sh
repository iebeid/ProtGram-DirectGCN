#!/bin/bash

# ==============================================================================
# SCRIPT: reset.sh
# PURPOSE: Completely resets the project by destroying the old repository and
#          re-cloning. It interactively handles two cases:
#          1. A normal, full clone for users without LFS issues.
#          2. A sparse clone for users with LFS budget errors, prompting for
#             manual data placement.
# WARNING: This script is ALWAYS DESTRUCTIVE and will remove the existing
#          project directory.
# VERSION: 9.0 (Installs a comprehensive system-level build toolchain)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pre-flight Check: Refresh sudo timestamp ---
echo "INFO: This script uses 'sudo' to manage system services and mounts."
echo "You may be prompted for your password once at the beginning."
sudo -v
echo "SUCCESS: Sudo credentials refreshed."

# --- NEW STRATEGY: Install a comprehensive system-level build toolchain ---
# This is more robust than relying on conda's compilers or letting pip build them.
# It provides gcc, g++, make, cmake, and the full GNU Autotools suite.
echo "INFO: Installing comprehensive system-level build tools..."
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    autoconf \
    automake \
    libtool \
    pkg-config
echo "SUCCESS: System-level build tools are installed."

# --- Configuration ---
ENV_NAME="ppi-env"
REPO_URL="https://github.com/iebeid/ProtGram-DirectGCN.git"
PROJECT_DIR_NAME="ProtGram-DirectGCN"
PYTHON_VERSION="3.11"
GIT_BRANCH="v2"

# --- Step 0: Define Project Structure and Find Conda ---
DOCUMENTS_DIR="$HOME/documents"
PROJECTS_DIR="$DOCUMENTS_DIR/projects"

echo "INFO: Ensuring project directory structure exists: $PROJECTS_DIR"
mkdir -p "$PROJECTS_DIR"
echo "SUCCESS: Project root will be in: $PROJECTS_DIR"

CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not find Conda base directory. Is Conda installed?"
    exit 1
fi
echo "INFO: Conda base found at: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Step 0.5: Dependency Checks (Git, Git LFS) ---
if ! command -v git &> /dev/null || ! command -v git-lfs &> /dev/null; then
    echo "ERROR: 'git' and 'git-lfs' are required. Please install them."
    exit 1
fi
echo "INFO: Git and Git LFS are installed."


# --- Step 1: Deactivate and Remove Old Environment ---
echo -e "\n--- STEP 1: Deactivating and Removing Conda Environment '$ENV_NAME' ---"
conda deactivate
if conda env list | grep -q "$ENV_NAME"; then
    echo "INFO: Environment '$ENV_NAME' found. Removing..."
    conda env remove -n "$ENV_NAME" -y
    echo "SUCCESS: Environment '$ENV_NAME' removed."
else
    echo "INFO: Environment '$ENV_NAME' not found. Skipping removal."
fi
conda clean --all -y > /dev/null
echo "SUCCESS: Conda cache cleaned."

# --- Step 2: Re-create Environment and Activate ---
echo -e "\n--- STEP 2: Re-creating Conda Environment '$ENV_NAME' ---"
conda create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"
echo "SUCCESS: Environment '$ENV_NAME' created and activated."
python --version

# --- Step 3: Reset Project Directory ---
echo -e "\n--- STEP 3: Resetting Project Directory ---"
cd "$PROJECTS_DIR"
echo "INFO: Current directory: $(pwd)"

# --- INTERACTIVE CHOICE FOR LFS HANDLING ---
LFS_ISSUE=false
read -r -p "Are you experiencing Git LFS budget errors that prevent 'git lfs pull' from working? (y/n): " lfs_response
if [[ "$lfs_response" == "y" || "$lfs_response" == "Y" ]]; then
    LFS_ISSUE=true
fi

# This is a reset script. If the directory exists, it will be destroyed to ensure a clean slate.
if [ -d "$PROJECT_DIR_NAME" ]; then
    echo "INFO: Existing project directory found. It will be completely removed for a clean reset."
    rm -rf "$PROJECT_DIR_NAME"
    echo "SUCCESS: Old project directory removed."
fi

# Now, clone the repository based on the user's LFS situation.
if [ "$LFS_ISSUE" = true ]; then
    # --- CASE 2: LFS PROBLEM (SPARSE CLONE) ---
    echo "INFO: LFS issue detected. Performing a sparse clone to exclude the 'data' directory."
    echo "INFO: Cloning repository structure without checking out files..."
    git clone --filter=blob:none --no-checkout "$REPO_URL"
    cd "$PROJECT_DIR_NAME"
    # Manually configure sparse-checkout for maximum compatibility, bypassing the 'set' command.
    git sparse-checkout init
    # This writes the patterns directly to the config file, which is more robust.
    echo "/*" > .git/info/sparse-checkout
    echo "!/data" >> .git/info/sparse-checkout
    echo "INFO: Checking out branch '$GIT_BRANCH'..."
    git checkout "$GIT_BRANCH"
else
    # --- CASE 1: NO LFS PROBLEM (FULL CLONE) ---
    echo "INFO: No LFS issues. Performing a standard, full clone..."
    git clone --branch "$GIT_BRANCH" "$REPO_URL"
    cd "$PROJECT_DIR_NAME"
    echo "INFO: Downloading LFS data..."
    git lfs pull
fi

# --- USER INTERVENTION STEP FOR LFS ISSUES ---
if [ "$LFS_ISSUE" = true ]; then
    echo -e "\n\n\n--- USER ACTION REQUIRED ---"
    echo "The repository has been set up WITHOUT the 'data' directory to avoid LFS errors."
    echo "Please manually place your complete 'data' directory into the following location:"
    echo "  -> $(pwd)"
    echo "You can download the files from the GitHub repository webpage and create the directory structure."
    read -p "Once the 'data' directory is in place, press [Enter] to continue the script..."

    if [ ! -d "data" ]; then
        echo "ERROR: The 'data' directory was not found. Aborting."
        exit 1
    fi
    echo "INFO: 'data' directory found. Proceeding with the pipeline."
fi

echo "SUCCESS: Project repository is ready."

# --- Step 4: Run the Main Application ---
echo -e "\n--- STEP 4: Executing the main application via run.py ---"
# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0