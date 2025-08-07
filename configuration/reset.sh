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
# VERSION: 9.1 (Added LD_LIBRARY_PATH export for runtime linking)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pre-flight Check: Refresh sudo timestamp ---
echo "INFO: This script uses 'sudo' to manage system services and mounts."
echo "You may be prompted for your password once at the beginning."
sudo -v
echo "SUCCESS: Sudo credentials refreshed."
read -r -p "This script needs to install system-level build tools (like build-essential, cmake). Is it OK to proceed? (y/n): " install_confirm
if [[ "$install_confirm" == "y" || "$install_confirm" == "Y" ]]; then
    echo "INFO: Installing comprehensive system-level build tools..."
    if command -v apt-get &> /dev/null; then
        echo "  - Debian/Ubuntu based system detected. Using apt-get."
        sudo apt-get update && sudo apt-get install -y build-essential cmake libssl-dev autoconf automake libtool pkg-config
    elif command -v dnf &> /dev/null || command -v yum &> /dev/null; then
        echo "  - RedHat/CentOS/Fedora based system detected. Using dnf/yum."
        sudo yum install -y gcc-c++ make cmake openssl-devel autoconf automake libtool pkgconfig
    elif command -v brew &> /dev/null; then
        echo "  - macOS detected. Using Homebrew."
        brew install cmake openssl pkg-config autoconf automake libtool
    else
        echo "  - WARNING: Could not detect package manager. Skipping system dependency installation."
        echo "  - Please ensure 'build-essential' (or equivalent), 'cmake', and 'libssl-dev' are installed."
    fi
    echo "SUCCESS: System-level build tools check complete."
else
    echo "Skipping system dependency installation as requested. The build may fail if dependencies are missing."
fi

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

# --- NEW: Add a flag to track if we should restore data ---
RESTORE_DATA=false

# This is a reset script. If the directory exists, it will be destroyed to ensure a clean slate.
if [ -d "$PROJECT_DIR_NAME" ]; then
    echo "INFO: Existing project directory found. It will be completely removed for a clean reset."
    # --- FIX: Interactively ask the user whether to keep or discard the existing data directory ---
    if [ "$LFS_ISSUE" = true ] && [ -d "$PROJECT_DIR_NAME/data" ]; then
        echo -e "\nAn existing 'data' directory was found."
        echo "This reset script will DELETE the entire project folder ('$PROJECT_DIR_NAME') and re-clone it."
        echo -e "\nWhat should be done with your current 'data' directory?"
        echo "  (k) Keep    - Back up the current 'data' directory and restore it in the new clone."
        echo "  (d) Discard - Delete the current 'data' directory. You will be prompted to provide a new one later."
        read -r -p "Choose an option [k/d]: " keep_data_response
        if [[ "$keep_data_response" == "k" || "$keep_data_response" == "K" ]]; then
            echo "INFO: Backing up existing 'data' directory to a safe location..."
            # Back up directly to the user's home directory for maximum safety.
            mv "$PROJECT_DIR_NAME/data" "$HOME/data_temp_backup"
            echo "INFO: 'data' directory temporarily backed up to '$HOME/data_temp_backup'."
            RESTORE_DATA=true
        else
            echo "INFO: The existing 'data' directory will be discarded along with the project."
        fi
    fi
    # --- END FIX ---
    rm -rf "$PROJECT_DIR_NAME" # Now it's safe to remove the old project
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
    echo "!data" >> .git/info/sparse-checkout
    echo "INFO: Checking out branch '$GIT_BRANCH'..."
    git checkout "$GIT_BRANCH"
    # --- FIX: Restore the backed-up data directory if the user chose to keep it ---
    if [ "$RESTORE_DATA" = true ] && [ -d "$HOME/data_temp_backup" ]; then
        echo "INFO: Restoring backed-up 'data' directory..."
        echo "INFO: Current directory for restore is: $(pwd)"
        # This command moves the backup into the current directory and renames it to 'data'.
        # The 'mv' command automatically removes the source directory ('$HOME/data_temp_backup').
        mv "$HOME/data_temp_backup" "./data"
        echo "SUCCESS: 'data' directory restored and backup automatically removed from home directory."
    fi
    # --- END FIX ---
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
    # Only prompt the user if the data directory wasn't restored from a backup
    if [ ! -d "data" ]; then
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
    fi
    echo "INFO: 'data' directory found. Proceeding with the pipeline."
fi

echo "SUCCESS: Project repository is ready."

# --- Step 4: Run the Main Application ---
echo -e "\n--- STEP 4: Executing the main application via run.py ---"

# --- CRITICAL FIX: Export the Conda environment's library path. ---
# This ensures that TensorFlow and other programs can find the CUDA libraries (.so files)
# that were installed by Conda. This resolves the "Cannot dlopen" errors at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- CRITICAL FIX for XLA/JIT: Point TensorFlow's XLA compiler to the Conda CUDA toolkit. ---
# This resolves the "libdevice not found" and "JIT compilation failed" errors when
# running Transformer models on the GPU.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0