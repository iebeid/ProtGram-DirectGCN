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

# This is a reset script. If the directory exists, it will be destroyed to ensure a clean slate.
if [ -d "$PROJECT_DIR_NAME" ]; then
    echo "INFO: Existing project directory found. It will be completely removed for a clean reset."
    rm -rf "$PROJECT_DIR_NAME" # Now it's safe to remove the old project
    echo "SUCCESS: Old project directory removed."
fi

# --- Always perform a standard, full clone. Data is not in the repo. ---
echo "INFO: Performing a standard, full clone..."
git clone --branch "$GIT_BRANCH" "$REPO_URL"
cd "$PROJECT_DIR_NAME"
echo "SUCCESS: Project repository is ready."

# --- NEW STEP: Install Python Dependencies ---
echo -e "\n--- STEP 3.5: Installing Python Dependencies ---"
if [ -f "requirements.txt" ]; then
    echo "INFO: Found requirements.txt. Installing packages..."
    # Use --no-cache-dir to ensure fresh installs and --upgrade to meet specified versions.
    pip install --no-cache-dir --upgrade -r requirements.txt
    echo "SUCCESS: Python dependencies installed."
else
    echo "ERROR: requirements.txt not found in the project root. Cannot install dependencies."
    exit 1
fi

# --- CRITICAL FIX: Export the Conda environment's library path. ---
# This ensures that TensorFlow and other programs can find the CUDA libraries (.so files)
# that were installed by Conda. This resolves the "Cannot dlopen" errors at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# --- DEFINITIVE FIX for Reproducibility: Configure CUDA workspace ---
# This environment variable is required by `torch.use_deterministic_algorithms(True)`
# to ensure that operations like `index_add` (used by PyG's scatter_add) are deterministic.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- CRITICAL FIX for XLA/JIT: Point TensorFlow's XLA compiler to the Conda CUDA toolkit. ---
# This resolves the "libdevice not found" and "JIT compilation failed" errors when
# running Transformer models on the GPU.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# --- Step 4: Check for cached data bundle and restore, or run full setup ---
CACHE_DIR="$HOME/.cache/protgram_directgcn"
DATA_BUNDLE_PATH="$CACHE_DIR/data_bundle.tar.gz"
DATA_MANIFEST_PATH="$CACHE_DIR/data_manifest.json"

if [ -f "$DATA_BUNDLE_PATH" ] && [ -f "$DATA_MANIFEST_PATH" ]; then
    echo -e "\n--- STEP 4: Found existing data bundle in cache. Restoring data... ---"
    # Call python to restore. The script will exit with an error if restoration fails.
    python -c "from configuration.config import Config; from configuration.data import DataManager; dm = DataManager(Config()); restored = dm.restore_data_from_bundle(); exit(0) if restored else exit(1)"
    echo "--- Data restoration from cache complete. ---"
else
    echo -e "\n--- STEP 4: No data bundle found in cache. Performing full data download and processing... ---"
    echo "--- This is a long-running process and will only be done once. ---"
    # Trigger the full data setup from configuration/manager.py
    python -c "from configuration.config import Config; from configuration.data import setup_data; print('--- Triggering DataManager full setup ---'); setup_data(Config())"
fi

echo -e "\n--- RESET SCRIPT FINISHED ---"
echo "--- The environment and data are now fully set up. ---"
echo "--- You can now use 'start.sh' for subsequent runs. ---"
exit 0