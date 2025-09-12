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
# VERSION: 9.2 (Added data cache removal for a true reset)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Pre-flight Check: Refresh sudo timestamp ---
echo "INFO: This script uses 'sudo' to manage system services and mounts."
echo "You may be prompted for your password once at the beginning."
sudo -v
echo "SUCCESS: Sudo credentials refreshed."
read -r -p "This script needs to install system-level tools (like git, git-lfs, build-essential, cmake). Is it OK to proceed? (y/n): " install_confirm
if [[ "$install_confirm" == "y" || "$install_confirm" == "Y" ]]; then
    echo "INFO: Installing comprehensive system-level tools..."
    if command -v apt-get &> /dev/null; then
        echo "  - Debian/Ubuntu based system detected. Using apt-get."
        sudo apt-get update && sudo apt-get install -y build-essential cmake libssl-dev autoconf automake libtool pkg-config git git-lfs
    elif command -v dnf &> /dev/null || command -v yum &> /dev/null; then
        echo "  - RedHat/CentOS/Fedora based system detected. Using dnf/yum."
        sudo yum install -y gcc-c++ make cmake openssl-devel autoconf automake libtool pkgconfig git git-lfs
    elif command -v pacman &> /dev/null; then
        echo "  - Arch-based system detected. Using pacman."
        sudo pacman -Syu --noconfirm base-devel cmake openssl pkg-config autoconf automake libtool git git-lfs
    elif command -v brew &> /dev/null; then
        echo "  - macOS detected. Using Homebrew."
        brew install cmake openssl pkg-config autoconf automake libtool git git-lfs
    else
        echo "  - WARNING: Could not detect package manager. Skipping system dependency installation."
        echo "  - Please ensure 'build-essential' (or equivalent), 'cmake', and 'libssl-dev' are installed."
    fi
    echo "SUCCESS: System-level build tools check complete."
else
    echo "Skipping system dependency installation as requested. The build may fail if dependencies are missing."
fi

# --- DEFINITIVE FIX: Ensure git-lfs is configured after installation ---
echo "INFO: Running 'git lfs install' to configure Git hooks..."
git lfs install

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

# --- NEW: Proactively accept Conda Terms of Service ---
# On fresh Anaconda installations, the ToS for default channels must be accepted.
echo "INFO: Proactively accepting Conda Terms of Service to prevent interactive prompts..."
# --- DEFINITIVE FIX: Handle multiple Conda versions and their ToS mechanisms ---
# Newer versions use a single config key. Older versions use the 'tos' subcommand.
# We try all known methods, and `|| true` ensures the script continues if a command is not supported.
conda config --set anaconda_tos_accepted yes || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "SUCCESS: Conda Terms of Service handled."

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
echo -e "\n--- STEP 2: Re-creating a minimal Conda Environment '$ENV_NAME' ---"
echo "INFO: This creates a bare-bones Python environment. The full set of packages"
echo "      will be installed later by the 'setup.py' script."
conda create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y

# --- DEFINITIVE FIX: Dynamically find the new environment's path ---
# Instead of assuming the env is in `$CONDA_BASE/envs`, we parse conda's output
# to find the actual location. This handles system vs. user-level installations.
NEW_ENV_PATH=$(conda info --envs | grep -w "$ENV_NAME" | awk '{print $NF}')
if [ -z "$NEW_ENV_PATH" ]; then
    echo "ERROR: Could not find the path for the newly created environment '$ENV_NAME'."
    exit 1
fi
NEW_ENV_PYTHON="$NEW_ENV_PATH/bin/python"
NEW_ENV_PIP="$NEW_ENV_PATH/bin/pip"

echo "SUCCESS: Environment '$ENV_NAME' created."
"$NEW_ENV_PYTHON" --version

# --- Step 3: Reset Project Directory and Data Cache ---
echo -e "\n--- STEP 3: Resetting Project Directory and Data Cache ---"
cd "$PROJECTS_DIR"
echo "INFO: Current directory: $(pwd)"

echo -e "\n\n\033[1;31m" # Bold Red
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: DESTRUCTIVE ACTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "This script is about to COMPLETELY REMOVE the following directories:"
echo "  1. Project Directory: $PROJECTS_DIR/$PROJECT_DIR_NAME"
echo "  2. Persistent Cache:  $HOME/.cache/protgram_directgcn"
echo ""
echo "This will delete all local code changes, results, and cached data."
echo "This action CANNOT be undone."
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo -e "\033[0m" # Reset color

read -r -p "Are you absolutely sure you want to proceed with this destructive reset? (y/n): " reset_confirm
if [[ "$reset_confirm" != "y" && "$reset_confirm" != "Y" ]]; then
    echo "INFO: Reset cancelled by user. Exiting."
    exit 0
fi

# This is a reset script. If the directory exists, it will be destroyed to ensure a clean slate.
if [ -d "$PROJECT_DIR_NAME" ]; then
    rm -rf "$PROJECT_DIR_NAME"
    echo "SUCCESS: Old project directory removed."
fi

# --- DEFINITIVE FIX: Also remove the persistent data cache to ensure a true reset ---
CACHE_DIR="$HOME/.cache/protgram_directgcn"
if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    echo "SUCCESS: Data cache removed."
fi

# --- Always perform a standard, full clone. Data is not in the repo. ---
echo "INFO: Performing a standard, full clone..."
git clone --branch "$GIT_BRANCH" "$REPO_URL"
cd "$PROJECT_DIR_NAME"
echo "SUCCESS: Project repository is ready."

# --- NEW: Verification Step ---
# This step makes the script's behavior more transparent by showing exactly
# what was cloned from the remote repository. This helps diagnose issues
# where the remote branch might not contain the expected code.
echo -e "\n--- STEP 3.1: Verifying Cloned Repository State ---"
echo "  - Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  - Latest commit: $(git log -1 --oneline)"
echo "  - Verifying contents of 'configuration/__init__.py':"
echo "    --- start of file ---"
cat configuration/__init__.py || echo "    [INFO: File is empty or does not exist, which is the correct state.]"
echo "    --- end of file ---"

# --- NEW STEP: Install Python Dependencies ---
# This step installs the minimal bootstrap dependencies (like PyYAML) needed for the main setup scripts to run.
echo -e "\n--- STEP 3.5: Installing Python Dependencies ---"
REQUIREMENTS_FILE="configuration/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "INFO: Found bootstrap requirements at '$REQUIREMENTS_FILE'. Installing packages..."
    # Use the explicit path to pip from the new environment
    "$NEW_ENV_PIP" install --no-cache-dir --upgrade -r "$REQUIREMENTS_FILE"
    echo "SUCCESS: Python dependencies installed."
else
    echo "ERROR: Bootstrap requirements file not found at '$REQUIREMENTS_FILE'. Cannot install dependencies."
    exit 1
fi

echo -e "\n\n"
echo "================================================================================"
echo "--- RESET SCRIPT FINISHED ---"
echo "--- The project code has been reset and a minimal Conda environment created. ---"
echo -e "\n--- NEXT STEP: The main 'start.sh' script will complete the setup. ---"
echo "--- Change to the project directory and run it with the following command: ---"
echo "    cd $PROJECTS_DIR/$PROJECT_DIR_NAME && bash start.sh"
exit 0
