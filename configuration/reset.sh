#!/bin/bash

# ==============================================================================
# SCRIPT: reset.sh
# PURPOSE: Completely resets the project by creating a standard directory
#          structure in the user's home (~/Documents/Projects), setting up
#          system services, removing the old environment, then re-cloning and running.
# WARNING: This is a DESTRUCTIVE script. It will delete your local
#          'ppi-env' Conda environment.
# VERSION: 2.2 (Added system service setup for SSH, VSFTPD, and WSL mounts)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_NAME="ppi-env"
REPO_URL="https://github.com/iebeid/ProtGram-DirectGCN.git"
PROJECT_DIR_NAME="ProtGram-DirectGCN"
PYTHON_VERSION="3.11"
GIT_BRANCH="v2"

# --- Step 0: Define Project Structure and Find Conda ---
# Define the standard project location within the user's home directory.
DOCUMENTS_DIR="$HOME/documents"
PROJECTS_DIR="$DOCUMENTS_DIR/projects"

echo "INFO: Ensuring project directory structure exists: $PROJECTS_DIR"
# The '-p' flag creates parent directories (like Documents) as needed.
mkdir -p "$PROJECTS_DIR"
echo "SUCCESS: Project root will be in: $PROJECTS_DIR"

# Find the base conda directory to source the activation script
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not find Conda base directory. Is Conda installed?"
    echo "Please install Anaconda or Miniconda and ensure it is in your system's PATH."
    exit 1
fi
echo "INFO: Conda base found at: $CONDA_BASE"

# Source the conda script to make 'conda activate' available
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Step 0.5: Install Git and Git LFS if needed (for Debian/Ubuntu) ---
echo -e "\n--- STEP 0.5: Checking for Git and Git LFS ---"
# Check for apt package manager (Debian/Ubuntu)
if command -v apt &> /dev/null; then
    echo "INFO: 'apt' package manager found. Checking dependencies..."
    # Install git if not present
    if ! command -v git &> /dev/null; then
        echo "INFO: Git not found. Installing git..."
        sudo apt update
        sudo apt install git -y
        echo "SUCCESS: Git installed."
    else
        echo "INFO: Git is already installed."
    fi

    # Install git-lfs if not present
    if ! command -v git-lfs &> /dev/null; then
        echo "INFO: Git LFS not found. Installing git-lfs..."
        sudo apt install git-lfs -y
        echo "SUCCESS: Git LFS installed."
    else
        echo "INFO: Git LFS is already installed."
    fi
else
    echo "INFO: 'apt' not found. Assuming Git and Git LFS are already installed."
fi

# Initialize Git LFS if it's available
if command -v git-lfs &> /dev/null; then
    echo "INFO: Initializing Git LFS..."
    git lfs install
    echo "SUCCESS: Git LFS initialized."
else
    echo "WARNING: git-lfs command not found. Large files might not be downloaded correctly."
fi

# --- Step 0.7: Setting up System Services (SSH, VSFTPD, Mounts) ---
echo -e "\n--- STEP 0.7: Setting up System Services (SSH, VSFTPD, Mounts) ---"

# This section is primarily for Debian/Ubuntu-based systems
if command -v apt &> /dev/null; then
    # --- SSH Server ---
    echo "INFO: Attempting to start the SSH server..."
    sudo service ssh start &> /dev/null
    if pgrep -x "sshd" &> /dev/null; then
      echo "SUCCESS: SSH server process is running."
    else
      echo "WARNING: SSH server does not appear to be running."
    fi

    # --- VSFTPD Server ---
    echo "INFO: Checking for vsftpd..."
    if ! command -v vsftpd &> /dev/null; then
        echo "INFO: vsftpd not found. Installing..."
        sudo apt install vsftpd -y
        echo "SUCCESS: vsftpd installed."
    else
        echo "INFO: vsftpd is already installed."
    fi
    echo "INFO: Restarting vsftpd service..."
    sudo systemctl restart vsftpd.service
    echo "SUCCESS: vsftpd service restarted."

else
    echo "INFO: 'apt' not found. Skipping system service setup (SSH, VSFTPD)."
fi

# --- WSL-Specific Drive Mount ---
# Check if running in a WSL environment by looking for 'Microsoft' in /proc/version
if grep -q -i "microsoft" /proc/version &> /dev/null; then
    echo "INFO: WSL environment detected. Attempting to mount G: drive..."
    MOUNT_POINT="/mnt/g"
    echo "INFO: Ensuring mount point directory '$MOUNT_POINT' exists."
    sudo mkdir -p "$MOUNT_POINT"
    echo "INFO: Attempting to unmount '$MOUNT_POINT' to ensure a clean state."
    # The '|| true' prevents the script from exiting if the drive wasn't mounted.
    sudo umount "$MOUNT_POINT" &> /dev/null || true
    echo "INFO: Executing mount command..."
    sudo mount -t drvfs G: "$MOUNT_POINT" -o metadata
    if mountpoint -q "$MOUNT_POINT"; then
        echo "SUCCESS: The G: drive has been mounted to $MOUNT_POINT."
    else
        echo "ERROR: The mount command failed. The drive is not mounted."
    fi
else
    echo "INFO: Not a WSL environment. Skipping Windows drive mount."
fi


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
# Create the environment using conda-forge from the start to ensure consistency.
conda create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"
echo "SUCCESS: Environment '$ENV_NAME' created and activated."
python --version

# --- Step 3: Reset or Clone the Repository ---
echo -e "\n--- STEP 3: Resetting Project Directory ---"
# Navigate to the standard projects directory
cd "$PROJECTS_DIR"
echo "INFO: Current directory: $(pwd)"

# If the directory exists, clean it with git. Otherwise, clone it.
if [ -d "$PROJECT_DIR_NAME" ]; then
    echo "INFO: Project directory exists. Resetting to a clean state..."
    cd "$PROJECT_DIR_NAME"
    # --- FIX: This is the minimal change ---
    # Reset any changes to tracked files, but leave untracked (e.g., manual LFS) files alone.
    git reset --hard HEAD
    echo "SUCCESS: Project directory has been reset."
    # --- END FIX ---
else
    echo "INFO: Project directory not found. Cloning fresh repository..."
    git clone "$REPO_URL"
    cd "$PROJECT_DIR_NAME"
    echo "SUCCESS: Repository cloned."
fi

# --- Step 4: Checkout Branch and Pull Latest ---
echo -e "\n--- STEP 4: Checking out branch '$GIT_BRANCH' and pulling data ---"
git checkout "$GIT_BRANCH"
echo "INFO: Checked out branch '$GIT_BRANCH'."
git pull
echo "INFO: Pulled latest changes for the branch."
git lfs pull
echo "INFO: Attempted to pull LFS data. This may show errors for files over budget, which is expected."

# --- Step 5: Run the Main Application ---
echo -e "\n--- STEP 5: Executing the main application via run.py ---"
# The run.py script will handle the rest of the setup and execution.
python run.py

echo -e "\n--- SCRIPT FINISHED ---"
exit 0