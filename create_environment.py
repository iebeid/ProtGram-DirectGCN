# ==============================================================================
# SCRIPT: create_environment.py
# PURPOSE: A robust script to programmatically create the Conda environment
#          for the ProtDiGCN project.
#
# USAGE:
#   1. Make sure you are in your 'base' Conda environment.
#   2. Run the script from your terminal: > python create_environment.py
# ==============================================================================

import os
import sys
import subprocess
import platform

# --- Configuration ---
ENV_NAME = "protdigcn-win-gpu-env"
PYTHON_VERSION = "3.9"

# Conda packages for the base environment
# These are packages that are safe to install together.
CONDA_PACKAGES = [
    # Core ML & CUDA
    "python=" + PYTHON_VERSION,
    "cudatoolkit=11.2.2",
    "cudnn=8.1.0",
    "tensorflow=2.10.0",
    "pytorch=1.12.1",
    "torchvision",
    "torchaudio",
    # Key Dependencies
    "dask", "dask-expr", "h5py", "matplotlib",
    "pandas", "pyarrow", "pyqt", "requests",
    "scikit-learn", "seaborn", "tqdm", "mlflow",
    "biopython", "networkx"
]

# Pip packages to be installed after the environment is created
# This list includes the special PyG dependencies.
PIP_PACKAGES = [
    # First, install the compiled PyG dependencies from the official wheel index
    "torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
    "torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
    "torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
    "torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
    # Then, install the main PyG library
    "torch-geometric",
    # Finally, install the other remaining pip dependencies
    "gensim",
    "python-louvain",
    "transformers",
    "torch-geometric-signed-directed"
]

# Define conda channels
CHANNELS = ["pytorch", "nvidia", "conda-forge", "defaults"]

# --- Helper Functions ---

def run_command(command, error_message):
    """Executes a command and exits if it fails."""
    try:
        print(f"\n>--- Running Command: {' '.join(command)} ---\n")
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"{error_message}")
        print(f"Failed command: {' '.join(command)}")
        print(f"Error details: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Command not found. Is Conda installed and in your system's PATH?")
        sys.exit(1)

def get_conda_base_prefix():
    """Finds the base directory of the Conda installation."""
    try:
        # The 'conda info --base' command is the most reliable way to get the base path
        result = subprocess.run(['conda', 'info', '--base'], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception as e:
        print("\n--- ERROR ---")
        print("Could not determine Conda base directory.")
        print(f"Error details: {e}")
        sys.exit(1)

def check_and_remove_env(env_name, conda_base):
    """Checks if the environment exists and asks the user to remove it."""
    env_path = os.path.join(conda_base, 'envs', env_name)
    if os.path.exists(env_path):
        print(f"An environment named '{env_name}' already exists.")
        answer = input("Would you like to remove it and continue? (y/n): ").lower().strip()
        if answer == 'y':
            run_command(
                ['conda', 'env', 'remove', '--name', env_name],
                f"Failed to remove existing environment '{env_name}'."
            )
            print(f"Successfully removed old environment '{env_name}'.")
        else:
            print("Aborting installation.")
            sys.exit(0)

# --- Main Script ---

if __name__ == "__main__":
    print("======================================================")
    print("=== Starting ProtDiGCN Environment Creation Script ===")
    print("======================================================")

    conda_base_path = get_conda_base_prefix()
    check_and_remove_env(ENV_NAME, conda_base_path)

    # Step 1: Create the base Conda environment
    print("\n[Step 1/2] Creating the base Conda environment with core packages...")
    print("This may take several minutes...")

    create_command = ['conda', 'create', '--name', ENV_NAME, '-y']
    for channel in CHANNELS:
        create_command.extend(['-c', channel])
    create_command.extend(CONDA_PACKAGES)

    run_command(create_command, "Failed to create the base Conda environment.")
    print("\n--- Base environment created successfully! ---")

    # Step 2: Install pip packages into the new environment
    print("\n[Step 2/2] Installing PyTorch Geometric and other pip packages...")

    # Determine the correct path to the new environment's Python/Pip executable
    if platform.system() == "Windows":
        pip_executable = os.path.join(conda_base_path, 'envs', ENV_NAME, 'Scripts', 'pip.exe')
    else: # Linux or macOS
        pip_executable = os.path.join(conda_base_path, 'envs', ENV_NAME, 'bin', 'pip')
    
    if not os.path.exists(pip_executable):
        print(f"--- ERROR ---")
        print(f"Could not find pip executable at: {pip_executable}")
        print("The base environment may not have been created correctly.")
        sys.exit(1)

    for package in PIP_PACKAGES:
        # Pip command needs to be split properly for subprocess
        install_command = [pip_executable, 'install'] + package.split()
        run_command(install_command, f"Failed to install pip package: '{package}'.")
    
    print("\n--- Pip packages installed successfully! ---")
    print("\n======================================================")
    print("=== Environment creation complete! ===")
    print(f"To activate your new environment, run:")
    print(f"\n> conda activate {ENV_NAME}\n")
    print("======================================================")