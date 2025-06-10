# ==============================================================================
# SCRIPT: create_environment.py (v12 - Two-Stage Constraint Locking)
# PURPOSE: A robust, OS-agnostic script that uses a two-stage process
#          to first "lock in" the GPU-enabled PyTorch build, and then
#          installs TensorFlow and all other dependencies. This provides a
#          more reliable solution than previous methods.
#
# USAGE:
#   1. For best results, install mamba in your base env: > conda install -n base mamba
#   2. Run the script from your terminal: > python create_environment.py
# ==============================================================================

import os
import sys
import subprocess
import platform
import shutil

# --- Configuration ---
ENV_NAME = "protgram-directgcn-8"
PYTHON_VERSION = "3.9"

# Stage 1: The critical packages to create the GPU foundation.
# We give the solver a simple problem to solve first.
GPU_FOUNDATION_PACKAGES = [f"python={PYTHON_VERSION}", "pytorch=1.12.1", "torchvision", "torchaudio", "cudatoolkit=11.3.1", "cudnn=8.2.1", ]

# Stage 2: The rest of the packages to be installed into the foundation.
ADDITIONAL_CONDA_PACKAGES = ["tensorflow=2.10.0", "tqdm", "dask", "dask-expr", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx"]

# Stage 3: Pip packages to be installed last.
PIP_PACKAGES = ["torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
    "torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-geometric", "gensim", "python-louvain", "transformers",
    "torch-geometric-signed-directed"]

# Define conda channels, prioritizing pytorch and nvidia
CHANNELS = ["pytorch", "nvidia", "conda-forge", "defaults"]


# --- Helper Functions ---

def get_solver():
    """Check if 'mamba' is available, otherwise use 'conda'."""
    if shutil.which("mamba"):
        print("--- Mamba solver detected. Using Mamba for faster installation. ---")
        return "mamba"
    else:
        print("--- Mamba not found. Using the default Conda solver. ---")
        print("    (For a much faster experience, run: conda install -n base mamba)")
        return "conda"


def run_command(command, error_message, step_description=""):
    """Executes a command in real-time and exits if it fails."""
    try:
        print(f"\n>--- {step_description}: {' '.join(command)} ---\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n--- ERROR in step: {step_description} ---")
        print(f"{error_message}")
        print(f"Failed command: {' '.join(command)}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Return code: {e.returncode}")
        sys.exit(1)


def get_conda_base_prefix():
    """Finds the base directory of the Conda installation."""
    try:
        result = subprocess.run(['conda', 'info', '--base'], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        print("\n--- ERROR: Could not determine Conda base directory. Is Conda installed and in your PATH? ---")
        sys.exit(1)


def check_and_remove_env(env_name, conda_base):
    """Checks if the environment directory exists and removes it if necessary."""
    env_path = os.path.join(conda_base, 'envs', env_name)
    if os.path.exists(env_path):
        print(f"An environment directory named '{env_name}' already exists.")
        answer = input("Would you like to remove it and continue? (y/n): ").lower().strip()
        if answer == 'y':
            try:
                print(f"Forcefully removing directory: {env_path}")
                shutil.rmtree(env_path)
                print(f"Successfully removed old environment directory '{env_name}'.")
            except OSError as e:
                print(f"\n--- ERROR: Failed to remove directory '{env_path}'. Please close any programs using it and try again. ---\nDetails: {e}")
                sys.exit(1)
        else:
            print("Aborting installation.")
            sys.exit(0)


# --- Main Script ---

if __name__ == "__main__":
    print("======================================================")
    print("=== Starting ProtDiGCN Environment Creation Script ===")
    print("======================================================")

    solver = get_solver()
    conda_base_path = get_conda_base_prefix()
    check_and_remove_env(ENV_NAME, conda_base_path)

    # Step 1: Create the core GPU foundation environment
    print(f"\n[Step 1/3] Creating GPU foundation with PyTorch using '{solver}'...")
    print("This is the most important step and may take several minutes...")

    create_command = [solver, 'create', '--name', ENV_NAME, '-y']
    for channel in CHANNELS:
        create_command.extend(['-c', channel])
    create_command.extend(GPU_FOUNDATION_PACKAGES)

    run_command(create_command, "Failed to create the GPU foundation environment.", f"Creating GPU Foundation with {solver}")
    print("\n--- GPU foundation created successfully! ---")

    # Step 2: Install additional Conda packages sequentially
    print(f"\n[Step 2/3] Installing TensorFlow and other packages using '{solver}'...")
    try:
        from tqdm import tqdm
    except ImportError:
        # Install tqdm first so we can use it
        run_command([solver, 'install', '-n', ENV_NAME, '-y', 'tqdm'], "Failed to install tqdm.", "Installing tqdm")
        from tqdm import tqdm

    for package in tqdm(ADDITIONAL_CONDA_PACKAGES, desc="Installing Conda Packages"):
        install_command = [solver, 'install', '-n', ENV_NAME, '-y', package]
        for channel in CHANNELS:
            install_command.extend(['-c', channel])
        run_command(install_command, f"Failed to install conda package: '{package}'.", f"Installing {package}")
    print("\n--- Additional Conda packages installed successfully! ---")

    # Step 3: Install pip packages sequentially
    print("\n[Step 3/3] Installing PyTorch Geometric and other pip packages...")
    if platform.system() == "Windows":
        pip_executable = os.path.join(conda_base_path, 'envs', ENV_NAME, 'Scripts', 'pip.exe')
    else:  # Linux, macOS, WSL
        pip_executable = os.path.join(conda_base_path, 'envs', ENV_NAME, 'bin', 'pip')

    for package in tqdm(PIP_PACKAGES, desc="Installing Pip Packages"):
        install_command = [pip_executable, 'install'] + package.split()
        run_command(install_command, f"Failed to install pip package: '{package}'.", f"Installing {package}")

    print("\n--- Pip packages installed successfully! ---")
    print("\n======================================================")
    print("=== Environment creation complete! ===")
    print(f"To activate your new environment, run:")
    print(f"\n> conda activate {ENV_NAME}\n")
    print("======================================================")
