# ==============================================================================
# MODULE: configuration/setup-environment.py
# PURPOSE: Sets up the Python environment for the project using a robust,
#          conda-first strategy for managing GPU dependencies.
# VERSION: 3.0 (Consolidated conda/pip installs for stability and speed)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import argparse
import os
import platform
import subprocess
import sys

# --- Configuration ---
# This script uses Conda to create a stable environment with a specific CUDA version.
# Both PyTorch and TensorFlow will be installed to use these shared libraries.
PYTHON_VERSION = "3.11"
CUDA_VERSION_MAJOR_MINOR = "12.1"

# PyTorch versions should be compatible with the target CUDA version.
PYTORCH_VERSION = "2.4.0"
TORCHVISION_VERSION = "0.19.0"
TORCHAUDIO_VERSION = "2.4.0"


# --- End Configuration ---

def create_setup_script(commands: list[str]) -> str:
    """Creates a platform-specific shell script from a list of commands."""
    is_windows = platform.system() == "Windows"
    script_extension = ".bat" if is_windows else ".sh"
    script_filename = f"setup_script{script_extension}"

    with open(script_filename, "w") as f:
        if not is_windows:
            f.write("#!/bin/bash\n")
            # Exit immediately if a command exits with a non-zero status.
            f.write("set -e\n")
        for command in commands:
            f.write(command + "\n")

    # Make the script executable on non-Windows systems
    if not is_windows:
        os.chmod(script_filename, 0o755)
    return script_filename


def run_script(script_filename: str):
    """Executes the setup script and streams its output."""
    is_windows = platform.system() == "Windows"
    print(f"--- Starting Environment Setup using '{script_filename}' ---")
    try:
        executor = ['cmd', '/c'] if is_windows else []
        # For Linux/macOS, execute the script directly from the current directory
        script_path = script_filename if is_windows else f"./{script_filename}"

        process = subprocess.Popen(
            executor + [script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream the output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"\n--- Script failed with exit code {process.returncode} ---")
            sys.exit(process.returncode)
        else:
            print("\n--- Environment setup completed successfully! ---")

    finally:
        # Clean up the generated script file
        if os.path.exists(script_filename):
            os.remove(script_filename)
            print(f"--- Cleaned up temporary script file: {script_filename} ---")


def check_conda_installed() -> bool:
    """Checks if conda is installed and available in the system's PATH."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True, text=True, shell=False)
        print("--- Conda is installed. ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install required packages into the active Conda environment.")
    parser.parse_args()
    if not check_conda_installed():
        sys.exit(1)

    system = platform.system()
    compiler_commands = []
    if system == "Linux":
        # For Linux, install the GNU compiler toolchain from conda-forge.
        compiler_commands.extend([
            "echo '--- Installing GCC/G++ compilers for Linux ---'",
            "conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y",
            "echo '--- Clearing PyCUDA cache to prevent stale compiler paths ---'",
            "rm -rf ~/.config/pycuda"
        ])

    # This command sequence is designed for robustness and clarity.
    # 1. A single, unified Conda command installs all complex binary dependencies.
    #    This allows Conda's solver to create a consistent environment one time.
    # 2. A single, unified Pip command installs the remaining packages.
    command_sequence = [
        "conda clean --all -y",
        "conda update --all -y",
        *compiler_commands,

        # --- 1. UNIFIED CONDA INSTALL: Install all conda packages in a single, coherent step. ---
        "echo '--- Installing PyTorch, PyG, and Core Libraries via Conda ---'",
        (
            # Prioritize channels correctly for the solver: specific (pytorch, pyg) before general (nvidia, conda-forge)
            f"conda install -c pytorch -c pyg -c nvidia -c conda-forge -y "
            # PyTorch Ecosystem
            f"pytorch={PYTORCH_VERSION} torchvision={TORCHVISION_VERSION} torchaudio={TORCHAUDIO_VERSION} "
            # This metapackage tells conda to install a compatible cuda-toolkit and cudnn as dependencies.
            f"pytorch-cuda={CUDA_VERSION_MAJOR_MINOR} "
            # PyG Ecosystem
            f"pyg "
            # Core Libraries
            f"dask tqdm biopython matplotlib scipy scikit-learn "
            f"gensim python-louvain seaborn pycuda networkx=3.2.1 "
            f"pandas h5py"
        ),

        # --- 2. UNIFIED PIP INSTALL: Install all remaining packages via pip. ---
        "echo '--- Installing TensorFlow and other pip packages ---'",
        (
            # Install TensorFlow and other packages that are best sourced from pip.
            "pip install tensorflow tf-keras mlflow transformers==4.41.2"
        ),

        # --- 3. VERIFICATION & CLEANUP ---
        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence)
    run_script(script_file)