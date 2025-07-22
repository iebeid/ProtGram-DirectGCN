# ==============================================================================
# MODULE: configuration/setup-environment.py
# PURPOSE: Sets up the Python environment for the project using a robust,
#          conda-first strategy for managing GPU dependencies.
# VERSION: 2.0
# AUTHOR: Islam Ebeid
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
CUDA_VERSION_MAJOR_MINOR = "12.5"
CUDNN_VERSION_MAJOR = "9.3"  # Conda will select the latest compatible minor version.

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
    # 1. Establish the base CUDA environment using Conda from the official NVIDIA channel.
    # 2. Install the main ML frameworks (PyTorch, TF) and other heavy dependencies using Conda.
    #    Conda is better at managing the complex binary dependencies for these packages.
    # 3. Install PyG and its dependencies, which are also best handled by Conda.
    # 4. Use pip only for packages not available on Conda or for which pip is preferred (like tf-keras).
    command_sequence = [
        "conda clean --all -y",
        "conda update --all -y",
        *compiler_commands,

        # --- 1. GPU LIBRARIES: Establish a single, authoritative source for CUDA/cuDNN ---
        "echo '--- Installing CUDA and cuDNN from the official nvidia channel ---'",
        f"conda install -c nvidia -y cuda-toolkit={CUDA_VERSION_MAJOR_MINOR} cudnn={CUDNN_VERSION_MAJOR}",

        # --- 2. ML FRAMEWORKS: Install PyTorch and TensorFlow from their recommended Conda channels ---
        "echo '--- Installing PyTorch and TensorFlow via Conda ---'",
        (
            f"conda install -c pytorch -c conda-forge -y "
            f"pytorch={PYTORCH_VERSION} torchvision={TORCHVISION_VERSION} torchaudio={TORCHAUDIO_VERSION} "
            # This metapackage ensures PyTorch links against the right CUDA version.
            # The driver from the system/conda env (12.5) is forward-compatible with the 12.1 runtime.
            f"pytorch-cuda=12.1 tensorflow"
        ),

        # --- 3. PyG (PyTorch Geometric): Install from its own channel for best compatibility ---
        "echo '--- Installing PyTorch Geometric (PyG) ---'",
        "conda install -c pyg -y pyg",

        # --- 4. CORE LIBRARIES: Install remaining packages via Conda ---
        "echo '--- Installing core data science and ML libraries via Conda ---'",
        (
            "conda install -c conda-forge -y "
            "dask tqdm biopython matplotlib scipy scikit-learn "
            "gensim python-louvain seaborn pycuda networkx=3.2.1 "
            "pandas h5py"
        ),

        # --- 5. PIP PACKAGES: Use pip for packages not available/ideal on Conda ---
        "echo '--- Installing remaining packages via pip ---'",
        (
            "pip install mlflow transformers==4.41.2 tf-keras"
        ),

        # --- 6. VERIFICATION & CLEANUP ---
        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence)
    run_script(script_file)
