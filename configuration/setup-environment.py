# ==============================================================================
# MODULE: configuration/setup-environment.py
# PURPOSE: Sets up the Python environment for the project using a robust,
#          sequential installation strategy based on a proven working configuration.
# VERSION: 5.1 (Corrected CUDA package name for the nvidia channel)
# AUTHOR: Islam Ebeid
# ==============================================================================

import argparse
import os
import platform
import subprocess
import sys

# --- Configuration ---
# This script uses a sequential installation process to ensure compatibility.
PYTHON_VERSION = "3.11"
# These versions are now used for documentation and to guide pip,
# while the core CUDA/cuDNN is installed first via conda.
CUDA_VERSION_FOR_PYTORCH = "12.1"
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

    # This command sequence follows the user-provided successful installation logic.
    # It installs components in a specific order to ensure a working, albeit complex, environment.
    command_sequence = [
        "conda clean --all -y",
        "conda update --all -y",
        *compiler_commands,

        # --- Stage 1: Install CUDA Toolkit and cuDNN from Conda ---
        "echo '--- Stage 1: Installing CUDA Toolkit and cuDNN from nvidia channel ---'",
        # FIX: Use the correct package name 'cuda' instead of 'cudatoolkit' for the nvidia channel.
        "conda install -c nvidia -c conda-forge -y cuda=12.5 cudnn=9.3",

        # --- Stage 2: Install TensorFlow from Conda ---
        # This uses conda-forge, which is known to work in this sequence.
        "echo '--- Stage 2: Installing TensorFlow from conda-forge ---'",
        "conda install -c conda-forge -y tensorflow",

        # --- Stage 3: Install PyTorch, PyG, and other pip-first packages ---
        # This will use pip and may install its own CUDA libraries, leading to the
        # non-fatal cuDNN factory warnings, but results in a working setup.
        "echo '--- Stage 3: Installing PyTorch, PyG, and other pip-managed packages ---'",
        (
            f"pip install "
            f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "
            f"torch-geometric pyg_lib torch-scatter torch-sparse "
            f"mlflow transformers==4.41.2 tf-keras "
            f"--index-url https://download.pytorch.org/whl/cu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')} "
            f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')}.html"
        ),

        # --- Stage 4: Install remaining core libraries from Conda ---
        "echo '--- Stage 4: Installing remaining core libraries from conda-forge ---'",
        (
            "conda install -c conda-forge -y "
            "dask tqdm biopython matplotlib scipy scikit-learn "
            "gensim python-louvain seaborn pycuda networkx=3.2.1 "
            "pandas h5py"
        ),

        # --- Stage 5: Verification & Cleanup ---
        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence)
    run_script(script_file)
