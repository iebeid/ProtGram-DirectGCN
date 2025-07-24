# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 8.1 (Harmonized PyTorch/TensorFlow CUDA dependencies via Conda)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
PYTHON_VERSION = "3.11"
CUDA_VERSION_FOR_PYTORCH = "12.1"
PYTORCH_VERSION = "2.4.0"
TORCHVISION_VERSION = "0.19.0"
TORCHAUDIO_VERSION = "2.4.0"
# The name of the file that will store the environment's "fingerprint".
ENVIRONMENT_YML_FILE = "environment.yml"


# --- End Configuration ---

def create_setup_script(commands: list[str], project_root: Path) -> Path:
    """
    Creates a platform-specific shell script from a list of commands.
    This approach is used to ensure that a sequence of shell commands can be
    executed reliably across different platforms (Windows vs. Linux/macOS).
    """
    is_windows = platform.system() == "Windows"
    script_extension = ".bat" if is_windows else ".sh"
    # Create the script in the project root to ensure consistent execution paths.
    script_path = project_root / f"temp_setup_script{script_extension}"

    with open(script_path, "w", encoding='utf-8') as f:
        if not is_windows:
            f.write("#!/bin/bash\n")
            # 'set -e' ensures the script will exit immediately if a command fails.
            f.write("set -e\n")
        for command in commands:
            f.write(command + "\n")

    if not is_windows:
        # Make the script executable on Unix-like systems.
        os.chmod(script_path, 0o755)
    return script_path


def run_script(script_path: Path):
    """Executes the setup script and streams its output."""
    is_windows = platform.system() == "Windows"
    print(f"--- Starting Environment Setup using temporary script: '{script_path.name}' ---")
    try:
        # On Windows, 'cmd /c' is needed to run .bat files.
        # On Linux/macOS, we execute the .sh file directly.
        executor = ['cmd', '/c'] if is_windows else []
        command_to_run = executor + [str(script_path)]

        # Use Popen to stream output in real-time.
        process = subprocess.Popen(
            command_to_run,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True,
            cwd=script_path.parent  # Ensure script runs from the project root
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
        process.wait()

        if process.returncode != 0:
            print(f"\n--- Script failed with exit code {process.returncode}. Please check the logs above. ---")
            sys.exit(process.returncode)
        else:
            print("\n--- Environment setup completed successfully! ---")

    finally:
        # Always clean up the temporary script file.
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"--- Cleaned up temporary script file: {script_path.name} ---")


def check_conda_installed() -> bool:
    """Checks if conda is installed and available in the system's PATH."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True, text=True, shell=False)
        print("--- Conda is installed and detected. ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        print("--- Please install Anaconda/Miniconda and ensure it's activated. ---")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install required packages into the active Conda environment.")
    parser.parse_args()
    if not check_conda_installed():
        sys.exit(1)

    # Determine the project root from this script's location.
    # This script is in 'configuration/', so the root is its parent's parent.
    project_root = Path(__file__).parent.parent.resolve()
    config_dir = project_root / "configuration"

    system = platform.system()
    compiler_commands = []
    if system == "Linux":
        compiler_commands.extend([
            "echo '--- Installing GCC/G++ compilers for Linux (required by PyCUDA) ---'",
            "conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y",
            "echo '--- Clearing PyCUDA cache to prevent stale compiler paths ---'",
            "rm -rf ~/.config/pycuda"
        ])

    # Define the full path for the output environment file.
    # Using a full path in the command makes it robust, regardless of where the script runs.
    env_yml_output_path = config_dir / ENVIRONMENT_YML_FILE

    # --- FIX: Install PyTorch and PyG via Conda to harmonize CUDA/cuDNN dependencies ---
    command_sequence = [
        "conda clean --all -y",
        "conda update --all -y",
        *compiler_commands,

        "echo '--- Stage 1: Installing PyTorch, TensorFlow, and all CUDA/cuDNN dependencies via Conda ---'",
        (
            f"conda install -c pytorch -c nvidia -c conda-forge -y "
            f"python={PYTHON_VERSION} "
            f"pytorch={PYTORCH_VERSION} torchvision={TORCHVISION_VERSION} torchaudio={TORCHAUDIO_VERSION} "
            f"pytorch-cuda={CUDA_VERSION_FOR_PYTORCH} "
            f"tensorflow cudnn=9.3 " # Explicitly request cuDNN 9.3 for TF
            f"pyg" # PyG will be pulled from conda-forge
        ),

        "echo '--- Stage 2: Installing remaining packages ---'",
        (
            "conda install -c conda-forge -y "
            "dask tqdm biopython matplotlib scipy scikit-learn "
            "gensim python-louvain seaborn pycuda networkx=3.2.1 "
            "pandas h5py pyyaml"
        ),

        "echo '--- Stage 3: Installing pip-only packages ---'",
        (
            "pip install mlflow transformers==4.41.2 tf-keras"
        ),

        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",

        "echo '--- Stage 4: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',

        "conda clean --all -y",
        "pip cache purge"
    ]
    # --- END FIX ---

    # Create and run the setup script from the project root for consistency.
    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)