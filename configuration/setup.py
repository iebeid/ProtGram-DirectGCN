# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 24.0 (Definitive fix: Unified Conda install from pytorch channel)
# AUTHOR: Islam Ebeid
# ==============================================================================

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
ENV_NAME = "ppi-env"
PYTHON_VERSION = "3.11"
CUDA_VERSION = "12.1"
PYTORCH_VERSION = "2.4.0"
TORCHVISION_VERSION = "0.19.0"
TORCHAUDIO_VERSION = "2.4.0"
ENVIRONMENT_YML_FILE = "environment.yml"


# --- End Configuration ---

def create_setup_script(commands: list[str], project_root: Path) -> Path:
    """Creates a platform-specific shell script from a list of commands."""
    is_windows = platform.system() == "Windows"
    script_extension = ".bat" if is_windows else ".sh"
    script_path = project_root / f"temp_setup_script{script_extension}"

    with open(script_path, "w", encoding='utf-8') as f:
        if not is_windows:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
        for command in commands:
            f.write(command + "\n")

    if not is_windows:
        os.chmod(script_path, 0o755)
    return script_path


def run_script(script_path: Path):
    """Executes the setup script and streams its output."""
    is_windows = platform.system() == "Windows"
    print(f"--- Starting Environment Setup using temporary script: '{script_path.name}' ---")
    try:
        executor = ['cmd', '/c'] if is_windows else []
        command_to_run = executor + [str(script_path)]

        process = subprocess.Popen(
            command_to_run,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True,
            cwd=script_path.parent
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
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"--- Cleaned up temporary script file: {script_path.name} ---")


def get_conda_base_path() -> str | None:
    """Checks if conda is installed and returns the base path if found."""
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            check=True, capture_output=True, text=True, shell=False
        )
        conda_base_path = result.stdout.strip()
        print(f"--- Conda is installed and detected. ---")
        return conda_base_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        print("--- Please install Anaconda/Miniconda and ensure it's activated. ---")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install required packages into the active Conda environment.")
    parser.parse_args()

    conda_base = get_conda_base_path()
    if not conda_base:
        sys.exit(1)

    project_root = Path(__file__).parent.parent.resolve()
    config_dir = project_root / "configuration"
    env_yml_output_path = config_dir / ENVIRONMENT_YML_FILE

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("FATAL ERROR: CONDA_PREFIX environment variable not found.")
        print("This script must be run from within an activated conda environment.")
        sys.exit(1)

    print(f"--- Using Conda prefix for library paths: {conda_prefix} ---")

    command_sequence = [
        f'source "{conda_base}/etc/profile.d/conda.sh"',
        f'conda activate {ENV_NAME}',
        "conda clean --all -y",
        "echo '--- Clearing PyCUDA cache to ensure rediscovery of system compiler ---'",
        "rm -rf ~/.config/pycuda",

        # STAGE 1: UNIFIED CONDA INSTALL
        # This is the most robust method. We let Conda's solver handle the complex dependencies
        # between PyTorch, its specific CUDA toolkit, and the compiler from the correct channels.
        "echo '--- Stage 1: Unified Conda installation for all core packages ---'",
        (f"conda install -y "
         # CRITICAL: Prioritize the pytorch channel, then nvidia, then conda-forge.
         f"-c pytorch -c nvidia -c conda-forge "
         f"python={PYTHON_VERSION} "
         # Install PyTorch, its CUDA toolkit, and other libraries in one go.
         # `pytorch-cuda` is a meta-package that ensures the correct CUDA runtime and cuDNN are installed.
         f"pytorch={PYTORCH_VERSION} torchvision={TORCHVISION_VERSION} torchaudio={TORCHAUDIO_VERSION} pytorch-cuda={CUDA_VERSION} "
         # Other data science libraries
         f"dask tqdm biopython matplotlib scipy scikit-learn gensim python-louvain seaborn pandas h5py pyyaml networkx=3.2.1"),

        # STAGE 2: PIP INSTALLATIONS
        # Install packages not available on Conda or that need specific versions.
        # TensorFlow will now find the consistent CUDA toolkit installed by Conda in Stage 1.
        "echo '--- Stage 2: Installing remaining packages via pip ---'",
        "pip install --no-cache-dir tensorflow tf-keras mlflow transformers==4.41.2",

        # STAGE 3: PYCUDA INSTALL
        # This must come after the main conda install, which provides the CUDA compiler.
        "echo '--- Stage 3: Building PyCUDA from source ---'",
        (f"CXXFLAGS=\"-std=c++14\" "
         f"PATH=\"{conda_prefix}/bin:$PATH\" "
         f"CUDA_HOME=\"{conda_prefix}\" "
         f"LDFLAGS=\"-L{conda_prefix}/lib\" "
         f"CPPFLAGS=\"-I{conda_prefix}/include\" "
         # Install build deps first, then pycuda itself.
         f"pip install --no-cache-dir pytools appdirs && "
         f"pip install --no-cache-dir --no-binary :all: --no-deps --no-use-pep517 pycuda"),

        # STAGE 4: PYG INSTALL
        "echo '--- Stage 4: Installing PyTorch Geometric (PyG) ---'",
        (f"pip install torch-geometric pyg_lib torch-scatter torch-sparse "
         f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION.replace('.', '')}.html"),

        # STAGE 5: VERIFICATION & CLEANUP
        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",
        "python -c \"import pycuda.autoinit; print('PyCUDA initialized successfully.')\"",
        "echo '--- Stage 5: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)