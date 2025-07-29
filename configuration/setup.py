# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 17.0 (Final WSL fix: Use LDFLAGS/CPPFLAGS for robust PyCUDA build)
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
CUDA_VERSION_FOR_PYTORCH = "12.1"
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

    # The run.py script ensures this script is run inside the activated environment.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("FATAL ERROR: CONDA_PREFIX environment variable not found.")
        print("This script must be run from within an activated conda environment.")
        sys.exit(1)

    print(f"--- Using Conda prefix for library paths: {conda_prefix} ---")

    command_sequence = [
        # 1. Activate the environment for a consistent session.
        f'source "{conda_base}/etc/profile.d/conda.sh"',
        f'conda activate {ENV_NAME}',

        # 2. General cleanup.
        "conda clean --all -y",
        "conda update --all -y",
        "echo '--- Clearing PyCUDA cache to ensure rediscovery of system compiler ---'",
        "rm -rf ~/.config/pycuda",

        # 3. Install all non-GPU data science and utility libraries with Conda.
        "echo '--- Stage 1: Installing core data science and utility libraries ---'",
        ("conda install -c conda-forge -y "
         "dask tqdm biopython matplotlib scipy scikit-learn "
         "gensim python-louvain seaborn pandas h5py pyyaml networkx=3.2.1"),

        # 4. Install PyTorch and TensorFlow first. These packages bring the required CUDA toolkit.
        "echo '--- Stage 2a: Installing PyTorch & TensorFlow to provide CUDA toolkit ---'",
        (f"pip install "
         f"tensorflow "
         f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "
         f"--extra-index-url https://download.pytorch.org/whl/cu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')}"),

        # 5. CRITICAL STEP: Install PyCUDA by setting environment variables
        #    that the compiler and linker will use directly. We install its
        #    dependencies first to avoid PEP 517 conflicts.
        "echo '--- Stage 2b: Installing PyCUDA dependencies (pytools, appdirs) ---'",
        "pip install --no-cache-dir pytools appdirs",

        "echo '--- Stage 2c: Installing PyCUDA with forced library paths and legacy setup ---'",
        (f"CXXFLAGS=\"-std=c++14\" "  # Force a compatible C++ standard
         f"PATH=\"{conda_prefix}/bin:$PATH\" "
         f"CUDA_HOME=\"{conda_prefix}\" "
         f"LDFLAGS=\"-L{conda_prefix}/lib -L{conda_prefix}/lib/stubs\" "
         f"CPPFLAGS=\"-I{conda_prefix}/include\" "
         f"pip install --no-cache-dir --no-binary :all: --no-deps --no-use-pep517 pycuda"),

        # 6. Install other pip packages.
        "echo '--- Stage 2d: Installing remaining pip packages ---'",
        "pip install mlflow transformers==4.41.2 tf-keras",

        # 7. Install PyG, which depends on the PyTorch version just installed.
        "echo '--- Stage 3: Installing PyTorch Geometric (PyG) ---'",
        (f"pip install torch-geometric pyg_lib torch-scatter torch-sparse "
         f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')}.html"),

        # 8. Final verification and cleanup.
        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",
        "python -c \"import pycuda.autoinit; print('PyCUDA initialized successfully.')\"",
        "echo '--- Stage 4: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)