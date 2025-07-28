# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 10.0 (Corrected order of compiler installation and environment export)
# AUTHOR: Islam Ebeid
# ==============================================================================

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
ENV_NAME = "ppi-env"  # The name of the conda environment
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

    # system = platform.system()
    #
    # # --- DEFINITIVE FIX: Correct the order of operations ---
    # compiler_install_commands = []
    # compiler_export_commands = []
    # if system == "Linux":
    #     compiler_install_commands.extend([
    #         "echo '--- Installing GCC/G++ compilers for Linux (required by PyCUDA) ---'",
    #         "conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y",
    #     ])
    #     compiler_export_commands.extend([
    #         "echo '--- Exporting compiler paths for the installation process ---'",
    #         'export CC="$CONDA_PREFIX/bin/gcc"',
    #         'export CXX="$CONDA_PREFIX/bin/g++"',
    #         "echo '--- Clearing PyCUDA cache to prevent stale compiler paths ---'",
    #         "rm -rf ~/.config/pycuda"
    #     ])

    pycuda_cache_clear_command = ["echo '--- Clearing PyCUDA cache ---'", "rm -rf ~/.config/pycuda"]

    command_sequence = [
        # 1. Source the main conda script to make 'conda activate' available
        f'source "{conda_base}/etc/profile.d/conda.sh"',
        # 2. Activate the specific environment for this project
        f'conda activate {ENV_NAME}',

        # 3. General cleanup first
        "conda clean --all -y",
        "conda update --all -y",

        # # 4. Install the compilers FIRST.
        # *compiler_install_commands,
        #
        # # 5. NOW that the compilers exist, export their paths.
        # *compiler_export_commands,

        *pycuda_cache_clear_command,

        # 6. Proceed with all other installation commands, which will now inherit the correct environment.
        "echo '--- Stage 1a: Installing GPU drivers and core TensorFlow ---'",
        "conda install -c nvidia -c conda-forge -y cuda=12.5 cudnn=9.3 tensorflow",

        "echo '--- Stage 1b: Installing core data science and utility libraries ---'",
        ("conda install -c conda-forge -y "
         "dask tqdm biopython matplotlib scipy scikit-learn "
         "gensim python-louvain seaborn pycuda networkx=3.2.1 "
         "pandas h5py pyyaml"),

        "echo '--- Stage 2: Installing PyTorch and other standard pip-managed packages ---'",
        (f"pip install "
         f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "
         f"mlflow transformers==4.41.2 tf-keras "
         f"--extra-index-url https://download.pytorch.org/whl/cu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')}"),

        "echo '--- Stage 3: Installing PyTorch Geometric (PyG) ---'",
        (f"pip install torch-geometric pyg_lib torch-scatter torch-sparse "
         f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION_FOR_PYTORCH.replace('.', '')}.html"),

        "echo '--- Verifying installations ---'",
        "python -c \"import tensorflow as tf; print('TensorFlow GPUs found: ' + str(len(tf.config.list_physical_devices('GPU'))))\"",
        "python -c \"import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))\"",

        "echo '--- Stage 4: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',

        "conda clean --all -y",
        "pip cache purge"
    ]
    # --- END FIX ---

    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)