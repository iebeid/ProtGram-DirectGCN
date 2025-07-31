# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 26.0 (Definitive fix: Staged hybrid install with forced linker path)
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

        # STAGE 1: CONDA FOR THE CUDA FOUNDATION
        # Install the CUDA toolkit and compiler from the official nvidia channel.
        # This creates a stable, framework-agnostic base that other packages will link against.
        "echo '--- Stage 1: Installing CUDA Toolkit and core data science libraries from Conda ---'",
        (f"conda install -y "
         f"-c nvidia -c conda-forge "
         f"python={PYTHON_VERSION} "
         # CRITICAL: Use the correct package names 'cuda-toolkit' and 'cuda-compiler'.
         f"'cuda-toolkit={CUDA_VERSION}' 'cuda-compiler={CUDA_VERSION}' 'cudnn' "
         # Other data science libraries
         f"dask tqdm biopython matplotlib scipy scikit-learn gensim python-louvain seaborn pandas h5py pyyaml networkx=3.2.1"),

        # STAGE 2: PIP INSTALLATIONS FOR ML FRAMEWORKS
        # Install the main ML frameworks. They will find and use the CUDA toolkit provided by Conda.
        # This avoids the library conflicts seen when installing pytorch+cuda from conda and tensorflow from pip.
        "echo '--- Stage 2: Installing ML Frameworks (PyTorch, TensorFlow) via pip ---'",
        (f"pip install --no-cache-dir "
         f"tensorflow "
         f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} --extra-index-url https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"
        ),

        # STAGE 3: PYCUDA INSTALL
        # This must come after Stage 1. We add LD_LIBRARY_PATH to force the linker to use the Conda env's libraries.
        "echo '--- Stage 3: Building PyCUDA from source with forced library paths ---'",
        (f"# CRITICAL: Add LD_LIBRARY_PATH to force the linker to use the Conda env's libraries.\n"
         f"LD_LIBRARY_PATH=\"{conda_prefix}/lib:$LD_LIBRARY_PATH\" "
         f"CXXFLAGS=\"-std=c++14\" "
         f"PATH=\"{conda_prefix}/bin:$PATH\" "
         f"CUDA_HOME=\"{conda_prefix}\" "
         f"LDFLAGS=\"-L{conda_prefix}/lib\" "
         f"CPPFLAGS=\"-I{conda_prefix}/include\" "
         # Install build deps first, then pycuda itself.
         f"pip install --no-cache-dir pytools appdirs && "
         f"pip install --no-cache-dir --no-binary :all: --no-deps --no-use-pep517 pycuda"),

        # STAGE 4: Install remaining pip packages that depend on the frameworks
        "echo '--- Stage 4: Installing remaining pip packages (MLflow, Transformers, PyG) ---'",
        "pip install --no-cache-dir tf-keras mlflow transformers==4.41.2",
        (f"pip install torch-geometric pyg_lib torch-scatter torch-sparse "
         f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION.replace('.', '')}.html"),

        # STAGE 5: VERIFICATION & CLEANUP
        "echo '--- Verifying installations ---'",
        # Use the robust, combined, multi-line test
        (f"python -c '\n"
         f"import sys\n"
         f"print(\"--- Verifying GPU Libraries ---\")\n"
         f"try:\n"
         f"    import torch\n"
         f"    print(\"\\n--- PyTorch ---\")\n"
         f"    is_avail = torch.cuda.is_available()\n"
         f"    print(f\"CUDA Available: {{is_avail}}\")\n"
         f"    if is_avail: print(f\"Device Name: {{torch.cuda.get_device_name(0)}}\")\n"
         f"except Exception as e: print(f\"\\n--- PyTorch ---\\nERROR: {{e}}\")\n"
         f"try:\n"
         f"    import tensorflow as tf\n"
         f"    print(\"\\n--- TensorFlow ---\")\n"
         f"    gpus = tf.config.list_physical_devices(\"GPU\")\n"
         f"    print(f\"GPUs Found: {{len(gpus)}}\")\n"
         f"except Exception as e: print(f\"\\n--- TensorFlow ---\\nERROR: {{e}}\")\n"
         f"try:\n"
         f"    import pycuda.autoinit\n"
         f"    print(\"\\n--- PyCUDA ---\")\n"
         f"    print(\"PyCUDA initialized successfully.\")\n"
         f"except Exception as e: print(f\"\\n--- PyCUDA ---\\nERROR: {{e}}\")\n"
         f"'"
        ),
        "echo '--- Stage 5: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)