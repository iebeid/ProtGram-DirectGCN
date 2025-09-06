# ==============================================================================
# MODULE: configuration/setup.py
# PURPOSE: Sets up the Python environment and generates a validation file.
# VERSION: 29.0 (Ensures a clean data setup by removing old cache)
# AUTHOR: Islam Ebeid
# NOTE: This setup script is designed for Unix-like environments (Linux, macOS, WSL)
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
PYTORCH_VERSION = "2.1.2"
TORCHVISION_VERSION = "0.16.2"
TORCHAUDIO_VERSION = "2.1.2"
ENVIRONMENT_YML_FILE = "environment.yml"


# --- End Configuration ---

def create_setup_script(commands: list[str], project_root: Path) -> Path:
    """Creates a shell script from a list of commands for Unix-like systems."""
    script_path = project_root / "temp_setup_script.sh"

    with open(script_path, "w", encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        for command in commands:
            f.write(command + "\n")

    os.chmod(script_path, 0o755)
    return script_path

def run_script(script_path: Path):
    """Executes the setup script and streams its output."""
    print(f"--- Starting Environment Setup using temporary script: '{script_path.name}' ---")
    try:
        command_to_run = [str(script_path)]

        process = subprocess.Popen(
            command_to_run,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=script_path.parent
        )

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

    # --- DEFINITIVE FIX: Use sys.prefix to robustly find the environment path ---
    # This is more reliable than os.environ.get("CONDA_PREFIX") because it works
    # even when the script is called directly by the env's python executable
    # from a non-activated shell.
    conda_prefix = sys.prefix
    if not conda_prefix:
        print(f"FATAL ERROR: Could not determine Conda environment prefix from sys.prefix: {conda_prefix}")
        print("This script must be run using the Python executable from the target Conda environment.")
        sys.exit(1)

    print(f"--- Using Conda prefix for library paths: {conda_prefix} ---")

    activate_script_content = (
        'export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"\\n'
        'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"'
    )
    deactivate_script_content = (
        'export LD_LIBRARY_PATH="${OLD_LD_LIBRARY_PATH}"\\n'
        'unset OLD_LD_LIBRARY_PATH'
    )

    command_sequence = [
        f'source "{conda_base}/etc/profile.d/conda.sh"',
        f'conda activate {ENV_NAME}',
        "conda clean --all -y",
        "echo '--- Clearing PyCUDA cache to ensure rediscovery of system compiler ---'",
        "rm -rf ~/.config/pycuda",

        # --- ANTICIPATORY DEBUGGING: Force update of certificate store ---
        "echo '--- Stage 0.5: Updating SSL certificate bundle to prevent verification errors ---'",
        "pip install --no-cache-dir --upgrade certifi",

        # STAGE 1: CONDA FOR THE CUDA FOUNDATION
        "echo '--- Stage 1: Installing CUDA Toolkit and core data science libraries from Conda ---'",
        "MAX_RETRIES=3",
        "COUNT=0",
        "until " + (f"conda install -y "
                    f"-c nvidia -c conda-forge "
                    f"python={PYTHON_VERSION} "
                    f"'cuda-toolkit={CUDA_VERSION}' 'cuda-compiler={CUDA_VERSION}' 'cudnn=8.9' "
                    f"dask tqdm biopython matplotlib 'scipy<1.14.0' scikit-learn gensim python-louvain seaborn pandas h5py pyyaml networkx=3.2.1"
                    ) + "; do",
        "    COUNT=$((COUNT+1))",
        "    if [ \"$COUNT\" -ge \"$MAX_RETRIES\" ]; then",
        "        echo \"Conda install failed after $MAX_RETRIES attempts. Aborting.\"",
        "        exit 1",
        "    fi",
        "    echo \"Conda install failed due to a likely network issue. Retrying in 5 seconds... (Attempt $((COUNT+1))/$MAX_RETRIES)\"",
        "    sleep 5",
        "done",

        # STAGE 2: PIP INSTALLATIONS FOR ML FRAMEWORKS
        "echo '--- Stage 2: Installing ML Frameworks (PyTorch, TensorFlow) via pip ---'",
        (f"pip install --no-cache-dir "
         f"\"tensorflow<2.16\" tf-keras "
         f"torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} --extra-index-url https://download.pytorch.org/whl/cu{CUDA_VERSION.replace('.', '')}"
         ),
        "echo '--- Stage 2.5: Forcing library consistency by removing ALL pip-installed CUDA libs ---'",
        # --- DEFINITIVE FIX: Dynamically generate the uninstall command ---
        # This makes the script robust to changes in the CUDA_VERSION variable.
        f"CUDA_SUFFIX=cu{CUDA_VERSION.split('.')[0]}",
        "PACKAGES_TO_UNINSTALL=("
        "    nvidia-cudnn-$CUDA_SUFFIX nvidia-cublas-$CUDA_SUFFIX nvidia-cufft-$CUDA_SUFFIX "
        "    nvidia-curand-$CUDA_SUFFIX nvidia-cusolver-$CUDA_SUFFIX nvidia-cusparse-$CUDA_SUFFIX "
        "    nvidia-nccl-$CUDA_SUFFIX nvidia-nvtx-$CUDA_SUFFIX nvidia-cuda-nvrtc-$CUDA_SUFFIX "
        "    nvidia-cuda-runtime-$CUDA_SUFFIX nvidia-cuda-cupti-$CUDA_SUFFIX nvidia-nvjitlink-$CUDA_SUFFIX"
        ")",
        "pip uninstall -y ${PACKAGES_TO_UNINSTALL[@]} 2>/dev/null || true",

        # STAGE 3: PYCUDA INSTALL
        "echo '--- Stage 3: Building PyCUDA from source ---'",
        (f"pip install --no-cache-dir pytools appdirs && "
         f"pip install --no-cache-dir --no-binary :all: --no-deps --no-use-pep517 pycuda"),

        # STAGE 4: Install remaining pip packages
        "echo '--- Stage 4: Installing remaining pip packages (MLflow, Transformers, PyG) ---'",
        # --- DEFINITIVE FIX for ModuleNotFoundError: Add the optuna-integration package ---
        # This package is now required for MLflow callbacks.
        "pip install --no-cache-dir optuna mlflow gdown 'transformers==4.41.2' 'safetensors==0.4.3' 'optuna-integration[mlflow]'",
        (f"pip install torch-geometric pyg_lib torch-scatter torch-sparse torch-cluster "
         f"-f https://data.pyg.org/whl/torch-{PYTORCH_VERSION}%2Bcu{CUDA_VERSION.replace('.', '')}.html"),

        # STAGE 5: Create Conda activation scripts for LD_LIBRARY_PATH
        "echo '--- Stage 5: Creating Conda activation scripts for library paths ---'",
        f'mkdir -p "{conda_prefix}/etc/conda/activate.d"',
        f'mkdir -p "{conda_prefix}/etc/conda/deactivate.d"',
        f'printf \'{activate_script_content}\' > "{conda_prefix}/etc/conda/activate.d/env_vars.sh"',
        f'printf \'{deactivate_script_content}\' > "{conda_prefix}/etc/conda/deactivate.d/env_vars.sh"',
        f'chmod +x "{conda_prefix}/etc/conda/activate.d/env_vars.sh"',
        f'chmod +x "{conda_prefix}/etc/conda/deactivate.d/env_vars.sh"',

        # STAGE 7: VERIFICATION & CLEANUP
        "echo '--- Stage 7: Verifying installations ---'",
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
        "echo '--- Stage 7: Generating environment validation file ---'",
        f'conda env export > "{env_yml_output_path}"',
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence, project_root)
    run_script(script_file)
