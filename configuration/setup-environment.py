import os
import subprocess
import sys

# --- Configuration for a Stable Environment ---
ENV_NAME = "ppi-env"
PYTHON_VERSION = "3.11"
# We select versions of CUDA, PyTorch, and TensorFlow that are known to be compatible.
# Conda and Mamba will handle the specific library versions.
CUDA_VERSION = "12.1"
PYTORCH_VERSION = "2.4.0"
TENSORFLOW_VERSION = "2.16.1"
TORCH_GEOMETRIC_VERSION = "2.6.0"


def create_environment_yaml():
    """Creates the environment.yml file for a robust conda/mamba setup."""
    yaml_content = f"""
name: {ENV_NAME}
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python={PYTHON_VERSION}
  - pytorch={PYTORCH_VERSION}
  - torchvision
  - torchaudio
  - pytorch-cuda={CUDA_VERSION}
  - tensorflow-gpu={TENSORFLOW_VERSION}
  - torch-geometric={TORCH_GEOMETRIC_VERSION}
  - cudatoolkit={CUDA_VERSION}
  - cudnn
  - dask
  - tqdm
  - biopython
  - matplotlib
  - scipy
  - scikit-learn
  - mlflow
  - transformers=4.41.2
  - gensim
  - python-louvain
  - seaborn
  - pycuda
  - networkx=3.2.1
  - pip
  - pip:
    - tf-keras
    - pyarrow
    - h5py
"""
    with open("environment.yml", "w") as f:
        f.write(yaml_content)
    print("--- Successfully created environment.yml file. ---")


def run_setup():
    """Installs Mamba and then creates the new conda environment using Mamba."""
    try:
        # Step 1: Install Mamba into the base conda environment if it's not there.
        print("\n--- Checking for Mamba and installing if necessary... ---")
        subprocess.run(
            ["conda", "install", "-n", "base", "-c", "conda-forge", "mamba", "-y"],
            check=True,
            capture_output=True  # Hide output unless there's an error
        )
        print("--- Mamba is installed. Proceeding with environment creation. ---")

        # Step 2: Use Mamba to create the environment. It's much faster.
        print(f"\n--- Creating the '{ENV_NAME}' environment with Mamba... ---")
        print("--- This should be significantly faster than using the standard conda solver. ---")
        subprocess.run(["mamba", "env", "create", "-f", "environment.yml"], check=True)

        print("\n" + "=" * 80)
        print("🎉 Environment setup with Mamba completed successfully! 🎉")
        print("\nTo activate and use this environment, run the following command:")
        print(f"conda activate {ENV_NAME}")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR: The setup process failed. ---")
        print(f"--- The command returned a non-zero exit code: {e.returncode} ---")
        print("--- Please check the error messages above. ---")
        sys.exit(1)
    except FileNotFoundError:
        print("--- ERROR: 'conda' command not found. Please ensure Conda is installed and in your PATH. ---")
        sys.exit(1)


def check_conda_installed():
    """Checks if conda is installed."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        return False


if __name__ == "__main__":
    if not check_conda_installed():
        sys.exit(1)

    create_environment_yaml()
    run_setup()