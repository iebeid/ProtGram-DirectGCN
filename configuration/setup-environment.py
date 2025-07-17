import os
import subprocess
import sys

# --- Configuration for a Stable Environment ---
ENV_NAME = "ppi-env"
PYTHON_VERSION = "3.11"
# We guide conda with the major CUDA version and let it find the compatible packages.
CUDA_VERSION = "12.1"


def create_environment_yaml():
    """Creates a more flexible and solvable environment.yml file."""
    yaml_content = f"""
name: {ENV_NAME}
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  # --- Core Python and GPU Setup ---
  - python={PYTHON_VERSION}
  - pytorch-cuda={CUDA_VERSION} # This is the key: let this package manage cudnn and cudatoolkit

  # --- Frameworks ---
  # Conda will find compatible versions of PyTorch, torchvision, and TensorFlow
  - pytorch
  - torchvision
  - torchaudio
  - tensorflow
  - torch-geometric

  # --- Core Libraries ---
  - dask
  - tqdm
  - biopython
  - matplotlib
  - scipy
  - scikit-learn
  - mlflow
  - transformers=4.41.2 # Keep this pinned for stability
  - gensim
  - python-louvain
  - seaborn
  - pycuda
  - networkx=3.2.1 # Keep this pinned to avoid known issues
  - pip

  # --- Pip for packages not well-supported on Conda ---
  - pip:
    - tf-keras
    - pyarrow
    - h5py
"""
    with open("environment.yml", "w") as f:
        f.write(yaml_content)
    print("--- Successfully created a flexible environment.yml file. ---")


def run_setup():
    """Installs Mamba and creates the new conda environment."""
    try:
        print("\n--- Ensuring Mamba is installed in the base environment... ---")
        subprocess.run(
            ["conda", "install", "-n", "base", "-c", "conda-forge", "mamba", "-y"],
            check=True,
            capture_output=True
        )
        print("--- Mamba is ready. ---")

        print(f"\n--- Creating the '{ENV_NAME}' environment with Mamba... ---")
        print("--- This will be much faster and more reliable. ---")
        subprocess.run(["mamba", "env", "create", "-f", "environment.yml"], check=True)

        print("\n" + "=" * 80)
        print("🎉 Environment setup with Mamba completed successfully! 🎉")
        print("\nTo activate and use this environment, run the following command:")
        print(f"conda activate {ENV_NAME}")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR: The setup process failed. ---")
        print("--- Please check the error messages above. The environment may be partially created. ---")
        print("--- It's recommended to run 'conda env remove -n ppi-env' before trying again. ---")
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