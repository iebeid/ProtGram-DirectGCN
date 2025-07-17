import os
import subprocess
import sys
import platform

# --- Configuration for a Stable, Hybrid Environment ---
ENV_NAME = "ppi-env"
PYTHON_VERSION = "3.11"
CUDA_VERSION = "12.1"  # For PyTorch and other conda packages


def create_environment_yaml():
    """
    Creates a robust environment.yml file WITHOUT TensorFlow.
    TensorFlow will be installed via pip in a second step to avoid conflicts.
    """
    yaml_content = f"""
name: {ENV_NAME}
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  # --- Core Python and GPU Setup for PyTorch ---
  - python={PYTHON_VERSION}
  - pytorch-cuda={CUDA_VERSION}
  - pytorch
  - torchvision
  - torchaudio
  - torch-geometric

  # --- Other Core Libraries ---
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
  - h5py
  - pyarrow
"""
    with open("environment.yml", "w") as f:
        f.write(yaml_content)
    print("--- Successfully created a flexible environment.yml for the base environment. ---")


def run_setup():
    """
    Creates the conda environment using Mamba, then runs a second script
    to activate it and pip install TensorFlow.
    """
    try:
        # Step 1: Ensure Mamba is installed
        print("\n--- Ensuring Mamba is installed in the base environment... ---")
        subprocess.run(
            ["conda", "install", "-n", "base", "-c", "conda-forge", "mamba", "-y"],
            check=True,
            capture_output=True
        )
        print("--- Mamba is ready. ---")

        # Step 2: Create the base environment using Mamba
        print(f"\n--- Creating the base '{ENV_NAME}' environment with Mamba (without TensorFlow)... ---")
        subprocess.run(["mamba", "env", "create", "-f", "environment.yml"], check=True)
        print(f"--- Base environment '{ENV_NAME}' created successfully. ---")

        # Step 3: Pip install TensorFlow and other specific packages inside the new environment
        print(f"\n--- Installing TensorFlow and pip dependencies into '{ENV_NAME}'... ---")

        # This command runs the pip installation using the python executable from the new environment
        conda_python_path = os.path.join(os.environ['CONDA_PREFIX'], 'envs', ENV_NAME, 'bin', 'python')

        pip_commands = [
            "-m", "pip", "install",
            "tf-keras",
            "\"tensorflow[and-cuda]==2.19.0\""  # Use the official pip package for TF + CUDA
        ]

        subprocess.run([conda_python_path] + pip_commands, check=True)

        print("\n" + "=" * 80)
        print("🎉 Full environment setup completed successfully! 🎉")
        print("\nTo activate and use this environment, run the following command:")
        print(f"conda activate {ENV_NAME}")
        print("=" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR: The setup process failed. ---")
        print(f"--- A command returned a non-zero exit code: {e.returncode} ---")
        print("--- Please check the error messages above. The environment may be partially created. ---")
        print(f"--- It's recommended to run 'conda env remove -n {ENV_NAME}' before trying again. ---")
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
        return False


if __name__ == "__main__":
    if not check_conda_installed():
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        sys.exit(1)

    create_environment_yaml()
    run_setup()