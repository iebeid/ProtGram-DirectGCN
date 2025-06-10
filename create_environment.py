import subprocess
import sys
import os
import platform
import argparse


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 6)
    print(f"\n{border}")
    print(f"|| {title} ||")
    print(f"{border}")


def print_step(message):
    """Prints a formatted step message."""
    print(f"\n>>> {message}")


def run_command(command_list, check=True):
    """
    Runs a command, streams its output in real-time, and checks for errors.
    Uses subprocess.run for robustness.
    """
    print(f"    Running Command: {' '.join(command_list)}")
    try:
        # Popen is used here specifically to stream output in real-time
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(f"      {line}", end='')
        process.stdout.close()
        return_code = process.wait()
        if check and return_code != 0:
            raise subprocess.CalledProcessError(return_code, command_list)
        return True
    except FileNotFoundError as e:
        print(f"\n*** ERROR: Command not found: '{e.filename}'. Is Conda in your system's PATH? ***")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n*** ERROR: Command failed with exit code {e.returncode}. See output above. ***")
        sys.exit(1)
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***")
        sys.exit(1)


def get_conda_base_path():
    """Finds the base directory of the Conda installation."""
    print_step("Locating Conda installation...")
    try:
        return subprocess.check_output(['conda', 'info', '--base'], text=True).strip()
    except Exception:
        print("\n*** ERROR: Could not find Conda. Please ensure Conda is installed and its 'bin' directory is in your system's PATH. ***")
        sys.exit(1)


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a robust, self-contained Conda environment for ProtDiGCN.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    # --- Configuration ---
    python_version = "3.11"
    torch_version = "2.3.1"
    cuda_version_for_pytorch = "cu121"

    base_conda_packages = [f"python={python_version}", "pip"]
    if platform.system() == "Linux":
        base_conda_packages.append("gxx_linux-64")

    pyg_dependencies = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    other_pip_packages = ["pytorch-geometric", "tensorflow[and-cuda]", "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx",
        "gensim", "python-louvain", "transformers", "torch-geometric-signed-directed"]

    print_header(f"Starting Environment Setup for '{env_name}'")

    conda_base_path = get_conda_base_path()
    env_path = os.path.join(conda_base_path, 'envs', env_name)

    # --- Step 1: Create the minimal Conda environment ---
    print_step("STEP 1: Creating minimal Conda environment with Python & Compilers...")
    if os.path.exists(env_path):
        print(f"    Environment '{env_name}' already exists. Skipping creation.")
    else:
        run_command(["conda", "create", "--name", env_name, "-y", "-c", "conda-forge"] + base_conda_packages)
    print("--- Minimal environment is ready. ---")

    # --- Get the absolute path to the new environment's pip ---
    pip_exe = os.path.join(env_path, 'bin', 'pip')
    if not os.path.exists(pip_exe):
        print(f"*** FATAL ERROR: Cannot find pip executable at '{pip_exe}'. The environment was not created correctly. ***")
        sys.exit(1)

    # --- Step 2: Install PyTorch ---
    print_step("STEP 2: Installing PyTorch for CUDA 12.1...")
    run_command([pip_exe, "install", "--no-cache-dir", f"torch=={torch_version}", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cuda_version_for_pytorch}"])
    print("--- PyTorch installed successfully. ---")

    # --- Step 3: Install PyTorch Geometric dependencies ---
    print_step("STEP 3: Installing PyTorch Geometric dependencies...")
    run_command([pip_exe, "install", "--no-cache-dir"] + pyg_dependencies + ["-f", f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version_for_pytorch}.html"])
    print("--- PyG dependencies installed successfully. ---")

    # --- Step 4: Install TensorFlow and remaining packages ---
    print_step("STEP 4: Installing TensorFlow and all remaining packages...")
    run_command([pip_exe, "install", "--no-cache-dir"] + other_pip_packages)
    print("--- All remaining packages installed successfully. ---")

    # --- Step 5: Final Verification ---
    print_step("STEP 5: Verifying core libraries...")
    print("    Checking PyTorch...")
    run_command([os.path.join(env_path, 'bin', 'python'), "-c", "import torch; print(f'PyTorch version: {torch.__version__}'); print('GPU available:', torch.cuda.is_available())"])
    print("    Checking TensorFlow...")
    run_command([os.path.join(env_path, 'bin', 'python'), "-c", "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"])

    print_header(f"✅✅✅ Environment '{env_name}' created and verified! ✅✅✅")
    print("To activate and use your new environment, run:\n")
    print(f"    conda activate {env_name}\n")
