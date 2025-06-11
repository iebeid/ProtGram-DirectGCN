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
    """
    print(f"    Running Command: {' '.join(command_list)}")
    try:
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

    # These are the specialized dependencies that live at the PyG URL
    pyg_dependencies = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]

    # This is the corrected list of packages for the final installation step
    other_pip_packages = ["torch-geometric",  # Corrected package name
        "tensorflow[and-cuda]", "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx", "gensim", "python-louvain", "transformers",
        "torch-geometric-signed-directed"]

    print_header(f"Final Robust GPU Environment Setup for '{env_name}'")

    conda_base_path = get_conda_base_path()
    env_path = os.path.join(conda_base_path, 'envs', env_name)
    pip_exe = os.path.join(env_path, 'bin', 'pip')

    # --- Step 1: Create the minimal Conda environment ---
    print_step("STEP 1: Creating minimal Conda environment...")
    if os.path.exists(env_path):
        print(f"    Environment '{env_name}' already exists. Skipping creation.")
    else:
        run_command(["conda", "create", "--name", env_name, "-y", "-c", "conda-forge"] + base_conda_packages)
    print("--- Minimal environment is ready. ---")

    # --- Step 2: Install PyTorch ---
    print_step("STEP 2: Installing PyTorch...")
    run_command([pip_exe, "install", "--no-cache-dir", f"torch=={torch_version}", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cuda_version_for_pytorch}"])
    print("--- PyTorch installed successfully. ---")

    # --- Step 3: Install PyG dependencies from the special URL ---
    print_step("STEP 3: Installing PyTorch Geometric dependencies from special source...")
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version_for_pytorch}.html"
    run_command([pip_exe, "install", "--no-cache-dir", "-f", pyg_url] + pyg_dependencies)
    print("--- PyG dependencies installed successfully. ---")

    # --- Step 4: Install all remaining packages from the standard PyPI ---
    print_step("STEP 4: Installing TensorFlow, PyG, and all remaining packages...")
    run_command([pip_exe, "install", "--no-cache-dir"] + other_pip_packages)
    print("--- All remaining packages installed successfully. ---")

    print_header(f"✅✅✅ Environment '{env_name}' created successfully! ✅✅✅")
    print("To activate and use your new environment, run:\n")
    print(f"    conda activate {env_name}\n")
