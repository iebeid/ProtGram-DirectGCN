import subprocess
import sys
import os
import platform
import argparse


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"=== {title} ===")
    print(f"{border}")


def print_step(current, total, description):
    """Prints a formatted step message."""
    print(f"\n[Step {current}/{total}] {description}")


def print_success(message):
    """Prints a success message."""
    print(f"\n--- {message} ---")


def print_error_and_exit(message, e=None):
    """Prints an error message and exits the script."""
    print(f"\n*** ERROR: {message} ***", file=sys.stderr)
    if e:
        print(f"*** Exception: {e} ***", file=sys.stderr)
    print("*** ABORTING SCRIPT. Please review the error messages above. ***", file=sys.stderr)
    sys.exit(1)


def run_command(command_list):
    """
    Runs a command specified as a list of arguments.
    Streams its output and checks for errors.
    """
    print(f"\n>--- Running Command: {' '.join(command_list)} ---")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\n*** Command failed with exit code {process.returncode} ***", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print_error_and_exit("An unexpected error occurred while running a command.", e)
        return False


def get_conda_base_prefix():
    """Finds the base directory of the Conda installation."""
    try:
        result = subprocess.run(['conda', 'info', '--base'], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception as e:
        print_error_and_exit("Could not determine Conda base directory. Is Conda installed and in your PATH?", e)


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Conda environment for ProtDiGCN using a robust, step-by-step method.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    # --- Configuration ---
    python_version = "3.11"
    torch_version = "2.3.1"
    cuda_version_for_pytorch = "cu121"

    base_conda_packages = [f"python={python_version}", "pip", "gxx_linux-64"]
    pyg_dependencies = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    other_pip_packages = ["pytorch-geometric", "tensorflow[and-cuda]", "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx",
        "gensim", "python-louvain", "transformers", "torch-geometric-signed-directed"]

    print_header(f"Robust GPU Environment Setup for '{env_name}'")

    conda_base_path = get_conda_base_prefix()
    TOTAL_STEPS = 5

    # --- Step 1: Create the minimal Conda environment ---
    print_step(1, TOTAL_STEPS, "Creating minimal Conda environment with Python and compilers...")
    conda_create_cmd = ["conda", "create", "--name", env_name, "-y", "-c", "conda-forge"] + base_conda_packages
    if not run_command(conda_create_cmd):
        print_error_and_exit("Failed to create the base Conda environment.")
    print_success("Base environment created successfully!")

    # --- Get the absolute path to the new environment's pip ---
    pip_exe_path = os.path.join(conda_base_path, 'envs', env_name, 'bin', 'pip')

    # --- Step 2: Install PyTorch using the new pip executable ---
    print_step(2, TOTAL_STEPS, "Installing PyTorch for CUDA 12.1...")
    torch_install_cmd = [pip_exe_path, "install", "--no-cache-dir", f"torch=={torch_version}", "torchvision", "torchaudio", "--index-url", f"https://download.pytorch.org/whl/{cuda_version_for_pytorch}"]
    if not run_command(torch_install_cmd):
        print_error_and_exit("Failed to install PyTorch.")
    print_success("PyTorch installed successfully!")

    # --- Step 3: Install PyG dependencies from their official wheel index ---
    print_step(3, TOTAL_STEPS, "Installing PyTorch Geometric dependencies...")
    pyg_install_cmd = [pip_exe_path, "install", "--no-cache-dir"] + pyg_dependencies + ["-f", f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version_for_pytorch}.html"]
    if not run_command(pyg_install_cmd):
        print_error_and_exit("Failed to install PyTorch Geometric dependencies.")
    print_success("PyG dependencies installed successfully!")

    # --- Step 4: Install TensorFlow ---
    print_step(4, TOTAL_STEPS, "Installing TensorFlow with CUDA support...")
    tf_install_cmd = [pip_exe_path, "install", "--no-cache-dir", "tensorflow[and-cuda]"]
    if not run_command(tf_install_cmd):
        print_error_and_exit("Failed to install TensorFlow.")
    print_success("TensorFlow installed successfully!")

    # --- Step 5: Install all remaining packages ---
    print_step(5, TOTAL_STEPS, "Installing all remaining packages...")
    remaining_install_cmd = [pip_exe_path, "install", "--no-cache-dir"] + other_pip_packages
    if not run_command(remaining_install_cmd):
        print_error_and_exit("Failed to install one or more of the remaining packages.")
    print_success("All remaining packages installed successfully!")

    print_header(f"✅✅✅ Environment '{env_name}' created successfully! ✅✅✅")
    print("To activate and use the new environment, run:\n")
    print(f"    conda activate {env_name}\n")
