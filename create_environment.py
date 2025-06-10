import subprocess
import sys
import os
import platform
import argparse
from tqdm import tqdm


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


def print_error_and_exit(message):
    """Prints an error message and exits the script."""
    print(f"\n*** ERROR: {message} ***", file=sys.stderr)
    print("*** ABORTING SCRIPT. Please review the error messages above. ***", file=sys.stderr)
    sys.exit(1)


def run_command(command, description=""):
    """Runs a command in the shell, showing a description and streaming its output."""
    if description:
        print(f"\n>--- Running: {description} ---")
    print(f">--- Command: {command} ---")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\n*** Command failed with exit code {process.returncode} ***", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(f"\n*** ERROR: Command not found. Is Conda/Mamba installed and in your PATH? ***", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***", file=sys.stderr)
        return False


def get_pip_executable(env_name, conda_base_path):
    """Gets the full path to the pip executable for the target environment."""
    if platform.system() == "Windows":
        pip_exe = os.path.join(conda_base_path, 'envs', env_name, 'Scripts', 'pip.exe')
    else:  # Linux or macOS
        pip_exe = os.path.join(conda_base_path, 'envs', env_name, 'bin', 'pip')

    return f'"{pip_exe}"'


def check_mamba():
    """Checks if mamba is installed and available."""
    try:
        subprocess.run("mamba --version", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_conda_base_prefix():
    """Finds the base directory of the Conda installation."""
    try:
        result = subprocess.run('conda info --base', shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        print_error_and_exit("Could not determine Conda base directory. Is Conda installed and in your PATH?")


def is_env_created(env_name, conda_base_path):
    """Checks if a conda environment already exists."""
    env_path = os.path.join(conda_base_path, 'envs', env_name)
    return os.path.exists(env_path)


# Main execution block
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Create a Conda environment with GPU support for TensorFlow and PyTorch.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    # --- Configuration ---
    python_version = "3.11"
    channels = ["conda-forge", "defaults"]

    conda_packages = [f"python={python_version}", "tqdm", "dask", "dask-expr", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx"]

    pip_packages = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric", "gensim", "python-louvain", "transformers", "torch-geometric-signed-directed"]

    print_header(f"GPU Environment Creation for '{env_name}' (Python {python_version})")

    solver = "mamba" if check_mamba() else "conda"
    print(f"--- Using {solver.capitalize()} for installation. ---")

    conda_base_path = get_conda_base_prefix()
    TOTAL_STEPS = 4

    # === [Step 1/4] Create Base Conda Environment ===
    if not is_env_created(env_name, conda_base_path):
        print_step(1, TOTAL_STEPS, f"Creating environment '{env_name}' with base packages...")
        create_command = f"{solver} create --name {env_name} -y {' '.join(['-c ' + c for c in channels])} {' '.join(conda_packages)}"
        if not run_command(create_command, "Conda Environment Creation"):
            print_error_and_exit("Failed to create the conda environment.")
        print_success("Base environment created successfully!")
    else:
        print_step(1, TOTAL_STEPS, f"Environment '{env_name}' already exists. Skipping creation.")

    pip_exe = get_pip_executable(env_name, conda_base_path)

    # === [Step 2/4] Install TensorFlow with CUDA ===
    print_step(2, TOTAL_STEPS, "Installing TensorFlow and its CUDA dependencies via pip...")
    tf_command = f'{pip_exe} install "tensorflow[and-cuda]"'
    if not run_command(tf_command, "Installing TensorFlow with CUDA support"):
        print_error_and_exit("Failed to install TensorFlow.")
    print_success("TensorFlow and CUDA libraries installed successfully!")

    # === [Step 3/4] Install PyTorch with CUDA ===
    print_step(3, TOTAL_STEPS, "Installing PyTorch with CUDA support via pip...")
    pytorch_command = f'{pip_exe} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    if not run_command(pytorch_command, "Installing PyTorch for CUDA 12.1"):
        print_error_and_exit("Failed to install PyTorch.")
    print_success("PyTorch installed successfully!")

    # === [Step 4/4] Install Remaining Pip Packages ===
    print_step(4, TOTAL_STEPS, "Installing PyTorch Geometric and other remaining packages...")
    with tqdm(total=len(pip_packages), desc="Installing Pip Packages") as pbar:
        for pkg in pip_packages:
            command = f"{pip_exe} install {pkg}"
            if not run_command(command, f"Installing {pkg}"):
                print(f"\n--- WARNING: Failed to install pip package: {pkg}. Continuing... ---", file=sys.stderr)
            pbar.update(1)
    print_success("Remaining pip packages installed successfully!")

    print_header("Environment creation complete!")
    print("To activate your new environment, run:\n")
    print(f"> conda activate {env_name}\n")
    print("You can then run the unit tests to verify the GPU setup for both frameworks.")
