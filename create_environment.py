# ==============================================================================
# MODULE: create_environment.py
# PURPOSE: Centralized unified machine learning python environment builder for both pytorch and tensorflow.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import subprocess
import sys
import os
import platform
import argparse


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 6)
    print(f"\n{border}\n|| {title} ||\n{border}")


def print_step(message):
    """Prints a formatted step message."""
    print(f"\n>>> {message}")


def run_command(command_list, check=True):
    """
    Runs a command, streams its output in real-time, and checks for errors.
    """
    print(f"    Running Command: {' '.join(command_list)}")
    try:
        # The 'creation' part can be long, so we stream the output
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(f"      {line}", end='')
        process.stdout.close()
        return_code = process.wait()
        if check and return_code != 0:
            raise subprocess.CalledProcessError(return_code, command_list)
        return True
    except FileNotFoundError as e:
        print(f"\n*** ERROR: Command '{e.filename}' not found. Is Conda in your system's PATH? ***")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n*** ERROR: Command failed with exit code {e.returncode}. See output above. ***")
        sys.exit(1)
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***")
        sys.exit(1)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Check for the required environment file first
    if not os.path.exists("unified_environment.yml"):
        print("\n*** ERROR: `unified_environment.yml` not found in the current directory. ***")
        print("Please create it before running this script.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Create a robust, unified ML Conda environment from a YAML file.")
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment (e.g., 'ml-unified-env').")
    args = parser.parse_args()
    env_name = args.env_name

    print_header(f"Unified ML Environment Setup for '{env_name}'")

    # --- Step 1: Remove Old Environment if it Exists ---
    print_step("STEP 1: Removing old environment if it exists to ensure a clean slate...")
    # We use `run_command` but with check=False so it doesn't fail if the env doesn't exist
    run_command(["conda", "env", "remove", "--name", env_name], check=False)
    print("--- Old environment removed (if present). ---")

    # --- Step 2: Create New Environment from YAML File ---
    print_step(f"STEP 2: Creating Conda environment '{env_name}' from file...")
    run_command(["conda", "env", "create", "-f", "unified_environment.yml", "--name", env_name])
    print("--- Main environment created successfully. ---")

    # --- Step 3: Install PyG Dependencies inside the New Environment ---
    print_step("STEP 3: Installing PyTorch Geometric dependencies...")

    # This is the key: we build the full path to the new environment's executables
    # This avoids the need for shell-specific `conda activate` commands
    conda_base = subprocess.check_output(['conda', 'info', '--base'], text=True).strip()
    env_path = os.path.join(conda_base, 'envs', env_name)

    if platform.system() == "Windows":
        python_exe = os.path.join(env_path, "python.exe")
        pip_exe = os.path.join(env_path, "Scripts", "pip.exe")
    else:  # Linux or macOS
        python_exe = os.path.join(env_path, "bin", "python")
        pip_exe = os.path.join(env_path, "bin", "pip")

    # Programmatically get the exact PyTorch version from the new environment
    pytorch_version_str = subprocess.check_output([python_exe, "-c", "import torch; print(torch.__version__)"], text=True).strip()
    pytorch_version = pytorch_version_str.split('+')[0]

    # We know the CUDA version from our YAML file
    cuda_version = "cu121"
    pyg_url = f"https://data.pyg.org/whl/torch-{pytorch_version}+{cuda_version}.html"

    print(f"    Installing PyG for PyTorch {pytorch_version} and CUDA {cuda_version}")

    pyg_deps = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]

    run_command([pip_exe, "install", "--no-cache-dir", "-f", pyg_url] + pyg_deps)
    print("--- PyG dependencies installed successfully. ---")

    # --- Finalization ---
    print_header(f"✅✅✅ Unified Environment '{env_name}' created successfully! ✅✅✅")
    print("To activate and use your new environment, run:\n")
    print(f"    conda activate {env_name}\n")
