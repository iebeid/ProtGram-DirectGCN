import os
import subprocess
import sys
import argparse
from pathlib import Path

# --- Configuration ---
CONFIG_DIR = Path(__file__).parent.resolve()
ENV_FILE = CONFIG_DIR / "environment.yml"

def check_conda_installed():
    """Checks if conda is installed and available in the system's PATH."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True, text=True)
        print("--- Conda is installed. ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        print("Please install Miniconda or Anaconda and try again.")
        print("Installation instructions: https://docs.conda.io/projects/miniconda/en/latest/")
        return False

def run_command(command: list):
    """Executes a command and streams its output."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        process.wait()

        if process.returncode != 0:
            print(f"\n--- Command failed with exit code {process.returncode} ---")
            sys.exit(process.returncode)

    except Exception as e:
        print(f"An unexpected error occurred while running command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or update a Conda environment from the environment.yml file."
    )
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    if not check_conda_installed():
        sys.exit(1)

    if not ENV_FILE.exists():
        print(f"--- ERROR: Environment definition file not found at '{ENV_FILE}' ---")
        sys.exit(1)

    print(f"--- Creating/Updating Conda environment '{env_name}' from '{ENV_FILE}' ---")
    print("This may take several minutes...")

    # Using 'conda env create' is idempotent and safe.
    # If the env exists, it will fail, so we can try to update. A more robust script could check first.
    # For simplicity, we just create. Users can remove old envs if needed.
    command = ["conda", "env", "create", "--name", env_name, "--file", str(ENV_FILE)]
    run_command(command)

    print("\n" + "=" * 60)
    print("--- Environment setup completed successfully! ---")
    print(f"To activate the new environment, run: conda activate {env_name}")
    print("=" * 60)