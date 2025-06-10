import os
import sys
import platform
import argparse
import stat


def create_installer_script(env_name):
    """
    Generates a robust, self-contained shell script to create the environment.
    This script activates the environment correctly before running pip.
    """

    # --- Configuration ---
    python_version = "3.11"
    torch_version = "2.3.1"
    cuda_version_for_pytorch = "cu121"

    # --- Define Package Lists ---
    # Minimal set for conda to handle
    base_conda_packages = f"python={python_version} pip gxx_linux-64"

    # Command for PyTorch (Step 2)
    torch_install_cmd = f"pip install --no-cache-dir torch=={torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version_for_pytorch}"

    # Command for PyG dependencies (Step 3)
    pyg_install_cmd = f"pip install --no-cache-dir --find-links https://data.pyg.org/whl/torch-{torch_version}+{cuda_version_for_pytorch}.html torch-scatter torch-sparse torch-cluster torch-spline-conv"

    # Command for all other packages (Step 4)
    other_packages_to_install = ["pytorch-geometric", "tensorflow[and-cuda]", "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx",
        "gensim", "python-louvain", "transformers", "torch-geometric-signed-directed"]
    other_pip_install_cmd = f"pip install --no-cache-dir {' '.join(other_packages_to_install)}"

    # --- Generate the Shell Script Content ---
    # This uses a "here document" which is a standard and robust way to embed
    # a script-within-a-script.
    script_content = f"""
#!/bin/bash
#
# This is a self-contained installer for the '{env_name}' environment.
# It will create, activate, and install all packages in the correct order.
#

# Stop the script if any command fails
set -e

# Define a function to print nice headers
print_header() {{
    echo ""
    echo "======================================================================"
    echo "=== $1"
    echo "======================================================================"
}}

# Step 1: Create the minimal environment
print_header "STEP 1: Creating minimal Conda environment '{env_name}'..."
conda create --name {env_name} -y -c conda-forge {base_conda_packages}

# ------------------------------------------------------------------
# The rest of the script will be executed *inside* the new environment
# This is the key to making it robust.
# ------------------------------------------------------------------
print_header "STEP 2-4: Activating environment and installing all pip packages..."
conda run -n {env_name} /bin/bash <<'EOF'

# Stop this sub-script on any error
set -e

# Define a function to print sub-step headers
print_step_header() {{
    echo ""
    echo "--- $1 ---"
}}

# Step 2: Install PyTorch
print_step_header "STEP 2: Installing PyTorch and CUDA toolkit..."
{torch_install_cmd}

# Step 3: Install PyTorch Geometric dependencies
print_step_header "STEP 3: Installing PyTorch Geometric dependencies..."
{pyg_install_cmd}

# Step 4: Install TensorFlow and all remaining packages
print_step_header "STEP 4: Installing TensorFlow and remaining packages..."
{other_pip_install_cmd}

EOF
# --- End of the inner script ---

# Final success message
print_header "✅✅✅ Environment '{env_name}' created successfully! ✅✅✅"
echo "To activate your new environment, run:"
echo ""
echo "    conda activate {env_name}"
echo ""
"""

    # --- Write the script to a file ---
    script_filename = f"install_{env_name}.sh"
    with open(script_filename, "w") as f:
        f.write(script_content)

    # Make the generated script executable
    st = os.stat(script_filename)
    os.chmod(script_filename, st.st_mode | stat.S_IEXEC)

    # --- Print instructions for the user ---
    print("=" * 60)
    print("✅ Generated new, robust installer script!")
    print(f"   To create your environment, please run the following command in your terminal:")
    print(f"\n   ./{script_filename}\n")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a robust installation script for the ProtDiGCN environment.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()

    # Check if on a Unix-like system
    if platform.system() not in ["Linux", "Darwin"]:
        print("ERROR: This script generator is designed for Linux or macOS and creates a .sh file.")
        sys.exit(1)

    create_installer_script(args.env_name)
