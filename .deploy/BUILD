#!/usr/bin/env bash

# Exit immediately on errors, treat unset variables as an error, and fail on error in any pipeline
set -euo pipefail



# Attempt to initialize Conda in a Bash shell

# Find the path to the conda executable
conda_path=$(which conda)

# Determine the path to conda.sh based on the conda executable location
if [[ "$conda_path" == */condabin/conda ]]; then
  conda_sh_path="${conda_path%/condabin/conda}/etc/profile.d/conda.sh"
elif [[ "$conda_path" == */bin/conda ]]; then
  conda_sh_path="${conda_path%/bin/conda}/etc/profile.d/conda.sh"
else
  echo "Error: Unable to locate the conda.sh file."
  exit 1
fi

# Source conda.sh if it exists, else export PATH
if [ -f "$conda_sh_path" ]; then
  echo "Sourcing: $conda_sh_path"
  source "$conda_sh_path"
else
  echo "Exporting PATH with Conda bin directory"
  export PATH="${conda_path%/bin/conda}/bin:$PATH"
fi



# Initialize and update git submodules
git submodule init
git submodule update

# Activate the base Conda environment
conda activate base

# Clean up any previous Conda environment and build targets
rm -rf .deploy/conda_environment/
rm -rf dacbench_adjustments/onell_algs_rs/target/

# Create and activate a new Conda environment based on .conda.yaml configuration
conda env create --prefix .deploy/conda_environment/ --file .deploy/conda.yaml --debug
conda activate .deploy/conda_environment/

# Install Python dependencies from the requirements file
pip install --requirement .deploy/pip.txt

# Uncomment the following lines to install and configure Rust (if necessary for your project)
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# export PATH="$HOME/.cargo/bin:$PATH"

# Build Rust project using maturin, within its directory (isolated in a subshell)
(
  cd dacbench_adjustments/onell_algs_rs/
  maturin build --release
)

# Find and install the Python wheel files built by maturin
find dacbench_adjustments/onell_algs_rs/target/wheels -name "*.whl" -print0 | xargs -0 pip install --force-reinstall
