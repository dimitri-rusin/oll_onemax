#!/bin/bash

#SBATCH --account=sc122-dimitri
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=1
#SBATCH --time=7:00:00


# Exit immediately on errors, treat unset variables as an error, and fail on error in any pipeline
set -euo pipefail


module load anaconda3/2023.09
cd /work/sc122/sc122/dimitri_rusin/oll_onemax/

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



# Check for the YAML file path argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <yaml_path>"
  exit 1
fi

yaml_path="$1"


conda activate .deploy/conda_environment

# Generate a temporary file in /tmp
temp_env_file=$(mktemp /tmp/env.XXXXXX)

# Generate the environment variables and capture them in the temporary file
python .deploy/apply.py "$yaml_path" > "$temp_env_file"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
  echo "Error occurred in apply.py"
  rm -f "$temp_env_file"
  exit 1
fi

# Source the environment variables if the Python script was successful
source "$temp_env_file"

# Continue with the rest of the script
python src/generate.py

# Clean up the temporary file
rm -f "$temp_env_file"

