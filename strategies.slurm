#!/bin/bash

#SBATCH --account=sc122-dimitri
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=1
#SBATCH --time=05:00:00



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


conda activate ./.conda_environment
source <(python yaml2env.py "$yaml_path")
python strategies.py
