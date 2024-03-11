#!/bin/bash

#SBATCH --account=sc122-dimitri
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --tasks-per-node=1
#SBATCH --time=00:02:00

module load anaconda/python3



# Dynamically find the Conda executable path
conda_path=$(which conda)

# Determine the correct path for conda.sh based on the location of the conda executable
if [[ "$conda_path" == */condabin/conda ]]; then
  conda_sh_path="${conda_path%/condabin/conda}/etc/profile.d/conda.sh"
elif [[ "$conda_path" == */bin/conda ]]; then
  conda_sh_path="${conda_path%/bin/conda}/etc/profile.d/conda.sh"
else
  echo "Error: Unable to locate the conda.sh file."
  exit 1
fi

echo "Sourcing: $conda_sh_path"
# Source the conda.sh script for proper initialization
source "$conda_sh_path"



cd /work/sc122/sc122/dimitri_rusin/oll_onemax/
conda activate ./.conda_environment
python strategies.py
