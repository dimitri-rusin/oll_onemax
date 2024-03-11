#!/bin/bash
# Slurm job options (name, compute nodes, job time)

#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=200-659

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=sc122-dimitri
# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=standard
# We use the "standard" QoS as our runtime is less than 4 days
#SBATCH --qos=standard

module load anaconda/python3

echo "Wrote it" > /mnt/lustre/indy2lfs/work/sc122/sc122/dimitri_rusin/oll_onemax/test.slurmfileCREATEDYAEH
