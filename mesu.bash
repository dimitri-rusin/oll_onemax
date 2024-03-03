#!/usr/bin/env bash

#PBS -q beta
#PBS -l select=1:ncpus=24
#PBS -l walltime=00:10:00
#PBS -N onemaxoll
#PBS -j oe

module add cmake/3.22
module add conda3-2023.02
module add gcc/11.2
module add git/2.42.0
module add LLVM/clang-llvm-10.0

export PATH="/scratchbeta/rusind/oll/.conda_environment/bin:$PATH"

python /scratchbeta/rusind/oll/strategies.py
