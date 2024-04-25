#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch milan.serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J hyper-trpo       # Job name
#SBATCH -o %x.o%j          # Name of stdout output file
#SBATCH -e %x.e%j          # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 03:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A IRI24006        # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=nicolas.hsu@utexas.edu
#SBATCH --cpus-per-task 2  # halving pruner requires 2 cpus

# Any other commands must follow all #SBATCH directives...
module list
pwd
date
echo $PATH

# Launch serial code...
python -m rl_zoo3.train --env multitask-atari -optimize --n-jobs 8 --env-kwargs env_names:\[\"Breakout\",\"Pong\"\] max_episode_steps:600 autoreset:True --algo trpo
