#!/bin/bash

#SBATCH --job-name Chip1240220
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user alessandro.tonin@hsancamillo.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --ntasks 2
#SBATCH --partition allgroups
#SBATCH --mem 250G

cd $SLURM_SUBMIT_DIR

srun singularity exec ./spikeinterface.sif python3 mea_padova.py