#!/bin/bash

#SBATCH --job-name Chip1240220
#SBATCH --output log/output_%j.txt
#SBATCH --error log/errors_%j.txt
#SBATCH --mail-user alessandro.tonin@hsancamillo.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH -c 32
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 200G

cd $SLURM_SUBMIT_DIR

srun singularity exec ./spikeinterface.sif python3 mea_padova.py