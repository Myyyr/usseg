#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=logs/fine_nnf_2.out # output file name
#SBATCH --error=logs/fine_nnf_2.err  # error file name


source /opt/server-env.sh
source /home/lthemyr/usenv/bin/activate


srun python main.py