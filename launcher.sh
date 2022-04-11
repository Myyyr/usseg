#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=logs/nnunet.out # output file name
#SBATCH --error=logs/nnunet.err  # error file name


source /opt/server-env.sh
# source /home/lthemyr/usenv/bin/activate
conda activate usenv

# srun python main.py #training.dbg=False
srun python main.py -m model=nnunet #training.dbg=False