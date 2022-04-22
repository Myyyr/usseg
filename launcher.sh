#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=logs/nnunet_small_v5.out # output file name
#SBATCH --error=logs/nnunet_small_v5.err  # error file name


source /opt/server-env.sh
# source /home/lthemyr/usenv/bin/activate
conda activate usenv

# srun python main.py training.dbg=False
# srun python main.py -m model=nnunet # training.dbg=False

# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128
# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128 training.name=CROP_SMALL_V4 training.epochs=1400 training.batch_size=4
srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128 training.name=CROP_SMALL_V5 training.epochs=2000 training.batch_size=4