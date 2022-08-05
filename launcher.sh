#!/bin/bash
#SBATCH -p public
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --time=100:00:00 
#SBATCH --output=logs/multi_nnunet_128on128.out # output file name
#SBATCH --error=logs/multi_nnunet_128on128.err  # error file name


source /opt/server-env.sh
conda activate usenv

# srun python main.py training.dbg=False
# srun python main.py -m model=nnunet # training.dbg=False

# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128
# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128 training.name=CROP_SMALL_V4 training.epochs=1400 training.batch_size=4
# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128 training.name=CROP_SMALL_V5 training.epochs=2000 training.batch_size=4
# srun python main.py -m model=vnet dataset=us128 training=crop128_128_128 training.name=CROP_SMALL_VNET_V3 training.epochs=2000 training.batch_size=4 training.gpu=0 training.do_clip=False training.do_schedul=False training.checkpoint.load=True




# srun convert_dataset.py /scratch/lthemyr/20220318_US_DATA/USmask_cropped /scratch/lthemyr/20220318_US_DATA/USmask_cropped256_npz 256



## Runs 128x128x128 on 128x128x128 images

# srun python main.py -m model=glam dataset=us128 training=crop128_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA

### NNUNET
# srun python main.py -m model=nnunet dataset=us128 training=crop128_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
#### multi split
srun python multi_main.py -m model=nnunet dataset=us128 training=crop128_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA


## Runs 128x128x128 on 256x256x256 images
# srun python main.py -m model=glam dataset=us256 training=crop128_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
# srun python main.py -m model=nnunet dataset=us256 training=crop128_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA

## Runs 64x128x128 on 256x256x256 images

### COTR
# srun python main.py -m model=cotr_64 dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
# srun python main.py -m model=cotr_64 dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA training.checkpoint.load=True
# srun python main.py -m model=cotr_64 dataset=us256 training=crop64_128_128_nnu training.name=CROP_SMALL_64_nnu_augdbg dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
#### multi split
# srun python multi_main.py -m model=cotr_64 dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA

### NNUNET
# srun python main.py -m model=nnunet dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
# srun python main.py -m model=nnunet dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA training.checkpoint.load=True
#### multi split
# srun python multi_main.py -m model=nnunet dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA

### GLAM
# srun python main.py -m model=glam_64 dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA
#### multi split
# srun python multi_main.py -m model=glam_64 dataset=us256 training=crop64_128_128_nnu dataset.path.pth=/scratch/lthemyr/20220318_US_DATA training.pth=/scratch/lthemyr/20220318_US_DATA model.pth=/scratch/lthemyr/20220318_US_DATA