#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=160:0:00
#SBATCH --mem=35GB
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=pcd
#SBATCH --account=ACCOUNTNAME
#SBATCH --mail-user=USERNAME@bristol.ac.uk
#SBATCH --mail-type=ALL

. ~/initConda.sh

conda activate py37

cd /user/work/USERNAME/PCDUNet/

python main_heathaze.py --network PCDUNet --resultDir AtmPCDUNet


