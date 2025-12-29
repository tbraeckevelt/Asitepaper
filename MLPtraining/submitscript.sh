#!/bin/sh

#SBATCH --account=project_465000315
#SBATCH --time=71:59:00
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH -e error.txt
#SBATCH -o output.txt
#SBATCH --export=NONE


source /pfs/lustrep4/scratch/project_465000315/svandenhaute/psiflow-1.0.0rc0/activate.sh

python -u run_passive.py
