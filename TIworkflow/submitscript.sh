#!/bin/sh
​
#PBS -o output.txt
#PBS -e error.txt
#PBS -l nodes=1:ppn=4
#PBS -A 2023_029
#PBS -l mem=20G
#PBS -l walltime=71:59:00
​
cd ${PBS_O_WORKDIR}
​
source /dodrio/scratch/projects/starting_2022_006/Forinstall/psiflow_env/activate.sh

mprof run python main.py
