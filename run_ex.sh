#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --partition=mercurial
#SBATCH -J braggNN 
#SBATCH -e ./%J.err
#SBATCH -o ./%J.out
#SBATCH --mail-user=$tristen.pool@mercurial-ai.com
#SBATCH --mail-type=ALL

# execute program
python -u main.py 
echo "program done"
date
