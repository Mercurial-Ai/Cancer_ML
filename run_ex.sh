#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --partition=All
#SBATCH -J braggNN 
#SBATCH --time=12:00:00
#SBATCH -e ./%J.err
#SBATCH -o ./%J.out

# execute program
python -u main.py 
echo "program done"
date
