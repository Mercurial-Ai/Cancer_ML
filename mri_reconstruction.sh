#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --partition=All
#SBATCH -J rf_insert
##SBATCH --time=12:00:00
#SBATCH -e ./%J.err
#SBATCH -o ./%J.out

# execute program

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
hostname
date
readarray -d '' paths < <(find . -maxdepth 3 -mindepth 3 -type d)
for path in ${paths[@]}
do
    base=$(echo "$path" | cut -d "/" -f2)
    save="mesh.stl"
    save_path="$base$save"
    dicom2mesh -i $path -t 557 -o $save_path
done
echo "program done"
date
