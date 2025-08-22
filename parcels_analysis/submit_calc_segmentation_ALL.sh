#!/bin/bash
#SBATCH -J xrsegmentation      # name of the job
#SBATCH -p normal              # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 1                   # number of cores
#SBATCH -t 5-00:00:00          # number of hours you want to reserve the cores
#SBATCH -o logfiles/calc_segmentation_ALL_2024.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/calc_segmentation_ALL_2024.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

now="$(date)"
printf "Start date and time %s\n" "$now"

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/transport_in_3D_project/parcels_analysis_PUBLISH

conda activate parcels-shared

s_time=$(date +%s)

combinations=("2024 1")

for combo in "${combinations[@]}"; do
    read yr mnt <<< "$combo"  # Split year and month
    python -u 3_calc_segmentation_ALL.py $yr $mnt > logfiles/calc_segmentation_ALL_Y${yr}M${mnt}.log 2>&1 &
done

wait

e_time=$(date +%s)
elapsed=$(( e_time - s_time ))
hours=$(echo "scale=2; $elapsed / 3600" | bc)
echo "Task completed Time: $hours hours"