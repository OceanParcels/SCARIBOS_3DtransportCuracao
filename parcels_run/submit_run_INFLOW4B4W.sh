#!/bin/bash
#SBATCH -J INFLOW4B4M        # name of the job
#SBATCH -p normal             # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 1                  # number of cores
#SBATCH -t 5-00:00:00         # number of hours you want to reserve the cores
#SBATCH -o logfiles/log_run_INFLOW4B4M_2024.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/log_run_INFLOW4B4M_2024.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

now="$(date)"
printf "Start date and time %s\n" "$now"

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/transport_in_3D_project/parcels_run

conda activate parcels-dev-local

s_time=$(date +%s)


combinations=("2024 1")

for combo in "${combinations[@]}"; do
    read yr mnt <<< "$combo"  # Split year and month
    python -u 2_run_INFLOW4B4M.py $yr $mnt > logfiles/run_INFLOW4B4M_starting_Y${yr}M${mnt}.log 2>&1 &
done

wait

e_time=$(date +%s)
echo "Task completed Time: $(( e_time - s_time )) seconds"