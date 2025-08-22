#!/bin/bash
#SBATCH -J rampupALL # name of the job
#SBATCH -p normal # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 1 # number of cores
#SBATCH -t 5-00:00:00 # number of hours you want to reserve the cores
#SBATCH -o logfiles/rampupALL.out # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/rampupALL.err # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)" # this makes sure that conda works in the batch environment

now="$(date)"
printf "Start date and time %s\n" "$now"

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/transport_in_3D_project/parcels_run_PUBLISH
conda activate parcels-dev-local

s_time=$(date +%s)
# Remove the & to run in foreground - this will wait for completion
python -u 4_calc_rampup_multiple_batches.py > logfiles/rampupALL.log 2>&1
e_time=$(date +%s)

elapsed=$(( e_time - s_time ))
hours=$(echo "scale=2; $elapsed / 3600" | bc)
echo "Task completed Time: $hours hours"

now="$(date)"
printf "End date and time %s\n" "$now"