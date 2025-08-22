#!/bin/bash
#SBATCH -J nearshoreall          # name of the job
#SBATCH -p normal              # for jobs upto 120 hours, there is also a short partition for jobs upto 3 hours
#SBATCH -n 16                   # number of cores
#SBATCH -t 5-00:00:00          # number of hours you want to reserve the cores
#SBATCH -o logfiles/calc_nearshore_all.out     # name of the output file (=stuff you normally see on screen)
#SBATCH -e logfiles/calc_nearshore_all.err     # name of the error file (if there are errors)

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

now="$(date)"
printf "Start date and time %s\n" "$now"

# once submitted to the batch tell the compute nodes where they should be
cd /nethome/berto006/transport_in_3D_project/parcels_analysis_PUBLISH

conda activate parcels-dev-local

s_time=$(date +%s)

# combinations=("2020 4" "2020 7" "2020 10" "2021 1" "2021 4" "2021 7" "2021 10" "2022 1" "2022 4" "2022 7" "2022 10" "2023 1" "2023 4" "2023 7" "2023 10")
# combinations=("2020 7" "2020 10" "2021 4" "2021 7" "2021 10" "2022 1" "2022 4" "2022 10")
combinations=("2020 4" "2020 7" "2020 10" "2021 1" "2021 4" "2021 7" "2021 10" "2022 1" "2022 4" "2022 7" "2022 10" "2023 1" "2023 4" "2023 7" "2023 10" "2024 1")

for combo in "${combinations[@]}"; do
    read yr mnt <<< "$combo"  # Split year and month
    python -u 7_calc_nearshore_trajectories.py $yr $mnt > logfiles_calc_nearshore_all_Y${yr}M${mnt}.log 2>&1 &
done

wait

e_time=$(date +%s)
elapsed=$(( e_time - s_time ))
hours=$(echo "scale=2; $elapsed / 3600" | bc)
echo "Task completed Time: $hours hours"