'''
Project: 3D flow and volume transport around Curaçao. 

In this script we create a timeline of crossings, for each simulated month period and each particle.
We need this in order to create the correct transition matrix.
Note: we are not listing the repeated crossings if there are no crossings of different segments in between them 
(e.g. if a particle crosses the same segment multiple times in a row, we only keep the first crossing).

Author: V Bertoncelj
'''

# %%
# Imports
import json
import os
import numpy as np
import xarray as xr

# list all months or the months that the calculations are needed for:
part_months = ['Y2020M04', 'Y2020M07', 'Y2020M10', 
               'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10', 
               'Y2022M01', 'Y2022M04', 'Y2022M07', 'Y2022M10', 
               'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10']

part_months = ['Y2024M01']

# loop over the months and load the segment names:
for i in range(len(part_months)):
    part_month = part_months[i]
    print(f"Processing {part_month}...")

    sections = ['KC', 'WP', 'MP', 'SS', 'NS']
    segment_names_KC = {}
    segment_names_WP = {}
    segment_names_MP = {}
    segment_names_SS = {}
    segment_names_NS = {}

    with open(f'../parcels_analysis/segmentation/final/segment_names_KC_{part_month}.json') as f:
        segment_names_KC.update(json.load(f))
    with open(f'../parcels_analysis/segmentation/final/segment_names_WP_{part_month}.json') as f:
        segment_names_WP.update(json.load(f))
    with open(f'../parcels_analysis/segmentation/final/segment_names_MP_{part_month}.json') as f:
        segment_names_MP.update(json.load(f))
    with open(f'../parcels_analysis/segmentation/final/segment_names_SS_{part_month}.json') as f:
        segment_names_SS = json.load(f)
    with open(f'../parcels_analysis/segmentation/final/segment_names_NS_{part_month}.json') as f:
        segment_names_NS = json.load(f)

    segment_names = {'KC': segment_names_KC,  'MP': segment_names_MP, 'WP': segment_names_WP, 'SS': segment_names_SS, 'NS': segment_names_NS}

    # Create a dictionary to store the combined segments
    combined_segments = {}
    for section, segments in segment_names.items():
        for traj_id, segment_name in segments.items():  # Use traj_id as key
            if traj_id not in combined_segments:
                combined_segments[traj_id] = []
            combined_segments[traj_id].append(segment_name)  # Append full segment name

    ordered_segments = {}
    for traj_id, segment_list in combined_segments.items():
        time_segment_pairs = []

        # Collect all time-segment pairs
        for segment_entry in segment_list:
            segments = segment_entry['segments']
            times = segment_entry['times']
            
            for seg, time_list in zip(segments, times):
                for t in time_list:
                    time_segment_pairs.append((t, seg))
        
        # Sort by time
        time_segment_pairs.sort()

        # Remove consecutive duplicates while keeping order
        sorted_segments = []
        for i, (_, seg) in enumerate(time_segment_pairs):
            if i == 0 or seg != sorted_segments[-1]:  # Avoid consecutive duplicates
                sorted_segments.append(seg)
        
        # Store ordered segments
        ordered_segments[traj_id] = ', '.join(sorted_segments)

    print(ordered_segments)

    # save
    with open(f'../parcels_analysis/segmentation/final/ordered_segments_{part_month}.json', 'w') as f:
        json.dump(ordered_segments, f)


# %%
