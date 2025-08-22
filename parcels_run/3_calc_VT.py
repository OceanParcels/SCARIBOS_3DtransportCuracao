'''
Project: 3D flow and volume transport around Curaçao. 

In this script we caculate the volume transport (VT) of each particle at the time of release.

Author: V Bertoncelj
Kernel: parcels-dev-local
'''

# load libraries
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    
    year       = int(sys.argv[1])
    month      = int(sys.argv[2])
    part_month = f'Y{year}M{str(month).zfill(2)}'
    part_config = 'SAMPLEVEL'
    print(f"Month: {part_month}")

    ds = xr.open_zarr(f"{part_config}/{part_config}_starting_{part_month}.zarr")

    # each .nc output has particles released in 3 months
    if month == 4 or month == 10:
        release_days = 91
    elif month == 7:
        release_days = 92
    elif month == 1:
        release_days = 90
    else:
        print('Check release days for this month')

    print(f"Number of release days: {release_days}")

    # load area
    area_all   = np.load('INPUT/particles_area_ALL.npy')
    area_all   = np.repeat(area_all, release_days)
    paricle_ID = np.arange(1, area_all.size + 1)
    speed_all  = np.full(area_all.shape, np.nan)

    # Initialize our results dataframe with all necessary columns for storage
    results = pd.DataFrame({
        'MONTH': [],
        'PARTICLE_ID': [],
        'AREA': [],
        'SPEED': [],
        'VT': []
    })

    # Get all unique trajectory IDs
    trajectories = ds.trajectory.values

    # Loop through each trajectory
    for traj_id in trajectories:

        particle_data = ds.sel(trajectory=traj_id)
        print(f"Processing trajectory ID: {traj_id}")
        
        # Convert U and V to m/s        
        lat = particle_data.lat.values[0] # needed to calculate U velocity in m/s
        U = particle_data.U.values * 1852 * 60 * np.cos(lat * np.pi / 180)
        V = particle_data.V.values * 1852 * 60
        
        # W is already in correct units
        W = particle_data.W.values

        speed = np.sqrt(U**2 + V**2 + W**2)
        idx = int(traj_id) - 1  # Adjust if your trajectory IDs have a different offset
        area = area_all[idx]
        
        # Calculate volume transport
        vt = speed * area / 10**6
        
        # Create a dataframe for this particle
        particle_df = pd.DataFrame({
            'MONTH': [part_month],
            'PARTICLE_ID': traj_id, 
            'AREA': area,
            'SPEED': speed,
            'VT': vt
            })  
        # Append to results
        results = pd.concat([results, particle_df], ignore_index=True)

    # Save the results:
    results.to_csv(f'VOLUME_TRANSPORT/{part_config}_speeds_vt_{part_month}.csv', index=False)

    print(f"Results saved to VOLUME_TRANSPORT/{part_config}_speeds_vt_{part_month}.csv")


