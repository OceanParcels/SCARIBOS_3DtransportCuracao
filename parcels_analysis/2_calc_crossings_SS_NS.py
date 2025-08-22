'''
Project: 3D flow and volume transport around Curaçao. 

In this script we calculate the crossings of particle trajectories with defined cross-sections.
Names:
- NS: North Section (in final manuscript: North-of-Curacao NC)
- WP: West Point (another script)
- MP: Mid Point (another script)
- KC: Klein Curacao (another script)
- SS: South Section (in final manuscript: South-of-Curacao SC)

Author: V Bertoncelj
kernel: parcels_shared
'''

# ==============================================================================
# 1. Import libraries and set up parameters
# ==============================================================================
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from scipy.interpolate import griddata
from geopy.distance import geodesic
import dask.array as da
from dask.diagnostics import ProgressBar

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    
    year       = int(sys.argv[1])
    month      = int(sys.argv[2])
    part_month = f'Y{year}M{str(month).zfill(2)}'

    # Simulation parameters
    part_config = 'INFLOW4B4M'

    # Calculate seeding times based on days in month
    days = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30 if month != 2 else 28 + (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_month2 = 31 if (month + 1) % 12 in [1, 3, 5, 7, 8, 10, 12] else 30 if (month + 1) % 12 != 2 else 28 + (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_month3 = 31 if (month + 2) % 12 in [1, 3, 5, 7, 8, 10, 12] else 30 if (month + 2) % 12 != 2 else 28 + (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days += days_month2 + days_month3
    seeding_times = days

    # =============================================================================
    # 2. Open Dataset with Dask
    # =============================================================================
    filename = f'../parcels_run/{part_config}/{part_config}_starting_{part_month}.zarr'
    ds = xr.open_zarr(filename)
    # Rechunk after loading
    ds = ds.chunk({'trajectory': 10000, 'obs': 'auto'})

    # =============================================================================
    # 3. Define Cross-Sections and Seeding Regions
    # =============================================================================
    sections = {
        "NS": [(12.03, -69.83), (12.93, -68.93)],
        "SS": [(11.41, -68.80), (12.40, -67.81)],
    }

    # Number of particles per seeding region (# of particles released each seeding time)
    num_south = 900
    num_east  = 8095
    num_north = 9081
    num_west  = 1217

    # =============================================================================
    # 4. Process Data and Find Crossings
    # =============================================================================
    # Create empty datasets to store crossing information for each section
    crossing_coords = {
        'trajectory': [],
        'time': [],
        'crossing_lat': [],
        'crossing_lon': [],
        'crossing_depth': [],
        'initial_depth': [],
        'distance_from_start': [],
        'source': []
    }

    # Initialize empty xarray Datasets for each section
    crossings_SS_ds = xr.Dataset()
    crossings_NS_ds = xr.Dataset()

    # Process in batches to manage memory
    batch_size = 19293
    total_trajectories = len(ds.trajectory)
    num_batches = (total_trajectories + batch_size - 1) // batch_size
    for batch_num in range(num_batches):
        print(f"Processing batch {batch_num+1} of {num_batches}")
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_trajectories)
        
        # Process current batch
        ds_batch = ds.isel(trajectory=slice(batch_start, batch_end))
        
        # Extract particle positions and depths
        par_lon = ds_batch.lon.values
        par_lat = ds_batch.lat.values
        par_z = ds_batch.z.values
        par_traje = ds_batch.trajectory.values
        par_time = ds_batch.time.values
        
        # Initialize dictionary to store crossings for this batch
        batch_crossings = {section: [] for section in sections}
        
        # Process each section
        for section_name, ((lat1, lon1), (lat2, lon2)) in sections.items():
            print(f"Processing section: {section_name}")
            
            # Particle movement vectors (from time t to t+1)
            p1 = np.stack((par_lat[:, :-1], par_lon[:, :-1]), axis=-1)
            p2 = np.stack((par_lat[:, 1:], par_lon[:, 1:]), axis=-1)
            
            # Define the cross-section endpoints and vector
            q1 = np.array([lat1, lon1])
            q2 = np.array([lat2, lon2])
            sec_vec = q2 - q1
            
            # Compute particle displacement vector and vector from q1 to p1
            r = p2 - p1
            qp = q1 - p1
            
            # Convert 2D vectors to 3D for cross product calculation
            r_3d = np.concatenate((r, np.zeros(r.shape[:-1] + (1,))), axis=-1)
            sec_vec_3d = np.concatenate((sec_vec, np.zeros(1)))
            qp_3d = np.concatenate((qp, np.zeros(qp.shape[:-1] + (1,))), axis=-1)
            
            # Compute cross products
            r_cross_s = np.cross(r_3d, sec_vec_3d)[..., 2]
            valid = r_cross_s != 0
            
            qp_cross_r = np.cross(qp_3d, r_3d)[..., 2]
            t = np.cross(qp_3d, sec_vec_3d)[..., 2] / np.where(valid, r_cross_s, np.nan)
            u = qp_cross_r / np.where(valid, r_cross_s, np.nan)
            
            # Valid crossing if intersection lies within both segments
            valid_crossings = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)
            
            # Interpolate crossing positions and depths along particle trajectory
            crossing_points = p1 + t[:, :, np.newaxis] * r
            crossing_depths = par_z[:, :-1] + t * (par_z[:, 1:] - par_z[:, :-1])

            # Replace NaNs in crossing_depths with previous depth (par_z[:, :-1])
            nan_mask = np.isnan(crossing_depths)
            crossing_depths[nan_mask] = par_z[:, :-1][nan_mask]
            
            # Get source information for particles
            def get_source(particle_id):
                if particle_id < num_south:
                    return "South"
                elif particle_id < num_south + num_east:
                    return "East"
                elif particle_id < num_south + num_east + num_north:
                    return "North"
                else:
                    return "West"
            
            # Extract crossings for each particle
            for i, (cross, lat_arr, lon_arr, depth_arr) in enumerate(zip(
                    valid_crossings, 
                    crossing_points[..., 0], 
                    crossing_points[..., 1], 
                    crossing_depths)):
                
                particle_id = i + batch_start
                indices = np.where(cross)[0]
                
                if len(indices) > 0:
                    source = get_source(particle_id)
                    
                    for t_idx in indices:
                        lat_val = lat_arr[t_idx]
                        lon_val = lon_arr[t_idx]
                        depth_val = depth_arr[t_idx]
                        initial_depth = par_z[i, 0]
                        traje_val = par_traje[i]
                        time_val = par_time[i, t_idx]
                        
                        # Calculate distance from start
                        distance = geodesic(sections[section_name][0], (lat_val, lon_val)).meters
                        
                        # Store crossing data
                        batch_crossings[section_name].append({
                            'trajectory': traje_val,
                            'time': time_val,
                            'crossing_lat': lat_val,
                            'crossing_lon': lon_val,
                            'crossing_depth': depth_val,
                            'initial_depth': initial_depth,
                            'distance_from_start': distance,
                            'source': source
                        })
        
        # Convert batch crossings to xarray Datasets
        for section_name, crossings in batch_crossings.items():
            if crossings:  # If we have crossings for this section
                # Create a temporary dataset for this batch's crossings
                crossing_data = {key: [] for key in crossing_coords}
                
                for crossing in crossings:
                    for key in crossing_coords:
                        crossing_data[key].append(crossing[key])
                
                # Create temporary dataset
                temp_ds = xr.Dataset(
                    {
                        'crossing_lat': ('crossing', np.array(crossing_data['crossing_lat'])),
                        'crossing_lon': ('crossing', np.array(crossing_data['crossing_lon'])),
                        'crossing_depth': ('crossing', np.array(crossing_data['crossing_depth'])),
                        'initial_depth': ('crossing', np.array(crossing_data['initial_depth'])),
                        'distance_from_start': ('crossing', np.array(crossing_data['distance_from_start'])),
                        'trajectory': ('crossing', np.array(crossing_data['trajectory'], dtype=np.int32)),
                        'time': ('crossing', np.array(crossing_data['time'], dtype=np.float64)),
                        'source': ('crossing', np.array(crossing_data['source']))
                    }
                )
                
                # Append to the appropriate dataset
                if section_name == "SS":
                    crossings_SS_ds = xr.concat([crossings_SS_ds, temp_ds], dim='crossing') if 'crossing' in crossings_SS_ds.dims else temp_ds
                elif section_name == "NS":
                    crossings_NS_ds = xr.concat([crossings_NS_ds, temp_ds], dim='crossing') if 'crossing' in crossings_NS_ds.dims else temp_ds

        
        # Save resutls only for the last batch
        if batch_num == num_batches - 1:
            if 'crossing' in crossings_SS_ds.dims:
                crossings_SS_ds.to_netcdf(f'../parcels_analysis/crossings_calculations/FINAL/xrcrossings_SS_{part_month}_ALL.nc')
            if 'crossing' in crossings_NS_ds.dims:
                crossings_NS_ds.to_netcdf(f'../parcels_analysis/crossings_calculations/FINAL/xrcrossings_NS_{part_month}_ALL.nc')

        print(f"Batch {batch_num+1} processed and saved")

    print("All batches processed")
