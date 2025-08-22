'''
Project: 3D flow and volume transport around Curaçao. 

In this script we collect timeseries of trajectories that are used then to plot the
timeseries of depths of particles arriving or leaving the enrshroe segments. Script is modified to 
be run in parallel (more months at the same time). 

Here we are only taking the maximum 60-days long tiemseries, if the particle trajectory is longer than 100 days, we only look
at the last 100 days of the trajectory (in case of 'backward' direction) or the first 100 days of the 
trajectory (in case of 'forward' direction). You can modify to extend that time window.

This version processes all combinations of nearshore segments and directions automatically:
- All segments: KC_5D1, KC_6D1, MP_5D1, MP_6D1, WP_5D1, WP_6D1
- Both directions: backward and forward

Author: V Bertoncelj
kernel: parcels-dev-local
'''

# Import libraries
import sys
import numpy as np
import xarray as xr
import json
import pickle
import os
from datetime import datetime
from tqdm import tqdm

# Define all combinations to process
ALL_SEGMENTS = ['KC_5D1', 'KC_6D1', 'MP_5D1', 'MP_6D1', 'WP_5D1', 'WP_6D1']
ALL_DIRECTIONS = ['backward', 'forward']

# Other parameters (unchanged)
UPWARD_ONLY = False
TIME_WINDOW = 100  # days
config = 'SCARIBOS_V8'
part_config = 'INFLOW4B4M'
MAX_TRAJECTORIES = None  # Process all trajectories

# Function to get reference date for a month string
def get_reference_date(month_str):
    year = int(month_str.split('Y')[1].split('M')[0])
    month = int(month_str.split('M')[1])
    return np.datetime64(f'{year:04d}-{month:02d}-01')

# Main processing function (modified to accept TARGET_SECTIONS and DIRECTION as parameters)
def process_single_month_and_config(year, month, target_sections, direction):
    part_month = f'Y{year}M{str(month).zfill(2)}'
    print(f"Processing {part_month} for {target_sections} in {direction} direction...")
    
    # Extract section type and create file suffix
    SECTION_TYPE = target_sections[0].split('_')[0]
    file_suffix = target_sections[0] if len(target_sections) == 1 else f"multiple_{len(target_sections)}_sections"
    if UPWARD_ONLY:
        file_suffix += "_upward"
    file_suffix += f"_{direction}_{part_month}"
    
    # Create output directory
    OUTPUT_DIR = f'nearshore/{target_sections[0][:2]}/{direction}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # File to save trajectory data
    trajectory_cache_file = f"{OUTPUT_DIR}/trajectory_data_{file_suffix}.pkl"
    
    # Dictionary to store all trajectory data
    all_trajectory_data = {}
    all_times, all_depths, all_rel_times, all_abs_times = [], [], [], []
    all_lats, all_lons, all_dz_values, all_particle_ids = [], [], {}, []
    
    # Load section locations
    all_locations = {}
    for section_type in ['KC', 'WP', 'MP', 'SS', 'NS']:
        try:
            section_locs = np.load(f'../parcels_analysis/segmentation/final/locations_{section_type}_{part_month}.npy', allow_pickle=True).item()
            all_locations.update(section_locs)
        except FileNotFoundError:
            continue
    
    # Validate target sections
    for section in target_sections:
        if section not in all_locations:
            print(f"Warning: Target section '{section}' not found in location data for {part_month}.")
    
    all_trajectory_data[part_month] = {
        'target_particles': [],
        'upward_particles': [],
        'particles_times': {},
        'ordered_segments': {},
        'segment_names': {}
    }
    
    # Load segment data
    ordered_segments_file = f'../parcels_analysis/segmentation/final/ordered_segments_{part_month}.json'
    if not os.path.exists(ordered_segments_file):
        print(f"Warning: {ordered_segments_file} not found, skipping")
        return
        
    with open(ordered_segments_file) as f:
        ordered_segments = json.load(f)
    all_trajectory_data[part_month]['ordered_segments'] = ordered_segments
    
    # Load segment names with timing information
    segment_names_file = f'../parcels_analysis/segmentation/final/segment_names_{SECTION_TYPE}_{part_month}.json'
    if os.path.exists(segment_names_file):
        with open(segment_names_file) as f:
            segment_names = json.load(f)
        all_trajectory_data[part_month]['segment_names'] = segment_names
    
    # Find particles that reach target sections
    for particle_id, path in ordered_segments.items():
        segments = path.split(', ')
        
        target_found = False
        for target_section in target_sections:
            if target_section in segments:
                target_found = True
                target_index = segments.index(target_section)
                break
        
        if target_found:
            all_trajectory_data[part_month]['target_particles'].append(particle_id)
            
            if UPWARD_ONLY and any('D2' in seg or 'D3' in seg for seg in segments[:target_index]):
                all_trajectory_data[part_month]['upward_particles'].append(particle_id)
    
    # Get timing information
    for particle_id in all_trajectory_data[part_month]['target_particles']:
        if particle_id in all_trajectory_data[part_month]['segment_names']:
            segments = all_trajectory_data[part_month]['segment_names'][particle_id]['segments']
            times = all_trajectory_data[part_month]['segment_names'][particle_id]['times']
            
            earliest_time = None
            for target_section in target_sections:
                if target_section in segments:
                    idx = segments.index(target_section)
                    section_time = times[idx][0]
                    
                    if earliest_time is None or section_time < earliest_time:
                        earliest_time = section_time
            
            if earliest_time is not None:
                all_trajectory_data[part_month]['particles_times'][particle_id] = earliest_time
    
    print(f"Found {len(all_trajectory_data[part_month]['target_particles'])} particles reaching target sections")
    if UPWARD_ONLY:
        print(f"Of these, {len(all_trajectory_data[part_month]['upward_particles'])} are moving upward")
    print(f"Found timing information for {len(all_trajectory_data[part_month]['particles_times'])} particles")
    
    # Extract trajectory data
    reference_date = get_reference_date(part_month)
    filename = f'../parcels_run/{part_config}/{part_config}_starting_{part_month}.zarr'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping")
        return

    # Determine particles of interest
    if UPWARD_ONLY:
        particles_of_interest = all_trajectory_data[part_month]['upward_particles']
    else:
        particles_of_interest = all_trajectory_data[part_month]['target_particles']
    
    particles_with_timing = [p for p in particles_of_interest if p in all_trajectory_data[part_month]['particles_times']]
    particles_int = [int(p) for p in particles_with_timing]
    
    # Sample if needed
    if MAX_TRAJECTORIES and len(particles_int) > 0:
        max_per_month = max(1, min(len(particles_int), MAX_TRAJECTORIES))
        if len(particles_int) > max_per_month:
            np.random.seed(42)
            particles_int = np.random.choice(particles_int, size=max_per_month, replace=False)
    
    if not particles_int:
        print(f"No valid particles found for {part_month}, skipping")
        return
    
    print(f"Loading trajectory data from {filename}...")
    ds = xr.open_zarr(filename)
    
    print(f"Extracting trajectory data for {len(particles_int)} particles...")
    for particle_id in tqdm(particles_int):
        str_particle_id = str(particle_id)
        if str_particle_id not in all_trajectory_data[part_month]['particles_times']:
            continue
            
        try:
            particle_data = ds.sel(trajectory=particle_id)
            
            times = particle_data.time.values
            depths = particle_data.z.values
            lats = particle_data.lat.values
            lons = particle_data.lon.values
            
            passing_time = all_trajectory_data[part_month]['particles_times'][str_particle_id]
            passing_time_ns = np.timedelta64(int(passing_time), 'ns')
            
            valid_mask = ~np.isnat(times)
            if not np.any(valid_mask):
                continue
                
            times = times[valid_mask]
            depths = depths[valid_mask]
            lats = lats[valid_mask]
            lons = lons[valid_mask]
            
            rel_times = []
            abs_times = []
            dt_times = []
            for t in times:
                if isinstance(t, np.timedelta64):
                    dt_times.append(t)
                    abs_times.append(reference_date + t)
                    time_diff = (t - passing_time_ns)
                    rel_times.append(time_diff.astype('timedelta64[s]').astype(float) / (24*3600))
            
            # Filter based on direction and time window
            if direction == 'backward':
                valid_indices = [i for i, rt in enumerate(rel_times) if -TIME_WINDOW <= rt <= 0]
            else:
                valid_indices = [i for i, rt in enumerate(rel_times) if 0 <= rt <= TIME_WINDOW]
            
            if valid_indices:
                valid_times = [dt_times[i] for i in valid_indices]
                valid_depths = [depths[i] for i in valid_indices]
                valid_rel_times = [rel_times[i] for i in valid_indices]
                valid_abs_times = [abs_times[i] for i in valid_indices]
                valid_lats = [lats[i] for i in valid_indices]
                valid_lons = [lons[i] for i in valid_indices]
                
                unique_particle_id = f"{part_month}_{particle_id}"
                
                # Calculate depth change
                depth_change = max(valid_depths) - min(valid_depths)
                all_dz_values[unique_particle_id] = {
                    'dz': depth_change,
                    'times': valid_times,
                    'depths': valid_depths,
                    'rel_times': valid_rel_times,
                    'abs_times': valid_abs_times,
                    'lats': valid_lats,
                    'lons': valid_lons,
                    'month': part_month
                }
                
                all_times.append(valid_times)
                all_depths.append(valid_depths)
                all_rel_times.append(valid_rel_times)
                all_abs_times.append(valid_abs_times)
                all_lats.append(valid_lats)
                all_lons.append(valid_lons)
                all_particle_ids.append(unique_particle_id)
                
        except Exception as e:
            print(f"Error processing particle {particle_id}: {e}")
    
    print(f"Successfully extracted data for {len(all_times)} trajectories")
    
    # Save trajectory data to file for future reuse
    cache_data = {
        'all_times': all_times,
        'all_depths': all_depths,
        'all_rel_times': all_rel_times,
        'all_abs_times': all_abs_times,
        'all_lats': all_lats,
        'all_lons': all_lons,
        'all_dz_values': all_dz_values,
        'all_particle_ids': all_particle_ids,
        'all_trajectory_data': all_trajectory_data
    }
    
    with open(trajectory_cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Saved trajectory data to {trajectory_cache_file} for future use")
    
    # Close dataset
    ds.close()
    print(f"Processing complete for {target_sections} in {direction} direction.")

def process_all_combinations(year, month):
    """Process all combinations of segments and directions"""
    total_combinations = len(ALL_SEGMENTS) * len(ALL_DIRECTIONS)
    current_combination = 0
    
    print(f"Starting processing of all {total_combinations} combinations for {year}-{month:02d}")
    print("=" * 80)
    
    for segment in ALL_SEGMENTS:
        for direction in ALL_DIRECTIONS:
            current_combination += 1
            print(f"\nCombination {current_combination}/{total_combinations}")
            print(f"Segment: {segment}, Direction: {direction}")
            print("-" * 40)
            
            try:
                process_single_month_and_config(year, month, [segment], direction)
            except Exception as e:
                print(f"Error processing {segment} in {direction} direction: {e}")
                continue
    
    print("\n" + "=" * 80)
    print(f"Completed processing all {total_combinations} combinations for {year}-{month:02d}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>")
        print("Example: python script.py 2021 04")
        print("This will process all nearshore segments (KC_5D1, KC_6D1, MP_5D1, MP_6D1, WP_5D1, WP_6D1)")
        print("in both backward and forward directions automatically.")
        sys.exit(1)
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    # Validate input
    if not (2000 <= year <= 2100):
        print("Error: Year should be between 2000 and 2100")
        sys.exit(1)
    
    if not (1 <= month <= 12):
        print("Error: Month should be between 1 and 12")
        sys.exit(1)
    
    process_all_combinations(year, month)

