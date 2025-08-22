
"""
Project: 3D flow and volume transport around Curaçao. 

Script to calculate ramp-up time from Parcels output for multiple batches.
Ramp-up time is defined as the time when 90% of a specific batch of particles 
have left the domain (either deleted or out of bounds). This is only applied to initally active particles
(the ones that flow into the domain and are not immediately deleted).

Author: V Bertoncelj
"""
#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

def load_parcels_output(filename):
    """Load parcels output from zarr file"""
    try:
        ds = xr.open_zarr(filename)
        print(f"Successfully loaded: {filename}")
        print(f"Dataset dimensions: {ds.dims}")
        print(f"Variables: {list(ds.variables.keys())}")
        return ds
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def identify_batch(ds, batch_no=0, particles_per_batch=19293):

    # Calculate the range of trajectory IDs for this batch
    all_trajectory_ids = ds.trajectory.values
    
    # Check if trajectory IDs start from 0 or 1
    min_id = np.min(all_trajectory_ids)
    if min_id == 0:
        # 0-indexed: batch 0 is IDs 0 to particles_per_batch-1
        #            batch 1 is IDs particles_per_batch to 2*particles_per_batch-1, etc.
        start_id = batch_no * particles_per_batch
        end_id = (batch_no + 1) * particles_per_batch - 1
        batch_ids = np.arange(start_id, end_id + 1)
    else:
        # 1-indexed: batch 0 is IDs 1 to particles_per_batch
        #            batch 1 is IDs particles_per_batch+1 to 2*particles_per_batch, etc.
        start_id = batch_no * particles_per_batch + 1
        end_id = (batch_no + 1) * particles_per_batch
        batch_ids = np.arange(start_id, end_id + 1)
    
    # Filter to only include IDs that actually exist in the dataset
    batch_ids = batch_ids[np.isin(batch_ids, all_trajectory_ids)]
    
    if len(batch_ids) == 0:
        print(f"Warning: No particles found for batch {batch_no}!")
        return None, None
    
    # Get the seeding time for this batch (time at obs=0 for first particle in batch)
    first_particle_id = batch_ids[0]
    first_particle_data = ds.sel(trajectory=first_particle_id)
    batch_seeding_time = first_particle_data.time.isel(obs=0).values
    
    return batch_ids, batch_seeding_time

def calculate_particle_exit_times(ds, batch_ids):

    # Select data for all batch particles at once
    batch_data = ds.sel(trajectory=batch_ids)
    
    # Get coordinates for all particles
    lon_data = batch_data.lon.values  # Shape: (n_particles, n_obs)
    lat_data = batch_data.lat.values  # Shape: (n_particles, n_obs)
    time_data = batch_data.time.values  # Shape: (n_particles, n_obs)
    
    # Find valid observations (both lon and lat are not NaN)
    valid_mask = ~np.isnan(lon_data) & ~np.isnan(lat_data)  # Shape: (n_particles, n_obs)
    
    # Find last valid index for each particle using vectorized operations
    # This creates an array where each row has indices 0, 1, 2, ... n_obs-1
    obs_indices = np.arange(valid_mask.shape[1])[np.newaxis, :]
    
    # Set invalid observations to -1, valid ones keep their index
    masked_indices = np.where(valid_mask, obs_indices, -1)
    
    # Find the last valid index for each particle
    last_valid_indices = np.max(masked_indices, axis=1)
    
    # Create exit times dictionary
    exit_times = {}
    total_obs = time_data.shape[1]
    
    # Vectorized condition checking
    has_valid_obs = last_valid_indices >= 0
    exited_early = (last_valid_indices < total_obs - 1) & has_valid_obs
    still_in_domain = (last_valid_indices == total_obs - 1) & has_valid_obs
    no_valid_obs = ~has_valid_obs
    
    # NEW: Filter out particles that exit within 1 timestep (index 0 or 1)
    immediate_exits = (last_valid_indices <= 1) & has_valid_obs
    
    for i, particle_id in enumerate(batch_ids):
        if no_valid_obs[i] or immediate_exits[i]:
            # Skip particles that exit immediately (within 1 timestep)
            continue
        elif exited_early[i]:
            # Particle left domain after more than 1 timestep
            exit_times[particle_id] = time_data[i, last_valid_indices[i]]
        elif still_in_domain[i]:
            # Particle still in domain at end of simulation
            exit_times[particle_id] = None
    
    return exit_times

def calculate_rampup_time(exit_times, batch_seeding_time, percentile=90):
    """
    Calculate ramp-up time based on when X% of particles have left
    
    Parameters:
    exit_times: dict of particle_id: exit_time
    batch_seeding_time: time when batch was seeded (as timedelta)
    percentile: percentage of particles that need to exit (default 90%)
    
    Returns:
    rampup_time_days: ramp-up time in days
    """
    # Filter out particles that haven't left (None values)
    exited_particles = {k: v for k, v in exit_times.items() if v is not None}
    
    total_particles = len(exit_times)
    exited_count = len(exited_particles)
    
    if exited_count == 0:
        return None
    
    # Calculate time since seeding for each exit
    exit_times_since_seeding = []
    for exit_time in exited_particles.values():
        # Both times are timedelta objects, so we can subtract them directly
        time_diff = exit_time - batch_seeding_time
        # Convert to days (timedelta64 to float)
        time_diff_days = time_diff / np.timedelta64(1, 'D')
        exit_times_since_seeding.append(float(time_diff_days))
    
    # Sort exit times
    exit_times_since_seeding.sort()
    
    # Calculate percentile
    if exited_count >= total_particles * percentile / 100:
        # Enough particles have exited
        percentile_idx = int(np.ceil(total_particles * percentile / 100)) - 1
        percentile_idx = min(percentile_idx, len(exit_times_since_seeding) - 1)
        rampup_time_days = exit_times_since_seeding[percentile_idx]
    else:
        # Use the latest exit time available
        rampup_time_days = max(exit_times_since_seeding) if exit_times_since_seeding else None
    
    return rampup_time_days

def analyze_multiple_batches(ds, num_batches=48, particles_per_batch=19293):

    batch_results = {}
    
    print(f"Analyzing {num_batches} batches...")
    print("="*50)
    
    for batch_no in range(num_batches):
        print(f"Processing batch {batch_no}...")
        
        # Identify batch
        batch_ids, batch_seeding_time = identify_batch(ds, batch_no=batch_no, 
                                                      particles_per_batch=particles_per_batch)
        
        if batch_ids is None:
            print(f"  Skipping batch {batch_no} - no particles found")
            batch_results[batch_no] = None
            continue
        
        # Calculate exit times
        exit_times = calculate_particle_exit_times(ds, batch_ids)
        
        # Calculate ramp-up time
        rampup_time_days = calculate_rampup_time(exit_times, batch_seeding_time, percentile=90)
        
        batch_results[batch_no] = rampup_time_days
        
        if rampup_time_days is not None:
            print(f"  Batch {batch_no} - 90% ramp-up time: {rampup_time_days:.2f} days")
        else:
            print(f"  Batch {batch_no} - Could not calculate ramp-up time")
    
    return batch_results

def plot_batch_rampup_times(batch_results, part_config, part_month):

    # Filter out None values
    valid_results = {k: v for k, v in batch_results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot!")
        return None
    
    batch_numbers = list(valid_results.keys())
    rampup_times = list(valid_results.values())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(batch_numbers, rampup_times, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('90% Ramp-up Time (days)')
    ax.set_title(f'90% Ramp-up Time by Batch ({part_config} - {part_month})')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    mean_rampup = np.mean(rampup_times)
    std_rampup = np.std(rampup_times)
    
    ax.axhline(mean_rampup, color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {mean_rampup:.2f} ± {std_rampup:.2f} days')
    
    ax.legend()
    
    # Set x-axis to show all batch numbers
    ax.set_xlim(-1, max(batch_numbers) + 1)
    
    plt.tight_layout()

    
    return fig

def main():
    """Main function to calculate ramp-up time for multiple batches"""
    
    # Configuration - adjust these parameters
    year = 2020  # Adjust based on your run
    month = 4    # Adjust based on your run
    part_month = f'Y{year}M{str(month).zfill(2)}'
    part_config = 'INFLOW4B4M'
    
    # BATCH CONFIGURATION
    num_batches = 48*15  # Number of batches to analyze
    particles_per_batch = 19293  # Number of particles per batch
    
    # Path to your output file
    output_file = f"../parcels_run/{part_config}/{part_config}_starting_{part_month}.zarr"
    
    print("="*50)
    print("MULTI-BATCH RAMP-UP TIME ANALYSIS")
    print(f"Analyzing first {num_batches} batches")
    print("="*50)
    
    # Load data
    print("Loading parcels output...")
    ds = load_parcels_output(output_file)
    if ds is None:
        return
    
    # Analyze all batches
    batch_results = analyze_multiple_batches(ds, num_batches=num_batches, 
                                           particles_per_batch=particles_per_batch)
    
    # Plot results
    print("\nGenerating plot...")
    fig = plot_batch_rampup_times(batch_results, part_config, part_month)
    
    if fig is not None:
        # Save figure
        filename = f'figures/rampup/rampup_times_all_batches_{part_config}_{part_month}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")

if __name__ == "__main__":
    main()

# %%
