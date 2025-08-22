'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot timeseries of depths of particles arriving or leaving the nearshore segments. 

Here we are only taking the maximum 60-days long tiemseries, if the particle trajectory is longer than 60 days, we only look
at the last 60 days of the trajectory (in case of 'backward' direction) or the first 60 days of the 
trajectory (in case of 'forward' direction). You can modify to extend that time window.

This version plots all combinations of nearshore segments:
- All segments: KC_5D1, KC_6D1, MP_5D1, MP_6D1, WP_5D1, WP_6D1

You need to run it separately for each direction: backward and forward by changing the parameter DIRECTION

Author: V Bertoncelj
kernel: parcels-dev-local
'''

#%%
# load libraries
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import cmocean
import os
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.dates import date2num, num2date
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize
import geopandas as gpd
import matplotlib.colors as mcolors
import time
import matplotlib.patches as mpatches
from datetime import timedelta
import matplotlib.dates as mdates
import pandas as pd

# CONFIGURATION
all_segments = ['WP_5D1', 'MP_5D1', 'KC_5D1', 'WP_6D1', 'MP_6D1', 'KC_6D1']  # Fixed order
DIRECTION = 'forward' #'backward'
part_months = ['Y2020M04', 'Y2020M07', 'Y2020M10', 'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10', 'Y2022M01', 'Y2022M04', 'Y2022M07', 'Y2022M10', 'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10', 'Y2024M01']
depth_range = [-2000, 0]
y_max_vt = 32

# Directories
FIGURES_DIR = 'figures/results'
CACHE_DIR = f'../Parcels_analysis/nearshore/cache_final'
VT_BASE_DIR = '../parcels_run/VOLUME_TRANSPORT'

# Thresholds
SURFACE_DEPTH_THRESHOLD = -162      
SUBSURFACE_DEPTH_THRESHOLD = -458.5  

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Processing all segments with continuous trajectories:")
print(f"  Segments: {all_segments}")
print(f"  Direction: {DIRECTION}")

#%%
# CONTINUOUS DATA LOADING FUNCTIONS (same as test, but for all segments)

def load_volume_transport_data_all():
    """Load VT data for all months (shared across segments)"""
    vt_cache_file = f"{CACHE_DIR}/vt_data_all_segments_{DIRECTION}.pkl"
    
    if os.path.exists(vt_cache_file):
        print(f"Loading cached VT data for all segments...")
        with open(vt_cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading VT data from CSV files...")
    vt_data = {}
    for month in part_months:
        vt_file_path = f'{VT_BASE_DIR}/SAMPLEVEL_speeds_vt_{month}.csv'
        
        if os.path.exists(vt_file_path):
            try:
                df = pd.read_csv(vt_file_path)
                month_vt_data = dict(zip(df['PARTICLE_ID'], df['VT']))
                vt_data[month] = month_vt_data
                print(f"  Loaded VT data for {month}: {len(month_vt_data)} particles")
            except Exception as e:
                print(f"  Error loading VT data for {month}: {e}")
    
    # Cache the VT data
    with open(vt_cache_file, 'wb') as f:
        pickle.dump(vt_data, f)
    print(f"Cached VT data for all segments")
    
    return vt_data

def load_continuous_trajectory_data_single_segment(segment, vt_data):
    """Load continuous trajectory data for ONE segment"""
    
    # Check for cached continuous data
    continuous_cache_file = f"{CACHE_DIR}/continuous_all_data_{segment}_{DIRECTION}.pkl"
    
    if os.path.exists(continuous_cache_file):
        print(f"  Loading cached continuous data for {segment}...")
        with open(continuous_cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  Processing continuous trajectory data for {segment}...")
    
    # Initialize storage for ALL trajectories (no year division)
    all_continuous_trajectories = []
    
    # Stats tracking
    total_particles = 0
    matched_vt_particles = 0
    
    # Load trajectory data from all months
    section_output_dir = f'../parcels_analysis_PUBLISH/nearshore/{segment[:2]}/{DIRECTION}'
    
    for month in part_months:
        file_suffix = f"{segment}_{DIRECTION}_{month}"
        trajectory_cache_file = f"{section_output_dir}/trajectory_data_{file_suffix}.pkl"
        
        if not os.path.exists(trajectory_cache_file):
            continue
            
        # Load this month's trajectory data
        with open(trajectory_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Extract trajectory data (these are the raw trajectories)
        month_abs_times = cache_data['all_abs_times']
        month_depths = cache_data['all_depths'] 
        month_particle_ids = cache_data['all_particle_ids']
        
        # Get VT data for this month
        month_vt_dict = vt_data.get(month, {})
        
        # Process each trajectory from this month
        for traj_times, traj_depths, particle_id in zip(month_abs_times, month_depths, month_particle_ids):
            
            if not traj_times or not traj_depths:
                continue
                
            total_particles += 1
            
            # Get VT value for this particle
            try:
                if '_' in str(particle_id):
                    numeric_pid = int(str(particle_id).split('_')[-1])
                else:
                    numeric_pid = int(particle_id)
                
                if numeric_pid in month_vt_dict:
                    vt_value = month_vt_dict[numeric_pid]
                    matched_vt_particles += 1
                else:
                    vt_value = 0.0
                    
            except (ValueError, TypeError):
                vt_value = 0.0
            
            # Convert to numpy arrays
            traj_times_arr = np.array(traj_times).astype('datetime64[ns]')
            traj_depths_arr = np.array(traj_depths)
            
            # Categorize by max depth
            max_depth = min(traj_depths) if traj_depths else 0
            if max_depth >= SURFACE_DEPTH_THRESHOLD:
                category = 'Surface'
                color = 'cornflowerblue'
            elif SUBSURFACE_DEPTH_THRESHOLD <= max_depth < SURFACE_DEPTH_THRESHOLD:
                category = 'Mid'
                color = 'tomato'
            else:
                category = 'Deep'
                color = 'lightseagreen'
            
            # Store as one continuous trajectory (NO YEAR SPLITTING!)
            trajectory = {
                'times': traj_times_arr,
                'depths': traj_depths_arr,
                'color': color,
                'category': category,
                'vt_value': vt_value,
                'particle_id': particle_id,
                'month': month
            }
            
            all_continuous_trajectories.append(trajectory)
    
    print(f"    Loaded {len(all_continuous_trajectories)} continuous trajectories")
    print(f"    VT matching: {matched_vt_particles}/{total_particles} ({matched_vt_particles/total_particles*100:.1f}%)")
    
    # Create monthly histogram data for VT bars
    monthly_histograms = create_monthly_histograms(all_continuous_trajectories)
    
    # Package the data
    continuous_data = {
        'trajectories': all_continuous_trajectories,
        'monthly_histograms': monthly_histograms,
        'segment': segment,
        'direction': DIRECTION,
        'total_particles': total_particles,
        'matched_vt_particles': matched_vt_particles
    }
    
    # Cache the continuous data
    with open(continuous_cache_file, 'wb') as f:
        pickle.dump(continuous_data, f)
    print(f"    Cached continuous data for {segment}")
    
    return continuous_data

def create_monthly_histograms(trajectories):
    """Create monthly VT histograms from continuous trajectories"""
    
    # Initialize monthly bins from April 2020 to December 2023
    monthly_data = {}
    
    # Create bins for each month
    for year in range(2020, 2025):  # 2020-2024
        if year == 2020:
            start_month = 4  # April for 2020
            end_month = 12
        elif year == 2024:
            start_month = 1
            end_month = 3  # Only up to March 2024
        else:
            start_month = 1
            end_month = 12

        for month in range(start_month, end_month + 1):
            month_key = f"{year}-{month:02d}"
            if month < 12:
                month_end = datetime(year, month + 1, 1) - timedelta(days=1)
            else:
                if year == 2024:
                    month_end = datetime(2024, 3, 31)
                else:
                    month_end = datetime(year, 12, 31)
            monthly_data[month_key] = {
                'vt_sum_surface': 0.0,
                'vt_sum_mid': 0.0,
                'vt_sum_deep': 0.0,
                'count_surface': 0,
                'count_mid': 0,
                'count_deep': 0,
                'month_start': datetime(year, month, 1),
                'month_end': month_end
            }
    # Process each trajectory to assign to monthly bins
    for traj in trajectories:
        if len(traj['times']) == 0:
            continue
            
        # Get arrival time (last point for backward, first for forward)
        arrival_time = traj['times'][-1] if DIRECTION == 'backward' else traj['times'][0]
        arrival_dt = pd.to_datetime(arrival_time)
        
        # Skip Dec 31st arrivals (year-end artifacts)
        if arrival_dt.month == 12 and arrival_dt.day == 31:
            continue
            
        # Find the correct monthly bin
        month_key = f"{arrival_dt.year}-{arrival_dt.month:02d}"
        
        if month_key in monthly_data:
            category = traj['category']
            vt_value = traj['vt_value']
            
            if category == 'Surface':
                monthly_data[month_key]['vt_sum_surface'] += vt_value
                monthly_data[month_key]['count_surface'] += 1
            elif category == 'Mid':
                monthly_data[month_key]['vt_sum_mid'] += vt_value
                monthly_data[month_key]['count_mid'] += 1
            elif category == 'Deep':
                monthly_data[month_key]['vt_sum_deep'] += vt_value
                monthly_data[month_key]['count_deep'] += 1
    
    return monthly_data

def get_segment_locations(segment):
    """Get the lat/lon bounds for a given segment"""
    try:
        first_month = 'Y2020M04'
        if segment.startswith('KC'):
            section_locations = np.load(f'../parcels_analysis/segmentation/final/locations_KC_{first_month}.npy', allow_pickle=True).item()
        elif segment.startswith('WP'):
            section_locations = np.load(f'../parcels_analysis/segmentation/final/locations_WP_{first_month}.npy', allow_pickle=True).item()
        elif segment.startswith('MP'):
            section_locations = np.load(f'../parcels_analysis/segmentation/final/locations_MP_{first_month}.npy', allow_pickle=True).item()
        else:
            return None
        
        return section_locations.get(segment, None)
    except:
        return None

#%%
# STEP 1: Load VT data once (shared across all segments)
print("\n" + "="*60)
print("STEP 1: Loading Volume Transport data...")
print("="*60)

vt_data = load_volume_transport_data_all()

#%%
# STEP 2: Load continuous data for all segments
print("\n" + "="*60)
print("STEP 2: Loading continuous trajectory data for all segments...")
print("="*60)

all_segments_data = {}

for segment in all_segments:
    print(f"\nProcessing {segment}...")
    segment_data = load_continuous_trajectory_data_single_segment(segment, vt_data)
    all_segments_data[segment] = segment_data

print(f"\n✓ All segment data loaded successfully!")

# Print summary
for segment in all_segments:
    data = all_segments_data[segment]
    print(f"  {segment}: {len(data['trajectories'])} trajectories")

#%%
#%%
# STEP 3: Create the full comparison plot
print("\n" + "="*60)
print("STEP 3: Creating full comparison plot...")
print("="*60)

# Load Curacao shapefile
try:
    curacao = gpd.read_file("data/CUW_adm0.shp")
    print("✓ Loaded Curacao shapefile")
except:
    print("✗ Could not load Curacao shapefile")
    curacao = None

# Create figure with 6 rows (maps + timeseries)
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(6, 2, width_ratios=[1, 3], hspace=0.3, wspace=0.01)  # Increased hspace from 0.1 to 0.25

fig.suptitle(f"Timeseries of depth trajectories of volume transport-weighted particles:\nnearshore segments ({'backward' if DIRECTION == 'backward' else 'forward'} trajectories)", 
             fontsize=16, x=0.6, y=0.925)  # Added y=0.95 to move title closer to first row

# Define ramp-up period (first 43 days from April 1, 2020)
ramp_up_start = np.datetime64('2020-04-01')
ramp_up_end = ramp_up_start + np.timedelta64(43, 'D')  # 43 days later

# Plot each segment
for row_idx, segment in enumerate(all_segments):
    print(f"Plotting segment {segment} (row {row_idx + 1}/6)...")
    
    # Create map subplot (left column)
    map_ax = fig.add_subplot(gs[row_idx, 0])
    
    # Create timeseries subplot (right column)  
    ax = fig.add_subplot(gs[row_idx, 1])
    
    # === MAP PLOTTING ===
    if curacao is not None:
        curacao.plot(ax=map_ax, color='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
    map_ax.text(0.5, 0.92, f"{segment}", transform=map_ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.2))
    
    # Get the panel letter position in data coordinates
    panel_x_start = ax.transAxes.inverted().transform(ax.transData.transform([ramp_up_start, depth_range[1]]))[0]
    panel_x_end = min(0.15, (ramp_up_end - ramp_up_start) / (plot_end - plot_start))  # 15% of plot width or hatch width

    # Create hatching that avoids the panel area
    if panel_x_end < 0.15:  # If hatch is shorter than panel area
        # Just add normal hatching, panel will be on top
        ax.axvspan(ramp_up_start, ramp_up_end, facecolor='lightgrey', alpha=0.3, 
                hatch='///', edgecolor='grey', linewidth=0.5, zorder=5)

    # Add panel lettering (A, B, C, D, E, F)
    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    ax.text(0.01, 0.92, panel_letters[row_idx], transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=1), zorder=30)

    # Get segment location and plot line
    segment_loc = get_segment_locations(segment)
    if segment_loc:
        line_coords = [
            (segment_loc['min_lat'], segment_loc['min_lon']), 
            (segment_loc['max_lat'], segment_loc['max_lon'])
        ]
        map_ax.plot([line_coords[0][1], line_coords[1][1]], 
                   [line_coords[0][0], line_coords[1][0]], 
                   color='blue', linewidth=4)
        
        buffer = 0.15
        buffer_x = 0.18
        map_ax.set_xlim(segment_loc['min_lon'] - buffer_x, segment_loc['max_lon'] + buffer_x)
        map_ax.set_ylim(segment_loc['min_lat'] - buffer, segment_loc['max_lat'] + buffer)
        
        # Custom tick formatting
        def format_lat_ticks(lat_values):
            return [f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}" for lat in lat_values]
        
        def format_lon_ticks(lon_values):
            return [f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}" for lon in lon_values]

        lat_ticks = np.linspace(segment_loc['min_lat'] - buffer/2, segment_loc['max_lat'] + buffer/2, 2)
        lon_ticks = np.linspace(segment_loc['min_lon'] - buffer/2, segment_loc['max_lon'] + buffer/2, 2)
        map_ax.set_yticks(lat_ticks)
        map_ax.set_yticklabels(format_lat_ticks(lat_ticks))
        map_ax.set_xticks(lon_ticks)
        map_ax.set_xticklabels(format_lon_ticks(lon_ticks))
        
    else:
        map_ax.set_xlim(-69.2, -68.7)
        map_ax.set_ylim(12.0, 12.4)
    
    map_ax.set_facecolor('#f7f7f7')
    map_ax.tick_params(labelsize=10)
    
    
    # === TIMESERIES PLOTTING ===
    segment_data = all_segments_data[segment]
    
    # Create twin axis for VT bars
    ax2 = ax.twinx()
    
    # === PLOT ALL CONTINUOUS TRAJECTORIES ===
    trajectories = segment_data['trajectories']
    print(f"  Plotting {len(trajectories)} continuous trajectories...")
    
    plot_start = np.datetime64('2020-04-01')
    plot_end = np.datetime64('2024-03-31')
    
    trajectory_count = 0
    for traj in trajectories:  # Only plot first 50 trajectories for test
        if len(traj['times']) == 0:
            continue
            
        # Filter to plot range but keep trajectories continuous
        traj_times = traj['times']
        traj_depths = traj['depths']
        
        time_mask = (traj_times >= plot_start) & (traj_times <= plot_end)
        
        if np.any(time_mask):
            visible_times = traj_times[time_mask]
            visible_depths = traj_depths[time_mask]
            
            # Plot continuous trajectory - NO YEAR BREAKS!
            ax.plot(visible_times, visible_depths, '-', 
                   linewidth=0.2, alpha=0.1, color=traj['color'])
            
            # Plot end point
            if DIRECTION == 'backward' and len(visible_times) > 0:
                ax.plot(visible_times[-1], visible_depths[-1], 'o', 
                       markersize=1, color=traj['color'], alpha=0.3)
            elif DIRECTION == 'forward' and len(visible_times) > 0:
                ax.plot(visible_times[0], visible_depths[0], 'o', 
                       markersize=1, color=traj['color'], alpha=0.3)
            
            trajectory_count += 1
    
    print(f"    Plotted {trajectory_count} trajectories")
    
    # === VT HISTOGRAM BARS ===
    monthly_histograms = segment_data['monthly_histograms']
    
    for month_key, month_data in monthly_histograms.items():
        month_start = month_data['month_start']
        month_end = month_data['month_end']
        
        start_num = mdates.date2num(month_start)
        end_num = mdates.date2num(month_end)
        exact_width = end_num - start_num
        
        # Stacked bars
        bottom = 0
        ax2.bar(start_num, month_data['vt_sum_deep'],
                width=exact_width, bottom=bottom,
                color='lightseagreen', alpha=0.8, align='edge')
        
        bottom += month_data['vt_sum_deep']
        ax2.bar(start_num, month_data['vt_sum_mid'],
                width=exact_width, bottom=bottom,
                color='tomato', alpha=0.8, align='edge')
        
        bottom += month_data['vt_sum_mid']
        ax2.bar(start_num, month_data['vt_sum_surface'],
                width=exact_width, bottom=bottom,
                color='cornflowerblue', alpha=0.8, align='edge')

    
    # === AXIS FORMATTING ===
    # Reference lines
    ax.axhline(y=SUBSURFACE_DEPTH_THRESHOLD, color='tomato', 
              linestyle='--', linewidth=0.8, alpha=1)
    ax.axhline(y=SURFACE_DEPTH_THRESHOLD, color='cornflowerblue', 
              linestyle='--', linewidth=0.8, alpha=1)
    
    # Main axis formatting
    ax.set_xlim(plot_start, plot_end)
    ax.set_ylim(depth_range[0], depth_range[1])
    ax.set_ylabel('Depth [m]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500))
    ax.tick_params(axis='both', labelsize=10)
    
    # VT axis formatting
    ax2.set_ylim(0, y_max_vt)
    ax2.set_ylabel('\nVolume transport\nper month [Sv]', fontsize=12)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax2.tick_params(axis='y', labelsize=10)
    
    # X-axis formatting - 6-month intervals on 6th and 12th months
    # Custom tick locations: January (1st) and July (7th) of each year
    tick_dates = []
    for year in range(2020, 2024):  # 2020 to 2023
        if year == 2020:
            tick_dates.append(datetime(year, 7, 1))
        else:
            tick_dates.append(datetime(year, 1, 1))
            tick_dates.append(datetime(year, 7, 1))
    tick_dates.append(datetime(2024, 1, 1))

    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45, labelsize=10)

    if row_idx < len(all_segments) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Date [year-month]', fontsize=12)

print("✓ All segments plotted!")

# === LEGENDS ===
print("Adding legends...")
# Remove the rampup_legend_element section and replace the three fig.legend calls with:

# Add ramp-up to the trajectory legend instead
trajectory_legend_elements = [
    plt.Line2D([0], [0], color='cornflowerblue', linewidth=3, label='Surface (0 to -162 m)'),
    plt.Line2D([0], [0], color='tomato', linewidth=3, label='Mid-depth (-162 to -458.5 m)'),
    plt.Line2D([0], [0], color='lightseagreen', linewidth=3, label='Deep (<-458.5 m)'),
    mpatches.Patch(facecolor='lightgrey', alpha=0.3, hatch='///', 
                   edgecolor='grey', label='Ramp-up period (43 days)')  # ADD THIS LINE
]
vt_legend_elements = [
    mpatches.Patch(color='cornflowerblue', alpha=0.8, label='Surface'),
    mpatches.Patch(color='tomato', alpha=0.8, label='Mid-depth'),
    mpatches.Patch(color='lightseagreen', alpha=0.8, label='Deep')
]

fig.legend(handles=trajectory_legend_elements, 
           loc='lower left', bbox_to_anchor=(0.175, 0),  # BACK TO ORIGINAL POSITION
           ncol=4, fontsize=12, title='Trajectories:                          ', title_fontsize=12)  # CHANGE ncol=3 to ncol=4

fig.legend(handles=vt_legend_elements, 
           loc='lower right', bbox_to_anchor=(0.95, 0),  # BACK TO ORIGINAL POSITION
           ncol=3, fontsize=12, title='Volume transport:', title_fontsize=12)

print("✓ Legends added!")

# === SAVE FIGURE ===
print("Saving figure...")

fig_num = 8 if DIRECTION == 'backward' else 9
filename_jpg = f"{FIGURES_DIR}/CH2_Fig{fig_num}_nearshore_ALL_SEGMENTS_{DIRECTION}.jpeg"

try:
    plt.savefig(filename_jpg, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', format='jpeg')
    print(f"✓ Saved JPEG to {filename_jpg}")
except Exception as e:
    print(f"✗ Error saving JPEG: {e}")



