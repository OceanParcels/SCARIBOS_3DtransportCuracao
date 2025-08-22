'''
Project: 3D flow and volume transport around Curaçao. 

In this script we calculate for each crossing where exactly did the particle cross.
Here, we define the segments for each cross-section, based on distance from continental Venezuela, 
depth and proximity to the island (Curaçao or Klein Curaçao).
Names:
- NS: North Section (in final manuscript: North-of-Curacao NC)
- WP: West Point
- MP: Mid Point
- KC: Klein Curacao
- SS: South Sectio (in final manuscript: South-of-Curacao SC)
In script 3_plot_segmentation_crossings.py we plot these crossings and the segments for example month.
All cross-sections have three depth layers, based on water mass analysis.
Layers: (names of segments store information of a depth layer: D1, D2, D3)
- D1: 0 m to -162 m
- D2: -162 m to -458.5 m
- D3: -458.5 m to -4500 m


Author: V Bertoncelj
kernel: parcels_shared
'''

#%%

# LIBRARIES
import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.interpolate import griddata

# Parameters
part_config = 'INFLOW4B4M'

# cross-sections:
sections = {
    "NS": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SS": [(11.41, -68.80), (12.40, -67.81)],
}
section_plotting = sections
n_points = 300 
# Create linearly spaced coordinates along each cross-section
lats_KC = np.linspace(sections["KC"][0][0], sections["KC"][1][0], n_points)
lons_KC = np.linspace(sections["KC"][0][1], sections["KC"][1][1], n_points)
lats_WP = np.linspace(sections["WP"][0][0], sections["WP"][1][0], n_points)
lons_WP = np.linspace(sections["WP"][0][1], sections["WP"][1][1], n_points)
lats_MP = np.linspace(sections["MP"][0][0], sections["MP"][1][0], n_points)
lons_MP = np.linspace(sections["MP"][0][1], sections["MP"][1][1], n_points)
lats_SS = np.linspace(sections["SS"][0][0], sections["SS"][1][0], n_points)
lons_SS = np.linspace(sections["SS"][0][1], sections["SS"][1][1], n_points)
lats_NS = np.linspace(sections["NS"][0][0], sections["NS"][1][0], n_points)
lons_NS = np.linspace(sections["NS"][0][1], sections["NS"][1][1], n_points)

# SCARIBOS (for plotting and extracting bathymetry of the cross-sections)
config = 'SCARIBOS_V8'
file_path = f'/nethome/berto006/croco/CONFIG/{config}/CROCO_FILES/croco_avg_Y2020M04.nc'
ds_scarib = xr.open_dataset(file_path)
mask_rho = ds_scarib.mask_rho.where(ds_scarib.mask_rho == 0)
lon_rho = ds_scarib.lon_rho.values
lat_rho = ds_scarib.lat_rho.values
h = -ds_scarib['h'].values  # Bathymetry (negative depth)

# Interpolate bathymetry along each cross-section
points = np.column_stack((lat_rho.ravel(), lon_rho.ravel()))
values = h.ravel()
cross_section_bathymetry_KC = griddata(points, values, (lats_KC, lons_KC), method='linear')
cross_section_bathymetry_WP = griddata(points, values, (lats_WP, lons_WP), method='linear')
cross_section_bathymetry_MP = griddata(points, values, (lats_MP, lons_MP), method='linear')
cross_section_bathymetry_SS = griddata(points, values, (lats_SS, lons_SS), method='linear')
cross_section_bathymetry_NS = griddata(points, values, (lats_NS, lons_NS), method='linear')

# funciton to load crossings (calculated with 2_calc_crossings_....py)
def load_all_crossings(section):
    """Load all crossing data from NetCDF files for all sections using dask and combine them into one xarray Dataset."""
    datasets = []
    for section in sections:
        filepath = f'../parcels_analysis/crossings_calculations/FINAL/xrcrossings_{section}_{part_month}_ALL.nc'
        if os.path.exists(filepath):
            try:
                ds = xr.open_dataset(filepath)
                ds = ds.assign_coords(section=section)
                datasets.append(ds)
                print(f"Successfully loaded {section} with {ds.sizes['crossing']} crossings")
            except Exception as e:
                print(f"Error loading {section}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    if datasets:
        combined_crossings = xr.concat(datasets, dim='crossing')
    else:
        combined_crossings = None
    
    return combined_crossings

def make_segments(all_crossings, min_distance, max_distance, min_depth, max_depth):
    segments = all_crossings.where(
        (all_crossings.distance_from_start >= min_distance) & 
        (all_crossings.distance_from_start < max_distance) & 
        (all_crossings.crossing_depth >= min_depth) & 
        (all_crossings.crossing_depth < max_depth), 
        drop=True
    )
    return segments
    
# plotting parameters
color1 = 'cornflowerblue'
color2 = 'royalblue'
color4 = 'tomato'
color3 = 'coral'
color6 = 'teal'
color5 = 'lightseagreen'

# run paralell:
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    
    year       = int(sys.argv[1])
    month      = int(sys.argv[2])
    part_month = f'Y{year}M{str(month).zfill(2)}'
    print(f"Year: {year}, Month: {month}, Part month: {part_month}")
    print("-    -")

    # ==================== SECTION: KC ====================
    section = 'KC'
    print(f"Section: {section}")
    sections = [section]

    # LOAD CROSSINGS
    all_crossings = load_all_crossings(section)
    all_crossings = all_crossings.set_coords('trajectory')
    print(all_crossings)

    # cross-section dependent parameters:
    # values in km
    vline1, vline2, vline3, vline4 = 13, 26.8, 72, 74.5
    mid_island, vline5, vline6, vline7, vline8 = 77.4, 80.2, 82.7, 121, 210
    depth1 = -162
    depth2 = -458.5

    KC_1D1 = make_segments(all_crossings, 0, vline1 * 1000,     depth1, 0)
    KC_2D1 = make_segments(all_crossings, vline1 * 1000, vline2 * 1000, depth1, 0)
    KC_3D1 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth1, 0)
    KC_4D1 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth1, 0)
    KC_5D1 = make_segments(all_crossings, vline4 * 1000, mid_island * 1000, depth1, 0)
    KC_6D1 = make_segments(all_crossings, mid_island * 1000, vline5 * 1000, depth1, 0)
    KC_7D1 = make_segments(all_crossings, vline5 * 1000, vline6 * 1000, depth1, 0)
    KC_8D1 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth1, 0)
    KC_9D1 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth1, 0)

    KC_2D2 = make_segments(all_crossings, 0, vline2 * 1000, depth2, depth1)
    KC_3D2 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth2, depth1)
    KC_4D2 = make_segments(all_crossings, vline3 * 1000, mid_island * 1000, depth2, depth1)
    KC_7D2 = make_segments(all_crossings, mid_island * 1000, vline6 * 1000, depth2, depth1)
    KC_8D2 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth2, depth1)
    KC_9D2 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth2, depth1)

    KC_3D3 = make_segments(all_crossings, 0, mid_island * 1000, -4500, depth2)
    KC_8D3 = make_segments(all_crossings, mid_island * 1000, vline7 * 1000, -4500, depth2)
    KC_9D3 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, -4500, depth2)

    # save for bar charts (8_plot_barchart_nearshore.py)
    save_dir = 'segmentation/for_plotting_barcharts'
    os.makedirs(save_dir, exist_ok=True)
    output_filepath_bar = f'{save_dir}/KC_nearshore_segment_locations.npy'
    # only ilsited ones!!
    np.save(output_filepath_bar, {
        'KC_4D1': KC_4D1,
        'KC_4D2': KC_4D2,
        'KC_5D1': KC_5D1,
        'KC_6D1': KC_6D1,
        'KC_7D1': KC_7D1,
        'KC_7D2': KC_7D2
    }, allow_pickle=True)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5)) # 12, 5

    mp_distance = np.linspace(0, geodesic(section_plotting["KC"][0], section_plotting["KC"][1]).kilometers, n_points)
    ax.scatter(all_crossings.distance_from_start / 1000, all_crossings.crossing_depth, s=1, c='yellow', label='All Crossings')

    # Plot segments with alternating colors
    ax.scatter(KC_1D1.distance_from_start / 1000, KC_1D1.crossing_depth, s=1, c=color1, label='KC_1D1')
    ax.scatter(KC_2D1.distance_from_start / 1000, KC_2D1.crossing_depth, s=1, c=color2, label='KC_2D1')
    ax.scatter(KC_3D1.distance_from_start / 1000, KC_3D1.crossing_depth, s=1, c=color1, label='KC_3D1')
    ax.scatter(KC_4D1.distance_from_start / 1000, KC_4D1.crossing_depth, s=1, c=color2, label='KC_4D1')
    ax.scatter(KC_5D1.distance_from_start / 1000, KC_5D1.crossing_depth, s=1, c=color1, label='KC_5D1')
    ax.scatter(KC_6D1.distance_from_start / 1000, KC_6D1.crossing_depth, s=1, c=color2, label='KC_6D1')
    ax.scatter(KC_7D1.distance_from_start / 1000, KC_7D1.crossing_depth, s=1, c=color1, label='KC_7D1')
    ax.scatter(KC_8D1.distance_from_start / 1000, KC_8D1.crossing_depth, s=1, c=color2, label='KC_8D1')
    ax.scatter(KC_9D1.distance_from_start / 1000, KC_9D1.crossing_depth, s=1, c=color1, label='KC_9D1')

    ax.scatter(KC_2D2.distance_from_start / 1000, KC_2D2.crossing_depth, s=1, c=color3, label='KC_2D2')
    ax.scatter(KC_3D2.distance_from_start / 1000, KC_3D2.crossing_depth, s=1, c=color4, label='KC_3D2')
    ax.scatter(KC_4D2.distance_from_start / 1000, KC_4D2.crossing_depth, s=1, c=color3, label='KC_4D2')
    ax.scatter(KC_7D2.distance_from_start / 1000, KC_7D2.crossing_depth, s=1, c=color4, label='KC_7D2')
    ax.scatter(KC_8D2.distance_from_start / 1000, KC_8D2.crossing_depth, s=1, c=color3, label='KC_8D2')
    ax.scatter(KC_9D2.distance_from_start / 1000, KC_9D2.crossing_depth, s=1, c=color4, label='KC_9D2')

    ax.scatter(KC_3D3.distance_from_start / 1000, KC_3D3.crossing_depth, s=1, c=color5, label='KC_3D3')
    ax.scatter(KC_8D3.distance_from_start / 1000, KC_8D3.crossing_depth, s=1, c=color6, label='KC_8D3')
    ax.scatter(KC_9D3.distance_from_start / 1000, KC_9D3.crossing_depth, s=1, c=color5, label='KC_9D3')

    ax.set_xlabel('Distance from Venezuela (mainland) [km]')
    ax.set_ylabel('Depth [m]')

    ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(mid_island, color='w', alpha=1, linewidth=0.5)
    ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline5, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline6, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline7, color='w', linestyle='--', alpha=1, linewidth=0.5)

    ax.plot(mp_distance, cross_section_bathymetry_KC, color='dimgrey', lw=0.5)
    ax.fill_between(mp_distance, cross_section_bathymetry_KC, -4600, color='k', alpha=0.8)
    ax.set_ylim(-4600, 200)
    ax.set_xlim(0, vline8)

    ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)

    # add section names in the plot
    ax.text(vline1/2, -100, 'KC_1D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline1 + vline2)/2, -100, 'KC_2D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -100, 'KC_3D1', fontsize=8, ha='center', va='center', color='w')
    ax.annotate('KC_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-15, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('KC_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-5, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('KC_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+5, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('KC_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+15, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.text((vline6 + vline7)/2, -100, 'KC_8D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -100, 'KC_9D1', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline1 + vline2)/2, -350, 'KC_2D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -350, 'KC_3D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + mid_island)/2-3, -350, 'KC_4D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((mid_island + vline5)/2+3, -350, 'KC_7D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline6 + vline7)/2, -350, 'KC_8D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -350, 'KC_9D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -800, 'KC_3D3', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline6 + vline7)/2, -800, 'KC_8D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -800, 'KC_9D3', fontsize=8, ha='center', va='center', color='w')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    output_dir = 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/segments_{section}_{part_month}.png'
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    # find the largest distance
    max_distance = all_crossings.distance_from_start.max().item()
    print(f"The largest distance is: {max_distance / 1000} km")

    # Calculate mean lon, lat, and depth for each segment and save as npy
    segments = {
        'KC_1D1': KC_1D1,
        'KC_2D1': KC_2D1,
        'KC_3D1': KC_3D1,
        'KC_4D1': KC_4D1,
        'KC_5D1': KC_5D1,
        'KC_6D1': KC_6D1,
        'KC_7D1': KC_7D1,
        'KC_8D1': KC_8D1,
        'KC_9D1': KC_9D1,
        'KC_2D2': KC_2D2,
        'KC_3D2': KC_3D2,
        'KC_4D2': KC_4D2,
        'KC_7D2': KC_7D2,
        'KC_8D2': KC_8D2,
        'KC_9D2': KC_9D2,
        'KC_3D3': KC_3D3,
        'KC_8D3': KC_8D3,
        'KC_9D3': KC_9D3
    }

    mean_values = {}
    for name, segment in segments.items():
        mean_lon = (segment.crossing_lon.min().item() + segment.crossing_lon.max().item()) / 2
        min_lon = segment.crossing_lon.min().item()
        max_lon = segment.crossing_lon.max().item()
        mean_lat = (segment.crossing_lat.min().item() + segment.crossing_lat.max().item()) / 2
        min_lat = segment.crossing_lat.min().item()
        max_lat = segment.crossing_lat.max().item()
        mean_depth = segment.crossing_depth.mean().item()
        mean_values[name] = {'mean_lon': mean_lon, 'mean_lat': mean_lat, 'mean_depth': mean_depth, 'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/locations_{section}_{part_month}.npy'
    np.save(output_filepath, mean_values)

    # save segments 
    unique_traj = np.unique(all_crossings['trajectory'].values)
    segment_names = {}

    for traj in unique_traj:
        segments = []
        segments_time = []
        if traj in KC_1D1.trajectory.values:
            segments.append('KC_1D1')
            mask = KC_1D1.trajectory == traj
            segments_time.append(KC_1D1['time'].values[mask].tolist())
        if traj in KC_2D1.trajectory.values:
            segments.append('KC_2D1')
            mask = KC_2D1.trajectory == traj
            segments_time.append(KC_2D1['time'].values[mask].tolist())
        if traj in KC_3D1.trajectory.values:
            segments.append('KC_3D1')
            mask = KC_3D1.trajectory == traj
            segments_time.append(KC_3D1['time'].values[mask].tolist())
        if traj in KC_4D1.trajectory.values:
            segments.append('KC_4D1')
            mask = KC_4D1.trajectory == traj
            segments_time.append(KC_4D1['time'].values[mask].tolist())
        if traj in KC_5D1.trajectory.values:
            segments.append('KC_5D1')
            mask = KC_5D1.trajectory == traj
            segments_time.append(KC_5D1['time'].values[mask].tolist())
        if traj in KC_6D1.trajectory.values:
            segments.append('KC_6D1')
            mask = KC_6D1.trajectory == traj
            segments_time.append(KC_6D1['time'].values[mask].tolist())
        if traj in KC_7D1.trajectory.values:
            segments.append('KC_7D1')
            mask = KC_7D1.trajectory == traj
            segments_time.append(KC_7D1['time'].values[mask].tolist())
        if traj in KC_8D1.trajectory.values:
            segments.append('KC_8D1')
            mask = KC_8D1.trajectory == traj
            segments_time.append(KC_8D1['time'].values[mask].tolist())
        if traj in KC_9D1.trajectory.values:
            segments.append('KC_9D1')
            mask = KC_9D1.trajectory == traj
            segments_time.append(KC_9D1['time'].values[mask].tolist())
        if traj in KC_2D2.trajectory.values:
            segments.append('KC_2D2')
            mask = KC_2D2.trajectory == traj
            segments_time.append(KC_2D2['time'].values[mask].tolist())
        if traj in KC_3D2.trajectory.values:
            segments.append('KC_3D2')
            mask = KC_3D2.trajectory == traj
            segments_time.append(KC_3D2['time'].values[mask].tolist())
        if traj in KC_4D2.trajectory.values:
            segments.append('KC_4D2')
            mask = KC_4D2.trajectory == traj
            segments_time.append(KC_4D2['time'].values[mask].tolist())
        if traj in KC_7D2.trajectory.values:
            segments.append('KC_7D2')
            mask = KC_7D2.trajectory == traj
            segments_time.append(KC_7D2['time'].values[mask].tolist())
        if traj in KC_8D2.trajectory.values:
            segments.append('KC_8D2')
            mask = KC_8D2.trajectory == traj
            segments_time.append(KC_8D2['time'].values[mask].tolist())
        if traj in KC_9D2.trajectory.values:
            segments.append('KC_9D2')
            mask = KC_9D2.trajectory == traj
            segments_time.append(KC_9D2['time'].values[mask].tolist())
        if traj in KC_3D3.trajectory.values:
            segments.append('KC_3D3')
            mask = KC_3D3.trajectory == traj
            segments_time.append(KC_3D3['time'].values[mask].tolist())
        if traj in KC_8D3.trajectory.values:
            segments.append('KC_8D3')
            mask = KC_8D3.trajectory == traj
            segments_time.append(KC_8D3['time'].values[mask].tolist())
        if traj in KC_9D3.trajectory.values:
            segments.append('KC_9D3')
            mask = KC_9D3.trajectory == traj
            segments_time.append(KC_9D3['time'].values[mask].tolist())
        if segments:
            segment_names[str(traj)] = {'segments': segments, 'times': segments_time}
        else:
            print(f"No segment found for trajectory {traj}")

    # save the dictionary to a json file
    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/segment_names_{section}_{part_month}.json'
    with open(output_filepath, 'w') as f:
        json.dump(segment_names, f)

    print(f'DONE SEGMENTATION FOR {section} {part_month}')
    print("-    -")


    # ==================== SECTION: MP ====================
    section = 'MP'
    print(f"Section: {section}")
    sections = [section]

    # LOAD CROSSINGS
    all_crossings = load_all_crossings(section)
    all_crossings = all_crossings.set_coords('trajectory')
    print(all_crossings)
    # Define segmentation lines for MP section
    vline1, vline2, vline3, vline4 = 34.5, 47.5, 96.3, 99
    mid_island, vline5, vline6, vline7, vline8 = 106.5, 115, 117.8, 172, 227
    depth1 = -162
    depth2 = -458.5

    MP_1D1 = make_segments(all_crossings, 0, vline1 *1000,     depth1, 0)
    MP_2D1 = make_segments(all_crossings, vline1 * 1000, vline2 * 1000, depth1, 0)
    MP_3D1 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth1, 0)
    MP_4D1 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth1, 0)
    MP_5D1 = make_segments(all_crossings, vline4 * 1000, mid_island * 1000, depth1, 0)
    MP_6D1 = make_segments(all_crossings, mid_island * 1000, vline5 * 1000, depth1, 0)
    MP_7D1 = make_segments(all_crossings, vline5 * 1000, vline6 * 1000, depth1, 0)
    MP_8D1 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth1, 0)
    MP_9D1 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth1, 0)

    MP_2D2 = make_segments(all_crossings, 0, vline2 * 1000, depth2, depth1)
    MP_3D2 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth2, depth1)
    MP_4D2 = make_segments(all_crossings, vline3 * 1000, mid_island * 1000, depth2, depth1)
    MP_7D2 = make_segments(all_crossings, mid_island * 1000, vline6 * 1000, depth2, depth1)
    MP_8D2 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth2, depth1)
    MP_9D2 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth2, depth1)

    MP_3D3 = make_segments(all_crossings, 0, mid_island * 1000, -4500, depth2)
    MP_8D3 = make_segments(all_crossings, mid_island * 1000, vline7 * 1000, -4500, depth2)
    MP_9D3 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, -4500, depth2)

    # save for bar charts (8_plot_barchart_nearshore.py)
    save_dir = 'segmentation/for_plotting_barcharts'
    os.makedirs(save_dir, exist_ok=True)
    output_filepath_bar = f'{save_dir}/MP_nearshore_segment_locations.npy'
    # only ilsited ones!!
    np.save(output_filepath_bar, {
        'MP_4D1': MP_4D1,
        'MP_4D2': MP_4D2,
        'MP_5D1': MP_5D1,
        'MP_6D1': MP_6D1,
        'MP_7D1': MP_7D1,
        'MP_7D2': MP_7D2
    }, allow_pickle=True)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5))

    mp_distance = np.linspace(0, geodesic(section_plotting["MP"][0], section_plotting["MP"][1]).kilometers, n_points)
    ax.scatter(all_crossings.distance_from_start / 1000, all_crossings.crossing_depth, s=1, c='yellow', label='All Crossings')

    # Plot segments with alternating colors
    ax.scatter(MP_1D1.distance_from_start / 1000, MP_1D1.crossing_depth, s=1, c=color1, label='MP_1D1')
    ax.scatter(MP_2D1.distance_from_start / 1000, MP_2D1.crossing_depth, s=1, c=color2, label='MP_2D1')
    ax.scatter(MP_3D1.distance_from_start / 1000, MP_3D1.crossing_depth, s=1, c=color1, label='MP_3D1')
    ax.scatter(MP_4D1.distance_from_start / 1000, MP_4D1.crossing_depth, s=1, c=color2, label='MP_4D1')
    ax.scatter(MP_5D1.distance_from_start / 1000, MP_5D1.crossing_depth, s=1, c=color1, label='MP_5D1')
    ax.scatter(MP_6D1.distance_from_start / 1000, MP_6D1.crossing_depth, s=1, c=color2, label='MP_6D1')
    ax.scatter(MP_7D1.distance_from_start / 1000, MP_7D1.crossing_depth, s=1, c=color1, label='MP_7D1')
    ax.scatter(MP_8D1.distance_from_start / 1000, MP_8D1.crossing_depth, s=1, c=color2, label='MP_8D1')
    ax.scatter(MP_9D1.distance_from_start / 1000, MP_9D1.crossing_depth, s=1, c=color1, label='MP_9D1')

    ax.scatter(MP_2D2.distance_from_start / 1000, MP_2D2.crossing_depth, s=1, c=color3, label='MP_2D2')
    ax.scatter(MP_3D2.distance_from_start / 1000, MP_3D2.crossing_depth, s=1, c=color4, label='MP_3D2')
    ax.scatter(MP_4D2.distance_from_start / 1000, MP_4D2.crossing_depth, s=1, c=color3, label='MP_4D2')
    ax.scatter(MP_7D2.distance_from_start / 1000, MP_7D2.crossing_depth, s=1, c=color4, label='MP_7D2')
    ax.scatter(MP_8D2.distance_from_start / 1000, MP_8D2.crossing_depth, s=1, c=color3, label='MP_8D2')
    ax.scatter(MP_9D2.distance_from_start / 1000, MP_9D2.crossing_depth, s=1, c=color4, label='MP_9D2')

    ax.scatter(MP_3D3.distance_from_start / 1000, MP_3D3.crossing_depth, s=1, c=color5, label='MP_3D3')
    ax.scatter(MP_8D3.distance_from_start / 1000, MP_8D3.crossing_depth, s=1, c=color6, label='MP_8D3')
    ax.scatter(MP_9D3.distance_from_start / 1000, MP_9D3.crossing_depth, s=1, c=color5, label='MP_9D3')

    ax.set_xlabel('Distance from Venezuela (mainland) [km]')
    ax.set_ylabel('Depth [m]')

    ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(mid_island, color='w', alpha=1, linewidth=0.5)
    ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline5, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline6, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline7, color='w', linestyle='--', alpha=1, linewidth=0.5)

    ax.plot(mp_distance, cross_section_bathymetry_MP, color='dimgrey', lw=0.5)
    ax.fill_between(mp_distance, cross_section_bathymetry_MP, -4500, color='k', alpha=0.8)
    ax.set_ylim(-3400, 200)
    ax.set_xlim(0, vline8)
    ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.text(vline1/2, -100, 'MP_1D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline1 + vline2)/2, -100, 'MP_2D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -100, 'MP_3D1', fontsize=8, ha='center', va='center', color='w')
    ax.annotate('MP_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-10, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('MP_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-3, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('MP_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+3, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('MP_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+10, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.text((vline6 + vline7)/2, -100, 'MP_8D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -100, 'MP_9D1', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline1 + vline2)/2, -350, 'MP_2D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -350, 'MP_3D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + mid_island)/2-3, -350, 'MP_4D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((mid_island + vline5)/2+3, -350, 'MP_7D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline6 + vline7)/2, -350, 'MP_8D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -350, 'MP_9D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -800, 'MP_3D3', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline6 + vline7)/2, -800, 'MP_8D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -800, 'MP_9D3', fontsize=8, ha='center', va='center', color='w')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    output_dir = 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/segments_{section}_{part_month}.png'
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    # Calculate mean lon, lat, and depth for each segment and save as npy
    segments = {
        'MP_1D1': MP_1D1,
        'MP_2D1': MP_2D1,
        'MP_3D1': MP_3D1,
        'MP_4D1': MP_4D1,
        'MP_5D1': MP_5D1,
        'MP_6D1': MP_6D1,
        'MP_7D1': MP_7D1,
        'MP_8D1': MP_8D1,
        'MP_9D1': MP_9D1,
        'MP_2D2': MP_2D2,
        'MP_3D2': MP_3D2,
        'MP_4D2': MP_4D2,
        'MP_7D2': MP_7D2,
        'MP_8D2': MP_8D2,
        'MP_9D2': MP_9D2,
        'MP_3D3': MP_3D3,
        'MP_8D3': MP_8D3,
        'MP_9D3': MP_9D3
    }

    mean_values = {}
    for name, segment in segments.items():
        mean_lon = (segment.crossing_lon.min().item() + segment.crossing_lon.max().item()) / 2
        min_lon = segment.crossing_lon.min().item()
        max_lon = segment.crossing_lon.max().item()
        mean_lat = (segment.crossing_lat.min().item() + segment.crossing_lat.max().item()) / 2
        min_lat = segment.crossing_lat.min().item()
        max_lat = segment.crossing_lat.max().item()
        mean_depth = segment.crossing_depth.mean().item()
        mean_values[name] = {'mean_lon': mean_lon, 'mean_lat': mean_lat, 'mean_depth': mean_depth, 'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/locations_{section}_{part_month}.npy'
    np.save(output_filepath, mean_values)

    # save segments 
    unique_traj = np.unique(all_crossings['trajectory'].values)
    segment_names = {}

    for traj in unique_traj:
        segments = []
        segments_time = []
        if traj in MP_1D1.trajectory.values:
            segments.append('MP_1D1')
            mask = MP_1D1.trajectory == traj
            segments_time.append(MP_1D1['time'].values[mask].tolist())
        if traj in MP_2D1.trajectory.values:
            segments.append('MP_2D1')
            mask = MP_2D1.trajectory == traj
            segments_time.append(MP_2D1['time'].values[mask].tolist())
        if traj in MP_3D1.trajectory.values:
            segments.append('MP_3D1')
            mask = MP_3D1.trajectory == traj
            segments_time.append(MP_3D1['time'].values[mask].tolist())
        if traj in MP_4D1.trajectory.values:
            segments.append('MP_4D1')
            mask = MP_4D1.trajectory == traj
            segments_time.append(MP_4D1['time'].values[mask].tolist())
        if traj in MP_5D1.trajectory.values:
            segments.append('MP_5D1')
            mask = MP_5D1.trajectory == traj
            segments_time.append(MP_5D1['time'].values[mask].tolist())
        if traj in MP_6D1.trajectory.values:
            segments.append('MP_6D1')
            mask = MP_6D1.trajectory == traj
            segments_time.append(MP_6D1['time'].values[mask].tolist())
        if traj in MP_7D1.trajectory.values:
            segments.append('MP_7D1')
            mask = MP_7D1.trajectory == traj
            segments_time.append(MP_7D1['time'].values[mask].tolist())
        if traj in MP_8D1.trajectory.values:
            segments.append('MP_8D1')
            mask = MP_8D1.trajectory == traj
            segments_time.append(MP_8D1['time'].values[mask].tolist())
        if traj in MP_9D1.trajectory.values:
            segments.append('MP_9D1')
            mask = MP_9D1.trajectory == traj
            segments_time.append(MP_9D1['time'].values[mask].tolist())
        if traj in MP_2D2.trajectory.values:
            segments.append('MP_2D2')
            mask = MP_2D2.trajectory == traj
            segments_time.append(MP_2D2['time'].values[mask].tolist())
        if traj in MP_3D2.trajectory.values:
            segments.append('MP_3D2')
            mask = MP_3D2.trajectory == traj
            segments_time.append(MP_3D2['time'].values[mask].tolist())
        if traj in MP_4D2.trajectory.values:
            segments.append('MP_4D2')
            mask = MP_4D2.trajectory == traj
            segments_time.append(MP_4D2['time'].values[mask].tolist())
        if traj in MP_7D2.trajectory.values:
            segments.append('MP_7D2')
            mask = MP_7D2.trajectory == traj
            segments_time.append(MP_7D2['time'].values[mask].tolist())
        if traj in MP_8D2.trajectory.values:
            segments.append('MP_8D2')
            mask = MP_8D2.trajectory == traj
            segments_time.append(MP_8D2['time'].values[mask].tolist())
        if traj in MP_9D2.trajectory.values:
            segments.append('MP_9D2')
            mask = MP_9D2.trajectory == traj
            segments_time.append(MP_9D2['time'].values[mask].tolist())
        if traj in MP_3D3.trajectory.values:
            segments.append('MP_3D3')
            mask = MP_3D3.trajectory == traj
            segments_time.append(MP_3D3['time'].values[mask].tolist())
        if traj in MP_8D3.trajectory.values:
            segments.append('MP_8D3')
            mask = MP_8D3.trajectory == traj
            segments_time.append(MP_8D3['time'].values[mask].tolist())
        if traj in MP_9D3.trajectory.values:
            segments.append('MP_9D3')
            mask = MP_9D3.trajectory == traj
            segments_time.append(MP_9D3['time'].values[mask].tolist())
        if segments:
            segment_names[str(traj)] = {'segments': segments, 'times': segments_time}
        else:
            print(f"No segment found for trajectory {traj}")


    # save the dictionary to a json file
    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/segment_names_{section}_{part_month}.json'
    with open(output_filepath, 'w') as f:
        json.dump(segment_names, f)

    print(f'DONE SEGMENTATION FOR {section} {part_month}')
    print("-    -")

    # ==================== SECTION: WP ====================
    section = 'WP'
    print(f"Section: {section}")
    sections = [section]

    # LOAD CROSSINGS
    all_crossings = load_all_crossings(section)
    all_crossings = all_crossings.set_coords('trajectory')
    print(all_crossings)

    # Define segmentation lines for WP section
    vline1, vline2, vline3, vline4 = 22.5, 41.3, 88.7, 92
    mid_island, vline5, vline6, vline7, vline8 = 98, 103.2, 105.9, 146.4, 187
    depth1 = -162
    depth2 = -458.5

    WP_1D1 = make_segments(all_crossings, 0, vline1 * 1000,     depth1, 0)
    WP_2D1 = make_segments(all_crossings, vline1 * 1000, vline2 * 1000, depth1, 0)
    WP_3D1 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth1, 0)
    WP_4D1 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth1, 0)
    WP_5D1 = make_segments(all_crossings, vline4 * 1000, mid_island * 1000, depth1, 0)
    WP_6D1 = make_segments(all_crossings, mid_island * 1000, vline5 * 1000, depth1, 0)
    WP_7D1 = make_segments(all_crossings, vline5 * 1000, vline6 * 1000, depth1, 0)
    WP_8D1 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth1, 0)
    WP_9D1 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth1, 0)

    WP_2D2 = make_segments(all_crossings, 0, vline2 * 1000, depth2, depth1)
    WP_3D2 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth2, depth1)
    WP_4D2 = make_segments(all_crossings, vline3 * 1000, mid_island * 1000, depth2, depth1)
    WP_7D2 = make_segments(all_crossings, mid_island * 1000, vline6 * 1000, depth2, depth1)
    WP_8D2 = make_segments(all_crossings, vline6 * 1000, vline7 * 1000, depth2, depth1)
    WP_9D2 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, depth2, depth1)

    WP_3D3 = make_segments(all_crossings, 0, mid_island * 1000, -4500, depth2)
    WP_8D3 = make_segments(all_crossings, mid_island * 1000, vline7 * 1000, -4500, depth2)
    WP_9D3 = make_segments(all_crossings, vline7 * 1000, vline8 * 1000, -4500, depth2)

    # save for bar charts (8_plot_barchart_nearshore.py)
    save_dir = 'segmentation/for_plotting_barcharts'
    os.makedirs(save_dir, exist_ok=True)
    output_filepath_bar = f'{save_dir}/WP_nearshore_segment_locations.npy'
    # only ilsited ones!!
    np.save(output_filepath_bar, {
        'WP_4D1': WP_4D1,
        'WP_4D2': WP_4D2,
        'WP_5D1': WP_5D1,
        'WP_6D1': WP_6D1,
        'WP_7D1': WP_7D1,
        'WP_7D2': WP_7D2
    }, allow_pickle=True)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5))

    mp_distance = np.linspace(0, geodesic(section_plotting["WP"][0], section_plotting["WP"][1]).kilometers, n_points)
    ax.scatter(all_crossings.distance_from_start / 1000, all_crossings.crossing_depth, s=1, c='yellow', label='All Crossings')

    # Plot segments with alternating colors
    ax.scatter(WP_1D1.distance_from_start / 1000, WP_1D1.crossing_depth, s=1, c=color1, label='WP_1D1')
    ax.scatter(WP_2D1.distance_from_start / 1000, WP_2D1.crossing_depth, s=1, c=color2, label='WP_2D1')
    ax.scatter(WP_3D1.distance_from_start / 1000, WP_3D1.crossing_depth, s=1, c=color1, label='WP_3D1')
    ax.scatter(WP_4D1.distance_from_start / 1000, WP_4D1.crossing_depth, s=1, c=color2, label='WP_4D1')
    ax.scatter(WP_5D1.distance_from_start / 1000, WP_5D1.crossing_depth, s=1, c=color1, label='WP_5D1')
    ax.scatter(WP_6D1.distance_from_start / 1000, WP_6D1.crossing_depth, s=1, c=color2, label='WP_6D1')
    ax.scatter(WP_7D1.distance_from_start / 1000, WP_7D1.crossing_depth, s=1, c=color1, label='WP_7D1')
    ax.scatter(WP_8D1.distance_from_start / 1000, WP_8D1.crossing_depth, s=1, c=color2, label='WP_8D1')
    ax.scatter(WP_9D1.distance_from_start / 1000, WP_9D1.crossing_depth, s=1, c=color1, label='WP_9D1')

    ax.scatter(WP_2D2.distance_from_start / 1000, WP_2D2.crossing_depth, s=1, c=color3, label='WP_2D2')
    ax.scatter(WP_3D2.distance_from_start / 1000, WP_3D2.crossing_depth, s=1, c=color4, label='WP_3D2')
    ax.scatter(WP_4D2.distance_from_start / 1000, WP_4D2.crossing_depth, s=1, c=color3, label='WP_4D2')
    ax.scatter(WP_7D2.distance_from_start / 1000, WP_7D2.crossing_depth, s=1, c=color4, label='WP_7D2')
    ax.scatter(WP_8D2.distance_from_start / 1000, WP_8D2.crossing_depth, s=1, c=color3, label='WP_8D2')
    ax.scatter(WP_9D2.distance_from_start / 1000, WP_9D2.crossing_depth, s=1, c=color4, label='WP_9D2')

    ax.scatter(WP_3D3.distance_from_start / 1000, WP_3D3.crossing_depth, s=1, c=color5, label='WP_3D3')
    ax.scatter(WP_8D3.distance_from_start / 1000, WP_8D3.crossing_depth, s=1, c=color6, label='WP_8D3')
    ax.scatter(WP_9D3.distance_from_start / 1000, WP_9D3.crossing_depth, s=1, c=color5, label='WP_9D3')

    ax.set_xlabel('Distance from Venezuela (mainland) [km]')
    ax.set_ylabel('Depth [m]')

    ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(mid_island, color='w', alpha=1, linewidth=0.5)
    ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline5, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline6, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline7, color='w', linestyle='--', alpha=1, linewidth=0.5)

    ax.plot(mp_distance, cross_section_bathymetry_WP, color='dimgrey', lw=0.5)
    ax.fill_between(mp_distance, cross_section_bathymetry_WP, -4500, color='k', alpha=0.8)
    ax.set_ylim(-2800, 200)
    ax.set_xlim(0,vline8)
    ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)

    ax.text(vline1/2, -100, 'WP_1D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline1 + vline2)/2, -100, 'WP_2D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -100, 'WP_3D1', fontsize=8, ha='center', va='center', color='w')
    ax.annotate('WP_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-10, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('WP_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-3, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('WP_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+3, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.annotate('WP_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+10, 150),
                fontsize=8, ha='center', va='center', color='black',
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                rotation=10)
    ax.text((vline6 + vline7)/2, -100, 'WP_8D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -100, 'WP_9D1', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline1 + vline2)/2, -350, 'WP_2D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -350, 'WP_3D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + mid_island)/2-3, -350, 'WP_4D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((mid_island + vline5)/2+3, -350, 'WP_7D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline6 + vline7)/2, -350, 'WP_8D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline7 + vline8)/2, -350, 'WP_9D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -800, 'WP_3D3', fontsize=8, ha='center', va='center', color='k')

    ax.text((vline6 + vline7)/2, -800, 'WP_8D3', fontsize=8, ha='center', va='center', color='k')
    ax.text((vline7 + vline8)/2, -800, 'WP_9D3', fontsize=8, ha='center', va='center', color='k')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    output_dir = 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/segments_{section}_{part_month}.png'
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    # find the largest distance
    max_distance = all_crossings.distance_from_start.max().item()
    print(f"The largest distance is: {max_distance / 1000} km")


    # Calculate mean lon, lat, and depth for each segment and save as npy
    segments = {
        'WP_1D1': WP_1D1,
        'WP_2D1': WP_2D1,
        'WP_3D1': WP_3D1,
        'WP_4D1': WP_4D1,
        'WP_5D1': WP_5D1,
        'WP_6D1': WP_6D1,
        'WP_7D1': WP_7D1,
        'WP_8D1': WP_8D1,
        'WP_9D1': WP_9D1,
        'WP_2D2': WP_2D2,
        'WP_3D2': WP_3D2,
        'WP_4D2': WP_4D2,
        'WP_7D2': WP_7D2,
        'WP_8D2': WP_8D2,
        'WP_9D2': WP_9D2,
        'WP_3D3': WP_3D3,
        'WP_8D3': WP_8D3,
        'WP_9D3': WP_9D3
    }

    mean_values = {}
    for name, segment in segments.items():
        mean_lon = (segment.crossing_lon.min().item() + segment.crossing_lon.max().item()) / 2
        min_lon = segment.crossing_lon.min().item()
        max_lon = segment.crossing_lon.max().item()
        mean_lat = (segment.crossing_lat.min().item() + segment.crossing_lat.max().item()) / 2
        min_lat = segment.crossing_lat.min().item()
        max_lat = segment.crossing_lat.max().item()
        mean_depth = segment.crossing_depth.mean().item()
        mean_values[name] = {'mean_lon': mean_lon, 'mean_lat': mean_lat, 'mean_depth': mean_depth, 'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/locations_{section}_{part_month}.npy'
    np.save(output_filepath, mean_values)

    # save segments
    unique_traj = np.unique(all_crossings['trajectory'].values)
    segment_names = {}
    for traj in unique_traj:
        segments = []
        segments_time = []
        if traj in WP_1D1.trajectory.values:
            segments.append('WP_1D1')
            mask = WP_1D1.trajectory == traj
            segments_time.append(WP_1D1['time'].values[mask].tolist())
        if traj in WP_2D1.trajectory.values:
            segments.append('WP_2D1')
            mask = WP_2D1.trajectory == traj
            segments_time.append(WP_2D1['time'].values[mask].tolist())
        if traj in WP_3D1.trajectory.values:
            segments.append('WP_3D1')
            mask = WP_3D1.trajectory == traj
            segments_time.append(WP_3D1['time'].values[mask].tolist())
        if traj in WP_4D1.trajectory.values:
            segments.append('WP_4D1')
            mask = WP_4D1.trajectory == traj
            segments_time.append(WP_4D1['time'].values[mask].tolist())
        if traj in WP_5D1.trajectory.values:
            segments.append('WP_5D1')
            mask = WP_5D1.trajectory == traj
            segments_time.append(WP_5D1['time'].values[mask].tolist())
        if traj in WP_6D1.trajectory.values:
            segments.append('WP_6D1')
            mask = WP_6D1.trajectory == traj
            segments_time.append(WP_6D1['time'].values[mask].tolist())
        if traj in WP_7D1.trajectory.values:
            segments.append('WP_7D1')
            mask = WP_7D1.trajectory == traj
            segments_time.append(WP_7D1['time'].values[mask].tolist())
        if traj in WP_8D1.trajectory.values:
            segments.append('WP_8D1')
            mask = WP_8D1.trajectory == traj
            segments_time.append(WP_8D1['time'].values[mask].tolist())
        if traj in WP_9D1.trajectory.values:
            segments.append('WP_9D1')
            mask = WP_9D1.trajectory == traj
            segments_time.append(WP_9D1['time'].values[mask].tolist())
        if traj in WP_2D2.trajectory.values:
            segments.append('WP_2D2')
            mask = WP_2D2.trajectory == traj
            segments_time.append(WP_2D2['time'].values[mask].tolist())
        if traj in WP_3D2.trajectory.values:
            segments.append('WP_3D2')
            mask = WP_3D2.trajectory == traj
            segments_time.append(WP_3D2['time'].values[mask].tolist())
        if traj in WP_4D2.trajectory.values:
            segments.append('WP_4D2')
            mask = WP_4D2.trajectory == traj
            segments_time.append(WP_4D2['time'].values[mask].tolist())
        if traj in WP_7D2.trajectory.values:
            segments.append('WP_7D2')
            mask = WP_7D2.trajectory == traj
            segments_time.append(WP_7D2['time'].values[mask].tolist())
        if traj in WP_8D2.trajectory.values:
            segments.append('WP_8D2')
            mask = WP_8D2.trajectory == traj
            segments_time.append(WP_8D2['time'].values[mask].tolist())
        if traj in WP_9D2.trajectory.values:
            segments.append('WP_9D2')
            mask = WP_9D2.trajectory == traj
            segments_time.append(WP_9D2['time'].values[mask].tolist())
        if traj in WP_3D3.trajectory.values:
            segments.append('WP_3D3')
            mask = WP_3D3.trajectory == traj
            segments_time.append(WP_3D3['time'].values[mask].tolist())
        if traj in WP_8D3.trajectory.values:
            segments.append('WP_8D3')
            mask = WP_8D3.trajectory == traj
            segments_time.append(WP_8D3['time'].values[mask].tolist())
        if traj in WP_9D3.trajectory.values:
            segments.append('WP_9D3')
            mask = WP_9D3.trajectory == traj
            segments_time.append(WP_9D3['time'].values[mask].tolist())
        if segments:
            segment_names[str(traj)] = {'segments': segments, 'times': segments_time}
        else:
            print(f"No segment found for trajectory {traj}")

    # save the dictionary to a json file
    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/segment_names_{section}_{part_month}.json'
    with open(output_filepath, 'w') as f:
        json.dump(segment_names, f)

    print(f'DONE SEGMENTATION FOR {section} {part_month}')
    print("-    -")

    # ==================== SECTION: SS ====================
    section = 'SS'
    print(f"Section: {section}")
    sections = [section]

    # LOAD CROSSINGS
    all_crossings = load_all_crossings(section)
    all_crossings = all_crossings.set_coords('trajectory')
    print(all_crossings)

    # Segmentation lines for SS section
    vline1, vline2, vline3, vline4, vline8 = 10.5, 20.8, 57.5, 94.5, 154.5
    depth1 = -162
    depth2 = -458.5

    SS_1D1 = make_segments(all_crossings, 0, vline1 * 1000,     depth1, 0)
    SS_2D1 = make_segments(all_crossings, vline1 * 1000, vline2 * 1000, depth1, 0)
    SS_3D1 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth1, 0)
    SS_4D1 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth1, 0)
    SS_5D1 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, depth1, 0)

    SS_2D2 = make_segments(all_crossings, 0, vline2 * 1000, depth2, depth1)
    SS_3D2 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth2, depth1)
    SS_4D2 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth2, depth1)
    SS_5D2 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, depth2, depth1)

    SS_3D3 = make_segments(all_crossings, 0, vline3 * 1000, -4500, depth2)
    SS_4D3 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, -4500, depth2)
    SS_5D3 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, -4500, depth2)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5))

    mp_distance = np.linspace(0, geodesic(section_plotting["SS"][0], section_plotting["SS"][1]).kilometers, n_points)
    ax.scatter(all_crossings.distance_from_start / 1000, all_crossings.crossing_depth, s=1, c='yellow', label='All Crossings')

    # Plot segments with alternating colors
    ax.scatter(SS_1D1.distance_from_start / 1000, SS_1D1.crossing_depth, s=1, c=color1, label='SS_1D1')
    ax.scatter(SS_2D1.distance_from_start / 1000, SS_2D1.crossing_depth, s=1, c=color2, label='SS_2D1')
    ax.scatter(SS_3D1.distance_from_start / 1000, SS_3D1.crossing_depth, s=1, c=color1, label='SS_3D1')
    ax.scatter(SS_4D1.distance_from_start / 1000, SS_4D1.crossing_depth, s=1, c=color2, label='SS_4D1')
    ax.scatter(SS_5D1.distance_from_start / 1000, SS_5D1.crossing_depth, s=1, c=color1, label='SS_5D1')

    ax.scatter(SS_2D2.distance_from_start / 1000, SS_2D2.crossing_depth, s=1, c=color3, label='SS_2D2')
    ax.scatter(SS_3D2.distance_from_start / 1000, SS_3D2.crossing_depth, s=1, c=color4, label='SS_3D2')
    ax.scatter(SS_4D2.distance_from_start / 1000, SS_4D2.crossing_depth, s=1, c=color3, label='SS_4D2')
    ax.scatter(SS_5D2.distance_from_start / 1000, SS_5D2.crossing_depth, s=1, c=color4, label='SS_5D2')

    ax.scatter(SS_3D3.distance_from_start / 1000, SS_3D3.crossing_depth, s=1, c=color5, label='SS_3D3')
    ax.scatter(SS_4D3.distance_from_start / 1000, SS_4D3.crossing_depth, s=1, c=color6, label='SS_4D3')
    ax.scatter(SS_5D3.distance_from_start / 1000, SS_5D3.crossing_depth, s=1, c=color5, label='SS_5D3')

    ax.set_xlabel('Distance from Venezuela (mainland) [km]')
    ax.set_ylabel('Depth [m]')

    ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)

    ax.plot(mp_distance, cross_section_bathymetry_SS, color='dimgrey', lw=0.5)
    ax.fill_between(mp_distance, cross_section_bathymetry_SS, -4500, color='k', alpha=0.8)
    ax.set_ylim(-3500, 200)
    ax.set_xlim(0,vline8)

    ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.text(vline1/2, -100, 'SS_1D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline1 + vline2)/2, -100, 'SS_2D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -100, 'SS_3D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -100, 'SS_4D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -100, 'SS_5D1', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline1 + vline2)/2, -350, 'SS_2D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -350, 'SS_3D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -350, 'SS_4D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -350, 'SS_5D2', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline2 + vline3)/2, -600, 'SS_3D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -600, 'SS_4D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -600, 'SS_5D3', fontsize=8, ha='center', va='center', color='w')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    output_dir = 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/segments_{section}_{part_month}.png'
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    # find the largest distance
    max_distance = all_crossings.distance_from_start.max().item()
    print(f"The largest distance is: {max_distance / 1000} km")

    # Calculate mean lon, lat, and depth for each segment and save as npy
    segments = {
        'SS_1D1': SS_1D1,
        'SS_2D1': SS_2D1,
        'SS_3D1': SS_3D1,
        'SS_4D1': SS_4D1,
        'SS_5D1': SS_5D1,
        'SS_2D2': SS_2D2,
        'SS_3D2': SS_3D2,
        'SS_4D2': SS_4D2,
        'SS_5D2': SS_5D2,
        'SS_3D3': SS_3D3,
        'SS_4D3': SS_4D3,
        'SS_5D3': SS_5D3
    }

    mean_values = {}
    for name, segment in segments.items():
        mean_lon = (segment.crossing_lon.min().item() + segment.crossing_lon.max().item()) / 2
        min_lon = segment.crossing_lon.min().item()
        max_lon = segment.crossing_lon.max().item()
        mean_lat = (segment.crossing_lat.min().item() + segment.crossing_lat.max().item()) / 2
        min_lat = segment.crossing_lat.min().item()
        max_lat = segment.crossing_lat.max().item()
        mean_depth = segment.crossing_depth.mean().item()
        mean_values[name] = {'mean_lon': mean_lon, 'mean_lat': mean_lat, 'mean_depth': mean_depth, 'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/locations_{section}_{part_month}.npy'
    np.save(output_filepath, mean_values)

    # save segments 
    unique_traj = np.unique(all_crossings['trajectory'].values)
    segment_names = {}
    for traj in unique_traj:
        segments = []
        segments_time = []
        if traj in SS_1D1.trajectory.values:
            segments.append('SS_1D1')
            mask = SS_1D1.trajectory == traj
            segments_time.append(SS_1D1['time'].values[mask].tolist())
        if traj in SS_2D1.trajectory.values:
            segments.append('SS_2D1')
            mask = SS_2D1.trajectory == traj
            segments_time.append(SS_2D1['time'].values[mask].tolist())
        if traj in SS_3D1.trajectory.values:
            segments.append('SS_3D1')
            mask = SS_3D1.trajectory == traj
            segments_time.append(SS_3D1['time'].values[mask].tolist())
        if traj in SS_4D1.trajectory.values:
            segments.append('SS_4D1')
            mask = SS_4D1.trajectory == traj
            segments_time.append(SS_4D1['time'].values[mask].tolist())
        if traj in SS_5D1.trajectory.values:
            segments.append('SS_5D1')
            mask = SS_5D1.trajectory == traj
            segments_time.append(SS_5D1['time'].values[mask].tolist())
        if traj in SS_2D2.trajectory.values:
            segments.append('SS_2D2')
            mask = SS_2D2.trajectory == traj
            segments_time.append(SS_2D2['time'].values[mask].tolist())
        if traj in SS_3D2.trajectory.values:
            segments.append('SS_3D2')
            mask = SS_3D2.trajectory == traj
            segments_time.append(SS_3D2['time'].values[mask].tolist())
        if traj in SS_4D2.trajectory.values:
            segments.append('SS_4D2')
            mask = SS_4D2.trajectory == traj
            segments_time.append(SS_4D2['time'].values[mask].tolist())
        if traj in SS_5D2.trajectory.values:
            segments.append('SS_5D2')
            mask = SS_5D2.trajectory == traj
            segments_time.append(SS_5D2['time'].values[mask].tolist())
        if traj in SS_3D3.trajectory.values:
            segments.append('SS_3D3')
            mask = SS_3D3.trajectory == traj
            segments_time.append(SS_3D3['time'].values[mask].tolist())
        if traj in SS_4D3.trajectory.values:
            segments.append('SS_4D3')
            mask = SS_4D3.trajectory == traj
            segments_time.append(SS_4D3['time'].values[mask].tolist())
        if traj in SS_5D3.trajectory.values:
            segments.append('SS_5D3')
            mask = SS_5D3.trajectory == traj
            segments_time.append(SS_5D3['time'].values[mask].tolist())
        if segments:
            segment_names[str(traj)] = {'segments': segments, 'times': segments_time}
        else:
            print(f"No segment found for trajectory {traj}")


    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/segment_names_{section}_{part_month}.json'
    with open(output_filepath, 'w') as f:
        json.dump(segment_names, f)

    print(f'DONE SEGMENTATION FOR {section} {part_month}')
    print("-    -")

    # ==================== SECTION: NS ====================
    section = 'NS'
    print(f"Section: {section}")
    sections = [section]

    # LOAD CROSSINGS
    all_crossings = load_all_crossings(section)
    all_crossings = all_crossings.set_coords('trajectory')
    print(all_crossings)

    vline1, vline2, vline3, vline4, vline8 = 13, 29.5, 66, 102, 140
    depth1 = -162
    depth2 = -458.5

    NS_1D1 = make_segments(all_crossings, 0, vline1 * 1000,     depth1, 0)
    NS_2D1 = make_segments(all_crossings, vline1 * 1000, vline2 * 1000, depth1, 0)
    NS_3D1 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth1, 0)
    NS_4D1 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth1, 0)
    NS_5D1 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, depth1, 0)

    NS_2D2 = make_segments(all_crossings, 0, vline2 * 1000, depth2, depth1)
    NS_3D2 = make_segments(all_crossings, vline2 * 1000, vline3 * 1000, depth2, depth1)
    NS_4D2 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, depth2, depth1)
    NS_5D2 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, depth2, depth1)


    NS_3D3 = make_segments(all_crossings, 0, vline3 * 1000, -4500, depth2)
    NS_4D3 = make_segments(all_crossings, vline3 * 1000, vline4 * 1000, -4500, depth2)
    NS_5D3 = make_segments(all_crossings, vline4 * 1000, vline8 * 1000, -4500, depth2)

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5))

    mp_distance = np.linspace(0, geodesic(section_plotting["NS"][0], section_plotting["NS"][1]).kilometers, n_points)
    ax.scatter(all_crossings.distance_from_start / 1000, all_crossings.crossing_depth, s=1, c='yellow', label='All Crossings')

    ax.scatter(NS_1D1.distance_from_start / 1000, NS_1D1.crossing_depth, s=1, c=color1, label='NS_1D1')
    ax.scatter(NS_2D1.distance_from_start / 1000, NS_2D1.crossing_depth, s=1, c=color2, label='NS_2D1')
    ax.scatter(NS_3D1.distance_from_start / 1000, NS_3D1.crossing_depth, s=1, c=color1, label='NS_3D1')
    ax.scatter(NS_4D1.distance_from_start / 1000, NS_4D1.crossing_depth, s=1, c=color2, label='NS_4D1')
    ax.scatter(NS_5D1.distance_from_start / 1000, NS_5D1.crossing_depth, s=1, c=color1, label='NS_5D1')

    ax.scatter(NS_2D2.distance_from_start / 1000, NS_2D2.crossing_depth, s=1, c=color3, label='NS_2D2')
    ax.scatter(NS_3D2.distance_from_start / 1000, NS_3D2.crossing_depth, s=1, c=color4, label='NS_3D2')
    ax.scatter(NS_4D2.distance_from_start / 1000, NS_4D2.crossing_depth, s=1, c=color3, label='NS_4D2')
    ax.scatter(NS_5D2.distance_from_start / 1000, NS_5D2.crossing_depth, s=1, c=color4, label='NS_5D2')

    ax.scatter(NS_3D3.distance_from_start / 1000, NS_3D3.crossing_depth, s=1, c=color5, label='NS_3D3')
    ax.scatter(NS_4D3.distance_from_start / 1000, NS_4D3.crossing_depth, s=1, c=color6, label='NS_4D3')
    ax.scatter(NS_5D3.distance_from_start / 1000, NS_5D3.crossing_depth, s=1, c=color5, label='NS_5D3')

    ax.set_xlabel('Distance from Venezuela (mainland) [km]')
    ax.set_ylabel('Depth [m]')

    ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
    ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)

    ax.plot(mp_distance, cross_section_bathymetry_NS, color='dimgrey', lw=0.5)
    ax.fill_between(mp_distance, cross_section_bathymetry_NS, -4500, color='k', alpha=0.8)

    ax.set_ylim(-2500, 200)
    ax.set_xlim(0,vline8)
    ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)
    ax.text(vline1/2, -100, 'NS_1D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline1 + vline2)/2, -100, 'NS_2D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -100, 'NS_3D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -100, 'NS_4D1', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -100, 'NS_5D1', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline1 + vline2)/2, -350, 'NS_2D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline2 + vline3)/2, -350, 'NS_3D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -350, 'NS_4D2', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -350, 'NS_5D2', fontsize=8, ha='center', va='center', color='w')

    ax.text((vline2 + vline3)/2, -600, 'NS_3D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline3 + vline4)/2, -600, 'NS_4D3', fontsize=8, ha='center', va='center', color='w')
    ax.text((vline4 + vline8)/2, -600, 'NS_5D3', fontsize=8, ha='center', va='center', color='w')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    output_dir = 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/segments_{section}_{part_month}.png'
    plt.savefig(output_filepath, dpi=300)
    plt.close()

    max_distance = all_crossings.distance_from_start.max().item()
    print(f"The largest distance is: {max_distance / 1000} km")

    # Calculate mean lon, lat, and depth for each segment and save as npy
    segments = {
        'NS_1D1': NS_1D1,
        'NS_2D1': NS_2D1,
        'NS_3D1': NS_3D1,
        'NS_4D1': NS_4D1,
        'NS_5D1': NS_5D1,
        'NS_2D2': NS_2D2,
        'NS_3D2': NS_3D2,
        'NS_4D2': NS_4D2,
        'NS_5D2': NS_5D2,
        'NS_3D3': NS_3D3,
        'NS_4D3': NS_4D3,
        'NS_5D3': NS_5D3
    }

    mean_values = {}
    for name, segment in segments.items():
        mean_lon = (segment.crossing_lon.min().item() + segment.crossing_lon.max().item()) / 2
        min_lon = segment.crossing_lon.min().item()
        max_lon = segment.crossing_lon.max().item()
        mean_lat = (segment.crossing_lat.min().item() + segment.crossing_lat.max().item()) / 2
        min_lat = segment.crossing_lat.min().item()
        max_lat = segment.crossing_lat.max().item()
        mean_depth = segment.crossing_depth.mean().item()
        mean_values[name] = {'mean_lon': mean_lon, 'mean_lat': mean_lat, 'mean_depth': mean_depth, 'min_lon': min_lon, 'max_lon': max_lon, 'min_lat': min_lat, 'max_lat': max_lat}

    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/locations_{section}_{part_month}.npy'
    np.save(output_filepath, mean_values)

    # save segments 
    unique_traj = np.unique(all_crossings['trajectory'].values)
    segment_names = {}
    for traj in unique_traj:
        segments = []
        segments_time = []
        if traj in NS_1D1.trajectory.values:
            segments.append('NS_1D1')
            mask = NS_1D1.trajectory == traj
            segments_time.append(NS_1D1['time'].values[mask].tolist())
        if traj in NS_2D1.trajectory.values:
            segments.append('NS_2D1')
            mask = NS_2D1.trajectory == traj
            segments_time.append(NS_2D1['time'].values[mask].tolist())
        if traj in NS_3D1.trajectory.values:
            segments.append('NS_3D1')
            mask = NS_3D1.trajectory == traj
            segments_time.append(NS_3D1['time'].values[mask].tolist())
        if traj in NS_4D1.trajectory.values:
            segments.append('NS_4D1')
            mask = NS_4D1.trajectory == traj
            segments_time.append(NS_4D1['time'].values[mask].tolist())
        if traj in NS_5D1.trajectory.values:
            segments.append('NS_5D1')
            mask = NS_5D1.trajectory == traj
            segments_time.append(NS_5D1['time'].values[mask].tolist())
        if traj in NS_2D2.trajectory.values:
            segments.append('NS_2D2')
            mask = NS_2D2.trajectory == traj
            segments_time.append(NS_2D2['time'].values[mask].tolist())
        if traj in NS_3D2.trajectory.values:
            segments.append('NS_3D2')
            mask = NS_3D2.trajectory == traj
            segments_time.append(NS_3D2['time'].values[mask].tolist())
        if traj in NS_4D2.trajectory.values:
            segments.append('NS_4D2')
            mask = NS_4D2.trajectory == traj
            segments_time.append(NS_4D2['time'].values[mask].tolist())
        if traj in NS_5D2.trajectory.values:
            segments.append('NS_5D2')
            mask = NS_5D2.trajectory == traj
            segments_time.append(NS_5D2['time'].values[mask].tolist())
        if traj in NS_3D3.trajectory.values:
            segments.append('NS_3D3')
            mask = NS_3D3.trajectory == traj
            segments_time.append(NS_3D3['time'].values[mask].tolist())
        if traj in NS_4D3.trajectory.values:
            segments.append('NS_4D3')
            mask = NS_4D3.trajectory == traj
            segments_time.append(NS_4D3['time'].values[mask].tolist())
        if traj in NS_5D3.trajectory.values:
            segments.append('NS_5D3')
            mask = NS_5D3.trajectory == traj
            segments_time.append(NS_5D3['time'].values[mask].tolist())
        if segments:
            segment_names[str(traj)] = {'segments': segments, 'times': segments_time}
        else:
            print(f"No segment found for trajectory {traj}")


    output_dir = '../parcels_analysis/segmentation'
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = f'{output_dir}/final/segment_names_{section}_{part_month}.json'
    with open(output_filepath, 'w') as f:
        json.dump(segment_names, f)

    print(f'DONE SEGMENTATION FOR {section} {part_month}')
    print("-    -")
    print("Moving to the next month!")
    print("-    -")
    print("-    -")

print("DONE SEGMENTATION FOR ALL SECTIONS FOR THE SELECTED MONTHS")