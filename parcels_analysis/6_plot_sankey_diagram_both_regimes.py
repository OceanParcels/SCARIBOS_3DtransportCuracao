'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot Sankey diagram for months associated with EDDY-flow regime or NW-flow regime. We can decide on the number of top strongest (associated with the largest
volume transport) we want to plot (parameter top_n_flows).

Note: All simulation periods consit of 3 months worth of particle seeding. However, some of these periods are half EDDY domiated and half NW dominated.
This is why we divide them suing the paritcle ID ranges. For each period we calculated until which particle ID it was associated with paricle seeding
during certain month. 

Author: V Bertoncelj
kernel: parcels_dev_local
'''


#%%
# Import libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd 
from collections import defaultdict
import networkx as nx
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
from matplotlib.lines import Line2D
import os  

# parameters
top_n_flows = 80

# Define months for each regime
REGIME_MONTHS = {
    "EDDY": ['Y2020M04','Y2020M07', 'Y2020M10', 'Y2021M04','Y2022M07','Y2022M10', 'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10', 'Y2024M01', 'Y2024M01'],
    "NW": ['Y2020M04','Y2020M07','Y2020M10', 'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10','Y2022M01','Y2022M04','Y2022M07','Y2022M10', 'Y2023M01', 'Y2023M07', 'Y2023M10']
}

# Geographical shifts for different depth layers
X_SHIFT_MID = 0
Y_SHIFT_MID = -1.64
X_SHIFT_BOTTOM = 0
Y_SHIFT_BOTTOM = -3.3

# Cross-section definitions
SECTIONS = {
    "NS": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SS": [(11.41, -68.80), (12.40, -67.81)],
}

# Function to extract depth from segment name
def get_depth(segment):
    match = re.search(r'_\d+D(\d+)', segment)
    if match:
        return int(match.group(1))
    return 0

# Define the particle ID ranges for filtering by month and regime
PARTICLE_RANGES = {
    "Y2020M04": {"NW": (1, 578790), "EDDY": (578791, 1736370)}, # OK
    "Y2020M07": {"NW": (1, 598083), "EDDY": (598084, 1774956)}, # OK
    "Y2020M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)}, # OK
    "Y2021M04": {"NW": (1, 578790), "EDDY": (578791, 1736370)}, # OK
    "Y2022M07": {"NW": (1, 1196166), "EDDY": (1196167, 1774956)}, # OK
    "Y2022M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)}, # OK
    "Y2023M01": {"NW": (1, 1138287), "EDDY": (1138288, 1736370)}, # OK
    "Y2023M07": {"EDDY": (1, 1196166), "NW": (1196167, 1774956)}, # OK
    "Y2023M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)}
}

def load_volume_transport_data(months_list, regime):
    """Load volume transport data from CSV files for specified months and filter by regime"""
    vt_data = {}
    
    for month in months_list:
        # Load volume transport data from CSV file
        vt_file_path = f'../parcels_run/VOLUME_TRANSPORT/SAMPLEVEL_speeds_vt_{month}.csv'
        
        # Check if file exists
        if not os.path.exists(vt_file_path):
            print(f"Warning: Volume transport file not found for {month}: {vt_file_path}")
            continue
        
        try:
            df = pd.read_csv(vt_file_path)
            
            # Filter by particle ID range if the month is in our defined ranges
            if month in PARTICLE_RANGES and regime in PARTICLE_RANGES[month]:
                min_id, max_id = PARTICLE_RANGES[month][regime]
                df = df[(df['PARTICLE_ID'] >= min_id) & (df['PARTICLE_ID'] <= max_id)]
                print(f"Filtered {month} data for {regime} regime: using particles {min_id}-{max_id}")
            
            # Create a dictionary mapping particle IDs to their volume transport values
            month_vt_data = dict(zip(df['PARTICLE_ID'], df['VT']))
            vt_data[month] = month_vt_data
            print(f"Loaded volume transport data for {month}: {len(month_vt_data)} particles")
        except Exception as e:
            print(f"Error loading volume transport data for {month}: {e}")
    
    return vt_data


def load_shapefiles():
    """Load and return all required shapefiles"""
    curacao = gpd.read_file("data/cuw_adm0/CUW_adm0.shp")
    venezuela = gpd.read_file("data/ven_adm/ven_admbnda_adm0_ine_20210223.shp")
    bes_islands = gpd.read_file("data/bes_adm0/BES_adm0.shp")
    aruba = gpd.read_file("data/abw_adm0/abw_admbnda_adm0_2020.shp")
    
    return curacao, venezuela, bes_islands, aruba

def load_bathymetry():
    """Load and return bathymetry data"""
    config = 'SCARIBOS_V8'
    path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
    grid_orig = xr.open_dataset(path + 'croco_grd.nc')
    
    # Slice the grid for different layers
    grid = grid_orig.isel(xi_rho=slice(17, 270), eta_rho=slice(142, 302))
    grid_mid = grid_orig.isel(xi_rho=slice(17, 270), eta_rho=slice(142, 302))
    grid_bottom = grid_orig.isel(xi_rho=slice(17, 270), eta_rho=slice(142, 302))
    
    # Extract bathymetry values and coordinates
    bathymetry = grid.h.values
    bathy_lons = grid.lon_rho.values
    bathy_lats = grid.lat_rho.values
    
    bathymetry_mid = grid_mid.h.values
    bathy_lons_mid = grid_mid.lon_rho.values
    bathy_lats_mid = grid_mid.lat_rho.values
    
    bathymetry_bottom = grid_bottom.h.values
    bathy_lons_bottom = grid_bottom.lon_rho.values
    bathy_lats_bottom = grid_bottom.lat_rho.values
    
    return (
        bathymetry, bathy_lons, bathy_lats,
        bathymetry_mid, bathy_lons_mid, bathy_lats_mid,
        bathymetry_bottom, bathy_lons_bottom, bathy_lats_bottom
    )

def load_combined_segments(months_list):
    """Load segment data from multiple months"""
    combined_transitions = {}
    particle_ids_by_path = {}  # New dictionary to store particle IDs for each path
    
    # Load locations only from the first month
    first_month = months_list[0]
    section_locations_KC = np.load(f'../parcels_analysis/segmentation/final/locations_KC_{first_month}.npy', allow_pickle=True).item()
    section_locations_WP = np.load(f'../parcels_analysis/segmentation/final/locations_WP_{first_month}.npy', allow_pickle=True).item()
    section_locations_MP = np.load(f'../parcels_analysis/segmentation/final/locations_MP_{first_month}.npy', allow_pickle=True).item()
    section_locations_SS = np.load(f'../parcels_analysis/segmentation/final/locations_SS_{first_month}.npy', allow_pickle=True).item()
    section_locations_NS = np.load(f'../parcels_analysis/segmentation/final/locations_NS_{first_month}.npy', allow_pickle=True).item()
    
    # Combine transitions from all months and track particle IDs
    for month in months_list:
        # Load segment transitions
        with open(f'../parcels_analysis/segmentation/final/ordered_segments_{month}.json') as f:
            month_transitions = json.load(f)
        
        # Load particle IDs for each path
        try:
            with open(f'../parcels_analysis/segmentation/final/particle_ids_{month}.json') as f:
                month_particle_ids = json.load(f)
                for path_id, path_data in month_particle_ids.items():
                    # Store with month prefix to avoid collision between months
                    particle_ids_by_path[f"{month}_{path_id}"] = path_data
        except FileNotFoundError:
            print(f"Warning: particle_ids_{month}.json not found. Using path IDs as backup.")
            # If no particle IDs file exists, use path IDs as placeholders
            for path_id in month_transitions.keys():
                particle_ids_by_path[f"{month}_{path_id}"] = {"particle_id": int(path_id)}
        
        # Add month prefix to path IDs to avoid collision between months
        prefixed_transitions = {f"{month}_{path_id}": path for path_id, path in month_transitions.items()}
        combined_transitions.update(prefixed_transitions)
    
    return (
        combined_transitions, 
        particle_ids_by_path,
        section_locations_KC, 
        section_locations_WP, 
        section_locations_MP, 
        section_locations_SS, 
        section_locations_NS
    )

def unify_section_locations(section_locations):
    """Unify section locations to ensure consistent coordinates for base names"""
    base_location = {}  # Dictionary to store the first occurrence of each base name
    for section, data in section_locations.items():
        base_name = section[:-1]  # Remove the last character to get the base name
        if base_name not in base_location:
            base_location[base_name] = (data['mean_lon'], data['mean_lat'])
        data['mean_lon'], data['mean_lat'] = base_location[base_name]
    return section_locations

def parse_transitions(transitions_data):
    """Parse transitions data and organize segments by type and depth"""
    # Extract all unique segments
    all_segments = set()
    for _, path in transitions_data.items():
        segments = [seg.strip() for seg in path.split(',')]
        all_segments.update(segments)
    
    # Group segments by type (KC, MP, WP, SS, NS) and depth
    segment_groups = defaultdict(list)
    
    for segment in all_segments:
        segment_type = None
        if re.search(r'KC_\d+D\d+', segment):
            segment_type = 'KC'
        elif re.search(r'MP_\d+D\d+', segment):
            segment_type = 'MP'
        elif re.search(r'WP_\d+D\d+', segment):
            segment_type = 'WP'
        elif re.search(r'SS_\d+D\d+', segment):
            segment_type = 'SS'
        elif re.search(r'NS_\d+D\d+', segment):
            segment_type = 'NS'
        
        if segment_type:
            depth = get_depth(segment)
            segment_groups[(segment_type, depth)].append(segment)
    
    # Sort segments within each group
    for key in segment_groups:
        segment_groups[key].sort()
    
    # Extract sorted groups
    kc_segments = []
    mp_segments = []
    wp_segments = []
    ss_segments = []
    ns_segments = []
    
    # Get all depths
    all_depths = sorted(set(depth for _, depth in segment_groups.keys()))
    
    # Organize segments by type and depth
    for depth in all_depths:
        kc_segments.extend(segment_groups.get(('KC', depth), []))
        mp_segments.extend(segment_groups.get(('MP', depth), []))
        wp_segments.extend(segment_groups.get(('WP', depth), []))
        ss_segments.extend(segment_groups.get(('SS', depth), []))
        ns_segments.extend(segment_groups.get(('NS', depth), []))
    
    # Create a mapping from segment to index
    segment_to_idx = {}
    for i, seg in enumerate(kc_segments + mp_segments + wp_segments + ss_segments + ns_segments):
        segment_to_idx[seg] = i
    
    return kc_segments, mp_segments, wp_segments, ss_segments, ns_segments, segment_to_idx, segment_groups


def filter_top_flows(flows, top_n=10):
    """Return the top N flows by weight"""
    sorted_flows = sorted(flows.items(), key=lambda x: x[1], reverse=True)
    top_flows = dict(sorted_flows[:top_n])
    return top_flows


def calculate_flows_with_vt(transitions_data, particle_ids_by_path, vt_data, segment_to_idx, regime):
    """Calculate flow weights between segments using volume transport, filtered by regime"""
    flows = defaultdict(float)  # Changed to float for VT values
    
    for path_id, path in transitions_data.items():
        # Extract month from the path_id (in format "YYYYMXX_path_id")
        match = re.match(r'(Y\d+M\d+)_(.+)', path_id)
        if not match:
            print(f"Warning: Could not extract month from path_id: {path_id}")
            continue
            
        month, original_path_id = match.groups()
        
        # Get particle ID for this path
        particle_data = particle_ids_by_path.get(path_id)
        if not particle_data:
            print(f"Warning: No particle data found for path: {path_id}")
            continue
            
        particle_id = particle_data.get("particle_id")
        if particle_id is None:
            print(f"Warning: No particle ID in data for path: {path_id}")
            continue
        
        # Check if particle is in the correct regime range for the month
        if month in PARTICLE_RANGES and regime in PARTICLE_RANGES[month]:
            min_id, max_id = PARTICLE_RANGES[month][regime]
            if not (min_id <= float(particle_id) <= max_id):
                continue  # Skip particles outside the desired range for this regime
        
        # Get VT value for this particle ID
        month_vt = vt_data.get(month, {})
        if not month_vt:
            print(f"Warning: No VT data for month: {month}")
            continue
            
        vt_value = month_vt.get(float(particle_id))
        if vt_value is None:
            print(f"Warning: No VT value for particle ID {particle_id} in month {month}")
            continue
        
        # Parse segments in the path
        segments = [seg.strip() for seg in path.split(',')]
        
        # Count transitions between consecutive segments with VT weights
        for i in range(len(segments) - 1):
            from_seg = segments[i]
            to_seg = segments[i + 1]
            flows[(from_seg, to_seg)] += vt_value  # Add VT value instead of incrementing by 1
    
    return flows

def calculate_node_weights_with_vt(transitions_data, particle_ids_by_path, vt_data, regime):
    """Calculate weights for each node based on volume transport, filtered by regime"""
    node_weights = defaultdict(float)  # Changed to float for VT values
    
    for path_id, path in transitions_data.items():
        # Extract month from the path_id
        match = re.match(r'(Y\d+M\d+)_(.+)', path_id)
        if not match:
            continue
            
        month, original_path_id = match.groups()
        
        # Get particle ID for this path
        particle_data = particle_ids_by_path.get(path_id)
        if not particle_data or "particle_id" not in particle_data:
            continue
            
        particle_id = particle_data["particle_id"]
        
        # Check if particle is in the correct regime range for the month
        if month in PARTICLE_RANGES and regime in PARTICLE_RANGES[month]:
            min_id, max_id = PARTICLE_RANGES[month][regime]
            if not (min_id <= float(particle_id) <= max_id):
                continue  # Skip particles outside the desired range for this regime
        
        # Get VT value for this particle ID
        month_vt = vt_data.get(month, {})
        if not month_vt:
            continue
            
        vt_value = month_vt.get(float(particle_id))
        if vt_value is None:
            continue
        
        # Parse segments and add VT value to each segment's weight
        segments = [seg.strip() for seg in path.split(',')]
        for seg in segments:
            node_weights[seg] += vt_value
    
    return node_weights


def plot_single_regime(ax, regime, kc_segments, mp_segments, wp_segments, ss_segments, ns_segments, 
                      flows, segment_groups, node_weights, all_locations, panel_label):
    """
    Plot a single regime's Sankey diagram on a given axis
    """
    # Load required data
    curacao, venezuela, bes_islands, aruba = load_shapefiles()
    bathy_data = load_bathymetry()
    bathymetry, bathy_lons, bathy_lats = bathy_data[0:3]
    bathymetry_mid, bathy_lons_mid, bathy_lats_mid = bathy_data[3:6]
    bathymetry_bottom, bathy_lons_bottom, bathy_lats_bottom = bathy_data[6:9]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with their categories and depths
    for seg in kc_segments + mp_segments + wp_segments + ss_segments + ns_segments:
        segment_type = None
        if re.search(r'KC_\d+D\d+', seg):
            segment_type = 'KC'
        elif re.search(r'MP_\d+D\d+', seg):
            segment_type = 'MP'
        elif re.search(r'WP_\d+D\d+', seg):
            segment_type = 'WP'
        elif re.search(r'SS_\d+D\d+', seg):
            segment_type = 'SS'
        elif re.search(r'NS_\d+D\d+', seg):
            segment_type = 'NS'
        
        depth = get_depth(seg)
        G.add_node(seg, category=segment_type, depth=depth, weight=node_weights.get(seg, 0))
    
    # Add edges with weights
    for (source, target), weight in flows.items():
        G.add_edge(source, target, weight=weight)
    
    # Plot bathymetry contours for different layers
    ax.contourf(bathy_lons_bottom + X_SHIFT_BOTTOM, bathy_lats_bottom + Y_SHIFT_BOTTOM, 
                bathymetry_bottom, levels=[0, 162, 458.5, 5000], 
                colors=['w','w', '#f7f7f7'], zorder=0)
    
    ax.contourf(bathy_lons_mid + X_SHIFT_MID, bathy_lats_mid + Y_SHIFT_MID, 
                bathymetry_mid, levels=[0, 162, 458.5, 5000], 
                colors=['w', '#f7f7f7', '#f7f7f7'], zorder=0)
    
    ax.contourf(bathy_lons, bathy_lats, bathymetry, 
                levels=[0, 162, 458.5, 5000], 
                colors=['#f7f7f7', '#f7f7f7', '#f7f7f7'], zorder=0)

    ax.contourf(bathy_lons_mid + X_SHIFT_MID, bathy_lats_mid + Y_SHIFT_MID, 
                bathymetry_mid, levels=[0, 162], colors=['w'], zorder=2)
    
    ax.contourf(bathy_lons_bottom + X_SHIFT_BOTTOM, bathy_lats_bottom + Y_SHIFT_BOTTOM, 
                bathymetry_bottom, levels=[0, 458.5], colors=['w'], zorder=2)

    # Draw section lines for all depth layers
    shapefile_color = 'grey'
    for name, coords in SECTIONS.items():
        # Original layer
        x_coords = [coords[0][1], coords[1][1]]
        y_coords = [coords[0][0], coords[1][0]]
        ax.plot(x_coords, y_coords, color=shapefile_color, lw=2, label=name, zorder=1)

        # Mid-range depth
        x_coords_mid = [coord + X_SHIFT_MID for coord in x_coords]
        y_coords_mid = [coord + Y_SHIFT_MID for coord in y_coords]
        ax.plot(x_coords_mid, y_coords_mid, color=shapefile_color, lw=2, zorder=1)

        # Bottom depth
        x_coords_bottom = [coord + X_SHIFT_BOTTOM for coord in x_coords]
        y_coords_bottom = [coord + Y_SHIFT_BOTTOM for coord in y_coords]
        ax.plot(x_coords_bottom, y_coords_bottom, color=shapefile_color, lw=2, zorder=1)

    # Plot land masses for different depth layers
    # Top layer
    shapefile_color = 'saddlebrown'
    curacao.plot(ax=ax, color=shapefile_color, alpha=0.4, edgecolor=shapefile_color, zorder=5)
    bes_islands.plot(ax=ax, color=shapefile_color, alpha = 0.4, edgecolor=shapefile_color, zorder=5)
    aruba.plot(ax=ax, color=shapefile_color, alpha = 0.4, edgecolor=shapefile_color, zorder=5)

    # Define bounding box for Venezuela
    xmin, xmax = -72, -68.3
    ymin, ymax = 11.4, 12.5
    bounding_box = box(xmin, ymin, xmax, ymax)

    # Clip Venezuela shapefile and plot for each layer
    venezuela_clipped = gpd.clip(venezuela, bounding_box)
    venezuela_clipped.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=10)

    # Mid layer
    vene_clipped_shifted = venezuela_clipped.copy()
    vene_clipped_shifted['geometry'] = vene_clipped_shifted.translate(xoff=X_SHIFT_MID, yoff=Y_SHIFT_MID)
    vene_clipped_shifted.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=10)

    curacao_shifted = curacao.copy()
    curacao_shifted['geometry'] = curacao_shifted.translate(xoff=X_SHIFT_MID, yoff=Y_SHIFT_MID)
    curacao_shifted.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)

    bes_islands_shifted = bes_islands.copy()
    bes_islands_shifted['geometry'] = bes_islands_shifted.translate(xoff=X_SHIFT_MID, yoff=Y_SHIFT_MID)
    bes_islands_shifted.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)

    aruba_shifted = aruba.copy()
    aruba_shifted['geometry'] = aruba_shifted.translate(xoff=X_SHIFT_MID, yoff=Y_SHIFT_MID)
    aruba_shifted.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)

    # Bottom layer
    vene_clipped_shifted2 = venezuela_clipped.copy()
    vene_clipped_shifted2['geometry'] = vene_clipped_shifted2.translate(xoff=X_SHIFT_BOTTOM, yoff=Y_SHIFT_BOTTOM)
    vene_clipped_shifted2.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=4)

    curacao_shifted2 = curacao.copy()
    curacao_shifted2['geometry'] = curacao_shifted2.translate(xoff=X_SHIFT_BOTTOM, yoff=Y_SHIFT_BOTTOM)
    curacao_shifted2.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)
    
    bes_islands_shifted2 = bes_islands.copy()
    bes_islands_shifted2['geometry'] = bes_islands_shifted2.translate(xoff=X_SHIFT_BOTTOM, yoff=Y_SHIFT_BOTTOM)
    bes_islands_shifted2.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)
    
    aruba_shifted2 = aruba.copy()
    aruba_shifted2['geometry'] = aruba_shifted2.translate(xoff=X_SHIFT_BOTTOM, yoff=Y_SHIFT_BOTTOM)
    aruba_shifted2.plot(ax=ax, color=shapefile_color, edgecolor=shapefile_color, alpha=0.4, zorder=6)

    # Position nodes based on their locations and depths
    pos = {}
    for seg in G.nodes():
        if seg in all_locations:
            x = all_locations[seg].get('mean_lon')
            y = all_locations[seg].get('mean_lat')
            depth = get_depth(seg)
            
            if depth == 2:
                y = y + Y_SHIFT_MID
                x = x + X_SHIFT_MID
            elif depth == 3:
                y = y + Y_SHIFT_BOTTOM
                x = x + X_SHIFT_BOTTOM
        else:
            print(f"Location not found for segment: {seg}")
            continue
            
        pos[seg] = (x, y)

    # Draw segment boundaries
    for seg in G.nodes():
        if seg in all_locations:
            # Extract min and max coordinates
            min_lon = all_locations[seg].get('min_lon')
            max_lon = all_locations[seg].get('max_lon')
            min_lat = all_locations[seg].get('min_lat')
            max_lat = all_locations[seg].get('max_lat')
            
            # Skip if we don't have min/max data
            if None in [min_lon, max_lon, min_lat, max_lat]:
                continue
            
            # Apply the same shift as applied to the node positions
            depth = get_depth(seg)
            if depth == 2:
                min_lon += X_SHIFT_MID
                max_lon += X_SHIFT_MID
                min_lat += Y_SHIFT_MID
                max_lat += Y_SHIFT_MID
            elif depth == 3:
                min_lon += X_SHIFT_BOTTOM
                max_lon += X_SHIFT_BOTTOM
                min_lat += Y_SHIFT_BOTTOM
                max_lat += Y_SHIFT_BOTTOM
            
            # Determine segment type
            segment_type = None
            if re.search(r'KC_\d+D\d+', seg):
                segment_type = 'KC'
            elif re.search(r'MP_\d+D\d+', seg):
                segment_type = 'MP'
            elif re.search(r'WP_\d+D\d+', seg):
                segment_type = 'WP'
            elif re.search(r'SS_\d+D\d+', seg):
                segment_type = 'SS'
            elif re.search(r'NS_\d+D\d+', seg):
                segment_type = 'NS'
            
            # Get the angle of the cross-section from the sections dictionary
            if segment_type in SECTIONS:
                lat1, lon1 = SECTIONS[segment_type][0]
                lat2, lon2 = SECTIONS[segment_type][1]
                angle = np.arctan2(lat2 - lat1, lon2 - lon1)
                
                # Calculate perpendicular angle (90 degrees = π/2 radians)
                perp_angle = angle + np.pi/2
                
                # Calculate the length of the border extending from the line
                marker_length = 0.01  # Adjust as needed
                
                # For min boundary
                min_point_lon = min_lon
                min_point_lat = min_lat
                
                # Draw the min boundary marker (perpendicular to the cross-section)
                min_marker_end1_lon = min_point_lon + marker_length * np.cos(perp_angle)
                min_marker_end1_lat = min_point_lat + marker_length * np.sin(perp_angle)
                min_marker_end2_lon = min_point_lon - marker_length * np.cos(perp_angle)
                min_marker_end2_lat = min_point_lat - marker_length * np.sin(perp_angle)
                
                ax.plot([min_marker_end1_lon, min_marker_end2_lon], 
                       [min_marker_end1_lat, min_marker_end2_lat], 
                       color='grey', linewidth=1, zorder=6)
                
                # For max boundary
                max_point_lon = max_lon
                max_point_lat = max_lat
                
                # Draw the max boundary marker (perpendicular to the cross-section)
                max_marker_end1_lon = max_point_lon + marker_length * np.cos(perp_angle)
                max_marker_end1_lat = max_point_lat + marker_length * np.sin(perp_angle)
                max_marker_end2_lon = max_point_lon - marker_length * np.cos(perp_angle)
                max_marker_end2_lat = max_point_lat - marker_length * np.sin(perp_angle)
                
                ax.plot([max_marker_end1_lon, max_marker_end2_lon], 
                       [max_marker_end1_lat, max_marker_end2_lat], 
                       color='grey', linewidth=1, zorder=6)

    # Draw edges with width proportional to weight using discrete categories
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Define thickness categories
    def get_thickness_category(weight):
        if weight < 300:
            return 1, "0-300"
        elif weight < 600:
            return 2, "300-600"
        elif weight < 1200:
            return 3, "600-1200"
        elif weight < 2400:
            return 4, "1200-2400"
        else:
            return 5, "2400+"

    # Create a dictionary to store thickness and category label for each edge
    edge_thickness = {}
    thickness_categories = set()

    for i, (u, v) in enumerate(G.edges()):
        weight = edge_weights[i]
        thickness, category = get_thickness_category(weight)
        edge_thickness[(u, v)] = thickness
        thickness_categories.add(category)

    # Use the discrete thicknesses for drawing
    scaled_edge_weights = [edge_thickness[(u, v)] for u, v in G.edges()]

    edge_colors = []
    for u, v in G.edges():
        source_category = G.nodes[u]['category']
        target_category = G.nodes[v]['category']
        source_depth = G.nodes[u]['depth']
        target_depth = G.nodes[v]['depth']
        
        # First check: depth changes (highest priority)
        if source_depth != target_depth:
            # Flow goes upward (deeper to shallower)
            if source_depth > target_depth:
                edge_colors.append('orangered')
            # Flow goes downward (shallower to deeper)
            else:
                edge_colors.append('cornflowerblue')
        # Second check: flows within same section
        elif source_category == target_category:
            edge_colors.append('grey')
        # Third check: northwestern progression
        elif (source_category == 'SS' and target_category == 'KC') or \
             (source_category == 'KC' and target_category == 'MP') or \
             (source_category == 'MP' and target_category == 'WP') or \
             (source_category == 'WP' and target_category == 'NS'):
            edge_colors.append('olivedrab')
        # Fourth check: southeastern progression
        elif (source_category == 'NS' and target_category == 'WP') or \
             (source_category == 'WP' and target_category == 'MP') or \
             (source_category == 'MP' and target_category == 'KC') or \
             (source_category == 'KC' and target_category == 'SS'):
            edge_colors.append('blueviolet')
        # Default case
        else:
            edge_colors.append('grey')

    # Draw the network edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=scaled_edge_weights,
        edge_color=edge_colors,
        alpha=0.8,
        connectionstyle='arc3,rad=0.15'
    )
    # Surface layer (A or D)
    ax.text(-69.9, 12.8, panel_label[0], fontsize=20, weight='bold', 
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8))

    # Mid-range layer (B or E)
    ax.text(-69.9, 12.8 + Y_SHIFT_MID, panel_label[1], fontsize=20, weight='bold', 
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8))

    # Deep layer (C or F)
    ax.text(-69.9, 12.8 + Y_SHIFT_BOTTOM, panel_label[2], fontsize=20, weight='bold', 
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8))
    # Set plot limits and turn off axis
    ax.set_xlim(-70, -67.75)
    ax.set_ylim(8.1, 13)
    ax.axis('off')

    return G


def create_combined_sankey_diagram():
    """
    Create a combined Sankey diagram with NW regime on the left and EDDY regime on the right
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))
    
    panel_labels = [['A', 'C', 'E'], ['B', 'D', 'F']]
    
    # Process both regimes
    for i, regime in enumerate(['NW', 'EDDY']):
        ax = ax1 if i == 0 else ax2
        
        print(f"\nProcessing {regime} regime...")
        
        # Load volume transport data for this regime
        months_list = REGIME_MONTHS[regime]
        vt_data = load_volume_transport_data(months_list, regime)
        print(f"Loaded volume transport data for {len(vt_data)} months")
        
        # Load and combine data from multiple months
        combined_transitions, particle_ids_by_path, section_locations_KC, section_locations_WP, section_locations_MP, section_locations_SS, section_locations_NS = load_combined_segments(months_list)
        
        # Unify section locations
        section_locations_KC = unify_section_locations(section_locations_KC)
        section_locations_WP = unify_section_locations(section_locations_WP)
        section_locations_MP = unify_section_locations(section_locations_MP)
        section_locations_SS = unify_section_locations(section_locations_SS)
        section_locations_NS = unify_section_locations(section_locations_NS)
        
        # Combine all locations
        all_locations = {}
        all_locations.update(section_locations_KC)
        all_locations.update(section_locations_WP)
        all_locations.update(section_locations_MP)
        all_locations.update(section_locations_SS)
        all_locations.update(section_locations_NS)
        
        # Parse transitions
        kc_segments, mp_segments, wp_segments, ss_segments, ns_segments, segment_to_idx, segment_groups = parse_transitions(combined_transitions)
        print(f"Total segments: {len(kc_segments) + len(mp_segments) + len(wp_segments) + len(ss_segments) + len(ns_segments)}")
        
        # Calculate flows and node weights using volume transport data
        flows = calculate_flows_with_vt(combined_transitions, particle_ids_by_path, vt_data, segment_to_idx, regime)
        flows = filter_top_flows(flows, top_n=top_n_flows)
        print(f"Total flows: {len(flows)}")
        node_weights = calculate_node_weights_with_vt(combined_transitions, particle_ids_by_path, vt_data, regime)
        
        # Plot the regime on its respective axis
        G = plot_single_regime(
            ax, regime, kc_segments, mp_segments, wp_segments, ss_segments, ns_segments,
            flows, segment_groups, node_weights, all_locations, panel_labels[i]
        )

    # Add depth labels only on the left subplot (NW regime)
    ax1.text(-70.6, 12.85, 'Surface', fontsize=18, weight='bold')
    ax1.text(-70.6, 12.77, '(0 to -162 m)', fontsize=18)
    ax1.text(-70.6, 11.22, 'Mid-range', fontsize=18, weight='bold')
    ax1.text(-70.6, 11.14, '(-162 to -458.5 m)', fontsize=18)
    ax1.text(-70.6, 9.55, 'Deep', fontsize=18, weight='bold')
    ax1.text(-70.6, 9.47, '(<-458.5 m)', fontsize=18)

    # Add regime titles
    ax1.text(-69.5, 13.05, f'NW-flow dominated months', fontsize=20, weight='bold', color='black')
    ax2.text(-69.5, 13.05, f'EDDY-flow dominated months', fontsize=20, weight='bold', color='black')

    # Create legends (only once for the entire figure)
    # Width legend
    width_legend_elements = []
    for thickness, category in [
        (1, "0-300"),
        (2, "300-600"),
        (3, "600-1200"),
        (4, "1200-2400"),
        (5, "2400+")
    ]:
        width_legend_elements.append(
            Line2D([0], [0], color='black', lw=thickness,
                label=f'VT: {category}')
        )

    # Create a separate legend for the widths
    width_legend = ax2.legend(handles=width_legend_elements,
                            loc='lower center',
                            bbox_to_anchor=(0.84, 0),
                            fontsize=14,
                            title='Volume transport \nin 4 years [Sv]', 
                            title_fontsize='14')

    # Color legend
    legend_elements = [
        Line2D([0], [0], color='olivedrab', lw=5, label='Northwestern flow (SS→KC→MP→WP→NS)'),
        Line2D([0], [0], color='blueviolet', lw=5, label='Southeastern flow (NS→WP→MP→KC→SS)'),
        Line2D([0], [0], color='grey', lw=5, label='Within-section flow (e.g., KC→KC)'),
        Line2D([0], [0], color='orangered', lw=5, label='Upward flow (deeper→shallower)'),
        Line2D([0], [0], color='cornflowerblue', lw=5, label='Downward flow (shallower→deeper)')
    ]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=14, title='Flow Direction', title_fontsize=14)

    plt.tight_layout()

    # Save the figure
    fig_name = f'figures/results/CH2_Fig5_sankey_combined.jpeg'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"Combined diagram saved as {fig_name}")
    plt.show()


def main():
    """Main function to execute the combined Sankey diagram creation workflow"""
    create_combined_sankey_diagram()

if __name__ == "__main__":
    main()

# %%