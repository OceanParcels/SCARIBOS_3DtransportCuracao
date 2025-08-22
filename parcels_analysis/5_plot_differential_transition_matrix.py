'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot transition connectivity matrices:
- for the entire period
- differential matrix = EDDY minus NW periods (differences betweent regimes)
Matrices are weighted with Volume transports, matching each particle ID.

Note: All simulaiton periods consit of 3 months worth of particle seeding. However, some of these periods are half EDDY domiated and half NW dominated.
This is why we divide them suing the paritcle ID ranges. For each period we calculated until which particle ID it was associated with paricle seeding
during certain month. 

Author: V Bertoncelj
'''


#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import cmocean
from matplotlib.colors import LogNorm
import json
import os
import pandas as pd

# Load all simulation periods
MONTHS_LIST = ['Y2020M04', 'Y2020M07', 'Y2020M10', 
               'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10', 
               'Y2022M01', 'Y2022M04', 'Y2022M07', 'Y2022M10',
               'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10', 'Y2024M01']

# Define periods associated with each regime
EDDY_MONTHS = ['Y2020M04','Y2020M07', 'Y2020M10', 
               'Y2021M04',
               'Y2022M07','Y2022M10', 
               'Y2023M01', 'Y2023M04', 'Y2023M07', 'Y2023M10', 'Y2024M01']
NW_MONTHS =   ['Y2020M04','Y2020M07','Y2020M10', 
               'Y2021M01', 'Y2021M04', 'Y2021M07', 'Y2021M10',
               'Y2022M01','Y2022M04','Y2022M07','Y2022M10', 
               'Y2023M01', 'Y2023M07', 'Y2023M10']

# Define particle ID ranges for specific months (in order to divide periods into EDDY and NW months)
PARTICLE_RANGES = {
    "Y2020M04": {"NW": (1, 578790), "EDDY": (578791, 1736370)}, 
    "Y2020M07": {"NW": (1, 598083), "EDDY": (598084, 1774956)}, 
    "Y2020M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)}, 
    "Y2021M04": {"NW": (1, 578790), "EDDY": (578791, 1736370)}, 
    "Y2022M07": {"NW": (1, 1196166), "EDDY": (1196167, 1774956)},
    "Y2022M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)},
    "Y2023M01": {"NW": (1, 1138287), "EDDY": (1138288, 1736370)},
    "Y2023M07": {"EDDY": (1, 1196166), "NW": (1196167, 1774956)},
    "Y2023M10": {"EDDY": (1, 1176873), "NW": (1176874, 1755663)}
}

#%%
# Load volume transport data for each month
def load_volume_transport_data(months_list, regime=None):
    """
    Load volume transport data from CSV files for specified months and filter by regime
    
    Parameters:
    months_list: List of months to load data for
    regime: Optional regime filter ('EDDY' or 'NW')
    
    Returns:
    Dictionary of volume transport values by month and particle ID
    """
    vt_data = {}
    print("\n=== LOADING VOLUME TRANSPORT DATA ===")
    print(f"Filtering by regime: {regime if regime else 'None (using all particles)'}")
    
    for month in months_list:
        # Load volume transport data from CSV file
        vt_file_path = f'../parcels_run/VOLUME_TRANSPORT/SAMPLEVEL_speeds_vt_{month}.csv'
        # Check if file exists
        if not os.path.exists(vt_file_path):
            print(f"Warning: Volume transport file not found for {month}: {vt_file_path}")
            continue
        try:
            df = pd.read_csv(vt_file_path)
            print(f"Processing {month}: Found {len(df)} particles in VT file")
            
            # Apply particle ID filtering based on regime and month
            if regime:
                # Case 1: Month is in PARTICLE_RANGES and has the specified regime
                if month in PARTICLE_RANGES and regime in PARTICLE_RANGES[month]:
                    min_id, max_id = PARTICLE_RANGES[month][regime]
                    df = df[(df['PARTICLE_ID'] >= min_id) & (df['PARTICLE_ID'] <= max_id)]
                    print(f"  Filtered {month} data for {regime} regime: using particles {min_id}-{max_id}")
                # Case 2: Month not in PARTICLE_RANGES or regime not defined for this month 
                # (use all particles for the specified regime)
                else:
                    print(f"  No specific particle range for {regime} regime in {month}, using all particles")
            else:
                # Case 3: No regime specified - use all particles
                print(f"  Using all particles for {month}")
            
            # Check VT value distribution
            vt_mean = df['VT'].mean()
            vt_max = df['VT'].max()
            vt_min = df['VT'].min()
            
            # Create a dictionary mapping particle IDs to their volume transport values
            month_vt_data = dict(zip(df['PARTICLE_ID'], df['VT']))
            vt_data[month] = month_vt_data
            print(f"  Added {len(month_vt_data)} particles for {month}, VT range: {vt_min:.6f} to {vt_max:.6f}, Mean: {vt_mean:.6f}")
        except Exception as e:
            print(f"  Error loading volume transport data for {month}: {e}")
    
    print(f"Total months with VT data: {len(vt_data)}")
    return vt_data

vt_data_all = load_volume_transport_data(MONTHS_LIST)



  #%%
# laod ordered transitions
def load_combined_segments(months_list):
    """Load segment data from multiple months"""
    # Load locations only from the first month
    print("\n=== LOADING SEGMENT DATA ===")
    first_month = months_list[0]
    print(f"Loading segment locations from reference month: {first_month}")
    
    section_locations_KC = np.load(f'../parcels_analysis/segmentation/final/locations_KC_{first_month}.npy', allow_pickle=True).item()
    section_locations_WP = np.load(f'../parcels_analysis/segmentation/final/locations_WP_{first_month}.npy', allow_pickle=True).item()
    section_locations_MP = np.load(f'../parcels_analysis/segmentation/final/locations_MP_{first_month}.npy', allow_pickle=True).item()
    section_locations_SS = np.load(f'../parcels_analysis/segmentation/final/locations_SS_{first_month}.npy', allow_pickle=True).item()
    section_locations_NS = np.load(f'../parcels_analysis/segmentation/final/locations_NS_{first_month}.npy', allow_pickle=True).item()
    
    # Load transitions for each month separately (don't combine them yet)
    month_transitions = {}
    total_particles = 0
    for month in months_list:
        try:
            with open(f'../parcels_analysis/segmentation/final/ordered_segments_{month}.json') as f:
                transitions = json.load(f)
                month_transitions[month] = transitions
                total_particles += len(transitions)
                print(f"  Loaded {len(transitions)} particle trajectories for {month}")
        except Exception as e:
            print(f"  Error loading transitions for {month}: {e}")
    
    print(f"Total particle trajectories loaded: {total_particles} across {len(month_transitions)} months")
    return month_transitions, section_locations_KC, section_locations_WP, section_locations_MP, section_locations_SS, section_locations_NS


month_transitions, *_ = load_combined_segments(MONTHS_LIST)

#%%
# create matrix and plot it
def get_depth(segment):
    """Extract depth from segment name"""
    match = re.search(r'_\d+D(\d+)', segment)
    if match:
        return int(match.group(1))
    return 0

def get_section(segment):
    """Extract section from segment name"""
    if 'NS_' in segment:
        return 'NS'
    elif 'WP_' in segment:
        return 'WP'
    elif 'MP_' in segment:
        return 'MP'
    elif 'KC_' in segment:
        return 'KC'
    elif 'SS_' in segment:
        return 'SS'
    return None


def create_differential_connectivity_matrix(transitions_data, vt_data_nw, vt_data_eddy, nw_months, eddy_months):
    """
    Create connectivity matrix visualization showing differences between NW and EDDY regimes
    
    Parameters:
    transitions_data: Dictionary of transition paths by month
    vt_data_nw: Dictionary of volume transport values for NW regime by month and particle ID
    vt_data_eddy: Dictionary of volume transport values for EDDY regime by month and particle ID
    nw_months: List of months to include for NW regime
    eddy_months: List of months to include for EDDY regime
    """
    print(f"\n=== CREATING DIFFERENTIAL NW vs EDDY CONNECTIVITY MATRIX ===")
    print(f"Processing {len(nw_months)} NW months vs {len(eddy_months)} EDDY months")
    
    # First create connectivity matrices for both regimes using the same segment ordering
    # This ensures we can do a proper comparison between the two
    
    # Step 1: Filter transitions for both regimes
    filtered_transitions_nw = {}
    filtered_transitions_eddy = {}
    
    print("Filtering NW transitions:")
    for month in nw_months:
        if month not in transitions_data or month not in vt_data_nw:
            print(f"  Skipping {month} - missing data")
            continue
        
        month_transitions = transitions_data[month]
        month_vt = vt_data_nw[month]
        filtered_month_transitions = {}
        
        # Filter particles
        for particle_id_str, path in month_transitions.items():
            try:
                particle_id = float(particle_id_str)
                if particle_id in month_vt:
                    filtered_month_transitions[particle_id_str] = path
            except (ValueError, KeyError):
                continue
        
        filtered_transitions_nw[month] = filtered_month_transitions
        print(f"  {month}: {len(filtered_month_transitions)} particles kept")
    
    print("Filtering EDDY transitions:")
    for month in eddy_months:
        if month not in transitions_data or month not in vt_data_eddy:
            print(f"  Skipping {month} - missing data")
            continue
        
        month_transitions = transitions_data[month]
        month_vt = vt_data_eddy[month]
        filtered_month_transitions = {}
        
        # Filter particles
        for particle_id_str, path in month_transitions.items():
            try:
                particle_id = float(particle_id_str)
                if particle_id in month_vt:
                    filtered_month_transitions[particle_id_str] = path
            except (ValueError, KeyError):
                continue
        
        filtered_transitions_eddy[month] = filtered_month_transitions
        print(f"  {month}: {len(filtered_month_transitions)} particles kept")
    
    # Step 2: Extract all unique segments from both filtered transitions
    all_segments = set()
    
    # Add segments from NW
    for month in nw_months:
        if month not in filtered_transitions_nw:
            continue
        for particle_id, path in filtered_transitions_nw[month].items():
            segments = [seg.strip() for seg in path.split(',')]
            all_segments.update(segments)
    
    # Add segments from EDDY
    for month in eddy_months:
        if month not in filtered_transitions_eddy:
            continue
        for particle_id, path in filtered_transitions_eddy[month].items():
            segments = [seg.strip() for seg in path.split(',')]
            all_segments.update(segments)
    
    print(f"Found {len(all_segments)} unique segments across both regimes")
    
    # Group segments by depth
    segments_by_depth = {1: [], 2: [], 3: []}
    
    for segment in all_segments:
        depth = get_depth(segment)
        if depth > 0:
            segments_by_depth.setdefault(depth, []).append(segment)
    
    # Sort segments within each depth by section and position
    section_order = {'NS': 0, 'WP': 1, 'MP': 2, 'KC': 3, 'SS': 4}
    
    for depth in segments_by_depth:
        # Sort by section, then position
        segments_by_depth[depth].sort(key=lambda x: (
            section_order.get(get_section(x), 999),
            'N' in x,  # Sort North after South
            re.search(r'_(\d+)D', x).group(1) if re.search(r'_(\d+)D', x) else '999'
        ))
    
    # Create ordered list of all segments
    ordered_segments = []
    for depth in [1, 2, 3]:  # Surface to bottom
        if depth in segments_by_depth:
            ordered_segments.extend(segments_by_depth[depth])
            print(f"  Depth {depth}: {len(segments_by_depth[depth])} segments")
    
    # Create mapping from segment to index
    segment_to_idx = {seg: i for i, seg in enumerate(ordered_segments)}
    
    # Step 3: Calculate VT-weighted flows for NW regime
    vt_flows_nw = defaultdict(float)
    vt_outgoing_total_nw = defaultdict(float)
    
    print("Computing NW VT-weighted flows:")
    for month in nw_months:
        if month not in filtered_transitions_nw or month not in vt_data_nw:
            print(f"  Skipping {month} - missing data")
            continue
            
        month_transitions = filtered_transitions_nw[month]
        month_vt = vt_data_nw[month]
        
        month_processed = 0
        
        for particle_id_str, path in month_transitions.items():
            try:
                particle_id = float(particle_id_str)
                vt_value = month_vt[particle_id]
                segments = [seg.strip() for seg in path.split(',')]
                
                for i in range(len(segments) - 1):
                    from_seg = segments[i]
                    to_seg = segments[i + 1]
                    
                    # Weight the connection by VT
                    vt_flows_nw[(from_seg, to_seg)] += vt_value
                    vt_outgoing_total_nw[from_seg] += vt_value
                
                month_processed += 1
                
            except (ValueError, KeyError):
                continue
        
        print(f"  {month}: Processed {month_processed} particles")
    
    # Step 4: Calculate VT-weighted flows for EDDY regime
    vt_flows_eddy = defaultdict(float)
    vt_outgoing_total_eddy = defaultdict(float)
    
    print("Computing EDDY VT-weighted flows:")
    for month in eddy_months:
        if month not in filtered_transitions_eddy or month not in vt_data_eddy:
            print(f"  Skipping {month} - missing data")
            continue
            
        month_transitions = filtered_transitions_eddy[month]
        month_vt = vt_data_eddy[month]
        
        month_processed = 0
        
        for particle_id_str, path in month_transitions.items():
            try:
                particle_id = float(particle_id_str)
                vt_value = month_vt[particle_id]
                segments = [seg.strip() for seg in path.split(',')]
                
                for i in range(len(segments) - 1):
                    from_seg = segments[i]
                    to_seg = segments[i + 1]
                    
                    # Weight the connection by VT
                    vt_flows_eddy[(from_seg, to_seg)] += vt_value
                    vt_outgoing_total_eddy[from_seg] += vt_value
                
                month_processed += 1
                
            except (ValueError, KeyError):
                continue
        
        print(f"  {month}: Processed {month_processed} particles")
    
    # Step 5: Create connectivity matrices for each regime
    n_segments = len(ordered_segments)
    matrix_nw = np.zeros((n_segments, n_segments))
    matrix_eddy = np.zeros((n_segments, n_segments))
    
    # Fill NW matrix with normalized VT-weighted values
    print("Building NW connectivity matrix...")
    for (from_seg, to_seg), vt_weight in vt_flows_nw.items():
        if from_seg in segment_to_idx and to_seg in segment_to_idx:
            from_idx = segment_to_idx[from_seg]
            to_idx = segment_to_idx[to_seg]
            if vt_outgoing_total_nw[from_seg] > 0:
                matrix_nw[from_idx, to_idx] = (vt_weight / vt_outgoing_total_nw[from_seg]) * 100
    
    # Fill EDDY matrix with normalized VT-weighted values
    print("Building EDDY connectivity matrix...")
    for (from_seg, to_seg), vt_weight in vt_flows_eddy.items():
        if from_seg in segment_to_idx and to_seg in segment_to_idx:
            from_idx = segment_to_idx[from_seg]
            to_idx = segment_to_idx[to_seg]
            if vt_outgoing_total_eddy[from_seg] > 0:
                matrix_eddy[from_idx, to_idx] = (vt_weight / vt_outgoing_total_eddy[from_seg]) * 100
    
    # Step 6: Calculate the difference matrix (EDDY - NW)
    diff_matrix = matrix_eddy - matrix_nw
    
    # Create a mask for cells where both matrices have zero values
    zero_mask = (matrix_nw == 0) & (matrix_eddy == 0)
    
    # Analyze matrix values
    non_zero_diffs = diff_matrix[~zero_mask]
    if len(non_zero_diffs) > 0:
        print(f"Difference matrix - Min: {non_zero_diffs.min():.4f}%, Mean: {non_zero_diffs.mean():.4f}%, Max: {non_zero_diffs.max():.4f}%")
        print(f"Matrix has {np.sum(~zero_mask)} non-zero cells ({(np.sum(~zero_mask)/(n_segments*n_segments)*100):.2f}% filled)")
    
    # Get depth and section for each segment in the order
    segment_depths = [get_depth(seg) for seg in ordered_segments]
    
    # Find depth boundaries (where depth changes)
    depth_boundaries = [0]
    for i in range(1, len(segment_depths)):
        if segment_depths[i] != segment_depths[i-1]:
            depth_boundaries.append(i)
    depth_boundaries.append(len(segment_depths))
    
    # Find section boundaries within each depth
    section_boundaries = []
    segment_sections = [get_section(seg) for seg in ordered_segments]
    for d in range(len(depth_boundaries)-1):
        start = depth_boundaries[d]
        end = depth_boundaries[d+1]
        
        for i in range(start+1, end):
            if segment_sections[i] != segment_sections[i-1]:
                section_boundaries.append(i)
    
    # Step 7: Create a signed logarithmic transformation for the difference matrix
    # This preserves the sign while applying log scaling to the magnitude
    print("Creating signed logarithmic transformation for better visualization of small differences...")
    
    # Copy the difference matrix
    diff_matrix_log = np.copy(diff_matrix)
    
    # Define a minimum threshold to avoid log(0) issues
    min_threshold = 0.01
    
    # Apply signed log transformation
    # For positive values: sign(x) * log10(1 + |x|)
    # For values near zero (|x| < min_threshold): linear scaling to avoid discontinuity
    for i in range(diff_matrix_log.shape[0]):
        for j in range(diff_matrix_log.shape[1]):
            if not zero_mask[i, j]:  # Only transform non-masked values
                value = diff_matrix_log[i, j]
                if abs(value) < min_threshold:
                    # Use linear scaling for small values
                    diff_matrix_log[i, j] = value / min_threshold * np.log10(1 + min_threshold)
                else:
                    # Use log scaling for larger values
                    diff_matrix_log[i, j] = np.sign(value) * np.log10(1 + abs(value))
    
    # Analyze the transformed matrix
    non_zero_diffs_log = diff_matrix_log[~zero_mask]
    if len(non_zero_diffs_log) > 0:
        print(f"Transformed matrix - Min: {non_zero_diffs_log.min():.4f}, Mean: {non_zero_diffs_log.mean():.4f}, Max: {non_zero_diffs_log.max():.4f}")
    
    vmax_log = np.log10(1 + 30)  # For 30% difference
    vmin_log = -vmax_log

    # Plot the transformed matrix
    plt.figure(figsize=(15, 12.5))

    # Create a diverging colormap: blue for more EDDY connectivity, red for more NW connectivity
    cmap = plt.cm.RdBu_r  # Red-Blue reversed (red = negative, blue = positive)

    # Create heatmap with the transformed matrix
    ax = sns.heatmap(
        diff_matrix_log, cmap=cmap, 
        vmin=vmin_log, vmax=vmax_log,
        mask=zero_mask,
        xticklabels=ordered_segments, yticklabels=ordered_segments,
        cbar_kws={
            'label': 'Flow percentage difference (EDDY - NW) [%, signed log scale]',
            'extend': 'both'
        }
    )
    # Set colorbar label font size to 12
    cbar = ax.collections[0].colorbar
    cbar.set_label('Flow percentage difference (EDDY - NW) [%, signed log scale]', fontsize=12)

    # Mark ALL diagonal elements (self-connectivity) with an X, even if value is zero
    for i in range(len(ordered_segments)):
        plt.plot([i+0.5], [i+0.5], 'x', color='k', markersize=7, markeredgewidth=1, zorder=5)

    # Add depth dividers
    for boundary in depth_boundaries[1:-1]:
        plt.axhline(y=boundary, color='k', linestyle='-', linewidth=1)
        plt.axvline(x=boundary, color='k', linestyle='-', linewidth=1)
    
    # Add section dividers
    for boundary in section_boundaries:
        plt.axhline(y=boundary, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add depth labels
    depth_labels = ['Surface \n(0 to -162 m)', 'Mid-depth \n(-162 to -458.5 m)', 'Bottom \n(<-458.5 m)']
    for i in range(len(depth_boundaries) - 1):
        mid_point = (depth_boundaries[i] + depth_boundaries[i+1]) / 2
        plt.text(mid_point, -4, depth_labels[i], fontsize=12, ha='center', va='top', fontweight='bold')

    depth_labels = ['Surface', 'Mid-depth', 'Bottom']
    for i in range(len(depth_boundaries) - 1):
        mid_point = (depth_boundaries[i] + depth_boundaries[i+1]) / 2 - 0.8
        plt.text(-7, mid_point, depth_labels[i], fontsize=12, ha='right', va='center', fontweight='bold')

    depth_labels = ['(0 to -162 m)', '(-162 to -458.5 m)', '(<-458.5 m)']
    for i in range(len(depth_boundaries) - 1):
        mid_point = (depth_boundaries[i] + depth_boundaries[i+1]) / 2 + 0.8
        plt.text(-7, mid_point, depth_labels[i], fontsize=12, ha='right', va='center', fontweight='bold')
    
        # Add directional arrows to each quadrant of the matrix
    font_size = 20  # Size for the arrow symbols
    box_size = 3  # INCREASED size of the black box (was 0.8)
    arrow_offset = 3  # Offset from corner of quadrants
    
    # Define each quadrant's boundaries (row_start, row_end, col_start, col_end)
    quadrants = []
    for i in range(len(depth_boundaries) - 1):  # Source depth (rows)
        for j in range(len(depth_boundaries) - 1):  # Destination depth (columns)
            quadrants.append((
                depth_boundaries[i],  # row_start
                depth_boundaries[i+1],  # row_end
                depth_boundaries[j],  # col_start
                depth_boundaries[j+1]  # col_end
            ))
    
    # Dictionary to store arrow types for legend
    arrow_types = {
        "↔": "Within the same layer", 
        "↓": "Downward to adjacent layer",
        "⇓": "Downward two layers",
        "↑": "Upward to adjacent layer",
        "⇑": "Upward two layers"
    }
    
    used_arrows = set()  # Keep track of which arrows were used
    
    # Loop through each quadrant and add the appropriate arrow
    # for i, (row_start, row_end, col_start, col_end) in enumerate(quadrants):
    #     # Calculate row and column indices from quadrant index
    #     source_depth_idx = i // 3  # Integer division to get row (0, 1, or 2)
    #     dest_depth_idx = i % 3     # Modulo to get column (0, 1, or 2)
        
    #     # Determine arrow type based on source and destination depths
    #     arrow = ""
        
    #     if source_depth_idx == dest_depth_idx:
    #         # Same layer connectivity
    #         arrow = "↔"
    #     elif source_depth_idx < dest_depth_idx:
    #         # Downward connectivity
    #         if dest_depth_idx - source_depth_idx == 1:
    #             # To adjacent layer
    #             arrow = "↓"
    #         else:
    #             # Skipping a layer
    #             arrow = "⇓"
    #     else:  # source_depth_idx > dest_depth_idx
    #         # Upward connectivity
    #         if source_depth_idx - dest_depth_idx == 1:
    #             # To adjacent layer
    #             arrow = "↑"
    #         else:
    #             # Skipping a layer
    #             arrow = "⇑"
        
    #     used_arrows.add(arrow)  # Add to used arrows for legend
        
    #     # Position of arrow (top right of quadrant with offset)
    #     arrow_x = col_end - arrow_offset
    #     arrow_y = row_start + arrow_offset
        
    #     # Add black box with white arrow
    #     rect = plt.Rectangle((arrow_x - box_size/2, arrow_y - box_size/2), box_size, box_size, 
    #                        facecolor='black', edgecolor='none', zorder=10)
    #     ax.add_patch(rect)
        
    #     # Add the arrow text in white
    #     plt.text(arrow_x, arrow_y, arrow, fontsize=font_size, ha='center', va='center', 
    #              color='white', fontweight='bold', zorder=11)
    # Set labels
    # Map section names: SS -> SC, NS -> NC
    def map_section_name(segment_name):
        if segment_name.startswith('SS_'):
            return segment_name.replace('SS_', 'SC_', 1)
        elif segment_name.startswith('NS_'):
            return segment_name.replace('NS_', 'NC_', 1)
        else:
            return segment_name

    # Apply the mapping to ordered_segments for display
    display_segments = [map_section_name(seg) for seg in ordered_segments]

    # Set labels
    plt.xlabel('Destination segments', fontsize=12)
    plt.ylabel('Source segments', fontsize=12)
    plt.xlabel('Destination segments', fontsize=12)
    plt.ylabel('Source segments', fontsize=12)

    # aspect ratio equal
    ax.set_aspect('equal')#, adjustable='box')
    
    plt.title('Volume transport-weighted differential transition matrix (EDDY - NW)', fontsize=16, y=1.07)
    
    # Add a legend explaining the color coding and the log scale
    # plt.figtext(0.5, -0.02, 
    #             'Red: Stronger connectivity in EDDY regime\nBlue: Stronger connectivity in NW regime', 
    #             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
    # Add custom tickmarks to colorbar to help interpret the log scale


    # Add custom tickmarks to colorbar with round numbers
    cbar = ax.collections[0].colorbar

    # Set specific tick positions for round percentage values
    percentage_values = [-30, -10, -3, -1, 0, 1, 3, 10, 30]
    tick_locs = []
    tick_labels = []

    for pct in percentage_values:
        if pct == 0:
            tick_locs.append(0)
            tick_labels.append('0')
        else:
            # Convert percentage to log scale position
            log_pos = np.sign(pct) * np.log10(1 + abs(pct))
            tick_locs.append(log_pos)
            if pct > 0:
                tick_labels.append(f'+{pct}')
            else:
                tick_labels.append(f'{pct}')

    # Apply the custom ticks
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels, fontsize=12)


    # cbar = ax.collections[0].colorbar
    # # Get the original tick positions
    # tick_locs = cbar.get_ticks()
    # # Create new tick labels that show the actual percentage differences
    # tick_labels = []
    # for loc in tick_locs:
    #     if loc == 0:
    #         tick_labels.append('0%')
    #     elif loc > 0:
    #         # Convert back from log scale to percentage for positive values
    #         actual_value = 10**(abs(loc)) - 1
    #         tick_labels.append(f'+{actual_value:.1f}%')
    #     else:
    #         # Convert back from log scale to percentage for negative values
    #         actual_value = 10**(abs(loc)) - 1
    #         tick_labels.append(f'-{actual_value:.1f}%')
    
    # Add legend for arrow symbols CLOSER to the bottom of the figure
    # legend_y_position = 0.1 # -0.05  # Moved legend closer to plot (was -0.15)
    # plt.figtext(0.5, legend_y_position, "Direction indicators:", ha='center', fontsize=12, fontweight='bold')
    
    # # Create two-column layout for legend
    # legend_items = []
    # for arrow, description in arrow_types.items():
    #     if arrow in used_arrows:  # Only include arrows that were used
    #         legend_items.append((arrow, description))
    
    # # Calculate positions for legend items (2 columns)
    # num_items = len(legend_items)
    # items_per_col = (num_items + 1) // 2  # Ceiling division
    
    # for i, (arrow, description) in enumerate(legend_items):
    #     # Determine column and row
    #     col = i // items_per_col
    #     row = i % items_per_col
        
    #     # Calculate x position (0.25 for first column, 0.65 for second)
    #     x_pos = 0.3 if col == 0 else 0.6
    #     # Calculate y position (starting from legend_y_position - 0.03)
    #     y_pos = legend_y_position - 0.04 - (0.025 * row)
        
    #     # Add black box with white arrow - INCREASED SIZE OF BOXES
    #     box_size_pts = 18  # Size in points (increased from 15)
    #     box_rect = plt.Rectangle((x_pos - 0.02 - box_size_pts/1000, y_pos - 0.01 - box_size_pts/2000), 
    #                           box_size_pts/500, box_size_pts/500, 
    #                           facecolor='black', transform=plt.gcf().transFigure)
    #     plt.gcf().patches.append(box_rect)

    #         # Add white arrow text
    #     plt.figtext(x_pos - 0.02, y_pos, arrow, color='white', fontsize=17, 
    #                ha='center', va='center', fontweight='bold')
        
    #     # Add description
    #     plt.figtext(x_pos + 0.01, y_pos, description, fontsize=10, ha='left', va='center')
    
    plt.subplots_adjust(bottom=0.2)
    # Apply the new tick labels
    # cbar.set_ticklabels(tick_labels)
    
    # Adjust tick labels
    # Adjust tick labels with mapped names
    plt.xticks(np.arange(len(ordered_segments)) + 0.5, display_segments, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(ordered_segments)) + 0.5, display_segments, fontsize=8)
    
    # plt.tight_layout()
    
    filename = 'CH2_Fig7_differential_matrix_EDDY_vs_NW_noarrows.jpeg'
    
    return plt, filename

vt_data_eddy = load_volume_transport_data(EDDY_MONTHS, regime="EDDY")
vt_data_nw   = load_volume_transport_data(NW_MONTHS, regime="NW")
plt_diff, filename_diff = create_differential_connectivity_matrix(
    month_transitions, 
    vt_data_nw, 
    vt_data_eddy, 
    NW_MONTHS, 
    EDDY_MONTHS
)
output_path = f'figures/results/{filename_diff}'
plt_diff.savefig(output_path, dpi=300, bbox_inches='tight')


# %%
