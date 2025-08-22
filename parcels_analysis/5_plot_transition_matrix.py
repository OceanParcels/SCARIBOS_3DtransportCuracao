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

# %%
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


def create_vt_weighted_connectivity_matrix(transitions_data, vt_data, months):
    """
    Create connectivity matrix visualization weighted by volume transport
    
    Parameters:
    transitions_data: Dictionary of transition paths by month
    vt_data: Dictionary of volume transport values by month and particle ID
    months: List of months to include in this analysis
    """
    print(f"\n=== CREATING VT-WEIGHTED CONNECTIVITY MATRIX ===")
    period_name = "All Months"
    if months == EDDY_MONTHS:
        period_name = "EDDY Months"
    elif months == NW_MONTHS:
        period_name = "NW Months"
    print(f"Processing {period_name}: {len(months)} months")
    
    # First, filter transitions to only include particles with VT values
    filtered_transitions = {}
    total_transitions_before = 0
    total_transitions_after = 0
    
    print("Filtering transitions to only include particles with VT data:")
    for month in months:
        if month not in transitions_data or month not in vt_data:
            print(f"  Skipping {month} - missing data")
            continue
        
        month_transitions = transitions_data[month]
        month_vt = vt_data[month]
        filtered_month_transitions = {}
        
        # Count particles before filtering
        particle_count_before = len(month_transitions)
        total_transitions_before += particle_count_before
        
        # Filter particles
        for particle_id_str, path in month_transitions.items():
            try:
                particle_id = float(particle_id_str)
                if particle_id in month_vt:  # This is where we check if the particle has VT data
                    filtered_month_transitions[particle_id_str] = path
            except (ValueError, KeyError):
                continue
        
        # Count particles after filtering
        particle_count_after = len(filtered_month_transitions)
        total_transitions_after += particle_count_after
        filtered_transitions[month] = filtered_month_transitions
        
        print(f"  {month}: {particle_count_after}/{particle_count_before} particles kept ({particle_count_before - particle_count_after} filtered out)")
    
    print(f"Total: {total_transitions_after}/{total_transitions_before} particles kept ({total_transitions_before - total_transitions_after} filtered out)")
    
    # Extract all unique segments from filtered transitions
    all_segments = set()
    for month in months:
        if month not in filtered_transitions:  # Use filtered_transitions instead of original transitions_data
            continue
        for particle_id, path in filtered_transitions[month].items():
            segments = [seg.strip() for seg in path.split(',')]
            all_segments.update(segments)
    
    print(f"Found {len(all_segments)} unique segments across selected months")
    
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
    
    # Calculate VT-weighted flows
    vt_flows = defaultdict(float)
    vt_outgoing_total = defaultdict(float)
    processed_count = 0
    
    # Process each month's data separately
    print("Computing VT-weighted flows:")
    for month in months:
        if month not in filtered_transitions or month not in vt_data:  # Use filtered_transitions
            print(f"  Skipping {month} - missing data")
            continue
            
        month_transitions = filtered_transitions[month]  # Use filtered_transitions
        month_vt = vt_data[month]
        
        month_processed = 0
        
        for particle_id_str, path in month_transitions.items():
            try:
                # Convert particle ID to the format in VT data
                particle_id = float(particle_id_str)
                
                # We've already filtered, so this should always exist
                vt_value = month_vt[particle_id]
                segments = [seg.strip() for seg in path.split(',')]
                
                for i in range(len(segments) - 1):
                    from_seg = segments[i]
                    to_seg = segments[i + 1]
                    
                    # Weight the connection by VT
                    vt_flows[(from_seg, to_seg)] += vt_value
                    vt_outgoing_total[from_seg] += vt_value
                
                month_processed += 1
                processed_count += 1
                
            except (ValueError, KeyError) as e:
                # This shouldn't happen since we filtered, but handle just in case
                continue
        
        print(f"  {month}: Processed {month_processed} particles")
    
    print(f"Total: Processed {processed_count} particles")
    print(f"Found {len(vt_flows)} unique segment connections with VT weighting")
    
    # Create connectivity matrix
    n_segments = len(ordered_segments)
    matrix = np.zeros((n_segments, n_segments))
    
    # Fill matrix with normalized VT-weighted values
    print("Building connectivity matrix...")
    connection_count = 0
    for (from_seg, to_seg), vt_weight in vt_flows.items():
        if from_seg in segment_to_idx and to_seg in segment_to_idx:
            from_idx = segment_to_idx[from_seg]
            to_idx = segment_to_idx[to_seg]
            if vt_outgoing_total[from_seg] > 0:
                matrix[from_idx, to_idx] = (vt_weight / vt_outgoing_total[from_seg]) * 100
                connection_count += 1
    
    print(f"Matrix populated with {connection_count} VT-weighted connections")
    
    # Analyze matrix values
    non_zero_values = matrix[matrix > 0]
    if len(non_zero_values) > 0:
        print(f"Matrix values - Min: {non_zero_values.min():.4f}%, Mean: {non_zero_values.mean():.4f}%, Max: {non_zero_values.max():.4f}%")
    
    # Create a mask for zero values
    zero_mask = (matrix == 0)
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
    
    print(f"Matrix prepared with {len(depth_boundaries)-1} depth layers and {len(section_boundaries)} section boundaries")
    
    # Create simplified tick labels that group by segment prefix
    simplified_labels = []
    current_prefix = ""
    for seg in ordered_segments:
        # Extract prefix (first two characters, typically section code like NS, WP, etc.)
        prefix = seg[:2] if len(seg) >= 2 else seg
        
        # Extract position number from the segment (typically after underscore and before D)
        position_match = re.search(r'_(\d+)D', seg)
        position = position_match.group(1) if position_match else ""
        
        # If prefix changes, include full segment name, otherwise just position
        if prefix != current_prefix:
            current_prefix = prefix
            simplified_labels.append(seg)  # Full name for first instance
        else:
            # Extract just the position number or any other distinguishing part
            simplified_labels.append(position)
    
    # Plot the connectivity matrix
    plt.figure(figsize=(15, 12.5))
    
    # Create custom colormap with white for zeros
    cmap = cmocean.cm.rain
    rain_white_zeros = cmap.copy()
    rain_white_zeros.set_bad('white')
    
    # Create heatmap with logarithmic color scale and masked zeros
    ax = sns.heatmap(matrix, cmap=rain_white_zeros, 
                    norm=LogNorm(vmin=1, vmax=100),
                    mask=zero_mask,
                    xticklabels=ordered_segments, yticklabels=ordered_segments,
                    cbar_kws={'label': 'Flow percentage [%, log scale]', 'extend': 'min'})
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)
    ax.collections[0].colorbar.set_label('Flow percentage [%, log scale]', fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([1, 10, 100])
    cbar.set_ticklabels(['1', '10', '100'])
    # Fill masked areas with white
    ax.collections[0].colorbar.solids.set_rasterized(True)
    
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
    ax.set_aspect('equal')
    plt.xlabel('Destination segments', fontsize=12)
    plt.ylabel('Source segments', fontsize=12)
    ax.set_aspect('equal')
    
    # Define period name for the title
    period_name = "entire period"
    if months == EDDY_MONTHS:
        period_name = "EDDY Months"
    elif months == NW_MONTHS:
        period_name = "NW Months"
    
    plt.title(f'Volume transport-weighted transition matrix (April 2020 - March 2024)', fontsize=16, y=1.07)
    
    plt.xticks(np.arange(len(ordered_segments)) + 0.5, display_segments, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(ordered_segments)) + 0.5, display_segments, fontsize=8)
    plt.tight_layout()
    
    # Add legend for arrow symbols CLOSER to the bottom of the figure
    # legend_y_position = 0.1  # Moved legend closer to plot (was -0.15)
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
        
    #     # Add white arrow text
    #     plt.figtext(x_pos - 0.02, y_pos, arrow, color='white', fontsize=17, 
    #                ha='center', va='center', fontweight='bold')
        
    #     # Add description
    #     plt.figtext(x_pos + 0.01, y_pos, description, fontsize=10, ha='left', va='center')
    
    # Adjust figure size to accommodate legend
    plt.subplots_adjust(bottom=0.2)


    # Create filename based on period
    filename = 'CH2_Fig6_transition_matrix_ALL_MONTHS_noarrows.jpeg'
    
    return plt, filename


plt_all, filename_all = create_vt_weighted_connectivity_matrix(month_transitions, vt_data_all, MONTHS_LIST)
output_path = f'figures/results/{filename_all}'
plt_all.savefig(output_path, dpi=300, bbox_inches='tight')



# %%
