'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot bar chart of total volume transport for each nearshore segment. 
The bars represent the total volue transport (in size) and the color represents the depth 
range of the particles before they reach the segment (if direction is backward).

You first need to run 8_calc_barchart_nearshore.py to generate the data for this script.

Author: V Bertoncelj
kernel: parcels_dev_local
'''


#%%
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import geopandas as gpd
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# load shapefile of curacao
curacao = gpd.read_file("data/cuw_adm0/CUW_adm0.shp")

# load bar chart data made with 8_calc_barchart_nearshore.py
DIRECTION = 'backward'  # 'forward' or 'backward' (needs to be calcualted with another script)
sections_filename = 'KC5D1' 
KC_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_KC5D1_{DIRECTION}.csv')
perc = KC_5D1_bar[KC_5D1_bar['VT_Percentage'].notnull()]
VT_total = KC_5D1_bar[KC_5D1_bar['VT_Value'].notnull()]
KC_5D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
KC_5D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'MP5D1' 
MP_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_MP5D1_{DIRECTION}.csv')
perc = MP_5D1_bar[MP_5D1_bar['VT_Percentage'].notnull()]
VT_total = MP_5D1_bar[MP_5D1_bar['VT_Value'].notnull()]
MP_5D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
MP_5D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'WP5D1'
WP_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_WP5D1_{DIRECTION}.csv')
perc = WP_5D1_bar[WP_5D1_bar['VT_Percentage'].notnull()]
VT_total = WP_5D1_bar[WP_5D1_bar['VT_Value'].notnull()]
WP_5D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
WP_5D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'KC_6D1'
KC_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_KC6D1_{DIRECTION}.csv')
perc = KC_6D1_bar[KC_6D1_bar['VT_Percentage'].notnull()]
VT_total = KC_6D1_bar[KC_6D1_bar['VT_Value'].notnull()]
KC_6D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
KC_6D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'MP_6D1'
MP_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_MP6D1_{DIRECTION}.csv')
perc = MP_6D1_bar[MP_6D1_bar['VT_Percentage'].notnull()]
VT_total = MP_6D1_bar[MP_6D1_bar['VT_Value'].notnull()]
MP_6D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
MP_6D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'WP_6D1'
WP_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_WP6D1_{DIRECTION}.csv')
perc = WP_6D1_bar[WP_6D1_bar['VT_Percentage'].notnull()]
VT_total = WP_6D1_bar[WP_6D1_bar['VT_Value'].notnull()]
WP_6D1_percentages_B = perc['VT_Percentage'].iloc[:3].to_list()
WP_6D1_total_VT_B = float(VT_total['VT_Value'].iloc[-1])

DIRECTION = 'forward'
sections_filename = 'KC5D1' 
KC_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_KC5D1_{DIRECTION}.csv')
perc = KC_5D1_bar[KC_5D1_bar['VT_Percentage'].notnull()]
VT_total = KC_5D1_bar[KC_5D1_bar['VT_Value'].notnull()]
KC_5D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
KC_5D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'MP5D1' 
MP_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_MP5D1_{DIRECTION}.csv')
perc = MP_5D1_bar[MP_5D1_bar['VT_Percentage'].notnull()]
VT_total = MP_5D1_bar[MP_5D1_bar['VT_Value'].notnull()]
MP_5D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
MP_5D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'WP5D1'
WP_5D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_WP5D1_{DIRECTION}.csv')
perc = WP_5D1_bar[WP_5D1_bar['VT_Percentage'].notnull()]
VT_total = WP_5D1_bar[WP_5D1_bar['VT_Value'].notnull()]
WP_5D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
WP_5D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'KC_6D1'
KC_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_KC6D1_{DIRECTION}.csv')
perc = KC_6D1_bar[KC_6D1_bar['VT_Percentage'].notnull()]
VT_total = KC_6D1_bar[KC_6D1_bar['VT_Value'].notnull()]
KC_6D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
KC_6D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'MP_6D1'
MP_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_MP6D1_{DIRECTION}.csv')
perc = MP_6D1_bar[MP_6D1_bar['VT_Percentage'].notnull()]
VT_total = MP_6D1_bar[MP_6D1_bar['VT_Value'].notnull()]
MP_6D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
MP_6D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

sections_filename = 'WP_6D1'
WP_6D1_bar = pd.read_csv(f'nearshore/bar_chart_data/BAR_VT_WP6D1_{DIRECTION}.csv')
perc = WP_6D1_bar[WP_6D1_bar['VT_Percentage'].notnull()]
VT_total = WP_6D1_bar[WP_6D1_bar['VT_Value'].notnull()]
WP_6D1_percentages_F = perc['VT_Percentage'].iloc[:3].to_list()
WP_6D1_total_VT_F = float(VT_total['VT_Value'].iloc[-1])

# %%
# load locations of segments that you generate and store with 3_calc_segmentation_ALL.py

KC_cross = np.load("segmentation/for_plotting_barcharts/KC_nearshore_segment_locations.npy", allow_pickle=True).item()
MP_cross = np.load("segmentation/for_plotting_barcharts/MP_nearshore_segment_locations.npy", allow_pickle=True).item()
WP_cross = np.load("segmentation/for_plotting_barcharts/WP_nearshore_segment_locations.npy", allow_pickle=True).item()

KC_4D1 = KC_cross['KC_4D1']
KC_4D2 = KC_cross['KC_4D2']
KC_5D1 = KC_cross['KC_5D1']
KC_6D1 = KC_cross['KC_6D1']
KC_7D1 = KC_cross['KC_7D1']
KC_7D2 = KC_cross['KC_7D2']

MP_4D1 = MP_cross['MP_4D1']
MP_4D2 = MP_cross['MP_4D2']
MP_5D1 = MP_cross['MP_5D1']
MP_6D1 = MP_cross['MP_6D1']
MP_7D1 = MP_cross['MP_7D1']
MP_7D2 = MP_cross['MP_7D2']

WP_4D1 = WP_cross['WP_4D1']
WP_4D2 = WP_cross['WP_4D2']
WP_5D1 = WP_cross['WP_5D1']
WP_6D1 = WP_cross['WP_6D1']
WP_7D1 = WP_cross['WP_7D1']
WP_7D2 = WP_cross['WP_7D2']


# %%

# plotting parameters
categories = ['Surface', 'Mid', 'Deep']
colors = ['cornflowerblue', 'tomato', 'lightseagreen']
bar_width = 0.06  # Width of each individual bar (reduced since we have 3 bars)
bar_height_scale = 0.0008  # Scale factor for bar height
arrlw = 1.4
color_south = 'silver'
color_north = 'k'

# Define all locations in a dictionary for easy modification
locations = {
    # Format: 'location_name': {'center_f': (lon, lat), 'center_b': (lon, lat), 'target': (lon, lat), 'label_pos': (lon, lat)}
    'WP_South': {
        'center_f': (-69.4, 12.3),
        'center_b': (-69.4, 12.385),
        'target': (WP_5D1['crossing_lon'].min(), WP_5D1['crossing_lat'].min()),
        'label_pos': (-69.7, 12.38),
        'arc_rad_f': 0.2,
        'arc_rad_b': -0.2,
        'color': color_south,
        'percentages_f': WP_5D1_percentages_F,
        'percentages_b': WP_5D1_percentages_B,
        'total_vt': WP_5D1_total_VT_B,
        'label_text': 'West Point (South)',
        'arrow_offset_dir_f': 'lon',
        'arrow_offset_sign': 'none',
        'arrow_offset_dir_b': 'lon'
    },
    'MP_South': {
        'center_f': (-69.4, 11.82),
        'center_b': (-69.4, 12.02),
        'target': (MP_5D1['crossing_lon'].min(), MP_5D1['crossing_lat'].min()),
        'label_pos': (-69.7, 12.01),
        'arc_rad_f': 0.2,
        'arc_rad_b': -0.2,
        'color': color_south,
        'percentages_f': MP_5D1_percentages_F,
        'percentages_b': MP_5D1_percentages_B,
        'total_vt': MP_5D1_total_VT_B,
        'label_text': 'Mid Point (South)',
        'arrow_offset_dir_f': 'lon',
        'arrow_offset_sign': 'none',
        'arrow_offset_dir_b': 'lon'
    },
    'KC_South': {
        'center_f': (-68.75, 11.62),
        'center_b': (-68.97, 11.62),
        'target': (KC_5D1['crossing_lon'].min(), KC_5D1['crossing_lat'].min()),
        'label_pos': (-68.95, 11.57),
        'arc_rad_f': 0.2,
        'arc_rad_b': -0.2,
        'color': color_south,
        'percentages_f': KC_5D1_percentages_F,
        'percentages_b': KC_5D1_percentages_B,
        'total_vt': KC_5D1_total_VT_B,
        'label_text': 'Klein Curaçao (South)',
        'arrow_offset_dir_f': 'lat',
        'arrow_offset_sign': 'none',
        'arrow_offset_dir_b': 'lat'
    },
    'WP_North': {
        'center_b': (-69.165, 12.5),
        'center_f': (-68.94, 12.5),
        'target': (WP_6D1['crossing_lon'].max(), WP_6D1['crossing_lat'].max()),
        'label_pos': (-69.25, 12.93),
        'arc_rad_f': -0.2,
        'arc_rad_b': 0.2,
        'color': color_north,
        'percentages_f': WP_6D1_percentages_F,
        'percentages_b': WP_6D1_percentages_B,
        'total_vt': WP_6D1_total_VT_B,
        'label_text': 'West Point (North)',
        'arrow_offset_dir_f': 'costum',
        'arrow_offset_sign': 'neg',
        'arrow_offset_dir_b': 'costum'
    },
    'MP_North': {
        'center_f': (-68.43, 12.25),
        'center_b': (-68.43, 12.495),
        'target': (MP_6D1['crossing_lon'].max(), MP_6D1['crossing_lat'].max()),
        'label_pos': (-68.32, 12.48),
        'arc_rad_f': -0.2,
        'arc_rad_b': 0.2,
        'color': color_north,
        'percentages_f': MP_6D1_percentages_F,
        'percentages_b': MP_6D1_percentages_B,
        'total_vt': MP_6D1_total_VT_B,
        'label_text': 'Mid Point (North)',
        'arrow_offset_dir_f': 'lon',
        'arrow_offset_sign': 'neg',
        'arrow_offset_dir_b': 'lon'
    },
    'KC_North': {
        'center_f': (-68.43, 11.873),
        'center_b': (-68.43, 12.044),
        'target': (KC_6D1['crossing_lon'].max(), KC_6D1['crossing_lat'].max()),
        'label_pos': (-68.32, 12.03),
        'arc_rad_f': -0.2,
        'arc_rad_b': 0.2,
        'color': color_north,
        'percentages_f': KC_6D1_percentages_F,
        'percentages_b': KC_6D1_percentages_B,
        'total_vt': KC_6D1_total_VT_B,
        'label_text': 'Klein Curaçao (North)',
        'arrow_offset_dir_f': 'lon',
        'arrow_offset_sign': 'neg',
        'arrow_offset_dir_b': 'lon'
    }
}

# Function to create bar charts and arrows for a location
def create_visualization_for_location(ax, loc_data):
    # Calculate the bar height scale based on total VT
    height_scale = bar_height_scale * loc_data['total_vt'] / 100
    
    # Create side-by-side horizontal bar charts for forward direction
    center_x_f, center_y_f = loc_data['center_f']
    
    # Calculate starting x position for the leftmost bar (Surface)
    total_bar_width = 3 * bar_width  # 3 bars with no spacing
    start_x_f = center_x_f - total_bar_width/2
    
    for i, (percentage, color) in enumerate(zip(loc_data['percentages_f'], colors)):
        bar_height = percentage * height_scale
        bar_x = start_x_f + i * bar_width
        
        # Create rectangle for each category
        rect_f = Rectangle((bar_x, center_y_f), 
                          bar_width, bar_height, 
                          facecolor=color, edgecolor='white', linewidth=1, alpha=0.5)
        ax.add_patch(rect_f)
        
        # Add percentage text if significant
        if percentage >= 30:
            text_x = bar_x + bar_width/2
            text_y = center_y_f + bar_height/2
            ax.text(text_x, text_y, f'{percentage:.0f}%', 
                   ha='center', va='center', fontsize=8)
        elif percentage > 0:
            # Add text above bar for small percentages
            text_x = bar_x + bar_width/2
            text_y = center_y_f + bar_height + 0.005
            ax.text(text_x, text_y, f'{percentage:.0f}%', 
                   ha='center', va='bottom', fontsize=7)
    
    # Create side-by-side horizontal bar charts for backward direction
    center_x_b, center_y_b = loc_data['center_b']
    start_x_b = center_x_b - total_bar_width/2
    
    for i, (percentage, color) in enumerate(zip(loc_data['percentages_b'], colors)):
        bar_height = percentage * height_scale
        bar_x = start_x_b + i * bar_width
        
        # Create rectangle for each category
        rect_b = Rectangle((bar_x, center_y_b), 
                          bar_width, bar_height, 
                          facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect_b)
        
        # Add percentage text if significant
        if percentage >= 30:
            text_x = bar_x + bar_width/2
            text_y = center_y_b + bar_height/2
            ax.text(text_x, text_y, f'{percentage:.0f}%', 
                   ha='center', va='center', fontsize=8)
        elif percentage > 0:
            # Add text above bar for small percentages
            text_x = bar_x + bar_width/2
            text_y = center_y_b + bar_height + 0.005
            ax.text(text_x, text_y, f'{percentage:.0f}%', 
                   ha='center', va='bottom', fontsize=7)
    
    # Calculate arrow start points (from center of bar group)
    if loc_data['arrow_offset_sign'] == 'neg':
        offset_multiplier = -1
    else:
        offset_multiplier = 1
    
    # Calculate maximum height for each direction to position arrows
    max_height_f = max([p * height_scale for p in loc_data['percentages_f']])
    max_height_b = max([p * height_scale for p in loc_data['percentages_b']])
    
    # For forward arrow
    if loc_data['arrow_offset_dir_f'] == 'costum':
        start_point_f = loc_data['center_f']

    elif loc_data['arrow_offset_dir_f'] == 'lon':
        start_point_f = (center_x_f + offset_multiplier * total_bar_width/2, 
                        center_y_f + max_height_f/2)
    elif loc_data['arrow_offset_dir_f'] == 'lat':
        start_point_f = (center_x_f, 
                        center_y_f + max_height_f + offset_multiplier * 0.01)
    else:  # 'none'
        start_point_f = (center_x_f, center_y_f + max_height_f/2)
    
    # For backward arrow
    if loc_data['arrow_offset_dir_b'] == 'costum':
        start_point_b = loc_data['center_b']

    elif loc_data['arrow_offset_dir_b'] == 'lon':
        start_point_b = (center_x_b + offset_multiplier * total_bar_width/2, 
                        center_y_b + max_height_b/2)
    elif loc_data['arrow_offset_dir_b'] == 'lat':
        start_point_b = (center_x_b, 
                        center_y_b + max_height_b + offset_multiplier * 0.01)
    else:  # 'none'
        start_point_b = (center_x_b, center_y_b + max_height_b/2)
    
    # Create the arrows
    arc_f = FancyArrowPatch(
        start_point_f, 
        loc_data['target'],
        connectionstyle=f"arc3,rad={loc_data['arc_rad_f']}",
        arrowstyle='<|-', 
        lw=arrlw,
        color='silver',#loc_data['color'],
        linestyle='dashed',
        alpha=1,
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0
    )
    ax.add_patch(arc_f)

    arc_b = FancyArrowPatch(
        start_point_b, 
        loc_data['target'],
        connectionstyle=f"arc3,rad={loc_data['arc_rad_b']}",
        arrowstyle='-|>', 
        lw=arrlw,
        color='k',#loc_data['color'],
        linestyle='-.',
        alpha=1,
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0
    )
    ax.add_patch(arc_b)

    # Add the label
    ax.text(
        loc_data['label_pos'][0],
        loc_data['label_pos'][1],
        f"{loc_data['label_text']}\nVolume transport \nin 4 years: $\\mathbf{{{loc_data['total_vt']:.0f}\\,Sv}}$",
        fontsize=10,
        ha='left',
        va='center',
        color='black'
    )


def create_full_visualization(ax):
    # Draw Curacao shapefile first
    for geom in curacao.geometry:
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.coords.xy
                plt.fill(x, y, color='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
        else:
            print('not a multipolygon')
    
    # Draw crossing lines
    color_south = 'k'
    color_north = 'k'
    lw_highlight = 5

    # Add KC points
    ax.plot([KC_5D1['crossing_lon'].min(), KC_5D1['crossing_lon'].max()],
        [KC_5D1['crossing_lat'].min(), KC_5D1['crossing_lat'].max()],
        c=color_south, alpha=1, linewidth=lw_highlight)
    ax.plot([KC_6D1['crossing_lon'].min(), KC_6D1['crossing_lon'].max()],
        [KC_6D1['crossing_lat'].min(), KC_6D1['crossing_lat'].max()],
        c=color_north, alpha=1, linewidth=lw_highlight)

    # MP
    ax.plot([MP_5D1['crossing_lon'].min(), MP_5D1['crossing_lon'].max()],
        [MP_5D1['crossing_lat'].min(), MP_5D1['crossing_lat'].max()],
        c=color_south, alpha=1, linewidth=lw_highlight)
    ax.plot([MP_6D1['crossing_lon'].min(), MP_6D1['crossing_lon'].max()],
        [MP_6D1['crossing_lat'].min(), MP_6D1['crossing_lat'].max()],
        c=color_north, alpha=1, linewidth=lw_highlight)

    # WP
    ax.plot([WP_5D1['crossing_lon'].min(), WP_5D1['crossing_lon'].max()],
        [WP_5D1['crossing_lat'].min(), WP_5D1['crossing_lat'].max()],
        c=color_south, alpha=1, linewidth=lw_highlight)
    ax.plot([WP_6D1['crossing_lon'].min(), WP_6D1['crossing_lon'].max()],
        [WP_6D1['crossing_lat'].min(), WP_6D1['crossing_lat'].max()],
        c=color_north, alpha=1, linewidth=lw_highlight)

    # Loop through all locations and create visualizations
    for loc_name, loc_data in locations.items():
        create_visualization_for_location(ax, loc_data)
    
    # Create legends
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Surface (0 to -162m)', 
               markerfacecolor=colors[0], markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Mid-range (-162 to -458.5m)', 
               markerfacecolor=colors[1], markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Deep (<-458.5m)', 
               markerfacecolor=colors[2], markersize=10),
        Line2D([0], [0], marker='o', color='w', label=' ', markerfacecolor='w', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=' ', markerfacecolor='w', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=' ', markerfacecolor='w', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=' ', markerfacecolor='w', markersize=10),
    ]

    legend_elements2 = [
        Line2D([0], [0], color=color_south, lw=arrlw, linestyle='-.', label='Arriving'),
        Line2D([0], [0], color='silver', lw=arrlw, linestyle='dashed', label='Leaving'),
    ]

    # First legend for depth categories
    first_legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True,
                            title='Max depth of particles \nprior/after crossing:',
                            title_fontproperties={'size': 10, 'weight': 'bold'},
                            bbox_to_anchor=(1, 1.06))

    # Add the first legend manually to the axes
    ax.add_artist(first_legend)

    # Second legend for particle types
    second_legend = ax.legend(handles=legend_elements2, loc='upper left', fontsize=10, frameon=False,
                            title='                Type of particles:',
                            title_fontproperties={'size': 10, 'weight': 'bold'},
                            bbox_to_anchor=(0.74, 0.9614))
    
    # Set the limits of the plot
    ax.set_xlim(-69.7, -68.07)
    ax.set_ylim(11.55, 13)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Add title
    ax.set_title('Bar charts of arriving and leaving\nvolume transport-weighted particles', 
                fontsize=16, fontweight='bold')

    plt.savefig('figures/results/CH2_Fig10_barchart_nearshore.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(14, 14))
create_full_visualization(ax)
plt.show()


# %%
