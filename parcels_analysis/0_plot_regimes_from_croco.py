'''
Project: 3D flow and volume transport around Curaçao. 

Script to plot cross-section velocities from SCARIBOS. For this you need to first run:
0_calc_regimes_from_croco.py

Author: V Bertoncelj
'''

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import xroms
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import os
import geopandas as gpd
from matplotlib.patches import Patch


# Configuration parameters
angle_deg = 315
deflection = (-100, 180)
center_manual = (214, 170)
island_point_lon = -68.8
island_point_lat = 12.1

# Initialize dataset for coordinate calculations
config = 'SCARIBOS_V8'
file_path = f'~/croco/CONFIG/{config}/CROCO_FILES/croco_avg_Y2020M04.nc'

# Open with xroms to get coordinate information
ds = xroms.open_netcdf(file_path, chunks={'time': 1})
ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)
ds.xroms.set_grid(xgrid)

center = center_manual

# Coordinate rotation function
def rotate_coordinates(x, y, angle_deg, center):
    """Rotate coordinates by a given angle around a center."""
    angle_rad = np.deg2rad(angle_deg)
    x_defl, y_defl = deflection
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad) + x_defl
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad) + y_defl
    return x_rot, y_rot

# Calculate plot boundaries
x = ds['xi_rho'][50:250]
y = ds['eta_rho'][100:300]
x_rot, y_rot = rotate_coordinates(x, y, angle_deg, center)

lon_plot_min = ds['lon_rho'][0, 85].values
lon_plot_max = ds['lon_rho'][0, 217].values
lat_plot_min = ds['lat_rho'][152, 0].values
lat_plot_max = ds['lat_rho'][285, 0].values

# Directory where saved velocity data is stored
data_dir = 'croco_regimes'

# Define the years and months we want to include
years = ['Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
months = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']

# Create a list of all yearmonth combinations we need
sim_months = []
for year in years:
    if year == 'Y2020':
        # For 2020, start from M04
        for month in months[3:]:  # Skip M01, M02, M03
            sim_months.append(f"{year}{month}")
    elif year == 'Y2024':
        # For 2024, only include up to M03
        for month in months[:3]:  # Only M01, M02, M03
            sim_months.append(f"{year}{month}")
    else:
        # For all other years, include all months
        for month in months:
            sim_months.append(f"{year}{month}")

print(f"Total months to plot: {len(sim_months)}")
print(f"Months to plot: {sim_months}")

# Dictionary to store all monthly data
all_monthly_data = {}

# Loop through each month to load pre-calculated data
for sim_month in sim_months:
    print(f'Loading data for {sim_month}...')
    
    # File paths for the pre-calculated data
    cross_shore_filename = os.path.join(data_dir, f'{sim_month}_cross_shore_velocity.npy')
    along_shore_filename = os.path.join(data_dir, f'{sim_month}_along_shore_velocity.npy')
    
    # Check if files exist
    if os.path.exists(cross_shore_filename) and os.path.exists(along_shore_filename):
        # Load the data
        cross_rot_surf_transect = np.load(cross_shore_filename)
        along_rot_surf_transect = np.load(along_shore_filename)
        
        # Create standard x and y arrays for consistent plotting
        x_rot_newidx = np.linspace(15, 15, len(cross_rot_surf_transect))
        y_rot_newidx = np.linspace(100, 200, len(cross_rot_surf_transect))
        
        # Store the data
        all_monthly_data[sim_month] = {
            'cross_rot_surf_transect': cross_rot_surf_transect,
            'along_rot_surf_transect': along_rot_surf_transect,
            'x_rot_newidx': x_rot_newidx,
            'y_rot_newidx': y_rot_newidx
        }
    else:
        print(f"Warning: Could not find data files for {sim_month}")

# Dictionary to convert month codes to names
month_code_to_name = {
    'M01': 'JAN', 'M02': 'FEB', 'M03': 'MAR', 'M04': 'APR', 
    'M05': 'MAY', 'M06': 'JUN', 'M07': 'JUL', 'M08': 'AUG', 
    'M09': 'SEP', 'M10': 'OCT', 'M11': 'NOV', 'M12': 'DEC'
}

# Calculate month positions for consistent spacing across years
month_positions = {}
for idx, month_code in enumerate(months):
    month_positions[month_code] = idx * 20  # 20 units spacing between months

# Define months per year for plotting organization
months_per_year = {
    'Y2020': [f'Y2020{m}' for m in months if f'Y2020{m}' in sim_months],
    'Y2021': [f'Y2021{m}' for m in months if f'Y2021{m}' in sim_months],
    'Y2022': [f'Y2022{m}' for m in months if f'Y2022{m}' in sim_months],
    'Y2023': [f'Y2023{m}' for m in months if f'Y2023{m}' in sim_months],
    'Y2024': [f'Y2024{m}' for m in months if f'Y2024{m}' in sim_months]
}

def load_shapefiles():
    """Load and return all required shapefiles"""
    curacao = gpd.read_file("data/cuw_adm0/CUW_adm0.shp")
    venezuela = gpd.read_file("data/ven_adm/ven_admbnda_adm0_ine_20210223.shp")
    bes_islands = gpd.read_file("data/bes_adm0/BES_adm0.shp")
    aruba = gpd.read_file("data/abw_adm0/abw_admbnda_adm0_2020.shp")
    
    return curacao, venezuela, bes_islands, aruba

# Load shapefiles
curacao, venezuela, bes_islands, aruba = load_shapefiles()

#%%

# Create the final figure
fig = plt.figure(figsize=(15, 17))  # Increased height for the new row

num_rows = len(years)  # Now includes Y2024
gs = fig.add_gridspec(num_rows, 3, width_ratios=[1, 0.4, 3.6], height_ratios=[1.1] * num_rows)

# Add Curacao map to first row, left column
map_ax = fig.add_subplot(gs[0, 0])

map_ax.set_title("Cross-section", fontsize=14)
map_ax.set_xlabel("")
map_ax.set_ylabel("")
# Set up map ticks and labels
map_ax.set_xticks(np.arange(-71.5, -66.5, 0.5))
map_ax.set_yticks(np.arange(10.0, 13.5, 0.5))
map_ax.set_xticklabels(np.round(np.arange(-71.5, -66.5, 0.5), 1), fontsize=11)
map_ax.set_yticklabels(np.round(np.arange(10.0, 13.5, 0.5), 1), fontsize=11)
map_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.1f} °W'))
map_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f} °N'))

# Add transect line
map_ax.plot([lon_plot_min, lon_plot_max], [lat_plot_min, lat_plot_max], color='blue', linewidth=3)
map_ax.set_facecolor('#f7f7f7')
# Add section labels
map_ax.text(lon_plot_max-0.3, lat_plot_max-0.3, 'Northern\nsection', fontsize=14, rotation=angle_deg+90, ha='center', va='center', color='black')
map_ax.text(lon_plot_max-0.92, lat_plot_max-0.92, 'Southern\nsection', fontsize=14, rotation=angle_deg+90, ha='center', va='center', color='black')

# Set map limits
map_ax.set_xlim(lon_plot_min-0.2, lon_plot_max+0.2)
map_ax.set_ylim(lat_plot_min-0.1, lat_plot_max+0.1)
map_ax.set_xlabel(' ')

# Plot shapefiles
curacao.plot(ax=map_ax, color='saddlebrown',alpha = 0.4,  edgecolor='saddlebrown', zorder=20)
venezuela.plot(ax=map_ax, color='saddlebrown',alpha = 0.4,  edgecolor='saddlebrown', zorder=20)
bes_islands.plot(ax=map_ax, color='saddlebrown',alpha = 0.4,  edgecolor='saddlebrown', zorder=20)
aruba.plot(ax=map_ax, color='saddlebrown',alpha = 0.4,  edgecolor='saddlebrown', zorder=20)

# Create subplots for each year
for i, year in enumerate(['Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']):
    # For first row, use the right side of the grid
    if i == 0:
        ax = fig.add_subplot(gs[0, 1:])
        ax.text(0.05, 0.93, f"{year[1:]}", ha='center', va='center', color='k', fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='grey', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))
    
    # For subsequent rows (2021-2023), span the full width, for 2024 use the left column
    elif i == 1:
        ax = fig.add_subplot(gs[i, :])
        ax.text(0.03, 0.93, f"{year[1:]}", ha='center', va='center', color='k', fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='grey', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))
    elif i == 2:
        ax = fig.add_subplot(gs[i, :])
        ax.text(0.03, 0.93, f"{year[1:]}", ha='center', va='center', color='k', fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='grey', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))
    elif i == 3:
        ax = fig.add_subplot(gs[i, :])
        ax.text(0.03, 0.93, f"{year[1:]}", ha='center', va='center', color='k', fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='grey', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))

    else:
        ax = fig.add_subplot(gs[i, 0:2])
        ax.text(0.11, 0.93, f"{year[1:]}", ha='center', va='center', color='k', fontsize=16, transform=ax.transAxes,
            bbox=dict(facecolor='grey', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))
    
    # Plot each month for this year
    for month_idx, month_key in enumerate(months_per_year[year]):
        if month_key in all_monthly_data:
            data = all_monthly_data[month_key]
            
            # Get the month code (M01, M02, etc.)
            month_code = month_key[-3:]
            
            # Calculate x-offset for this month based on consistent month positions
            x_offset = month_positions[month_code]
            
            # Base y positions from the data
            y_positions = data['y_rot_newidx']
            
            # Define month label position
            month_label_x = x_offset + 15  # Center of the month's space
            month_label_y = np.max(y_positions) + 8  # Above the vectors
            
            # Vector plotting parameters
            skip = 10  # Skip more points for better visibility
            scale_factor = 15  # Scale factor for quiver plots
            
            # Plot original velocity field (u, v) in grey
            ax.quiver(
                data['x_rot_newidx'][::skip] + x_offset,
                y_positions[::skip],
                data['along_rot_surf_transect'][::skip],
                data['cross_rot_surf_transect'][::skip],
                scale=scale_factor,
                width=0.003 if i == 4 else (0.0013 if i == 0 else 0.001),
                color='grey',
            )
            
            # Plot alongshore velocity component in blue
            ax.quiver(
                data['x_rot_newidx'][::skip] + x_offset,
                y_positions[::skip],
                data['along_rot_surf_transect'][::skip],
                np.zeros_like(data['along_rot_surf_transect'][::skip]),
                scale=scale_factor,
                width=0.003 if i == 4 else (0.0013 if i == 0 else 0.001),
                color='mediumblue',
            )
            
            # Plot cross-shore velocity component in red
            # ax.quiver(
            #     data['x_rot_newidx'][::skip] + x_offset,
            #     y_positions[::skip],
            #     np.zeros_like(data['cross_rot_surf_transect'][::skip]),
            #     data['cross_rot_surf_transect'][::skip],
            #     scale=scale_factor,
            #     width=0.001,
            #     color='red',
            # )

            # Add reference line at y = 155
            ax.plot([-10, 300], [155, 155], color='grey', linewidth=1, linestyle='--')
            
            # Add month label with appropriate color coding
            month_name = month_code_to_name[month_code]
            
            # Color code months based on flow regime
            if year == 'Y2020' and month_code in ['M05', 'M06', 'M08', 'M09', 'M10', 'M11']:
                ax.text(month_label_x, month_label_y, month_name,
                       ha='center', va='top', fontsize=14,
                       bbox=dict(facecolor='blueviolet', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2020' and month_code in ['M04', 'M07', 'M12']:
                ax.text(month_label_x, month_label_y, month_name, 
                       ha='center', va='top', fontsize=14, 
                       bbox=dict(facecolor='olivedrab', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2021' and month_code in ['M01', 'M02', 'M03', 'M04', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']:
                ax.text(month_label_x, month_label_y, month_name, 
                       ha='center', va='top', fontsize=14, 
                       bbox=dict(facecolor='olivedrab', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2021' and month_code in ['M05', 'M06']:
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='blueviolet', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2022' and month_code in ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M12']:
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='olivedrab', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2022' and month_code in ['M09', 'M10', 'M11']:
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='blueviolet', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2023' and month_code in ['M01', 'M02', 'M09', 'M12']:
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='olivedrab', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2023' and month_code in ['M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M10', 'M11']:
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='blueviolet', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
            elif year == 'Y2024':
                # You can add specific color coding for 2024 months here if needed
                ax.text(month_label_x, month_label_y, month_name,
                          ha='center', va='top', fontsize=14,
                          bbox=dict(facecolor='blueviolet', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))

    # Set subplot properties
    ax.set_ylabel("Southern s.       Northern s.", fontsize=14)
    
    if i == 0:
        ax.set_xlim(50, 240)
    elif year == 'Y2024':
        # For Y2024, set xlim to show only 3 months worth of space
        ax.set_xlim(-5, 68)  # Adjusted to show only Jan-Mar
    else:
        ax.set_xlim(-5, 240)
    
    ax.set_ylim(105, 200)
    ax.set_xticks([])
    ax.set_yticks([])

# Create a new axes for the legend in the bottom row, shifted to the right
legend_ax = fig.add_axes([0.4, 0.02, 0.5, 0.05])  # Shifted to the right
legend_ax.axis('off')

# Create custom legend handles
legend_elements = [
    Patch(facecolor='none', alpha=0.5, label='Quivers:'),
    plt.Line2D([0], [0], color='grey', marker='>',
            markersize=8, label='Original (u, v)', linestyle=''),
    plt.Line2D([0], [0], color='mediumblue', marker='>',
            markersize=8, label='Alongshore component', linestyle=''),
    # plt.Line2D([0], [0], color='red', marker='>',
    #         markersize=8, label='Cross-shore component', linestyle=''),
    # Flow regime indicators
    Patch(facecolor='none', alpha=0.5, label='Flow regime:'),
    Patch(facecolor='olivedrab', alpha=0.5, label='NW flow'),
    Patch(facecolor='blueviolet', alpha=0.5, label='EDDY flow'),
    # Scale indicator
    Patch(facecolor='white', alpha=0, label='Quiver scale:     '),
    plt.Line2D([0], [0], color='none', label='', linestyle=''),
    Patch(facecolor='white', alpha=0, label='1 m/s')
]

# Create legend with 3 columns
legend = legend_ax.legend(
    handles=legend_elements, 
    loc='center', 
    bbox_to_anchor=(0.37, 3.67), #(0.5, 2.8), 
    fontsize=14,
    frameon=True,
    ncol=3,
    columnspacing=1.0,
    handletextpad=0.5
)

# Add a properly scaled quiver to represent 1 m/s
legend_ax2 = fig.add_axes([0.21, 0.183, 1, 0.04])   #([0.255, 0.14, 1, 0.04])  # Also shifted to the right
legend_ax2.axis('off')

# Add the reference quiver
reference_velocity = 1  # 1 m/s reference
legend_ax2.quiver(0, 0.5, reference_velocity, 0, 
                 scale=15, width=0.0018, color='k', zorder=30)

# Final adjustments and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.073)
plt.savefig('figures/methodology/CH2_Fig2_velocity_transects_Y2020-Y2024.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison plot saved as figures/methodology/CH2_Fig2_velocity_transects_Y2020-Y2024.png")

#%%

