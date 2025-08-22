"""
Project: 3D flow and volume transport around Curaçao. 

Water mass analysis for PE529 CTD data and SCARIBOS comparison. For that you need to extract 
the temperature and salinity data from SCARIBOS by running script:
0_extract_SCARIBOS_temperature_salinity.py
You also need the data from CTD stations during RV Pelagia expediton (64PE529)

Author: V Bertoncelj
"""

#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import pycnv
import gsw
from matplotlib.patches import Ellipse
import warnings
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import xarray as xr
warnings.filterwarnings('ignore')


# Load the bathymetry data
bathy_gebpel_file = 'data_large_files/gebco_and_pelagia_merged_SCARIBOS_V2.nc'
bathy_gebpel = xr.open_dataset(bathy_gebpel_file)
bathy_gebpel_topo = bathy_gebpel['topo']

# Define a custom colormap for the bathymetry
def custom_div_cmap(numcolors=50, name='custom_div_cmap',
                    mincol='blue', midcol2='yellow', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                              colors=[mincol, midcol, midcol2, maxcol],
                                              N=numcolors)
    return cmap
blevels = [-5373, -4000, -3500, -3000, -2000, -1500, -1000, -750, -500, -250, 0]   # define levels for plotting (transition of colorbar)
N       = (len(blevels)-1)*2
cmap2_bl   = custom_div_cmap(N, mincol='#3f3f3f', midcol='dimgrey', midcol2='#888888' ,maxcol='w')
levels = 10
vmin = -5373
vmax = 0

# Configuration
NUM_WATER_MASSES = 3
DEPTH_LIMIT = 1000  # meters
ZOOM_DEPTH = 250    # meters for zoomed profile
OUTPUT_DIR = 'figures_final_CH2'
DATA_DIR = 'data'
SCARIBOS_FILE = 'SCARIBOS_T_S_Y2024M01.txt' # this is extraction from the SCARIBOS model
# Colors for water masses
colors = ['dodgerblue', 'tomato', 'lightseagreen']
colors_text = ['darkblue', 'darkred', 'teal']

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load SCARIBOS data
print("Loading SCARIBOS data...")
try:
    scaribos_df = pd.read_csv(SCARIBOS_FILE)
    
    # Parse semicolon-separated values
    salt_lists = []
    temp_lists = []
    for _, row in scaribos_df.iterrows():
        salt_vals = [float(x) for x in str(row['salt']).split(';') if x]
        temp_vals = [float(x) for x in str(row['temp']).split(';') if x]
        salt_lists.append(salt_vals)
        temp_lists.append(temp_vals)
    
    # Flatten into arrays
    scaribos_salinity = np.concatenate(salt_lists)
    scaribos_temperature = np.concatenate(temp_lists)
    print(f"Loaded {len(scaribos_salinity)} SCARIBOS data points")
except Exception as e:
    print(f"Error loading SCARIBOS data: {e}")
    scaribos_salinity = None
    scaribos_temperature = None

# Load and process CTD station files
print(f"\nProcessing CTD files from {DATA_DIR}...")
ctd_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.cnv')]
print(f"Found {len(ctd_files)} CTD files")

station_data = []

for filepath in ctd_files:
    try:
        # Load CTD data
        cnv = pycnv.pycnv(filepath)
        
        # Extract parameters
        depth = cnv.data['p']
        temp_pot = cnv.data['potemp090C']
        sal_prac = cnv.data['sal00']
        density = gsw.rho(sal_prac, temp_pot, 0) - 1000  # sigma-t
        
        # Extract station ID
        station_id = os.path.basename(filepath).split('-')[-1].split('.')[0]
        
        # Extract coordinates from header
        lat_decimal, lon_decimal = None, None
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                if '* NMEA Latitude' in line:
                    parts = line.strip().split('=')[1].strip().split()
                    if len(parts) == 3:
                        lat_deg, lat_min, lat_dir = float(parts[0]), float(parts[1]), parts[2]
                        lat_decimal = lat_deg + lat_min/60
                        if lat_dir == 'S':
                            lat_decimal = -lat_decimal
                
                elif '* NMEA Longitude' in line:
                    parts = line.strip().split('=')[1].strip().split()
                    if len(parts) == 3:
                        lon_deg, lon_min, lon_dir = float(parts[0]), float(parts[1]), parts[2]
                        lon_decimal = lon_deg + lon_min/60
                        if lon_dir == 'W':
                            lon_decimal = -lon_decimal
                
                if lat_decimal is not None and lon_decimal is not None:
                    break
        
        station_data.append({
            'depth': depth,
            'density': density,
            'temperature': temp_pot,
            'salinity': sal_prac,
            'station': station_id,
            'max_depth': np.max(depth),
            'latitude': lat_decimal,
            'longitude': lon_decimal
        })
        
        print(f"Processed: {os.path.basename(filepath)}")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

print(f"\nSuccessfully processed {len(station_data)} stations")

# Filter deep stations
deep_stations = [station for station in station_data if station['max_depth'] > DEPTH_LIMIT]
print(f"Found {len(deep_stations)} stations with depth < -{DEPTH_LIMIT}m")

# Create figure
fig = plt.figure(figsize=(11, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 2], width_ratios=[1, 1])

ax_map = fig.add_subplot(gs[0, 0])  # Station map (top left)
ax1 = fig.add_subplot(gs[0, 1])  # Full depth profile (top right)
ax_zoom = fig.add_subplot(gs[1, :])  # Zoomed profile (middle)
ax2 = fig.add_subplot(gs[2, 0])  # T-S diagram without SCARIBOS (bottom left)
ax_scaribos = fig.add_subplot(gs[2, 1])  # T-S diagram with SCARIBOS (bottom right)

# Get data ranges
all_salinities = []
all_temperatures = []
all_densities = []

for station in deep_stations:
    all_salinities.extend(station['salinity'])
    all_temperatures.extend(station['temperature'])
    all_densities.extend(station['density'])

sal_min, sal_max = np.min(all_salinities), np.max(all_salinities)
temp_min, temp_max = np.min(all_temperatures), np.max(all_temperatures)
density_min, density_max = np.min(all_densities), np.max(all_densities)

# Create density contours for T-S diagram
sal_grid = np.linspace(sal_min-0.5, sal_max+0.5, 100)
temp_grid = np.linspace(temp_min-5, temp_max+5, 100)
SAL, TEMP = np.meshgrid(sal_grid, temp_grid)
DENSITY = gsw.rho(SAL, TEMP, 0) - 1000

# Plot density contours
cs = ax2.contour(SAL, TEMP, DENSITY, colors='gray', linestyles='--', alpha=0.5)
ax2.clabel(cs, inline=1, fontsize=8, fmt='%.2f')

# Process each station
all_water_masses = []
mld_values = []

for station in deep_stations:
    # Detect mixed layer depth (density threshold method)
    surface_density = station['density'][0]
    mld = 10.0  # default
    for i in range(len(station['depth'])):
        if station['density'][i] - surface_density > 0.03:
            mld = station['depth'][i]
            break
    mld_values.append(mld)
    
    # Detect water masses using K-means
    features = np.column_stack([station['density'], station['temperature'], station['salinity']])
    kmeans = KMeans(n_clusters=NUM_WATER_MASSES, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Create water mass info
    water_masses = []
    for cluster in range(NUM_WATER_MASSES):
        cluster_mask = labels == cluster
        cluster_depths = station['depth'][cluster_mask]
        
        water_masses.append({
            'cluster': cluster,
            'mean_depth': np.mean(cluster_depths),
            'mask': cluster_mask
        })
    
    # Sort by depth
    water_masses.sort(key=lambda x: x['mean_depth'])
    
    # Plot density profiles
    ax1.plot(station['density'], station['depth'], color='gray', alpha=0.5, linewidth=0.5)
    ax_zoom.plot(station['density'], station['depth'], color='gray', alpha=0.5, linewidth=0.5)
    
    # Plot water masses
    for i, mass in enumerate(water_masses):
        if i < len(colors):
            cluster_mask = mass['mask']
            color = colors[i]
            
            ax1.scatter(station['density'][cluster_mask], station['depth'][cluster_mask], 
                       s=20, color=color, alpha=0.5)
            ax_zoom.scatter(station['density'][cluster_mask], station['depth'][cluster_mask], 
                           s=20, color=color, alpha=0.5)
            ax2.scatter(station['salinity'][cluster_mask], station['temperature'][cluster_mask],
                       s=10, color=color, alpha=0.5)
            ax_scaribos.scatter(station['salinity'][cluster_mask], station['temperature'][cluster_mask],
                               s=10, color=color, alpha=0.3)

# Calculate mixed layer statistics
mean_mld = np.mean(mld_values)
std_mld = np.std(mld_values)

# Add mixed layer visualization with grey hatching
ax_zoom.axhspan(0, mean_mld, color='grey', alpha=0.1, zorder=0)
ax_zoom.axhspan(0, mean_mld, facecolor='none', edgecolor='grey', 
                hatch='///', alpha=0.5, zorder=0, linewidth=0)
ax_zoom.axhline(mean_mld, color='grey', linestyle='--', linewidth=2)
ax_zoom.text(26.9, mean_mld - 3, 
            f'Mixed layer: -{mean_mld:.1f}±{std_mld:.1f} m', 
            color='k', ha='left', va='center',
            bbox=dict(facecolor='white', alpha=0.7))

# Calculate water mass statistics
print(f"\nMixed Layer Depth: {mean_mld:.1f} ± {std_mld:.1f} m")
print("Identified Water Masses:")

for wm_idx in range(NUM_WATER_MASSES):
    all_depths = []
    all_temps = []
    all_sals = []
    
    # Collect data from all stations
    for station in deep_stations:
        features = np.column_stack([station['density'], station['temperature'], station['salinity']])
        kmeans = KMeans(n_clusters=NUM_WATER_MASSES, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Sort clusters by mean depth
        clusters = []
        for c in range(NUM_WATER_MASSES):
            mask = labels == c
            mean_depth = np.mean(station['depth'][mask])
            clusters.append((c, mean_depth, mask))
        clusters.sort(key=lambda x: x[1])
        
        if wm_idx < len(clusters):
            mask = clusters[wm_idx][2]
            all_depths.extend(station['depth'][mask])
            all_temps.extend(station['temperature'][mask])
            all_sals.extend(station['salinity'][mask])
    
    if not all_depths:
        continue
    
    # Calculate percentiles
    depth_5th = np.percentile(all_depths, 5)
    depth_95th = np.percentile(all_depths, 95)
    temp_5th = np.percentile(all_temps, 5)
    temp_95th = np.percentile(all_temps, 95)
    sal_5th = np.percentile(all_sals, 5)
    sal_95th = np.percentile(all_sals, 95)
    
    print(f"\nWater Mass {wm_idx+1}:")
    print(f"  Depth Range (5-95%): {depth_5th:.1f} - {depth_95th:.1f} m")
    print(f"  Temperature Range (5-95%): {temp_5th:.2f} - {temp_95th:.2f} °C")
    print(f"  Salinity Range (5-95%): {sal_5th:.2f} - {sal_95th:.2f} PSU")
    
    # Draw depth ranges
    color = colors[wm_idx]
    for ax in [ax1, ax_zoom]:
        ax.axhspan(depth_5th, depth_95th, color=color, alpha=0.1)
        ax.axhline(depth_5th+(depth_95th-depth_5th)/2, color=color, linestyle='--')
    
    color = colors_text[wm_idx]
    # Add labels
    ax1.text(density_min + (density_max - density_min) * 0.05, 
            depth_5th+(depth_95th-depth_5th)/2, 
            f'WM {wm_idx+1}: -{depth_5th:.0f} to -{depth_95th:.0f} m', 
            color=color, ha='left', va='center',
            bbox=dict(facecolor='white', alpha=0.7))
    
    if depth_5th < ZOOM_DEPTH:
        label_depth = min(depth_5th+(depth_95th-depth_5th)/2, ZOOM_DEPTH-20)
        ax_zoom.text(26.9, label_depth, 
                    f'WM {wm_idx+1}: -{depth_5th:.0f} to -{depth_95th:.0f} m', 
                    color=color, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw T-S boundaries
    width = (sal_95th - sal_5th) * 1.2
    height = (temp_95th - temp_5th) * 1.2
    center_sal = (sal_95th + sal_5th) / 2
    center_temp = (temp_95th + temp_5th) / 2
    color = colors[wm_idx]
    ellipse = Ellipse((center_sal, center_temp), width, height,
                     fill=False, edgecolor=color, linestyle='-', linewidth=1.5)
    ax2.add_patch(ellipse)

# Customize plots
# Station map (top left)
try:
    import geopandas as gpd


    # contourf_zoomed = ax_map.contourf(bathy_gebpel_topo['lon'], bathy_gebpel_topo['lat'], bathy_gebpel_topo, 25, 
    #                                     cmap=cmap2_bl,vmin=-5000, vmax = 0, rasterized=True, zorder = 1)

    blevels_500m = np.arange(-5000, 1, 500)  # Every 500m from -5000 to 0
    topo_clipped = np.clip(bathy_gebpel_topo, -5000, 0)

    contourf_zoomed = ax_map.contourf(
        bathy_gebpel_topo['lon'],
        bathy_gebpel_topo['lat'],
        topo_clipped,
        levels=blevels_500m,  # Use explicit levels instead of number of contours
        cmap=cmap2_bl,
        extend='min',  # Handle values below the minimum level
        rasterized=True,
        zorder=1
    )
    # Try to load shapefiles
    curacao = gpd.read_file("cuw_adm0/CUW_adm0.shp")
    venezuela = gpd.read_file("ven_adm/ven_admbnda_adm0_ine_20210223.shp")
    bes_islands = gpd.read_file("bes_adm0/BES_adm0.shp")
    aruba = gpd.read_file("abw_adm0/abw_admbnda_adm0_2020.shp")
    
    curacao.plot(ax=ax_map, color='saddlebrown', alpha = 0.4, edgecolor='saddlebrown')
    venezuela.plot(ax=ax_map, color='saddlebrown', alpha = 0.4, edgecolor='saddlebrown')
    bes_islands.plot(ax=ax_map, color='saddlebrown', alpha = 0.4, edgecolor='saddlebrown')
    aruba.plot(ax=ax_map, color='saddlebrown', alpha = 0.4, edgecolor='saddlebrown')
    # ax_map.set_facecolor('#f7f7f7')

    # Plot stations
    for station in deep_stations:
        if station['latitude'] is not None and station['longitude'] is not None:
            st_map = ax_map.scatter(station['longitude'], station['latitude'], 
                         s=45, color='mediumblue', edgecolor='white', zorder=10)

    ax_map.set_title(f'A) 64PE529 CTD stations and bathymetry')
    ax_map.text(-68.8, 12.3, 'Curaçao', fontsize=10, color='k', ha='center', va='center')
    # Add arrow pointing to Curaçao
    ax_map.annotate(
        '', 
        xy=(-68.9, 12.2),  # arrow tip (approximate location of Curaçao)
        xytext=(-68.8, 12.25),  # arrow tail (from label)
        arrowprops=dict(facecolor='k', edgecolor='k', arrowstyle='->', lw=1.5)
    )
    ax_map.text(-68.5, 12.18, 'Klein\nCuraçao', fontsize=10, color='k', ha='center', va='center')
    # Add arrow pointing to Klein Curaçao
    ax_map.annotate(
        '',
        xy=(-68.62, 12.0),  # arrow tip (approximate location of Klein Curaçao)
        xytext=(-68.5, 12.1),  # arrow tail (from label)
        arrowprops=dict(facecolor='k', edgecolor='k', arrowstyle='->', lw=1.5)
    )

    # Add legend for station markers
    st_map = ax_map.scatter([], [], s=45, color='mediumblue', edgecolor='white', label='CTD stations')
    ax_map.legend(loc='upper right')
    
    # Set custom ticks
    xticks = np.arange(-69.8, -68.1, 0.4)
    yticks = np.arange(11.75, 12.85, 0.4)
    ax_map.set_xticks(xticks)
    ax_map.set_yticks(yticks)
    ax_map.set_xlim(-69.7, -68.15)
    ax_map.set_ylim(11.5, 12.9)
    ax_map.set_xticklabels([f"{abs(x):.1f}° W" for x in xticks])
    ax_map.set_yticklabels([f"{y:.1f}° N" for y in yticks])
    ax_map.grid(True, linestyle='--', alpha=0.5)
    ax_map.set_aspect('equal', adjustable='box')
    # add colorbar
    cbar = plt.colorbar(contourf_zoomed, ax=ax_map, orientation='vertical', pad=0.02, shrink=0.95)
    cbar.set_label('Depth [m]')
    ticks = [-5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
    cbar.set_ticks(ticks) 

except:
    # Simple fallback if geopandas not available
    for station in deep_stations:
        if station['latitude'] is not None and station['longitude'] is not None:
            ax_map.scatter(station['longitude'], station['latitude'], 
                         s=70, color='dodgerblue', edgecolor='w')
    ax_map.set_title(f'A) 64PE529 CTD stations (depth < -{DEPTH_LIMIT}m)')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.grid(True, linestyle='--', alpha=0.5)

    
# Density profiles (top right)
ax1.set_xlabel('Density [σ-t, kg/m³]')
ax1.set_ylabel('Depth [m]')
ax1.set_title('B) 64PE529 observations: density profiles with water masses')
ax1.invert_yaxis()
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(density_min-0.5, density_max+0.5)
ax1.set_ylim(None, 0)
# y ticks: negative numbers (depths from 0 to -2500 with 500m intervals)
ax1.set_yticklabels([0, -500, -1000, -1500, -2000, -2500])


# Add legend
for i in range(NUM_WATER_MASSES):
    ax1.scatter([], [], color=colors[i], label=f'Water mass {i+1}')
ax1.legend(loc='lower left')

# Zoomed profile (middle)
ax_zoom.set_xlabel('Density [σ-t, kg/m³]')
ax_zoom.set_ylabel('Depth [m]')
ax_zoom.set_title(f'C) 64PE529 observations: density profiles (surface to -{ZOOM_DEPTH}m)')
ax_zoom.invert_yaxis()
ax_zoom.set_ylim(ZOOM_DEPTH, 0)
ax_zoom.grid(True, linestyle='--', alpha=0.3)
ax_zoom.set_xlim(density_min-0.5, density_max+0.5)
ax_zoom.set_yticklabels([0, -50, -100, -150, -200, -250])

# T-S diagram without SCARIBOS (bottom left)
ax2.set_xlabel('Salinity [PSU]')
ax2.set_ylabel('Potential temperature [°C]')
ax2.set_title('D) 64PE529 observations: T-S diagram with 5-95% boundaries')
ax2.grid(True, linestyle='--', alpha=0.3)

# T-S diagram with SCARIBOS (bottom right)
if scaribos_salinity is not None and scaribos_temperature is not None:
    ax_scaribos.scatter(scaribos_salinity, scaribos_temperature, 
                       s=2, color='k', alpha=1, label='SCARIBOS')

# Add density contours for SCARIBOS plot
cs_scaribos = ax_scaribos.contour(SAL, TEMP, DENSITY, 
                                 colors='gray', linestyles='--', alpha=0.5)
ax_scaribos.clabel(cs_scaribos, inline=1, fontsize=8, fmt='%.2f')

ax_scaribos.set_xlabel('Salinity [PSU]')
ax_scaribos.set_ylabel('Potential temperature [°C]')
ax_scaribos.set_title('E) SCARIBOS (black) and 64PE529 (colored) T-S diagram ')
ax_scaribos.grid(True, linestyle='--', alpha=0.3)

# Save figure
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'CH2_Fig1_water_masses.jpeg'), 
            dpi=300, bbox_inches='tight')


# Save deep stations list
deep_stations_info = []
for station in deep_stations:
    if station['latitude'] is not None and station['longitude'] is not None:
        deep_stations_info.append({
            'station': station['station'],
            'max_depth': station['max_depth'],
            'latitude': station['latitude'],
            'longitude': station['longitude']
        })

if deep_stations_info:
    df = pd.DataFrame(deep_stations_info)
    df.to_csv(os.path.join(OUTPUT_DIR, 'deep_stations_list.csv'), index=False)
    print(f"\nDeep stations list saved to '{OUTPUT_DIR}/deep_stations_list.csv'")
# %%
