'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot particle crossings and segmentation for one example month - figure in the manuscript.
Needed to run the script:
- parcels (INFLOW4B4M) output needs to alredy exist (to plot crossings)
- bathymetry from ETOPO - data/bathy_etopo2.nc
- bathymetry from GEBCO and Pelagia merged - data/gebco_and_pelagia_merged_SCARIBOS_V2.nc
- shapefile of Curacao as shapefile for plotting (CUW_adm0.shp), found at www.gadm.org, contributor: OCHA Field Information Services Section (FISS), available publicly
- calculated crossings and segmentations (run scripts: 2_calc... and 3_calc...)

Author: V Bertoncelj
'''

#%%
# Import libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from geopy.distance import geodesic
import matplotlib as mpl
import matplotlib.cm as cm

# example month: 
part_month = 'Y2021M04'

# Load the bathymetry data from ETOPO
etopo_data = xr.open_dataset('data/data_large_files/bathy_etopo2.nc')
bathymetry = etopo_data['z']
bathymetry_subregion = bathymetry.sel(latitude=slice(8.5, 16), longitude=slice(-73, -60))

# Load SCARIBOS grid data 
config = 'SCARIBOS_V8'
path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid = xr.open_dataset(path + 'croco_grd.nc')
lon = grid.lon_rho.values
lat = grid.lat_rho.values

# Define border indices - parcels simulation (seeding locations)
x_south = 100
y_east = 270
y_north = 300
x_west = 40
south_ymin = 218
south_ymax = y_east
east_xmin = x_south
east_xmax = y_north
north_xmin = x_west
north_xmax = 270
west_xmin = 219
west_xmax = y_north
lon_south = lon[x_south, south_ymin:south_ymax]
lat_south = lat[x_south, south_ymin:south_ymax]
lon_north = lon[y_north, north_xmin:north_xmax]
lat_north = lat[y_north, north_xmin:north_xmax]-0.01
lon_west = lon[west_xmin:west_xmax, x_west]+0.0001
lat_west = lat[west_xmin:west_xmax, x_west]
lon_east = lon[east_xmin:east_xmax, y_east]-0.01
lat_east = lat[east_xmin:east_xmax, y_east]

# Configuration (load parcels output INFLOW4B4M)


sections = {
    "NS": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SS": [(11.41, -68.80), (12.40, -67.81)],
}
section_plotting = sections

# Create linearly spaced coordinates along each cross-section
n_points = 300 
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

# SCARIBOS: for interpolation of bathymetry
file_path = f'~/croco/CONFIG/{config}/CROCO_FILES/croco_avg_{part_month}.nc'
ds_scarib = xr.open_dataset(file_path)
mask_rho = ds_scarib.mask_rho.where(ds_scarib.mask_rho == 0)
lon_rho = ds_scarib.lon_rho.values
lat_rho = ds_scarib.lat_rho.values
h = -ds_scarib['h'].values  # Bathymetry (negative depth)
points = np.column_stack((lat_rho.ravel(), lon_rho.ravel()))
values = h.ravel()

# Interpolate bathymetry along each cross-section
cross_section_bathymetry_KC = griddata(points, values, (lats_KC, lons_KC), method='linear')
cross_section_bathymetry_WP = griddata(points, values, (lats_WP, lons_WP), method='linear')
cross_section_bathymetry_MP = griddata(points, values, (lats_MP, lons_MP), method='linear')
cross_section_bathymetry_SS = griddata(points, values, (lats_SS, lons_SS), method='linear')
cross_section_bathymetry_NS = griddata(points, values, (lats_NS, lons_NS), method='linear')

# pick all crossings that have distance_from_start < 1000 km and depth < 1000 m
def make_segments(all_crossings, min_distance, max_distance, min_depth, max_depth):
    segments = all_crossings.where(
        (all_crossings.distance_from_start >= min_distance) & 
        (all_crossings.distance_from_start < max_distance) & 
        (all_crossings.crossing_depth >= min_depth) & 
        (all_crossings.crossing_depth < max_depth), 
        drop=True
    )
    return segments

def load_all_crossings(section_one):
    """Load all crossing data from NetCDF files for all sections using dask and combine them into one xarray Dataset."""
    datasets = []
    for section in section_one:
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

# depth limits - for all cross-sections the same
depth1 = -162
depth2 = -458.5

# %%
# Create segments for NS section
section = 'NS'
all_crossings = load_all_crossings(section_one=[section])
all_crossings = all_crossings.set_coords('trajectory')
vline1, vline2, vline3, vline4, vline8 = 13, 29.5, 66, 102, 140
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


#%%
# Create segments for WP section
section = 'WP'
all_crossings = load_all_crossings(section_one=[section])
all_crossings = all_crossings.set_coords('trajectory')
vline1, vline2, vline3, vline4, mid_island, vline5, vline6, vline7, vline8 = 22.5, 41.3, 88.7, 92, 98, 103.2, 105.9, 146.4, 187
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


# %%
# Create segments for MP section
section = 'MP'
all_crossings = load_all_crossings(section_one=[section])
all_crossings = all_crossings.set_coords('trajectory')
vline1, vline2, vline3, vline4 = 34.5, 47.5, 96.3, 99
mid_island, vline5, vline6, vline7, vline8 = 106.5, 115, 117.8, 172, 227
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

#%%
# Create segments for KC section
section = 'KC'
all_crossings = load_all_crossings(section_one=[section])
all_crossings = all_crossings.set_coords('trajectory')
vline1, vline2, vline3, vline4, mid_island, vline5, vline6, vline7, vline8 = 13, 26.8, 72, 74.5, 77.4, 80.2, 82.7, 121, 210
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

#%%
# Create segments for SS section
section = 'SS'
all_crossings = load_all_crossings(section_one=[section])
all_crossings = all_crossings.set_coords('trajectory')
vline1, vline2, vline3, vline4, vline8 = 10.5, 20.8, 57.5, 94.5, 154.5
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

# %%
# figure settings
font0 = 16
font1 = 14
square_color = 'darkorange' 
square_cur_color = 'k'
square_linewidth = 4 
square_cur_linewidth = 1
color1 = 'cornflowerblue'
color2 = 'royalblue'
color4 = 'tomato'
color3 = 'coral'
color6 = 'teal'
color5 = 'lightseagreen'

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

# FIGURE:
fig = plt.figure(figsize=(13, 20))

gs = GridSpec(6, 2, height_ratios=[2.9, 1.4, 1.4, 1.4, 1.4, 1.4], width_ratios=[4, 0.05], hspace=0.3, wspace=0.0001)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
contourf = ax1.contourf(bathymetry_subregion['longitude'], bathymetry_subregion['latitude'], 
                        bathymetry_subregion, levels, cmap=cmap2_bl, vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree(), extend='min', rasterized=True)

ax1.add_feature(cfeature.LAND, zorder=3, color='saddlebrown', alpha=0.4)
gridlines = ax1.gridlines(draw_labels=False, zorder=1, linewidth=0.5)
# aspect ratio equal
ax1.set_aspect('equal')
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlabels_bottom = False
gridlines.ylabels_left = False
ax1.set_yticks(np.arange(8, 18, 1))
ax1.set_xticks(np.arange(-76, -58, 1))
ax1.set_yticklabels(['{:.0f}° N'.format(abs(lat)) for lat in ax1.get_yticks()], fontsize=10)
ax1.set_xticklabels(['{:.0f}° W'.format(abs(lon)) for lon in ax1.get_xticks()], fontsize=10)
# aspect equal
ax1.set_aspect('equal', adjustable='box')
# x lim
ax1.set_xlim(-73, -64.6)
ax1.set_ylim(10.5, 13.5)

cbar_ax = fig.add_subplot(gs[0, 1])
box = cbar_ax.get_position()
cbar_ax.set_position([0.92, 0.7, 0.015, 0.18])#[box.x0 - box.width*0.5, box.y0, box.width*0.45, box.height])
cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='vertical', label='Depth [m]', shrink=0.9)
cbar.set_label('Depth [m]', fontsize=10)
ticks = [-5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
cbar.set_ticks(ticks)
cbar.ax.tick_params(labelsize=10)

color_release = 'k'
ax1.plot(lon_south, lat_south, color=color_release, linewidth=square_linewidth, zorder=3, label='South')
ax1.plot(lon_east, lat_east, color=color_release, linewidth=square_linewidth, zorder=3, label='East')
ax1.plot(lon_north, lat_north, color=color_release, linewidth=square_linewidth, zorder=3, label='North')
ax1.plot(lon_west, lat_west, color=color_release, linewidth=square_linewidth, zorder=3, label='West')

ax1.set_title('A) Map of cross-sections', fontsize=14)#, fontweight='bold')

sections = {
    "NC": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SC": [(11.41, -68.80), (12.40, -67.81)],
}

colors_map = ['b', 'b', 'b', 'b', 'b']

# Define different positions along each line (0 = start, 1 = end)
label_positions = [0.2, 0.27, 0.3, 0.17, 0.16]  # Each section gets a different position

for i, (section_key, coords) in enumerate(sections.items()):
    lats = [coords[0][0], coords[1][0]]
    lons = [coords[0][1], coords[1][1]]
    ax1.plot(lons, lats, color=colors_map[i], linewidth=4,
             label=section_key, transform=ccrs.PlateCarree())
    
    # Calculate label position based on the specified fraction along the line
    t = label_positions[i]  # Position fraction (0 to 1)
    label_lat = lats[0] + t * (lats[1] - lats[0])
    label_lon = lons[0] + t * (lons[1] - lons[0])
    
    # Add section labels at the calculated positions
    ax1.text(label_lon, label_lat, section_key, fontsize=12, fontweight='bold',
             ha='center', va='center', transform=ccrs.PlateCarree(),
             bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8))

sections = {
    "NS": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SS": [(11.41, -68.80), (12.40, -67.81)],
}
# NS ------------------------------------
mp_distance = np.linspace(0, geodesic(section_plotting["NS"][0], section_plotting["NS"][1]).kilometers, n_points)
vline1 = 13
vline2 = 29.5
vline3 = 66
vline4 = 102
vline8 = 140

# All subsequent plots span both columns to maintain their width
ax = fig.add_subplot(gs[1, 0])
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
ax.set_ylabel('Depth [m]', fontsize = 10)
#

ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)

ax.plot(mp_distance, cross_section_bathymetry_NS, color='saddlebrown', lw=0.5)
ax.fill_between(mp_distance, cross_section_bathymetry_NS, -4500, color='saddlebrown', alpha=0.4)

ax.set_ylim(-2500, 200)
ax.set_xlim(0, 140)
ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)
ax.text(vline1/2, -100, 'NC_1D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline1 + vline2)/2, -100, 'NC_2D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -100, 'NC_3D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -100, 'NC_4D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -100, 'NC_5D1', fontsize=8, ha='center', va='center', color='k')

ax.text((vline1 + vline2)/2, -350, 'NC_2D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -350, 'NC_3D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -350, 'NC_4D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -350, 'NC_5D2', fontsize=8, ha='center', va='center', color='k')

ax.text((vline2 + vline3)/2, -600, 'NC_3D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -600, 'NC_4D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -600, 'NC_5D3', fontsize=8, ha='center', va='center', color='k')

ax.text(0.02, 0.05, 'B) North-of-Curaçao (NC)', fontsize=14, ha='left', va='bottom', transform=ax.transAxes, color='k', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# WP ------------------------------------
vline1 = 22.5
vline2 = 41.3
vline3 = 88.7
vline4 = 92
mid_island = 98
vline5 = 103.2
vline6 = 105.9
vline7 = 146.4
vline8 = 187
mp_distance = np.linspace(0, geodesic(section_plotting["WP"][0], section_plotting["WP"][1]).kilometers, n_points)

ax = fig.add_subplot(gs[2, 0])

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

ax.plot(mp_distance, cross_section_bathymetry_WP, color='saddlebrown', lw=0.5)
ax.fill_between(mp_distance, cross_section_bathymetry_WP, -4500, color='saddlebrown', alpha=0.4)
ax.set_ylim(-2800, 200)
ax.set_xlim(0,vline8)
ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)

ax.text(vline1/2, -100, 'WP_1D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline1 + vline2)/2, -100, 'WP_2D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -100, 'WP_3D1', fontsize=8, ha='center', va='center', color='k')
ax.annotate('WP_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-10, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('WP_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-3, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('WP_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+3, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('WP_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+10, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.text((vline6 + vline7)/2, -100, 'WP_8D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -100, 'WP_9D1', fontsize=8, ha='center', va='center', color='k')

ax.text((vline1 + vline2)/2, -350, 'WP_2D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -350, 'WP_3D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + mid_island)/2-3, -350, 'WP_4D2', fontsize=8, ha='center', va='center', color='k')
ax.text((mid_island + vline5)/2+3, -350, 'WP_7D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline6 + vline7)/2, -350, 'WP_8D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -350, 'WP_9D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -800, 'WP_3D3', fontsize=8, ha='center', va='center', color='k')

ax.text((vline6 + vline7)/2, -800, 'WP_8D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -800, 'WP_9D3', fontsize=8, ha='center', va='center', color='k')
ax.text(0.02, 0.05, 'C) West Point (WP)', fontsize=14, ha='left', va='bottom', transform=ax.transAxes, color='k', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# MP ------------------------------------
vline1 = 34.5
vline2 = 47.5
vline3 = 96.3
vline4 = 99
mid_island = 106.5
vline5 = 115
vline6 = 117.8
vline7 = 172
vline8 = 227
mp_distance = np.linspace(0, geodesic(section_plotting["MP"][0], section_plotting["MP"][1]).kilometers, n_points)

ax = fig.add_subplot(gs[3, 0])
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

ax.plot(mp_distance, cross_section_bathymetry_MP, color='saddlebrown', lw=0.5)
ax.fill_between(mp_distance, cross_section_bathymetry_MP, -4500, color='saddlebrown', alpha=0.4)
ax.set_ylim(-3400, 200)
ax.set_xlim(0, vline8)
ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)

# add section names in the plot
ax.text(vline1/2, -100, 'MP_1D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline1 + vline2)/2, -100, 'MP_2D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -100, 'MP_3D1', fontsize=8, ha='center', va='center', color='k')
ax.annotate('MP_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-10, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('MP_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-3, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('MP_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+3, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('MP_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+10, 240),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.text((vline6 + vline7)/2, -100, 'MP_8D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -100, 'MP_9D1', fontsize=8, ha='center', va='center', color='k')

ax.text((vline1 + vline2)/2, -350, 'MP_2D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -350, 'MP_3D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + mid_island)/2-3, -350, 'MP_4D2', fontsize=8, ha='center', va='center', color='k')
ax.text((mid_island + vline5)/2+3, -350, 'MP_7D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline6 + vline7)/2, -350, 'MP_8D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -350, 'MP_9D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -800, 'MP_3D3', fontsize=8, ha='center', va='center', color='k')

ax.text((vline6 + vline7)/2, -800, 'MP_8D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -800, 'MP_9D3', fontsize=8, ha='center', va='center', color='k')
ax.text(0.02, 0.05, 'D) Mid-Point (MP)', fontsize=14, ha='left', va='bottom', transform=ax.transAxes, color='k', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# KC ------------------------------------
vline1 = 13
vline2 = 26.8
vline3 = 72
vline4 = 74.5
mid_island = 77.4
vline5 = 80.2
vline6 = 82.7
vline7 = 121
vline8 = 210
mp_distance = np.linspace(0, geodesic(section_plotting["KC"][0], section_plotting["KC"][1]).kilometers, n_points)
ax = fig.add_subplot(gs[4, 0])

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

# ax.set_xlabel('Distance from Venezuela (mainland) [km]')
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

ax.plot(mp_distance, cross_section_bathymetry_KC, color='saddlebrown', lw=0.5)
ax.fill_between(mp_distance, cross_section_bathymetry_KC, -4600, color='saddlebrown', alpha=0.4)
ax.set_ylim(-4600, 200)
ax.set_xlim(0, vline8)

ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)

# add section names in the plot
ax.text(vline1/2, -100, 'KC_1D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline1 + vline2)/2, -100, 'KC_2D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -100, 'KC_3D1', fontsize=8, ha='center', va='center', color='k')
ax.annotate('KC_4D1', xy=((vline3 + vline4)/2, -50), xytext=((vline3 + vline4)/2-15, 400),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('KC_5D1', xy=((vline4 + mid_island)/2, -50), xytext=((vline4 + mid_island)/2-5, 400),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('KC_6D1', xy=((mid_island + vline5)/2, -50), xytext=((mid_island + vline5)/2+5, 400),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.annotate('KC_7D1', xy=((vline5 + vline6)/2, -50), xytext=((vline5 + vline6)/2+15, 400),
            fontsize=8, ha='center', va='center', color='black',
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            rotation=10)
ax.text((vline6 + vline7)/2, -100, 'KC_8D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -100, 'KC_9D1', fontsize=8, ha='center', va='center', color='k')

ax.text((vline1 + vline2)/2, -350, 'KC_2D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -350, 'KC_3D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + mid_island)/2-3, -350, 'KC_4D2', fontsize=8, ha='center', va='center', color='k')
ax.text((mid_island + vline5)/2+3, -350, 'KC_7D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline6 + vline7)/2, -350, 'KC_8D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -350, 'KC_9D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -800, 'KC_3D3', fontsize=8, ha='center', va='center', color='k')

ax.text((vline6 + vline7)/2, -800, 'KC_8D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline7 + vline8)/2, -800, 'KC_9D3', fontsize=8, ha='center', va='center', color='k')

ax.text(0.02, 0.05, 'E) Klein Curaçao (KC)', fontsize=14, ha='left', va='bottom', transform=ax.transAxes, color='k', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# SS ------------------------------------
vline1 = 10.5
vline2 = 20.8
vline3 = 57.5
vline4 = 94.5
vline8 = 154.5
mp_distance = np.linspace(0, geodesic(section_plotting["SS"][0], section_plotting["SS"][1]).kilometers, n_points)

ax = fig.add_subplot(gs[5, 0])
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

ax.set_xlabel('Distance from Venezuela (mainland) [km]', fontsize=10)
ax.set_ylabel('Depth [m]')

ax.axhline(depth1, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axhline(depth2, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline1, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline2, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline3, color='w', linestyle='--', alpha=1, linewidth=0.5)
ax.axvline(vline4, color='w', linestyle='--', alpha=1, linewidth=0.5)


ax.plot(mp_distance, cross_section_bathymetry_SS, color='saddlebrown', lw=0.5)
ax.fill_between(mp_distance, cross_section_bathymetry_SS, -4500, color='saddlebrown', alpha=0.4)
ax.set_ylim(-3500, 200)
ax.set_xlim(0,vline8)

ax.grid(color='w', linestyle='--', linewidth=0.3, alpha=0.3)
ax.text(vline1/2, -100, 'SC_1D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline1 + vline2)/2, -100, 'SC_2D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -100, 'SC_3D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -100, 'SC_4D1', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -100, 'SC_5D1', fontsize=8, ha='center', va='center', color='k')

ax.text((vline1 + vline2)/2, -350, 'SC_2D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline2 + vline3)/2, -350, 'SC_3D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -350, 'SC_4D2', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -350, 'SC_5D2', fontsize=8, ha='center', va='center', color='k')

ax.text((vline2 + vline3)/2, -600, 'SC_3D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline3 + vline4)/2, -600, 'SC_4D3', fontsize=8, ha='center', va='center', color='k')
ax.text((vline4 + vline8)/2, -600, 'SC_5D3', fontsize=8, ha='center', va='center', color='k')
ax.text(0.02, 0.05, 'F) South-of-Curaçao (SC)', fontsize=14, ha='left', va='bottom', transform=ax.transAxes, color='k', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Replace the colorbar section at the end of your code with this:

# Create colorbar for b-f figures with striped patterns for each depth range
# Get the position of the bottom 5 subplots to align the colorbar
bottom_subplot_bottom = fig.get_axes()[-1].get_position().y0  # Bottom of subplot F (last subplot)
top_subplot_top = fig.get_axes()[1].get_position().y1       # Top of subplot B (first of the 5)

# Calculate the full height of subplots B-F
full_height = top_subplot_top - bottom_subplot_bottom

# Position the colorbar to span only from bottom of F to top of B
cbar_ax = fig.add_axes([0.92, bottom_subplot_bottom, 0.015, full_height-0.22])

# Define depth boundaries and colors for each range (0 at top, -1000 at bottom visible)
# We need to reverse the order for the colormap but keep depth values correct
depth_boundaries = [0, -162, -458.5, -1000]#, -458.5, -162, 0]  # Reversed for colormap (deepest to shallowest)
depth_labels = [ '-1000', '-458.5', '-162', '0']  # Labels in correct order (shallow to deep)
colors_d1 = ['cornflowerblue', 'royalblue']  # D1 colors (alternating)
colors_d2 = ['coral', 'tomato']              # D2 colors (alternating)  
colors_d3 = ['lightseagreen', 'teal']        # D3 colors (alternating)

# Create a custom colormap with the stripe patterns
import matplotlib.colors as mcolors

# Create stripe pattern for each segment
n_stripes_per_segment = 10
total_colors = []
boundaries = []

# Build the colormap with alternating stripes for each depth range
current_boundary = 0
for i, (start_depth, end_depth) in enumerate(zip(depth_boundaries[:-1], depth_boundaries[1:])):
    # Determine colors for this depth range
    if i == 0:  # Surface layer (0 to -162m)
        range_colors = colors_d3
    elif i == 1:  # Intermediate layer (-162 to -458.5m)
        range_colors = colors_d2
    else:  # Deep layer (-458.5 to -1000m)
        range_colors = colors_d1
    
    # Add alternating stripes for this segment
    for j in range(n_stripes_per_segment):
        color_idx = j % 2
        total_colors.append(range_colors[color_idx])
        boundaries.append(current_boundary + j)
    
    current_boundary += n_stripes_per_segment

# Add final boundary
boundaries.append(current_boundary)

# Create the custom colormap
custom_cmap = mcolors.ListedColormap(total_colors)
norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

# Create a dummy mappable for the colorbar

mappable = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
mappable.set_array([])

# Create the colorbar with extend='min'
cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='vertical', extend='min')

# Set the tick positions and labels to correspond to depth values
tick_positions = [0, n_stripes_per_segment, 2*n_stripes_per_segment, 3*n_stripes_per_segment]
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(depth_labels)

# Customize the colorbar appearance
cbar.set_label('Crossing depth [m]', rotation=90, fontsize=10)
cbar.ax.tick_params(labelsize=10)

# Continue with your existing save command
fig.tight_layout()
fig.savefig('figures/methodology/CH2_Fig4_segments_crossings.jpeg', dpi=300, bbox_inches='tight')


# %%
