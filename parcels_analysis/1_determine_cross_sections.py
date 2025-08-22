'''
Project: 3D flow and volume transport around Curaçao. 

In this script we define and plot the cross-sections used in the analysis of the particle trajectories.
We have 5 cross-sections, oriented at 45 degrees.
Names:
- NS: North Section (in final manuscript: North-of-Curacao NC)
- WP: West Point
- MP: Mid Point
- KC: Klein Curacao
- SS: South Section (in final manuscript: South-of-Curacao SC)

Author: V Bertoncelj
kernel: parcels_shared
'''

#%%
# Import libraries
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from scipy.interpolate import griddata
from geopy.distance import geodesic
import dask.array as da
from dask.diagnostics import ProgressBar

# ====================== DEFINED SECTIONS =========================
sections = {
    "NS": [(12.03, -69.83), (12.93, -68.93)],
    "WP": [(11.73, -69.77), (12.93, -68.57)],
    "MP": [(11.47, -69.63), (12.93, -68.17)],
    "KC": [(11.49, -69.16), (12.84, -67.81)],
    "SS": [(11.41, -68.80), (12.40, -67.81)],
}
# ==================================================================

# bathymetry
config = 'SCARIBOS_V8'
path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid       = xr.open_dataset(path + 'croco_grd.nc')

bathymetry = grid.h.values
lon = grid.lon_rho.values
lat = grid.lat_rho.values
land        = np.where(grid.mask_rho == 0, 1, np.nan)
land        = np.where(grid.mask_rho == 1, np.nan, land)

# define the release location edges: indices in SCARIBOS grid
x_south = 100 # SOUTH
y_east  = 270 # EAST
y_north = 300 # NORTH
x_west  = 40  # WEST

# index boarders (we need x and y indices)
south_ymin = 218
south_ymax = y_east
east_xmin  = x_south
east_xmax  = y_north
north_xmin = x_west
north_xmax = 270
west_xmin  = 219
west_xmax  = y_north
lon_south = lon[x_south, south_ymin:south_ymax]
lat_south = lat[x_south, south_ymin:south_ymax]
lon_north = lon[y_north, north_xmin:north_xmax]
lat_north = lat[y_north, north_xmin:north_xmax]-0.01
lon_west = lon[west_xmin:west_xmax, x_west]+0.0001
lat_west = lat[west_xmin:west_xmax, x_west]
lon_east = lon[east_xmin:east_xmax, y_east]-0.01
lat_east = lat[east_xmin:east_xmax, y_east]

# calculate angle for each: --> make sure all have 45 deg. angle !
for name, coords in sections.items():
    lat1, lon1 = coords[0]
    lat2, lon2 = coords[1]
    angle = np.arctan2(lat2 - lat1, lon2 - lon1)
    angle = np.degrees(angle)
    print(f"{name}: {angle:.2f} degrees")

# %%
color_seeding = 'blue'

# FIGURE
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(bathymetry, extent=[grid.lon_rho.min(), grid.lon_rho.max(), 
                            grid.lat_rho.min(), grid.lat_rho.max()], 
        origin='lower', cmap='Greys_r', alpha=0.5)
ax.pcolormesh(grid.lon_rho, grid.lat_rho, land, cmap='Greys', alpha=1, zorder = 10)


ax.hlines(y=lat_south, xmin=lon_south.min(), xmax=lon_south.max(), color=color_seeding, lw=2)
ax.hlines(y=lat_north, xmin=lon_north.min(), xmax=lon_north.max(), color=color_seeding, lw=2)
ax.vlines(x=lon_west, ymin=lat_west.min(), ymax=lat_west.max(), color=color_seeding, lw=2)
ax.vlines(x=lon_east, ymin=lat_east.min(), ymax=lat_east.max(), color=color_seeding, lw=2, label = 'Seeding positions')

for i, (name, coords) in enumerate(sections.items()):
    x_coords = [coords[0][1], coords[1][1]]
    y_coords = [coords[0][0], coords[1][0]]
    ax.plot(x_coords, y_coords, marker='o', color='dodgerblue', linewidth=4)

ax.legend(loc='upper right')
ax.grid(which='both', linestyle='--', linewidth=0.5)
ax.set_xticks(np.arange(-70, -66, 0.5))
ax.set_yticks(np.arange(10, 14, 0.5))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°W'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}°N'))
ax.set_xlim(-70.5, -67.5)
ax.set_ylim(10.6, 13.3)

plt.savefig('figures/methodology/cross_sections.png', dpi=300, bbox_inches='tight')


# %%
