'''
Project: 3D flow and volume transport around Curaçao. 

In this script we plot particle release lcoations at each boarder and the map of these boarders together with 
the location of particles for selected month and time (as an example).

Needed to run the script:
- parcels (INFLOW4B4M) output needs to alredy exist
- bathymetry from ETOPO - data/bathy_etopo2.nc
- bathymetry from GEBCO and Pelagia merged - data/gebco_and_pelagia_merged_SCARIBOS_V2.nc
- shapefile of Curacao as shapefile for plotting (CUW_adm0.shp), found at www.gadm.org, contributor: OCHA Field Information Services Section (FISS), available publicly

Author: V Bertoncelj
'''

#%%
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import geopandas as gpd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load the shapefile of Curacao
shapefile_path = 'data/CUW_adm0.shp'
land = gpd.read_file(shapefile_path)

# Load the bathymetry data (combined GEBCO and Pelagia)
bathy_gebpel_file = 'data/data_large_files/gebco_and_pelagia_merged_SCARIBOS_V2.nc'
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

# Load the bathymetry data from ETOPO
etopo_data = xr.open_dataset('data/data_large_files/bathy_etopo2.nc')
bathymetry = etopo_data['z']
bathymetry_subregion = bathymetry.sel(latitude=slice(8.5, 16), longitude=slice(-73, -60))

# Load SCARIBOS grid data
config = 'SCARIBOS_V8'
path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid = xr.open_dataset(path + 'croco_grd.nc')
bathymetry_grid = grid.h.values
lon = grid.lon_rho.values
lat = grid.lat_rho.values

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
bathymetry_south = -bathymetry_grid[x_south, south_ymin:south_ymax]
bathymetry_north = -bathymetry_grid[y_north, north_xmin:north_xmax]
bathymetry_west = -bathymetry_grid[west_xmin:west_xmax, x_west]
bathymetry_east = -bathymetry_grid[east_xmin:east_xmax, y_east]
lon_south_par = np.load('INPUT/particles_lon_south.npy')
lat_south_par = np.load('INPUT/particles_lat_south.npy')
depth_south_par = np.load('INPUT/particles_depth_south.npy')
area_south_par = np.load('INPUT/particles_area_south.npy')
lon_east_par = np.load('INPUT/particles_lon_east.npy')
lat_east_par = np.load('INPUT/particles_lat_east.npy')
depth_east_par = np.load('INPUT/particles_depth_east.npy')
area_east_par = np.load('INPUT/particles_area_east.npy')
lon_north_par = np.load('INPUT/particles_lon_north.npy')
lat_north_par = np.load('INPUT/particles_lat_north.npy')
depth_north_par = np.load('INPUT/particles_depth_north.npy')
area_north_par = np.load('INPUT/particles_area_north.npy')
lon_west_par = np.load('INPUT/particles_lon_west.npy')
lat_west_par = np.load('INPUT/particles_lat_west.npy')
depth_west_par = np.load('INPUT/particles_depth_west.npy')
area_west_par = np.load('INPUT/particles_area_west.npy')

# Example particle trajectories
part_month    = 'Y2020M04'
part_config   = 'INFLOW4B4M'
try:
    ds = xr.open_zarr(f"/nethome/berto006/transport_in_3D_project/parcels_run/{part_config}/{part_config}_starting_{part_month}.zarr")
    ds = ds.isel(trajectory=slice(0, 19293*2))
    lon_traj = ds.isel(obs=48)['lon'].values
    lat_traj = ds.isel(obs=48)['lat'].values
    z_traj = ds.isel(obs=48)['z'].values
    colors = np.full(z_traj.shape, 'lightseagreen', dtype=object)
    colors[(z_traj > -458.5) & (z_traj <= -162)] = 'tomato'
    colors[(z_traj > -162) & (z_traj <= 0)] = 'cornflowerblue'
    trajectories_loaded = True
except:
    print("Trajectory data not found. Will skip trajectory plotting.")
    trajectories_loaded = False


#%%
# plot

# colors and fonts
font0 = 16
font1 = 14
font3 =12
color_release = 'k'
square_color = 'darkorange' 
square_cur_color = 'k'
square_linewidth = 4 
square_cur_linewidth = 1
release_colors = ['cornflowerblue', 'tomato', 'lightseagreen']
release_bounds = [0, -162, -458.5, -4800]  # 0 to -162, -162 to -458.5, -458.5 to -4800
release_colors = release_colors[::-1] # Because BoundaryNorm expects increasing bounds, reverse both lists
release_bounds = release_bounds[::-1]
cmap_releases = ListedColormap(release_colors)
norm_releases = BoundaryNorm(release_bounds, cmap_releases.N)

# FIGURE
fig = plt.figure(figsize=(14, 15))
gs = gridspec.GridSpec(4, 4, height_ratios=[3, 1, 1, 1], width_ratios=[1, 1, 1, 1])

# Top plot: Southern Caribbean bathymetry
ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
contourf = ax1.contourf(bathymetry_subregion['longitude'], bathymetry_subregion['latitude'], 
                        bathymetry_subregion, levels, cmap=cmap2_bl, vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree(), extend='min', rasterized=True)
ax1.add_feature(cfeature.LAND, zorder=3, color='saddlebrown', alpha=0.4)#color='saddlebrown', alpha=0.4)
gridlines = ax1.gridlines(draw_labels=False, zorder=1, linewidth=0.5)
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlabels_bottom = False
gridlines.ylabels_left = False
ax1.set_yticks(np.arange(8, 18, 1))
ax1.set_xticks(np.arange(-74, -60, 1))
ax1.set_yticklabels(['{:.0f}° N'.format(abs(lat)) for lat in ax1.get_yticks()], fontsize=14)
ax1.set_xticklabels(['{:.0f}° W'.format(abs(lon)) for lon in ax1.get_xticks()], fontsize=14)
ax1.set_title('A) Map of SCARIBOS model domain (orange) and particle release locations (black)\nwith particle locations in April 2020 after 50 hours of advection', fontsize=16)#, fontweight='bold')
ax1.plot([-70.5, -70.5], [10, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-70.5, -66], [10, 10], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-66, -66], [10, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-70.5, -66], [13.5, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot(lon_south, lat_south, color=color_release, linewidth=square_linewidth, zorder=3, label='South')
ax1.plot(lon_east, lat_east, color=color_release, linewidth=square_linewidth, zorder=3, label='East')
ax1.plot(lon_north, lat_north, color=color_release, linewidth=square_linewidth, zorder=3, label='North')
ax1.plot(lon_west, lat_west, color=color_release, linewidth=square_linewidth, zorder=3, label='West')
# Plot trajectories if available
if trajectories_loaded:
    indices = np.arange(len(lon_traj))
    np.random.shuffle(indices)
    ax1.scatter(lon_traj[indices], lat_traj[indices], s=2, c=colors[indices])
ax1.set_xlim(-70.52-2, -65.98+2)
ax1.set_ylim(9.98-0.2, 13.52+0.2)
# Annotations
ax1.annotate('SCARIBOS model domain', xy=(-69, 13.5), 
             xytext=(-69.2, 13.55), fontsize=font0, color=square_color)
ax1.annotate('V e n e z u e l a', xy=(-69.5, 11.5), 
             xytext=(-68.3, 10.2), fontsize=18, color=square_cur_color)
ax1.annotate('Aruba', xy=(-69.4, 12.5), 
             xytext=(-70.05, 12.67), fontsize=font0, color=square_cur_color)
ax1.annotate('Curaçao', xy=(-68.9, 12.2), 
             xytext=(-69.37, 12.45), fontsize=font0, color='k')
ax1.annotate('Bonaire', xy=(-68.4, 12.1), 
             xytext=(-68.45, 12.35), fontsize=font0, color='k')
ax1.annotate('W border', xy=(-68.3, 12.1), 
             xytext=(-70.25, 12.3), fontsize=font3, color='k',
             fontstyle='italic', rotation=90)
ax1.annotate('E border', xy=(-68.3, 12.1), 
             xytext=(-67.7, 11.8), fontsize=font3, color='k',
             fontstyle='italic', rotation=90)
ax1.annotate('N border', xy=(-68.3, 12.1), 
             xytext=(-69.3, 13), fontsize=font3, color='k',
             fontstyle='italic')
ax1.annotate('S border', xy=(-68.3, 12.1), 
             xytext=(-68.2, 10.85), fontsize=font3, color='k',
             fontstyle='italic')
# Legend for trajectories
if trajectories_loaded:
    handles = [
        plt.scatter([], [], color='cornflowerblue', label='0 to -162 m', s=100),
        plt.scatter([], [], color='tomato', label='-162 to -458.5 m', s=100),
        plt.scatter([], [], color='lightseagreen', label='<-458.5 m', s=100),
    ]
    ax1.legend(handles=handles, loc='lower left', 
               fontsize=16, frameon=False, title='Particle depth:', title_fontsize=16)
# Colorbar for main plot
cbar_ax = fig.add_axes([1, 0.575, 0.015, 0.375])
cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='vertical', label='Depth [m]', shrink=0.9)
cbar.set_label('Depth [m]', fontsize=14)
ticks = [-5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
cbar.set_ticks(ticks)
cbar.ax.tick_params(labelsize=14)

# SUBPLOT 1: Western Border (row 2, left)
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(lat_west, bathymetry_west, color='saddlebrown', linewidth=1)
ax2.fill_between(lat_west, bathymetry_west, -1000,color='saddlebrown', alpha=0.4)# color='k', alpha=0.8)
sc = ax2.scatter(lat_west_par, depth_west_par, c=depth_west_par, 
                cmap=cmap_releases, norm=norm_releases, s=3, alpha=1)#s=3, alpha=0.7, vmin=-4800, vmax=0)
# ax2.set_title('b) Western boarder', fontsize=font1)#, fontweight='bold')
ax2.text(12.35, -850, 'B) Western border', fontsize=18, color='k', ha='center', fontweight='bold')
ax2.set_xlabel('Latitude [°N]', fontsize=font1)
ax2.set_ylabel('Depth [m]', fontsize=font1)
ax2.set_ylim([-1000, 0])
ax2.set_xlim([lat_west.min(), lat_west.max()])
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=14)

# SUBPLOT 2: Southern Border (row 2, right)
ax3 = fig.add_subplot(gs[1, 2:])
ax3.plot(lon_south, bathymetry_south, color='saddlebrown', linewidth=1)
ax3.fill_between(lon_south, bathymetry_south, -1000,color='saddlebrown', alpha=0.4)# color='k', alpha=0.8)
ax3.scatter(lon_south_par, depth_south_par, c=depth_south_par, 
                cmap=cmap_releases, norm=norm_releases, s=3, alpha = 1)
# ax3.set_title('c) Southern border', fontsize=font1)#, fontweight='bold')
ax3.text(-68.18, -850, 'C) Southern border', fontsize=18, color='k', ha='center', fontweight='bold')
ax3.set_xlabel('Longitude [°W]', fontsize=font1)
ax3.set_ylim([-1000, 0])
ax3.set_xlim([lon_south.min(), lon_south.max()])
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=14)
xticks = ax3.get_xticks()
ax3.set_xticklabels([f"{abs(x):.1f}" for x in xticks])

# SUBPLOT 3: Northern Border (entire row 3)
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(lon_north, bathymetry_north, color='saddlebrown', linewidth=1)
ax4.fill_between(lon_north, bathymetry_north, -3500,color='saddlebrown', alpha=0.4)# color='k', alpha=0.8)
ax4.scatter(lon_north_par, depth_north_par, c=depth_north_par, 
                cmap=cmap_releases, norm=norm_releases, s=3, alpha=1)
# ax4.set_title('d) Northern boarder', fontsize=font1)#, fontweight='bold')
ax4.text(-69.82, -3000, 'D) Northern border', fontsize=18, color='k', ha='center', fontweight='bold')
ax4.set_xlabel('Longitude [°W]', fontsize=font1)
ax4.set_ylabel('Depth [m]', fontsize=font1)
ax4.set_ylim([-3500, 0])
ax4.set_xlim([lon_north.min(), lon_north.max()])
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=14)
xticks = ax4.get_xticks()
ax4.set_xticklabels([f"{abs(x):.1f}" for x in xticks])

# SUBPLOT 4: Eastern Border (entire row 4)
ax5 = fig.add_subplot(gs[3, :])
ax5.plot(lat_east, bathymetry_east, color='saddlebrown', linewidth=1)
ax5.fill_between(lat_east, bathymetry_east, -4800,color='saddlebrown', alpha=0.4)# color='k', alpha=0.8)
ax5.scatter(lat_east_par, depth_east_par, c=depth_east_par, 
                cmap=cmap_releases, norm=norm_releases, s=3, alpha=1)
# ax5.set_title('E) Eastern border', fontsize=font1)#, fontweight='bold')
# add text:
ax5.text(11.2, -4000, 'E) Eastern border', fontsize=18, color='k', ha='center', fontweight='bold')
ax5.set_xlabel('Latitude [°N]', fontsize=font1)
ax5.set_ylabel('Depth [m]', fontsize=font1)
ax5.set_ylim([-4800, 0])
ax5.set_xlim([lat_east.min(), lat_east.max()])
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=14)

# Shared colorbar for all particle depth plots
cbar_particles_ax = fig.add_axes([1, 0.043, 0.015, 0.48])
cbar_particles = plt.colorbar(sc, cax=cbar_particles_ax, orientation='vertical', extend='min')
cbar_particles.set_label('Particle Depth [m]', fontsize=14)
cbar_particles.ax.tick_params(labelsize=14)
cbar_particles.set_ticks([0, -162, -458.5, -4800])
cbar_particles.set_ticklabels(['0', '-162', '-458.5', '-1000'])

plt.tight_layout()
ax1.set_aspect('equal', adjustable='box')

plt.savefig('figures/CH2_Fig3_release_locations.jpeg', 
            dpi=300, bbox_inches='tight')

# %%
