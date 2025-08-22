'''
Project: 3D flow and volume transport around Curaçao. 

In this script we define the locations of release of particles in the horizontal and vertical space.
The locations are defined in the horizontal space as lines in the model grid, enclosing the rectangualr area,
and in the vertical space as a set of depths (based on bathyemtry in SCARIBOS).

Author: V Bertoncelj
'''

#%%
# Import libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo

# load SCARIBOS for bathymetry
month     = 'Y2020M04' # example month
config    = 'SCARIBOS_V8'
file_path = f'~/croco/CONFIG/{config}/CROCO_FILES/croco_avg_{month}.nc'
ds        = xr.open_dataset(file_path)

# load SCARIBOS grid information
path       = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid       = xr.open_dataset(path + 'croco_grd.nc')
bathymetry = grid.h.values
lon        = grid.lon_rho.values
lat        = grid.lat_rho.values
land       = np.where(grid.mask_rho == 0, 1, np.nan)
land       = np.where(grid.mask_rho == 1, np.nan, land)

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

# plot bathyemtry and particle release lines (for check-up)
fig, ax = plt.subplots(figsize=(11,8))
ax.contourf(-bathymetry, levels=100, cmap=cmo.ice)
cbar = plt.colorbar(ax.contourf(-bathymetry, levels=100, cmap=cmo.ice), ax=ax)
cbar.set_label('Bathymetry [m]')
cbar.set_ticks(np.arange(-5000, 0, 1000))
plt.contourf(land, color='brown', levels=100)
plt.grid()
plt.hlines(y=x_south, xmin=south_ymin, xmax=south_ymax, color='red', linewidth=2) # south
plt.vlines(x=y_east, ymin=east_xmin, ymax=east_xmax, color='red', linewidth=2)  # east
plt.hlines(y=y_north, xmin=north_xmin, xmax=north_xmax, color='red', linewidth=2)  # north
plt.vlines(x=x_west, ymin=west_xmin, ymax=west_xmax, color='red', linewidth=2)  # west
ax.set_aspect('equal', 'box')
plt.xlim([0, 320])
plt.ylim([20, 320])
plt.title('Release locations of particles')
plt.tight_layout()


# %%
# create a list of all lon and lat coordinates that correspond 
# to the above hlines and vlines (which are in the space of eta and xi)
lon_south = lon[x_south, south_ymin:south_ymax]
lat_south = lat[x_south, south_ymin:south_ymax]
lon_north = lon[y_north, north_xmin:north_xmax]
lat_north = lat[y_north, north_xmin:north_xmax]-0.01
lon_west  = lon[west_xmin:west_xmax, x_west]+0.0001
lat_west  = lat[west_xmin:west_xmax, x_west]
lon_east  = lon[east_xmin:east_xmax, y_east]-0.01
lat_east  = lat[east_xmin:east_xmax, y_east]

# extract bathymetry at these locations
bathymetry_south = -bathymetry[x_south, south_ymin:south_ymax]
bathymetry_north = -bathymetry[y_north, north_xmin:north_xmax]
bathymetry_west  = -bathymetry[west_xmin:west_xmax, x_west]
bathymetry_east  = -bathymetry[east_xmin:east_xmax, y_east]


# %%
# generate smooth depth points (of particles) with logharitmically increasing spacing

def generate_smooth_depths(max_depth=-5000, num_points=200, method='quadratic'):
    """
    Generate depth points with gradually increasing spacing and calculate dz.
    Parameters:
    - max_depth: Maximum depth (negative)
    - num_points: Number of depth levels
    - method: 'quadratic' (default) or 'logarithmic' for different spacing styles
    Returns:
    - depth_loc: array of depths from surface (-0.5) to max_depth
    - dz: array of dz values (same length as depth_loc)
    """
    if method == 'quadratic':
        spacing = np.linspace(0, 1, num_points) ** 2.2  # Quadratic increase
    elif method == 'logarithmic':
        spacing = np.logspace(-2, 0, num_points) - 0.01  # Logarithmic, shift to avoid zero
    
    spacing   = spacing / spacing.max()  # Normalize to [0, 1]
    depth_loc = max_depth * spacing  # Scale to depth range
    depth_loc = np.concatenate(([-0.5], depth_loc))  # Ensure surface point at -0.5
    depth_loc = np.unique(np.clip(depth_loc, max_depth, -0.5))  # Clip to valid range
    
    # Calculate dz
    dz = np.zeros_like(depth_loc)
    dz[1:-1] = (depth_loc[2:] - depth_loc[:-2]) / 2  # dz[i] = (d[i+1] - d[i-1])/2
    dz[0] = (depth_loc[1] - depth_loc[0])  # First dz
    dz[-1] = (depth_loc[-1] - depth_loc[-2])  # Last dz
    
    print(f"depth_loc[1]: {depth_loc[1]}")
    print(f"depth_loc[-1]: {depth_loc[-1]}")
    print(f"depth_loc[0]: {depth_loc[0]}")
    print(f"depth_loc[-2]: {depth_loc[-2]}")
    print(f"depth_loc[-3]: {depth_loc[-3]}")
    
    return depth_loc, dz


depth_smooth, dz = generate_smooth_depths(-4800, num_points=50, method='logarithmic')
depth_smooth = np.flip(depth_smooth)
dz = np.flip(dz)

# Plot spacing
plt.figure(figsize=(6,4))
plt.plot(depth_smooth[:-1], np.diff(depth_smooth), marker='o', markersize=2)
plt.xlabel("Depth (m)")
plt.ylabel("Spacing (m)")
plt.title("Smooth Depth Spacing")
plt.grid()

# Plot dz
plt.figure(figsize=(6,4))
plt.plot(depth_smooth, dz, marker='o', markersize=2)
plt.xlabel("Depth (m)")
plt.ylabel("dz (m)")
plt.title("Calculated dz Values")
plt.grid()

# Calculate horizontal spacing
dx_south = 0.01 * 111000 * np.cos(np.radians(np.mean(lat_south)))  # in meters
dx_north = 0.01 * 111000 * np.cos(np.radians(np.mean(lat_north)))  # in meters
dy_east = np.diff(lat_east) * 111000  # in meters
dy_west = np.diff(lat_west) * 111000  # in meters
print(f"dx_south: {dx_south:.2f} m")
print(f"dx_north: {dx_north:.2f} m")
print(f"dy_east: {dy_east.mean():.2f} m")
print(f"dy_west: {dy_west.mean():.2f} m")

# Add one more value to match the dimensions (using the last value)
dy_east = np.append(dy_east, dy_east[-1])
dy_west = np.append(dy_west, dy_west[-1])

#%%
# Function to adjust dz for particles near the seabed based on bathymetry
def adjust_dz_for_bathymetry(depth, dz, bathymetry):
    """
    Adjust dz for particles near the bathymetry.
    Parameters:
    - depth: Depth of the particle (negative value)
    - dz: Original dz value for this depth
    - bathymetry: Bathymetry depth at this location (negative value)
    Returns:
    - Adjusted dz value that accounts for bathymetry
    """
    
    half_dz = dz / 2 # Half of dz extends below the particle
    dist_to_bathymetry = depth - bathymetry # Distance from particle to bathymetry
    
    # If the particle is more than half dz from bathymetry, return original dz
    if dist_to_bathymetry >= half_dz:
        return dz
    
    # If the particle is within half dz of bathymetry, adjust dz
    # The new dz will be the distance to bathymetry + half dz above the particle
    if dist_to_bathymetry > 0:
        return dist_to_bathymetry + half_dz
    
    return half_dz


# Adjust dz and calculate areas for each location:

# South border
lon_south_par   = []
lat_south_par   = []
depth_south_par = []
idx_south_par   = []
area_south_par  = []

for i in range(len(lon_south)):
    h_loc = bathymetry_south[i]
    max_depth = h_loc
    # Filter depths that are greater than or equal to the bathymetry
    valid_depths = np.array([j for j, d in enumerate(depth_smooth) if d >= max_depth])
    
    if len(valid_depths) > 0:
        depth_indices = valid_depths
        depth_loc = depth_smooth[depth_indices]
        dz_loc = dz[depth_indices]
        
        # Adjust dz for particles near bathymetry
        adjusted_dz = np.array([adjust_dz_for_bathymetry(d, dz_val, max_depth) 
                               for d, dz_val in zip(depth_loc, dz_loc)])
        
        # Calculate area for this location
        area_loc = dx_south * adjusted_dz
        
        lon_loc = np.tile(lon_south[i], len(depth_loc))
        lat_loc = np.tile(lat_south[i], len(depth_loc))
        idx_loc = np.full(len(depth_loc), i)
        
        lon_south_par.append(lon_loc)
        lat_south_par.append(lat_loc)
        depth_south_par.append(depth_loc)
        idx_south_par.append(idx_loc)
        area_south_par.append(area_loc)

lon_south_par   = np.concatenate(lon_south_par)
lat_south_par   = np.concatenate(lat_south_par)
depth_south_par = np.concatenate(depth_south_par)
idx_south_par   = np.concatenate(idx_south_par)
area_south_par  = np.concatenate(area_south_par)

# East border
lon_east_par = []
lat_east_par = []
depth_east_par = []
idx_east_par = []
area_east_par = []

for i in range(len(lat_east)):
    h_loc = bathymetry_east[i]
    max_depth = h_loc
    # Filter depths that are greater than or equal to the bathymetry
    valid_depths = np.array([j for j, d in enumerate(depth_smooth) if d >= max_depth])
    
    if len(valid_depths) > 0:
        depth_indices = valid_depths
        depth_loc = depth_smooth[depth_indices]
        dz_loc = dz[depth_indices]
        
        # Adjust dz for particles near bathymetry
        adjusted_dz = np.array([adjust_dz_for_bathymetry(d, dz_val, max_depth) 
                               for d, dz_val in zip(depth_loc, dz_loc)])
        
        # Calculate area for this location
        area_loc = dy_east[i] * adjusted_dz
        
        lon_loc = np.tile(lon_east[i], len(depth_loc))
        lat_loc = np.tile(lat_east[i], len(depth_loc))
        idx_loc = np.full(len(depth_loc), i)
        
        lon_east_par.append(lon_loc)
        lat_east_par.append(lat_loc)
        depth_east_par.append(depth_loc)
        idx_east_par.append(idx_loc)
        area_east_par.append(area_loc)

lon_east_par = np.concatenate(lon_east_par)
lat_east_par = np.concatenate(lat_east_par)
depth_east_par = np.concatenate(depth_east_par)
idx_east_par = np.concatenate(idx_east_par)
area_east_par = np.concatenate(area_east_par)

# North border
lon_north_par   = []
lat_north_par   = []
depth_north_par = []
idx_north_par   = []
area_north_par  = []

for i in range(len(lon_north)):
    h_loc = bathymetry_north[i]
    max_depth = h_loc
    # Filter depths that are greater than or equal to the bathymetry
    valid_depths = np.array([j for j, d in enumerate(depth_smooth) if d >= max_depth])
    
    if len(valid_depths) > 0:
        depth_indices = valid_depths
        depth_loc = depth_smooth[depth_indices]
        dz_loc = dz[depth_indices]
        
        # Adjust dz for particles near bathymetry
        adjusted_dz = np.array([adjust_dz_for_bathymetry(d, dz_val, max_depth) 
                               for d, dz_val in zip(depth_loc, dz_loc)])
        
        # Calculate area for this location
        area_loc = dx_north * adjusted_dz
        
        lon_loc = np.tile(lon_north[i], len(depth_loc))
        lat_loc = np.tile(lat_north[i], len(depth_loc))
        idx_loc = np.full(len(depth_loc), i)
        
        lon_north_par.append(lon_loc)
        lat_north_par.append(lat_loc)
        depth_north_par.append(depth_loc)
        idx_north_par.append(idx_loc)
        area_north_par.append(area_loc)

lon_north_par = np.concatenate(lon_north_par)
lat_north_par = np.concatenate(lat_north_par)
depth_north_par = np.concatenate(depth_north_par)
idx_north_par = np.concatenate(idx_north_par)
area_north_par = np.concatenate(area_north_par)

# West border
lon_west_par   = []
lat_west_par   = []
depth_west_par = []
idx_west_par   = []
area_west_par  = []

for i in range(len(lat_west)):
    h_loc = bathymetry_west[i]
    max_depth = h_loc
    # Filter depths that are greater than or equal to the bathymetry
    valid_depths = np.array([j for j, d in enumerate(depth_smooth) if d >= max_depth])
    
    if len(valid_depths) > 0:
        depth_indices = valid_depths
        depth_loc = depth_smooth[depth_indices]
        dz_loc = dz[depth_indices]
        
        # Adjust dz for particles near bathymetry
        adjusted_dz = np.array([adjust_dz_for_bathymetry(d, dz_val, max_depth) 
                               for d, dz_val in zip(depth_loc, dz_loc)])
        
        # Calculate area for this location
        area_loc = dy_west[i] * adjusted_dz
        
        lon_loc = np.tile(lon_west[i], len(depth_loc))
        lat_loc = np.tile(lat_west[i], len(depth_loc))
        idx_loc = np.full(len(depth_loc), i)
        
        lon_west_par.append(lon_loc)
        lat_west_par.append(lat_loc)
        depth_west_par.append(depth_loc)
        idx_west_par.append(idx_loc)
        area_west_par.append(area_loc)

lon_west_par = np.concatenate(lon_west_par)
lat_west_par = np.concatenate(lat_west_par)
depth_west_par = np.concatenate(depth_west_par)
idx_west_par = np.concatenate(idx_west_par)
area_west_par = np.concatenate(area_west_par)


# plot (check the areas at the seabed)
fig, ax = plt.subplots(4, 1, figsize=(10, 14))

# South border plot
ax[0].plot(lon_south, bathymetry_south, color='k')
ax[0].fill_between(lon_south, bathymetry_south, -900, color='grey', alpha=0.5)
sc0 = ax[0].scatter(lon_south_par, depth_south_par, c=area_south_par, cmap='viridis', s=2)
plt.colorbar(sc0, ax=ax[0], label='Area (m²)')
ax[0].set_title('South border')
ax[0].set_xlabel('Longitude')
ax[0].set_ylim([-900, 0])
ax[0].set_xlim([lon_south.min(), lon_south.max()])

# East border plot
ax[1].plot(lat_east, bathymetry_east, color='k')
ax[1].fill_between(lat_east, bathymetry_east, -4800, color='grey', alpha=0.5)
sc1 = ax[1].scatter(lat_east_par, depth_east_par, c=area_east_par, cmap='viridis', s=2)
plt.colorbar(sc1, ax=ax[1], label='Area (m²)')
ax[1].set_title('East border')
ax[1].set_xlabel('Latitude')
ax[1].set_ylim([-4800, 0])
ax[1].set_xlim([lat_east.min(), lat_east.max()])

# North border plot
ax[2].plot(lon_north, bathymetry_north, color='k')
ax[2].fill_between(lon_north, bathymetry_north, -3500, color='grey', alpha=0.5)
sc2 = ax[2].scatter(lon_north_par, depth_north_par, c=area_north_par, cmap='viridis', s=2)
plt.colorbar(sc2, ax=ax[2], label='Area (m²)')
ax[2].set_title('North border')
ax[2].set_xlabel('Longitude')
ax[2].set_ylim([-3500, 0])
ax[2].set_xlim([lon_north.min(), lon_north.max()])

# West border plot
ax[3].plot(lat_west, bathymetry_west, color='k')
ax[3].fill_between(lat_west, bathymetry_west, -1000, color='grey', alpha=0.5)
sc3 = ax[3].scatter(lat_west_par, depth_west_par, c=area_west_par, cmap='viridis', s=2)
plt.colorbar(sc3, ax=ax[3], label='Area (m²)')
ax[3].set_title('West border')
ax[3].set_xlabel('Latitude')
ax[3].set_ylim([-1000, 0])
ax[3].set_xlim([lat_west.min(), lat_west.max()])

for i in range(4):
    ax[i].set_ylabel('Depth [m]')
    ax[i].grid()

plt.tight_layout()

# %%
# some calcualtion for check-up
# count all particles, boarder by boarder
n_south = len(lon_south_par)
n_east  = len(lon_east_par)
n_north = len(lon_north_par)
n_west  = len(lon_west_par)
print(f"Number of particles released at the South border: {n_south}")
print(f"Number of particles released at the East border: {n_east}")
print(f"Number of particles released at the North border: {n_north}")
print(f"Number of particles released at the West border: {n_west}")

# calculate and print total area for each border
total_area_south = np.sum(area_south_par)
total_area_east  = np.sum(area_east_par)
total_area_north = np.sum(area_north_par)
total_area_west  = np.sum(area_west_par)
print(f"Total area at the South border: {total_area_south:.2f} m^2")
print(f"Total area at the East border: {total_area_east:.2f} m^2")
print(f"Total area at the North border: {total_area_north:.2f} m^2")
print(f"Total area at the West border: {total_area_west:.2f} m^2")

# Calculate area adjustments due to bathymetry
original_area_south = dx_south * np.sum(dz[np.array([j for j, d in enumerate(depth_smooth) if d >= bathymetry_south.min()])])
original_area_east = np.mean(dy_east) * np.sum(dz[np.array([j for j, d in enumerate(depth_smooth) if d >= bathymetry_east.min()])])
original_area_north = dx_north * np.sum(dz[np.array([j for j, d in enumerate(depth_smooth) if d >= bathymetry_north.min()])])
original_area_west = np.mean(dy_west) * np.sum(dz[np.array([j for j, d in enumerate(depth_smooth) if d >= bathymetry_west.min()])])

area_reduction_south = (original_area_south - total_area_south) / original_area_south * 100
area_reduction_east = (original_area_east - total_area_east) / original_area_east * 100
area_reduction_north = (original_area_north - total_area_north) / original_area_north * 100
area_reduction_west = (original_area_west - total_area_west) / original_area_west * 100

print(f"Area reduction due to bathymetry at South border: {area_reduction_south:.2f}%")
print(f"Area reduction due to bathymetry at East border: {area_reduction_east:.2f}%")
print(f"Area reduction due to bathymetry at North border: {area_reduction_north:.2f}%")
print(f"Area reduction due to bathymetry at West border: {area_reduction_west:.2f}%")

# Calculate average depth for each border
avg_depth_south = np.mean(depth_south_par)
avg_depth_east = np.mean(depth_east_par)
avg_depth_north = np.mean(depth_north_par)
avg_depth_west = np.mean(depth_west_par)

print(f"Average depth at the South border: {avg_depth_south:.2f} m")
print(f"Average depth at the East border: {avg_depth_east:.2f} m")
print(f"Average depth at the North border: {avg_depth_north:.2f} m")
print(f"Average depth at the West border: {avg_depth_west:.2f} m")

# calculate and print max diff between spacing 
if len(depth_south_par) > 1:
    spacing_south = np.diff(np.sort(depth_south_par))
    max_spacing_south = spacing_south.max()
    print(f"Max spacing between particles at the South border: {max_spacing_south:.2f} m")

if len(depth_east_par) > 1:
    spacing_east = np.diff(np.sort(depth_east_par))
    max_spacing_east = spacing_east.max()
    print(f"Max spacing between particles at the East border: {max_spacing_east:.2f} m")

if len(depth_north_par) > 1:
    spacing_north = np.diff(np.sort(depth_north_par))
    max_spacing_north = spacing_north.max()
    print(f"Max spacing between particles at the North border: {max_spacing_north:.2f} m")

if len(depth_west_par) > 1:
    spacing_west = np.diff(np.sort(depth_west_par))
    max_spacing_west = spacing_west.max()
    print(f"Max spacing between particles at the West border: {max_spacing_west:.2f} m")

# %%
# save particles as npy for each border separately

np.save('INPUT/particles_lon_south.npy', lon_south_par)
np.save('INPUT/particles_lat_south.npy', lat_south_par)
np.save('INPUT/particles_depth_south.npy', depth_south_par)
np.save('INPUT/particles_idx_south.npy', idx_south_par)
np.save('INPUT/particles_area_south.npy', area_south_par)

np.save('INPUT/particles_lon_east.npy', lon_east_par)
np.save('INPUT/particles_lat_east.npy', lat_east_par)
np.save('INPUT/particles_depth_east.npy', depth_east_par)
np.save('INPUT/particles_idx_east.npy', idx_east_par)
np.save('INPUT/particles_area_east.npy', area_east_par)

np.save('INPUT/particles_lon_north.npy', lon_north_par)
np.save('INPUT/particles_lat_north.npy', lat_north_par)
np.save('INPUT/particles_depth_north.npy', depth_north_par)
np.save('INPUT/particles_idx_north.npy', idx_north_par)
np.save('INPUT/particles_area_north.npy', area_north_par)

np.save('INPUT/particles_lon_west.npy', lon_west_par)
np.save('INPUT/particles_lat_west.npy', lat_west_par)
np.save('INPUT/particles_depth_west.npy', depth_west_par)
np.save('INPUT/particles_idx_west.npy', idx_west_par)
np.save('INPUT/particles_area_west.npy', area_west_par)

# add all particles together (concatenate)
lon_all = np.concatenate([lon_south_par, lon_east_par, lon_north_par, lon_west_par])
lat_all = np.concatenate([lat_south_par, lat_east_par, lat_north_par, lat_west_par])
depth_all = np.concatenate([depth_south_par, depth_east_par, depth_north_par, depth_west_par])
idx_all = np.concatenate([idx_south_par, idx_east_par, idx_north_par, idx_west_par])
area_all = np.concatenate([area_south_par, area_east_par, area_north_par, area_west_par])

# save all particles as npy
np.save('INPUT/particles_lon_ALL.npy', lon_all)
np.save('INPUT/particles_lat_ALL.npy', lat_all)
np.save('INPUT/particles_depth_ALL.npy', depth_all)
np.save('INPUT/particles_idx_ALL.npy', idx_all)
np.save('INPUT/particles_area_ALL.npy', area_all)

# Create a visualization of area distribution
plt.figure(figsize=(10, 6))
plt.hist(area_all, bins=50, alpha=0.7)
plt.xlabel('Area (m²)')
plt.ylabel('Count')
plt.title('Distribution of Particle Areas')
plt.grid(True)
