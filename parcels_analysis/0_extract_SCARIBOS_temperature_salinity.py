"""
Project: 3D flow and volume transport around Curaçao. 

Extraction of salinity and temperature data from SCARIBOS for the time in January 2024 and locations that 
match the locations of the CTD stations. 

Author: V Bertoncelj
"""

#%%
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw

print('Loading data...')
config = 'SCARIBOS_V8'
path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
month = 'Y2024M01'
ds = xr.open_dataset(path + f'croco_avg_{month}.nc')

# average ds over time from 4*24 to 22*24 th time step
ds = ds.isel(time=slice(4*24, 22*24))

# now average all variables over this time
ds_salt = ds.salt.mean(dim='time')
ds_temp = ds.temp.mean(dim='time')


# %%
# coordiantes extracted from CTD stations:
coordinates = """12.1255,-69.1585
12.091666666666667,-69.19916666666667
12.477,-68.752
11.972333333333333,-69.08433333333333
12.019333333333334,-69.0445
12.123333333333333,-69.10333333333334
12.587,-69.3165
12.648333333333333,-69.3695
11.979,-69.1775
12.157,-69.11916666666667
11.9705,-68.94816666666667
12.071833333333334,-69.09716666666667
12.025,-69.13683333333333
12.4365,-68.824
12.396,-68.8965
12.3065,-69.32066666666667
12.306833333333334,-69.23616666666666
12.306833333333334,-69.40316666666666
12.525,-69.26366666666667
12.445666666666666,-69.19766666666666
11.926333333333334,-69.12533333333333
12.125,-69.15866666666666
12.158166666666666,-69.11766666666666
11.865,-69.01033333333334
11.917833333333334,-68.97966666666667
12.0585,-69.24
11.867,-68.559
12.1175,-69.05683333333333"""

coords_list = [line.split(',') for line in coordinates.strip().split('\n')]
stations_df = pd.DataFrame(coords_list, columns=['latitude', 'longitude'])
stations_df['latitude'] = stations_df['latitude'].astype(float)
stations_df['longitude'] = stations_df['longitude'].astype(float)

print(f"Number of stations: {len(stations_df)}")

model_lats = ds.lat_rho.values
model_lons = ds.lon_rho.values

def find_nearest_grid_point(lat, lon, model_lats, model_lons):
    # Calculate distances to all grid points
    if model_lats.ndim == 2:
        # For 2D arrays
        distances = np.sqrt((model_lats - lat)**2 + (model_lons - lon)**2)
    else:
        # For 1D arrays
        lat_diff = np.abs(model_lats - lat)
        lon_diff = np.abs(model_lons - lon)
        distances = np.sqrt(lat_diff**2 + lon_diff**2)
    
    # Find the index of the minimum distance
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    return idx, distances[idx]

# Find the nearest model grid point for each station
stations_df['grid_i'] = None
stations_df['grid_j'] = None
stations_df['model_lat'] = None
stations_df['model_lon'] = None
stations_df['distance'] = None

for i, row in stations_df.iterrows():
    idx, dist = find_nearest_grid_point(row['latitude'], row['longitude'], model_lats, model_lons)
    
    stations_df.at[i, 'grid_i'] = idx[0] if len(idx) > 1 else idx
    stations_df.at[i, 'grid_j'] = idx[1] if len(idx) > 1 else 0
    
    if model_lats.ndim == 2:
        stations_df.at[i, 'model_lat'] = model_lats[idx]
        stations_df.at[i, 'model_lon'] = model_lons[idx]
    else:
        stations_df.at[i, 'model_lat'] = model_lats[idx[0]]
        stations_df.at[i, 'model_lon'] = model_lons[idx[1]]
    
    stations_df.at[i, 'distance'] = dist

# Calculate depths from s-coordinates
# Get average time step for simplicity
time_index = 0  # Using first time step

# Extract necessary variables for sigma coordinate transformation
h = ds.h.values  # Bathymetry
hc = ds.hc.values  # Critical depth
Cs_r = ds.Cs_r.values  # S-coordinate stretching curves at RHO points
s_rho = ds.s_rho.values  # S-coordinate at RHO points

# Vtransform should indicate which vertical transformation is used
vtransform = ds.Vtransform.values
print(f"Vertical transformation type: {vtransform}")

# Function to calculate z-levels from sigma coordinates
def calculate_depth(h, hc, Cs_r, s_rho, zeta=0, vtransform=1):
    N = len(s_rho)
    z = np.zeros((N))
    
    if vtransform == 1:  # Original ROMS transformation
        for k in range(N):
            z0 = (s_rho[k] - Cs_r[k]) * hc + Cs_r[k] * h
            z[k] = z0 + zeta * (1.0 + z0/h)
    elif vtransform == 2:  # New ROMS transformation
        for k in range(N):
            z0 = (hc * s_rho[k] + h * Cs_r[k]) / (hc + h)
            z[k] = zeta + (zeta + h) * z0
    
    return z

# Extract temperature, salinity, and calculate depths for each station
results = []

for i, row in stations_df.iterrows():
    i_idx = int(row['grid_i'])
    j_idx = int(row['grid_j'])
    
    # Extract zeta (free surface) for this location and time step
    if 'zeta' in ds:
        zeta = ds.zeta.isel(time=time_index, eta_rho=i_idx, xi_rho=j_idx).values
    else:
        zeta = 0  # Default if not available
    
    # Extract bathymetry at this location
    station_h = ds.h.isel(eta_rho=i_idx, xi_rho=j_idx).values
    
    # Calculate depths
    depths = calculate_depth(station_h, hc, Cs_r, s_rho, zeta, vtransform)
    
    # Extract temperature and salinity profiles
    temp = ds_temp.isel(eta_rho=i_idx, xi_rho=j_idx).values
    salt = ds_salt.isel(eta_rho=i_idx, xi_rho=j_idx).values
    
    results.append({
        'station_id': i,
        'lat': row['latitude'],
        'lon': row['longitude'],
        'model_lat': row['model_lat'],
        'model_lon': row['model_lon'],
        'distance': row['distance'],
        'temp': temp,
        'salt': salt,
        'depth': depths,
        'bathymetry': station_h
    })

# Function to calculate seawater density (simplified equation)
def calculate_density(temp, salt):
    density = gsw.rho(salt, temp, 0) - 1000  # sigma-t
    return density

# Function to plot T-S diagrams and density profiles
def plot_station_profiles(station_data, station_idx):
    data = station_data[station_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Calculate density
    density = calculate_density(data['temp'], data['salt'])
    
    # Temperature, Salinity, and Density vs Depth
    ax1.plot(data['temp'], data['depth'], 'r-', label='Temperature (°C)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Temperature (°C)')
    
    ax1_salt = ax1.twiny()
    ax1_salt.plot(data['salt'], data['depth'], 'b-', label='Salinity (psu)')
    ax1_salt.set_xlabel('Salinity (psu)')
    
    ax1_density = ax1.twiny()
    ax1_density.plot(density, data['depth'], 'g-', label='Density (kg/m³)')
    ax1_density.spines['top'].set_position(('outward', 60))
    ax1_density.set_xlabel('Density (kg/m³)')
    
    # Invert y-axis for depth
    ax1.invert_yaxis()
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_salt.get_legend_handles_labels()
    lines3, labels3 = ax1_density.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower right')
    
    # T-S diagram with density contours
    # Create density contours
    S, T = np.meshgrid(np.linspace(34, 37, 100), np.linspace(15, 30, 100))
    D = calculate_density(T, S)
    
    CS = ax2.contour(S, T, D, 10, colors='gray', alpha=0.5)
    ax2.clabel(CS, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot the data points colored by depth
    scatter = ax2.scatter(data['salt'], data['temp'], c=data['depth'], cmap='viridis')
    ax2.set_xlabel('Salinity (psu)')
    ax2.set_ylabel('Temperature (°C)')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Depth (m)')
    
    plt.suptitle(f'Station {station_idx}: Lat {data["lat"]:.4f}, Lon {data["lon"]:.4f}\n'
                f'Model Point: Lat {data["model_lat"]:.4f}, Lon {data["model_lon"]:.4f}, Distance: {data["distance"]:.6f}°')
    plt.tight_layout()
    
    return fig

# Plot the first station as an example
first_station_fig = plot_station_profiles(results, 0)
plt.show()

# Save profiles for all stations
print("Saving individual station profiles...")
for i in range(len(results)):
    fig = plot_station_profiles(results, i)
    plt.savefig(f'station_{i}_profile.png')
    plt.close(fig)

# Create a comprehensive T-S diagram with all stations
plt.figure(figsize=(12, 10))

# Create density contours
S, T = np.meshgrid(np.linspace(34.7592-0.5, 37.2881+0.5, 100), np.linspace(3.9084-5, 28.6345+5, 100))
D = calculate_density(T, S)
CS = plt.contour(S, T, D, 10, colors='gray', linestyles='--', alpha=0.5)
plt.clabel(CS, inline=True, fontsize=8, fmt='%.1f')

# Plot each station with a different color
for i, data in enumerate(results):
    plt.plot(data['salt'], data['temp'],color = 'grey', linewidth = 0.5, alpha=0.7, label=f'Station {i}')
    plt.scatter(data['salt'], data['temp'], s=10, color = 'k', alpha=0.7, label=f'Station {i}')

plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.title('T-S diagram - 4-22 January 2024 (SCARIBOS)')
plt.savefig('ts_diagram_Y2024M01.png', dpi=300)
plt.show()

# Save the station data to a CSV file
station_summary = []
for i, data in enumerate(results):
    station_summary.append({
        'station_id': i,
        'latitude': data['lat'],
        'longitude': data['lon'],
        'model_latitude': data['model_lat'],
        'model_longitude': data['model_lon'],
        'distance_to_model_point': data['distance'],
        'bathymetry': data['bathymetry'],
        'surface_temp': data['temp'][0],
        'bottom_temp': data['temp'][-1],
        'surface_salt': data['salt'][0],
        'bottom_salt': data['salt'][-1],
        'surface_depth': data['depth'][0],
        'bottom_depth': data['depth'][-1]
    })

summary_df = pd.DataFrame(station_summary)
summary_df.to_csv('station_summary_Y2024M01.csv', index=False)
print("Summary CSV file saved.")
# %%
# save also all data for creating T-S diagram to npy
np.save('SCARIBOS_T_S_Y2024M01.npy', results, allow_pickle=True)
print("Station data saved to .npy file.")

# Load the station data
station_data = np.load('SCARIBOS_T_S_Y2024M01.npy', allow_pickle=True)

# Save as txt (CSV-like, one row per station, arrays as semicolon-separated strings)
with open('SCARIBOS_T_S_Y2024M01.txt', 'w') as f:
    # Write header
    f.write('station_id,lat,lon,model_lat,model_lon,distance,bathymetry,temp,salt,depth\n')
    for i, data in enumerate(station_data):
        temp_str = ';'.join(map(str, data['temp']))
        salt_str = ';'.join(map(str, data['salt']))
        depth_str = ';'.join(map(str, data['depth']))
        f.write(f"{i},{data['lat']},{data['lon']},{data['model_lat']},{data['model_lon']},{data['distance']},{data['bathymetry']},{temp_str},{salt_str},{depth_str}\n")

# Convert to DataFrame for easier handling
station_df = pd.DataFrame([{
    'station_id': i,
    'lat': data['lat'],
    'lon': data['lon'],
    'model_lat': data['model_lat'],
    'model_lon': data['model_lon'],
    'distance': data['distance'],
    'temp': data['temp'],
    'salt': data['salt'],
    'depth': data['depth'],
    'bathymetry': data['bathymetry']
} for i, data in enumerate(station_data)])

#%%
# Plot T-S diagram with all stations
plt.figure(figsize=(12, 10))
S, T = np.meshgrid(np.linspace(34.7592-0.5, 37.2881+0.5, 100), np.linspace(3.9084-5, 28.6345+5, 100))
D = calculate_density(T, S)
CS = plt.contour(S, T, D, 10, colors='gray', linestyles='--', alpha=0.5)
plt.clabel(CS, inline=True, fontsize=8, fmt='%.1f')
for i, data in enumerate(station_df.itertuples()):
    plt.plot(data.salt, data.temp, color='grey', linewidth=0.5, alpha=0.7, label=f'Station {i}')
    plt.scatter(data.salt, data.temp, s=10, color='k', alpha=0.7, label=f'Station {i}')
plt.xlabel('Salinity (psu)')
plt.ylabel('Temperature (°C)')
plt.grid(True, linestyle='--', alpha=0.3)

# %%
