'''
Project: 3D flow and volume transport around Curaçao. 

Script to select transects across the island (perpendicular to the coast) 
and calculate alongshore and cross-shore component of the velocity field there 

Author: V Bertoncelj
'''
#%%
# load libraries
import numpy as np
import xarray as xr
import xroms
import os

# Configuration parameters
angle_deg = 315
deflection = (-100, 180)
island_point_lon = -68.8
island_point_lat = 12.1
config = 'SCARIBOS_V8'

# Output directory for saved data
output_dir = 'croco_regimes'
os.makedirs(output_dir, exist_ok=True)

# Months to loop through - full dataset
sim_months = [
    'Y2020M04', 'Y2020M05', 'Y2020M06', 
    'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
    'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 
    'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
    'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 
    'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
    'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 
    'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
    'Y2024M01', 'Y2024M02', 'Y2024M03'
]

# For testing, uncomment the line below to process only one month
# sim_months = ['Y2020M04']

def rotate_coordinates(x, y, angle_deg, center):
    """Rotate coordinates by a given angle around a center."""
    angle_rad = np.deg2rad(angle_deg)
    x_defl, y_defl = deflection
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad) + x_defl
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad) + y_defl
    return x_rot, y_rot

def process_velocity_data(sim_month):
    """Process velocity data for a single month and return transect data."""
    print(f'Processing {sim_month}...')
    
    # File paths
    file_path_u = f'../../croco/CONFIG/{config}/CROCO_FILES/croco_avg_{sim_month}.nc'
    file_path_v = f'../../croco/CONFIG/{config}/CROCO_FILES/surface_currents/{sim_month}_v.nc'
    
    # Check if files exist
    if not (os.path.exists(file_path_u) and os.path.exists(file_path_v)):
        print(f"Warning: Files not found in {file_path_u} or {file_path_v}. Skipping {sim_month}.")

        return None, None
    
    # Open with xroms
    dsu = xroms.open_netcdf(file_path_u, chunks={'time': 1})
    ds, xgrid = xroms.roms_dataset(dsu, include_cell_volume=True)
    ds.xroms.set_grid(xgrid)
    
    # Find center point
    center = xroms.argsel2d(ds.lon_rho, ds.lat_rho, island_point_lon, island_point_lat)
    
    # Calculate rotated coordinates
    x = ds['xi_rho'][50:250]
    y = ds['eta_rho'][100:300]
    x_rot, y_rot = rotate_coordinates(x, y, angle_deg, center)
    
    # Interpolate bathymetry to rotated coordinates
    h_rot = ds['h'].interp(xi_rho=x_rot, eta_rho=y_rot)
    
    # Interpolate velocity field to rho points
    u_rho = xroms.to_grid(ds.u, xgrid, hcoord='rho')
    v_rho = xroms.to_grid(ds.v, xgrid, hcoord='rho')
    w_rho = xroms.to_grid(ds.w, xgrid, hcoord='rho')
    
    # Rotate velocity field to along/cross-shore components
    along_rot, cross_rot = xroms.vector.rotate_vectors(
        u_rho, v_rho, angle_deg, isradians=False, 
        reference='compass', xgrid=None, hcoord='rho', attrs=None
    )
    
    # Calculate time average
    along_rot = along_rot.cf.mean('time')
    cross_rot = cross_rot.cf.mean('time')
    w_transect = w_rho.cf.mean('time')
    u_rho_plot = u_rho.mean('time')
    v_rho_plot = v_rho.mean('time')
    
    # Extract surface values
    along_rot_surf = along_rot.cf.isel(s_rho=-1)
    cross_rot_surf = cross_rot.cf.isel(s_rho=-1)
    u_rho_plot_surf = u_rho_plot.cf.isel(s_rho=-1)
    v_rho_plot_surf = v_rho_plot.cf.isel(s_rho=-1)
    
    # Extract transect data
    transect = 100
    along_rot_transect = along_rot.interp(xi_rho=x_rot[transect], eta_rho=y_rot[transect])
    cross_rot_transect = cross_rot.interp(xi_rho=x_rot[transect], eta_rho=y_rot[transect])
    w_transect_interp = w_transect.interp(xi_rho=x_rot[transect], eta_rho=y_rot[transect])
    
    # Surface transect data
    along_rot_surf_transect = along_rot_surf.interp(xi_rho=x_rot[transect], eta_rho=y_rot[transect])
    cross_rot_surf_transect = cross_rot_surf.interp(xi_rho=x_rot[transect], eta_rho=y_rot[transect])
    
    return along_rot_surf_transect, cross_rot_surf_transect



def save_transect_data(sim_month, along_rot_surf_transect, cross_rot_surf_transect):
    """Save transect data as .npy files."""
    # Save alongshore velocity
    along_filename = os.path.join(output_dir, f'{sim_month}_along_shore_velocity.npy')
    np.save(along_filename, along_rot_surf_transect.values)
    
    # Save cross-shore velocity
    cross_filename = os.path.join(output_dir, f'{sim_month}_cross_shore_velocity.npy')
    np.save(cross_filename, cross_rot_surf_transect.values)
    
    print(f'Saved data: {along_filename} and {cross_filename}')

# Main processing loop
def main():
    """Main processing function."""
    print(f"Processing {len(sim_months)} months...")
    print(f"Output directory: {output_dir}")
    
    for sim_month in sim_months:
        try:
            # Process velocity data
            along_rot_surf_transect, cross_rot_surf_transect = process_velocity_data(sim_month)
            
            if along_rot_surf_transect is not None and cross_rot_surf_transect is not None:
                # Save transect data
                save_transect_data(sim_month, along_rot_surf_transect, cross_rot_surf_transect)
                print(f'Successfully processed {sim_month}')
            else:
                print(f'Skipped {sim_month} due to missing data')
                
        except Exception as e:
            print(f'Error processing {sim_month}: {str(e)}')
            continue
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
# %%
