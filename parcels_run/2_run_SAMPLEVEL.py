'''
Project: 3D flow and volume transport around Curaçao. 

In this script we run parcels with the original set-up for this project, but each particle is only run for 1 time step.
Each particle output stores the local velocity of the particle at the time of release,
so that we can use it later for the analysis of the particle trajectories (to calculate volume transport).
In our case we run 3 months at once (3 months of seeding of particles). 
Each particle has a lifetime of 5min for this specific configuration, as we are only interested at velocities at t=0.

Author: V Bertoncelj
Kernel: parcels-dev-local
'''

import sys
import numpy as np
from glob import glob
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta as delta
from datetime import timedelta
import datetime
from datetime import datetime, timedelta
import parcels
from parcels import JITParticle, FieldSet, Variable, ParticleSet


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    year  = int(sys.argv[1])
    month = int(sys.argv[2])
    part_month    = f'Y{year}M{str(month).zfill(2)}'

    part_config   = 'SAMPLEVEL'
    seeding_dt    = 24    # in hours
    seeding_start = 0     # in hours
    adv_period    = 4     # in months
    ipart_dt      = 5*60  # in seconds (= 5 min)

    print(f"Start running 2_run_{part_config}.py...")
    print("Hydrodynamics from: SCARIBOS_V8 croco model")
    print(f"Month running (this is starting month): {part_month}")

    dir = '~/croco/CONFIG/SCARIBOS_V8/CROCO_FILES/'

    def generate_filenames(month):
        filenames = []
        for i in range(6):
            month_idx = int(month[6:8]) - 1 + i
            year = int(month[1:5])
            if month_idx > 11:
                year += 1
            month_idx %= 12
            month_str = str(month_idx + 1).zfill(2)
            filename = f"{dir}croco_avg_Y{year}M{month_str}.nc"
            filenames.append(filename)
        return filenames

    filenames = generate_filenames(part_month)
    print("Filenames taken for current simulation:", part_month)
    for filename in filenames:
        print(filename)

    # Load particle positions
    X_masked_south = np.load(f'INPUT/particles_lon_ALL.npy', allow_pickle=True)
    Y_masked_south = np.load(f'INPUT/particles_lat_ALL.npy', allow_pickle=True)
    lon_part_south = X_masked_south.flatten()
    lat_part_south = Y_masked_south.flatten()
    lon_part_south = lon_part_south[~np.isnan(lon_part_south)]  # delete nans
    lat_part_south = lat_part_south[~np.isnan(lat_part_south)]
    depth_part_south = np.load(f'INPUT/particles_depth_ALL.npy', allow_pickle=True)
    npart          = len(lon_part_south)
    print(f"Number of particles released: {npart}")

    days        = 31 if month in [1, 3, 5, 7, 8, 10] else 30 if month != 2 else 28
    days_month2 = 31 if month + 1 in [1, 3, 5, 7, 8, 10] else 30 if month + 1 != 2 else 28
    days_month3 = 31 if month + 2 in [1, 3, 5, 7, 8, 10] else 30 if month + 2 != 2 else 28

    days += days_month2 + days_month3
    days_toprint  = days
    time_releases = [(day - 1) * 24 * 60 * 60 + hour * 60 * 60 for day in range(1, days + 1) for hour in range(0, 24, 24)]
    time_releases = [time + seeding_start * 60 * 60 for time in time_releases]

    print(f"Number of days in the month: {days_toprint}")
    print(f"Number of time releases: {len(time_releases)}")

    lons  = np.tile(lon_part_south, len(time_releases))
    lats  = np.tile(lat_part_south, len(time_releases))
    Z     = np.tile(depth_part_south, len(time_releases))
    times = np.repeat(time_releases, len(lon_part_south))

    # variables, dimension, indices, fieldset
    variables = {"U": "u", "V": "v", "W": "w", "H": "h","Zeta": "zeta", "Cs_w": "Cs_w"}

    lon_rho = "lon_rho"
    lat_rho = "lat_rho"
    time    = "time"

    dimensions = {
        "time": "time",
        "U": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
        "V": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
        "W": {"lon": lon_rho, "lat": lat_rho, "depth": "s_w", "time": "time"},
        "H": {"lon": lon_rho, "lat": lat_rho},
        "Zeta": {"lon": lon_rho, "lat": lat_rho, "time": "time"},
        "Cs_w": {"depth": "s_w"}
    }

    # define indices (if particle goes beyond these indices, it will be 'out of bounds', so deleted/frozen)
    indices = {
        "lon": range(40, 270),
        "lat": range(100,300)
    }

    fieldset = parcels.FieldSet.from_croco(
        filenames,
        variables,
        dimensions,
        indices=indices,
        gridindexingtype="croco",
        hc=200 # in SCARIBOS it is 200 everywhere
    )

    SampleParticle = parcels.JITParticle.add_variables(
        [
            parcels.Variable("U", dtype=np.float32, initial=np.nan, to_write="once"), 
            parcels.Variable("V", dtype=np.float32, initial=np.nan, to_write="once"),
            parcels.Variable("W", dtype=np.float32, initial=np.nan, to_write="once"),
            parcels.Variable('particle_age', dtype=np.float32, initial=0.)
        ]
    )

    # freeze if out of bounds
    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 49:
            particle.delete()

    def UpdateAge(particle, fieldset, time):
        if particle.time > 0:
            particle.particle_age += particle.dt

    def SampleVel_correct(particle, fieldset, time): # with this kernel we sample the velocity at the particle location
        particle.U, particle.V, particle.W = fieldset.UVW[
            time, particle.depth, particle.lat, particle.lon, particle
        ]

    def DeleteOldParticles(particle, fieldset, time): # we stop the particle after one internal time step (=5min)
        if particle.particle_age > 5*60:  
            particle.delete()
    
    outputdt = 24*3600
    runtime  = 93 * 24 * 3600
    print(f"Runtime: {runtime / (24 * 3600)} days")

    pset = ParticleSet(
        fieldset=fieldset, pclass=SampleParticle, lon=lons, lat=lats, depth=Z, time=times
    )

    outputfile = pset.ParticleFile(
        name=f"{part_config}/{part_config}_starting_{part_month}.zarr", outputdt=outputdt, chunks=(int(len(pset) / 100), int(np.ceil(30 * 24 * 3600 / (100 * 3600))))
    )
    pset.execute(
        [parcels.AdvectionRK4_3D,UpdateAge, SampleVel_correct,DeleteParticle,DeleteOldParticles], runtime= runtime, dt=ipart_dt, output_file=outputfile
    )
