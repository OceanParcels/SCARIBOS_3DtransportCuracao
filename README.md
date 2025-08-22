# SCARIBOS_3DtransportCuracao

Repository for Vesna's project: Ocean-to-nearshore circulation patterns around Curaçao: A southern Caribbean reef island exposed to distinct flow regimes.

This project contains the following folders:

## Project Structure:

- [**parcels_run**](.o/parcels_run/): Contains the scripts to run the OceanParcels experiments
- [**parcels_analysis**](./parcels_analysis/): Contains scripts that process and analyze particle trajectory outputs from OceanParcels.

## Hydrodynamic input: SCARIBOS

For this project the hydrodynamic input used is SCARIBOS: South CARIBbean Ocean System, made with CROCO community model.

SCARIBOS is run for the period from December 2019 to March 2024. Period from December 2019 to March 2020 is discarded from further analysis, to account for the spin-up time.

### Information about the model:

- 3D hydrodynamic model
- CROCO community model, version 1.3.1: Auclair, F., Benshila, R., Bordois, L., Boutet, M., Brémond, M., Caillaud, M., Cambon, G., Capet, X., Debreu, L., Ducousso, N., Dufois, F., Dumas, F., Ethé, C., Gula, J., Hourdin, C., Illig, S., Jullien, S., Le Corre, M., Le Gac, S., Le Gentil, S., Lemarié, F., Marchesiello, P., Mazoyer, C., Morvan, G., Nguyen, C., Penven, P., Person, R., Pianezze, J., Pous, S., Renault, L., Roblou, L., Sepulveda, A., and Theetten, S.: Coastal and Regional Ocean COmmunity model (1.3.1), Zenodo [code], https://doi.org/10.5281/zenodo.7415055, 2022.
- Horizontal resolution: 1/100°
- Vertical resolution: 50 sigma-depth layers
- Horizontal extent: Longitudes: 70.5°W to 66.0°W; Latitudes: 10.0°N to 13.5°N
- Bathymetry input: global product GEBCO (version 2023; GEBCO Compilation Group, 2023) and bathymetry around Curaçao obtained using multibeam sonar during RV Pelagia expedition 64PE500
- Oceanographic initial and boundary conditions: GLORYS12V1 (Lellouche et al., 2021) - created using CROCO TOOLS product (V1.3.1)
- Atmospheric forcing: ERA-5 global atmosphere reanalysis (Hersbach et al., 2020)
- River runoff: four rivers Tocuyo, Yaracuy, Tuy and Grande - created using CROCO TOOLS product (V1.3.1)

To learn more about the model inputs and analysis, see repository **OceanParcels/SCARIBOS_ConnectivityCuracao**.

To download the model outputs (surface currents) or request the 3D model outputs, see: **Bertoncelj, Vesna, 2025, "SCARIBOS hydrodynamic model outputs", https://doi.org/10.25850/nioz/7b.b.7h, NIOZ, V3.**

## Lagrangian particle tracking using Parcels: simulations and diagnostics

In this project the Parcels version 3.1.2 is used. Particle tracking simulations are used to simulate the transport pathways of passive tracers across a large domain that includes the islands of Curaçao, Aruba and Bonaire. Virtual particles are released into the study area at 24-hour intervals from April 1, 2020, to March 31, 2024.

The simulated particles represent neutrally buoyant tracers representing the volumetric transport of water masses within the region. Particle deployment occurs across multiple depth levels, extending from the sea surface down to the maximum model depth, using logarithmic vertical spacing to provide higher resolution near the surface where the current is generally the strongest. This vertical arrangement creates depth intervals ranging from approximately 4.3 meters near the surface to 435 meters at the greatest depths. Horizontal particle placement follows a regular grid pattern with 0.01° spacing (approximately 1 km resolution).

### Running the Parcels simulations:

The structure of the scripts, found in [**parcels_run**](./parcels_run/), is as follows:

- `1_particle_release.py`: Determine locations and depths of particle release
- `1_plot_particle_release.py`: Plot particle release locations (**Figure 3 in the manuscript**)
- `2_run_INFLOW4B4M.py`: Script to run the main Parcels experiment - can be executed in parallel for all months. Each simulation includes particle seeding over a 3-month period.
- `2_run_SAMPLEVEL.py`: Script to execute single time-step particle seeding while recording initial particle velocities - required for calculating volume transport per particle.
- `3_calc_VT.py`: Script to calculate volume transport of each particle.
- `4_calc_rampup_multiple_batches.py`: Script to determine ramp-up period (ramp-up period is marked in plots in **Figures 8 and 9**).
- `submit_...`: these are example scripts that are used to submit SLURM jobs to IMAU Lorenz cluster (usually to run in parallel, but not necessary)

### Analyzing the Parcels outputs:

The structure of the scripts, found in [**parcels_analysis**](./parcels_analysis/), is as follows:
- `0_extract_SCARIBOS_temperature_salinity.py`: Script that extracts the salinity and temperature data from SCARIBOS in order to compare the T-S diagram from SCARIBOS with the observations (expedition 64PE529).
- `0_calc_and_plot_water_masses.py`: Script that calculates the water masses around Curaçao from the CTD observations during the RV _Pelagia_  expedition in January 2024 (64PE529). This scripts also plots **Figure 1**.
- `0_calc_regimes_from_croco.py`: Script to select transects across the island (perpendicular to the coast) and calculate alongshore and cross-shore component of the velocity field there.
- `0_plot_regimes_from_croco.py`: Script that plots these velocity transects (monthly for the 4 years from April 2020 to March 2024) and marks the months associated with NW-flow and EDDY-flow: **Figure 2**.
- `1_determine_cross_sections.py`: Script that determines the coordinates of the 5 cross-sections on which the analysis is performed: North-of-Curaçao, West Point, Mid Point, Klein Curaçao and South-of-Curaçao.
- `2_calc_crossings_KC_MP_WP.py`: Script that iterates through particle trajectories and records crossings at three cross-sections: West Point, Mid Point and Klein Curaçao.
- `2_calc_crossings_SS_NS.py`: Script that iterates through particle trajectories and records crossings at three cross-sections: North-of-Curaçao and South-of-Curaçao.
- `3_calc_segmentation_ALL.py`: Script to identify and record crossing locations along each cross-section for every particle intersection. Cross-sections are subdivided into segments using depth and distance from Venezuela and Curaçao as criteria.
- `3_plot_segmentation_crossings.py`: Script that visualizes each cross-section and its segments, along with example particle crossings for one time period (**Figure 4**)
- `4_calc_combine_segments_timeline.py`: Script that creates a timeline for each crossing event. 
- `5_plot_transition_matrix.py`: Script that plots the transition matrix for all crossings: **Figure 6**.
- `5_plot_differential_transition_matrix.py`: Script that plots the differential transition matrix, which is the difference between flow during EDDY-flow months and flow during NW_flow months: **Figure 7**.
- `6_plot_sankey_diagram_both_regimes.py`: Script that plots the Sankey diagram for both regimes (NW-flow and EDDY-flow) for the 80 most common flows: **Figure 5**.
- `7_calc_nearshore_trajectories.py`: Script that only takes the trajectories of the particles that have at some point passed at least one of the 6 nearshore segments. 
- `8_plot_nearshore_all_segments.py`: Script that plots the depth timeseries of the offshore-nearshore connectivity: **Figures 8 and 9**.
- `9_calc_barchart_nearshore.py`: Script that summarizes offshore-nearshore connectivity over the entire period of 4 years.
- `10_plot_barchart_nearshore.py`: Script that plots the summary of offshore-nearshore connectivity: **Figure 10**.
- `submit_...`: these are example scripts that are used to submit SLURM jobs to IMAU Lorenz cluster (usually to run in parallel, but not necessary)
 
