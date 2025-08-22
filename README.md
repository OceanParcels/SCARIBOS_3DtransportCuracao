# SCARIBOS_3DtransportCuracao

Repository for Vesna's project: Ocean-to-nearshore circulation patterns around Curaçao: A southern Caribbean reef island exposed to distinct flow regimes.

This project contains the following folders:

## Project Structure:

- [**parcels_run**](.o/parcels_run/): Contains the scripts to run the OceanParcels experiments
- [**parcels_analysis**](./parcels_analysis/): Contains the scripts related to particle tracking and trajectory analysis using OceanParcels.

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

The simulated particles represent neutrally buoyant tracersthe volumetric transport of water masses within the region. Particle deployment occurs across multiple depth levels, extending from the sea surface down to the maximum model depth, using logarithmic vertical spacing to provide higher resolution near the surface where the current is generally the strongest. This vertical arrangement creates depth intervals ranging from approximately 4.3 meters near the surface to 435 meters at the greatest depths. Horizontal particle placement follows a regular grid pattern with 0.01° spacing (approximately 1 km resolution).

### Running the Parcels simulations:

The structure of the scripts, found in [**parcels_run**](./parcels_run/), is as follows:

- `1_particle_release.py`: Determine locations and depths of particle release
- `1_plot_particle_release.py`: Plot particle release locations (**Figure 3 in manuscript**)
- `2_run_INFLOW4B4M.py`: Script to run the main Parcels experiment - can be executed in parallel for all months. Each simulation includes particle seeding over a 3-month period.
- `2_run_SAMPLEVEL.py`: Script to execute single time-step particle seeding while recording initial particle velocities - required for calculating volume transport per particle.
- `3_calc_VT.py`: Script to calcualte volume transport of each particle.
- `4_calc_rampup_multiple_batches.py`: Script to determine ramp-up period (ramp-up period is marked in polts in **Figures 8 and 9**).
- `submit_...`: these are example scripts that are used to submit SLURM jobs to IMAU Lorenz cluster (usually to run in parallel, but not necessary)

### Analysizng the Parcels outputs:

The structure of the scripts, found in [**parcels_analysis**](./parcels_analysis/), is as follows:
- `0_extract_SCARIBOS_temperature_salinity.py`:
- `0_calc_and_plot_water_masses.py`: 
- `0_calc_regimes_from_croco.py`:
- `0_plot_regimes_from_croco.py`: 
- `1_determine_cross_sections.py`: 
- `2_calc_crossings_KC_MP_WP.py`: 
- `2_calc_crossings_SS_NS.py`: 
- `3_calc_segmentation_ALL.py`: 
- `3_plot_segmentation_crossings.py`: Script that visualizes each cross-section and its segments, along with example particle crossings for one time period (**Figure 4**)
- `4_calc_combine_segments_timeline.py`:
- `5_plot_transition_matrix.py`: 
- `5_plot_differential_transition_matrix.py`: 
- `6_plot_sankey_diagram_both_regimes.py`: 
- `7_calc_nearshore_trajectories.py`: 
- `8_plot_nearshore_all_segments.py`: 
- `9_calc_barchart_nearshore.py`: 
- `10_plot_barchart_nearshore.py`: 
- `submit_...`: these are example scripts that are used to submit SLURM jobs to IMAU Lorenz cluster (usually to run in parallel, but not necessary)
 
