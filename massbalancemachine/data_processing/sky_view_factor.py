# This file should not be included into the MassBalanceMachine package!
# It is designed to be executed in a separate environment
# Computing sky view factor is done through rvt which is not compatible with the MassBalanceMachine environment


import os
import sys
import numpy as np
import xarray as xr
import rvt.vis
import argparse

parser = argparse.ArgumentParser("Generate sky view factor netCDF file.")
parser.add_argument(
    "rgi_folder",
    type=str,
    help="Folder of the glacier for which sky view factor data should be generated.",
)
args = parser.parse_args()

rgi_folder = args.rgi_folder


dem_file = os.path.join(rgi_folder, "dem.nc")

dem = xr.open_dataset(dem_file)

voxel_size = max(
    np.abs(np.mean(np.diff(dem.x))), np.abs(np.mean(np.diff(dem.y)))
)  # Meters

svf_n_dir = 16
svf_r_max = 10
svf_noise = 0
asvf_level = 1
asvf_dir = 315

out = rvt.vis.sky_view_factor(
    dem=dem.topo,
    resolution=voxel_size,
    compute_svf=True,
    compute_asvf=False,  # True,
    compute_opns=False,  # True,
    svf_n_dir=svf_n_dir,
    svf_r_max=svf_r_max,
    svf_noise=svf_noise,
    asvf_level=asvf_level,
    asvf_dir=asvf_dir,
)
svf = out["svf"]

svf_xr = xr.DataArray(svf, dims=("y", "x"), name="svf")

new_dem = dem.assign(svf=svf_xr)

new_dem.to_netcdf(os.path.join(rgi_folder, "svf.nc"))
