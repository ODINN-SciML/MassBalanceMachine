import os
import numpy as np
import xarray as xr


def create_masked_glacier_grid(path_RGIs, rgi_gl):
    """
    Create masked glacier dataset from OGGM .zarr file,
    """

    # --- Load dataset ---
    ds = xr.open_zarr(os.path.join(path_RGIs, f"{rgi_gl}.zarr"))

    # --- Check for glacier mask ---
    if "glacier_mask" not in ds:
        raise ValueError(f"'glacier_mask' variable not found in dataset {rgi_gl}")

    # --- Build mask (NaN outside glacier) ---
    glacier_mask = np.where(
        ds["glacier_mask"].values == 0, np.nan, ds["glacier_mask"].values
    )

    # --- Apply mask to core variables ---
    ds = ds.assign(masked_slope=glacier_mask * ds["slope"])
    ds = ds.assign(masked_elev=glacier_mask * ds["topo"])
    ds = ds.assign(masked_aspect=glacier_mask * ds["aspect"])
    ds = ds.assign(masked_dis=glacier_mask * ds["dis_from_border"])

    # --- Optional fields ---
    if "hugonnet_dhdt" in ds:
        ds = ds.assign(masked_hug=glacier_mask * ds["hugonnet_dhdt"])
    if "consensus_ice_thickness" in ds:
        ds = ds.assign(masked_cit=glacier_mask * ds["consensus_ice_thickness"])
    if "millan_v" in ds:
        ds = ds.assign(masked_miv=glacier_mask * ds["millan_v"])

    # --- Get indices where glacier_mask == 1 ---
    glacier_indices = np.where(ds["glacier_mask"].values == 1)

    return ds, glacier_indices
