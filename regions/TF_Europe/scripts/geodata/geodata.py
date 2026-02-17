import os
import xarray as xr
from pyproj import CRS


def export_glacier_dems_to_geotiff(
    path_RGIs,
    rgi_gl,
    path_out_tiff,
    *,
    dem_var="topo",
    mask_var="glacier_mask",
    masked=True,
):
    """
    Export a glacier DEM from an OGGM per-glacier Zarr file to a GeoTIFF,
    automatically deducing the CRS from the dataset.

    The function opens `<path_RGIs>/<rgi_gl>.zarr`, extracts the DEM variable
    (default: 'topo'), optionally masks it using `mask_var` (default: 'glacier_mask',
    setting non-glacier pixels to NaN), ensures a CRS is attached (preferring
    `ds.rio.crs`, falling back to OGGM's `ds.pyproj_srs`), and writes a GeoTIFF
    to `path_out_tiff`.

    Parameters
    ----------
    path_RGIs : str
        Directory containing per-glacier Zarr datasets named `<rgi_gl>.zarr`.
    rgi_gl : str
        Glacier identifier used for the Zarr filename (e.g., RGIId).
    path_out_tiff : str
        Output directory for the GeoTIFF.
    dem_var : str, optional
        Name of the DEM variable in the dataset (default: 'topo').
    mask_var : str, optional
        Name of the glacier mask variable (default: 'glacier_mask').
    masked : bool, optional
        If True, mask DEM outside glacier (non-glacier pixels → NaN). Default True.

    Returns
    -------
    str
        Path to the written GeoTIFF.

    Raises
    ------
    ValueError
        If `dem_var` (or `mask_var` when `masked=True`) is missing.
    RuntimeError
        If a CRS cannot be inferred from the dataset.

    Notes
    -----
    - Requires rioxarray (`.rio`) to be available on the DataArray/Dataset.
    - OGGM often stores CRS in `ds.pyproj_srs` even when `ds.rio.crs` is None.
    """

    # --- Load dataset ---
    zpath = os.path.join(path_RGIs, f"{rgi_gl}.zarr")
    ds = xr.open_zarr(zpath)

    if dem_var not in ds:
        raise ValueError(f"'{dem_var}' variable not found in dataset {rgi_gl}")

    dem = ds[dem_var]

    # --- Optional masking ---
    if masked:
        if mask_var not in ds:
            raise ValueError(f"'{mask_var}' variable not found in dataset {rgi_gl}")
        dem = dem.where(ds[mask_var] == 1)

    # --- Deduce CRS ---
    crs = None

    # 1) Prefer rioxarray CRS if present
    try:
        crs = ds.rio.crs
    except Exception:
        crs = None

    # 2) Fallback to OGGM's pyproj_srs
    if crs is None and hasattr(ds, "pyproj_srs") and ds.pyproj_srs:
        crs = CRS.from_string(ds.pyproj_srs)

    if crs is None:
        raise RuntimeError(
            f"Could not infer CRS for {rgi_gl}. "
            f"ds.rio.crs is None and ds.pyproj_srs is missing/empty."
        )

    # --- Attach CRS and write GeoTIFF ---
    dem = dem.rio.write_crs(crs, inplace=False)

    os.makedirs(path_out_tiff, exist_ok=True)
    out_tif = os.path.join(path_out_tiff, f"{rgi_gl}.tif")

    dem.rio.to_raster(
        out_tif,
        dtype="float32",
        compress="LZW",
        BIGTIFF="IF_SAFER",
        tiled=True,
        predictor=3,
    )

    print(f"Saved DEM GeoTIFF to: {out_tif}")
    return out_tif
