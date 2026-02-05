import numpy as np
import pandas as pd
import geopandas as gpd
import massbalancemachine as mbm

from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.geo_data.svf import open_and_merge_svf
from regions.Switzerland.scripts.geo_data.geodata import *


def process_yearly_gl_sgi_grid(
    cfg,
    glacier_name: str,
    year: int,
    data_root: str,
    path_glamos_topo: str,
    path_xr_grids: str,
    path_xr_svf: str,
) -> str:
    """
    Process a single (glacier, year) SGI topography grid into a masked WGS84 Zarr dataset.

    This function executes the complete processing workflow for one glacier and one year:
    1) Resolve SGI/RGI identifiers for the given glacier.
    2) Locate the yearly DEM `.grid` file stored in LV95 (EPSG:2056) coordinates.
    3) Load the grid and convert it to an xarray DataArray using metadata.
    4) Transform coordinates from LV95 to WGS84 geographic (lat/lon).
    5) Generate a glacier mask based on non-null DEM values.
    6) Enrich the dataset with SGI-derived topographic layers
       (masked aspect, slope, elevation) via `xr_GLAMOS_masked_topo`.
    7) Merge sky-view-factor (SVF) variables if a corresponding NetCDF file exists.
    8) Save the final enriched dataset as a consolidated Zarr store.

    The function is designed to be executed independently for many glacier–year
    combinations (e.g., in parallel via `ProcessPoolExecutor`). For robustness,
    it catches all internal exceptions and returns a descriptive status string
    instead of raising errors.

    Parameters
    ----------
    cfg : object
        Configuration object providing access to project paths and glacier metadata.
        Only lightweight attributes (e.g., `cfg.dataPath`) are used, making the
        function compatible with multiprocessing.
    glacier_name : str
        Short name of the glacier to process (e.g., "aletsch").
    year : int
        Year of the DEM grid to process.
    data_root : str
        Base directory where GLAMOS topography data are stored.
    path_glamos_topo : str
        Relative path to the GLAMOS topography folder within `data_root`.
    path_xr_grids : str
        Output directory where processed Zarr datasets will be written.
    path_xr_svf : str
        Directory containing optional sky-view-factor NetCDF files
        (`<glacier>_<year>_svf_latlon.nc`).

    Returns
    -------
    str
        Human-readable status message indicating one of:
        - successful processing with output Zarr path,
        - reason for skipping the glacier-year,
        - or an error message if an exception occurred.

    Raises
    ------
    None
        All exceptions are handled internally and reported through the return value
        to allow safe large-scale parallel execution.

    Notes
    -----
    - Years earlier than 1951 are automatically skipped.
    - The glacier “sanktanna” is handled via the folder name “stanna”
      to match GLAMOS directory naming conventions.
    - Missing input files result in a descriptive SKIP message rather than failure.
    - If an SVF file is present, variables such as `svf`, `asvf`, and `opns`
      (and their masked versions) are merged into the output dataset.
    - The saved Zarr dataset contains at least:
        * `dem`
        * `glacier_mask`
        * `masked_aspect`
        * `masked_slope`
        * `masked_elev`
        * optional SVF variables if available
    """
    try:
        # resolve SGI/RGI
        sgi_id, rgi_id, rgi_shp = get_rgi_sgi_ids(cfg, glacier_name)
        if not sgi_id or not rgi_shp:
            return f"SKIP {glacier_name} {year}: missing SGI or shapefile."

        # folder with .grid files
        folder_path = os.path.join(
            data_root,
            path_glamos_topo,
            "lv95",
            "stanna" if glacier_name == "sanktanna" else glacier_name,
        )
        if not os.path.exists(folder_path):
            return f"SKIP {glacier_name} {year}: folder missing."

        if year < 1951:
            return f"SKIP {glacier_name} {year}: <1951."

        file_name = f"gl_{year}_lv95.grid"
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            return f"SKIP {glacier_name} {year}: grid file missing."

        # load grid → xarray → lat/lon
        metadata, grid_data = load_grid_file(file_path)
        dem_y = convert_to_xarray_geodata(grid_data, metadata)
        dem_wgs84_y = transform_xarray_coords_lv95_to_wgs84(dem_y)

        ds_gl = xr.Dataset({"dem": dem_wgs84_y})
        ds_gl["glacier_mask"] = ds_gl["dem"].notnull().astype(np.uint8)

        # your topo enrichment (aspect/slope/etc) in WGS84
        ds_latlon = xr_GLAMOS_masked_topo(cfg, sgi_id, ds_gl)

        # # resolution and optional coarsening
        # dx_m, dy_m = get_res_from_degrees(ds_latlon)
        # if dx_m > 20:
        #     ds_latlon = coarsen_DS(ds_latlon, target_res_m=50)

        # merge SVF
        ds_latlon = open_and_merge_svf(ds_latlon, glacier_name, year, path_xr_svf)

        # save zarr
        save_path = os.path.join(path_xr_grids, f"{glacier_name}_{year}.zarr")
        ds_latlon.to_zarr(save_path, mode="w", consolidated=True)
        return f"OK {glacier_name} {year} → {save_path}"
    except Exception as e:
        return f"ERROR {glacier_name} {year}: {e}"


def create_glacier_grid_SGI(glacierName, year, rgi_id, ds, start_month="10"):
    """
    Generate a WGMS-style point grid dataframe from an SGI masked topography dataset.

    This function converts a glacier-masked xarray dataset into a tabular grid
    representation suitable for downstream mass-balance modeling. All grid cells
    where `glacier_mask == 1` are extracted and transformed into individual point
    records containing geographic coordinates and topographic attributes.

    Parameters
    ----------
    glacierName : str
        Name of the glacier (used for metadata fields in the output table).
    year : int
        Reference year of the grid; used to construct hydrological date ranges.
    rgi_id : str
        RGI identifier associated with the glacier.
    ds : xarray.Dataset
        Masked SGI topography dataset containing at least the variables:
        - `glacier_mask`
        - `masked_elev`
        - `masked_aspect`
        - `masked_slope`
        - `svf`
        and coordinates `lon` and `lat`.
    start_month : str, optional
        Hydrological start month used when constructing FROM_DATE.
        Default is "10" (October).

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per glacier grid cell, containing columns:

        Spatial attributes:
        - `POINT_LAT`, `POINT_LON`
        - `aspect`, `slope`, `topo`, `svf`

        WGMS-compatible fields:
        - `RGIId`
        - `POINT_ID`
        - `POINT_ELEVATION`
        - `POINT_BALANCE` (set to 0 as placeholder)
        - `N_MONTHS`
        - `PERIOD` (always "annual")
        - `GLACIER`
        - `YEAR`
        - `FROM_DATE`, `TO_DATE`

    Notes
    -----
    - The function assumes that `ds` has already been transformed to WGS84
      coordinates and masked to the glacier outline.
    - `POINT_BALANCE` is set to 0 as a dummy value; this grid is intended for
      use as model input rather than as observations.
    - Elevation is taken directly from `masked_elev` and duplicated into
      `POINT_ELEVATION` for compatibility with WGMS-style processing.
    - FROM_DATE is constructed as `<year><start_month>01`
      and TO_DATE as `<year+1>0930`, following a standard hydrological year.
    - All points are assigned a simple sequential `POINT_ID` starting from 1.

    Raises
    ------
    KeyError
        If required variables or coordinates are missing from `ds`.
    """
    glacier_indices = np.where(ds["glacier_mask"].values == 1)

    # Glacier mask as boolean array:
    gl_mask_bool = ds["glacier_mask"].values.astype(bool)

    lon_coords = ds["lon"].values
    lat_coords = ds["lat"].values

    lon = lon_coords[glacier_indices[1]]
    lat = lat_coords[glacier_indices[0]]

    # Create a DataFrame
    data_grid = {
        "RGIId": [rgi_id] * len(ds.masked_elev.values[gl_mask_bool]),
        "POINT_LAT": lat,
        "POINT_LON": lon,
        "aspect": ds.masked_aspect.values[gl_mask_bool],
        "slope": ds.masked_slope.values[gl_mask_bool],
        "topo": ds.masked_elev.values[gl_mask_bool],
        "svf": ds.svf.values[gl_mask_bool],
    }
    df_grid = pd.DataFrame(data_grid)

    # Match to WGMS format:
    df_grid["POINT_ID"] = np.arange(1, len(df_grid) + 1)
    df_grid["N_MONTHS"] = 12
    df_grid["POINT_ELEVATION"] = df_grid["topo"]  # no other elevation available
    df_grid["POINT_BALANCE"] = 0  # fake PMB for simplicity (not used)

    # Add metadata that is not in WGMS dataset
    df_grid["PERIOD"] = "annual"
    df_grid["GLACIER"] = glacierName
    # Add the 'year' and date columns to the DataFrame
    df_grid["YEAR"] = np.tile(year, len(df_grid))
    df_grid["FROM_DATE"] = df_grid["YEAR"].apply(lambda x: str(x) + f"{start_month}01")
    df_grid["TO_DATE"] = df_grid["YEAR"].apply(lambda x: str(x + 1) + "0930")

    return df_grid


def add_OGGM_features(df_y_gl, voi, path_OGGM):
    """
    Enrich a point dataset with gridded OGGM variables using nearest-neighbor sampling.

    For each unique RGIId present in the input dataframe, this function:
    1) Opens the corresponding OGGM gridded dataset `<path_OGGM>/xr_grids/<RGIId>.zarr`
    2) Transforms stake coordinates from WGS84 (EPSG:4326) to the projection of the OGGM dataset
    3) Samples the requested variables of interest at the nearest grid cell
    4) Appends these variables as new columns in the dataframe

    Parameters
    ----------
    df_y_gl : pandas.DataFrame
        Input dataframe containing at least the following columns:
        - `RGIId`
        - `POINT_LON`
        - `POINT_LAT`
    voi : list of str
        List of OGGM variable names to extract from the gridded datasets
        (e.g., ["topo", "slope", "aspect", "hugonnet_dhdt"]).
    path_OGGM : str
        Base path to the OGGM data directory. The function expects to find
        per-glacier datasets in `<path_OGGM>/xr_grids/<RGIId>.zarr`.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with additional columns corresponding to
        each variable in `voi`. Values are assigned using nearest-neighbor lookup
        from the OGGM grids. Rows for which OGGM data are unavailable remain NaN.

    Notes
    -----
    - If a Zarr file for a given RGIId is missing, the glacier is skipped and a
      message is printed.
    - If a requested variable is not present in the OGGM dataset, that variable
      is skipped for the affected glacier.
    - All coordinate transformations are performed using pyproj based on the
      CRS information stored in each OGGM dataset.
    - Sampling uses xarray `.sel(..., method="nearest")`, meaning no interpolation
      is performed beyond nearest-neighbor selection.

    Raises
    ------
    KeyError
        If required input columns (`RGIId`, `POINT_LON`, `POINT_LAT`) are missing
        from `df_y_gl`.
    """
    df_pmb = df_y_gl.copy()

    # Initialize empty columns for the variables
    for var in voi:
        df_pmb[var] = np.nan

    # Path to OGGM datasets
    path_to_data = path_OGGM + "xr_grids/"

    # Group rows by RGIId
    grouped = df_pmb.groupby("RGIId")

    # Process each group
    for rgi_id, group in grouped:
        file_path = f"{path_to_data}{rgi_id}.zarr"

        try:
            # Load the xarray dataset for the current RGIId
            ds_oggm = xr.open_dataset(file_path)
        except FileNotFoundError:
            print(f"File not found for RGIId: {file_path}")
            continue

        # Define the coordinate transformation
        transf = pyproj.Transformer.from_proj(
            pyproj.CRS.from_user_input("EPSG:4326"),  # Input CRS (WGS84)
            pyproj.CRS.from_user_input(ds_oggm.pyproj_srs),  # Output CRS from dataset
            always_xy=True,
        )

        # Transform all coordinates in the group
        lon, lat = group["POINT_LON"].values, group["POINT_LAT"].values
        x_stake, y_stake = transf.transform(lon, lat)
        # Select nearest values for all points
        try:
            stake = ds_oggm.sel(
                x=xr.DataArray(x_stake, dims="points"),
                y=xr.DataArray(y_stake, dims="points"),
                method="nearest",
            )

            # Extract variables of interest
            stake_var = stake[voi]

            # Convert the extracted data to a DataFrame
            stake_var_df = stake_var.to_dataframe()

            # Update the DataFrame with the extracted values
            for var in voi:
                df_pmb.loc[group.index, var] = stake_var_df[var].values
        except KeyError as e:
            print(f"Variable missing in dataset {file_path}: {e}")
            continue
    return df_pmb


# ============== worker initializer ==============
_WORKER = {"RGI_OUTLINES": None}


def init_worker(outlines_path: str):
    """Runs once per worker process. Load expensive, read-only resources here."""
    global _WORKER
    _WORKER = {"RGI_OUTLINES": None}

    if outlines_path and os.path.exists(outlines_path):
        _WORKER["RGI_OUTLINES"] = gpd.read_file(outlines_path)
    else:
        _WORKER["RGI_OUTLINES"] = None


def process_monthly_grids_gl(
    cfg,
    glacier_name: str,
    year: int,
    *,
    data_root: str,
    path_xr_grids: str,
    out_folder_root: str,
    vois_climate: list,
    vois_topo: list,
    meta_cols: list,
    era5_monthly_path: str,
    era5_geopot_path: str,
    pcsr_zarr_root: str,
    oggm_path: str,
    start_month: str,
    small_glaciers: list,
) -> str:
    """Process one (glacier, year) -> writes a parquet; returns status string."""
    try:
        if glacier_name in small_glaciers:
            return f"SKIP {glacier_name} {year}: too small"

        file_name = f"{glacier_name}_{year}.zarr"
        zarr_path = os.path.join(path_xr_grids, file_name)
        if not os.path.exists(zarr_path):
            return f"SKIP {glacier_name} {year}: zarr not found"

        ds = xr.open_zarr(zarr_path)

        sgi_id, rgi_id, rgi_shp = get_rgi_sgi_ids(cfg, glacier_name)
        if not sgi_id or not rgi_id or not rgi_shp:
            return f"SKIP {glacier_name} {year}: missing SGI/RGI"

        df_grid = create_glacier_grid_SGI(
            glacier_name, year, rgi_id, ds, start_month=start_month
        ).reset_index(drop=True)

        dataset_grid = mbm.data_processing.Dataset(
            cfg=cfg,
            data=df_grid,
            region_name="CH",
            region_id=11,
            data_path=os.path.join(data_root, path_PMB_GLAMOS_csv),
        )

        dataset_grid.get_climate_features(
            climate_data=era5_monthly_path,
            geopotential_data=era5_geopot_path,
            change_units=True,
            smoothing_vois={
                "vois_climate": vois_climate,
                "vois_other": ["ALTITUDE_CLIMATE"],
            },
        )

        dataset_grid.get_potential_rad(pcsr_zarr_root)

        df_y_gl = dataset_grid.data
        df_y_gl.rename(columns={"RGIId": "RGIId_old"}, inplace=True)

        outlines = _WORKER.get("RGI_OUTLINES")
        if outlines is None:
            return f"SKIP {glacier_name} {year}: outlines not loaded in worker"

        df_y_gl = mbm.data_processing.utils.get_rgi(
            data=df_y_gl, glacier_outlines=outlines
        )

        df_y_gl = df_y_gl.dropna(subset=["RGIId"])

        df_y_gl = add_OGGM_features(
            df_y_gl,
            ["hugonnet_dhdt", "consensus_ice_thickness", "millan_v"],
            oggm_path,
        )

        df_y_gl["GLWD_ID"] = df_y_gl.apply(
            lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"),
            axis=1,
        ).astype(str)

        dataset_grid_oggm = mbm.data_processing.Dataset(
            cfg=cfg,
            data=df_y_gl,
            region_name="CH",
            region_id=11,
            data_path=os.path.join(data_root, path_PMB_GLAMOS_csv),
        )

        dataset_grid_oggm.convert_to_monthly(
            meta_data_columns=meta_cols,
            vois_climate=vois_climate + ["pcsr"],
            vois_topographical=vois_topo,
        )

        df_oggm = dataset_grid_oggm.data

        if "svf" not in df_oggm.columns:
            return f"ERROR {glacier_name} {year}: 'svf' missing after conversion"
        if "pcsr" not in df_oggm.columns:
            return f"ERROR {glacier_name} {year}: 'pcsr' missing after conversion"

        df_oggm.rename(
            columns={"aspect": "aspect_sgi", "slope": "slope_sgi"}, inplace=True
        )

        if "POINT_ELEVATION" not in df_oggm.columns:
            return f"ERROR {glacier_name} {year}: 'POINT_ELEVATION' missing"

        out_folder = os.path.join(out_folder_root, glacier_name)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{glacier_name}_grid_{year}.parquet")
        df_oggm.to_parquet(out_path, engine="pyarrow", compression="snappy")

        return f"OK {glacier_name} {year} -> {out_path}"

    except Exception as e:
        return f"ERROR {glacier_name} {year}: {type(e).__name__}: {e}"


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
