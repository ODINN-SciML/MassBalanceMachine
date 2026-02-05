import pandas as pd
import os
import geopandas as gpd

from regions.French_Alps.scripts.config_FR import *


def get_gl_area_FR(
    df_pmb,
    shapefile_path,
    glacier_col="GLACIER",
    rgi_col_df="RGIId",
    rgi_col_shp="RGIId",
    area_col_shp="Area",
    verbose=True,
):
    """
    Extract glacier areas from an RGI shapefile and map them to glaciers
    present in a PMB dataframe.

    Parameters
    ----------
    df_pmb : pandas.DataFrame
        DataFrame containing at least:
        - glacier_col (e.g. 'GLACIER')
        - rgi_col_df (e.g. 'RGIId')
    shapefile_path : str
        Path to the RGI shapefile.
    glacier_col : str, default 'GLACIER'
        Column in df_pmb with glacier names.
    rgi_col_df : str, default 'RGIId'
        Column in df_pmb containing RGI IDs.
    rgi_col_shp : str, default 'RGIId'
        Column name in shapefile containing RGI IDs.
    area_col_shp : str, default 'Area'
        Column name in shapefile containing glacier area (km²).
    verbose : bool, default True
        Print warnings for missing or inconsistent IDs.

    Returns
    -------
    gl_area : dict
        Dictionary mapping glacier name → glacier area in km².
    """

    gdf_shapefiles = gpd.read_file(shapefile_path)

    # Ensure string matching works
    gdf_shapefiles[rgi_col_shp] = gdf_shapefiles[rgi_col_shp].astype(str)

    gl_area = {}

    for gl in df_pmb[glacier_col].dropna().unique():

        rgi_ids = df_pmb.loc[df_pmb[glacier_col] == gl, rgi_col_df].dropna().unique()

        if len(rgi_ids) == 0:
            if verbose:
                print(f"[WARN] No RGIId found for glacier '{gl}'")
            continue

        if len(rgi_ids) > 1 and verbose:
            print(f"[WARN] Multiple RGIIds for glacier '{gl}': {rgi_ids}")

        rgi_id = str(rgi_ids[0])

        gdf_mask_gl = gdf_shapefiles[gdf_shapefiles[rgi_col_shp] == rgi_id]

        if gdf_mask_gl.empty:
            if verbose:
                print(f"[WARN] No shapefile match for glacier '{gl}' ({rgi_id})")
            continue

        gl_area[gl.lower()] = float(gdf_mask_gl[area_col_shp].iloc[0])

    return gl_area
