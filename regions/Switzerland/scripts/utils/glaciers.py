import pandas as pd
import os
import geopandas as gpd

from regions.Switzerland.scripts.config_CH import *


def get_gl_area(cfg):
    """
    Retrieve glacier surface areas from the SGI 2016 glacier inventory shapefile.

    This function matches glacier identifiers from the project glacier ID table
    with polygons from the SGI 2016 inventory shapefile and extracts the glacier
    area for each glacier. The result is returned as a dictionary keyed by
    glacier short names.

    Special handling is applied to the glacier "clariden", which is stored in the
    inventory as two separate entities (claridenL / claridenU). In this case,
    the area corresponding to "claridenL" is used.

    Parameters
    ----------
    cfg : object
        Configuration object containing at least:
        - `dataPath` : base directory for input data
        - `path_glacier_ids` : relative path to the glacier ID CSV file
        - `path_SGI_topo` : relative path to the SGI topography folder

    Returns
    -------
    dict
        Dictionary mapping glacier short names (str) to glacier areas (float),
        as stored in the SGI shapefile attribute `Area`.

    Raises
    ------
    FileNotFoundError
        If the glacier ID CSV file or SGI shapefile cannot be found.
    KeyError
        If required columns are missing from the input tables.

    Notes
    -----
    - Only glaciers present in both the glacier ID table and the SGI shapefile
      are included in the output.
    - Glaciers with missing or undefined `rgi_id_v6_2016_shp` values are skipped.
    - Areas are taken directly from the shapefile attribute table without
      reprojection or recalculation.
    """
    # Load glacier metadata
    rgi_df = pd.read_csv(cfg.dataPath + path_glacier_ids, sep=",")
    rgi_df.rename(columns=lambda x: x.strip(), inplace=True)
    rgi_df.sort_values(by="short_name", inplace=True)
    rgi_df.set_index("short_name", inplace=True)

    # Load the shapefile
    shapefile_path = os.path.join(
        cfg.dataPath,
        path_SGI_topo,
        "inventory_sgi2016_r2020",
        "SGI_2016_glaciers_copy.shp",
    )
    gdf_shapefiles = gpd.read_file(shapefile_path)

    gl_area = {}

    for glacierName in rgi_df.index:
        if glacierName == "clariden":
            rgi_shp = (
                rgi_df.loc["claridenL", "rgi_id_v6_2016_shp"]
                if "claridenL" in rgi_df.index
                else None
            )
        else:
            rgi_shp = rgi_df.loc[glacierName, "rgi_id_v6_2016_shp"]

        # Skip if rgi_shp is not found
        if pd.isna(rgi_shp) or rgi_shp is None:
            continue

        # Ensure matching data types
        rgi_shp = str(rgi_shp)
        gdf_mask_gl = gdf_shapefiles[gdf_shapefiles.RGIId.astype(str) == rgi_shp]

        # If a glacier is found, get its area
        if not gdf_mask_gl.empty:
            gl_area[glacierName] = gdf_mask_gl.Area.iloc[0]  # Use .iloc[0] safely

    return gl_area
