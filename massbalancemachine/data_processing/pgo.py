import os

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import oggm.utils

from data_processing.oggm_utils import (
    _initialize_oggm_config,
    _initialize_custom_glacier_directories,
)
from data_processing.product_utils import (
    rgi_id_to_folders,
    region_id_folders,
    mbm_path,
    data_path,
)
from data_processing.Product import Product
from data_processing.glacier_utils import get_region_shape_file


def prepare_PGO_outlines(shp_file: str, csv_corresp: str, rgi_ids_to_keep=None):
    # Load correspondence file
    corresp = pd.read_csv(csv_corresp)

    # Build mapping of GLACIER_NR to RGI ID
    assert (
        corresp.GLACIER_NR.nunique() == corresp.shape[0]
    ), "There are multiple entries of the same GLACIER_NR."
    dict_corresp = {
        corresp.loc[i, "GLACIER_NR"]: corresp.loc[i, "rgi_id"]
        for i in range(corresp.shape[0])
    }

    # Load your custom shapefile
    custom_outlines = gpd.read_file(shp_file).to_crs("EPSG:4326")
    custom_outlines["RGIId"] = custom_outlines.GLACIER_NR.map(
        lambda glacier_nr: dict_corresp[glacier_nr]
    )
    if rgi_ids_to_keep is not None:
        to_keep = []
        for rgi_id in rgi_ids_to_keep:
            tmp = custom_outlines[custom_outlines.RGIId == rgi_id]
            for ind in tmp.index:
                to_keep.append(ind)
        custom_outlines = custom_outlines.loc[to_keep]
    return custom_outlines


def prepare_custom_glaciers_mask(
    custom_outlines, custom_tmp_working_dir="oggm-custom-pgo"
):
    # Initialize the OGGM Config
    working_dir = oggm.utils.gettempdir(dirname=custom_tmp_working_dir, reset=True)
    _initialize_oggm_config(working_dir)

    # Retrieve projection
    crs = CRS(custom_outlines.crs)
    utm_zone = crs.utm_zone
    if utm_zone is None:
        centroid_lon = custom_outlines.geometry.centroid.x.mean()
        centroid_lat = custom_outlines.geometry.centroid.y.mean()

        zone_number = int((centroid_lon + 180) / 6) + 1
        hemisphere = "N" if centroid_lat >= 0 else "S"
        utm_zone = f"{zone_number}{hemisphere}"
    utm_crs = CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "south": centroid_lat < 0}
    )
    custom_outlines = custom_outlines.to_crs(utm_crs)

    # Cook dataframe into an RGI-compatible GeoDataFrame
    # o1_region should match the RGI region of your area (e.g. '11' for Central Europe)
    rgidf = oggm.utils.cook_rgidf(
        custom_outlines,
        o1_region="11",
        assign_column_values={"RGIId": "RGIId"},
    )
    rgidf["utm_zone"] = utm_zone

    # Drop samples for which there is no associated RGI ID
    rgidf = rgidf[rgidf["RGIId"].notna()]
    # Check RGI ID format
    for rgi_id in rgidf.RGIId.unique():
        assert "RGI2000-v7.0" in rgi_id

    # geom_utm = rgidf.geometry.to_crs(utm_crs)
    # rgidf["Area"] = geom_utm.area / 1e6

    # Merge entries that share the same RGI ID
    merged_df = rgidf.dissolve(by="RGIId").reset_index()

    # Reproject and compute area in km²
    geom_utm = merged_df.geometry.to_crs(utm_crs)
    merged_df.Area = geom_utm.area / 1e6

    # Replace the geometry by the convex hull of the set of geometries to get a grid that covers all of the entries
    # That geometry is overwritten in _initialize_custom_glacier_directories by the union of all geometries since OGGM does not handle MultiPolygon
    merged_df["geometry"] = merged_df.geometry.convex_hull

    df = merged_df
    df = df.rename(
        columns={
            "RGIId": "rgi_id",
            "O1Region": "o1region",
            "O2Region": "o2region",
        }
    )
    df["src_date"] = "2019-01-01 00:00:00"

    # Run OGGM and overwrite geometry
    gdirs = _initialize_custom_glacier_directories(df, rgidf)

    return gdirs, rgidf


def geodetic_target_PGO(geo_file):
    df = pd.read_csv(geo_file)
    df = df[df["rgi_id"].notna()]
    df = df.drop(
        columns=[
            "begin_platform",
            "begin_method",
            "end_platform",
            "end_method",
            "agencies",
            "references",
            "remarks",
            "investigators",
        ]
    )

    # Drop time and keep only date
    df.begin_date_min = pd.to_datetime(
        df.begin_date_min.map(lambda begin_date_min: begin_date_min.split("_")[0])
    )
    df.begin_date_max = pd.to_datetime(
        df.begin_date_max.map(lambda begin_date_max: begin_date_max.split("_")[0])
    )
    df.end_date_min = pd.to_datetime(
        df.end_date_min.map(lambda end_date_min: end_date_min.split("_")[0])
    )
    df.end_date_max = pd.to_datetime(
        df.end_date_max.map(lambda end_date_max: end_date_max.split("_")[0])
    )

    # Sanity checks
    assert (df.begin_date_min.dt.year == df.begin_date_max.dt.year).all()
    assert (df.end_date_min.dt.year == df.end_date_max.dt.year).all()

    index_rgi = (
        df.begin_date_min.dt.year.astype(str)
        + "_"
        + df.end_date_min.dt.year.astype(str)
        + "_"
        + df.rgi_id
    )
    df["index_rgi"] = index_rgi
    # assert index_rgi.nunique()==index_rgi.shape[0], f"Number of unique index_rgi: {index_rgi.nunique()} but size of dataframe is {index_rgi.shape}"

    index_glacier_nr = (
        df.begin_date_min.dt.year.astype(str)
        + "_"
        + df.end_date_min.dt.year.astype(str)
        + "_"
        + df.GLACIER_NR.astype(str)
    )
    df["index_glacier_nr"] = index_glacier_nr
    assert (
        index_glacier_nr.nunique() == index_glacier_nr.shape[0]
    ), f"Number of unique index_glacier_nr: {index_glacier_nr.nunique()} but size of dataframe is {index_glacier_nr.shape}"

    # Check that start and end dates are close
    assert max(abs((df.begin_date_min - df.begin_date_max).dt.days.unique())) <= 32
    assert max(abs((df.end_date_min - df.end_date_max).dt.days.unique())) <= 10

    # Sanity check and filter entries
    cnts = {}
    to_keep = []
    for rgi_id in df.rgi_id.unique():
        tmp = df[df.rgi_id == rgi_id]
        cnt = tmp.rgi_id.count()
        cnts[rgi_id] = cnt
        if cnts[rgi_id] == 1:
            for ind in tmp.index:
                to_keep.append(ind)
        else:
            if tmp.GLACIER_NR.nunique() == cnt:
                # This is fine, it's just that the glacier is split across isolated blocks of ice
                # We need to: 1) make sure that the begin/end dates correspond 2) merge them
                for ind in tmp.index:
                    to_keep.append(ind)
            else:
                if tmp.index_rgi.nunique() == cnt:
                    # We have multiple starting date, we need to take the larger time window
                    assert (tmp.begin_date_min == tmp.begin_date_max).all()
                    assert (tmp.end_date_min == tmp.end_date_max).all()
                    year_range = tmp.end_date_min.dt.year - tmp.begin_date_min.dt.year
                    for ind in tmp[year_range == year_range.max()].index:
                        to_keep.append(ind)
                else:
                    if tmp.GLACIER_NR.nunique() * tmp.index_rgi.nunique() == cnt:
                        # We have multiple starting date and the glacier is split across isolated blocks of ice, we need to take the larger time window
                        year_range = (
                            tmp.end_date_min.dt.year - tmp.begin_date_min.dt.year
                        )
                        for ind in tmp[year_range == year_range.max()].index:
                            to_keep.append(ind)
                    elif tmp.GLACIER_NR.nunique() * tmp.index_rgi.nunique() > cnt:
                        # We have multiple starting date, the glacier is split across isolated blocks of ice and the time windows do not all include all the isolated blocks, we need to take the time window that covers all blocks
                        year_range = (
                            tmp.end_date_min.dt.year - tmp.begin_date_min.dt.year
                        )
                        d = {
                            tmp[year_range == y].shape[0]: y
                            for y in year_range.unique()
                        }
                        nunique_max_year_range = max(list(d.keys()))
                        for ind in tmp[year_range == d[nunique_max_year_range]].index:
                            to_keep.append(ind)
                    else:
                        # Unique time windows
                        assert (
                            tmp.begin_date_min.nunique() == 1
                        ), "More than one unique begin_date_min"
                        assert (
                            tmp.begin_date_max.nunique() == 1
                        ), "More than one unique begin_date_max"
                        assert (
                            tmp.end_date_min.nunique() == 1
                        ), "More than one unique end_date_min"
                        assert (
                            tmp.end_date_max.nunique() == 1
                        ), "More than one unique end_date_max"
                        for ind in tmp.index:
                            to_keep.append(ind)
    df["cnt"] = df.rgi_id.map(lambda rgi_id: cnts[rgi_id])
    to_keep.sort()
    df = df.loc[to_keep]

    to_keep = []
    # Merge entries with the same RGI ID
    for rgi_id in df.rgi_id.unique():
        tmp = df[df.rgi_id == rgi_id]
        if len(tmp) == 1:
            for ind in tmp.index:
                to_keep.append(ind)
        else:
            # Check the time windows of what we are about to merge are close
            delta_begin_min = (
                tmp.begin_date_min.unique().max() - tmp.begin_date_min.unique().min()
            ).days
            assert delta_begin_min <= 26
            delta_begin_max = (
                tmp.begin_date_max.unique().max() - tmp.begin_date_max.unique().min()
            ).days
            assert delta_begin_max <= 32
            delta_end_min = (
                tmp.end_date_min.unique().max() - tmp.end_date_min.unique().min()
            ).days
            assert delta_end_min <= 10
            delta_end_max = (
                tmp.end_date_max.unique().max() - tmp.end_date_max.unique().min()
            ).days
            assert delta_end_max <= 10

            ind = tmp[tmp.a_m2.max() == tmp.a_m2].index
            assert len(ind) == 1
            total_area = tmp.a_m2.sum()
            df.loc[ind, "a_m2"] = total_area
            df.loc[ind, "total_dh_per_year"] = (
                tmp.total_dh_per_year * tmp.a_m2
            ).sum() / total_area
            df.loc[ind, "elevation_change"] = (
                tmp.elevation_change * tmp.a_m2
            ).sum() / total_area
            df.loc[ind, "elevation_change_unc_2sigma"] = (
                tmp.elevation_change_unc_2sigma * tmp.a_m2
            ).sum() / total_area

            to_keep.append(ind[0])
    to_keep.sort()
    df = df.loc[to_keep]
    df = df.drop(columns=["GLACIER_NR", "index_rgi", "index_glacier_nr", "cnt"])
    df = df.rename(columns={"rgi_id": "RGIId"})

    # Round start and end dates
    df["FROM_DATE"] = (
        df.begin_date_min.where(
            df.begin_date_min.dt.day <= 15, df.begin_date_min + pd.offsets.MonthBegin(0)
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    df["TO_DATE"] = (
        df.end_date_min.where(
            df.end_date_min.dt.day <= 15, df.end_date_min + pd.offsets.MonthBegin(0)
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    density_ice = 0.850  # T/m³
    sigma_ice = 0.060
    sigma = df.elevation_change_unc_2sigma / 2
    m = df.elevation_change
    mwe = m * density_ice
    sigma_mwe = (
        density_ice**2 * sigma**2 + m**2 * sigma_ice**2
    ) ** 0.5  # Get 1sigma uncertainty assuming independent uncertainties

    t = (df.end_date_min - df.begin_date_min).dt.days / 365.25
    mwe_per_year = mwe / t
    sigma_mwe_per_year = sigma_mwe / t
    df = df.assign(mwe_per_year=mwe_per_year, sigma_mwe_per_year=sigma_mwe_per_year)

    return df


def outlines_path():
    shp_file = "/home/gossarda/Téléchargements/PGO_data/transfer_12853201_files_c8b76236/c3s_gi_rgi11_s2_2015_v2.shp"
    return shp_file


def csv_correspondence_file():
    csv_correspondence = "/home/gossarda/Téléchargements/PGO_data/transfer_12853201_files_c8b76236/correspondance_GLACIER_NR_rgi_id.csv"
    return csv_correspondence


def pgo_target_file():
    geo_file = "/home/gossarda/Téléchargements/PGO_data/transfer_12853201_files_c8b76236/merge_final_by_glacier_WGMS_versionGeoid_rgi2016_CEU_2m_with_rgiid.csv"
    return geo_file


def _find_corresponding_custom_id_in_rgi(rgi_gdf, custom_gdf):
    assert rgi_gdf.crs == custom_gdf.crs
    custom = custom_gdf.to_crs(custom_gdf.crs)
    rgi = rgi_gdf.to_crs(rgi_gdf.crs)

    rgi_small = rgi[["RGIId", "geometry"]].copy()
    custom_small = custom[["RGIId", "geometry"]].copy()
    custom_small = custom_small.rename(columns={"RGIId": "custom_id"})
    custom_small = custom_small.reset_index(drop=True)
    custom_small["geom_area"] = custom_small.geometry.area

    # --- Precompute TOTAL area per custom_id, across the whole dataset ---
    # (this is the denominator for containment fraction - full glacier area,
    #  regardless of how many pieces it's split into or where they fall)
    id_total_area = (
        custom_small.groupby("custom_id")["geom_area"].sum().rename("id_total_area")
    )

    # --- Spatial index on custom geometries ---
    custom_sindex = custom_small.sindex

    MIN_CONTAINMENT = 0.2

    matches = []

    for _, rgi_row in rgi_small.iterrows():
        rgi_geom = rgi_row.geometry
        rgi_id = rgi_row["RGIId"]

        candidate_idx = list(custom_sindex.intersection(rgi_geom.bounds))
        if not candidate_idx:
            continue
        candidates = custom_small.iloc[candidate_idx]

        # compute intersection area per candidate geometry
        rows = []
        for _, cust_row in candidates.iterrows():
            cust_geom = cust_row.geometry
            if not rgi_geom.intersects(cust_geom):
                continue
            inter_area = rgi_geom.intersection(cust_geom).area
            if inter_area > 0:
                rows.append((cust_row["custom_id"], inter_area))

        if not rows:
            continue

        # aggregate intersection area per custom_id within this RGI polygon
        inter_df = pd.DataFrame(rows, columns=["custom_id", "inter_area"])
        inter_agg = inter_df.groupby("custom_id")["inter_area"].sum()

        # join against the *global* total area for that custom_id
        for custom_id, inter_area in inter_agg.items():
            total_area = id_total_area.loc[custom_id]
            containment_frac = inter_area / total_area if total_area > 0 else 0

            if containment_frac >= MIN_CONTAINMENT:
                matches.append(
                    {
                        "RGIId": rgi_id,
                        "custom_id": custom_id,
                        "containment_frac": containment_frac,
                        "intersection_area": inter_area,
                        "custom_id_total_area": total_area,
                    }
                )

    matches_df = pd.DataFrame(matches)
    return matches_df


def table_RGI62_to_PGO(region_id):
    if not isinstance(region_id, str):
        region_id = f"{region_id:02d}"
    grid_path = os.path.join(data_path, "grids", "PGO")
    region_folder = region_id_folders(region_id, "7")
    path_region_folder = os.path.join(grid_path, region_folder)
    save_path = os.path.abspath(os.path.join(path_region_folder, f"RGI62_to_PGO.csv"))
    p = Product(save_path)
    if not p.is_up_to_date():

        df_target = geodetic_target_PGO(pgo_target_file())
        rgi_ids = df_target.RGIId.unique()
        custom_outlines = prepare_PGO_outlines(
            outlines_path(), csv_correspondence_file(), rgi_ids_to_keep=rgi_ids
        )

        # Build glacier directories to retrieve the merged dataframe with the custom outlines
        gdirs, custom_gdf = prepare_custom_glaciers_mask(custom_outlines)

        assert all(
            [
                int(rgi_id.split("-")[3]) == int(region_id)
                for rgi_id in custom_gdf.RGIId.values
            ]
        ), f"Not all entries in the custom outlines are in RGI region {region_id}."
        shp_path = get_region_shape_file(region_id)

        # Get RGI62 outlines
        rgi_gdf = gpd.read_file(shp_path)

        # Compute mapping
        matches_df = _find_corresponding_custom_id_in_rgi(rgi_gdf, custom_gdf)
        matches_df.to_csv(save_path)

        p.gen_chk()
    else:
        matches_df = pd.read_csv(save_path)

    return matches_df
