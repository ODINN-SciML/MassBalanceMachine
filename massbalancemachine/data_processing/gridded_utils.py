import os
import pandas as pd
import numpy as np
import tqdm
import multiprocessing
import xarray as xr
from calendar import month_abbr

from oggm import utils

from data_processing.Dataset import Dataset
from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders, mbm_path, data_path
from data_processing.get_topo_data import (
    glacier_cell_area,
    get_glacier_mask,
    get_custom_glacier_mask,
)
from data_processing.pgo import (
    prepare_PGO_outlines,
    prepare_custom_glaciers_mask,
    outlines_path,
    csv_correspondence_file,
)
from data_processing.glacier_utils import (
    create_glacier_grid_RGI,
    create_dem_file_RGI,
    create_custom_dem_file,
    generate_svf_file,
)
from data_processing.utils.data_preprocessing import get_hash


def create_gridded_features_PGO(
    cfg,
    time_ranges,
    multi=True,
):
    rgi_ids = list(time_ranges.keys())
    custom_outlines = prepare_PGO_outlines(
        outlines_path(), csv_correspondence_file(), rgi_ids_to_keep=rgi_ids
    )

    grid_path = os.path.join(data_path, "grids", "PGO")

    reprocess = False
    products = {}
    rgi_id_to_years = {}
    for rgi_id in rgi_ids:
        path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
        start, end = time_ranges[rgi_id]
        year_start = pd.Timestamp(start).year
        offset = (
            1 if (pd.Timestamp(end).month == 1) and (pd.Timestamp(end).day == 1) else 0
        )  # No need to generate year of end if this corresponds to the 1st of January
        year_end = pd.Timestamp(end).year - offset
        years = range(year_start, year_end + 1)
        rgi_id_to_years[rgi_id] = years
        products[rgi_id] = {}
        for year in years:
            save_path = os.path.abspath(os.path.join(path_rgi_id, f"{year}.parquet"))
            products[rgi_id][year] = Product(save_path)

        # Add sky view factor product
        svf_file = os.path.join(path_rgi_id, "svf.nc")
        products[rgi_id]["svf"] = Product(svf_file)

        if any([not p.is_up_to_date() for p in products[rgi_id].values()]):
            reprocess = True

    if reprocess:
        gdirs, rgidf = prepare_custom_glaciers_mask(custom_outlines)
    else:
        return

    rgi_ids_gdirs = [gdir.rgi_id for gdir in gdirs]
    assert all(
        [rgi_id in rgi_ids_gdirs for rgi_id in rgi_ids]
    ), "Not all keys in time_ranges have an associated glacier directory. There is probably a bug in the processing somewhere."

    gdirs.sort(key=lambda gdir: gdir.rgi_id)
    for gdir in gdirs:
        rgi_id = gdir.rgi_id
        region_id = int(rgi_id.split("-")[3])
        path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))

        if all([p.is_up_to_date() for p in products[rgi_id].values()]):
            print(f"All gridded products are already generated for {rgi_id}")
            continue

        # Check if sky view factor needs to be generated
        p = products[rgi_id]["svf"]
        if not p.is_up_to_date():

            # Create DEM grid
            create_custom_dem_file(gdir, path_rgi_id)

            # Generate sky view factor
            generate_svf_file(path_rgi_id)

            p.gen_chk()

        # Get glacier mask from OGGM
        ds, glacier_indices = get_custom_glacier_mask(gdir)

        years = rgi_id_to_years[rgi_id]
        with tqdm.tqdm(total=len(years)) as pbar:
            if multi:
                # Create a pool of workers
                with multiprocessing.Pool(processes=7) as pool:
                    for year in pool.imap_unordered(
                        create_gridded_features_from_mask_per_year,
                        [
                            (
                                rgi_id,
                                year,
                                region_id,
                                cfg,
                                ds,
                                glacier_indices,
                                gdir,
                                path_rgi_id,
                            )
                            for year in years
                        ],
                    ):
                        pbar.update(1)  # Update progress bar
                        pbar.set_description(
                            "%s: Generating gridded data for %i" % (rgi_id, year)
                        )  # Update description
            else:
                for year in years:
                    create_gridded_features_from_mask_per_year(
                        (
                            rgi_id,
                            year,
                            region_id,
                            cfg,
                            ds,
                            glacier_indices,
                            gdir,
                            path_rgi_id,
                        )
                    )
                    pbar.update(1)  # Update progress bar
                    pbar.set_description(
                        "%s: Generating gridded data for %i" % (rgi_id, year)
                    )  # Update description


def create_gridded_features_RGI(
    cfg,
    rgi_ids,
    years=range(2000, 2020),
    multi=True,
):
    grid_path = os.path.join(data_path, "grids", "Hugonnet21")
    for rgi_id in rgi_ids:
        region_id = int(rgi_id.split("-")[1].split(".")[0])

        products = {}
        path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
        for year in years:
            save_path = os.path.abspath(os.path.join(path_rgi_id, f"{year}.parquet"))
            products[year] = Product(save_path)

        # Add sky view factor product
        svf_file = os.path.join(path_rgi_id, "svf.nc")
        products["svf"] = Product(svf_file)

        if all([p.is_up_to_date() for p in products.values()]):
            # print(f"All gridded products are already generated for {rgi_id}")
            continue

        # Check if sky view factor needs to be generated
        p = products["svf"]
        if not p.is_up_to_date():

            # Create DEM grid
            create_dem_file_RGI(cfg, rgi_id, path_rgi_id)

            # Generate sky view factor
            generate_svf_file(path_rgi_id)

            p.gen_chk()

        # Get glacier mask from OGGM
        ds, glacier_indices, gdir = get_glacier_mask(rgi_id, "", cfg)

        with tqdm.tqdm(total=len(years)) as pbar:
            if multi:
                # Create a pool of workers
                with multiprocessing.Pool(processes=7) as pool:
                    for year in pool.imap_unordered(
                        create_gridded_features_from_mask_per_year,
                        [
                            (
                                rgi_id,
                                year,
                                region_id,
                                cfg,
                                ds,
                                glacier_indices,
                                gdir,
                                path_rgi_id,
                            )
                            for year in years
                        ],
                    ):
                        pbar.update(1)  # Update progress bar
                        pbar.set_description(
                            "%s: Generating gridded data for %i" % (rgi_id, year)
                        )  # Update description
            else:
                for year in years:
                    create_gridded_features_from_mask_per_year(
                        (
                            rgi_id,
                            year,
                            region_id,
                            cfg,
                            ds,
                            glacier_indices,
                            gdir,
                            path_rgi_id,
                        )
                    )


def create_gridded_features_from_mask_per_year(args):
    rgi_id, year, region_id, cfg, ds, glacier_indices, gdir, path_rgi_id = args
    try:
        save_path = os.path.abspath(os.path.join(path_rgi_id, f"{year}.parquet"))
        p = Product(save_path)

        if not p.is_up_to_date():

            # Load sky view factor
            svf = xr.open_dataset(os.path.join(path_rgi_id, "svf.nc"))

            # Create glacier grid
            df_grid = create_glacier_grid_RGI(
                ds,
                [year],
                glacier_indices,
                gdir,
                rgi_id,
                ds_svf=svf,
                calendar_year=True,
            )

            dataset_grid = Dataset(
                cfg=cfg,
                data=df_grid,
                region_name="",
                region_id=region_id,
            )

            # Climate columns
            vois_climate = [
                "t2m",
                "tp",
                "slhf",
                "sshf",
                "ssrd",
                "fal",
                "str",
                "u10",
                "v10",
            ]
            # Topographical columns
            voi_topographical = [
                "aspect",
                "slope",
                "hugonnet_dhdt",
                "consensus_ice_thickness",
                "millan_v",
                "topo",
                "svf",
            ]
            # Some glaciers do not have velocity data
            # Or depending on the product we want to generate, we do not necessarily have all variables available
            # For example with PGO, the glacier grid does not come from an official RGI
            if "hugonnet_dhdt" not in df_grid.columns:
                voi_topographical.remove("hugonnet_dhdt")
            if "consensus_ice_thickness" not in df_grid.columns:
                voi_topographical.remove("consensus_ice_thickness")
            if "millan_v" not in df_grid.columns:
                voi_topographical.remove("millan_v")
            del df_grid  # Free up memory

            # Add climate data
            dataset_grid.get_climate_features(
                change_units=True,
                smoothing_vois={
                    "vois_climate": vois_climate,
                    "vois_other": ["ALTITUDE_CLIMATE"],
                },
            )

            df_grid_y = dataset_grid.data[dataset_grid.data.YEAR == year]

            dataset_grid_yearly = Dataset(
                cfg=cfg, data=df_grid_y, region_name="", region_id=region_id
            )
            del df_grid_y  # Free up memory

            # Convert to monthly time resolution
            dataset_grid_yearly.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate,
                vois_topographical=voi_topographical,
            )

            # Save the dataset for the specific year
            data = dataset_grid_yearly.data.loc[
                :, ~dataset_grid_yearly.data.columns.duplicated()
            ]
            data.to_parquet(p.file_path, engine="pyarrow", compression="snappy")
            del data  # Free up memory

            p.gen_chk()
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        raise Exception(
            "Exception occurred during gridded features generation. Look at the traceback above."
        )
    return year


def geodetic_input_PGO(rgi_id, time_range):
    assert (
        len(time_range) == 1
    ), "Only one geodetic target per glacier is supported for the moment."
    grid_path = os.path.join(data_path, "grids", "PGO")
    start_date, end_date = time_range[0]
    year_start = pd.Timestamp(start_date).year
    offset = (
        1
        if (pd.Timestamp(end_date).month == 1) and (pd.Timestamp(end_date).day == 1)
        else 0
    )  # No need to generate year of end if this corresponds to the 1st of January
    year_end = pd.Timestamp(end_date).year - offset
    years = range(year_start, year_end + 1)
    month_to_id = {
        month_abbr[i].lower() + ("_" if i > 9 else ""): i for i in range(1, 13)
    }

    df_X_geod = pd.DataFrame()
    maxId = -1
    for year in years:
        file_path = os.path.abspath(
            os.path.join(grid_path, *rgi_id_to_folders(rgi_id), f"{year}.parquet")
        )
        df_grid = pd.read_parquet(file_path)
        if year == year_start or year == year_end:
            df_grid["MONTHS_NUM"] = df_grid.MONTHS.map(lambda month: month_to_id[month])
            if year == year_start:
                df_grid = df_grid[df_grid.MONTHS_NUM >= pd.Timestamp(start_date).month]
            elif year == year_end:
                df_grid = df_grid[df_grid.MONTHS_NUM < pd.Timestamp(end_date).month]
            df_grid = df_grid.drop(columns=["MONTHS_NUM"])

        # Remap ID so that one ID covers only one year
        df_grid["ID"] = df_grid["ID"] + maxId + 1

        # df_grid["GLWD_M_ID"] = df_grid.apply(
        #     lambda x: get_hash(f"{rgi_id}_{year}_{x.MONTHS}"),
        #     axis=1,
        # ).astype(str)
        df_grid["GLWD_M_ID"] = f"{rgi_id}_{year}_" + df_grid.MONTHS

        # Append to the final dataframe
        df_X_geod = pd.concat([df_X_geod, df_grid], ignore_index=True)

        # Update the ID counter
        maxId = df_X_geod.ID.max()

    # df_X_geod["GLWD_ID"] = df_X_geod.apply(
    #     lambda x: get_hash(f"{rgi_id}"),
    #     axis=1,
    # ).astype(str)
    df_X_geod["GLWD_ID"] = f"{rgi_id}"

    df_X_geod["aspect"] = 180 * df_X_geod["aspect"] / np.pi
    df_X_geod["slope"] = 180 * df_X_geod["slope"] / np.pi

    return df_X_geod


def geodetic_input_Hugonnet21(
    rgi_id,
    years=range(2000, 2020),
):
    grid_path = os.path.join(data_path, "grids", "Hugonnet21")

    df_X_geod = pd.DataFrame()
    maxId = -1
    for year in years:
        file_path = os.path.abspath(
            os.path.join(grid_path, *rgi_id_to_folders(rgi_id), f"{year}.parquet")
        )
        df_grid = pd.read_parquet(file_path)

        # Remap ID so that one ID covers only one year
        df_grid["ID"] = df_grid["ID"] + maxId + 1

        # df_grid["GLWD_M_ID"] = df_grid.apply(
        #     lambda x: get_hash(f"{rgi_id}_{year}_{x.MONTHS}"),
        #     axis=1,
        # ).astype(str)
        df_grid["GLWD_M_ID"] = f"{rgi_id}_{year}_" + df_grid.MONTHS

        # Append to the final dataframe
        df_X_geod = pd.concat([df_X_geod, df_grid], ignore_index=True)

        # Update the ID counter
        maxId = df_X_geod.ID.max()

    # df_X_geod["GLWD_ID"] = df_X_geod.apply(
    #     lambda x: get_hash(f"{rgi_id}"),
    #     axis=1,
    # ).astype(str)
    df_X_geod["GLWD_ID"] = f"{rgi_id}"

    df_X_geod["aspect"] = 180 * df_X_geod["aspect"] / np.pi
    df_X_geod["slope"] = 180 * df_X_geod["slope"] / np.pi

    return df_X_geod


def geodetic_target_Hugonnet21(rgi_ids, cfg):
    period_range = 20
    mbdf = utils.get_geodetic_mb_dataframe()
    geo_target_data = {}
    for rgi_id in rgi_ids:
        glacier_geo_mb_data = mbdf.loc[rgi_id]
        data = glacier_geo_mb_data[
            glacier_geo_mb_data.period == "2000-01-01_2020-01-01"
        ]
        assert len(data) == 1
        data = data.iloc[0]

        # 1. Convert to mass equivalent
        # density_ice = 916.7  # kg/m³
        density_water = 1000  # kg/m³
        area = data.area  # glacier area m²
        # print(f"{area=}")
        dmdtda = data.dmdtda  # m.w.e. / year
        # print(f"{dmdtda=}")
        V_water = dmdtda * area * period_range  # m³ of water equivalent
        # print(f"{V_water=}")
        m = V_water * density_water  # kg
        # print(m)

        # # 2. Retrieve the cell area of the geodetic grid
        # cell_area = glacier_cell_area(rgi_id, "", cfg)

        # 3. Convert to point-wise meter water equivalent (m.w.e.)
        # cumulative_pmb = V_water / cell_area # cumulative m.w.e.
        # mean_pmb = cumulative_pmb / period_range # mean m.w.e. / year
        cumulative_pmb = V_water / area  # cumulative m.w.e.
        mean_pmb = cumulative_pmb / period_range  # mean m.w.e. / year

        # 4. Do the same for the error
        # err_dmdtda = data.err_dmdtda
        # err_V_water = err_dmdtda * area * period_range
        # err_cumulative_pmb = err_V_water / cell_area
        # err_pmb = err_cumulative_pmb / period_range
        err_pmb = data.err_dmdtda

        geo_target_data[rgi_id] = {"mean": mean_pmb, "err": err_pmb, "area": area}

    return geo_target_data

    # # 3. Convert to meter snow equivalent (m.s.e.)
    # V_ice = m / density_ice # m³ of snow equivalent
    # cumulative_pmb = V_ice / cell_area # cumulative m.s.e.
    # mean_pmb = cumulative_pmb / period_range # mean m.s.e.

    # return mean_pmb

    # annual_pred = ...
    # cell_area = abs( np.diff(nds.x).mean() * np.diff(nds.y).mean() )
    # total_area = (nds.hugonnet_dhdt*0+1).sum().data*cell_area
    # print(f"{total_area=}")
    # sum_dhdt = nds.hugonnet_dhdt.sum().data * cell_area # m.s.e. * m² / year
    # print(f"{sum_dhdt=}")
    # V_ice = sum_dhdt * 20 # m³ of snow equivalent
    # print(f"{V_ice=}")
    # mass_change = V_ice * density_ice # kg
    # print(f"{mass_change=}")


def geodetic_target_region_Hugonnet21(region_id, cfg, thres_area=None):
    mbdf = utils.get_geodetic_mb_dataframe()
    ind = mbdf.index.str.contains("RGI60-%02d." % region_id)
    reg_mbdf = mbdf[ind]
    reg_mbdf = reg_mbdf[reg_mbdf.period == "2000-01-01_2020-01-01"]
    reg_mbdf = reg_mbdf[
        reg_mbdf.is_cor == False
    ]  # Remove data which has been corrected
    if thres_area is not None:
        reg_mbdf = reg_mbdf[reg_mbdf.area > thres_area]
    rgi_ids = reg_mbdf.index.values

    return geodetic_target_Hugonnet21(rgi_ids, cfg)


def generate_grid_multi_years(rgi_id, years, product_source):
    assert product_source in ["Hugonnet21", "PGO"]
    path_prepared = os.path.join(
        data_path,
        "grids_multiyears",
        product_source,
        "_".join([str(y) for y in years]),
        *(rgi_id_to_folders(rgi_id)[:-1]),
    )
    path_prepared_df = os.path.join(path_prepared, f"{rgi_id}.parquet")
    p = Product(path_prepared_df)
    if not p.is_up_to_date():
        os.makedirs(path_prepared, exist_ok=True)
        if product_source == "Hugonnet21":
            df_X_geod_rgi_id = geodetic_input_Hugonnet21(rgi_id, years=years)
        elif product_source == "PGO":
            df_X_geod_rgi_id = geodetic_input_PGO(rgi_id, years=years)
        df_X_geod_rgi_id.to_parquet(
            path_prepared_df, engine="pyarrow", compression="snappy"
        )
        p.gen_chk()


def load_grid_multi_years(rgi_id, years, product_source):
    assert product_source in ["Hugonnet21", "PGO"]
    path_prepared = os.path.join(
        data_path,
        "grids_multiyears",
        product_source,
        "_".join([str(y) for y in years]),
        *(rgi_id_to_folders(rgi_id)[:-1]),
    )
    path_prepared_df = os.path.join(path_prepared, f"{rgi_id}.parquet")
    # TODO: determine these based on features and metadata
    columns = [
        "ALTITUDE_CLIMATE",
        "POINT_LAT",
        "POINT_BALANCE",
        "N_MONTHS",
        "PERIOD",
        "POINT_ELEVATION",
        "ID",
        "YEAR",
        "svf",
        "MONTHS",
        "slope",
        "RGIId",
        "ELEVATION_DIFFERENCE",
        "aspect",
        "POINT_LON",
        "t2m",
        "tp",
        "slhf",
        "sshf",
        "ssrd",
        "fal",
        "str",
        "u10",
        "v10",
        "GLWD_M_ID",
        "GLWD_ID",
    ]
    df = pd.read_parquet(path_prepared_df, columns=columns)
    return df
