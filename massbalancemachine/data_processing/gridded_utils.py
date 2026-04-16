import os
import pandas as pd
import tqdm
import multiprocessing
import xarray as xr

from oggm import utils

from data_processing.Dataset import Dataset
from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders, mbm_path, data_path
from data_processing.get_topo_data import (
    glacier_cell_area,
    get_glacier_mask,
)
from data_processing.glacier_utils import (
    create_glacier_grid_RGI,
    create_dem_file_RGI,
    generate_svf_file,
)
from data_processing.utils.data_preprocessing import get_hash


def create_gridded_features_RGI(
    cfg,
    rgi_ids,
    years=range(2000, 2020),
    multi=True,
):
    grid_path = os.path.join(data_path, "grids")
    for rgi_id in rgi_ids:
        region_id = int(rgi_id.split("-")[1].split(".")[0])

        products = {}
        path_rgi_id = os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
        for year in years:
            save_path = os.path.abspath(os.path.join(path_rgi_id, f"{year}.parquet"))
            products[year] = Product(save_path)

        # Add clear sky radiation product
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
                ds, [year], glacier_indices, gdir, rgi_id, ds_svf=svf
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
            if "millan_v" not in df_grid.columns:
                # Some glaciers do not have velocity data
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


def geodetic_input(
    rgi_id,
    years=range(2000, 2020),
):
    grid_path = os.path.join(data_path, "grids")

    df_X_geod = pd.DataFrame()
    maxId = -1
    for year in years:
        file_path = os.path.abspath(
            os.path.join(grid_path, *rgi_id_to_folders(rgi_id), f"{year}.parquet")
        )
        df_grid = pd.read_parquet(file_path)

        # Remap ID so that one ID covers only one year
        df_grid["ID"] = df_grid["ID"] + maxId + 1

        df_grid["GLWD_ID"] = df_grid.apply(
            lambda x: get_hash(f"{rgi_id}_{year}"),
            axis=1,
        ).astype(str)

        # Append to the final dataframe
        df_X_geod = pd.concat([df_X_geod, df_grid], ignore_index=True)

        # Update the ID counter
        maxId = df_X_geod.ID.max()

    return df_X_geod


def geodetic_target(rgi_ids, cfg):
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
        density_ice = 916.7  # kg/m³
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


def geodetic_target_region(region_id, cfg, thres_area=None):
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

    return geodetic_target(rgi_ids, cfg)
