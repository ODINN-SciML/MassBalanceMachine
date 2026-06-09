import os
import shutil
import urllib.request
import zipfile
import pandas as pd
import tqdm
import multiprocessing
import xarray as xr
import oggm
from data_processing.Dataset import Dataset
from data_processing.Product import Product
from data_processing.product_utils import rgi_id_to_folders, mbm_path, data_path
from data_processing.oggm_utils import _initialize_oggm_config
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

wgms_zip_file = "DOI-WGMS-FoG-2026-02-10.zip"

wgms_source_data_link = f"https://wgms.ch/downloads/{wgms_zip_file}"
local_path_wgms = f"{data_path}/WGMS/{wgms_zip_file}"

wgms_folder = f"{data_path}/WGMS/{wgms_zip_file.replace('.zip', '')}"


def _clean_extracted_wgms():
    if os.path.isdir(wgms_folder):
        shutil.rmtree(wgms_folder)


def check_and_download_wgms():
    os.makedirs(f"{data_path}/WGMS/", exist_ok=True)
    if not os.path.isdir(wgms_folder):
        if not os.path.isfile(local_path_wgms):
            print("Downloading data from WGMS website")
            urllib.request.urlretrieve(wgms_source_data_link, local_path_wgms)
        print("Unzipping WGMS archive")
        with zipfile.ZipFile(local_path_wgms, "r") as zip_ref:
            zip_ref.extractall(wgms_folder)


def load_wgms_data():
    """
    Load WGMS data and enrich mass balance data with rgi_region.

    Returns:
        pd.DataFrame: mass balance data with added 'rgi_region' column
    """
    check_and_download_wgms()

    point_mb_file = f"{wgms_folder}/data/mass_balance_point.csv"
    glacier_file = f"{wgms_folder}/data/glacier.csv"

    data_mb = pd.read_csv(point_mb_file)
    data_glacier = pd.read_csv(glacier_file)

    # Build mapping: id -> rgi_region (extract number before "_")
    mapping = data_glacier.assign(
        rgi_region=data_glacier["gtng_region"].str.split("_").str[0].astype(int)
    ).set_index("id")["rgi_region"]

    # Apply mapping to data_mb
    data_mb["rgi_region"] = data_mb["glacier_id"].map(mapping)

    return data_mb


def parse_wgms_format(data_mb):
    """
    Converts the WGMS point balance DataFrame to a dataframe ready to be used by MBM Data preparation notebook.

    Args:
        df_pb (pd.DataFrame): dataframe loaded by load_wgms_data "mass_balance_point.csv" from WGMS.
    Returns:
        pd.DataFrame
    """

    new_df = data_mb.drop(
        columns=[
            "country",
            "glacier_name",
            "original_id",
            "glacier_id",
            "time_system",
            "begin_date_unc",
            "end_date_unc",
            "balance_unc",
            "density",
            "density_unc",
            "method",
            "remarks",
        ]
    )
    new_df = new_df.rename(
        columns={
            "id": "ID",
            "year": "YEAR",
            "balance": "POINT_BALANCE",
            "latitude": "POINT_LAT",
            "longitude": "POINT_LON",
            "elevation": "POINT_ELEVATION",
            "begin_date": "FROM_DATE",
            "end_date": "TO_DATE",
            "balance_code": "PERIOD",
        },
    )
    assert new_df.ID.nunique() == new_df.shape[0], "It seems that ID are not unique"

    new_df["FROM_DATE"] = pd.to_datetime(new_df["FROM_DATE"]).dt.strftime("%Y%m%d")
    new_df["TO_DATE"] = pd.to_datetime(new_df["TO_DATE"]).dt.strftime("%Y%m%d")

    return new_df


def filter_dates(df):
    # Remove points for which the dates have a too large uncertainty
    threshold_date_uncertainty = 5
    filtered_df = df[
        (df.end_date_unc <= threshold_date_uncertainty)
        & (df.begin_date_unc <= threshold_date_uncertainty)
    ]

    return filtered_df


def load_processed_wgms(rgi_region=None):
    check_and_download_wgms()
    df = load_wgms_data()
    df = filter_dates(df)
    df = parse_wgms_format(df)
    if rgi_region is not None:
        df = df.loc[df.rgi_region == rgi_region]
    return df


def _prepare_glacier_wide_mb(rgi_ids):
    """Load glacier-wide MB data from the WGMS database and map `glacier_id` to the RGI ID"""
    # 1. Map the RGIId to WGMS ID
    _initialize_oggm_config("")
    gdirs = oggm.workflow.init_glacier_directories(
        rgi_ids,
        from_prepro_level=5,
        prepro_base_url=oggm.DEFAULT_BASE_URL,
        prepro_border=80,
    )
    rgi_id_to_wgms_id = {}
    wgms_id_to_rgi_id = {}
    for gdir in gdirs:
        df = gdir.get_ref_mb_data()
        assert df.WGMS_ID.nunique() == 1
        wgms_id = df.WGMS_ID.unique()[0]
        rgi_id_to_wgms_id[gdir.rgi_id] = wgms_id
        wgms_id_to_rgi_id[wgms_id] = gdir.rgi_id

    # 2. Read glacier wide data from WGMS
    check_and_download_wgms()

    mb_file = f"{wgms_folder}/data/mass_balance.csv"
    glacier_file = f"{wgms_folder}/data/glacier.csv"

    data_mb = pd.read_csv(mb_file)
    data_glacier = pd.read_csv(glacier_file)

    # Build mapping: id -> rgi_region (extract number before "_")
    mapping = data_glacier.assign(
        rgi_region=data_glacier["gtng_region"].str.split("_").str[0].astype(int)
    ).set_index("id")["rgi_region"]

    # Apply mapping to data_mb
    data_mb["rgi_region"] = data_mb["glacier_id"].map(mapping)
    glacier_ids = [rgi_id_to_wgms_id[rgi_id] for rgi_id in rgi_ids]
    df = (
        data_mb[data_mb.glacier_id.isin(glacier_ids)]
        .drop(
            columns=[
                "country",
                "outline_id",
                "ela_position",
                "ela",
                "ela_unc",
                "aar",
                "investigators",
                "agencies",
                "midseason_date",
                "midseason_date_unc",
                "winter_balance",
                "winter_balance_unc",
                "summer_balance",
                "summer_balance_unc",
            ]
        )
        .reset_index()
    )
    df["RGIId"] = df.glacier_id.map(lambda glacier_id: wgms_id_to_rgi_id[glacier_id])

    # 3. Format to the same format as the stakes data
    df = filter_dates(df)
    df = df.rename(
        columns={
            "year": "YEAR",
            "annual_balance": "GLWD_BALANCE",
            "begin_date": "FROM_DATE",
            "end_date": "TO_DATE",
        },
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"]).dt.strftime("%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"]).dt.strftime("%Y%m%d")
    df["PERIOD"] = "annual"

    df = df.drop(
        columns=["time_system", "begin_date_unc", "end_date_unc", "annual_balance_unc"]
    )

    for rgi_id in rgi_ids:
        years = df[df.RGIId == rgi_id].YEAR.unique()

    return df


def load_glacier_wide_annual_mb(rgi_ids, cfg, multi=True):

    grid_path = os.path.join(data_path, "grids", "WGMS")
    path_rgi_ids = {
        rgi_id: os.path.join(grid_path, *rgi_id_to_folders(rgi_id))
        for rgi_id in rgi_ids
    }

    # Check table data
    products = {}
    for rgi_id, path_rgi_id in path_rgi_ids.items():
        table_file = os.path.join(path_rgi_id, "table_data.csv")
        p = Product(table_file)
        products[rgi_id] = p

    if not all([p.is_up_to_date() for p in products.values()]):
        df = _prepare_glacier_wide_mb(rgi_ids)
        for rgi_id, path_rgi_id in path_rgi_ids.items():
            p = products[rgi_id]
            if not p.is_up_to_date():
                table_file = os.path.join(path_rgi_id, "table_data.csv")
                df_gl = df[df.RGIId == rgi_id]
                df_gl.to_csv(table_file, index=False)
                p.gen_chk()
        del df
    del products

    years_per_rgi_id = {}
    # Generate products if needed
    for rgi_id, path_rgi_id in path_rgi_ids.items():
        region_id = int(rgi_id.split("-")[1].split(".")[0])

        table_file = os.path.join(path_rgi_id, "table_data.csv")
        df_gl = pd.read_csv(table_file)
        years = df_gl.YEAR.unique()
        years_per_rgi_id[rgi_id] = years.tolist()
        products = {}
        for year in years:
            sel_year = df_gl[df_gl.YEAR == year]
            assert (
                sel_year.shape[0] == 1
            ), f"Expected to find only one entry for {rgi_id} and year {year} in the WGMS database but found {sel_year.shape[0]}."
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
                                df_gl,
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
                            df_gl,
                            ds,
                            glacier_indices,
                            gdir,
                            path_rgi_id,
                        )
                    )

    df_annual = pd.DataFrame()
    for rgi_id, path_rgi_id in path_rgi_ids.items():

        maxId = -1
        for year in years:
            file_path = os.path.abspath(os.path.join(path_rgi_id, f"{year}.parquet"))
            df_grid = pd.read_parquet(file_path)

            # Remap ID so that one ID covers only one year
            df_grid["ID"] = df_grid["ID"] + maxId + 1

            df_grid["GLWD_ID"] = df_grid.apply(
                lambda x: get_hash(f"{rgi_id}_{year}"),
                axis=1,
            ).astype(str)

            # Append to the final dataframe
            df_annual = pd.concat([df_annual, df_grid], ignore_index=True)

            # Update the ID counter
            maxId = df_annual.ID.max()

    return df_annual


def create_gridded_features_from_mask_per_year(args):
    rgi_id, year, region_id, cfg, df_gl, ds, glacier_indices, gdir, path_rgi_id = args
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
            )  # We don't care if it's calendar year or not because FROM_DATE and TO_DATE are overwritten just after

            df_year = df_gl[df_gl.YEAR == year]
            df_grid["FROM_DATE"] = df_year.FROM_DATE.values[0]
            df_grid["TO_DATE"] = df_year.TO_DATE.values[0]
            df_grid["POINT_BALANCE"] = df_year.GLWD_BALANCE.values[0]
            df_grid["area"] = df_year.area.values[0]
            df_grid["PERIOD"] = df_year.PERIOD.values[0]

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

            # Add climate data
            dataset_grid.get_climate_features(
                change_units=True,
                smoothing_vois={
                    "vois_climate": vois_climate,
                    "vois_other": ["ALTITUDE_CLIMATE"],
                },
            )

            # Convert to monthly time resolution
            dataset_grid.convert_to_monthly(
                meta_data_columns=cfg.metaData,
                vois_climate=vois_climate,
                vois_topographical=voi_topographical,
            )

            # Save the dataset for the specific year
            data = dataset_grid.data.loc[:, ~dataset_grid.data.columns.duplicated()]
            data.to_parquet(p.file_path, engine="pyarrow", compression="snappy")
            del data  # Free up memory

            p.gen_chk()
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        raise Exception(
            "Exception occurred during gridded features generation. Look at the traceback above."
        )
    return year
