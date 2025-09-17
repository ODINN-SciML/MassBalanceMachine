import os
import logging
import massbalancemachine as mbm
from tqdm import tqdm
from argparse import ArgumentParser

# Scripts
from regions.Switzerland.scripts.helpers import *
from regions.Switzerland.scripts.glamos_preprocess import *
from regions.Switzerland.scripts.config_CH import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def generate_pmb_df(cfg):

    # ------------------------------------------------------------------------------
    # 1. Transform seasonal and winter PMB .dat files to .csv for simplicity
    # ------------------------------------------------------------------------------

    log.info("Processing PMB .dat files to .csv")
    process_pmb_dat_files(cfg)

    # ------------------------------------------------------------------------------
    # 2. Assemble measurement periods
    # ------------------------------------------------------------------------------

    log.info("-- Processing annual measurements")
    df_annual_raw = process_annual_stake_data(cfg.dataPath + path_PMB_GLAMOS_csv_a)

    log.info("-- Processing winter measurements")
    process_winter_stake_data(
        df_annual_raw,
        cfg.dataPath + path_PMB_GLAMOS_csv_w,
        cfg.dataPath + path_PMB_GLAMOS_csv_w_clean,
    )

    log.info("-- Assembling all measurements (winter & annual)")
    df_all_raw = assemble_all_stake_data(
        df_annual_raw,
        cfg.dataPath + path_PMB_GLAMOS_csv_w_clean,
        cfg.dataPath + path_PMB_GLAMOS_csv,
    )

    # ------------------------------------------------------------------------------
    # 3. Add RGI Ids
    # ------------------------------------------------------------------------------

    log.info("Adding RGI Ids")
    df_pmb = add_rgi_ids_to_df(df_all_raw, cfg.dataPath + path_rgi_outlines)

    rgiids6 = df_pmb[["GLACIER", "RGIId"]].drop_duplicates()
    log.info("-- RGIs before pre-processing")

    if check_multiple_rgi_ids(rgiids6):
        log.info(
            "-- Alert: The following glaciers have more than one RGIId. Cleaning up."
        )
        df_pmb_clean = clean_rgi_ids(df_pmb.copy())
        df_pmb_clean.reset_index(drop=True, inplace=True)

        rgiids6_clean = df_pmb_clean[["GLACIER", "RGIId"]].drop_duplicates()
        if check_multiple_rgi_ids(rgiids6_clean):
            log.error("-- Error: Some glaciers still have more than one RGIId.")
        else:
            log.info("-- All glaciers are correctly associated with a single RGIId.")
    else:
        log.info("-- All glaciers are correctly associated with a single RGIId.")
        df_pmb_clean = df_pmb

    # ------------------------------------------------------------------------------
    # 4. Cut from 1951 (start of ERA5-Land)
    # ------------------------------------------------------------------------------

    log.info("Cutting data from 1951 (start of ERA5-Land)")
    df_pmb_50s = df_pmb_clean[df_pmb_clean.YEAR > 1950].sort_values(
        by=["GLACIER", "YEAR"], ascending=[True, True]
    )

    df_pmb_50s["POINT_BALANCE"] = df_pmb_50s["POINT_BALANCE"] / 1000

    df_pmb_50s.loc[df_pmb_50s.GLACIER == "claridenU", "GLACIER"] = "clariden"
    df_pmb_50s.loc[df_pmb_50s.GLACIER == "claridenL", "GLACIER"] = "clariden"

    # ------------------------------------------------------------------------------
    # 5. Merge stakes that are close
    # ------------------------------------------------------------------------------

    log.info("Merging stakes that are close")
    df_pmb_50s_clean = pd.DataFrame()
    for gl in tqdm(df_pmb_50s.GLACIER.unique(), desc="Merging stakes"):
        log.info(f"-- {gl.capitalize()}:")
        df_gl = df_pmb_50s[df_pmb_50s.GLACIER == gl]
        df_gl_cleaned = remove_close_points(df_gl)
        df_pmb_50s_clean = pd.concat([df_pmb_50s_clean, df_gl_cleaned])

    df_pmb_50s_clean.drop(["x", "y"], axis=1, inplace=True)

    # ------------------------------------------------------------------------------
    # Save intermediate output
    # ------------------------------------------------------------------------------
    log.info("Saving intermediate output df_pmb_50s.csv to {path_PMB_GLAMOS_csv}")
    df_pmb_50s_clean.to_csv(
        os.path.join(cfg.dataPath, path_PMB_GLAMOS_csv, "df_pmb_50s.csv"), index=False
    )

    df_pmb_50s_clean[
        ["GLACIER", "POINT_ID", "POINT_LAT", "POINT_LON", "PERIOD"]
    ].to_csv(
        os.path.join(cfg.dataPath, path_PMB_GLAMOS_csv, "coordinate_50s.csv"),
        index=False,
    )


def generate_topo_data(cfg):

    # ------------------------------------------------------------------------------
    # 6. Add OGGM topographical variables
    # ------------------------------------------------------------------------------

    df_pmb_50s_clean = pd.read_csv(
        cfg.dataPath + path_PMB_GLAMOS_csv + "df_pmb_50s.csv"
    )

    log.info("Merge with OGGM topography:")
    log.info("-- Initializing OGGM glacier directories:")
    gdirs, rgidf = initialize_oggm_glacier_directories(
        cfg,
        rgi_region="11",
        rgi_version="6",
        base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/",
        log_level="WARNING",
        task_list=None,
    )

    export_oggm_grids(cfg, gdirs)

    log.info("-- Adding OGGM topographical variables:")
    df_pmb_topo = merge_pmb_with_oggm_data(
        df_pmb=df_pmb_50s_clean,
        gdirs=gdirs,
        rgi_region="11",
        rgi_version="6",
    )

    # Remove 'pers' glacier
    # only if pers is in the dataset
    if "pers" in df_pmb_topo.GLACIER.unique():
        # Remove 'pers' glacier
        df_pmb_topo = df_pmb_topo.loc[df_pmb_topo.GLACIER != "pers"]

    # Drop points that are not within the glacier shape
    df_pmb_topo = df_pmb_topo[df_pmb_topo["within_glacier_shape"]]
    df_pmb_topo = df_pmb_topo.drop(columns=["within_glacier_shape"])

    # ------------------------------------------------------------------------------
    # 7. Add SGI topographical variables
    # ------------------------------------------------------------------------------

    log.info("Merge with SGI topography:")
    # Paths and variables of interest
    path_masked_grids = os.path.join(cfg.dataPath, path_SGI_topo, "xr_masked_grids/")

    # First create the masked topographical arrays per glacier:
    glacier_list = sorted(df_pmb_topo.GLACIER.unique())
    create_sgi_topo_masks(cfg, glacier_list)

    # Merge PMB with SGI data
    df_pmb_sgi = merge_pmb_with_sgi_data(
        df_pmb_topo,  # cleaned PMB DataFrame
        path_masked_grids,  # path to SGI grids
        voi=["masked_aspect", "masked_slope", "masked_elev"],
    )

    # ------------------------------------------------------------------------------
    # 8. Give new stake IDs
    # ------------------------------------------------------------------------------
    log.info("Renaming stake IDs:")
    # Give new stake IDs with glacier name and then a number according to the elevation.
    # This is because accross glaciers some stakes have the same ID which is not practical.
    df_pmb_sgi = rename_stakes_by_elevation(df_pmb_sgi)

    # Check the condition
    check_point_ids_contain_glacier(df_pmb_sgi)

    # Save to CSV
    fname = "CH_wgms_dataset_all.csv"
    df_pmb_sgi.to_csv(
        os.path.join(cfg.dataPath, path_PMB_GLAMOS_csv, fname), index=False
    )
    log.info(f"-- Saved final dataset {fname} to: {path_PMB_GLAMOS_csv}")

    # Check information
    log.info(f"-- Number of glaciers: {len(df_pmb_sgi.GLACIER.unique())}")
    log.info(f"-- Number of winter and annual samples: {len(df_pmb_sgi)}")
    log.info(
        f'-- Number of annual samples: {len(df_pmb_sgi[df_pmb_sgi.PERIOD == "annual"])}'
    )
    log.info(
        f'-- Number of winter samples: {len(df_pmb_sgi[df_pmb_sgi.PERIOD == "winter"])}'
    )


def main(process_pmb=True, process_topo=True):
    # ------------------------------------------------------------------------------
    # Config and Setup
    # ------------------------------------------------------------------------------

    cfg = mbm.SwitzerlandConfig()
    seed_all(cfg.seed)
    free_up_cuda()

    if process_pmb:
        generate_pmb_df(cfg)

    if process_topo:
        generate_topo_data(cfg)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--noPMB", type=bool, default=False, help="Process the point mass balance data"
    )
    parser.add_argument(
        "--noTopo", type=bool, default=False, help="Process the topographical data"
    )
    args = parser.parse_args()

    main(process_pmb=not args.noPMB, process_topo=not args.noTopo)
