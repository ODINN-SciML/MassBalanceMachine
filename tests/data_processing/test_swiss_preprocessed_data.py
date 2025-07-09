import os
import pytest
import massbalancemachine as mbm
from regions.Switzerland.scripts.glamos_preprocess import get_geodetic_MB, getStakesData
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.xgb_helpers import process_or_load_data, get_CV_splits

if "CI" in os.environ:
    pathDataDownload = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../dataDownload/data/'))
    dataPath = pathDataDownload
else:
    dataPath = None

@pytest.mark.order1
def test_geodetic_data():
    cfg = mbm.SwitzerlandConfig(dataPath=dataPath)

    geodetic_mb = get_geodetic_MB(cfg)
    print("geodetic_mb.shape=",geodetic_mb.shape)
    assert geodetic_mb.shape == (292, 17)

@pytest.mark.order1
def test_process_or_load_data():
    cfg = mbm.SwitzerlandConfig(dataPath=dataPath)

    data_glamos = getStakesData(cfg)
    assert data_glamos.shape == (32574, 20)

    vois_climate = [
        't2m', 'tp', 'slhf', 'sshf', 'ssrd', 'fal', 'str', 'u10', 'v10'
    ]
    vois_topographical = [
        # "aspect", # OGGM
        # "slope", # OGGM
        "aspect_sgi",  # SGI
        "slope_sgi",  # SGI
        "hugonnet_dhdt",  # OGGM
        "consensus_ice_thickness",  # OGGM
        "millan_v",  # OGGM
    ]
    paths = {
        'csv_path': cfg.dataPath + path_PMB_GLAMOS_csv,
        'era5_climate_data': cfg.dataPath + path_ERA5_raw + 'era5_monthly_averaged_data.nc',
        'geopotential_data': cfg.dataPath + path_ERA5_raw + 'era5_geopotential_pressure.nc',
        'radiation_save_path': cfg.dataPath + path_pcsr + 'zarr/'
    }

    dataloader_gl = process_or_load_data(
        run_flag=True,
        data_glamos=data_glamos,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        output_file='CH_wgms_dataset_monthly_silvretta.csv'
    )
    assert dataloader_gl.data.shape == (284645, 30)

@pytest.mark.order2
def test_geodataloader():
    # This test needs to run after test_process_or_load_data since we use the
    # results of process_or_load_data by reading on disk
    cfg = mbm.SwitzerlandConfig(dataPath=dataPath)

    data_glamos = getStakesData(cfg)

    vois_climate = [
        't2m', 'tp', 'slhf', 'sshf', 'ssrd', 'fal', 'str', 'u10', 'v10'
    ]
    vois_topographical = [
        # "aspect", # OGGM
        # "slope", # OGGM
        "aspect_sgi",  # SGI
        "slope_sgi",  # SGI
        "hugonnet_dhdt",  # OGGM
        "consensus_ice_thickness",  # OGGM
        "millan_v",  # OGGM
    ]
    paths = {
        'csv_path': cfg.dataPath + path_PMB_GLAMOS_csv,
        'era5_climate_data': cfg.dataPath + path_ERA5_raw + 'era5_monthly_averaged_data.nc',
        'geopotential_data': cfg.dataPath + path_ERA5_raw + 'era5_geopotential_pressure.nc',
        'radiation_save_path': cfg.dataPath + path_pcsr + 'zarr/'
    }

    dataloader_gl = process_or_load_data(
        run_flag=False,
        data_glamos=data_glamos,
        paths=paths,
        cfg=cfg,
        vois_climate=vois_climate,
        vois_topographical=vois_topographical,
        output_file='CH_wgms_dataset_monthly_silvretta.csv',
    )

    data_monthly = dataloader_gl.data

    data_monthly['GLWD_ID'] = data_monthly.apply(
        lambda x: mbm.data_processing.utils.get_hash(f"{x.GLACIER}_{x.YEAR}"), axis=1)
    data_monthly['GLWD_ID'] = data_monthly['GLWD_ID'].astype(str)

    dataloader_gl = mbm.dataloader.DataLoader(cfg,
                                data=data_monthly,
                                random_seed=cfg.seed,
                                meta_data_columns=cfg.metaData)

    # Split on measurements (IDs)
    splits, test_set, train_set = get_CV_splits(dataloader_gl,
                                                test_split_on='ID',
                                                random_state=cfg.seed,
                                                test_size=0.1)

    feature_columns = list(data_monthly.columns.difference(cfg.metaData).drop(cfg.notMetaDataNotFeatures))
    cfg.setFeatures(feature_columns)

    gdl = mbm.dataloader.GeoDataLoader(cfg, ['silvretta'], train_set['df_X'])
    for g in gdl.glaciers():
        print(f"Glacier {g}")
    g = 'silvretta'
    s, m, gt = gdl.stakes(g)
    nRows = 40107
    assert s.shape == (nRows, 16)
    assert m.shape == (nRows, 8)
    assert gt.shape == (nRows, )
    x, m, y = gdl.geo(g)
    nRows = 215952
    assert x.shape == (nRows, 16)
    assert m.shape == (nRows, 8)
    assert y.shape == (50,)


if __name__ == "__main__":
    test_geodetic_data()
    test_process_or_load_data()
    test_geodataloader()
