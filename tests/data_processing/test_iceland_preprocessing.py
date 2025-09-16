import os
import pytest
import tempfile
import pandas as pd
import geopandas as gpd
import massbalancemachine as mbm

_wgms_data_columns = {
    "yr": "YEAR",
    "stake": "POINT_ID",
    "lat": "POINT_LAT",
    "lon": "POINT_LON",
    "elevation": "POINT_ELEVATION",
    "TO_DATE": "TO_DATE",
    "FROM_DATE": "FROM_DATE",
    "POINT_BALANCE": "POINT_BALANCE",
}


@pytest.mark.order1
def test_data_preprocessing():
    target_raw_data_fname = (
        "./notebooks/example_data/iceland/files/iceland_stake_dataset.csv"
    )
    data = pd.read_csv(target_raw_data_fname)
    assert data.shape == (19, 18)

    assert all(
        data.keys()
        == [
            "stake",
            "yr",
            "d1",
            "d2",
            "d3",
            "lat",
            "lon",
            "elevation",
            "rhow",
            "rhos",
            "bw_stratigraphic",
            "bs_stratigraphic",
            "ba_stratigraphic",
            "bw_floating_date",
            "bs_floating_date",
            "ba_floating_date",
            "GLIMSId",
            "Name",
        ]
    )

    column_names_dates = ["d1", "d2", "d3"]
    column_names_smb = ["bw_stratigraphic", "bs_stratigraphic", "ba_stratigraphic"]

    Nrows = 57

    data = mbm.data_processing.utils.convert_to_wgms(
        wgms_data_columns=_wgms_data_columns,
        data=data,
        date_columns=column_names_dates,
        smb_columns=column_names_smb,
    )
    assert all(data.keys() == list(_wgms_data_columns.values()))
    assert data.shape == (Nrows, len(_wgms_data_columns))

    data = mbm.data_processing.utils.convert_to_wgs84(data=data, from_crs=4659)
    assert all(data.keys() == list(_wgms_data_columns.values()))
    assert data.shape == (Nrows, len(_wgms_data_columns))


@pytest.mark.order2
def test_data_processing_wgms():
    cfg = mbm.Config()

    # Specify the filename of the input file with the raw data
    target_data_fname = (
        "./notebooks/example_data/iceland/files/iceland_wgms_dataset.csv"
    )

    # Load the target data
    data = pd.read_csv(target_data_fname)

    Nrows = 57
    data_path = tempfile.gettempdir() + "/MBM/"

    # Get the RGI ID for each stake measurement for the region of interest
    data = mbm.data_processing.utils.get_rgi(data=data, region=6)
    assert data.shape == (Nrows, len(_wgms_data_columns) + 1)

    dataset = mbm.data_processing.Dataset(
        cfg, data=data, region_name="iceland", region_id=6, data_path=data_path
    )

    voi_topographical = ["aspect", "slope"]
    dataset.get_topo_features(vois=voi_topographical)

    # Specify the files of the climate data
    # Setting to None automatically downloads the data using the CDSAPI
    era5_climate_data = (
        "./notebooks/example_data/iceland/climate/era5_monthly_averaged_data.nc"
    )
    geopotential_data = (
        "./notebooks/example_data/iceland/climate/era5_geopotential_pressure.nc"
    )
    dataset.get_climate_features(
        climate_data=era5_climate_data, geopotential_data=geopotential_data
    )

    vois_climate = ["t2m", "tp", "slhf", "sshf", "ssrd", "fal", "str"]
    dataset.convert_to_monthly(
        vois_climate=vois_climate, vois_topographical=voi_topographical
    )

    assert dataset.data.shape == (445, 21)


if __name__ == "__main__":
    test_data_preprocessing()
    test_data_processing_wgms()
