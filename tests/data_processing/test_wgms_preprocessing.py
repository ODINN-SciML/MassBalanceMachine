import os
import pytest
import tempfile
import pandas as pd
import geopandas as gpd
import massbalancemachine as mbm


@pytest.mark.order1
def test_data_retrieval():
    mbm.data_processing.wgms._clean_extracted_wgms()
    mbm.data_processing.check_and_download_wgms()


@pytest.mark.order2
def test_data_preprocessing_wgms():
    df = mbm.data_processing.wgms.load_processed_wgms()
    expected_columns = [
        "YEAR",
        "ID",
        "FROM_DATE",
        "TO_DATE",
        "POINT_LAT",
        "POINT_LON",
        "POINT_ELEVATION",
        "POINT_BALANCE",
        "PERIOD",
        "rgi_region",
    ]
    assert set(expected_columns).issubset(
        set(df.columns)
    ), f"Not all features are in the dataframe. Expected {set(expected_columns)} but {set(expected_columns).difference(set(df.columns))} are missing."
    assert df.shape == (64143, 10)
    df_alps = mbm.data_processing.wgms.load_processed_wgms(rgi_region=11)
    assert df_alps.shape == (27137, 10)


if __name__ == "__main__":
    test_data_retrieval()
    test_data_preprocessing_wgms()
