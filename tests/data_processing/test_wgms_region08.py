"""Curation of the WGMS stake measurements of Scandinavia (RGI 08). Skipped when the
WGMS Fluctuations of Glaciers database is not in the data tree."""

import glob
import os

import pandas as pd
import pytest

from data_processing.product_utils import get_data_path

FOG = glob.glob(
    os.path.join(
        get_data_path(), "WGMS", "DOI-WGMS-FoG-*", "data", "mass_balance_point.csv"
    )
)


@pytest.mark.skipif(not FOG, reason="WGMS FoG not available")
def test_region08_curation():
    import data_preprocessing.wgms as wgms
    from data_processing.wgms import load_processed_wgms

    points = wgms.prepare_points(load_processed_wgms(rgi_region=8), 8)
    winter = points[points.PERIOD == "winter"]
    # No winter balance from one autumn to the next
    assert (winter.MONTH_DIFF <= 10).all()
    # Storglaciären 2023 and Riukojietna 2022: annual dates from the neighbouring years
    sg = points[
        (points.RGIId == "RGI60-08.00213")
        & (points.YEAR == 2023)
        & (points.PERIOD == "annual")
    ]
    assert len(sg) and (sg.TO_DATE == "20230903").all()
    rk = points[
        (points.RGIId == "RGI60-08.00006")
        & (points.YEAR == 2022)
        & (points.PERIOD == "annual")
    ]
    assert len(rk) and (rk.FROM_DATE == "20210911").all()
    # Juvfonne: kept, relabelled winter and annual
    juv = points[points.RGIId == "RGI60-08.00868"]
    assert len(juv) == 10 and set(juv.PERIOD) == {"winter", "annual"}
    assert (pd.to_datetime(juv[juv.PERIOD == "winter"].TO_DATE).dt.month <= 6).all()
