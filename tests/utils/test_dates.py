import os
import pytest
import numpy as np
import pandas as pd
import massbalancemachine as mbm
from massbalancemachine.data_processing.utils import _compute_head_tail_pads_from_df
from massbalancemachine.data_processing.utils.hydro_year import assign_hydro_year


@pytest.mark.order(1)
def test_padding():
    # 1. Exact hydrological year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20241001"],
            "TO_DATE": ["20250930"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == []
    assert months_tail_pad == []

    # 2. Shifted one month in advance
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240901"],
            "TO_DATE": ["20250831"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == []
    assert months_tail_pad == ["sep_"]

    # 3. Shifted one month late
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20241101"],
            "TO_DATE": ["20251031"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_"]
    assert months_tail_pad == []

    # 4. One month before and after
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240901"],
            "TO_DATE": ["20251031"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_"]
    assert months_tail_pad == ["sep_"]

    # 5. Start one month late and spring of next year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20241101"],
            "TO_DATE": ["20260331"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_", "nov_", "dec_", "jan_", "feb_", "mar_"]
    assert months_tail_pad == []

    # 6. Start one month in advance and spring of next year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240901"],
            "TO_DATE": ["20260331"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_", "nov_", "dec_", "jan_", "feb_", "mar_"]
    assert months_tail_pad == ["sep_"]

    # 7. Accumulation season only
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20241001"],
            "TO_DATE": ["20250531"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == []
    assert months_tail_pad == []

    # 8. Accumulation season only shifted in advance
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240901"],
            "TO_DATE": ["20250531"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == []
    assert months_tail_pad == ["sep_"]

    # 9. Ablation season only
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20250601"],
            "TO_DATE": ["20250930"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == []
    assert months_tail_pad == []

    # 10. Ablation season only shifted late
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20250601"],
            "TO_DATE": ["20251031"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_"]
    assert months_tail_pad == []

    # 11. Incomplete months with previous month included
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240914"],
            "TO_DATE": ["20251015"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_"]
    assert months_tail_pad == ["sep_"]

    # 12. Incomplete months with previous month discarded
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240915"],
            "TO_DATE": ["20251015"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_"]
    assert months_tail_pad == []

    # 13. Calendar year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20250101"],
            "TO_DATE": ["20251231"],
        }
    )
    months_head_pad, months_tail_pad = _compute_head_tail_pads_from_df(df)
    assert months_head_pad == ["oct_", "nov_", "dec_"]
    assert months_tail_pad == []


@pytest.mark.order(1)
def test_generate_monthly_ranges():
    # 1. Start before and end after hydrological year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240814"],
            "TO_DATE": ["20251116"],
        }
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")
    from massbalancemachine.data_processing.transform_to_monthly import (
        _generate_monthly_ranges,
    )

    months = _generate_monthly_ranges(df)["MONTHS"][0]
    assert months == [
        "aug_",
        "sep_",
        "oct",
        "nov",
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct_",
        "nov_",
    ]

    # 2. Start before and end after hydrological year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240816"],
            "TO_DATE": ["20251101"],
        }
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")
    from massbalancemachine.data_processing.transform_to_monthly import (
        _generate_monthly_ranges,
    )

    months = _generate_monthly_ranges(df)["MONTHS"][0]
    assert months == [
        "sep_",
        "oct",
        "nov",
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct_",
    ]

    # 3. Start after and end after hydrological year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20241023"],
            "TO_DATE": ["20251116"],
        }
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")
    from massbalancemachine.data_processing.transform_to_monthly import (
        _generate_monthly_ranges,
    )

    months = _generate_monthly_ranges(df)["MONTHS"][0]
    assert months == [
        "nov",
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct_",
        "nov_",
    ]

    # 4. Start before and end before hydrological year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20240905"],
            "TO_DATE": ["20250923"],
        }
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")
    from massbalancemachine.data_processing.transform_to_monthly import (
        _generate_monthly_ranges,
    )

    months = _generate_monthly_ranges(df)["MONTHS"][0]
    assert months == [
        "sep_",
        "oct",
        "nov",
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
    ]

    # 5. Calendar year
    df = pd.DataFrame(
        {
            "FROM_DATE": ["20250101"],
            "TO_DATE": ["20251231"],
        }
    )
    df["FROM_DATE"] = pd.to_datetime(df["FROM_DATE"].astype(str), format="%Y%m%d")
    df["TO_DATE"] = pd.to_datetime(df["TO_DATE"].astype(str), format="%Y%m%d")
    from massbalancemachine.data_processing.transform_to_monthly import (
        _generate_monthly_ranges,
    )

    months = _generate_monthly_ranges(df)["MONTHS"][0]
    assert months == [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct_",
        "nov_",
        "dec_",
    ]


def test_assign_hydro_year():
    row = pd.Series(
        {"FROM_DATE": pd.Timestamp("2022-11-01"), "TO_DATE": pd.Timestamp("2025-02-28")}
    )
    assert assign_hydro_year(row) == 2024

    row = pd.Series(
        {"FROM_DATE": pd.Timestamp("2023-07-01"), "TO_DATE": pd.Timestamp("2023-11-30")}
    )
    assert assign_hydro_year(row) == 2023


if __name__ == "__main__":
    test_padding()
    test_generate_monthly_ranges()
    test_assign_hydro_year()
