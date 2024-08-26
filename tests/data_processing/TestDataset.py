import pytest
import pandera as pa
import pandas as pd
from massbalancemachine import data_processing
from pandera import Column, DataFrameSchema


df_schema_wgms = pa.DataFrameSchema({
    "YEAR": Column(),
    "POINT_ID": Column(),
    "POINT_LAT": Column(),
    "POINT_LON": Column(),
    "POINT_ELEVATION": Column(),
    "TO_DATE": Column(),
    "FROM_DATE": Column(),
    "POINT_BALANCE": Column(),
})

class TestDataset:

