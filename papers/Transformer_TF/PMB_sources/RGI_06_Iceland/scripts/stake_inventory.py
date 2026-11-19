"""
Inventory of the Icelandic stakes used in the paper.

The Icelandic stake data are downloaded from the live IMO API (`1.0`), which has
no versions. This script lists, for every IMO stake in the processed dataset
(`ISL_wgms_dataset_all.csv`, written by `1.1`), the number of annual and winter
measurements and the years covered, so that users can check that a new download
gives the same data.

The processed dataset renames the stakes by elevation (`POINT_ID`), so each
processed measurement is linked back to its IMO stake ID through its coordinates,
which are taken unchanged from the raw stake files.

Usage (from the repository root):
    python -m papers.Transformer_TF.PMB_sources.RGI_06_Iceland.scripts.stake_inventory
"""

import glob
import os

import pandas as pd

import massbalancemachine as mbm
from papers.Transformer_TF.PMB_sources.RGI_06_Iceland.scripts.config_ICE import *

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "ISL_stake_inventory.csv")


def _coord_key(lat, lon, decimals=5):
    return lat.round(decimals).astype(str) + "_" + lon.round(decimals).astype(str)


def build_stake_inventory(raw_dir, processed_csv):
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    raw["key"] = _coord_key(raw["lat"], raw["lon"])
    # one coordinate can in principle be shared by several stakes: keep all IDs
    key_to_stake = (
        raw.groupby("key")["stake"]
        .agg(lambda s: ";".join(sorted(set(s.astype(str)))))
        .rename("IMO_STAKE_ID")
    )

    df = pd.read_csv(processed_csv)
    df["key"] = _coord_key(df["POINT_LAT"], df["POINT_LON"])
    df = df.join(key_to_stake, on="key")
    n_unmatched = df["IMO_STAKE_ID"].isna().sum()
    if n_unmatched:
        print(f"[warn] {n_unmatched} processed measurements not matched to a stake")
    df["IMO_STAKE_ID"] = df["IMO_STAKE_ID"].fillna("unmatched")

    inv = (
        df.groupby(["IMO_STAKE_ID", "GLACIER", "RGIId"])
        .agg(
            N_ANNUAL=("PERIOD", lambda p: int((p == "annual").sum())),
            N_WINTER=("PERIOD", lambda p: int((p == "winter").sum())),
            FIRST_YEAR=("YEAR", "min"),
            LAST_YEAR=("YEAR", "max"),
        )
        .reset_index()
    )
    inv["N_TOTAL"] = inv["N_ANNUAL"] + inv["N_WINTER"]
    inv[["FIRST_YEAR", "LAST_YEAR"]] = inv[["FIRST_YEAR", "LAST_YEAR"]].astype(int)
    return inv.sort_values(["GLACIER", "IMO_STAKE_ID"]).reset_index(drop=True)


if __name__ == "__main__":
    cfg = mbm.IcelandConfig()
    inv = build_stake_inventory(
        raw_dir=os.path.join(cfg.dataPath, path_PMB_WGMS_raw),
        processed_csv=os.path.join(
            cfg.dataPath, path_PMB_WGMS_csv, "ISL_wgms_dataset_all.csv"
        ),
    )
    inv.to_csv(OUT_CSV, index=False)
    print(
        f"{len(inv)} stakes, {inv['GLACIER'].nunique()} glaciers, "
        f"{inv['N_ANNUAL'].sum()} annual + {inv['N_WINTER'].sum()} winter = "
        f"{inv['N_TOTAL'].sum()} measurements -> {os.path.abspath(OUT_CSV)}"
    )
