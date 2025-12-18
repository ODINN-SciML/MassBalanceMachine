"""
The VeryPoorlyNamedClass class is part of the massbalancemachine package and is in charge
of handling the train/validation/test datasets. It acts as a layer on top of the
Dataset class. It automatically preprocess data depending if it has been
preprocessed and saved to the disk beforehand or not.

Date Created: 03/11/2025
"""

import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import pandas as pd

from data_processing.Dataset import Dataset
from data_processing.utils import build_head_tail_pads_from_monthly_df, get_hash
from dataloader.DataLoader import DataLoader, set_dataloader_splits
import config

###
from regions.Switzerland.scripts.glamos_preprocess import getStakesData
from regions.Switzerland.scripts.config_CH import (
    path_PMB_GLAMOS_csv,
    path_ERA5_raw,
    path_pcsr,
)
from regions.Switzerland.scripts.helpers import (
    process_or_load_data,
)

###

import pdb


_default_test_glaciers_switzerland = [
    "tortin",
    "plattalva",
    "sanktanna",
    "schwarzberg",
    "hohlaub",
    "pizol",
    "corvatsch",
    "tsanfleuron",
    "forno",
]
_default_train_glaciers_switzerland = [
    "clariden",
    "oberaar",
    "otemma",
    "gietro",
    "rhone",
    "silvretta",
    "gries",
    "sexrouge",
    "allalin",
    "corbassiere",
    "aletsch",
    "joeri",
    "basodino",
    "morteratsch",
    "findelen",
    "albigna",
    "gorner",
    "murtel",
    "plainemorte",
    "adler",
    "limmern",
    "schwarzbach",
]


_default_additional_var = [
    "ALTITUDE_CLIMATE",
    "ELEVATION_DIFFERENCE",
    "POINT_ELEVATION",
    "pcsr",
]
_default_vois_climate = [
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
_default_vois_topographical = [
    "aspect_sgi",
    "slope_sgi",
    "hugonnet_dhdt",
    "consensus_ice_thickness",
    "millan_v",
]
_default_input = (
    _default_additional_var + _default_vois_climate + _default_vois_topographical
)


_default_test_glaciers_iceland = []
_default_train_glaciers_iceland = ["RGI60-06.00228", "RGI60-06.00232"]


class VeryPoorlyNamedClass:
    def __init__(self, cfg, params, test_split_on="GLACIER"):
        self.cfg = cfg
        self.params = params
        # self.train_glaciers = []
        # self.test_glaciers =  []
        if len(self.test_glaciers) > 0:
            self.glacierNamesProvided = not all(
                e.startswith("RGI") for e in self.test_glaciers
            )
        elif len(self.train_glaciers) > 0:
            self.glacierNamesProvided = not all(
                e.startswith("RGI") for e in self.train_glaciers
            )
        else:
            self.glacierNamesProvided = False
        self.test_split_on = test_split_on
        self.glaciers_list = []
        self.mean_stakes_elevation = {}

    def train_test_glaciers(self):
        return self.train_glaciers, self.test_glaciers

    def load_stakes_data(self):
        raise NotImplemented(
            "This function must be implemented by the child class of VeryPoorlyNamedClass."
        )

    def train_test_sets(self):
        data_monthly = self.load_stakes_data()
        self.glaciers_list = data_monthly[
            "GLACIER" if self.glacierNamesProvided else "RGIId"
        ].unique()
        self.mean_stakes_elevation = (
            data_monthly.groupby("GLACIER" if self.glacierNamesProvided else "RGIId")[
                "POINT_ELEVATION"
            ]
            .mean()
            .to_dict()
        )
        months_head_pad, months_tail_pad = build_head_tail_pads_from_monthly_df(
            data_monthly
        )
        dataloader = DataLoader(
            self.cfg,
            data=data_monthly,
            random_seed=self.cfg.seed,
            meta_data_columns=self.cfg.metaData,
        )

        # Ensure all test glaciers exist in the dataset
        existing_glaciers = set(
            dataloader.data.GLACIER.unique()
            if self.glacierNamesProvided
            else dataloader.data.RGIId.unique()
        )
        missing_glaciers = [g for g in self.test_glaciers if g not in existing_glaciers]

        if missing_glaciers:
            print(
                f"Warning: The following test glaciers are not in the dataset: {missing_glaciers}"
            )

        # TODO: add some prints and checks

        # Split between train and test sets
        train_set, test_set = set_dataloader_splits(
            dataloader,
            test_split_on=self.test_split_on,
            test_splits=self.test_glaciers,
            random_state=self.cfg.seed,
        )
        return train_set, test_set, months_head_pad, months_tail_pad


class VeryPoorlyNamedClassSwitzerland(VeryPoorlyNamedClass):
    def __init__(self, cfg, params, *args, **kwargs):
        self.train_glaciers = (
            params["training"].get("train_glaciers")
            or _default_train_glaciers_switzerland
        )
        self.test_glaciers = (
            params["training"].get("test_glaciers")
            or _default_test_glaciers_switzerland
        )
        super().__init__(cfg, params, *args, **kwargs)

    def load_stakes_data(self):
        # TODO: determine this flag based on existing files
        ###########
        csvFileName = "CH_wgms_dataset_monthly_NN_nongeo.csv"
        process = False
        ###########

        data_glamos = getStakesData(self.cfg)
        data_glamos.drop(
            data_glamos[data_glamos.GLACIER == "taelliboden"].index, inplace=True
        )
        downscale = self.params["model"]["downscale"]
        if downscale is not None:
            assert False, "The downscale option is not supported yet."
            # assert downscale == "linear"

        vois_climate = self.params["model"].get("vois_climate", _default_vois_climate)
        vois_topographical = self.params["model"].get(
            "vois_topographical", _default_vois_topographical
        )
        paths = {
            "csv_path": self.cfg.dataPath + path_PMB_GLAMOS_csv,
            "era5_climate_data": self.cfg.dataPath
            + path_ERA5_raw
            + "era5_monthly_averaged_data.nc",
            "geopotential_data": self.cfg.dataPath
            + path_ERA5_raw
            + "era5_geopotential_pressure.nc",
            "radiation_save_path": self.cfg.dataPath + path_pcsr + "zarr/",
        }

        # Transform data to monthly format (run or load data)
        data_monthly = process_or_load_data(
            run_flag=process,
            data_glamos=data_glamos,
            paths=paths,
            cfg=self.cfg,
            vois_climate=vois_climate,
            vois_topographical=vois_topographical,
            output_file=csvFileName,
        )

        data_monthly["GLWD_ID"] = data_monthly.apply(
            lambda x: get_hash(f"{x.GLACIER}_{x.YEAR}"), axis=1
        )
        data_monthly["GLWD_ID"] = data_monthly["GLWD_ID"].astype(str)

        return data_monthly


class VeryPoorlyNamedClassIceland(VeryPoorlyNamedClass):
    def __init__(self, cfg, params, *args, **kwargs):
        self.train_glaciers = (
            params["training"].get("train_glaciers") or _default_train_glaciers_iceland
        )
        self.test_glaciers = (
            params["training"].get("test_glaciers") or _default_test_glaciers_iceland
        )
        super().__init__(cfg, params, *args, **kwargs)

    def load_stakes_data(self):
        # TODO: for the moment the arguments are the RGIId but we should manage this properly in the future
        data = pd.read_csv(
            os.path.join(
                mbm_path,
                "notebooks/example_data/iceland/files/iceland_monthly_dataset.csv",
            )
        )

        ### Add period column ###
        data = data.assign(PERIOD="")
        for ID in data[data.N_MONTHS <= 8].ID.unique():
            sub = data[data.ID == ID]
            months = sub.MONTHS.to_numpy()
            if "jan" in months:
                data.loc[data.ID == ID, "PERIOD"] = "winter"
            elif "jul" in months:
                data.loc[data.ID == ID, "PERIOD"] = "summer"
        data.loc[data.N_MONTHS > 8, "PERIOD"] = "annual"
        assert data.PERIOD.nunique() == 3
        #########################

        existing_glaciers = set(data.RGIId.unique())
        missing_glaciers = [g for g in self.test_glaciers if g not in existing_glaciers]
        if missing_glaciers:
            print(
                f"Warning: The following test glaciers are not in the dataset: {missing_glaciers}"
            )

        train_glaciers = [i for i in existing_glaciers if i not in self.test_glaciers]

        dataloader = DataLoader(
            self.cfg,
            data=data,
            random_seed=self.cfg.seed,
            meta_data_columns=self.cfg.metaData,
        )
        data_monthly = dataloader.data
        # months_head_pad, months_tail_pad = (
        #     mbm.data_processing.utils.build_head_tail_pads_from_monthly_df(data_monthly)
        # )

        # data_test = dataloader.data[dataloader.data.RGIId.isin(test_glaciers)]
        # print("Size of monthly test data:", len(data_test))

        # data_train = dataloader.data[dataloader.data.RGIId.isin(train_glaciers)]
        # print("Size of monthly train data:", len(data_train))

        # if len(data_train) == 0:
        #     print("Warning: No training data available!")
        # else:
        #     test_perc = (len(data_test) / len(data_train)) * 100
        #     print("Percentage of test size: {:.2f}%".format(test_perc))

        return data_monthly

    # # Split on measurements (IDs)
    # test_set, train_set = get_CV_splits_iceland(
    #     dataloader,
    #     test_split_on="RGIId",
    #     test_splits=test_glaciers,
    #     random_state=cfg.seed,
    # )
    # return train_set, test_set, months_head_pad, months_tail_pad
