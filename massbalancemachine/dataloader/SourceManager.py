"""
The SourceManager class is part of the massbalancemachine package and is in charge
of handling the train/validation/test datasets. It acts as a layer on top of the
Dataset class. It automatically preprocess data depending if it has been
preprocessed and saved to the disk beforehand or not.

Date Created: 03/11/2025
"""

import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import pandas as pd
import numpy as np
import datetime
import urllib
import json
import git

# from data_processing.Dataset import Dataset
# from data_processing.utils import build_head_tail_pads_from_monthly_df, get_hash
from data_processing.utils.hydro_year import build_head_tail_pads_from_monthly_df
from data_processing.utils.data_preprocessing import get_hash
from data_processing.Product import Product
from dataloader.DataLoader import DataLoader, set_dataloader_splits
import data_preprocessing.iceland
import data_preprocessing.wgms
import data_processing.wgms
import config

###
from regions.RGI_11_Switzerland.scripts.dataset.data_loader import (
    process_or_load_data,
    get_stakes_data,
)
from regions.RGI_11_Switzerland.scripts.config_CH import (
    path_PMB_GLAMOS_csv,
    path_ERA5_raw,
    path_pcsr,
)

###

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
_default_input_switzerland = (
    _default_additional_var + _default_vois_climate + _default_vois_topographical
)

_default_test_glaciers_iceland = []
_default_train_glaciers_iceland = [
    "RGI60-06.00228",
    "RGI60-06.00232",
    "RGI60-06.00236",
    "RGI60-06.00238",
]

_default_test_glaciers_norway = [
    "RGI60-08.01258",
    "RGI60-08.01026",
    "RGI60-08.02384",
    "RGI60-08.01598",
    "RGI60-08.01484",
    "RGI60-08.02650",
    "RGI60-08.00434",
    "RGI60-08.01286",
    "RGI60-08.00449",
    "RGI60-08.01013",
    "RGI60-08.02916",
    "RGI60-08.02918",
    "RGI60-08.02920",
    "RGI60-08.02969",
]
_default_train_glaciers_norway = [
    "RGI60-08.02436",
    "RGI60-08.02458",
    "RGI60-08.00287",
    "RGI60-08.01657",
    "RGI60-08.00295",
    "RGI60-08.02666",
    "RGI60-08.02017",
    "RGI60-08.01126",
    "RGI60-08.01186",
    "RGI60-08.01217",
    "RGI60-08.00868",
    "RGI60-08.00987",
    "RGI60-08.00966",
    "RGI60-08.01779",
    "RGI60-08.02966",
    "RGI60-08.02963",
    "RGI60-08.02962",
    "RGI60-08.02643",
]


def _format_data_credit(source, usage_conditions, papers):
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(
        "------------------------------------------------------------------------------------------------------------------"
    )
    print(
        "||  Your are using an external dataset. Please respect the usage conditions and cite the scientific references! ||"
    )
    print(BOLD + "Source: " + RESET + source)
    print(BOLD + "Data usage conditions: " + RESET + usage_conditions)
    if isinstance(papers, (list, tuple)):
        if len(papers) > 0:
            print(BOLD + "Papers:" + RESET)
            for p in papers:
                print("- " + p)
    else:
        print(BOLD + "Paper: " + RESET + papers)
    print(
        "------------------------------------------------------------------------------------------------------------------"
    )
    print()


def _default_input(sourceData):
    if sourceData == "switzerland":
        return _default_input_switzerland
    elif sourceData == "iceland":
        return []
    elif sourceData == "norway":
        return []
    elif "wgms" in sourceData:
        return []
    else:
        raise ValueError(f"source_data={sourceData} is unknown")


class SourceManager:
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

    # def train_test_glaciers(self):
    #     return self.train_glaciers, self.test_glaciers

    def load_stakes_data(self):
        raise NotImplemented(
            "This function must be implemented by the child class of SourceManager."
        )

    def train_test_sets(self):
        data_monthly = self.load_stakes_data()
        self.glaciers_list = data_monthly[
            "GLACIER" if self.glacierNamesProvided else "RGIId"
        ].unique()
        self.mean_stakes_elevation = (
            data_monthly.groupby("GLACIER" if self.glacierNamesProvided else "RGIId")[
                (
                    "POINT_ELEVATION"
                    if "POINT_ELEVATION" in data_monthly.keys()
                    else "ELEVATION_DIFFERENCE"
                )
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

        existing_glaciers = set(
            dataloader.data.GLACIER.unique()
            if self.glacierNamesProvided
            else dataloader.data.RGIId.unique()
        )

        if self.test_glaciers is None:
            assert (
                self.train_glaciers is not None
            ), "If test_glaciers is not defined, train_glaciers should be defined to automatically determine the test glaciers based on available data."
            self.test_glaciers = list(
                set(existing_glaciers).difference(self.train_glaciers)
            )
            print(
                f"Determining test glaciers based on available data: {self.test_glaciers}"
            )
        if self.train_glaciers is None:
            assert (
                self.test_glaciers is not None
            ), "If train_glaciers is not defined, test_glaciers should be defined to automatically determine the train glaciers based on available data."
            self.train_glaciers = list(
                set(existing_glaciers).difference(self.test_glaciers)
            )
            print(
                f"Determining train glaciers based on available data: {self.train_glaciers}"
            )

        # Ensure all test glaciers exist in the dataset
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


class SourceManagerSwitzerland(SourceManager):
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

        data_glamos = get_stakes_data(self.cfg)
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


class SourceManagerIceland(SourceManager):
    def __init__(self, cfg, params, *args, light=False, **kwargs):
        self.train_glaciers = (
            params["training"].get("train_glaciers") or _default_train_glaciers_iceland
        )
        self.test_glaciers = (
            params["training"].get("test_glaciers") or _default_test_glaciers_iceland
        )
        super().__init__(cfg, params, *args, **kwargs)
        self.light = light

    def load_stakes_data(self):
        # TODO: for the moment the arguments are the RGIId but we should manage this properly in the future
        if self.light:
            data = pd.read_csv(
                os.path.join(
                    mbm_path,
                    "notebooks/example_data/iceland/files/iceland_monthly_dataset.csv",
                )
            )
        else:
            p = Product(data_preprocessing.iceland.processed_features_stakes_path)
            if not p.is_up_to_date():
                data = data_preprocessing.iceland.raw_data()
                data_preprocessing.iceland.build_monthly_data(data, self.cfg)
                p.gen_chk()
            data = pd.read_csv(
                data_preprocessing.iceland.processed_features_stakes_path
            )
            _format_data_credit(
                "https://icelandicglaciers.is/#/page/map",
                "CC-BY license with following references provided for the mass balance data",
                [
                    "Finnur Pálsson. 2022. Vatnajökull. Mass balance, meltwater drainage and surface velocity of the glacial year 2021–22. Reykjavík, Landsvirkjun, Institute of Earth Sciences University of Iceland, Rep. LV-2022-054. https://gogn.lv.is/files/2022/2022-054.pdf",
                    "Finnur Pálsson. 2022. Afkomu- og hraðamælingar á Langjökli jökulárið 2021–2022 (Mass balance and ice flow measurements of Langjökull in 2021–2022). Reykjavík, Landsvirkjun, Institute of Earth Sciences University of Iceland, Rep. LV-2022-053. https://gogn.lv.is/files/2022/2022-053.pdf",
                    "Helgi Björnsson, Finnur Pálsson, Magnús Tumi Guðmundsson and Hannes H. Haraldsson. 1998. Mass balance of western and northern Vatnajökull, Iceland, 1991–1995, Jökull, 45, 35–58, https://doi.org/10.33799/jokull1998.45.035",
                    "Þorsteinn Þorsteinsson, Tómas Jóhannesson, Bergur Einarsson, Vilhjálmur S. Kjartansson. 2017. Afkomumælingar á Hofsjökli 1988–2017 [Mass balance measurements on Hofsjökull 1988–2017]. Reykjavík, Veðurstofa Íslands, skýrsla 2017-016. https://www.vedur.is/media/vedurstofan-utgafa-2017/2017_016_hofsjokull30_rs.pdf",
                    "Guðfinna Aðalgeirsdóttir, Eyjólfur Magnússon, Finnur Pálsson, Þorsteinn Þorsteinsson, JMC Belart, Tómas Jóhannesson, Hrafnhildur Hannesdóttir, Oddur Sigurðsson, Andri Gunnarsson, Bergur Einarsson, E Berthier, LS Schmidt, Hannes H. Haraldsson, Helgi Björnsson. 2020. Glacier changes in Iceland in the 20th and the beginning of the 21st century. Frontiers of Earth Sciences, 8, 523646, https://doi.org/10.3389/feart.2020.523646",
                ],
            )

        data["aspect"] = 180 * data["aspect"] / np.pi
        data["slope"] = 180 * data["slope"] / np.pi
        data["t2m"] = data["t2m"] - 273.15

        if self.light:
            # Add period column for the light data, for the full dataset this is already done in the preprocessing
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


class SourceManagerNorway(SourceManager):
    def __init__(self, cfg, params, *args, **kwargs):
        self.train_glaciers = (
            params["training"].get("train_glaciers") or _default_train_glaciers_norway
        )
        self.test_glaciers = (
            params["training"].get("test_glaciers") or _default_test_glaciers_norway
        )
        super().__init__(cfg, params, *args, **kwargs)

        url_monthly_dataset_train = "https://raw.githubusercontent.com/khsjursen/ML_MB_Norway/refs/heads/main/src/Data/2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate_svf_monthly.csv"
        url_monthly_dataset_test = "https://raw.githubusercontent.com/khsjursen/ML_MB_Norway/refs/heads/main/src/Data/2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate_test_svf.csv"
        folder_csv = os.path.abspath(os.path.join(mbm_path, ".data/stakes/norway/"))
        self.path_csv_train = os.path.abspath(
            os.path.join(
                folder_csv,
                "2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate_svf_monthly.csv",
            )
        )
        self.path_csv_test = os.path.abspath(
            os.path.join(
                folder_csv,
                "2023-08-28_stake_mb_norway_cleaned_ids_latlon_wattributes_climate_test_svf.csv",
            )
        )
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        path_info_download = os.path.abspath(os.path.join(folder_csv, "checksum.json"))
        commit_match = False
        if os.path.isfile(path_info_download):
            with open(path_info_download, "r") as f:
                d = json.load(f)
                # if d.get("commit_hash") == commit_hash:
                if (
                    True
                ):  # TODO: change this, for the moment we do not redownload but we should build a more robust system dependencies in the future
                    commit_match = True
        if not os.path.isfile(self.path_csv_train) or not commit_match:
            print("Downloading monthly train CSV file")
            if not os.path.isdir(folder_csv):
                os.makedirs(folder_csv, exist_ok=True)
            urllib.request.urlretrieve(url_monthly_dataset_train, self.path_csv_train)

            info = {
                "commit_hash": commit_hash,
                "date": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S%z"
                ),
            }
            with open(path_info_download, "w") as f:
                json.dump(info, f, indent=4, sort_keys=True)

        if not os.path.isfile(self.path_csv_test) or not commit_match:
            print("Downloading monthly test CSV file")
            if not os.path.isdir(folder_csv):
                os.makedirs(folder_csv, exist_ok=True)
            urllib.request.urlretrieve(url_monthly_dataset_test, self.path_csv_test)

            info = {
                "commit_hash": commit_hash,
                "date": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S%z"
                ),
            }
            with open(path_info_download, "w") as f:
                json.dump(info, f, indent=4, sort_keys=True)

    def load_stakes_data(self):
        data_train = pd.read_csv(self.path_csv_train)
        data_train.drop(["Unnamed: 0", "BREID"], axis=1, inplace=True)
        data_train.rename(
            columns={
                "id": "ID",
                "RGIID": "RGIId",
                "year": "YEAR",
                "altitude_diff": "ELEVATION_DIFFERENCE",
                "balance": "POINT_BALANCE",
                "skyview_factor": "svf",
                "n_months": "N_MONTHS",
                "month": "MONTHS",
            },
            inplace=True,
        )
        data_train["POINT_ELEVATION"] = 0.0
        data_test = pd.read_csv(self.path_csv_test)
        data_test.drop(
            ["Unnamed: 0", "BREID", "altitude_climate"], axis=1, inplace=True
        )
        data_test.rename(
            columns={
                "id": "ID",
                "RGIID": "RGIId",
                "altitude": "POINT_ELEVATION",
                "year": "YEAR",
                "altitude_diff": "ELEVATION_DIFFERENCE",
                "balance": "POINT_BALANCE",
                "skyview_factor": "svf",
                "n_months": "N_MONTHS",
                "month": "MONTHS",
            },
            inplace=True,
        )
        data_test["ID"] = data_test["ID"] + data_train["ID"].max() + 1

        # In the Norway data altitude_diff is the opposite of what we are actually computing with ELEVATION_DIFFERENCE in MBM
        # See https://github.com/khsjursen/ML_MB_Norway/blob/32d8175adab27963c6ca2766f2420f24cdb72a6b/src/scripts/data_processing.py#L96
        data_train["ELEVATION_DIFFERENCE"] = -data_train["ELEVATION_DIFFERENCE"]
        data_test["ELEVATION_DIFFERENCE"] = -data_test["ELEVATION_DIFFERENCE"]

        # Apply some transformations
        data_train["aspect"] = 180 * data_train["aspect"] / np.pi
        data_train["slope"] = 180 * data_train["slope"] / np.pi
        data_train["t2m"] = data_train["t2m"] - 273.15
        data_test["aspect"] = 180 * data_test["aspect"] / np.pi
        data_test["slope"] = 180 * data_test["slope"] / np.pi
        data_test["t2m"] = data_test["t2m"] - 273.15

        # Add period column to train data
        data_train = data_train.assign(PERIOD="")
        for ID in data_train[data_train.N_MONTHS <= 8].ID.unique():
            sub = data_train[data_train.ID == ID]
            months = sub.MONTHS.to_numpy()
            if "jan" in months:
                data_train.loc[data_train.ID == ID, "PERIOD"] = "winter"
            elif "jul" in months:
                data_train.loc[data_train.ID == ID, "PERIOD"] = "summer"
        data_train.loc[data_train.N_MONTHS > 8, "PERIOD"] = "annual"
        assert data_train.PERIOD.nunique() == 3

        # Add period column to test data
        data_test = data_test.assign(PERIOD="")
        for ID in data_test[data_test.N_MONTHS <= 8].ID.unique():
            sub = data_test[data_test.ID == ID]
            months = sub.MONTHS.to_numpy()
            if "jan" in months:
                data_test.loc[data_test.ID == ID, "PERIOD"] = "winter"
            elif "jul" in months:
                data_test.loc[data_test.ID == ID, "PERIOD"] = "summer"
        data_test.loc[data_test.N_MONTHS > 8, "PERIOD"] = "annual"
        assert data_test.PERIOD.nunique() == 3

        data = pd.concat([data_train, data_test]).reset_index(drop=True)

        dataloader = DataLoader(
            self.cfg,
            data=data,
            random_seed=self.cfg.seed,
            meta_data_columns=self.cfg.metaData,
        )
        data_monthly = dataloader.data

        return data_monthly


class SourceManagerWGMS(SourceManager):
    def __init__(self, cfg, params, *args, rgi_region=None, **kwargs):
        self.train_glaciers = params["training"].get("train_glaciers")
        self.test_glaciers = params["training"].get("test_glaciers")
        self.rgi_region = rgi_region
        super().__init__(cfg, params, *args, **kwargs)

    def load_stakes_data(self):
        # TODO: for the moment the arguments are the RGIId but we should manage this properly in the future

        path_preprocessed = data_preprocessing.wgms.processed_features_stakes_path(
            self.rgi_region
        )
        p = Product(path_preprocessed)
        if not p.is_up_to_date():
            data = data_processing.wgms.load_processed_wgms(rgi_region=self.rgi_region)
            data_preprocessing.wgms.build_monthly_data(data, self.cfg, self.rgi_region)
            p.gen_chk()
        data = pd.read_csv(path_preprocessed)
        _format_data_credit(
            "WGMS (2026): Fluctuations of Glaciers (FoG) Database. World Glacier Monitoring Service (WGMS), Zurich, Switzerland. https://doi.org/10.5904/wgms-fog-2026-02-10",
            "Open access under the requirement of correct citation",
            [
                "WGMS (2023): Global Glacier Change Bulletin No. 5 (2020-2021). Michael Zemp, Isabelle Gärtner-Roer, Samuel U. Nussbaumer, Ethan Z. Welty, Inés Dussaillant, and Jacqueline Bannwart (eds.), ISC (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland, 134 pp. Based on database version https://doi.org/10.5904/wgms-fog-2023-09.",
                "WGMS (2013): Glacier Mass Balance Bulletin No. 12 (2010-2011). Michael Zemp, Samuel U. Nussbaumer, Kathrin Naegeli, Isabelle Gärtner-Roer, Frank Paul, Martin Hoelzle, and Wilfried Haeberli (eds.), ICSU (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland, 106 pp. Based on database version https://doi.org/10.5904/wgms-fog-2013-11.",
                "WGMS (2012): Fluctuations of Glaciers 2005-2010 (Vol. X): Michael Zemp, Holger Frey, Isabelle Gärtner-Roer, Samuel U. Nussbaumer, Martin Hoelzle, Frank Paul, and Wilfried Haeberli (eds.), ICSU (WDS) / IUGG (IACS) / UNEP / UNESCO / WMO, World Glacier Monitoring Service, Zurich, Switzerland. Based on database version https://doi.org/10.5904/wgms-fog-2012-11.",
            ],
        )

        data["aspect"] = 180 * data["aspect"] / np.pi
        data["slope"] = 180 * data["slope"] / np.pi
        data["t2m"] = data["t2m"] - 273.15

        dataloader = DataLoader(
            self.cfg,
            data=data,
            random_seed=self.cfg.seed,
            meta_data_columns=self.cfg.metaData,
        )
        data_monthly = dataloader.data

        return data_monthly
