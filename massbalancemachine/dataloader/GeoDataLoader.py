import random
from typing import List
import numpy as np
import pandas as pd
import time
import torch
from concurrent.futures import ThreadPoolExecutor

from regions.RGI_11_Switzerland.scripts.geodetic.geodetic_processing import (
    prepare_geo_targets,
    build_periods_per_glacier,
    get_geodetic_MB,
    create_geodetic_input,
    has_geodetic_input,
)

from data_processing.Dataset import Normalizer
from data_processing.utils import _rebuild_month_index
from data_processing.gridded_utils import (
    create_gridded_features_RGI,
    geodetic_input,
    geodetic_target,
    geodetic_target_region,
)
from models.TorchNeuralNetworkRegressor import aggrMetadataId, aggrMetadataGlwdId


class GeoDataLoader:
    """
    The class that handles both stakes and geodetic data loading. It prepares the
    features and the metadata and add extra columns for the aggregations. It also
    retrieves the ground truth values.

    Args:
        cfg (config.Config): Configuration instance.
        glacierList (list of str): List of glaciers to use in the instanciated
            dataloader.
        trainStakesDf (pd.DataFrame): A pandas dataframe containing the training stake data.
        valStakesDf (pd.DataFrame): A pandas dataframe containing the validation stake data.
        ignoreStakesWithoutGeo (bool): Whether to discard stake measurements whose glacier
            don't have geodetic data.
    """

    def __init__(
        self,
        cfg,
        glacierList: List[str],
        trainStakesDf: pd.DataFrame,
        months_head_pad: list[str],
        months_tail_pad: list[str],
        valStakesDf: pd.DataFrame = None,
        glacierListVal: List[str] = [],
        ignoreStakesWithoutGeo: bool = False,
        geodeticOggm: bool = True,
        preloadGeodetic: bool = False,
        keyGlacierSel: str = "GLACIER",
        geoGlaciers: str = "stakes",
        ignoreGlaciers: list[str] = [],
        device=torch.device("cpu"),
    ) -> None:
        self.cfg = cfg
        self.glacierList = glacierList.copy()  # Copy for shuffling
        self.glacierListVal = (
            glacierListVal.copy()
        )  # Copy just in case but we don't shuffle this
        random.shuffle(self.glacierList)
        self.indGlacier = 0
        self.indGlacierVal = 0
        self.indGlacierGeo = 0
        self.periodToInt = {"annual": 0, "winter": 1, "summer": 2}

        _, self.month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

        self.trainStakesDf = trainStakesDf
        self.valStakesDf = valStakesDf
        self.geodeticOggm = geodeticOggm
        self.preloadGeodetic = preloadGeodetic
        self.keyGlacierSel = keyGlacierSel
        self.geoGlaciers = geoGlaciers
        self.ignoreGlaciers = ignoreGlaciers
        self.device = device

        if valStakesDf is not None:
            assert (
                len(glacierListVal) > 0
            ), "If validation data stakes are provided you need to provide a list of validation glaciers."
        else:
            assert (
                len(glacierListVal) == 0
            ), "If validation data stakes are not provided we don't expect a list of validation glaciers."

        # Prepare geodetic data
        self.prepareGeoData()
        self.glacierListGeo = self.glaciersWithGeo
        if ignoreStakesWithoutGeo:
            raise NotImplementedError(
                "We need to implement an intersection between glaciersWithGeo and train/validation glaciers"
            )
            self.glacierList = self.glaciersWithGeo
            self.glacierListGeo = self.glaciersWithGeo  # TODO: change this

        if len(self.glaciersWithGeo) == 1:
            if self.geodeticOggm:
                raise NotImplementedError()
            # Preload geodetic data into memory if there is only one glacier
            self.df_X_geod = create_geodetic_input(
                self.cfg,
                self.glaciersWithGeo[0],
                self.periods_per_glacier,
                to_seasonal=False,
            )
        else:
            if self.geodeticOggm and self.preloadGeodetic:
                print("Preloading geodetic grids")
                self.df_X_geod = {}
                for rgi_id in self.glaciersWithGeo:
                    self.df_X_geod[rgi_id] = geodetic_input(rgi_id)
            else:
                self.df_X_geod = None

        if self.df_X_geod is not None:
            if len(self.glaciersWithGeo) == 1:
                self.precomputed_meta = {
                    self.glaciersWithGeo[0]: self._metadata_groups(self.df_X_geod)
                }
            else:
                self.precomputed_meta = {}
                for rgi_id in self.glaciersWithGeo:
                    self.precomputed_meta[rgi_id] = self._metadata_groups(
                        self.df_X_geod[rgi_id]
                    )
        else:
            self.precomputed_meta = None

        self.normalizer = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

        self._geo_executor = ThreadPoolExecutor(max_workers=1)

    def prepareGeoData(self) -> None:
        if self.geodeticOggm:
            stakesDf = (
                self.trainStakesDf
                if self.valStakesDf is None
                else pd.concat(
                    (self.trainStakesDf, self.valStakesDf), ignore_index=True
                )
            )
            # TODO: implement this in a more clever way
            if self.geoGlaciers == "stakes":
                rgi_ids = list(stakesDf.RGIId.unique())
                for g in self.ignoreGlaciers:
                    if g in rgi_ids:
                        rgi_ids.remove(g)
                create_gridded_features_RGI(self.cfg, rgi_ids)
                geo_target_data = geodetic_target(rgi_ids, self.cfg)
            elif "region-" in self.geoGlaciers:
                s = self.geoGlaciers.split("-")
                region_id = int(s[1])
                thres_area = float(s[2])
                geo_target_data = geodetic_target_region(
                    region_id, self.cfg, thres_area
                )
                rgi_ids = list(geo_target_data.keys())
                for g in self.ignoreGlaciers:
                    if g in rgi_ids:
                        rgi_ids.remove(g)
                create_gridded_features_RGI(self.cfg, rgi_ids)
            self.periods_per_glacier = {}
            self.y_target_geo = {}
            self.err_target_geo = {}
            self.glaciersWithGeo = []
            for rgi_id in rgi_ids:
                if rgi_id in geo_target_data:
                    mean_pmb = geo_target_data[rgi_id]["mean"]
                    err_pmb = geo_target_data[rgi_id]["err"]
                    self.periods_per_glacier[rgi_id] = [(2000, 2021)]
                    self.y_target_geo[rgi_id] = np.array([mean_pmb])
                    self.err_target_geo[rgi_id] = np.array([err_pmb])
                    self.glaciersWithGeo.append(rgi_id)
        else:
            # This works only with Swiss data
            geodetic_mb = get_geodetic_MB(self.cfg)
            self.periods_per_glacier, _ = build_periods_per_glacier(geodetic_mb)
            self.y_target_geo = prepare_geo_targets(
                geodetic_mb, self.periods_per_glacier
            )
            self.err_target_geo = {}

            self.glaciersWithGeo = []
            for g in self.glacierList:
                if g in self.periods_per_glacier and has_geodetic_input(
                    self.cfg, g, self.periods_per_glacier
                ):
                    self.glaciersWithGeo.append(g)
                    self.err_target_geo[g] = (
                        self.y_target_geo[g] * 0
                    )  # Needed in the geodetic loss, so we just fill with zeros
            print(
                f"Geodetic data contain {len(self.glaciersWithGeo)} glaciers out of {len(self.glacierList)}."
            )

    def onEpochEnd(self) -> None:
        random.shuffle(self.glacierList)
        random.shuffle(self.glacierListGeo)
        self.indGlacier = 0
        self.indGlacierVal = 0
        self.indGlacierGeo = 0

    def __len__(self):
        return len(self.glacierList)

    def lenVal(self):
        return len(self.glacierListVal)

    def lenGeo(self):
        return len(self.glacierListGeo)

    def glaciers(self):
        """
        Iterator that returns a glacier as a string each time it is called.
        """
        while self.indGlacier < len(self.glacierList):
            yield self.glacierList[self.indGlacier]
            self.indGlacier += 1
        self.indGlacier = 0

    def glaciersVal(self):
        """
        Iterator that returns a glacier as a string each time it is called.
        """
        while self.indGlacierVal < len(self.glacierListVal):
            yield self.glacierListVal[self.indGlacierVal]
            self.indGlacierVal += 1
        self.indGlacierVal = 0

    def glaciersGeo(self):
        """
        Iterator that returns a glacier in the list of geodetic available glaciers as a string each time it is called.
        """
        while self.indGlacierGeo < len(self.glacierListGeo):
            yield self.glacierListGeo[self.indGlacierGeo]
            self.indGlacierGeo += 1
        self.indGlacierGeo = 0

    def stakes(self, glacierName: str, overwriteDf: pd.DataFrame = None):
        """
        Returns the training stake data to be used in the model.

        Args:
            glacierName (str): The glacier associated with the stake data to be
                returned.
            overwriteDf (pd.DataFrame): For internal use only. It allows applying
                the same processing steps to the validation dataset.

        Returns:
            features (np.ndarray): The normalized features.
            metadata (pd.DataFrame): The metadata with non-numeric columns replaced
                by numerical equivalents where each ID has been replaced by values
                ranging from 0 to N-1 where N is the number of unique values for a
                given column. If a column is named "ID", the column in the returned
                dataframe is named "ID_int" where "_int" stands for integer.
            groundTruth (np.ndarray): The ground truth mass balance values.
        """
        X = overwriteDf if overwriteDf is not None else self.trainStakesDf
        X = X[X[self.keyGlacierSel] == glacierName]  # .dropna()

        feature_columns = self.cfg.featureColumns
        non_feature_columns = X.columns.difference(feature_columns)

        # Extract metadata and features
        metadata = X[non_feature_columns]  # .values
        features = X[feature_columns].values

        groundTruth = X.POINT_BALANCE.values

        features = self.normalizer.normalize(features)
        # metadata = self._mapStrColToInt(metadata, True)

        return features, metadata, groundTruth

    def stakesVal(self, glacierName: str):
        """
        Returns the validation stake data to be used in the model.

        Args:
            glacierName (str): The glacier associated with the stake data to be
                returned.

        See the `stakes` docstring for more information.
        """
        return self.stakes(glacierName, overwriteDf=self.valStakesDf)

    def stakesKeys(self):
        """
        Returns the keys of the stake measurements for the features and metadata in
        the same order as they are returned in the `stakes` method.
        """
        meta_data_columns = self.cfg.metaData
        feature_columns = self.trainStakesDf.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # Convert feature_columns and meta_data_columns to a list (if needed)
        return list(feature_columns), list(meta_data_columns)

    def hasGeo(self, glacierName: str):
        return glacierName in self.glaciersWithGeo

    def geo(self, glacierName: str):
        return self._geo_sync(glacierName)

    def submit_geo(self, glacierName: str):
        if not self.hasGeo(glacierName):
            return None
        return self._geo_executor.submit(self._geo_sync, glacierName)

    def _metadata_groups(self, df):
        # Retrieve feature columns directly from the config
        feature_columns = self.cfg.featureColumns
        non_feature_columns = df.columns.difference(feature_columns)
        metadata = df[non_feature_columns]

        # If one day we add stochasticity we will have to make sure that the precomputed indices correspond to the ones obtained in the aggregation of the lost function

        int_id, unique_id = pd.factorize(metadata["ID"].values)
        metadata = metadata.assign(ID_int=int_id)
        grouped_ids = aggrMetadataId(metadata, "ID_int")

        int_glwd_id, unique_id = pd.factorize(grouped_ids["GLWD_ID"].values)
        grouped_ids = grouped_ids.assign(GLWD_ID_int=int_glwd_id)
        metadataAggrYear = aggrMetadataGlwdId(grouped_ids, "GLWD_ID_int")

        return {
            "metadata": metadata,
            "grouped_ids": grouped_ids,
            "nunique_ids": metadata["ID_int"].nunique(),
            "grouped_glwd_ids": metadataAggrYear,
            "nunique_glwd_ids": grouped_ids["GLWD_ID_int"].nunique(),
        }

    def _geo_sync(self, glacierName: str, async_transfer: bool = False):
        """
        Returns the geodetic data to be used in the model.

        Args:
            glacierName (str): The glacier associated with the geodetic data to be
                returned.

        Returns:
            features (np.ndarray): The normalized features.
            metadata (pd.DataFrame): The metadata with non-numeric columns replaced
                by numerical equivalents where each ID has been replaced by values
                ranging from 0 to N-1 where N is the number of unique values for a
                given column. If a column is named "ID", the column in the returned
                dataframe is named "ID_int" where "_int" stands for integer.
            groundTruth (np.ndarray): The ground truth geodetic mass balance values.
        """
        assert (
            glacierName in self.glaciersWithGeo
        ), f"Glacier {glacierName} is not in the list of glaciers with available geodetic data for this dataloader."
        if self.df_X_geod is None:
            if self.geodeticOggm:
                df_X_geod = geodetic_input(glacierName)
            else:
                df_X_geod = create_geodetic_input(
                    self.cfg, glacierName, self.periods_per_glacier, to_seasonal=False
                )
            precomputed_meta = self._metadata_groups(df_X_geod)
        else:
            if self.preloadGeodetic:
                df_X_geod = self.df_X_geod[glacierName]
            else:
                df_X_geod = self.df_X_geod
            if self.precomputed_meta is not None:
                precomputed_meta = self.precomputed_meta[glacierName]
            else:
                precomputed_meta = self._metadata_groups(df_X_geod)

        # # NaN values are in aspect_sgi and slope_sgi columns
        # # That's because the GLAMOS and SGI grids don't match exactly on the borders
        # df_X_geod = df_X_geod.dropna()

        assert (
            len(df_X_geod) > 0
        ), f"Geodetic dataframe of glacier {glacierName} is empty."

        # Retrieve feature columns directly from the config
        feature_columns = self.cfg.featureColumns
        # With the geodetic grid we need more features (like POINT_LAT, POINT_LON and GLWD_ID)
        # than what is usually defined for stakes data

        metadata = precomputed_meta["metadata"]

        # int_glwd_id, unique_id = pd.factorize(grouped_ids["GLWD_ID"].values)
        # grouped_ids = grouped_ids.assign(GLWD_ID_int=int_glwd_id)

        # Extract metadata and features
        # metadata = df_X_geod[meta_data_columns]  # .values
        features = df_X_geod[feature_columns].values

        features = self.normalizer.normalize(features)

        # return (
        #     features,
        #     metadata,
        #     self.y_target_geo[glacierName],
        #     self.err_target_geo.get(glacierName),
        # )

        err = self.err_target_geo.get(glacierName)
        if async_transfer:
            features = torch.from_numpy(features.astype(np.float32)).pin_memory()
            y = torch.from_numpy(
                self.y_target_geo[glacierName].astype(np.float32)
            ).pin_memory()
            if err is not None:
                err = torch.from_numpy(err.astype(np.float32)).pin_memory()
        else:
            features = torch.from_numpy(features.astype(np.float32))
            y = torch.from_numpy(self.y_target_geo[glacierName].astype(np.float32))
            if err is not None:
                err = torch.from_numpy(err.astype(np.float32))
        return features, metadata, y, err, precomputed_meta

    def close(self):
        self._geo_executor.shutdown(wait=False)
