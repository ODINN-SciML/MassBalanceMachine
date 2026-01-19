import random
from typing import List
import pandas as pd

from regions.Switzerland.scripts.geodetic.geodetic_processing import (
    prepare_geo_targets,
    build_periods_per_glacier,
    get_geodetic_MB,
    create_geodetic_input,
    has_geodetic_input,
)

from data_processing.Dataset import Normalizer
from data_processing.utils import _rebuild_month_index


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
        ignoreStakesWithoutGeo: bool = False,
    ) -> None:
        self.cfg = cfg
        self.glacierList = glacierList.copy()  # Copy for shuffling
        random.shuffle(self.glacierList)
        self.indGlacier = 0
        self.periodToInt = {"annual": 0, "winter": 1}

        _, self.month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

        self.trainStakesDf = trainStakesDf
        self.valStakesDf = valStakesDf
        self.prepareGeoData()
        if ignoreStakesWithoutGeo:
            self.glacierList = self.glaciersWithGeo

        if len(self.glaciersWithGeo) == 1:
            # Preload geodetic data into memory if there is only one glacier
            self.df_X_geod = create_geodetic_input(
                self.cfg,
                self.glaciersWithGeo[0],
                self.periods_per_glacier,
                to_seasonal=False,
            )
        else:
            self.df_X_geod = None

        self.normalizer = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

    def prepareGeoData(self) -> None:
        geodetic_mb = get_geodetic_MB(self.cfg)
        self.periods_per_glacier, _ = build_periods_per_glacier(geodetic_mb)
        self.y_target_geo = prepare_geo_targets(geodetic_mb, self.periods_per_glacier)

        self.glaciersWithGeo = []
        for g in self.glacierList:
            if g in self.periods_per_glacier and has_geodetic_input(
                self.cfg, g, self.periods_per_glacier
            ):
                self.glaciersWithGeo.append(g)
        print(
            f"Geodetic data contain {len(self.glaciersWithGeo)} glaciers out of {len(self.glacierList)}."
        )

    def onEpochEnd(self) -> None:
        random.shuffle(self.glacierList)
        self.indGlacier = 0

    def __len__(self):
        return len(self.glacierList)

    def glaciers(self):
        """
        Iterator that returns a glacier as a string each time it is called.
        """
        while self.indGlacier < len(self.glacierList):
            yield self.glacierList[self.indGlacier]
            self.indGlacier += 1
        self.indGlacier = 0

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
        X = (overwriteDf or self.trainStakesDf)[
            self.trainStakesDf.GLACIER == glacierName
        ].dropna()

        meta_data_columns = self.cfg.metaData

        # Split features from metadata
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # Convert feature_columns to a list (if needed)
        if "y" in feature_columns:
            feature_columns = feature_columns.drop("y")
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[meta_data_columns]  # .values
        features = X[feature_columns].values

        groundTruth = X.POINT_BALANCE.values

        features = self.normalizer.normalize(features)
        metadata = self._mapStrColToInt(metadata)

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
            df_X_geod = create_geodetic_input(
                self.cfg, glacierName, self.periods_per_glacier, to_seasonal=False
            )
        else:
            df_X_geod = self.df_X_geod

        # NaN values are in aspect_sgi and slope_sgi columns
        # That's because the GLAMOS and SGI grids don't match exactly on the borders
        df_X_geod = df_X_geod.dropna()

        assert (
            len(df_X_geod) > 0
        ), f"Geodetic dataframe of glacier {glacierName} is empty."

        meta_data_columns = self.cfg.metaData

        # Split features from metadata
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = df_X_geod.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(
            self.cfg.notMetaDataNotFeatures + ["topo"]
        )
        if "y" in feature_columns:
            feature_columns = feature_columns.drop("y")
        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = df_X_geod[meta_data_columns]  # .values
        features = df_X_geod[feature_columns].values

        features = self.normalizer.normalize(features)
        metadata = self._mapStrColToInt(metadata)

        return features, metadata, self.y_target_geo[glacierName]

    def _mapStrColToInt(self, metadata):
        """
        Maps columns that contain string values to new ones with integer values.
        Non-numeric columns are replaced by numerical equivalents where each ID has
        been replaced by values ranging from 0 to N-1 where N is the number of
        unique values for a given column. If a column is named "ID", the column in
        the returned dataframe is named "ID_int" where "_int" stands for integer.
        """
        metadata = metadata.copy()
        stringCol = ["RGIId", "ID", "GLWD_ID", "MONTHS", "PERIOD", "GLACIER"]
        colToRemove = []
        for col in stringCol:
            if col in metadata.keys():
                colToRemove.append(col)
                if col == "MONTHS":
                    string_to_index = {
                        s: i - 1 for s, i in self.month_pos.items()  # index starts at 0
                    }
                    col_int = metadata[col].map(string_to_index)
                elif col == "PERIOD":
                    # Ensure always the same convention
                    # Otherwise glacier with only winter data could result in winter
                    # being 0 instead of 1
                    col_int = metadata[col].map(self.periodToInt)
                else:
                    col_int, unique_val = pd.factorize(metadata[col])
                metadata[col + "_int"] = col_int
        metadata.drop(colToRemove, axis=1, inplace=True)
        return metadata
