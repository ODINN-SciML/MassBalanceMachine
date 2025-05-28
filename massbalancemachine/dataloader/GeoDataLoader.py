import random
from typing import List
from regions.Switzerland.scripts.geodata import prepareGeoTargets, buildPeriodsPerGlacier
from regions.Switzerland.scripts.glamos_preprocess import get_geodetic_MB
from regions.Switzerland.scripts.xgb_helpers import create_geodetic_input

import pandas as pd

from data_processing.Dataset import Normalizer


class GeoDataLoader:
    """
    The class that handles both stakes and geodetic data loading. It prepares the
    features and the metadata and add extra columns for the aggregations. It also
    retrieves the ground truth values.

    Args:
        cfg (config.Config): Configuration instance.
        glacierList (list of str): List of glaciers to use in the instanciated
            dataloader.
        stakesDf (pd.DataFrame): A pandas dataframe containing the stake data.
    """
    def __init__(self, cfg, glacierList: List[str], stakesDf: pd.DataFrame) -> None:
        self.cfg = cfg
        self.glacierList = glacierList.copy() # Copy for shuffling
        random.shuffle(self.glacierList)
        self.indGlacier = 0

        self.stakesDf = stakesDf
        self.prepareGeoData()

        if len(self.glacierList) == 1:
            # Preload geodetic data into memory if there is only one glacier
            self.df_X_geod = create_geodetic_input(
                self.cfg,
                self.glacierList[0],
                self.periods_per_glacier,
                to_seasonal=False
            )
        else:
            self.df_X_geod = None

        self.normalizer = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

    def prepareGeoData(self) -> None:
        geodetic_mb = get_geodetic_MB(self.cfg)
        self.periods_per_glacier, _ = buildPeriodsPerGlacier(geodetic_mb)
        self.y_target_geo = prepareGeoTargets(geodetic_mb, self.periods_per_glacier)

    def onEpochEnd(self) -> None:
        random.shuffle(self.glacierList)
        self.indGlacier = 0

    def glaciers(self):
        """
        Iterator that returns a glacier as a string each time it is called.
        """
        while self.indGlacier < len(self.glacierList):
            yield self.glacierList[self.indGlacier]
            self.indGlacier += 1
        self.indGlacier = 0

    def stakes(self, glacierName: str):
        """
        Returns the stake data to be used in the model.

        Args:
            glacierName (str): The glacier associated with the stake data to be
                returned.

        Returns:
            features (np.ndarray): The normalized features.
            metadata (pd.DataFrame): The metadata with non-numeric columns replaced
                by numerical equivalents where each ID has been replaced by values
                ranging from 0 to N-1 where N is the number of unique values for a
                given column. If a column is named "ID", the column in the returned
                dataframe is named "ID_int" where "_int" stands for integer.
            groundTruth (np.ndarray): The ground truth mass balance values.
        """
        X = self.stakesDf[self.stakesDf.GLACIER == glacierName].dropna()

        meta_data_columns = self.cfg.metaData

        # Split features from metadata
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = X.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = X[meta_data_columns]#.values
        features = X[feature_columns].values

        groundTruth = X.POINT_BALANCE.values

        features = self.normalizer.normalize(features)
        metadata = self._mapStrColToInt(metadata)

        return features, metadata, groundTruth

    def stakesKeys(self):
        """
        Returns the keys of the stake measurements for the features and metadata in
        the same order as they are returned in the `stakes` method.
        """
        meta_data_columns = self.cfg.metaData
        feature_columns = self.stakesDf.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # Convert feature_columns and meta_data_columns to a list (if needed)
        return list(feature_columns), list(meta_data_columns)

    def geo(self, glacierName):
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
        assert glacierName in self.glacierList, f"Glacier {glacierName} is not in the list of glaciers for this dataloader."
        if self.df_X_geod is None:
            df_X_geod = create_geodetic_input(
                self.cfg,
                glacierName,
                self.periods_per_glacier,
                to_seasonal=False
            )
        else:
            df_X_geod = self.df_X_geod

        # NaN values are in aspect_sgi and slope_sgi columns
        # That's because the GLAMOS and SGI grids don't match exactly on the borders
        df_X_geod = df_X_geod.dropna()

        assert len(df_X_geod) > 0; f"Geodetic dataframe of glacier {glacierName} is empty."

        meta_data_columns = self.cfg.metaData

        # Split features from metadata
        # Get feature columns by subtracting metadata columns from all columns
        feature_columns = df_X_geod.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures + ["topo"])
        # Convert feature_columns to a list (if needed)
        feature_columns = list(feature_columns)

        # Extract metadata and features
        metadata = df_X_geod[meta_data_columns]#.values
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
        stringCol = ['RGIId', 'ID', 'GLWD_ID', 'MONTHS', 'PERIOD', 'GLACIER']
        colToRemove = []
        for col in stringCol:
            if col in metadata.keys():
                colToRemove.append(col)
                if col == 'MONTHS':
                    string_to_index = {
                        s: i-1 # index starts at 0
                            for s, i in self.cfg.month_abbr_hydr.items()
                        }
                    col_int = metadata[col].map(string_to_index)
                else:
                    col_int, unique_val = pd.factorize(metadata[col])
                metadata[col + '_int'] = col_int
        metadata.drop(colToRemove, axis=1, inplace=True)
        return metadata
