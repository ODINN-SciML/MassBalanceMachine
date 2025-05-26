import random
from regions.Switzerland.scripts.geodata import prepareGeoTargets, buildPeriodsPerGlacier
from regions.Switzerland.scripts.glamos_preprocess import get_geodetic_MB
from regions.Switzerland.scripts.xgb_helpers import create_geodetic_input

import pandas as pd
import time

from massbalancemachine.data_processing.Dataset import Normalizer


class GeoDataLoader:
    def __init__(self, cfg, glacierList, stakesDf, bnds):
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

        self.normalizer = Normalizer(bnds) # TODO: use cfg.bnds instead

    def prepareGeoData(self):

        geodetic_mb = get_geodetic_MB(self.cfg)
        self.periods_per_glacier, _ = buildPeriodsPerGlacier(geodetic_mb)
        self.y_target_geo = prepareGeoTargets(geodetic_mb, self.periods_per_glacier)

    def onEpochEnd(self):
        random.shuffle(self.glacierList)
        self.indGlacier = 0

    def glaciers(self):
        while self.indGlacier < len(self.glacierList):
            yield self.glacierList[self.indGlacier]
            self.indGlacier += 1
        self.indGlacier = 0

    def stakes(self, glacierName):
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
        meta_data_columns = self.cfg.metaData
        feature_columns = self.stakesDf.columns.difference(meta_data_columns)

        # remove columns that are not used in metadata or features
        feature_columns = feature_columns.drop(self.cfg.notMetaDataNotFeatures)
        # Convert feature_columns and meta_data_columns to a list (if needed)
        return list(feature_columns), list(meta_data_columns)

    def geo(self, glacierName):
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
        st = time.time()
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
        strToIntColTime = time.time()-st
        print("strToIntColTime=",strToIntColTime)
        return metadata
