import random
from regions.Switzerland.scripts.geodata import prepareGeoTargets, buildPeriodsPerGlacier
from regions.Switzerland.scripts.glamos_preprocess import get_geodetic_MB
from regions.Switzerland.scripts.xgb_helpers import create_geodetic_input

import numpy as np
import pandas as pd



class GeoDataLoader:
    def __init__(self, cfg, glacierList, stakesDf):
        self.cfg = cfg
        self.glacierList = glacierList.copy() # Copy for shuffling
        random.shuffle(self.glacierList)
        self.indGlacier = 0

        self.stakesDf = stakesDf
        self.prepareGeoData()

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

    def stakes(self, glacierName):
        return self.stakesDf[self.stakesDf.GLACIER == glacierName]

    def geo(self, glacierName):
        df_X_geod = create_geodetic_input(
            self.cfg,
            glacierName,
            self.periods_per_glacier,
            to_seasonal=False
        )
        return df_X_geod, self.y_target_geo[glacierName]
