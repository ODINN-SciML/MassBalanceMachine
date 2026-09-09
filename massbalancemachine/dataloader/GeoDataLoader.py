import random
from typing import List
import numpy as np
import pandas as pd
import time
import torch
import tqdm
from concurrent.futures import ThreadPoolExecutor

# from regions.RGI_11_Switzerland.scripts.geodetic.geodetic_processing import (
#     prepare_geo_targets,
#     build_periods_per_glacier,
#     get_geodetic_MB,
#     create_geodetic_input,
#     has_geodetic_input,
# )

from data_processing.Dataset import Normalizer
from data_processing.utils import _rebuild_month_index
from data_processing.gridded_utils import (
    create_gridded_features_RGI,
    create_gridded_features_PGO,
    create_gridded_features_GLAMOS,
    geodetic_input_Hugonnet21,
    geodetic_target_Hugonnet21,
    geodetic_target_region_Hugonnet21,
    generate_grid_multi_years,
    load_grid_multi_years,
    load_precomputed_metadata,
    prepare_precomputed_metadata,
    GEODETIC_INPUT_FN,
    WINDOWED_SOURCES,
)
from data_processing.pgo import pgo_target_file, geodetic_target_PGO, table_RGI62_to_PGO
from data_processing.glamos import geodetic_target_GLAMOS, table_RGI62_to_GLAMOS
from models.TorchNeuralNetworkRegressor import aggrMetadata


def isNativeGlacierId(glacierName: str, geodeticSource: str) -> bool:
    """Whether `glacierName` is already expressed in the identifier scheme of
    `geodeticSource`, and therefore needs no translation from an RGI 6.2 id.

    PGO glaciers are named by their RGI v7 id, GLAMOS ones by their SGI id
    ("B36-26"), which is anything that is not an RGI id.
    """
    if geodeticSource == "PGO":
        return glacierName.startswith("RGI2000-v7.0-G-")
    if geodeticSource == "GLAMOS":
        return not glacierName.startswith("RGI")
    return True


def buildGlacierMapping(glacierList, geodeticSource: str):
    """Map every glacier of `glacierList` onto the identifier the geodetic source
    uses for it.

    Several RGI ids mapping onto one entity is expected and kept: the Swiss inventory
    and the RGI do not cut the ice into the same glaciers, and Claridenfirn is one SGI
    glacier that RGI 6.2 splits into four. The reverse - one RGI id matching several
    entities - is dropped, see below.
    """
    if isNativeGlacierId(glacierList[0], geodeticSource):
        # Glacier list already follows the source's own identifiers
        return {glacier_id: glacier_id for glacier_id in glacierList}

    # We have to find the mapping between the glacier list, which is in RGI v6, and
    # the identifiers of the geodetic source
    rgi_ids_rgi6 = glacierList
    region_id = int(
        rgi_ids_rgi6[0].split(".")[0].split("-")[1]
    )  # Use first glacier to retrieve region
    if geodeticSource == "PGO":
        table_df = table_RGI62_to_PGO(region_id)
    elif geodeticSource == "GLAMOS":
        table_df = table_RGI62_to_GLAMOS(region_id=region_id)
    else:
        raise ValueError(f"No glacier id mapping available for {geodeticSource}.")

    # A glacier is only mapped when the crosswalk gives it exactly one entity. Several
    # rows mean the RGI outline straddles two of them and there is no basis for
    # picking one; attributing a whole glacier-wide geodetic rate to the wrong entity
    # is worse than leaving that glacier out of the geodetic loss altogether.
    mapping = {}
    for rgi_id_rgi6 in rgi_ids_rgi6:
        tmp = table_df[table_df.RGIId == rgi_id_rgi6]
        if tmp.shape[0] == 1:
            mapping[rgi_id_rgi6] = tmp.custom_id.values[0]
    return mapping


def buildPGOMapping(glacierList):
    return buildGlacierMapping(glacierList, "PGO")


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
        glacierListGeo=None,
        glacierListValGeo=None,
        ignoreStakesWithoutGeo: bool = False,
        geodeticSource: str = "Hugonnet21",
        preloadGeodetic: bool = False,
        keyGlacierSel: str = "GLACIER",
        geoGlaciers: str = "stakes",
        ignoreGlaciers: list[str] = [],
        device=torch.device("cpu"),
        allStakesPerIter=False,
        additionalYears=[],  # years for which to ensure the gridded products are generated in addition to what is required to process the dataset (which is based on the geodeticSource argument); this is used only when geoGlaciers="stakes"
        noGeo=False,
        geodeticSourceOptions: dict = None,  # extra options handed to the geodetic target of `geodeticSource`; see `_prepareGeoDataWindowed`
    ) -> None:
        self.cfg = cfg
        self.glacierList = (
            glacierList.copy() if glacierList is not None else None
        )  # Copy for shuffling
        self.glacierListVal = (
            glacierListVal.copy()
        )  # Copy just in case but we don't shuffle this
        if self.glacierList is not None:
            random.shuffle(self.glacierList)
        self.indGlacier = 0
        self.indGlacierVal = 0
        self.indGlacierGeo = 0
        self.indGlacierValGeo = 0
        self.indGlacierAllGeo = 0
        self.periodToInt = {"annual": 0, "winter": 1, "summer": 2}

        _, self.month_pos = _rebuild_month_index(months_head_pad, months_tail_pad)

        self.trainStakesDf = trainStakesDf
        self.valStakesDf = valStakesDf
        self.geodeticSource = geodeticSource
        self.preloadGeodetic = preloadGeodetic
        self.keyGlacierSel = keyGlacierSel
        self.geoGlaciers = geoGlaciers
        self.ignoreGlaciers = ignoreGlaciers
        self.device = device
        self.allStakesPerIter = allStakesPerIter
        self.additionalYears = additionalYears
        self.geodeticSourceOptions = geodeticSourceOptions or {}

        if valStakesDf is not None:
            assert (
                len(glacierListVal) > 0
            ), "If validation data stakes are provided you need to provide a list of validation glaciers."
        else:
            assert (
                len(glacierListVal) == 0
            ), "If validation data stakes are not provided we don't expect a list of validation glaciers."

        # Prepare geodetic data
        if not noGeo:
            self.prepareGeoData()
            self.glacierListGeo = self.glaciersWithGeo
            self.glacierListValGeo = self.glaciersValWithGeo
            self.glacierListAllGeo = self.glaciersAllWithGeo
        if ignoreStakesWithoutGeo:
            raise NotImplementedError(
                "We need to implement an intersection between glaciersWithGeo and train/validation glaciers"
            )
            self.glacierList = self.glaciersWithGeo
            self.glacierListGeo = self.glaciersWithGeo  # TODO: change this
            self.glacierListValGeo = self.glaciersValWithGeo  # TODO: change this
            self.glacierListAllGeo = self.glaciersAllWithGeo  # TODO: change this

        if not noGeo and len(self.glaciersWithGeo) == 1:
            if self.geodeticSource in {"Hugonnet21"} | WINDOWED_SOURCES:
                raise NotImplementedError()
            # # Preload geodetic data into memory if there is only one glacier
            # self.df_X_geod = create_geodetic_input(
            #     self.cfg,
            #     self.glaciersWithGeo[0],
            #     self.periods_per_glacier,
            #     to_seasonal=False,
            # )
        elif not noGeo:
            if self.preloadGeodetic:
                if self.geodeticSource == "Hugonnet21":
                    print("Preloading Hugonnet21 geodetic grids")
                    self.df_X_geod = {}
                    for rgi_id in tqdm.tqdm(
                        self.glaciersWithGeo + self.glaciersValWithGeo
                    ):
                        self.df_X_geod[rgi_id] = geodetic_input_Hugonnet21(
                            rgi_id, years=self.years
                        )
                elif self.geodeticSource in WINDOWED_SOURCES:
                    print(f"Preloading {self.geodeticSource} geodetic grids")
                    geodetic_input = GEODETIC_INPUT_FN[self.geodeticSource]
                    self.df_X_geod = {}
                    for glacier_id in tqdm.tqdm(
                        self.glaciersWithGeo + self.glaciersValWithGeo
                    ):
                        self.df_X_geod[glacier_id] = geodetic_input(
                            glacier_id,
                            time_range=self.periods_per_glacier[glacier_id],
                        )
                else:
                    raise ValueError(f"Unknown geodetic source {self.geodeticSource}.")
            elif self.geodeticSource is not None:
                self.df_X_geod = None
                if self.geodeticSource == "Hugonnet21":
                    print("Preparing Hugonnet21 geodetic grids")
                    for rgi_id in tqdm.tqdm(
                        self.glaciersWithGeo + self.glaciersValWithGeo
                    ):
                        generate_grid_multi_years(
                            rgi_id,
                            self.years,
                            "Hugonnet21",
                        )
                        prepare_precomputed_metadata(
                            rgi_id,
                            self.years,
                            "Hugonnet21",
                            self.cfg.featureColumns,
                        )
                elif self.geodeticSource in WINDOWED_SOURCES:
                    print(f"Preparing {self.geodeticSource} geodetic grids")
                    for glacier_id in tqdm.tqdm(
                        self.glaciersWithGeo + self.glaciersValWithGeo
                    ):
                        generate_grid_multi_years(
                            glacier_id,
                            self.periods_per_glacier[glacier_id],
                            self.geodeticSource,
                        )
                        prepare_precomputed_metadata(
                            glacier_id,
                            self.periods_per_glacier[glacier_id],
                            self.geodeticSource,
                            self.cfg.featureColumns,
                        )
                else:
                    raise ValueError(f"Unknown geodetic source {self.geodeticSource}.")

        if not noGeo and self.df_X_geod is not None:
            if len(self.glaciersWithGeo) == 1:
                self.precomputed_meta = {
                    self.glaciersWithGeo[0]: self._metadata_groups(self.df_X_geod)
                }
            else:
                self.precomputed_meta = {}
                for rgi_id in self.glaciersWithGeo + self.glaciersValWithGeo:
                    self.precomputed_meta[rgi_id] = self._metadata_groups(
                        self.df_X_geod[rgi_id]
                    )
        else:
            self.precomputed_meta = None

        if self.allStakesPerIter:
            self.precomputed_meta_stakes = (
                self._metadata_groups_stakes(self.trainStakesDf)
                if self.trainStakesDf is not None
                else None
            )
        else:
            self.precomputed_meta_stakes = None

        self.normalizer = Normalizer({k: cfg.bnds[k] for k in cfg.featureColumns})

        self._prefetch_depth = 3
        self._geo_executor = ThreadPoolExecutor(max_workers=self._prefetch_depth)

    def prepareGeoData(self) -> None:
        if self.geodeticSource == "Hugonnet21":
            stakesDf = (
                self.trainStakesDf
                if self.valStakesDf is None
                else pd.concat(
                    (self.trainStakesDf, self.valStakesDf), ignore_index=True
                )
            )
            # TODO: implement this in a more clever way
            if self.geoGlaciers == "stakes":
                # rgi_ids = list(stakesDf.RGIId.unique())
                rgi_ids = list(set(self.glacierList).union(self.glacierListVal))
                for g in self.ignoreGlaciers:
                    if g in rgi_ids:
                        rgi_ids.remove(g)
                self.years = list(range(2000, 2020)) + self.additionalYears
                create_gridded_features_RGI(self.cfg, rgi_ids, years=self.years)
                geo_target_data = geodetic_target_Hugonnet21(rgi_ids, self.cfg)
            elif "region-" in self.geoGlaciers:
                s = self.geoGlaciers.split("-")
                region_id = int(s[1])
                thres_area = float(s[2])
                geo_target_data = geodetic_target_region_Hugonnet21(
                    region_id, self.cfg, thres_area
                )
                rgi_ids = list(geo_target_data.keys())
                for g in self.ignoreGlaciers:
                    if g in rgi_ids:
                        rgi_ids.remove(g)
                self.years = list(range(2000, 2020))
                create_gridded_features_RGI(self.cfg, rgi_ids, years=self.years)
            self.periods_per_glacier = {}
            self.y_target_geo = {}
            self.err_target_geo = {}
            self.glaciersWithGeo = []
            for rgi_id in rgi_ids:
                if rgi_id in geo_target_data:
                    mean_pmb = geo_target_data[rgi_id]["mean"]
                    err_pmb = geo_target_data[rgi_id]["err"]
                    self.periods_per_glacier[rgi_id] = [(2000, 2020)]
                    self.y_target_geo[rgi_id] = np.array([mean_pmb])
                    self.err_target_geo[rgi_id] = np.array([err_pmb])
                    self.glaciersWithGeo.append(rgi_id)
            if self.valStakesDf is None:
                self.glaciersValWithGeo = []
            else:
                # Split glaciersWithGeo into validation and training glaciers
                # self.glaciersValWithGeo = list(set(self.valStakesDf.RGIId.unique()).intersection(self.glaciersWithGeo))
                # self.glaciersWithGeo = list(set(self.trainStakesDf.RGIId.unique()).intersection(self.glaciersWithGeo))
                trainNoValGlaciers = set(self.glacierList).difference(
                    self.glacierListVal
                )
                self.glaciersValWithGeo = list(
                    set(self.glacierListVal).intersection(self.glaciersWithGeo)
                )
                self.glaciersWithGeo = list(
                    set(trainNoValGlaciers).intersection(self.glaciersWithGeo)
                )
            self.glaciersAllWithGeo = list(
                set(self.glaciersWithGeo).union(self.glaciersValWithGeo)
            )
        elif self.geodeticSource in WINDOWED_SOURCES:
            self._prepareGeoDataWindowed()
        # else:
        #     # This works only with Swiss data
        #     geodetic_mb = get_geodetic_MB(self.cfg)
        #     self.periods_per_glacier, _ = build_periods_per_glacier(geodetic_mb)
        #     self.y_target_geo = prepare_geo_targets(
        #         geodetic_mb, self.periods_per_glacier
        #     )
        #     self.err_target_geo = {}

        #     self.glaciersWithGeo = []
        #     for g in self.glacierList:
        #         if g in self.periods_per_glacier and has_geodetic_input(
        #             self.cfg, g, self.periods_per_glacier
        #         ):
        #             self.glaciersWithGeo.append(g)
        #             self.err_target_geo[g] = (
        #                 self.y_target_geo[g] * 0
        #             )  # Needed in the geodetic loss, so we just fill with zeros
        #     print(
        #         f"Geodetic data contain {len(self.glaciersWithGeo)} glaciers out of {len(self.glacierList)}."
        #     )

    def _prepareGeoDataWindowed(self) -> None:
        """Prepare a geodetic source whose target covers an arbitrary date window
        rather than a fixed set of calendar years.

        PGO and GLAMOS share this shape: a table with one window and one rate per
        glacier, identifiers of their own that the RGI 6.2 glacier list has to be
        mapped onto, and grids built on the source's own outlines over exactly the
        years its window spans.

        `geodeticSourceOptions` is forwarded to the source's target function. It
        matters for GLAMOS, whose windows reach back to the 19th century: pass
        `min_year` to keep the selection inside the climate forcing, along with
        `max_year`, `min_window_years` and `min_covered` to choose which windows are
        eligible at all (see `data_processing.glamos.select_glamos_windows`).
        """
        source = self.geodeticSource
        assert (
            len(self.additionalYears) == 0
        ), f"Option additionalYears is not available yet with {source} geodetic data."

        if source == "PGO":
            dfGeo = geodetic_target_PGO(pgo_target_file(), **self.geodeticSourceOptions)
            create_gridded_features = create_gridded_features_PGO
        elif source == "GLAMOS":
            dfGeo = geodetic_target_GLAMOS(**self.geodeticSourceOptions)
            create_gridded_features = create_gridded_features_GLAMOS
        else:
            raise ValueError(f"Unknown windowed geodetic source {source}.")

        glacier_ids_val = []
        if self.glacierList is None:
            # We are working with geodetic data only, no need to map RGI62 IDs to the
            # identifiers of the geodetic source
            glacier_ids = list(dfGeo.RGIId.unique())
            self.glacier_id_map = {g: g for g in glacier_ids}
            self.glacier_id_map_val = {}
        else:
            self.glacier_id_map = buildGlacierMapping(self.glacierList, source)
            self.glacier_id_map_val = {}
            if len(self.glacierListVal) > 0:
                self.glacier_id_map_val = buildGlacierMapping(
                    self.glacierListVal, source
                )
                glacier_ids_val = list(self.glacier_id_map_val.values())
            glacier_ids = list(set(self.glacier_id_map.values()).union(glacier_ids_val))
        for g in self.ignoreGlaciers:
            if g in glacier_ids:
                glacier_ids.remove(g)
            if g in glacier_ids_val:
                glacier_ids_val.remove(g)

        self.years = None
        time_ranges = {}
        for glacier_id in glacier_ids:
            tmp = dfGeo[dfGeo.RGIId == glacier_id]
            time_ranges[glacier_id] = (
                tmp.FROM_DATE.values[0],
                tmp.TO_DATE.values[0],
            )
        create_gridded_features(self.cfg, time_ranges)

        self.periods_per_glacier = {}
        self.y_target_geo = {}
        self.err_target_geo = {}
        self.glaciersWithGeo = []
        for glacier_id in glacier_ids:
            if glacier_id in dfGeo.RGIId.values:
                tmp = dfGeo[dfGeo.RGIId == glacier_id]
                mean_pmb = tmp.mwe_per_year.values[0]
                err_pmb = tmp.sigma_mwe_per_year.values[0]
                self.periods_per_glacier[glacier_id] = [time_ranges[glacier_id]]
                self.y_target_geo[glacier_id] = np.array([mean_pmb])
                self.err_target_geo[glacier_id] = np.array([err_pmb])
                self.glaciersWithGeo.append(glacier_id)
        # Split glaciersWithGeo into validation and training glaciers
        self.glaciersValWithGeo = list(
            set(self.glaciersWithGeo).intersection(glacier_ids_val)
        )
        self.glaciersWithGeo = list(
            set(self.glaciersWithGeo).difference(self.glaciersValWithGeo)
        )
        self.glaciersAllWithGeo = list(
            set(self.glaciersWithGeo).union(self.glaciersValWithGeo)
        )

    def _toSourceGlacierId(self, glacierName: str) -> str:
        """Translate an RGI 6.2 glacier name into the identifier the geodetic source
        uses, leaving it alone if it already is one."""
        if self.geodeticSource not in WINDOWED_SOURCES:
            return glacierName
        if isNativeGlacierId(glacierName, self.geodeticSource):
            return glacierName
        return (
            self.glacier_id_map.get(glacierName) or self.glacier_id_map_val[glacierName]
        )

    def geodetic_periods(self, g):
        return self.periods_per_glacier[self._toSourceGlacierId(g)]

    def elevation_diff_range(self, g: str):
        assert self.hasGeo(g)
        g = self._toSourceGlacierId(g)
        if self.df_X_geod is None:
            if self.geodeticSource == "Hugonnet21":
                df_X_geod = load_grid_multi_years(g, self.years, "Hugonnet21")
            else:
                df_X_geod = load_grid_multi_years(
                    g, self.periods_per_glacier[g], self.geodeticSource
                )
            precomputed_meta = self._metadata_groups(df_X_geod)
        else:
            if self.preloadGeodetic:
                df_X_geod = self.df_X_geod[g]
            else:
                df_X_geod = self.df_X_geod
            if self.precomputed_meta is not None:
                precomputed_meta = self.precomputed_meta[g]
            else:
                precomputed_meta = self._metadata_groups(df_X_geod)
        return (
            precomputed_meta["metadata"].ELEVATION_DIFFERENCE.min(),
            precomputed_meta["metadata"].ELEVATION_DIFFERENCE.max(),
        )

    def onEpochEnd(self) -> None:
        random.shuffle(self.glacierList)
        random.shuffle(self.glacierListGeo)
        self.indGlacier = 0
        self.indGlacierVal = 0
        self.indGlacierGeo = 0
        self.indGlacierValGeo = 0
        self.indGlacierAllGeo = 0

    def __len__(self):
        return len(self.glacierList)

    def lenVal(self):
        return len(self.glacierListVal)

    def lenGeo(self):
        return len(self.glacierListGeo)

    def lenValGeo(self):
        return len(self.glacierListValGeo)

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
        # TODO: make sure that we don't use val glaciers during training
        while self.indGlacierGeo < len(self.glacierListGeo):
            yield self.glacierListGeo[self.indGlacierGeo]
            self.indGlacierGeo += 1
        self.indGlacierGeo = 0

    def glaciersValGeo(self):
        """
        Iterator that returns a glacier in the list of geodetic available glaciers as a string each time it is called.
        """
        while self.indGlacierValGeo < len(self.glacierListValGeo):
            yield self.glacierListValGeo[self.indGlacierValGeo]
            self.indGlacierValGeo += 1
        self.indGlacierValGeo = 0

    def glaciersAllGeo(self):
        """
        Iterator that returns a glacier in the list of geodetic available glaciers as a string each time it is called.
        It corresponds to the union of glaciersGeo() and glaciersValGeo()
        """
        while self.indGlacierAllGeo < len(self.glacierListAllGeo):
            yield self.glacierListAllGeo[self.indGlacierAllGeo]
            self.indGlacierAllGeo += 1
        self.indGlacierAllGeo = 0

    def _metadata_groups_stakes(self, df):
        feature_columns = self.cfg.featureColumns
        non_feature_columns = df.columns.difference(feature_columns)
        if "ELEVATION_DIFFERENCE" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["ELEVATION_DIFFERENCE"]))
            )

        # Extract metadata and features
        metadata = df[non_feature_columns]
        features = df[feature_columns].values

        idAggr = metadata["ID"].values
        int_id, unique_id = pd.factorize(idAggr)
        metadata = metadata.assign(ID_int=int_id)
        return {
            "metadata": metadata,
            "int_id": int_id,
            "nunique_ids": metadata["ID"].nunique(),
        }

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
        if not self.allStakesPerIter:
            X = X[X[self.keyGlacierSel] == glacierName]
            precomputed_meta = self._metadata_groups_stakes(X)
        else:
            if overwriteDf is None:
                precomputed_meta = self.precomputed_meta_stakes
            else:
                precomputed_meta = self._metadata_groups_stakes(X)

        return self._extract_stakes(X, precomputed_meta)

    def _extract_stakes(self, X: pd.DataFrame, precomputed_meta):
        """Split a stakes-like dataframe into normalized features, metadata,
        and ground truth. Shared by `stakes` and `stakesFromDf`.
        """
        feature_columns = self.cfg.featureColumns
        non_feature_columns = X.columns.difference(feature_columns)
        if "ELEVATION_DIFFERENCE" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["ELEVATION_DIFFERENCE"]))
            )
        if "POINT_ELEVATION" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["POINT_ELEVATION"]))
            )
        if "POINT_LAT" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["POINT_LAT"]))
            )
        if "POINT_LON" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["POINT_LON"]))
            )

        # Extract metadata and features
        metadata = X[non_feature_columns]
        features = X[feature_columns].values

        groundTruth = X.POINT_BALANCE.values

        features = self.normalizer.normalize(features)
        # metadata = self._mapStrColToInt(metadata, True)

        return features, metadata, groundTruth, precomputed_meta

    def stakesFromDf(self, df: pd.DataFrame, glacierName: str = None):
        """
        Runs an arbitrary stakes-like dataframe through the same processing as
        `stakes`, without requiring it to be part of `trainStakesDf` /
        `valStakesDf`. Useful for dataframes built outside the usual stake
        pipeline, such as the AWS monthly climate/topographical features from
        `create_aws_climate_topo_features` / `monthly_features`.

        Args:
            df (pd.DataFrame): A stakes-like dataframe. Must contain
                `cfg.featureColumns` and `POINT_BALANCE`. If it has no "ID"
                column, one is added automatically (`np.arange(len(df))`),
                treating each row as an independent measurement, which is
                correct for dataframes already at monthly resolution (one row
                per month, as opposed to the wide hydrological-year rows that
                `transform_to_monthly` explodes into several monthly rows
                sharing one "ID").
            glacierName (str, optional): If given and `df` has a
                `self.keyGlacierSel` column, `df` is filtered to
                `df[self.keyGlacierSel] == glacierName` first, like `stakes`
                does. Ignored if `df` doesn't have that column (e.g. AWS
                dataframes, which use "RGIId" rather than "GLACIER").

        See the `stakes` docstring for the meaning of the returned values.
        """
        df = df.copy()
        if "ID" not in df.columns:
            df["ID"] = np.arange(len(df))
        if glacierName is not None and self.keyGlacierSel in df.columns:
            df = df[df[self.keyGlacierSel] == glacierName]

        precomputed_meta = self._metadata_groups_stakes(df)
        return self._extract_stakes(df, precomputed_meta)

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
        if self.geodeticSource in WINDOWED_SOURCES:
            if glacierName in self.glacier_id_map:
                return self.glacier_id_map[glacierName] in self.glaciersWithGeo
            elif glacierName in self.glacier_id_map_val:
                return self.glacier_id_map_val[glacierName] in self.glaciersValWithGeo
            else:
                return False
        else:
            return (
                glacierName in self.glaciersWithGeo
                or glacierName in self.glaciersValWithGeo
            )

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
        if "ELEVATION_DIFFERENCE" in feature_columns:
            # We also want this in the metadata as this is useful to have it not unnormalized
            non_feature_columns = list(
                set(non_feature_columns).union(set(["ELEVATION_DIFFERENCE"]))
            )
        metadata = df[non_feature_columns]

        # If one day we add stochasticity we will have to make sure that the precomputed indices correspond to the ones obtained in the aggregation of the lost function

        int_id, unique_id = pd.factorize(metadata["ID"].values)
        int_glwd_m_id, _ = pd.factorize(metadata["GLWD_M_ID"].values)
        metadata = metadata.assign(ID_int=int_id, GLWD_M_ID_int=int_glwd_m_id)
        grouped_ids = aggrMetadata(metadata, "ID_int")

        metadataAggrGlWdM = aggrMetadata(metadata, "GLWD_M_ID_int")

        return {
            "metadata": metadata,
            "grouped_ids": grouped_ids,
            "grouped_glwd_m_ids": metadataAggrGlWdM,
            "nunique_glwd_m_ids": metadata["GLWD_M_ID_int"].nunique(),
            "nunique_ids": metadata["ID_int"].nunique(),
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
        glacierName = self._toSourceGlacierId(glacierName)
        assert (glacierName in self.glaciersWithGeo) or (
            glacierName in self.glaciersValWithGeo
        ), f"Glacier {glacierName} is not in the list of glaciers with available geodetic data for this dataloader."
        if self.df_X_geod is None:
            if self.geodeticSource == "Hugonnet21":
                years = self.years
                df_X_geod = load_grid_multi_years(glacierName, years, "Hugonnet21")
                precomputed_meta = load_precomputed_metadata(
                    glacierName,
                    years,
                    "Hugonnet21",
                    self.cfg.featureColumns,
                )
            elif self.geodeticSource in WINDOWED_SOURCES:
                years = self.periods_per_glacier[glacierName]
                df_X_geod = load_grid_multi_years(
                    glacierName, years, self.geodeticSource
                )
                precomputed_meta = load_precomputed_metadata(
                    glacierName,
                    years,
                    self.geodeticSource,
                    self.cfg.featureColumns,
                )
            # else:
            #     df_X_geod = create_geodetic_input(
            #         self.cfg, glacierName, self.periods_per_glacier, to_seasonal=False
            #     )
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

        err = self.err_target_geo[glacierName]
        if async_transfer:
            features = torch.from_numpy(features.astype(np.float32)).pin_memory()
            y = torch.from_numpy(
                self.y_target_geo[glacierName].astype(np.float32)
            ).pin_memory()
            err = torch.from_numpy(err.astype(np.float32)).pin_memory()
            precomputed_meta["GLWD_M_ID_int"] = torch.from_numpy(
                metadata["GLWD_M_ID_int"].values.astype(np.int64)
            ).pin_memory()
        else:
            features = torch.from_numpy(features.astype(np.float32))
            y = torch.from_numpy(self.y_target_geo[glacierName].astype(np.float32))
            err = torch.from_numpy(err.astype(np.float32))
            precomputed_meta["GLWD_M_ID_int"] = torch.from_numpy(
                metadata["GLWD_M_ID_int"].values.astype(np.int64)
            )
            precomputed_meta["ID_int"] = torch.from_numpy(
                metadata["ID_int"].values.astype(np.int64)
            )
        return features, metadata, y, err, precomputed_meta

    def close(self):
        self._geo_executor.shutdown(wait=False)
