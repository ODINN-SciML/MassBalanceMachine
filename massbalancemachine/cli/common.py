"""Helpers shared by the command-line tools of the package and by the training and
evaluation scripts of the repository (`scripts/`), which re-export them."""

import os

import yaml
import pandas as pd

import massbalancemachine as mbm

# Keys of a region block holding glacier lists, merged by union over the regions
REGION_LIST_KEYS = [
    "train_glaciers",
    "val_glaciers",
    "test_glaciers",
    "train_glaciers_geo",
    "val_glaciers_geo",
    "test_glaciers_geo",
]
# Keys a region block owns, which the training section must not also set
REGION_OWNED_KEYS = REGION_LIST_KEYS + [
    "splitVal",
    "val_years",
    "geodetic_source",
    "geodetic_source_options",
    "geodetic_source_options_val",
    "geodetic_source_test",
    "geodetic_source_test_options",
]


def mergeRegions(training, netcfgFolder):
    """Merge the per-region blocks of the `training` section of a config.

    Each region of `training["regions"]` either references a split file (`split_file`,
    relative to `netcfgFolder`) or holds its keys inline. The glacier lists of the regions
    are merged by union, their geodetic sources and source options by source, and
    `source_data` is the union of their RGI regions. The returned section has
    `splitVal="per-region"`, and keeps every region block, as read, under `regions`:
    `trainValData` splits each region with its own rule, and the evaluation tests each
    region on its own geodetic source. A region may set its own split by year,
    `splitTest` (e.g. "YEAR:<2010"); the others keep the `splitTest` of the training
    section, see `dataloader.SourceManager.yearSplitPerRgiRegion`.
    """
    owned = [k for k in REGION_OWNED_KEYS if k in training]
    assert (
        not owned
    ), f"With regions, {owned} must be set in the region blocks, not in the training section."

    regions = {}
    for name, block in training["regions"].items():
        block = dict(block or {})
        splitFile = block.pop("split_file", None)
        if splitFile is not None:
            with open(os.path.join(netcfgFolder, splitFile)) as f:
                block = {**yaml.safe_load(f), **block}
        assert block.get("rgi_regions"), f"Region {name} declares no rgi_regions."
        assert block.get("splitVal") in (
            "group-year",
            "group-rgi",
        ), f"Region {name}: splitVal must be group-year or group-rgi, not {block.get('splitVal')}."
        regions[name] = block

    merged = {k: v for k, v in training.items() if k != "regions"}
    for k in REGION_LIST_KEYS:
        merged[k] = list(
            dict.fromkeys(g for b in regions.values() for g in (b.get(k) or []))
        )

    rgiRegions = [r for b in regions.values() for r in b["rgi_regions"]]
    assert len(set(rgiRegions)) == len(
        rgiRegions
    ), f"An RGI region belongs to two regions: {rgiRegions}"
    sourceData = "wgms:" + ",".join(str(r) for r in sorted(rgiRegions))
    assert training.get("source_data", sourceData) == sourceData, (
        f"source_data={training['source_data']} does not match the RGI regions of the "
        f"region blocks ({sourceData}): leave it out."
    )

    # The sources of all regions go to one GeoDataLoader per side, which combines them
    # glacier by glacier and takes its options keyed by source
    sources, options, optionsVal = [], {}, {}
    for name, block in regions.items():
        regionSources = block.get("geodetic_source") or []
        if isinstance(regionSources, str):
            regionSources = [regionSources]
        regionOptions = block.get("geodetic_source_options") or {}
        regionOptionsVal = block.get("geodetic_source_options_val") or regionOptions
        for source in regionSources:
            assert (
                source not in sources
            ), f"Geodetic source {source} is declared by two regions."
            sources.append(source)
            options[source] = regionOptions.get(source) or {}
            optionsVal[source] = regionOptionsVal.get(source) or {}

    merged.update(
        source_data=sourceData,
        splitVal="per-region",
        geodetic_source=sources,
        geodetic_source_options=options,
        geodetic_source_options_val=optionsVal,
        regions=regions,
    )
    return merged


def rgiRegionOf(rgiIds):
    """RGI region of RGI 6.0 ids: "RGI60-11.01450" -> 11."""
    return (
        pd.Series(rgiIds).str.split("-").str[1].str.split(".").str[0].astype(int).values
    )


def testSplits(params, df_test, keyGlacier="RGIId"):
    """The test sets a model is evaluated on, each with its stakes and its geodetic test.

    Without `regions` there is a single test set, named None: every test stake, and the
    geodetic test glaciers on `geodetic_source_test` (the training source by default).
    With `regions` there is one test set per region, holding the test stakes of its RGI
    regions and its own geodetic test source, since the sources of the regions cannot
    always share a dataloader (Hugonnet21 does not combine with the windowed sources).

    Returns {name: {"stakes", "glaciers", "geodeticSource", "geodeticSourceOptions"}}.
    """
    training = params["training"]
    regions = training.get("regions")
    if not regions:
        return {
            None: {
                "stakes": df_test,
                "glaciers": training.get("test_glaciers_geo")
                or list(df_test[keyGlacier].unique()),
                "geodeticSource": training.get("geodetic_source_test")
                or training["geodetic_source"],
                "geodeticSourceOptions": training.get("geodetic_source_test_options")
                or training.get("geodetic_source_options"),
            }
        }
    stakeRegion = pd.Series(rgiRegionOf(df_test.RGIId), index=df_test.index)
    splits = {}
    for name, region in regions.items():
        stakes = df_test[stakeRegion.isin(region["rgi_regions"])]
        assert len(stakes) > 0, f"Region {name} has no test stake."
        splits[name] = {
            "stakes": stakes,
            "glaciers": region.get("test_glaciers_geo")
            or list(stakes[keyGlacier].unique()),
            "geodeticSource": region.get("geodetic_source_test")
            or region["geodetic_source"],
            "geodeticSourceOptions": region.get("geodetic_source_test_options")
            or region.get("geodetic_source_options"),
        }
    return splits


def _valMask(df, splitVal, val_years=None, val_glaciers=None):
    """Stakes of `df` that go to validation with a split by year or by glacier."""
    if splitVal == "group-year":
        assert (
            val_years is not None
        ), "With splitVal='group-year', val_years must be provided."
        return df.YEAR.isin(set(int(y) for y in val_years)).values
    if splitVal == "group-rgi":
        return df.RGIId.isin(val_glaciers or []).values
    raise ValueError(f"No validation mask for splitVal={splitVal}.")


def getMetaData(featuresInpModel, sourceData):
    featuresToRemove = list(
        set(mbm.dataloader._default_input(sourceData)) - set(featuresInpModel)
    )
    if sourceData == "switzerland":
        metaData = list(
            set(
                [
                    "RGIId",
                    "POINT_ID",
                    "ID",
                    "GLWD_ID",
                    "N_MONTHS",
                    "MONTHS",
                    "PERIOD",
                    "GLACIER",
                    "YEAR",
                    "POINT_LAT",
                    "POINT_LON",
                ]
            ).union(set(featuresToRemove))
        )
    elif sourceData == "iceland":
        metaData = list(
            set(
                [
                    "RGIId",
                    "POINT_ID",
                    "ID",
                    "GLWD_ID",
                    "N_MONTHS",
                    "MONTHS",
                    "PERIOD",
                    "GLACIER",
                    "YEAR",
                    "POINT_LAT",
                    "POINT_LON",
                ]
            ).union(set(featuresToRemove))
        )
    elif sourceData == "norway":
        metaData = list(
            set(
                [
                    "RGIId",
                    "ID",
                    "N_MONTHS",
                    "MONTHS",
                    "PERIOD",
                    "YEAR",
                ]
            ).union(set(featuresToRemove))
        )
    elif "wgms" in sourceData:
        metaData = list(
            set(
                [
                    "RGIId",
                    "ID",
                    "N_MONTHS",
                    "MONTHS",
                    "PERIOD",
                    "YEAR",
                ]
            ).union(set(featuresToRemove))
        )
    else:
        raise ValueError(f"source_data={sourceData} is unknown")
    return metaData


def setFeatures(cfg, data_train, featuresInpModel):
    # feature_columns = list(
    #     data_train.columns.difference(cfg.metaData)
    #     .drop(cfg.notMetaDataNotFeatures)
    #     .drop("y")
    # )
    assert set(featuresInpModel).issubset(
        set(data_train.columns)
    ), f"Asked features are {featuresInpModel} but the dataframe columns are {data_train.columns}. The following features are missing: {set(featuresInpModel).difference(data_train.columns)}."
    cfg.setFeatures(featuresInpModel)


def trainValData(
    cfg,
    train_set,
    feature_columns,
    split_key="group-meas-id",
    val_glaciers=None,
    val_years=None,
    regions=None,
):
    """
    Split training dataset into train and validation sets.

    Args:
        - cfg: A configuration instance.
        - train_set: Dictionary with at least the following keys: `df_X` (pd.DataFrame) and `y` (pd.Series) which represent respectively the features and the targets.
        - feature_columns: List of string representing the columns to be used as features in the dataframe.
        - split_key (str): Type of split between the train and validation sets.
        - val_glaciers (list or None): Optional list of glaciers to be kept aside for validation. If this option is used the split is done per glacier and split_key is ignored.
        - val_years (list or None): Years kept aside for validation, with `split_key="group-year"`. A measurement goes to the side holding the year it was measured in, so that the geodetic windows of each side stay in that side's years.
        - regions (dict or None): With `split_key="per-region"`, the regions of `mergeRegions`: each stake is split with the rule (`splitVal`, `val_years`, `val_glaciers`) of the region its RGI id belongs to.
    """
    # Validation and train split:
    data_train = train_set["df_X"]
    data_train["y"] = train_set["y"]
    dataloader = mbm.dataloader.DataLoader(cfg, data=data_train)

    if split_key == "group-rgi" and val_glaciers is not None:

        full_df = data_train.reset_index()
        is_val = _valMask(full_df, "group-rgi", val_glaciers=val_glaciers)
        val_indices = full_df.loc[is_val].index.values
        train_indices = full_df.loc[~is_val].index.values

    elif split_key == "group-year":

        full_df = data_train.reset_index()
        is_val = _valMask(full_df, "group-year", val_years=val_years)
        val_indices = full_df.loc[is_val].index.values
        train_indices = full_df.loc[~is_val].index.values
        print(
            f"Split per year: {len(set(val_years))} validation years, "
            f"{full_df.loc[~is_val].YEAR.nunique()} train years"
        )

    elif split_key == "per-region":

        assert regions, "With splitVal='per-region', regions must be provided."
        full_df = data_train.reset_index()
        stakeRegion = rgiRegionOf(full_df.RGIId)
        is_val = pd.Series(False, index=full_df.index).values
        declared = set()
        for name, region in regions.items():
            inRegion = pd.Series(stakeRegion).isin(region["rgi_regions"]).values
            declared.update(region["rgi_regions"])
            regionVal = inRegion & _valMask(
                full_df,
                region["splitVal"],
                val_years=region.get("val_years"),
                val_glaciers=region.get("val_glaciers"),
            )
            is_val |= regionVal
            print(
                f"Region {name} ({region['splitVal']}): "
                f"{full_df.loc[inRegion & ~regionVal].ID.nunique()} train and "
                f"{full_df.loc[regionVal].ID.nunique()} validation measurements"
            )
        undeclared = sorted(set(stakeRegion) - declared)
        assert (
            not undeclared
        ), f"Stakes of RGI regions {undeclared} belong to no region."
        val_indices = full_df.loc[is_val].index.values
        train_indices = full_df.loc[~is_val].index.values

    else:

        train_itr, val_itr = dataloader.set_train_test_split(
            test_size=0.2, type_fold=split_key
        )

        # Get all indices of the training and valing dataset at once from the iterators. Once called, the iterators are empty.
        train_indices, val_indices = list(train_itr), list(val_itr)

    df_X_train = data_train.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values

    # Get val set
    df_X_val = data_train.iloc[val_indices]
    y_val = df_X_val["POINT_BALANCE"].values

    assert all(data_train.POINT_BALANCE == train_set["y"])

    all_columns = list(set(feature_columns + cfg.fieldsNotFeatures))
    print("Shape of training dataset:", df_X_train[all_columns].shape)
    print("Shape of validation dataset:", df_X_val[all_columns].shape)
    print("Running with features:", feature_columns)

    return df_X_train, y_train, df_X_val, y_val


def testData(cfg, test_set, feature_columns):
    all_columns = list(set(feature_columns + cfg.fieldsNotFeatures))
    df_X_test_subset = test_set["df_X"][all_columns]
    print("Shape of testing dataset:", df_X_test_subset.shape)
    return df_X_test_subset


def geodetic_table(geoTarget, geoErr, geoPred, gdl):
    """Geodetic targets and predictions returned by `mbm.training.eval_geodetic`, as
    a dataframe with one row per geodetic window of every glacier. The columns start
    and end are the bounds of the window."""
    rows = []
    for g in geoTarget:
        for (start, end), target, err, pred in zip(
            gdl.geodetic_periods(g), geoTarget[g], geoErr[g], geoPred[g]
        ):
            rows.append(
                {
                    "RGIId": g,
                    "start": start,
                    "end": end,
                    "target": target,
                    "err": err,
                    "pred": pred,
                }
            )
    return pd.DataFrame(rows)


def geodetic_windows(df_geo, with_target=True, default_period=None):
    """Geodetic windows of every glacier in the format `mbm.plots.cumulatedMassChange`
    expects, from a table built by `geodetic_table`.

    Args:
        df_geo (pd.DataFrame): table with one row per geodetic window.
        with_target (bool): include the observed rates and their uncertainty.
            Without them only the bounds are given, which restricts the plot to the
            geodetic windows.
        default_period (tuple): bounds used when the table has no start and end
            columns, which is the case of the tables saved when a glacier could only
            have one geodetic window.
    """
    geo = {}
    for g, df in df_geo.groupby("RGIId", sort=False):
        if "start" in df.columns:
            entry = {"start": df.start.tolist(), "end": df.end.tolist()}
        else:
            entry = {
                "start": [default_period[0]] * len(df),
                "end": [default_period[1]] * len(df),
            }
        if with_target:
            entry["mean"] = df.target.to_numpy()
            entry["err"] = df.err.to_numpy()
        geo[g] = entry
    return geo


def default_glacier_name(rgi_id):
    return {
        # # Norway
        # RGI60-08.00038;  # Nigardsbreen
        # RGI60-08.00087;  # Jostedalsbreen
        # RGI60-08.00147;  # Folgefonna
        # RGI60-08.00203;  # Hardangerjøkulen
        # Italy
        "RGI60-11.00695": "Glatschiu dil segnas",
        "RGI60-11.03005": "Miage",
        "RGI60-11.03001": "Brenva",
        "RGI60-11.01473": "Laaser Ferner",
        "RGI60-11.00597": "Übeltalferner",
        "RGI60-11.01776": "Langenferner/Vedretta Lunga",
        "RGI60-11.03166": "Grand Etret",
        "RGI60-11.00647": "Gigante Occidentale (Ries Ovest) / Westl. Rieser",
        # France, Mont Blanc
        "RGI60-11.03643": "Mer de Glace/Geant",
        "RGI60-11.03638": "Argentière",
        "RGI60-11.03646": "Bossons",
        "RGI60-11.03647": "Taconnaz",
        "RGI60-11.03296": "Tricot",
        "RGI60-11.03438": "Tete Rousse",
        "RGI60-11.03648": "Bionnassay",
        "RGI60-11.03601": "Armancette",
        "RGI60-11.03650": "Covagnet",
        "RGI60-11.03276": "Miage 1",
        "RGI60-11.03388": "Miage 2",
        "RGI60-11.03579": "Miage 3",
        "RGI60-11.03649": "Miage 4",
        "RGI60-11.03651": "Tré-la-Tête",
        "RGI60-11.03339": "Glaciers",
        # France, Belledonne
        "RGI60-11.03674": "Saint Sorlin",
        # France, Ecrins
        "RGI60-11.03677": "Meije",
        "RGI60-11.03684": "Blanc",
        # France, Pyrénées
        "RGI60-11.03232": "Ossoue",
        "RGI60-11.03208": "Aneto",
        # Austria
        "RGI60-11.00897": "Hintereisferner",
        "RGI60-11.00787": "Kesselwandferner",
        "RGI60-11.00781": "Jamtalferner",
        "RGI60-11.00116": "Venedigerkees",
        "RGI60-11.00251": "Kleinfleisskees",
        "RGI60-11.00289": "Goldbergkees",
        "RGI60-11.00006": "Schladminger",
        # Switzerland
        "RGI60-11.01270": "Grindelwald",
        "RGI60-11.01450": "Aletsch",
        "RGI60-11.01733": "Hangend",
        "RGI60-11.01328": "Unteraar",
        "RGI60-11.01238": "Rhone",
        "RGI60-11.02249": "Tsanfleuron",
        "RGI60-11.01702": "Kander",
        "RGI60-11.00872": "Hüfifirn",
        "RGI60-11.02774": "Giétro",
        "RGI60-11.01876": "Gries",
        "RGI60-11.02746": "Schwarzberg",
        "RGI60-11.02810": "Arolla",
        "RGI60-11.02775": "Orny",
        "RGI60-11.02507": "Brunegg",
        "RGI60-11.00804": "Silvretta",
        "RGI60-11.00752": "Vorab",
        "RGI60-11.02787": "Mont Collon",
        "RGI60-11.01267": "Porchabella",
        "RGI60-11.02634": "Prafleuri",
        "RGI60-11.01946": "Morteratsch",
        "RGI60-11.00878": "Claridenfirn I",
        "RGI60-11.00843": "Claridenfirn II",
        "RGI60-11.00819": "Claridenfirn III",
        "RGI60-11.01962": "Corvatsch",
        "RGI60-11.02740": "Trient",
        "RGI60-11.01367": "St. Annafirn",
        "RGI60-11.02745": "Allalin",
        "RGI60-11.01280": "Glatscher da Plattas",
        "RGI60-11.02679": "Hohlaubgletscher",
        "RGI60-11.02773": "Findelen",
        "RGI60-11.02448": "Plan Névé",
        "RGI60-11.02282": "Vadrec dal Castel Nord",
        "RGI60-11.02624": "Feegletscher",
    }.get(rgi_id)


def recursive_update(target, source):
    for key in source:
        if key not in target.keys():
            print(f"Creating key {key} as it does not exist in target")
            if isinstance(source[key], dict):
                target[key] = {}
            else:
                target[key] = None
        if isinstance(target[key], dict):
            recursive_update(target[key], source[key])
        else:
            target[key] = source[key]
