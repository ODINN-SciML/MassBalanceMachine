from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import os


def _normalize_data_path(dataPath: Optional[str]) -> str:
    if dataPath is None:
        mbmDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
        return mbmDir + "../data/"
    if dataPath != "" and not dataPath.endswith("/"):
        dataPath += "/"
    return dataPath


DEFAULT_META = ["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"]
DEFAULT_NOT_META_NOT_FEATURES = [
    "POINT_BALANCE",
    "YEAR",
    "POINT_LAT",
    "POINT_LON",
    "ALTITUDE_CLIMATE",
]

DEFAULT_BNDS: Dict[str, Tuple[float, float]] = {
    "ALTITUDE_CLIMATE": (1500, 3000),
    "ELEVATION_DIFFERENCE": (0, 1000),
    "POINT_ELEVATION": (2000, 3500),
    "aspect": (0, 360),
    "consensus_ice_thickness": (0, 300),
    "fal": (0, 1),
    "hugonnet_dhdt": (-5, 5),
    "millan_v": (0, 300),
    "pcsr": (0, 500),
    "slhf": (-10e6, 10e6),
    "slope": (0, 90),
    "sshf": (-10e6, 10e6),
    "ssrd": (-10e6, 10e6),
    "str": (-10e6, 10e6),
    "t2m": (-20, 15),
    "tp": (0, 0.1),
    "u10": (-10, 10),
    "v10": (-10, 10),
    "svf": (0, 1),
}

# Small, declarative region differences live here.
REGION_SPECS: Dict[str, Dict[str, Any]] = {
    "switzerland": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "GLWD_ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
            "ALTITUDE_CLIMATE",
            "POINT_ELEVATION",
        ],
        "bnds_add": {  # aliases for SGI features
            "slope_sgi": "slope",
            "aspect_sgi": "aspect",
        },
        "bnds_remove": [],  # keep pcsr
        "numJobs": 28,
    },
    "france": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
            "GLACIER_ZONE",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    # These 4 are identical: just remove pcsr and use same fields (might remove later)
    "italy_austria": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    "norway": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    "svalbard": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    "iceland": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    "europe": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
    "europe_tf": {
        "metaData": [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
            "SOURCE_CODE",
            "RGI_REGION",
        ],
        "notMetaDataNotFeatures": ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
        "bnds_remove": ["pcsr"],
        "numJobs": 28,
    },
}


@dataclass
class Config:
    """
    Configuration class that defines the variables related to processing resources and the features to use.
    Attributes:
        - numJobs: Number of jobs to run in parallel for XGBoost. If not provided,
            the value used is equal to the number of logical cores minus 2 clamped
            between 1 and 25.
        - testSize: Proportion of the dataset to include in the test split.
        - nSplits: Number of splits for cross-validation.
        - seed (int): Seed for random operations to ensure reproducibility.
        - metaData (list of str): Metadata fields.
        - notMetaDataNotFeatures (list of str): Fields that are neither metadata nor
            features.
        - loss (str): Type of loss to use
        - bnds (dict of float tuple): Upper and lower bounds of each variable to
            scale them (useful for the neural network). These bounds don't clip
            the data and if a variable exceeds the bounds, its normalized
            counterpart will simply be outside of [0, 1].
    """

    numJobs: int = -1
    testSize: float = 0.3
    nSplits: int = 5
    seed: int = 20
    metaData: List[str] = field(default_factory=lambda: DEFAULT_META.copy())
    notMetaDataNotFeatures: List[str] = field(
        default_factory=lambda: DEFAULT_NOT_META_NOT_FEATURES.copy()
    )
    loss: str = "MSE"
    bnds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_BNDS.copy()
    )
    dataPath: str = field(default_factory=lambda: _normalize_data_path(None))

    # runtime populated
    featureColumns: List[str] = field(default_factory=list)

    # constant attribute
    base_url_w5e5: str = (
        "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/"
        "elev_bands/W5E5_w_data/"
    )

    def __post_init__(self) -> None:
        if "CI" in os.environ:
            assert self.seed is not None, "In CI/tests the seed must be defined."

        self.numJobs = self.numJobs or max(1, min((os.cpu_count() or 4) - 2, 25))

    # ---------------- Properties / setters ----------------
    @property
    def fieldsNotFeatures(self) -> List[str]:
        return self.metaData + self.notMetaDataNotFeatures

    def setFeatures(self, featureColumns: List[str]) -> None:
        self.featureColumns = featureColumns


class RegionConfig(Config):
    """
    One config class for all regions. Differences are applied via REGION_SPECS.
    You can also override anything via kwargs.
    """

    def __init__(
        self,
        region: str,
        *,
        dataPath: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        region_key = region.lower()
        spec = REGION_SPECS.get(region_key, {})

        # Start from region spec, then user overrides win
        meta = overrides.pop("metaData", spec.get("metaData", None))
        notmeta = overrides.pop(
            "notMetaDataNotFeatures", spec.get("notMetaDataNotFeatures", None)
        )
        numJobs = overrides.pop("numJobs", spec.get("numJobs", -1))
        bnds = overrides.pop("bnds", None)

        super().__init__(
            numJobs=numJobs,
            metaData=meta if meta is not None else DEFAULT_META.copy(),
            notMetaDataNotFeatures=(
                notmeta if notmeta is not None else DEFAULT_NOT_META_NOT_FEATURES.copy()
            ),
            bnds=bnds if bnds is not None else DEFAULT_BNDS.copy(),
            dataPath=_normalize_data_path(dataPath),
            **overrides,
        )

        # Apply bnds modifications from region spec
        for k in spec.get("bnds_remove", []):
            self.bnds.pop(k, None)

        # Add alias bnds (Switzerland case)
        # spec["bnds_add"] maps new_key -> existing_key
        for new_key, existing_key in spec.get("bnds_add", {}).items():
            if existing_key in self.bnds:
                self.bnds[new_key] = self.bnds[existing_key]

        self.region = region_key  # handy for logging


# Optional: keep old names as convenience aliases (no new classes needed)
def SwitzerlandConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("switzerland", *args, **kwargs)


def FranceConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("france", *args, **kwargs)


# These 4 below are identical so might remove them later on
def ItalyAustriaConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("italy_austria", *args, **kwargs)


def NorwayConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("norway", *args, **kwargs)


def SvalbardConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("svalbard", *args, **kwargs)


def IcelandConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("iceland", *args, **kwargs)


def EuropeConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("europe", *args, **kwargs)


def EuropeTFConfig(*args, **kwargs) -> RegionConfig:
    return RegionConfig("europe_tf", *args, **kwargs)
