from typing import List, Dict, Tuple, Optional
import os


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

    def __init__(
        self,
        numJobs: int = -1,
        testSize: float = 0.3,
        nSplits: int = 5,
        seed: int = 20,
        metaData: List[str] = ["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
            "ALTITUDE_CLIMATE",
        ],
        loss: str = "MSE",
        bnds: Dict[str, Tuple[float, float]] = {},
    ) -> None:

        if "CI" in os.environ:
            assert (
                seed is not None
            ), "In the CI and in the tests the seed must be defined."

        self.numJobs = numJobs or max(1, min((os.cpu_count() or 4) - 2, 25))
        self.testSize = testSize
        self.nSplits = nSplits
        self.seed = seed
        self.metaData = metaData
        self.notMetaDataNotFeatures = notMetaDataNotFeatures
        self.featureColumns: List[str] = []
        self.loss = loss

        # Constant attributes
        self.base_url_w5e5 = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"

        # Scaling bounds (unchanged)
        self.bnds = bnds or {
            "ALTITUDE_CLIMATE": (1500, 3000),
            "ELEVATION_DIFFERENCE": (0, 1000),
            "POINT_ELEVATION": (2000, 3500),
            "aspect": (0, 360),
            "consensus_ice_thickness": (0, 300),
            "fal": (0, 1),
            "hugonnet_dhdt": (-5, 5),
            "millan_v": (0, 300),
            "pcsr": (0, 500),  # Potential incoming clear sky solar radiation
            "slhf": (-10e6, 10e6),  # Surface latent heat flux
            "slope": (0, 90),
            "sshf": (-10e6, 10e6),  # Surface sensible heat flux
            "ssrd": (-10e6, 10e6),  # Surface solar radiation downwards
            "str": (-10e6, 10e6),  # Surface net thermal radiation
            "t2m": (-20, 15),  # Temperature
            "tp": (0, 0.1),  # Precipitation
            "u10": (-10, 10),  # Horizontal wind component
            "v10": (-10, 10),  # Vertical wind component
            "svf": (0, 1),  # Sky view factor
        }

    # ---------------- Properties / setters ----------------

    @property
    def fieldsNotFeatures(self) -> List[str]:
        return self.metaData + self.notMetaDataNotFeatures

    def setFeatures(self, featureColumns: List[str]) -> None:
        self.featureColumns = featureColumns


class SwitzerlandConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId",
            "POINT_ID",
            "ID",
            "GLWD_ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
            "ALTITUDE_CLIMATE",
            "POINT_ELEVATION",
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = (
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
            )
            self.dataPath = mbmDir + "../data/"
        else:
            if dataPath != "" and not dataPath.endswith("/"):
                dataPath = dataPath + "/"
            self.dataPath = dataPath
        super().__init__(
            *args,
            metaData=metaData,
            notMetaDataNotFeatures=notMetaDataNotFeatures,
            numJobs=numJobs,
            **kwargs,
        )
        self.bnds["slope_sgi"] = self.bnds["slope"]
        self.bnds["aspect_sgi"] = self.bnds["aspect"]


class FranceConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
            "GLACIER_ZONE",
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = (
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
            )
            self.dataPath = mbmDir + "../data/"
        else:
            if dataPath != "" and not dataPath.endswith("/"):
                dataPath = dataPath + "/"
            self.dataPath = dataPath
        super().__init__(
            *args,
            metaData=metaData,
            notMetaDataNotFeatures=notMetaDataNotFeatures,
            numJobs=numJobs,
            **kwargs,
        )


class ItalyAustriaConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = (
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
            )
            self.dataPath = mbmDir + "../data/"
        else:
            if dataPath != "" and not dataPath.endswith("/"):
                dataPath = dataPath + "/"
            self.dataPath = dataPath
        super().__init__(
            *args,
            metaData=metaData,
            notMetaDataNotFeatures=notMetaDataNotFeatures,
            numJobs=numJobs,
            **kwargs,
        )


class NorwayConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = (
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
            )
            self.dataPath = mbmDir + "../data/"
        else:
            if dataPath != "" and not dataPath.endswith("/"):
                dataPath = dataPath + "/"
            self.dataPath = dataPath
        super().__init__(
            *args,
            metaData=metaData,
            notMetaDataNotFeatures=notMetaDataNotFeatures,
            numJobs=numJobs,
            **kwargs,
        )


class IcelandConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId",
            "POINT_ID",
            "ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = (
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")) + "/"
            )
            self.dataPath = mbmDir + "../data/"
        else:
            if dataPath != "" and not dataPath.endswith("/"):
                dataPath = dataPath + "/"
            self.dataPath = dataPath
        super().__init__(
            *args,
            metaData=metaData,
            notMetaDataNotFeatures=notMetaDataNotFeatures,
            numJobs=numJobs,
            **kwargs,
        )
