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
            - months_tail_pad (list of str): Months to pad the start of the hydrological year.
            - months_head_pad (list of str): Months to pad the end of the hydrological year.
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
        notMetaDataNotFeatures: List[str] = ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON", "ALTITUDE_CLIMATE"],
        loss: str = 'MSE',
        bnds: Dict[str, Tuple[float, float]] = {},
    ) -> None:

        if "CI" in os.environ:
            assert seed is not None, "In the CI and in the tests the seed must be defined."

        self.numJobs = numJobs or max(1, min((os.cpu_count() or 4) - 2, 25))
        self.testSize = testSize
        self.nSplits = nSplits
        self.seed = seed
        self.metaData = metaData
        self.notMetaDataNotFeatures = notMetaDataNotFeatures
        self.featureColumns: List[str] = []
        self.loss = loss

        # Padding to allow for flexible month ranges (customize freely)
        self.months_tail_pad: List[str] = ['aug_', 'sep_']  # before 'oct'
        self.months_head_pad: List[str] = ['oct_']           # after 'sep'
        
        # self.months_tail_pad: List[str] = []  # before 'oct'
        # self.months_head_pad: List[str] =  [] # after 'sep'
        
        # Constant attributes
        self.base_url_w5e5 = (
            "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"
        )

        # Scaling bounds (unchanged)
        self.bnds = bnds or {
            'ALTITUDE_CLIMATE': (1500, 3000),
            'ELEVATION_DIFFERENCE': (0, 1000),
            'POINT_ELEVATION': (2000, 3500),
            'aspect': (0, 360),
            'consensus_ice_thickness': (0, 300),
            'fal': (0, 1),
            'hugonnet_dhdt': (-5, 5),
            'millan_v': (0, 300),
            'pcsr': (0, 500),
            'slhf': (-10e6, 10e6),
            'slope': (0, 90),
            'sshf': (-10e6, 10e6),
            'ssrd': (-10e6, 10e6),
            'str': (-10e6, 10e6),
            't2m': (-20, 15),
            'tp': (0, 0.1),
            'u10': (-10, 10),
            'v10': (-10, 10),
        }

    # ---------------- Properties / setters ----------------

    @property
    def fieldsNotFeatures(self) -> List[str]:
        return self.metaData + self.notMetaDataNotFeatures

    def setFeatures(self, featureColumns: List[str]) -> None:
        self.featureColumns = featureColumns

    def set_month_padding(self, months_tail_pad: Optional[List[str]] = None,
                          months_head_pad: Optional[List[str]] = None) -> None:
        """
        Update padding tokens and rebuild month indexing.
        """
        if months_tail_pad is not None:
            self.months_tail_pad = list(months_tail_pad)
        if months_head_pad is not None:
            self.months_head_pad = list(months_head_pad)
        self._rebuild_month_index()

    # ---------------- Flexible month mapping ----------------

    def _rebuild_month_index(self) -> None:
        """
        Recompute month list and index mappings given current tail/head pads.
        """
        self.month_list = self.make_month_abbr_hydr(
            self.months_tail_pad, self.months_head_pad
        )  # returns ordered list of tokens

        # 0-based positions for array indexing
        self.month_pos0: Dict[str, int] = {m: i for i, m in enumerate(self.month_list)}
        # 1-based positions (if needed for display)
        self.month_pos1: Dict[str, int] = {m: i + 1 for i, m in enumerate(self.month_list)}

        # convenience
        self.max_T: int = len(self.month_list)
        
    def make_month_abbr_hydr(self, months_tail_pad: Optional[List[str]] = None,
                             months_head_pad: Optional[List[str]] = None) -> List[str]:
        """
        Create a flexible hydrological month token list depending on tail/head padding.

        Returns
        -------
        list[str] : ordered tokens, e.g.
            ['aug_', 'sep_', 'oct','nov','dec','jan','feb','mar','apr','may','jun','jul','aug','sep','oct_']
        """
        # use provided args; fall back to current attributes if None
        tail = list(months_tail_pad) if months_tail_pad is not None else list(self.months_tail_pad)
        head = list(months_head_pad) if months_head_pad is not None else list(self.months_head_pad)

        # Standard hydro year (oct..sep)
        hydro = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar',
                 'apr', 'may', 'jun', 'jul', 'aug', 'sep']

        full = tail + hydro + head
        return full


class SwitzerlandConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId", "POINT_ID", "ID", "GLWD_ID", "N_MONTHS", "MONTHS",
            "PERIOD", "GLACIER"
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON",
            "ALTITUDE_CLIMATE", "POINT_ELEVATION"
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../')) + '/'
            self.dataPath = mbmDir + '../data/'
        else:
            if dataPath != '' and not dataPath.endswith('/'):
                dataPath = dataPath + '/'
            self.dataPath = dataPath
        super().__init__(*args,
                         metaData=metaData,
                         notMetaDataNotFeatures=notMetaDataNotFeatures,
                         numJobs=numJobs,
                         **kwargs)
        self.bnds['slope_sgi'] = self.bnds['slope']
        self.bnds['aspect_sgi'] = self.bnds['aspect']


class FranceConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD",
            "GLACIER", "GLACIER_ZONE"
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../')) + '/'
            self.dataPath = mbmDir + '../data/'
        else:
            if dataPath != '' and not dataPath.endswith('/'):
                dataPath = dataPath + '/'
            self.dataPath = dataPath
        super().__init__(*args,
                         metaData=metaData,
                         notMetaDataNotFeatures=notMetaDataNotFeatures,
                         numJobs=numJobs,
                         **kwargs)


class ItalyAustriaConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD",
            "GLACIER"
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../')) + '/'
            self.dataPath = mbmDir + '../data/'
        else:
            if dataPath != '' and not dataPath.endswith('/'):
                dataPath = dataPath + '/'
            self.dataPath = dataPath
        super().__init__(*args,
                         metaData=metaData,
                         notMetaDataNotFeatures=notMetaDataNotFeatures,
                         numJobs=numJobs,
                         **kwargs)


class NorwayConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD",
            "GLACIER"
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../')) + '/'
            self.dataPath = mbmDir + '../data/'
        else:
            if dataPath != '' and not dataPath.endswith('/'):
                dataPath = dataPath + '/'
            self.dataPath = dataPath
        super().__init__(*args,
                         metaData=metaData,
                         notMetaDataNotFeatures=notMetaDataNotFeatures,
                         numJobs=numJobs,
                         **kwargs)


class IcelandConfig(Config):

    def __init__(
        self,
        *args,
        metaData: List[str] = [
            "RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS", "PERIOD",
            "GLACIER"
        ],
        notMetaDataNotFeatures: List[str] = [
            "POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"
        ],
        dataPath: str = None,
        numJobs: int = 28,
        **kwargs,
    ):
        if dataPath is None:
            mbmDir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../')) + '/'
            self.dataPath = mbmDir + '../data/'
        else:
            if dataPath != '' and not dataPath.endswith('/'):
                dataPath = dataPath + '/'
            self.dataPath = dataPath
        super().__init__(*args,
                         metaData=metaData,
                         notMetaDataNotFeatures=notMetaDataNotFeatures,
                         numJobs=numJobs,
                         **kwargs)
