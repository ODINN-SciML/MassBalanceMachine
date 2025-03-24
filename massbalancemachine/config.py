from typing import List
import os


class Config:
    def __init__(
            self,
            numJobs: int = -1,
            testSize: float = 0.3,
            nSplits: int = 5,
            seed: int = 30,
            metaData: List[str] = ["RGIId", "POINT_ID", "ID", "N_MONTHS", "MONTHS"],
            notMetaDataNotFeatures: List[str] = ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON", "ALTITUDE_CLIMATE"],
            loss: str = 'MSE',
        ) -> None:
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
        """

        # Customizable attributes
        self.numJobs = numJobs or max(1, min(os.cpu_count()-2, 25)) # Use provided value otherwise use number of logical cores minus 2 to keep resources
        self.testSize = testSize
        self.nSplits = nSplits
        self.seed = seed
        self.metaData = metaData
        self.notMetaDataNotFeatures = notMetaDataNotFeatures
        self.loss = loss

        # Constant attributes
        self.base_url_w5e5 = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/"
        self.month_abbr_hydr = {
            'sep': 1,
            'oct': 2,
            'nov': 3,
            'dec': 4,
            'jan': 5,
            'feb': 6,
            'mar': 7,
            'apr': 8,
            'may': 9,
            'jun': 10,
            'jul': 11,
            'aug': 12,
            #'sep_': 13,
        }

    @property
    def fieldsNotFeatures(self):
        return self.metaData + self.notMetaDataNotFeatures

class SwitzerlandConfig(Config):
    def __init__(
            self,
            *args,
            metaData: List[str] = ["RGIId", "POINT_ID", "ID", "GLWD_ID", "N_MONTHS", "MONTHS", "PERIOD", "GLACIER", "YEAR", "POINT_LAT", "POINT_LON"],
            # notMetaDataNotFeatures: List[str] = ["POINT_BALANCE", "YEAR", "POINT_LAT", "POINT_LON"],
            notMetaDataNotFeatures: List[str] = ["POINT_BALANCE", ],
            **kwargs,
        ):
        super().__init__(*args, **kwargs, metaData=metaData, notMetaDataNotFeatures=notMetaDataNotFeatures)