import seaborn as sns
from cmcrameri import cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
from matplotlib.patches import Rectangle
from typing import Sequence, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import colormaps as cmaps

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)

import massbalancemachine as mbm

from regions.Switzerland.scripts.utils import *
from regions.Switzerland.scripts.config_CH import *
from regions.Switzerland.scripts.geo_data import *


def mbm_glwd_pred(PATH_PREDICTIONS, GLACIER_NAME):
    # Define the path to model predictions
    path_results = os.path.join(PATH_PREDICTIONS, GLACIER_NAME)

    # Extract available years from NetCDF filenames
    years = sorted(
        [
            int(f.split("_")[1])
            for f in os.listdir(path_results)
            if f.endswith("_annual.zarr")
        ]
    )

    # Extract model-predicted mass balance
    pred_gl = []
    for year in years:
        file_path = os.path.join(path_results, f"{GLACIER_NAME}_{year}_annual.zarr")
        if not os.path.exists(file_path):
            print(f"Warning: Missing MBM file for {GLACIER_NAME} ({year}). Skipping...")
            pred_gl.append(np.nan)
            continue

        ds = xr.open_dataset(file_path)
        pred_gl.append(ds.pred_masked.mean().item())

    # Create DataFrame
    MBM_glwmb = pd.DataFrame(pred_gl, index=years, columns=["MBM Balance"])
    return MBM_glwmb
