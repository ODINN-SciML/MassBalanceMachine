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
    root_mean_squared_error,
)

from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.config_CH import *
from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.geo_data import *

from papers.Transformer_TF.PMB_sources.RGI_11_Switzerland.scripts.plotting.palettes import (
    get_color_maps,
)

import massbalancemachine as mbm

mbm.plots.use_mbm_style()


def _load_lstm_ds(glacier, year, path_pred_lstm, period, apply_smoothing_fn=None):
    zpath = os.path.join(path_pred_lstm, glacier, f"{glacier}_{year}_{period}.zarr")
    if not os.path.exists(zpath):
        return None
    ds = xr.open_zarr(zpath)
    if apply_smoothing_fn is not None:
        ds = apply_smoothing_fn(ds)
    return ds


import string
