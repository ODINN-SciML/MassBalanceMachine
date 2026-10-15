"""Colour maps, norms and default line styles shared across plots."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex


def _default_style(color_annual, color_winter):
    """Return default colour/line-style/label settings for annual and winter series."""
    return {
        "annual": {"color": color_annual, "ls": "-", "label": "Annual"},
        "winter": {"color": color_winter, "ls": "-", "label": "Winter"},
    }


def get_color_maps(vmin, vmax):
    """
    Choose a colour map and norm for a mass balance map given its value range.

    - Range crossing zero: diverging ``coolwarm_r`` centred on 0. The norm is
      made symmetric using ``vmin`` (i.e. ``vmax`` is set to ``-vmin``).
    - Only negative values: ``Reds_r``.
    - Otherwise (only non-negative values): ``Blues``.

    Parameters
    ----------
    vmin, vmax : float
        Minimum and maximum of the data to plot.

    Returns
    -------
    tuple
        (cmap_name, norm) to pass to matplotlib/xarray plotting functions.
    """
    if vmin < 0 and vmax > 0:
        norm_ann = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=-vmin)
        cmap_ann = "coolwarm_r"
    elif vmin < 0 and vmax < 0:
        norm_ann = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_ann = "Reds_r"
    else:
        norm_ann = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_ann = "Blues"
    return cmap_ann, norm_ann


def get_cmap_hex(cmap, length):
    """
    Function to get a get a list of colours as hex codes

    :param cmap:    name of colourmap
    :type cmap:     str

    :return:        list of hex codes
    :rtype:         list
    """
    # Get cmap
    rgb = plt.get_cmap(cmap)(np.linspace(0, 1, length))

    # Convert to hex
    hex_codes = [to_hex(rgb[i, :]) for i in range(rgb.shape[0])]

    return hex_codes
