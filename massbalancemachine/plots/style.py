# utils/plotting/style.py
import matplotlib.pyplot as plt
from pathlib import Path

_STYLE_APPLIED = True


def use_mbm_style(force: bool = False):
    """
    Apply the MBM matplotlib style.

    Parameters
    ----------
    force : bool, optional
        If True, re-apply the style even if it was already applied.
        Default is False (safe, idempotent).
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return

    style_path = Path(__file__).with_name("style.mplstyle")
    plt.style.use(style_path)
    _STYLE_APPLIED = True


COLOR_ANNUAL = "#c51b7d"
COLOR_WINTER = "#011959"
