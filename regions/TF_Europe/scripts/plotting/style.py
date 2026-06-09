# utils/plotting/style.py
import matplotlib.pyplot as plt
import matplotlib as mpl


def alpha_labels(n: int):
    """
    Generate subplot labels '(a)', '(b)', ... '(z)', '(aa)', '(ab)', ... .

    Parameters
    ----------
    n : int
        Number of labels to generate.

    Returns
    -------
    list of str
        List of n labels formatted as '(...)'.
    """

    def to_label(k: int) -> str:
        # 0 -> a, 25 -> z, 26 -> aa, ...
        s = ""
        k += 1
        while k > 0:
            k, r = divmod(k - 1, 26)
            s = chr(97 + r) + s
        return f"({s})"

    return [to_label(i) for i in range(n)]


NATURE_PALETTE = {
    "black": "#000000",
    "orange": "#e69f00",
    "sky_blue": "#56b4e9",
    "bluish_green": "#009e73",
    "yellow": "#f0e442",
    "blue": "#0072b2",
    "vermillion": "#d55e00",
    "reddish_purple": "#cc79a7",
}

# Convenient ordered list for sequential assignment
NATURE_COLORS = list(NATURE_PALETTE.values())

# Nature figure specs (for reference when setting figsize)
NATURE_SPECS = {
    "single_col_mm": 89,
    "double_col_mm": 183,
    "max_height_mm": 170,
    "font_min_pt": 5,
    "font_max_pt": 7,
    "font_family": "Arial",
    "dpi": 300,
}


def nature_figsize(cols=1, height_mm=80):
    """Return figsize in inches for Nature single or double column."""
    width_mm = (
        NATURE_SPECS["single_col_mm"] if cols == 1 else NATURE_SPECS["double_col_mm"]
    )
    return (width_mm / 25.4, height_mm / 25.4)


def apply_nature_style(ax, fontsize=6, box=False):
    if box:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
    else:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_linewidth(0.5)
    ax.tick_params(labelsize=fontsize, width=0.5, length=3)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize + 1)
    ax.grid(color="#e0e0e0", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.width": 0.5,
        "ytick.major.size": 3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
