# utils/plotting/style.py
import matplotlib.pyplot as plt
from pathlib import Path

_STYLE_APPLIED = False


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

    style_path = Path(__file__).with_name("example.mplstyle")
    plt.style.use(style_path)
    _STYLE_APPLIED = True


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
