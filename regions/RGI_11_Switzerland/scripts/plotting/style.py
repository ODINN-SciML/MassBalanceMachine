# utils/plotting/style.py
import matplotlib.pyplot as plt


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
