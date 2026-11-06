import os
import urllib.request

from data_processing.product_utils import get_data_path

"""
This script downloads the stake mass balance measurements of the GLACIOCLIM observatory
for the French Alps, published at https://glacioclim.osug.fr/Donnees-des-Alpes
("Données de bilans de masse").

Every dataset of the page is a public share of the OSUG Nextcloud, downloaded at
`<NEXTCLOUD>/<token>/download`. The tokens are those of the links of the page. Each
share of `SHARES` is a zip archive with one CSV per year for one zone of one glacier
and one period, except Sarennes, whose stakes come as one Excel workbook with a sheet per
year and the three periods in it (`SARENNES_SHARE`).

Only the measurements made at the stakes are downloaded. The other mass balance shares of
the page are left out, because they are not point measurements:
- the glacier-wide series: Saint-Sorlin s5GK7sgjXEXqmEQ, Gébroulaz iejdY4DpfmEpd6x,
  Argentière bAym9FbE89WjBAs, Mer de Glace oDGCSjT9LCR2yqo, Sarennes ToNQAYbQdZyqEkK and
  Glacier Blanc 24x3SjgsEezRtt6 (the only mass balance share of Glacier Blanc);
- the reconstructed mass balances since 1905 ("Bilans de masse reconstitués"):
  Saint-Sorlin xj3KqXyRBpjf7Ma, Gébroulaz b87RFgbQX7Yq2LR, Argentière aKdfDqnAZYxn3jL
  and Mer de Glace YjgfaPxDgHwMmLj.
"""

NEXTCLOUD = "https://nextcloud.osug.fr/index.php/s"

# (token, glacier, zone, period) of the zip archives of stake measurements
SHARES = [
    ("a3cCqWoJdgRWgiX", "saint_sorlin", "accumulation", "annual"),
    ("N3mwC5mXmpPnyDM", "saint_sorlin", "ablation", "annual"),
    ("4Naxf8i5K8wbzRd", "saint_sorlin", "accumulation", "winter"),
    ("g6rxMiyzNz4P9Gj", "saint_sorlin", "ablation", "winter"),
    ("o8L96wPdcFc6Brf", "saint_sorlin", "accumulation", "summer"),
    ("reyweAknJebGgAJ", "saint_sorlin", "ablation", "summer"),
    ("KqLraLsNqjqFwja", "gebroulaz", "accumulation", "annual"),
    ("JCHfDxYrMgYE4eD", "gebroulaz", "ablation", "annual"),
    ("GTf9bN2P3xndcMr", "gebroulaz", "accumulation", "winter"),
    ("tQ4jcmCQeMzgJZc", "gebroulaz", "ablation", "winter"),
    ("cG6SSYNMDnSH4BK", "gebroulaz", "accumulation", "summer"),
    ("XKX6966FzmFzA2f", "gebroulaz", "ablation", "summer"),
    ("MJSJ3SDgDE7gwY7", "argentiere", "accumulation", "annual"),
    ("65R2kJP8DMkFGzC", "argentiere", "ablation", "annual"),
    ("TtGnFc644woEJGB", "argentiere", "tour_noir", "annual"),
    ("MiXn2w62SqBqT38", "argentiere", "accumulation", "winter"),
    ("ptpwXwbdpNxBNoX", "argentiere", "ablation", "winter"),
    ("YxTaXjLt8aFTxLg", "argentiere", "tour_noir", "winter"),
    ("gtL5JK9QfRSawEH", "argentiere", "accumulation", "summer"),
    ("mAEJ6LKH8bJqzXm", "argentiere", "ablation", "summer"),
    ("J2tFtnsfsfJGfJ9", "argentiere", "tour_noir", "summer"),
    ("dHs5DC4HMqLA3zr", "mer_de_glace", "accumulation", "annual"),
    ("nQHZr7ADWmqq33S", "mer_de_glace", "langue", "annual"),
    ("7QQHXCktpfyG8bk", "mer_de_glace", "leschaux", "annual"),
    ("MmbfyZfgwSJx7KH", "mer_de_glace", "tacul", "annual"),
    ("oxawoedLeeyiBDt", "mer_de_glace", "talefre", "annual"),
    ("TcSrxbi6xYcpqGn", "mer_de_glace", "accumulation", "winter"),
    ("5cQLfyrQQPSYJCK", "mer_de_glace", "langue", "winter"),
    ("jyxbC86GdQrJ6Xw", "mer_de_glace", "leschaux", "winter"),
    ("TjdZqc54ZyJ2tpd", "mer_de_glace", "tacul", "winter"),
    ("mzxeoFtaK8i3SAz", "mer_de_glace", "talefre", "winter"),
    ("CPg9m8tdaR5XmgQ", "mer_de_glace", "accumulation", "summer"),
    ("d6dF787s3GR97tQ", "mer_de_glace", "langue", "summer"),
    ("GdfmwoZbDMYPn9t", "mer_de_glace", "leschaux", "summer"),
    ("x8mS5HDsHPy9Qii", "mer_de_glace", "tacul", "summer"),
    ("6HzS69bpZtbfesR", "mer_de_glace", "talefre", "summer"),
]
# Excel workbook of the stakes of Sarennes, 1994-2020
SARENNES_SHARE = "DEgEWn3NpD6L5Ab"


def glacioclim_path():
    return os.path.join(get_data_path(), "stakes", "glacioclim")


def raw_path():
    return os.path.join(glacioclim_path(), "raw")


def share_file(token):
    """Local file of a share: the zip archive, or the workbook of Sarennes."""
    ext = ".xls" if token == SARENNES_SHARE else ".zip"
    return os.path.join(raw_path(), token + ext)


def all_tokens():
    return [token for token, *_ in SHARES] + [SARENNES_SHARE]


def download_all(overwrite=False):
    """Download the stake shares of `SHARES` and `SARENNES_SHARE` to `raw_path()`, as
    they are served (nothing is extracted). The files already present are kept unless
    `overwrite`."""
    os.makedirs(raw_path(), exist_ok=True)
    for token in all_tokens():
        path = share_file(token)
        if os.path.isfile(path) and not overwrite:
            continue
        print(f"Downloading GLACIOCLIM share {token}")
        with urllib.request.urlopen(f"{NEXTCLOUD}/{token}/download", timeout=300) as r:
            content = r.read()
        # Write through a temporary file, so that an interrupted download leaves no
        # truncated file behind to be taken as complete
        with open(path + ".part", "wb") as f:
            f.write(content)
        os.replace(path + ".part", path)
    print(f"GLACIOCLIM shares in {raw_path()}")
