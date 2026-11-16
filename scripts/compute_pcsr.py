import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import argparse
import multiprocessing

import tqdm

import massbalancemachine as mbm
from data_processing.parallel_utils import worker_count
from data_processing.potential_radiation import (
    PCSR_FILE,
    ensure_pcsr_file,
    horizon_tiles_of_dem,
)
from data_processing.Product import Product
from data_processing.product_utils import get_data_path

GIB = 1024**3

parser = argparse.ArgumentParser(
    "Write the potential clear-sky direct radiation (pcsr.nc) of every glacier of a "
    "grid tree from its dem.nc. The geodetic grids and the stakes read it when they "
    "are prepared; computing it beforehand, in parallel, keeps that step quick."
)
parser.add_argument(
    "source",
    type=str,
    help="Grid tree under .data/grids/, e.g. Hugonnet21, GLAMOS, Rabatel16, NVE_GAO.",
)
parser.add_argument(
    "--region",
    type=str,
    default=None,
    help="Only the glaciers of this RGI region, e.g. 11 (folders RGI60-11, "
    "RGI2000-v7.0-G-11).",
)
parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Number of processes (default: as many as the free memory allows).",
)
parser.add_argument(
    "--dataPath",
    type=str,
    default=None,
    help="Data tree to use instead of the repository .data/.",
)
args = parser.parse_args()

if args.dataPath is not None:
    mbm.set_data_path(args.dataPath)

root = os.path.join(get_data_path(), "grids", args.source)
assert os.path.isdir(root), f"No grid tree {root}."

folders = []
for dirpath, dirnames, filenames in os.walk(root):
    if "dem.nc" not in filenames:
        continue
    if args.region is not None:
        rel = os.path.relpath(dirpath, root).split(os.sep)[0]
        if not (
            rel == f"RGI60-{int(args.region):02d}"
            or rel == f"RGI2000-v7.0-G-{int(args.region):02d}"
        ):
            continue
    if not Product(os.path.join(dirpath, PCSR_FILE)).is_up_to_date():
        folders.append(dirpath)
folders.sort()
print(f"{len(folders)} glaciers of {root} without {PCSR_FILE}.")


def _compute(folder):
    try:
        ensure_pcsr_file(folder)
    except Exception as e:
        return folder, repr(e)
    return folder, None


if __name__ == "__main__" and folders:
    # The DEM tiles of the horizon first, one glacier at a time: parallel workers
    # would otherwise download the same tile at once
    print("Fetching the horizon DEM tiles")
    for folder in tqdm.tqdm(folders):
        horizon_tiles_of_dem(os.path.join(folder, "dem.nc"))

    n_workers = worker_count(per_worker_bytes=1 * GIB, requested=args.workers)
    print(f"Computing {PCSR_FILE} with {n_workers} workers")
    failed = []
    with multiprocessing.Pool(n_workers) as pool:
        for folder, error in tqdm.tqdm(
            pool.imap_unordered(_compute, folders), total=len(folders)
        ):
            if error is not None:
                failed.append((folder, error))
    for folder, error in failed:
        print(f"Failed: {folder}: {error}")
    print(f"Done: {len(folders) - len(failed)} written, {len(failed)} failed.")
