import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import argparse
import hashlib
import shutil

import geopandas as gpd

import massbalancemachine as mbm
from data_processing import rabatel16
from data_processing.Product import Product

parser = argparse.ArgumentParser(
    "Write the Rabatel16 files of the Hugging Face dataset "
    f"{rabatel16.HUB_REPO_ID} from the raw outlines and IGN DEM: the outlines of the "
    "glaciers of the ALPGM table, and the DEM with its missing values declared, "
    "cropped around them. The files are written to a folder to upload; nothing is "
    "uploaded."
)
parser.add_argument(
    "rawFolder",
    type=str,
    help=f"Folder holding the raw files, {rabatel16.RAW_OUTLINES_FILE} and "
    f"{rabatel16.RAW_DEM_FILE}.",
)
parser.add_argument("outFolder", type=str, help="Folder to write the files to.")
parser.add_argument(
    "--border",
    type=int,
    default=80,
    help="OGGM border, in grid cells, the DEM must cover around every glacier "
    "(default: %(default)s).",
)
parser.add_argument(
    "--padCells",
    type=int,
    default=20,
    help="DEM cells kept beyond that border, for the resampling (default: "
    "%(default)s).",
)
parser.add_argument(
    "--install",
    default=False,
    action="store_true",
    help="Also copy the files to the data tree, .data/Rabatel16, where MBM would "
    "download them, replacing what is there.",
)
parser.add_argument(
    "--dataPath",
    type=str,
    default=None,
    help="Data tree of MBM, if not the .data/ of the repository.",
)
args = parser.parse_args()

if args.dataPath is not None:
    mbm.set_data_path(args.dataPath)

paths = rabatel16.write_rabatel16_hub_files(
    args.rawFolder, args.outFolder, border=args.border, pad_cells=args.padCells
)
n = len(gpd.read_file(paths[0]))
print(f"{n} glaciers")
for path in paths:
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB, sha256 {sha256}")

if args.install:
    for path in paths:
        dst = os.path.join(rabatel16.rabatel16_data_folder(), os.path.basename(path))
        p = Product(dst)
        shutil.copyfile(path, dst)
        p.gen_chk()
        print(f"installed {dst}")

print(
    "\nUpload with\n\n"
    f"    hf upload {rabatel16.HUB_REPO_ID} {os.path.abspath(args.outFolder)} "
    f"{rabatel16.HUB_FOLDER} --repo-type dataset\n\n"
    "then pin the revision in rabatel16.HUB_REVISION."
)
