import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import argparse
import datetime
import hashlib
import json
import shutil

import git

import massbalancemachine as mbm

parser = argparse.ArgumentParser(
    "Export a trained model with only what is needed to evaluate it: its parameters "
    "and its best checkpoint. The exported folder can be given to mbm-eval as is."
)
parser.add_argument("modelFolder", type=str, help="Name of the run under logs/.")
parser.add_argument(
    "destination",
    type=str,
    help="Folder in which the run is exported, as <destination>/<run name>.",
)
parser.add_argument(
    "-f",
    "--force",
    dest="force",
    default=False,
    action="store_true",
    help="Replace an existing export of the same run.",
)
args = parser.parse_args()

runName = os.path.basename(os.path.normpath(args.modelFolder))
pathFolder = os.path.join(mbm_path, "logs", runName)
assert os.path.isdir(
    pathFolder
), f"No run {runName} in {os.path.join(mbm_path, 'logs')}"
exportFolder = os.path.join(os.path.abspath(args.destination), runName)
if os.path.exists(exportFolder):
    assert args.force, f"{exportFolder} already exists, use --force to replace it."
    shutil.rmtree(exportFolder)
os.makedirs(exportFolder)

# The best checkpoint is the one the evaluation loads; it keeps its file name, since
# the selection of the evaluation reads the epoch and the score from it.
bestFile, bestVal = mbm.training.bestModelFile(pathFolder)
shutil.copy2(os.path.join(pathFolder, "params.json"), exportFolder)
shutil.copy2(bestFile, exportFolder)

with open(os.path.join(pathFolder, "params.json")) as f:
    params = json.load(f)
with open(bestFile, "rb") as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()
repo = git.Repo(mbm_path)
info = {
    "run": runName,
    "checkpoint": os.path.basename(bestFile),
    "checkpoint_sha256": sha256,
    "best_validation_score": bestVal,
    "training_commit": params.get("commit_hash"),
    "export_commit": repo.head.object.hexsha,
    "export_commit_dirty": repo.is_dirty(),
    "export_branch": None if repo.head.is_detached else repo.active_branch.name,
    "export_date": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    ),
}
with open(os.path.join(exportFolder, "export.json"), "w") as f:
    json.dump(info, f, indent=4)

print(f"Exported {runName} to {exportFolder}:")
for name in sorted(os.listdir(exportFolder)):
    print(f"  {name}")
