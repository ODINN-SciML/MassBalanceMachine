import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

from pathlib import Path
import shutil
import argparse
from huggingface_hub import login, upload_folder, HfApi

parser = argparse.ArgumentParser()
parser.add_argument("modelFolder", type=str, help="Folder of the model to export.")
parser.add_argument("tag", type=str, help="Tag to identify the model to publish.")
args = parser.parse_args()

modelFolder = args.modelFolder
tag = args.tag

repo_id = "MassBalanceMachine/MLP"

api = HfApi()
login()

source_folder = os.path.join(mbm_path, "logs", modelFolder)
source_model = os.path.join(source_folder, "best_model.json")
source_params = os.path.join(source_folder, "params.json")
if not os.path.isfile(source_model):
    source_model = os.path.join(source_folder, "model.json")
    assert os.path.isfile(source_model), f"The model file does not exist."
assert os.path.isfile(source_params), f"File {source_params} does not exist."
assert len(tag) > 0, "Tag cannot be empty"

publish_folder = os.path.join(mbm_path, "publish", modelFolder)
print(f"{publish_folder=}")
if os.path.isdir(publish_folder):
    shutil.rmtree(publish_folder)
os.makedirs(publish_folder, exist_ok=True)
shutil.copy2(source_model, os.path.join(publish_folder, "model.json"))
shutil.copy2(source_params, publish_folder)


commit_info = upload_folder(
    folder_path=publish_folder, repo_id=repo_id, repo_type="model"
)
api.create_tag(repo_id=repo_id, tag=tag, revision=commit_info.oid)
print(f"Model was published under the tag {tag}")
