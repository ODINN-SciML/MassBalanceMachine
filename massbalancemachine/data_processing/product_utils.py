import os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(mbm_path, ".data")


def rgi_id_to_folders(rgi_id):
    region_folder, id_per_region = rgi_id.split(".")
    return region_folder, region_folder + "." + id_per_region[:2], rgi_id
