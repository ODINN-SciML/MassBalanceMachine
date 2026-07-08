import os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(mbm_path, ".data")


def region_id_folders(region_id, rgi_version: str):
    if not isinstance(region_id, str):
        region_id = f"{region_id:02d}"
    if rgi_version == "60":
        return f"RGI60-{region_id}"
    else:
        return f"RGI2000-v7.0-G-{region_id}"


def rgi_id_to_folders(rgi_id):
    if "RGI6" in rgi_id:
        # For example "RGI60-11.00695"
        region_folder, id_per_region = rgi_id.split(".")  # "RGI60-11", "00695"
        return region_folder, region_folder + "." + id_per_region[:2], rgi_id
    else:
        assert rgi_id.startswith("RGI2000-v7.0-G-")
        # For example "RGI2000-v7.0-G-11-00147"
        # should return "RGI2000-v7.0-G-11", "RGI2000-v7.0-G-11-00", "RGI2000-v7.0-G-11-00147"
        region_folder, id_per_region = rgi_id.split("-G-")  # "RGI2000-v7.0", "11-00147"
        region_id, id_per_region = id_per_region.split("-")  # "11", "00147"
        region_folder = region_folder + "-G-" + region_id  # "RGI2000-v7.0-G-11"
        return region_folder, region_folder + "-" + id_per_region[:2], rgi_id
