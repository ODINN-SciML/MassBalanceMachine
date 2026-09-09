import os
import re

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


def is_rgi_id(glacier_id: str) -> bool:
    """Whether `glacier_id` follows one of the two RGI naming schemes handled by
    `rgi_id_to_folders` (RGI v6 like "RGI60-11.00695", or RGI v7 like
    "RGI2000-v7.0-G-11-00147")."""
    return "RGI6" in glacier_id or glacier_id.startswith("RGI2000-v7.0-G-")


def glacier_id_to_folders(glacier_id: str):
    """Folder tree in which the gridded products of one glacier are stored.

    RGI ids keep the existing three-level tree produced by `rgi_id_to_folders`.
    Any other identifier - the native id of a custom outline dataset, such as the
    SGI id "B36-26" of the Swiss inventory - gets a two-level tree whose first
    level is the part of the id before the first separator, so that a source with
    thousands of glaciers does not end up with one flat directory. The id is
    sanitized because it becomes a path component.
    """
    if is_rgi_id(glacier_id):
        return rgi_id_to_folders(glacier_id)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", glacier_id)
    assert sanitized not in (
        "",
        ".",
        "..",
    ), f"Glacier id {glacier_id!r} does not yield a usable folder name."
    group = re.split(r"[-_.]", sanitized, maxsplit=1)[0] or sanitized
    return group, sanitized
