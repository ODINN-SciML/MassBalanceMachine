def rgi_id_to_folders(rgi_id):
    region_folder, id_per_region = rgi_id.split(".")
    return region_folder, region_folder + "." + id_per_region[:2], rgi_id
