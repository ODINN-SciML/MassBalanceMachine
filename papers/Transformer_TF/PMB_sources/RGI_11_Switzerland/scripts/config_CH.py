# <------------------ PATHS (relative to cfg.dataPath) ------------------>
# GLAMOS point data
path_PMB_GLAMOS_raw = "GLAMOS/point/point_raw/"
path_PMB_GLAMOS_w_raw = path_PMB_GLAMOS_raw + "winter/"
path_PMB_GLAMOS_a_raw = path_PMB_GLAMOS_raw + "annual/"

path_PMB_GLAMOS_csv = (
    "GLAMOS/point/csv/"  # Converted .dat files and intermediate outputs
)
path_PMB_GLAMOS_csv_w = path_PMB_GLAMOS_csv + "winter/"
path_PMB_GLAMOS_csv_w_clean = path_PMB_GLAMOS_csv + "winter_clean/"
path_PMB_GLAMOS_csv_a = path_PMB_GLAMOS_csv + "annual/"

# Final stake dataset (CH_wgms_dataset_all.csv), read by
# papers/Transformer_TF/scripts/dataset/data_loader.py
path_PMB_WGMS_csv = "WGMS/Switzerland/csv/"

# Topo data
path_SGI_topo = "GLAMOS/topo/SGI2020/"  # DEMs & topo from SGI
path_GLAMOS_topo = "GLAMOS/topo/GLAMOS_DEM/"  # yearly DEMs from GLAMOS
path_glacier_ids = "GLAMOS/CH_glacier_ids_long.csv"  # glacier name <-> SGI/RGI ids

path_rgi_outlines = "RGI_v6/RGI_11_CentralEurope/11_rgi60_CentralEurope.shp"
