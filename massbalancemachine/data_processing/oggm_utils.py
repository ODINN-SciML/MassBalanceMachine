import os
from oggm import workflow, tasks
from oggm import cfg as oggmCfg

import config


def _initialize_oggm_config(custom_working_dir):
    """Initialize OGGM configuration."""
    oggmCfg.initialize(logging_level="WARNING")
    oggmCfg.PARAMS["border"] = 10
    oggmCfg.PARAMS["use_multiprocessing"] = True
    oggmCfg.PARAMS["continue_on_error"] = True
    if len(custom_working_dir) == 0:
        current_path = os.getcwd()
        oggmCfg.PATHS["working_dir"] = os.path.join(current_path, "OGGM")
    else:
        oggmCfg.PATHS["working_dir"] = custom_working_dir


def _initialize_glacier_directories(rgi_ids_list: list, cfg: config.Config) -> list:
    """Initialize glacier directories."""
    base_url = cfg.base_url_w5e5 if cfg.prepro_level >= 3 else cfg.base_url_l2
    glacier_directories = workflow.init_glacier_directories(
        rgi_ids_list,
        reset=False,
        from_prepro_level=cfg.prepro_level,
        prepro_base_url=base_url,
        prepro_border=10,
    )

    workflow.execute_entity_task(
        tasks.gridded_attributes, glacier_directories, print_log=False
    )
    return glacier_directories


def _glacier_name(rgi_ids_list: list, cfg: config.Config, custom_working_dir=""):

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)
    glacier_directories = _initialize_glacier_directories(rgi_ids_list, cfg)
    return {gdir.rgi_id: gdir.name for gdir in glacier_directories}
