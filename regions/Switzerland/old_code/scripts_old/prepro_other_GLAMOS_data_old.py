import os
import logging

import massbalancemachine as mbm
from regions.Switzerland.scripts.glamos_preprocess import *

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

cfg = mbm.SwitzerlandConfig()


def main():
    log.info("Processing SMB GLAMOS data")
    process_SMB_GLAMOS(cfg)

    log.info("Processing GLAMOS pcsr data")
    process_pcsr(cfg)
