import os
import logging

from regions.Switzerland.scripts.glamos_preprocess import *

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    log.info('Processing SMB GLAMOS data')
    process_SMB_GLAMOS() 
    
    log.info('Processing GLAMOS pcsr data')
    process_pcsr()   