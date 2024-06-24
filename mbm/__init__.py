import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

__all__ = ['data_processing']

from .data_processing import get_oggm_data
