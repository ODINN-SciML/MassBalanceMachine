import torch
import numpy as np
import random as rd
import os
import gc
import shutil
from matplotlib.colors import to_hex
from matplotlib import pyplot as plt
import random
import logging
import massbalancemachine as mbm
import pandas as pd

from sklearn.model_selection import train_test_split


def seed_all(seed=None):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Forbid nondeterministic ops (warn if an op has no deterministic impl)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Setting CUBLAS environment variable (helps in newer versions)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def free_up_cuda():
    """Frees up unused CUDA memory in PyTorch."""
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Free unused cached memory
    torch.cuda.ipc_collect()  # Collect inter-process memory


def get_cmap_hex(cmap, length):
    """
    Function to get a get a list of colours as hex codes

    :param cmap:    name of colourmap
    :type cmap:     str

    :return:        list of hex codes
    :rtype:         list
    """
    # Get cmap
    rgb = plt.get_cmap(cmap)(np.linspace(0, 1, length))

    # Convert to hex
    hex_codes = [to_hex(rgb[i, :]) for i in range(rgb.shape[0])]

    return hex_codes


def emptyfolder(path):
    """Removes all files and subdirectories in the given folder."""
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove folder and all contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists


# difference between two lists
def Diff(li1, li2):
    li_dif = list(set(li1) - set(li2))
    return li_dif


def format_rgi_code(X):
    # Convert X to a string, and pad with leading zeros if its length is less than 5
    Y = str(X).zfill(5)
    # Return the final formatted string
    return f"RGI60-11.{Y}"
