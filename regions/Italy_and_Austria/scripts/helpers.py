import torch
import numpy as np
import random as rd
import os
import gc
import shutil
from matplotlib.colors import to_hex
from matplotlib import pyplot as plt


def seed_all(seed=None):
    """Sets the random seed everywhere for reproducibility."""
    if seed is None:
        seed = 10  # Default seed value

    # Python built-in random
    rd.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # Ensuring deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setting CUBLAS environment variable (helps in newer versions)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
