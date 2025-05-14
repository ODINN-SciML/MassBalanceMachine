import torch
import numpy as np
import random as rd
import os
import gc 
import shutil
import pyproj
from matplotlib.colors import to_hex
from matplotlib import pyplot as plt


def seed_all(seed=None):
    """Sets the random seed everywhere for reproducibility.
    """
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

# difference between two lists
def Diff(li1, li2):
    li_dif = list(set(li1) - set(li2))
    return li_dif

def format_rgi_code(X):
    # Convert X to a string, and pad with leading zeros if its length is less than 5
    Y = str(X).zfill(5)
    # Return the final formatted string
    return f"RGI60-11.{Y}"


def lamberttoWGS84(df, lambert_type="III"):
    """Converts from x & y Lambert III (EPSG:27563) or Lambert II (EPSG:27562) to lat/lon WGS84 (EPSG:4326) coordinate system
    Args:
        df (pd.DataFrame): data in x/y Lambert3
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """

    if lambert_type == "II":
        transformer = pyproj.Transformer.from_crs("EPSG:27562",
                                              "EPSG:4326",
                                              always_xy=True)
    else:
        transformer = pyproj.Transformer.from_crs("EPSG:27563",
                                              "EPSG:4326",
                                              always_xy=True)

    # Transform to Latitude and Longitude (WGS84)
    lon, latitude = transformer.transform(df.x_lambert3, df.y_lambert3)

    df['lat'] = latitude
    df['lon'] = lon
    df.drop(['x_lambert3', 'y_lambert3'], axis=1, inplace=True)
    return df