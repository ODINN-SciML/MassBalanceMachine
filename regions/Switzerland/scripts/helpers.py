import os
from os.path import isfile, join, isdir
import numpy as np
import random as rd
import torch
import gc

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

# Paths
path_PMB_GLAMOS_raw = '../../../data/GLAMOS/point/raw/'
path_PMB_GLAMOS_w_raw = path_PMB_GLAMOS_raw + 'winter/'
path_PMB_GLAMOS_a_raw = path_PMB_GLAMOS_raw + 'annual/'

path_PMB_GLAMOS_csv = '../../../data/GLAMOS/point/csv/'
path_PMB_GLAMOS_csv_w = path_PMB_GLAMOS_csv + 'winter/'
path_PMB_GLAMOS_csv_w_clean = path_PMB_GLAMOS_csv + 'winter_clean/'
path_PMB_GLAMOS_csv_a = path_PMB_GLAMOS_csv + 'annual/'
path_SMB_GLAMOS_raw = '../../../data/GLAMOS/glacier-wide/raw/'
path_SMB_GLAMOS_csv = '../../../data/GLAMOS/glacier-wide/csv/'
path_glacier_grid = '../../../data/GLAMOS/gridded_products/RGI_grid/'
path_glacier_grid_sgi = '../../../data/GLAMOS/gridded_products/SGI_grid/'
path_SGI_topo = '../../../data/GLAMOS/topo/SGI2020/'
path_OGGM = '../../../data/OGGM/'
# Potential incoming clear sky solar radiation from GLAMOS:
path_direct = '../../../data/GLAMOS/direct/raw/'
path_direct_save = '../../../data/GLAMOS/direct/csv/'

path_rgi = '../../../data/GLAMOS/CH_glacier_ids_long.csv'
path_glogem = '../../../data/GloGEM'
path_geodetic_MB = '../../../data/GLAMOS/geodetic/'

# ERA5-Land
path_ERA5_raw = '../../../data/ERA5Land/raw/'

# Sentinel-2
path_S2 = '../../../data/Sentinel/'

vois_climate_long_name = {
    't2m': 'Temperature',
    'tp': 'Precipitation',
    't2m_corr': 'Temperature corr.',
    'tp_corr': 'Precipitation corr.',
    'slhf': 'Surf. latent heat flux',
    'sshf': 'Surf. sensible heat flux',
    'ssrd': 'Surf. solar rad. down.',
    'fal': 'Albedo',
    'str': 'Surf. net thermal rad.',
    'pcsr': 'Pot. in. clear sky solar rad.',
    'u10': '10m E wind',
    'v10': '10m N wind',
}

vois_units = {
    't2m': 'C',
    'tp': 'm w.e.',
    'slhf': 'J m-2',
    'sshf': 'J m-2',
    'ssrd': 'J m-2',
    'fal': '',
    'str': 'J m-2',
    'pcsr': 'J m-2',
    'u10': 'm s-1',
    'v10': 'm s-1',
}

month_abbr_hydr_full = {
    'sep': 1,
    'oct': 2,
    'nov': 3,
    'dec': 4,
    'jan': 5,
    'feb': 6,
    'mar': 7,
    'apr': 8,
    'may': 9,
    'jun': 10,
    'jul': 11,
    'aug': 12,
    'sep_': 13,
}

loss_units = {'RMSE': '[m w.e.]', 'MSE': '[]'}


# sets the same random seed everywhere so that it is reproducible
def seed_all(seed):
    if not seed:
        seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def emptyfolder(path):
    if os.path.exists(path):
        # Loop through all items in the directory
        for item in os.listdir(path):
            item_path = join(path, item)
            if isfile(item_path):
                os.remove(item_path)  # Remove file
            elif isdir(item_path):
                emptyfolder(item_path)  # Recursively empty the folder
                os.rmdir(item_path)  # Remove the now-empty folder
    else:
        createPath(path)


def createPath(path):
    os.makedirs(path, exist_ok=True)


# difference between two lists
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


# Updates a dictionnary at key with value
def updateDic(dic, key, value):
    if key not in dic.keys():
        dic[key] = [value]
    else:
        dic[key].append(value)

    return dic


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


def free_up_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def makeCombNum(combin_clim, combi_topo):
    topo_primes = {
        'aspect': 2,
        'slope': 3,
        'dis_from_border': 5,
        'topo': 7,
    }
    climate_primes = {
        't2m': 11,
        'tp': 13,
        'slhf': 17,
        'sshf': 19,
        'ssrd': 23,
        'fal': 29,
        'str': 31,
    }
    mult = 1
    for var in combi_topo:
        mult *= topo_primes[var]
    for var in combin_clim:
        mult *= climate_primes[var]
    return mult


# Get all  combinations of length min 3:
def powerset(original_list, min_length=3):
    # The number of subsets is 2^n
    num_subsets = 2**len(original_list)

    # Create an empty list to hold all the subsets
    subsets = []

    # Iterate over all possible subsets
    for subset_index in range(num_subsets):
        # Create an empty list to hold the current subset
        subset = []
        # Iterate over all elements in the original list
        for index in range(len(original_list)):
            # Check if index bit is set in subset_index
            if (subset_index & (1 << index)) != 0:
                # If the bit is set, add the element at this index to the current subset
                subset.append(original_list[index])
        # Add the current subset to the list of all subsets
        if len(subset) >= min_length:
            subsets.append(subset)
    return subsets


def format_rgi_code(X):
    # Convert X to a string, and pad with leading zeros if its length is less than 5
    Y = str(X).zfill(5)
    # Return the final formatted string
    return f"RGI60-11.{Y}"

def reformat_SGI_id(input_str):
    # Split the string by "/"
    part1, part2 = input_str.split("/")

    # Convert part1 to lowercase for the letter and retain the number
    part1 = part1[:-1] + part1[-1].lower()

    # Combine part1 and part2 with a hyphen in between
    return f"{part1}-{part2}"