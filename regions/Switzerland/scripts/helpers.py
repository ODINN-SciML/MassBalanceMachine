import os
from os.path import isfile, join
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
path_glacier_grid = '../../../data/GLAMOS/glacier-wide/grid/'

# Potential incoming clear sky solar radiation from GLAMOS:
path_direct = '../../../data/GLAMOS/direct/raw/'
path_direct_save = '../../../data/GLAMOS/direct/csv/'

path_rgi = '../../../data/GLAMOS/CH_glacier_ids_long.csv'

# ERA5-Land
path_ERA5_raw = '../../../data/ERA5Land/raw/'

vois_climate_long_name = {
    't2m': 'Temperature',
    'tp': 'Precipitation',
    'slhf': 'Surface latent heat flux',
    'sshf': 'Surface sensible heat flux',
    'ssrd': 'Surface solar radiation downwards',
    'fal': 'Forecast albedo',
    'str': 'Surface net thermal radiation'
}

vois_long_name = {
    't2m': 'Temperature',
    'tp': 'Precipitation',
    'slhf': 'Surface latent heat flux',
    'sshf': 'Surface sensible heat flux',
    'ssrd': 'Surface solar radiation downwards',
    'fal': 'Forecast albedo',
    'str': 'Surface net thermal radiation',
    'pcsr': 'Pot. incoming clear sky solar rad.'
}

vois_units = {
    't2m': 'C',
    'tp': 'm w.e.',
    'slhf': 'J m-2',
    'sshf': 'J m-2',
    'ssrd': 'J m-2',
    'fal': '',
    'str': 'J m-2',
    'pcsr': 'J m-2'
}

loss_units = {
    'RMSE': '[m w.e.]',
    'MSE': '[]'
}


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


def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


# empties a folder
def emptyfolder(path):
    if os.path.exists(path):
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            os.remove(path + f)
    else:
        createPath(path)


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
