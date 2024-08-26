import os
from os.path import isfile, join
import numpy as np
import random as rd
import torch
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

# ERA5-Land
path_ERA5_raw = '../../../data/ERA5Land/raw/'

# Constants:
SEED = 5

vois_climate_long_name = {
    't2m': 'Temperature',
    'tp': 'Precipitation',
    'slhf': 'Surface latent heat flux',
    'sshf': 'Surface sensible heat flux',
    'ssrd': 'Surface solar radiation downwards',
    'fal': 'Forecast albedo',
    'str': 'Surface net thermal radiation'
}

vois_units = {
    't2m': 'C',
    'tp': 'm w.e.',
    'slhf': 'J m-2',
    'sshf': 'J m-2',
    'ssrd': 'J m-2',
    'fal': '',
    'str': 'J m-2'
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
    hex_codes = [to_hex(rgb[i,:]) for i in range(rgb.shape[0])]

    return hex_codes
