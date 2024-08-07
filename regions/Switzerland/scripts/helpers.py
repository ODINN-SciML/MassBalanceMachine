import os
from os.path import isfile, join

# Paths
path_PMB_GLAMOS_raw = '../../../data/GLAMOS/point/raw/'
path_PMB_GLAMOS_w_raw = path_PMB_GLAMOS_raw + 'winter/'
path_PMB_GLAMOS_a_raw = path_PMB_GLAMOS_raw + 'annual/'

path_PMB_GLAMOS_csv = '../../../data/GLAMOS/point/csv/'
path_PMB_GLAMOS_csv_w = path_PMB_GLAMOS_csv + 'winter/'
path_PMB_GLAMOS_csv_w_clean = path_PMB_GLAMOS_csv + 'winter_clean/'
path_PMB_GLAMOS_csv_a = path_PMB_GLAMOS_csv + 'annual/'


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