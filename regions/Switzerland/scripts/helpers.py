import os
from os.path import isfile, join

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