# OGGM

This folder is purely dedicated to OGGM, as running the code on a Windows machine requires a remote environment. Future plans are to have this code with the other scripts in the data processing pipeline and automate this process with a Docker Image (and possibly with Kubernetes).

## Installation

If you haven't already, please consult [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install). A list of steps is provided for Windows users to run this code on their local machine in a remote environment:

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:
   1. [Visual Studio](https://code.visualstudio.com/docs/remote/wsl)
   2. [PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   3. [Juypyter Notebook](https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/)
2. Installing Anaconda on Linux:
   1. [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/linux/), or
   2. [Steps to Install Anaconda on Windows Ubuntu Terminal](https://docs.anaconda.com/free/anaconda/install/linux/)
3. [Create a new Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from the ```oggm_env.yml```
4. Access the remote environment, and ```conda activate oggm_env``` via the terminal, and start ```jupyter notebook``` if your OGGM code is in a Jupyter Notebook. Else, if your code is a Python script, run your code with: ```python scripts/get_oggm_data.py```.

When you run the code for the first time, the RGI directories for your RGI IDs are downloaded to the ```data``` folder this can take some while.

## File Structure

.
├── data                    # Data files
│   └── OGGM                # OGGM, RGIV6 per Glacier
├── misc                    # Miscellaneous files
└── scripts                 # Scripts

- ```scripts``` contains ```get_oggm_data.py``` that takes as input: ```Iceland_Stake_Data_Reprojected.csv``` and outputs: ```Iceland_Stake_Data_T_Attributes.csv```. For each RGI ID,
the 'slope', 'slope factor', 'topo', 'aspect', and the 'distance from the border' are retrieved via OGGM for each individual stake. It also retrieves the total area for each glacier, needed for the hypsometry plot, and outputs this in: ```areas_rgiids.csv```.  

- ```misc``` contains ```oggm_env.yml``` that contains all the required libraries to run the OGGM script. 