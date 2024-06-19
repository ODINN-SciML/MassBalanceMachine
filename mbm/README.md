# MassBalanceMachine Core

In the ```mbm``` folder, the core code of MassBalanceMachine can be found. The code is divided into partitions; each process step has dedicated scripts, which can be found in folders with their respective names. Each folder has a ```README``` file explaining each script, the order in which the scripts should be executed, and the script's output. Depending on the dataset and tasks, either all or only some scripts are useful to new users. Inspiration can be taken from other regions, already available in this repository in the folder ```regions```, for more extended data (pre)processing and model training and testing. The core of the MassBalanceMachine, currently consists of: 

- ```scripts-data-processing```, data processing and visualisation. 
- ```scripts-model-training```, model training, testing, and visualisation of results.

**Note**: please be aware that all scripts should be run manually, each with its own input and output file. The location of these files should be adjusted to your environment, desired location, and file names.

## Requirements

You can run the MassBalanceMachine core scripts with the following software installed:

- [Python](https://www.python.org/downloads/)
- Conda
  - [Anaconda](https://docs.anaconda.com/anaconda/install/)
  - [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

## Installation

You can run the scripts that are part of the MassBalanceMachine in a dedicated Conda environment. To do this, install the environment and all the necessary packages using the `environment.yml` file with the following command:

```
conda env create -f environment.yml
```

Ensure you are using the Conda terminal and are in the same directory of the ```environment.yml``` file or put in the correct path to the file in the command above. Then activate the environment with:

```
conda activate MassBalanceMachine
```

All packages should now be installed correctly, and you are ready to use the MassBalanceMachine core (mbm) and the OGGM environment that is part of the ```scripts-data-processing``` directory. In the ```MassBalanceMachine``` environment, you can execute the code as follows: 

- Jupyter notebooksare started with the following command: ```jupyter notebook```. Click the link that navigates to localhost in order to connect to Jupyter lab. 
- Python scripts can be run with the following command: ```python script.py```.

## Usage

Please consult the directories listed above for a detailed outline of how the scripts should be used. 