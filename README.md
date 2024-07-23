# MassBalanceMachine

A bridge between mass balance modelling and observations. Global machine learning glacier mass balance modelling that assimilates all glaciological and remote sensing data sources.

- 🏔️ MassBalanceMachine takes meteorological, topographical and/or other features to predict the surface mass balance of glaciers for a region of interest.
- ❄️ MassBalanceMachine uses glaciological (stake) and geodetic mass balance data as targets.
- 📅 MassBalanceMachine can make predictions or fill data gaps on an annual, seasonal (summer and winter), and monthly temporal scale for any spatial resolution.

This project is in **ongoing development**, and new features will be added over the coming months. Please see the contribution guidelines for more information on contributing to this project.

## Requirements

You can run the MassBalanceMachine core scripts and notebooks with the following software installed:

- Conda Environment (either of the following):
  - [Anaconda](https://docs.anaconda.com/anaconda/install/)
  - [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

## Installation (for all users)

To run the Jupyter Notebooks, you'll need to set up a Conda environment. Within this environment, Poetry will handle the installation of all necessary packages and dependencies. Follow these steps to create a new Conda environment named MassBalanceMachine:

```
conda env create -f environment.yml
```

Activate the MassBalanceMachine environment:

```
conda activate MassBalanceMachine # for linux and unix users alternatively: source activate MassBalanceMachine
```

Install all required packages and dependencies needed in the environment via poetry:

```
poetry install
```

All packages and dependencies should now be installed correctly, and you are ready to use the MassBalanceMachine core (```massbalancemachine```). For example, by importing the packing in a Jupyter Notebook by: ```import massbalancemachine as mbm```. Make sure you have selected the right interpreter or kernel before that, in your editor of choice.

**Note:** If you are working on a remote server running JupyterLab or Jupyter Notebook (e.g. Binder) instead of locally, the virtual environment of the notebook will be different from the Conda environment. As an additional step, you need to create a new kernel that includes the Conda environment in Jupyter Notebook. Here’s how you can do it:

```
poetry run ipython kernel install --user --name=mbm_env
```

At last, make sure your Jupyter kernel, select in the top right corner of the notebook, or via the Launcher (sometimes this requires a refresh),has been configured to work with the 'mbm_env' Conda environment. You should now be ready to go and use ```massbalancemachine``` in your Jupyter Notebooks.

### Known Installation Issues

- Poetry occasionally flags duplicate package folders, but it simplifies dependency and version control in Python projects, ensuring seamless integration of libraries and packages, which can typically be resolved by locating and removing unwanted package versions from your Conda environment folder.

### Additional Installation for Windows Users

**Note:** Topographical features are retrieved using OGGM in the data processing stage, which for now requires a Unix environment. **However, it is not required to run the model training and evaluation in a remote environment**. Window users can either choose to work with the MassBalanceMachine for the entire project in a Unix environment, or just for the data processing part (this requires two times installing the Conda environment).

If you haven't already, please consult [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install). A list of steps is provided for Windows users to run this code on their local machine in a remote environment:

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:
   1. [Visual Studio](https://code.visualstudio.com/docs/remote/wsl)
   2. [PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   3. [Juypyter Notebook](https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/)
2. Installing Anaconda on Linux:
   1. [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/linux/), or
   2. [Steps to Install Anaconda on Windows Ubuntu Terminal](https://docs.anaconda.com/free/anaconda/install/linux/)
3. Follow the steps as specified in the section: **Installation**.
4. Access the remote environment in the terminal, select the right kernel or interpreter and run the Jupyter Notebook or Python scripts.

## Usage

After installing the `massbalancemachine` package and setting up the Conda environment successfully, you can start exploring the example notebooks found in the `notebooks` directory. These notebooks are designed to walk you through using MassBalanceMachine with WGMS data, focusing initially on extracting data from the [Open Global Glacier Model (OGGM)](https://github.com/OGGM/oggm). This data includes comprehensive topographical information for nearly all glaciers worldwide.

Specifically, the example notebooks concentrate on glaciers documented in the WGMS database, particularly those in Iceland. They cover various topics, including:

1. **Data Pre-processing 🌍**: Users have two options for preparing their data. They can choose to follow a notebook that converts their data into the WGMS format (available [here](https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/notebooks/data_preprocessing.ipynb)), or they can start with their data already formatted in the WGMS standard (found [here](https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/notebooks/data_processing_wgms.ipynb)). In both workflows, topographical and climate data are fetched and aligned with the stake measurements. Subsequently, the data is aggregated to a monthly resolution, preparing it for use as training data for the model.
   - **Note:** If the OGGM cluster is shut down, users will be unable to retrieve topographical features for their region of interest. If you encounter a 403 error in your notebook while trying to retrieve these features, it likely means that the OGGM cluster is down. You can check the status of the cluster on their [Slack channel](https://oggm.org/2022/10/11/Welcome-to-the-OGGM-Slack/).
2. **Data Exploration 🔍**: Users can gain deeper insights into their data by visualizing time series of the available stake measurements, which are related to either the region-wide surface mass balance or the point surface mass balance. The example is available [here](https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/notebooks/date_exploration.ipynb).
3. **Model Training 🚀**: Users can choose from two models. One option is the XGBoost model, with an example available in this [notebook](https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/notebooks/xgboost_model_training.ipynb). The other option is a neural network, which will be released in the future. Both models are customized to handle the monthly resolution of the data. In the notebooks, the models will be trained using the data obtained earlier.
4. **Model Testing 🎯** [WIP]

## Project Structure

- The ```massbalancemachine``` package contains the core components of MassBalanceMachine, including scripts, classes, and example Jupyter Notebooks that are essential for new users to start a MassBalanceMachine project. This core package, named massbalancemachine, can be imported into scripts and Jupyter Notebooks as needed.
- ```regions``` contains additional scripts, classes, and Jupyter Notebooks that are tailored for MassBalanceMachine instances that operate in different regions in the world. If the region you are interested in is not on this list, you can, with a pull request, add this to the repository. Please make sure you do not upload any confidential or unpublished data. Regions that are covered so far:
  - [WIP] ```Iceland```
  - [COMING SOON] ```Switzerland```
  - [COMING SOON] ``Norway``
  - [ADD YOUR OWN REGION]. PRs welcome! Message us if you have questions 🙂

## Project Roadmap

The following features are on the roadmap to be implemented in the coming months:

- 🛰️ MassBalanceMachine uses geodetic mass balance data as an extra target variable on top of glaciological data. This will help calibrate the bias/trend in long simulations where the cumulative mass balance matters.
- 🔄 MassBalanceMachine can do transfer learning for new regions, reducing the training time and making more accurate predictions.
- 📊 MassBalanceMachine can incorporate physical constraints, in order to merge physical knowledge with data-driven discovery.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JulianBiesheuvel"><img src="https://avatars.githubusercontent.com/u/16390017?v=4?s=100" width="100px;" alt="Julian"/><br /><sub><b>Julian</b></sub></a><br /><a href="#code-JulianBiesheuvel" title="Code">💻</a> <a href="#doc-JulianBiesheuvel" title="Documentation">📖</a> <a href="#maintenance-JulianBiesheuvel" title="Maintenance">🚧</a> <a href="#data-JulianBiesheuvel" title="Data">🔣</a> <a href="#research-JulianBiesheuvel" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/khsjursen"><img src="https://avatars.githubusercontent.com/u/69296367?v=4?s=100" width="100px;" alt="khsjursen"/><br /><sub><b>khsjursen</b></sub></a><br /><a href="#research-khsjursen" title="Research">🔬</a> <a href="#code-khsjursen" title="Code">💻</a> <a href="#ideas-khsjursen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-khsjursen" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jordibolibar.wordpress.com"><img src="https://avatars.githubusercontent.com/u/2025815?v=4?s=100" width="100px;" alt="Jordi Bolibar"/><br /><sub><b>Jordi Bolibar</b></sub></a><br /><a href="#research-JordiBolibar" title="Research">🔬</a> <a href="#projectManagement-JordiBolibar" title="Project Management">📆</a> <a href="#financial-JordiBolibar" title="Financial">💵</a> <a href="#ideas-JordiBolibar" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-JordiBolibar" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marvande"><img src="https://avatars.githubusercontent.com/u/22401294?v=4?s=100" width="100px;" alt="Marijn  "/><br /><sub><b>Marijn  </b></sub></a><br /><a href="#ideas-marvande" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-marvande" title="Data">🔣</a> <a href="#research-marvande" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zekollari"><img src="https://avatars.githubusercontent.com/u/19975538?v=4?s=100" width="100px;" alt="zekollari"/><br /><sub><b>zekollari</b></sub></a><br /><a href="#research-zekollari" title="Research">🔬</a> <a href="#financial-zekollari" title="Financial">💵</a> <a href="#ideas-zekollari" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-zekollari" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Contribution Guidelines

- The MassBalanceMachine project is an open-source community initiative that welcomes new users to fork the repository, add new regions, or modify the existing code and submit a [pull request](https://github.com/ODINN-SciML/MassBalanceMachine/pulls).
- **Currently, uploading data is not allowed unless it is accompanied by a license that explicitly permits open access, allowing it to be shared and used by others.** Pull requests containing data will be rejected. In the future, data sharing will be supported.
- If you have any questions, please contact one of the contributors listed above. You can also create new Git issues via the [issue tracker](https://github.com/ODINN-SciML/MassBalanceMachine/issues) to propose new features or changes to existing ones.
