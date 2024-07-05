# MassBalanceMachine

A bridge between mass balance modelling and observations. Global machine learning glacier mass balance modelling that assimilates all glaciological and remote sensing data sources.

- 🏔️ MassBalanceMachine takes meteorological, topographical and/or other features to predict the surface mass balance of glaciers for a region of interest.
- ❄️ MassBalanceMachine uses glaciological (stake) and geodetic mass balance data as targets.
- 📅 MassBalanceMachine can make predictions or fill data gaps on an annual, seasonal (summer and winter), and monthly temporal scale for any spatial resolution.

This project is in **ongoing development**, and new features will be added over the coming months. Please see the contribution guidelines for more information on contributing to this project.

## Requirements

You can run the MassBalanceMachine core scripts and notebooks with the following software installed:

- [Python](https://www.python.org/downloads/) (version 10 or higher is required)
- Conda Environment (either of the following):
  - [Anaconda](https://docs.anaconda.com/anaconda/install/)
  - [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

## Installation (for all users)

To run the Jupyter Notebooks, you'll need to set up a Conda environment. Within this environment, Poetry will handle the installation of all necessary packages and dependencies. Follow these steps to create a new Conda environment named MassBalanceMachine:

```
conda create --name mbm_env --no-default-packages
conda env update --name mbm_env --file environment.yml
```

Activate the MassBalanceMachine environment and install Poetry:


```
conda activate MassBalanceMachine
```

Install all required packages and dependencies needed in the environment via poetry:

```
poetry install
```

All packages and dependencies should now be installed correctly, and you are ready to use the MassBalanceMachine core (```massbalancemachine```). For example, by importing the packing in a Jupyter Notebook by: ```import massbalancemachine as mbm```. Make sure you have selected the right interpreter or kernel before that, in your editor of choice.

**Note:** If you are working on a remote server running JupyterLab or Jupyter Notebook instead of locally, the virtual environment of the notebook will be different from the Conda environment. As an additional step, you need to create a new kernel that includes the Conda environment in Jupyter Notebook. Here’s how you can do it:

```
poetry run ipython kernel install --user --name=massbalancemachine
```

At last, select in the Jupyter Notebook in the top right corner the new installed kernel: massbalancemachine. You should now be ready to go and use ```massbalancemachine``` in your Jupyter Notebooks.

### Known Installation Issues

- Poetry occasionally flags duplicate package folders, but it simplifies dependency and version control in Python projects, ensuring seamless integration of libraries and packages, which can typically be resolved by locating and removing unwanted package versions from your Conda environment folder.

### Additional Installation for Windows Users

**Note:** Topographical and climatological features are retrieved using OGGM in the data processing stage, which for now requires a Unix environment. **However, it is not required to run the model training and evaluation in a remote environment**. Window users can either choose to work with the MassBalanceMachine for the entire project in a Unix environment, or just for the data processing part (this requires two times installing the Conda environment).

If you haven't already, please consult [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install). A list of steps is provided for Windows users to run this code on their local machine in a remote environment:

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:
   1. [Visual Studio](https://code.visualstudio.com/docs/remote/wsl)
   2. [PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   3. [Juypyter Notebook](https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/)
2. Installing Anaconda on Linux:
   1. [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/linux/), or
   2. [Steps to Install Anaconda on Windows Ubuntu Terminal](https://docs.anaconda.com/free/anaconda/install/linux/)
3. Follow the steps as specified in the section: **Installation and Usage**.
4. Access the remote environment in the terminal, select the right kernel or interpreter and run the Jupyter Notebook or Python scripts.

## Usage

After successfully installing the ```massbalancemachine``` package and the Conda environment, you can begin with the example notebooks located in the `notebooks` directory. These notebooks will guide you through how to use MassBalanceMachine, using WGMS data for the examples.

## Project Structure

- The ```massbalancemachine``` package contains the core components of MassBalanceMachine, including scripts, classes, and example Jupyter Notebooks that are essential for new users to start a MassBalanceMachine project. This core package, named massbalancemachine, can be imported into scripts and Jupyter Notebooks as needed.
- ```regions``` contains additional scripts, classes, and Jupyter Notebooks that are tailored for MassBalanceMachine instances that operate in different regions in the world. If the region you are interested in is not on this list, you can, with a pull request, add this to the repository. Please make sure you do not upload any confidential or unpublished data. Regions that are covered so far:
  - [WIP]```Iceland```
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

- The MassBalanceMachine project is an open-source community initiative that welcomes new users to fork the repository, add new regions, or modify the existing code and submit a pull request.

- **Currently, uploading data is not permitted**. Pull requests containing data will be rejected. In the future, data sharing will be supported.

- If you have any questions, please contact one of the contributors listed above. You can also create new Git issues to propose new features or changes to existing ones.