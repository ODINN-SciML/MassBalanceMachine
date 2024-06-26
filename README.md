# MassBalanceMachine

Global machine learning glacier mass balance modelling that assimilates all glaciological and remote sensing data sources.

- 🏔️ MassBalanceMachine takes meteorological, topographical and/or other features to predict the surface mass balance of glaciers for a region of interest.
- ❄️ MassBalanceMachine uses glaciological (stake) and geodetic mass balance data as targets.
- 📅 MassBalanceMachine can make predictions on an annual, seasonal (summer and winter), and monthly temporal scale for any spatial resolution.

This project is in ongoing development, and new features will be added over the coming months. Please see the contribution guidelines for more information on contributing to this project.

## Requirements

You can run the MassBalanceMachine core scripts and notebooks with the following software installed:

- [Python](https://www.python.org/downloads/)
- Conda
  - [Anaconda](https://docs.anaconda.com/anaconda/install/)
  - [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

## Installation and Usage

To run the scripts and the notebooks, install the environment and all the necessary packages using the `environment.yml` file with the following command:

conda env create -f environment.yml
Ensure you are using the Conda terminal and are in the same directory of the ```environment.yml``` file or put in the correct path to the file in the command above. Then activate the environment with:

```
conda activate MassBalanceMachine
```
All packages should now be installed correctly, and you are ready to use the MassBalanceMachine core (mbm).

### Installation for Windows Users


**Note:** Topographical features are retrieved using OGGM, which for now requires a Linux environment. For Windows users it is not required to run the model training and evaluation in the remote environment. 

If you haven't already, please consult [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install). A list of steps is provided for Windows users to run this code on their local machine in a remote environment:

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:
   1. [Visual Studio](https://code.visualstudio.com/docs/remote/wsl)
   2. [PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   3. [Juypyter Notebook](https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/)
2. Installing Anaconda on Linux:
   1. [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/linux/), or
   2. [Steps to Install Anaconda on Windows Ubuntu Terminal](https://docs.anaconda.com/free/anaconda/install/linux/)
3. [Create a new Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) from the ```environment.yml``` file that is located in the ```mbm``` directory.
4. Access the remote environment, ```conda activate MassBalanceMachine``` via the terminal and start a Jupyter Notebook.

## Project Structure

- ```mbm``` contains the most important scripts, classes, and notebooks for the MassBalanceMachine project. New users can use these files to get started and add more or change existing files to meet their needs.
- ```regions``` contains MassBalanceMachine instances for different regions in the world. If the region you are interested in is not on this list, you can, with a pull request, add this to the repository. Please make sure you do not upload any confidential or unpublished data. Regions that are covered:
  - ```Iceland```
  - [COMING SOON] ```Switzerland```
  - [COMING SOON] ``Norway``

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/khsjursen"><img src="https://avatars.githubusercontent.com/u/69296367?v=4?s=100" width="100px;" alt="khsjursen"/><br /><sub><b>khsjursen</b></sub></a><br /><a href="#research-khsjursen" title="Research">🔬</a> <a href="#code-khsjursen" title="Code">💻</a> <a href="#ideas-khsjursen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-khsjursen" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JulianBiesheuvel"><img src="https://avatars.githubusercontent.com/u/16390017?v=4?s=100" width="100px;" alt="Julian"/><br /><sub><b>Julian</b></sub></a><br /><a href="#code-JulianBiesheuvel" title="Code">💻</a> <a href="#doc-JulianBiesheuvel" title="Documentation">📖</a> <a href="#maintenance-JulianBiesheuvel" title="Maintenance">🚧</a> <a href="#data-JulianBiesheuvel" title="Data">🔣</a> <a href="#research-JulianBiesheuvel" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jordibolibar.wordpress.com"><img src="https://avatars.githubusercontent.com/u/2025815?v=4?s=100" width="100px;" alt="Jordi Bolibar"/><br /><sub><b>Jordi Bolibar</b></sub></a><br /><a href="#research-JordiBolibar" title="Research">🔬</a> <a href="#projectManagement-JordiBolibar" title="Project Management">📆</a> <a href="#financial-JordiBolibar" title="Financial">💵</a> <a href="#ideas-JordiBolibar" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-JordiBolibar" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Contribution Guidelines

The MassBalanceMachine project is an open-source community project, welcoming new users to fork the repository, start adding new regions, or make changes to the existing code and make a pull request. We encourage you to contact one of the contributors listed above if you have any questions. You can also create new Git issues for new features or changes to existing features.
