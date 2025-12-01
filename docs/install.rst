Installation instructions
=========================

.. tip::
    If you want to contribute to MassBalanceMachine, don't forget to read the :doc:`contributing` section. This is especially useful to know the coding rules and code formatting tools that should be applied to your commits.

Requirements
************

You can run the MassBalanceMachine core scripts and notebooks with the following software installed:
Conda Environment (either of the following):

* `Anaconda <https://docs.anaconda.com/anaconda/install/>`_
* `Miniconda <https://docs.anaconda.com/miniconda/miniconda-install/>`_
* `Micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_ (recommended)

Installation
************

To run the MassBalanceMachine, you'll need to set up a Conda environment.
Within this environment, Poetry will handle the installation of all necessary packages and dependencies.
Follow these steps to create a new Conda environment named :code:`MassBalanceMachine`:

Environment setup
-----------------

**Note:** if you are using micromamba, replace :code:`conda` by :code:`micromamba` in the commands below.

CPU only
^^^^^^^^

.. code-block:: bash

    conda env create -f environment_cpu.yml

GPU
^^^

Linux
"""""

.. code-block:: bash

    conda env create -f environment_gpu.yml

Windows
"""""""

Windows users need to manually install PyTorch with GPU support.
You can refer to the `installation instructions of PyTorch <https://pytorch.org/get-started/locally/>` and run commands similar to:

.. code-block:: bash

    conda env create -f environment_gpu.yml
    # Adapt command below to your needs
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


MacOS
"""""

After having created and activated the :code:`MassBalanceMachine` environment, you need to install cupy separately:

.. code-block:: bash

    conda env create -f environment_gpu.yml
    conda activate MassBalanceMachine
    conda install -c conda-forge cupy

Install dependencies
--------------------

Activate the MassBalanceMachine environment:

.. code-block:: bash

    conda activate MassBalanceMachine

Install all required packages and dependencies needed in the environment via poetry:

.. code-block:: bash

    poetry install

All packages and dependencies should now be installed correctly, and you are ready to use the MassBalanceMachine core. For example, by importing the packing in a Jupyter Notebook by: :code:`import massbalancemachine as mbm`. Make sure you have selected the right interpreter or kernel before that in your editor of choice.

.. tip::
    If you are working on a remote server running JupyterLab or Jupyter Notebook (e.g. Binder) instead of locally, the virtual environment of the notebook will be different from the Conda environment. As an additional step, you need to create a new kernel that includes the Conda environment in Jupyter Notebook. Here’s how you can do it:

    .. code-block:: bash

        poetry run ipython kernel install --user --name=mbm_env


Finally, ensure that your Jupyter kernel is set to use the MassBalanceMachine Conda environment. You can select the kernel from the top right corner of the notebook or through the Launcher (you might need to refresh for the changes to take effect). With this setup, you should be ready to use the :code:`massbalancemachine` package in your Jupyter Notebooks.

Known Installation Issues
-------------------------

- Poetry sometimes identifies duplicate package folders, but it streamlines dependency and version management in Python projects, ensuring smooth library and package integration. Any duplicate packages can usually be resolved by locating and removing the unnecessary versions from your Conda environment folder.

Additional Installation for Windows Users
-----------------------------------------

.. note::
    Topographical features are retrieved using OGGM in the data processing stage, which, for now, requires a Unix environment. **However, the model training and evaluation are not required to run in a remote environment**. Window users can either choose to work with the :code:`MassBalanceMachine` for the entire project in a Unix environment or just for the data processing part (this requires two times installing the Conda environment)

If you haven't already, please consult `How to install Linux on Windows with WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`. A list of steps is provided for Windows users to run this code on their local machine in a remote environment:

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:

   1. `Visual Studio <https://code.visualstudio.com/docs/remote/wsl>`_
   2. `PyCharm <https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter>`_
   3. `Juypyter Notebook <https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/>`_

2. Installing Anaconda on Linux:

   1. `Anaconda Docs <https://docs.anaconda.com/free/anaconda/install/linux/>`_, or
   2. `Steps to Install Anaconda on Windows Ubuntu Terminal <https://docs.anaconda.com/free/anaconda/install/linux/>`_

3. Follow the steps as specified in the section: :doc:`install`.
4. Access the remote environment in the terminal, select the right kernel or interpreter and run the Jupyter Notebook or Python scripts.
