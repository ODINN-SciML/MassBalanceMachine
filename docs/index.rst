:github_url: https://github.com/ODINN-SciML/MassBalanceMachine

Welcome to the MassBalanceMachine documentation!
################################################

**MassBalanceMachine** is global machine learning glacier mass balance model that assimilates all glaciological and remote sensing data sources.

.. note::

    This project is under active development.

Check out the :doc:`install` section for installation information.
The notebooks will guide you through the data preprocessing and your first `MassBalanceMachine` training.

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Getting started

    install
    structure

Tutorials
*********

After installing the `massbalancemachine` package, you can start exploring the tutorials. These notebooks are designed to walk you through using MassBalanceMachine with WGMS data, focusing initially on extracting data from the `Open Global Glacier Model (OGGM) <https://github.com/OGGM/oggm>`_. These data include comprehensive topographical information for nearly all glaciers worldwide.

Specifically, the example notebooks concentrate on glaciers documented in the WGMS database, particularly those in Iceland. They cover various topics, including:

1. **Data pre-processing 🌍**: If users have already their data formatted in the WGMS format, they can directly jump into the :doc:`notebooks/data_processing_wgms` tutorial. Alternatively, if the data are not properly formatted, the :doc:`notebooks/data_preprocessing` notebook shows an example of how to convert the data. In the :doc:`notebooks/data_processing_wgms` workflow, topographical and climate data are fetched and aligned with the stake measurements. Subsequently, the data is aggregated to a monthly resolution, preparing it for use as training data for the model.

.. note::

    If the OGGM cluster is shut down, users will be unable to retrieve topographical features for their region of interest. If you encounter a 403 error in your notebook while trying to retrieve these features, it likely means that the OGGM cluster is down. You can check the status of the cluster on their `Slack channel <https://community.oggm.org/guides/slack-intro.html>`_.

2. **Data Exploration 🔍**: Users can gain deeper insights into their data by visualizing the time series of the available stake measurements, which are related to either the region-wide surface mass balance or the point surface mass balance. See the :doc:`notebooks/data_exploration` tutorial.
3. **Model Training 🚀 & Testing 🎯**: Two tutorials cover the use of two models: XGBoost and neural networks. Refer to the :doc:`notebooks/model_training_xgboost` and :doc:`notebooks/model_training_neuralnetwork` tutorials.

.. toctree::
    :maxdepth: 1
    :caption: Tutorials
    :hidden:

    notebooks/data_preprocessing
    notebooks/data_processing_wgms
    notebooks/data_exploration
    notebooks/model_training_xgboost
    notebooks/model_training_neuralnetwork

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Project information

    support
    goals
    contributing

.. toctree::
    :glob:
    :maxdepth: 2
    :caption: API

    api
