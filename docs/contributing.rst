Contributing
============

How to contribute?
******************

We welcome contributions! Here's how to get started:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description.

Make sure to conform with the `Contribution Guidelines <https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/CONTRIBUTING.md>`_.


- The MassBalanceMachine project is an open-source community initiative that welcomes new users to fork the repository, add new regions, or modify the existing code and submit a `pull request <https://github.com/ODINN-SciML/MassBalanceMachine/pulls>`_.
- **Currently, uploading data is not allowed unless it is accompanied by a license that explicitly permits open access, allowing it to be shared and used by others.** Pull requests containing data will be rejected. In the future, data sharing will be supported.
- If you have any questions, please contact one of the contributors listed in the `Readme <https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/README.md>`_. You can also create new Git issues via the [issue tracker](https://github.com/ODINN-SciML/MassBalanceMachine/issues) to propose new features, and changes to existing ones, or report bugs.

Developing locally
******************

MassBalanceMachine should run locally without problems.
However, in order to build a live representation of the documentation on your own computer, you need to:

- Create a dedicated `MassBalanceMachine` environment (if not already made) and instead of installing the classical environment as described in :doc:`install`, install the one with the dependencies needed to build the documentation using ``poetry install --with docs``.
- `Install pandoc <https://pandoc.org/installing.html>`_.
- Run ``sphinx-autobuild docs docs/_build/html`` which will print a link that you can open in your browser to visualize the documentation built locally. This will also execute the notebooks and render their outputs in the documentation. Notebooks execution can take a while, in order to build the documentatin and skip these execution steps run ``sphinx-autobuild docs docs/_build/html --define nbsphinx_execute=never``.

Formatting the code
*******************

Before committing your changes, make sure that they comply with the coding style.
You can format the code by running `black <https://github.com/psf/black>`_.
This code formatter can be automatically called upon commit by installing the `pre-commit hook <https://github.com/ODINN-SciML/MassBalanceMachine/blob/main/.pre-commit-config.yaml>`_ defined at the root of the MBM repository.
For this, install `pre-commit <https://pre-commit.com/>`_ in the `MassBalanceMachine` environment by running:
.. code-block:: bash

    pip install pre-commit

The hook can be installed by running:
.. code-block:: bash

    pre-commit install

Then once you have staged your changes, when running the :code:`git commit` command, the hook will trigger and black will ask you to confirm the formatting that have been applied (if changes to the code format were necessary).
