# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))


import io
import re
import shutil
import sphinx_rtd_theme  # noqa

### Code below is a copy paste from gpytorch ###
# Cf https://github.com/cornellius-gp/gpytorch/blob/main/docs/source/conf.py

# - Copy over notebooks folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images

examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "notebooks"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

logo_file_name = "MBM_logo.png"
# Copy logo file
shutil.copyfile("../"+logo_file_name, logo_file_name)

# Include examples in documentation
# This adds a lot of time to the doc buiod; to bypass use the environment variable SKIP_EXAMPLES=true
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        ext = os.path.splitext(fil)[1]
        if ext in [".ipynb", ".md", ".rst", ".csv", ".dbf", ".prj", ".shp", ".shx", ".nc"]:
            if "example_data" in root and ext == ".md":
                # Do not copy the tutorials inside the example_data/ folder
                continue
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)

            # If we're skipping examples, put a dummy file in place
            if os.getenv("SKIP_EXAMPLES"):
                if dest_filename.endswith("index.rst"):
                    shutil.copyfile(source_filename, dest_filename)
                else:
                    with open(os.path.splitext(dest_filename)[0] + ".rst", "w") as f:
                        basename = os.path.splitext(os.path.basename(dest_filename))[0]
                        f.write(f"{basename}\n" + "=" * 80)

            # Otherwise, copy over the real example files
            else:
                shutil.copyfile(source_filename, dest_filename)

################################################


# -- Project information -----------------------------------------------------

project = "MassBalanceMachine"
copyright = "2024, MassBalanceMachine core team"
author = "MassBalanceMachine core team"


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.autodoc",
    "myst_parser",
    "nbsphinx",
    # "sphinx_autorun",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False, # keep sidebar sections expanded
    "navigation_depth": 4,        # how deep to show headings
    "titles_only": False,         # if True, only show top-level page titles
    "logo_only": False,
    "display_version": True,
}

language = "en"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "ODINN-SciML", # Username
    "github_repo": "MassBalanceMachine", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}

html_logo = logo_file_name

html_favicon = 'favicon.ico'
