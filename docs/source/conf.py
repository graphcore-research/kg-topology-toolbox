# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Copyright (c) 2022 Read the Docs Inc. All rights reserved.

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "KG Topology Toolbox"
copyright = "(c) 2023 Graphcore Ltd. All rights reserved"
author = "Alberto Cattaneo, Daniel Justus"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]
numpydoc_show_class_members = False
todo_include_todos = True
autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
