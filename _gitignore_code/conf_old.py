# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import sphinx_rtd_theme

# sys.path.insert(0, os.path.abspath('../MappingMovement/'))
# sys.path.insert(0, os.path.abspath('../GetDataFrame/'))
# sys.path.insert(0, os.path.abspath('..'))

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functional'))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xyz'
copyright = '2024, AF'
author = 'AF'
release = '00.00.01'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
