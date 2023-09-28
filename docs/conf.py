# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from sphinx.builders.html import StandaloneHTMLBuilder
import subprocess, os, sys


# Doxygen
subprocess.call('doxygen Doxyfile.in', shell=True)


# Generate API
subprocess.call(f'{sys.executable} build_api.py', shell=True)


# -- Project information -----------------------------------------------------

project = 'Kernel Float'
copyright = '2023, Stijn Heldens, Ben van Werkhoven'
author = 'Stijn Heldens, Ben van Werkhoven'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        "breathe",
        "myst_parser",
        #"sphinx_mdinclude",
]

#source_suffix = {
#    '.rst': 'restructuredtext',
#    '.txt': 'markdown',
#    #'.md': 'markdown',
#}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'logo_only': False,

    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']



# -- Breathe configuration -------------------------------------------------

# Setup the breathe extension
breathe_projects = {
    "Kernel Launcher": "./_doxygen/xml"
}

breathe_default_members = ("members",)
breathe_default_project = "Kernel Launcher"

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

