# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pyvista
from sphinx_gallery.sorting import FileNameSortKey
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
from pyepri import __version__

# -- Project information -----------------------------------------------------

project = u"PyEPRI"
copyright = u"2024, Université Paris Cité & Centre National de la Recherche Scientifique"
author = u"Rémy Abergel"
version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
#    "myst_nb",
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode", # provide link toward source code
    "sphinx_gallery.gen_gallery",
#    "sphinx_tabs.tabs",
#    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "pyvista.ext.viewer_directive",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.video",    
]
autoapi_dirs = ["../src"]
autoapi_template_dir = '_autoapi_templates'
mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = [
    ".md",
    ".rst",
]

# -- bibtex ------------------------------------------------------------------
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'alpha'

# -- Sphinx Gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': ['../examples/filtered_back_projection',
                      '../examples/tv_regularization',
                      '../examples/sources_separation',
                      #'../examples/plug_and_play_regularization',
                      #'../tutorials',
                      '../examples/getting_started',
                      ], # path to input example scripts
    'gallery_dirs': ['_gallery/filtered_back_projection',
                     '_gallery/tv_regularization',
                     '_gallery/sources_separation',
                     #'_gallery/plug_and_play_regularization',
                     #'_gallery/tutorials',
                     '_gallery/getting_started',
                     ], # path to where to save gallery generated outputs
    'within_subsection_order': FileNameSortKey,
    'example_extensions': {'.py'},
    'nested_sections': False,
    'filename_pattern': r'/(example|tutorial)_',
    'example_extensions': {'.py'},
    'show_memory': True,
    'image_scrapers': (DynamicScraper(), "matplotlib"),
    'remove_config_comments': True,
}
suppress_warnings = ["config.cache"] # suppress warning about caching
                                     # caused by the use of
                                     # FileNameSortKey (could be
                                     # removed in a near future, see
                                     # https://github.com/sphinx-gallery/sphinx-gallery/pull/1289)

# -- PyVista configuration --------------------------------------------------
#
# Adapted from PyVista conf.py in https://github.com/pyvista Manage
# errors.
#
pyvista.set_error_output_file("pyvista-errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# -- Options for HTML output ------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = "PyEPRI documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "PyEPRI documentation"

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path or fully
# qualified paths (eg. https://...)
html_css_files = [
    'custom.css',
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo2.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logo.ico"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# other configurations
html_show_sphinx = False

# -- PyVista configuration ---------------------------------------------------
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
pyvista.set_jupyter_backend(None)
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'
