# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add source code to path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("[DOCS] MOSAIC library path: {}".format(sys.path[0]))

# -- Project information -----------------------------------------------------

project = 'MOSAIC'
copyright = '2024, MOSAIC Contributors'
author = 'Abdulhamid Mousa'
release = '0.1.0'

# The master toctree document.
master_doc = 'index'

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # External extensions
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_favicon",
    "sphinxcontrib.mermaid",
]

# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# MyST parser extensions
myst_enable_extensions = [
    "dollarmath",
]

# Mock imports for autodoc (packages not installed on ReadTheDocs)
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "gymnasium",
    "pygame",
    "tqdm",
    "pyglet",
    "pettingzoo",
    "tensorboard",
    "wandb",
    "moviepy",
    "imageio",
    "mpi4py",
    "torch",
    "tensorflow",
    "mindspore",
    "PyQt6",
    "grpcio",
    "ray",
    "cleanrl",
    "xuance",
]

# Pygments style
pygments_style = "tango"
pygments_dark_style = "zenburn"

# Templates and exclusions
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "MOSAIC"
html_short_title = "MOSAIC"
html_scaled_image_link = False
html_static_path = ['_static']

html_theme_options = {
    # Logo
    "light_logo": "figures/logo.png",
    "dark_logo": "figures/logo.png",
    # Source links
    "source_repository": "https://github.com/Abdulhamid97Mousa/MOSAIC",
    "source_branch": "main",
    "source_directory": "docs/source",
    "top_of_page_buttons": ["view", "edit"],
    # Navigation
    "navigation_with_keys": True,
}

html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/sidebar-collapse.js',
]

favicons = [
    {"rel": "icon", "type": "image/x-icon", "href": "figures/favicon.ico"},
    {"rel": "icon", "type": "image/png", "sizes": "32x32", "href": "figures/favicon_32.png"},
    {"rel": "icon", "type": "image/png", "sizes": "16x16", "href": "figures/favicon_16.png"},
]

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    'pettingzoo': ('https://pettingzoo.farama.org/', None),
}
