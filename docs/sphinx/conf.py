# Configuration file for the Sphinx documentation builder.
#
# Build with:
#     venv/bin/python -m sphinx -b html docs/sphinx docs/sphinx/_build
# Then open docs/sphinx/_build/index.html.
#
# For the full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Make the bridge package importable for autodoc. Source layout is
# `gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/...`, so the outer
# `gazebo_gymnasium_bridge/` dir needs to be on sys.path.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'gazebo_gymnasium_bridge'))

# -- Project info ------------------------------------------------------------

project = 'Gazebo Gymnasium'
copyright = '2026, Lucas Wendland'
author = 'Lucas Wendland'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',          # autogenerate from docstrings
    'sphinx.ext.viewcode',         # link to source on the rendered docs
    'sphinx.ext.napoleon',         # accept Google/Numpy-style docstrings
    'sphinx.ext.intersphinx',      # cross-link to external project docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# Autodoc: pull docstrings and treat undocumented members as visible too —
# the package isn't fully docstringed yet and we still want a reference page.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
# Mock heavy/native deps so autodoc can import the env modules even when
# gz-transport, torch, etc. aren't available in the docs build environment.
autodoc_mock_imports = [
    'gz', 'gz.transport13', 'gz.msgs10', 'gz.sim8',
    'torch', 'gymnasium', 'stable_baselines3',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    'sb3': ('https://stable-baselines3.readthedocs.io/en/master/', None),
}

# -- HTML output -------------------------------------------------------------

# Use the sphinx_rtd_theme if installed (cleaner than the default alabaster);
# fall back gracefully when it isn't available.
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

html_static_path = ['_static']
