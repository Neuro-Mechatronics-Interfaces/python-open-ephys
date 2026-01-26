# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'python-oephys'
copyright = '2025, Jonathan Shulgach'
author = 'Jonathan Shulgach'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

autodoc_mock_imports = [
    'open_ephys', 
    'pylsl',
    'PyQt5',
    'pyqtgraph',
    'zmq'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
