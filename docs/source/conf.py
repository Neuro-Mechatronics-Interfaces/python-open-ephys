# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

import pyoephys
project = 'python-oephys'
copyright = '2025, Jonathan Shulgach'
author = 'Jonathan Shulgach'
release = pyoephys.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

autodoc_mock_imports = [
    'matplotlib',
    'pandas',
    'open_ephys', 
    'pylsl',
    'PyQt5',
    'pyqtgraph',
    'scipy',
    'sklearn',
    'torch',
    'zmq'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False
source_suffix = ['.rst', '.md']
master_doc = 'index'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
