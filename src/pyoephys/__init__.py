"""
pyoephys: Python interface for reading, processing, and visualizing data
from the Open Ephys acquisition systems.

This package includes modules for:
- Reading and parsing `.oebin` and `.rec` files
- Streamed acquisition via TCP/IP from the Intan RHX software
- Visualization of high-density EMG or LFP signals
- Configuration and device control
"""

__version__ = "0.0.1"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/jshulgach/intan-python"
__description__ = "Python interface for streaming, parsing, and analyzing Open Ephys files"

import importlib as _importlib

submodules = [
    # 'decomposition',
    'applications',
    'io',
    #'plotting',
    # 'control',
    'processing',
    'interface',
    #'samples',
    # 'stream',
]

__all__ = submodules + [
    #'LowLevelCallable',
    #'tests',
    #'show_config',
    '__version__',
]


def __dir__():
    return __all__
