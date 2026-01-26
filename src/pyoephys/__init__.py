"""
pyoephys: Python tools for reading, streaming, processing, and visualizing
Open Ephys and related electrophysiology data.
"""

try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("python-oephys")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0+unknown"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/jshulgach/intan-python"
__description__ = "Python interface for streaming, parsing, and analyzing Open Ephys files"

submodules = [
    'applications',
    'interface',
    'io',
    'ml',
    'plotting',
    'processing',
]

__all__ = submodules + [
    #'LowLevelCallable',
    #'tests',
    #'show_config',
    '__version__',
]


def __dir__():
    return __all__
