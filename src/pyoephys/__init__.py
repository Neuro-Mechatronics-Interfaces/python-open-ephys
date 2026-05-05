"""
pyoephys: Python tools for reading, streaming, processing, and visualizing
Open Ephys and related electrophysiology data.
"""

from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    PackageNotFoundError = Exception
    version = None

repo_root = Path(__file__).resolve().parents[2]
__version__ = None

if (repo_root / ".git").exists():
    try:
        from setuptools_scm import get_version

        __version__ = get_version(
            root="../..",
            relative_to=__file__,
            local_scheme="no-local-version",
            tag_regex=r"^v?(?P<version>\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc)\d+)?)$",
            git_describe_command="git describe --dirty --tags --long --match v[0-9]*",
        )
    except Exception:
        __version__ = None

if __version__ is None:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = None

if __version__ is None:
    if version is not None:
        try:
            __version__ = version("python-oephys")
        except PackageNotFoundError:
            __version__ = "0.0.0+unknown"
    else:
        __version__ = "0.0.0+unknown"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys"
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
