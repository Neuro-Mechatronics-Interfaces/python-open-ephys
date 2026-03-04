"""
Example script demonstrating how to use the interactive conversion tool
included in the pyoephys library.

This script delegates to the core library function.
"""

import sys
from pathlib import Path


def main():
    try:
        from pyoephys.io import convert_session_ui
    except ImportError:
        # If running from source without install
        current_dir = Path(__file__).resolve().parent
        src_dir = current_dir.parent.parent / "src"
        sys.path.append(str(src_dir))
        from pyoephys.io import convert_session_ui

    # Launch the interactive converter
    convert_session_ui()


if __name__ == "__main__":
    main()
