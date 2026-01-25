"""
Lightweight helpers to keep heavy/GUI/ML deps optional.

Usage pattern:

    from pyoephys._optional import optional_import

    np, _ = optional_import("numpy")  # required at runtime -> raises clean ImportError
    torch, have_torch = optional_import("torch", allow_none=True)
    if not have_torch:
        raise ImportError("Install with: pip install 'pyoephys[ml]'")

For GUI modules, import inside functions so headless users don't pay the import cost.
"""

from __future__ import annotations

from importlib import import_module
from typing import Tuple, Any


def optional_import(modname: str, *, allow_none: bool = False) -> Tuple[Any, bool]:
    """
    Attempt to import `modname`. Return (module_or_None, available_flag).

    If allow_none=False and the import fails, raises ImportError with a friendly message.
    """
    try:
        mod = import_module(modname)
        return mod, True
    except Exception as e:  # pragma: no cover - env dependent
        if allow_none:
            return None, False
        raise ImportError(
            f"Optional dependency '{modname}' is not installed. "
            f"Install the extra that provides it, e.g.: "
            f"pip install 'pyoephys[gui]' or 'pyoephys[ml]'. Original error: {e}"
        )
