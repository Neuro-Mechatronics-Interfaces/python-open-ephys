from __future__ import annotations
from importlib import resources as ir
from pathlib import Path
from typing import Iterable, Optional
import tempfile
import shutil


_KNOWN_ALIASES = {
    "gestures": "gestures",
}


def list_example_data() -> list[str]:
    """
    Return the list of example dataset folder names bundled under `pyoephys.io.data`,
    if any are present. If none are bundled (preferred for lightweight wheels),
    returns known aliases (may not be available locally).
    """
    try:
        root = ir.files("pyoephys.io.data")
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    except Exception:
        return sorted(_KNOWN_ALIASES.keys())


def _copy_traversable_dir(src: ir.abc.Traversable, dst: Path) -> None:
    """Recursively copy a Traversable directory to a real folder."""
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_traversable_dir(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with child.open("rb") as fsrc, open(target, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)


def _resolve_alias(name: str) -> str:
    return _KNOWN_ALIASES.get(name, name)


def get_example_path(
    name: str,
    dest: Optional[Path] = None,
    strict: bool = True,
) -> Optional[Path]:
    """
    Return a filesystem path to an example dataset folder named `name`.

    If the examples are zipped inside the wheel, the folder is extracted to a temporary
    location (or to `dest` if provided). Returns None if missing and `strict=False`.
    """
    base = _resolve_alias(name)

    try:
        root = ir.files("pyoephys.io.data")
    except Exception:
        if strict:
            raise FileNotFoundError(
                "No bundled examples found. Download example data from your repo "
                "or call this function with strict=False."
            )
        return None

    # Try to get a Traversable directory
    try:
        trav = root.joinpath(base)
        if not trav.is_dir():
            raise FileNotFoundError
    except Exception:
        if strict:
            raise FileNotFoundError(f"Example dataset '{name}' not found.")
        return None

    # Ensure we have a real path on disk (works for zipped dists too)
    with ir.as_file(trav) as real_path:
        if real_path.exists():
            return real_path

    # Fallback: extract manually to temp/dest
    out_dir = dest or Path(tempfile.mkdtemp(prefix=f"pyoephys-example-{base}-"))
    _copy_traversable_dir(trav, out_dir)
    return out_dir


def get_example_oebin_path(name: str = "gestures", strict: bool = True) -> Optional[Path]:
    """
    Return a folder path containing at least one `.oebin` file for the named example.
    """
    p = get_example_path(name, strict=strict)
    if p is None:
        return None
    if not any(p.rglob("*.oebin")):
        if strict:
            raise FileNotFoundError(f"No .oebin file found under example '{name}' at {p}")
        return None
    return p