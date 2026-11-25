"""Read TTL transitions from an Open Ephys Binary recording."""

import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from ..sync import DigitalEdge


def _integers(path):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or data.dtype.kind != "i":
        raise ValueError(f"Expected a one-dimensional signed integer array in {path}.")
    return data


def read_digital_edges(path, *, event_folder=None, segment=0):
    """Return edges from a Binary recording's structure.oebin (GUI 0.6+).

    Both polarities and absolute sample numbers are preserved. Use
    `event_folder` to select one event channel by its metadata folder name.
    `segment` labels this recording when combining multiple recordings.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "structure.oebin"
    metadata = json.loads(path.read_bytes())
    event_root = (path.parent / "events").resolve()
    edges = []
    matched = False
    for event in metadata.get("events", []):
        name = event["folder_name"].replace("\\", "/").rstrip("/")
        if event_folder is not None and name != str(event_folder).replace("\\", "/").rstrip("/"):
            continue
        folder = (event_root / name).resolve()
        if not folder.is_relative_to(event_root):
            raise ValueError(f"Open Ephys event folder is outside the recording: {name}.")
        if not (folder / "states.npy").is_file():
            continue
        matched = True
        rate = float(event["sample_rate"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid Open Ephys TTL sample rate in {name}.")
        states = _integers(folder / "states.npy")
        numbers = _integers(folder / "sample_numbers.npy")
        if (len(states) != len(numbers) or np.any(states == 0)
                or np.any((states < -256) | (states > 256)) or np.any(numbers[1:] < numbers[:-1])):
            raise ValueError(f"Invalid TTL state or sample number sequence in {name}.")
        stream_id = str(PurePosixPath(name).parent)
        edges.extend(
            DigitalEdge(int(number), abs(int(state)), bool(state > 0), rate, stream_id, segment, name)
            for number, state in zip(numbers, states)
        )
    if event_folder is not None and not matched:
        raise ValueError(f"No saved Open Ephys TTL channel at {event_folder!r}.")
    return sorted(edges, key=lambda edge: (edge.stream_id, edge.sample_number, edge.source))
