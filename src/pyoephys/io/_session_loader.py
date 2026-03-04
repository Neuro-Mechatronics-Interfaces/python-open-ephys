from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._file_utils import find_oebin_files


@dataclass
class SessionData:
    amplifier_data: np.ndarray  # shape (C, S), float32 (microvolts by default)
    t_amplifier: np.ndarray  # shape (S,), float64 seconds
    sample_rate: float
    channel_names: List[str]


def load_open_ephys_session(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load an Open Ephys session (or a simple NPZ dump) into memory.

    Supported inputs:
      - .npz with keys {emg, timestamps, fs_hz} (preferred quick path)
      - CSV with a ``timestamp`` column and one column per channel
      - .npz with keys {emg, timestamps, fs_hz} (preferred quick path)
      - folder/file with .oebin (+ binary 'continuous.dat' and 'timestamps.npy')
      - folder containing 'amplifier_data.npy' and 'timestamps.npy' and 'sample_rate.txt'

    Returns dict(SessionData.__dict__).

    Notes
    -----
    - For Open Ephys Binary Format, we parse `.oebin` for meta (channel count, names, fs)
      and read 'continuous.dat' (int16, little-endian, interleaved by channel). Data are
      scaled to microvolts using per-channel bitVolts if available, else 0.195 µV/count.
    - If per-sample timestamps aren't available, we synthesize a strictly monotonic vector.
    """
    p = Path(path)

    # Case 0: CSV with a timestamp column
    if p.is_file() and p.suffix.lower() == ".csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load CSV files: pip install pandas"
            )
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            t = df["timestamp"].to_numpy(dtype=np.float64)
            ch_cols = [c for c in df.columns if c != "timestamp"]
        else:
            t = df.iloc[:, 0].to_numpy(dtype=np.float64)
            ch_cols = list(df.columns[1:])
        y = df[ch_cols].to_numpy(dtype=np.float32).T  # (C, S)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 200.0
        fs = round(1.0 / dt) if dt > 0 else 200.0
        return SessionData(y, t, fs, ch_cols).__dict__

    # Case 1: NPZ convenience (emg/timestamps/fs_hz)
    if p.is_file() and p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=False)
        y = np.asarray(z["emg"], dtype=np.float32)
        t = np.asarray(z["timestamps"], dtype=np.float64)
        fs = float(z["fs_hz"])
        ch = [f"ch{i}" for i in range(y.shape[0])]
        return SessionData(y, t, fs, ch).__dict__

    # Case 1.5: XDF
    if p.is_file() and p.suffix.lower() == ".xdf":
        return load_xdf_session(p)

    # Case 2: 'simple' folder with pre-extracted arrays
    if p.is_dir():
        ad = p / "amplifier_data.npy"
        ts = p / "timestamps.npy"
        fs_txt = p / "sample_rate.txt"
        if ad.exists() and ts.exists() and fs_txt.exists():
            y = np.load(ad).astype(np.float32)
            t = np.load(ts).astype(np.float64)
            fs = float(fs_txt.read_text().strip())
            ch = [f"ch{i}" for i in range(y.shape[0])]
            return SessionData(y, t, fs, ch).__dict__

    # Case 3: OEBin
    oebins = find_oebin_files(p if p.is_dir() else p.parent)
    if not oebins:
        raise FileNotFoundError(
            f"Could not find a supported session at: {p}. "
            "Provide a .npz (emg/timestamps/fs_hz) or a folder with .oebin + binary files."
        )
    oebin = oebins[0]
    meta = _load_oebin_meta(oebin)

    stream = _pick_continuous_stream(meta)
    if stream is None:
        raise RuntimeError("No continuous stream found in .oebin metadata.")

    fs = float(
        stream.get("sample_rate")
        or stream.get("sampleRate")
        or stream.get("rate")
        or meta.get("sample_rate", 0.0)
    )
    channels_meta = stream.get("channels") or stream.get("source_channels") or []
    channel_names = [
        c.get("channel_name") or c.get("name") or f"ch{i}"
        for i, c in enumerate(channels_meta)
    ]
    n_channels = int(
        stream.get("num_channels")
        or stream.get("channel_count")
        or len(channel_names)
        or 0
    )
    if n_channels == 0:
        n_channels = len(channel_names)
    if n_channels == 0:
        raise RuntimeError("Cannot determine number of channels from .oebin")

    # Locate binary files
    root = oebin.parent
    dat = _find_first(root, "continuous.dat")
    ts = _find_first(root, "timestamps.npy")

    if dat is None:
        raise FileNotFoundError("continuous.dat not found under session directory.")
    # timestamps are optional; we can synthesize if missing
    have_ts = ts is not None

    # Read int16 interleaved -> (S, C) -> transpose to (C, S)
    raw = np.fromfile(dat, dtype="<i2")
    if raw.size % n_channels != 0:
        remainder = raw.size % n_channels
        print(
            f"Warning: continuous.dat size {raw.size} not divisible by n_channels={n_channels}. "
            f"Truncating last {remainder} items."
        )
        raw = raw[:-remainder]

    samples = raw.size // n_channels
    y_i16 = raw.reshape(samples, n_channels)
    # Scale to microvolts using bitVolts if available, else 0.195 µV/count
    bitvolts = _extract_bitvolts(channels_meta, default_uv_per_count=0.195)
    y = (y_i16 * bitvolts[None, :]).astype(np.float32)  # (S, C)
    y = y.T  # (C, S)

    if have_ts:
        try:
            t = np.load(ts).astype(np.float64)
            # Verify length align
            if t.size > samples:
                print(
                    f"Warning: timestamps length ({t.size}) > samples ({samples}). Truncating timestamps."
                )
                t = t[:samples]

            if t.ndim != 1 or t.size != samples:
                print(
                    f"Warning: timestamps.npy has wrong shape/len ({t.size} vs {samples}). Synthesizing timestamps."
                )
                t = np.arange(samples, dtype=np.float64) / fs
        except Exception as e:
            print(f"Warning: Failed to load timestamps.npy: {e}. Synthesizing.")
            t = np.arange(samples, dtype=np.float64) / fs
    else:
        # Synthesize strictly monotonic timestamps
        t0 = 0.0
        t = t0 + np.arange(samples, dtype=np.float64) / fs

    # If channel_names length mismatches, fix it
    if len(channel_names) != n_channels:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    # Load events (TTL)
    events = _load_open_ephys_events(root)

    return {
        "amplifier_data": y,
        "t_amplifier": t,
        "sample_rate": fs,
        "channel_names": channel_names,
        "events": events,
    }


def _load_open_ephys_events(root: Path) -> Dict[str, Any]:
    """
    Search for and load TTL events from the Open Ephys events directory.
    Returns a dict with 'timestamps', 'channels', 'states'.
    """
    # Typical path: events/*/TTL/timestamps.npy
    # We search specifically for a 'TTL' folder
    ttl_dirs = list(root.rglob("TTL"))

    if not ttl_dirs:
        # Fallback: Look for 'events' folder with standard files
        # Sometimes events are directly in events/
        events_dir = root / "events"
        if events_dir.exists() and (events_dir / "timestamps.npy").exists():
            ttl_dirs = [events_dir]
        else:
            return {}

    # Take the first TTL folder found (usually only one per recording)
    ttl_dir = ttl_dirs[0]

    ts_file = ttl_dir / "timestamps.npy"
    states_file = ttl_dir / "channel_states.npy"
    ch_file = ttl_dir / "channels.npy"

    events = {}

    if ts_file.exists():
        events["timestamps"] = np.load(ts_file).astype(np.float64)

        if states_file.exists():
            st = np.load(states_file)
            events["states"] = st
            # In some versions, channel info is encoded in states or separate
            if ch_file.exists():
                events["channels"] = np.load(ch_file)
            else:
                events["channels"] = np.zeros_like(st)

        elif ch_file.exists():
            events["channels"] = np.load(ch_file)
            # Look for states.npy?
            st_file_alt = ttl_dir / "states.npy"
            if st_file_alt.exists():
                events["states"] = np.load(st_file_alt)
            else:
                events["states"] = np.ones_like(events["channels"])

    return events


def load_xdf_session(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load an XDF file (LabRecorder) into the session dictionary format.

    Requires 'pyxdf' installed.
    Searches for a stream with type="EMG" (or "EEG", "Bio", etc. logic).
    """
    try:
        import pyxdf
    except ImportError:
        raise ImportError(
            "pyxdf is required to load .xdf files. Install with 'pip install pyxdf'"
        )

    streams, header = pyxdf.load_xdf(str(path))

    # Heuristic to find the 'main' amplifier stream
    target_stream = None

    # Priority 1: Type is EMG
    for s in streams:
        if s["info"]["type"][0].lower() == "emg":
            target_stream = s
            break

    # Priority 2: Name contains "OpenEphys" or "TMSi"
    if target_stream is None:
        for s in streams:
            name = s["info"]["name"][0]
            if "OpenEphys" in name or "TMSi" in name:
                target_stream = s
                break

    # Priority 3: Fallback to first stream if only one exists (and isn't Markers)
    if target_stream is None:
        candidates = [s for s in streams if s["info"]["type"][0].lower() != "markers"]
        if len(candidates) == 1:
            target_stream = candidates[0]

    if target_stream is None:
        raise ValueError(
            f"Could not automatically identify an EMG/Amplifier stream in {path}. Found types: {[s['info']['type'][0] for s in streams]}"
        )

    # Extract data
    # XDF time_series shape is (n_samples, n_channels)
    # We want (n_channels, n_samples)
    y = target_stream["time_series"].T.astype(np.float32)
    t = np.array(target_stream["time_stamps"], dtype=np.float64)

    # Try to get nominal fs
    NominalSamplingRate = float(target_stream["info"]["nominal_srate"][0])
    fs = NominalSamplingRate if NominalSamplingRate > 0 else 2000.0  # Fallback

    # Channel Names
    try:
        # info -> desc -> channels -> channel -> label
        ch_list = target_stream["info"]["desc"][0]["channels"][0]["channel"]
        channel_names = [ch["label"][0] for ch in ch_list]
    except (KeyError, IndexError):
        channel_names = [f"ch{i}" for i in range(y.shape[0])]

    # Extract Events (Markers)
    events = {}
    for s in streams:
        if s["info"]["type"][0].lower() == "markers":
            # XDF marker streams usually have strings in time_series
            mrk_ts = np.array(s["time_stamps"], dtype=np.float64)
            mrk_dat = s["time_series"]  # list of lists of strings usually

            # Flatten if necessary
            if isinstance(mrk_dat, list):
                if len(mrk_dat) > 0 and isinstance(mrk_dat[0], list):
                    mrk_dat = [
                        item[0] for item in mrk_dat
                    ]  # Flatten nested single-element lists
                mrk_dat = np.array(mrk_dat)

            # Assign to events dict. If multiple marker streams, maybe dict of dicts?
            # For simplicity, we assume one main marker stream or combine them
            # Let's use the stream name as the key if we have multiple
            s_name = s["info"]["name"][0].replace(" ", "_").replace("-", "_")
            events[s_name] = {"timestamps": mrk_ts, "data": mrk_dat}

            # If this is the main "Markers" stream, also map to generic keys
            if "Markers" in s_name or len(events) == 1:
                events["timestamps"] = mrk_ts
                events["data"] = mrk_dat

    return {
        "amplifier_data": y,
        "t_amplifier": t,
        "sample_rate": fs,
        "channel_names": channel_names,
        "events": events,
    }


def _load_oebin_meta(oebin_path: Path) -> dict:
    with open(oebin_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_continuous_stream(meta: dict) -> Optional[dict]:
    """
    Traverse common .oebin shapes and return the first 'continuous' stream dict.

    Handles all known Open Ephys GUI v0.5/v0.6/v1.x layout variants:
      - meta["continuous"][0]                           (GUI ≥ 1.0, top-level)
      - meta["recordings"][0]["continuous"][0]
      - meta["recordings"][0]["streams"]["continuous"][0]
      - meta["streams"]["continuous"][0]
    """

    def _listify(x):
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return list(x.values())
        return []

    # Top-level "continuous" (GUI >= 1.0 .oebin format)
    if "continuous" in meta:
        items = _listify(meta["continuous"])
        if items:
            return items[0]

    candidates: list[dict] = []
    if "recordings" in meta:
        for rec in _listify(meta["recordings"]):
            streams = rec.get("streams") or rec
            if "continuous" in streams:
                candidates.extend(_listify(streams["continuous"]))
    if "streams" in meta and "continuous" in meta["streams"]:
        candidates.extend(_listify(meta["streams"]["continuous"]))

    return candidates[0] if candidates else None


def _find_first(root: Path, filename: str) -> Optional[Path]:
    for p in root.rglob(filename):
        return p
    return None


def _extract_bitvolts(
    channels_meta: List[dict], default_uv_per_count: float = 0.195
) -> np.ndarray:
    uv = []
    for ch in channels_meta:
        bv = ch.get("bit_volts") or ch.get("bitVolts") or None
        if bv is None:
            uv.append(default_uv_per_count)
        else:
            # bit_volts is in µV/count when units=="uV", otherwise assume V/count
            units = (ch.get("units") or "").strip().lower()
            if units in ("uv", "µv", "microvolts"):
                uv.append(float(bv))  # already µV/count
            else:
                uv.append(float(bv) * 1e6)  # convert V/count → µV/count
    if not uv:
        return np.full((1,), default_uv_per_count, dtype=np.float32)
    return np.asarray(uv, dtype=np.float32)


# import os
# from open_ephys.analysis.session import Session
#
#
# def load_session(path: str) -> Session:
#     """
#     Load an Open Ephys session from the specified directory.
#     Raises informative errors if loading fails.
#     """
#     if not os.path.isdir(path):
#         raise FileNotFoundError(f"Path '{path}' is not a valid directory.")
#
#     try:
#         session = Session(path)
#         print(f"[Loaded] Session from {path}")
#         print(f"[Info] {len(session.recordnodes)} record node(s) found.")
#         return session
#     except Exception as e:
#         raise RuntimeError(f"Failed to load session from {path}: {e}")
