"""
pyoephys.io._dataset_utils

Dataset building utilities (Ported from python-intan).
"""

import os
import re
import json
import logging
import numpy as np
from typing import List, Tuple, Optional

from pyoephys.processing import EMGPreprocessor
from ._file_utils import labels_from_events, find_event_for_file
from ._grid_utils import infer_grid_dimensions, apply_grid_permutation, parse_orientation_from_filename
from ..io import load_open_ephys_session

def normalize_name(s: str) -> str:
    s = s.strip().upper()
    s = s.replace("_", "").replace("-", "")
    if s.startswith("CH") and s[2:].isdigit():
        return f"CH{int(s[2:])}"
    return s

def build_indices_from_mapping(raw_names: List[str], mapping_names: List[str], strict: bool = True) -> List[int]:
    lookup = {normalize_name(n): i for i, n in enumerate(raw_names)}
    indices = []
    missing = []
    for nm in mapping_names:
        key = normalize_name(nm)
        if key in lookup:
            indices.append(lookup[key])
        else:
            missing.append(nm)
    
    if strict and missing:
        raise ValueError(f"Channel mapping references missing names: {missing[:5]}...")
    return indices

def select_channels(
    raw_names: List[str],
    channels: List[int] | None,
    channel_map: str | None,
    channel_map_file: str,
    non_strict: bool = False
) -> Tuple[List[int], List[str]]:
    
    if channel_map:
        with open(channel_map_file, 'r') as f:
            mappings = json.load(f)
        if channel_map not in mappings:
            raise KeyError(f"Mapping '{channel_map}' not in {channel_map_file}")
        
        mapping_names = mappings[channel_map]
        indices = build_indices_from_mapping(raw_names, mapping_names, strict=not non_strict)
        return indices, [raw_names[i] for i in indices]
        
    if channels:
        return channels, [raw_names[i] for i in channels]
        
    return list(range(len(raw_names))), raw_names


def load_open_ephys_data(path: str) -> dict:
    """
    Load Open Ephys session and adapt to standardized dict format.
    """
    try:
        session_data = load_open_ephys_session(path)
    except Exception as e:
        raise ValueError(f"Failed to load Open Ephys session at {path}: {e}")

    # Map SessionData dict to process_recording format
    # SessionData: amplifier_data (C, S), t_amplifier (S,), sample_rate, channel_names
    
    return {
        "amplifier_data": session_data["amplifier_data"],
        "t_amplifier": session_data["t_amplifier"],
        "frequency_parameters": {"amplifier_sample_rate": session_data["sample_rate"]},
        "sample_rate": session_data["sample_rate"],
        "channel_names": session_data["channel_names"]
    }



def assess_channel_quality(
    emg: np.ndarray,
    fs: float,
    min_rms_uv: float = 0.5,
    max_rms_uv: float = 5000.0,
    flat_fraction: float = 0.01,
) -> dict:
    """
    Assess per-channel signal quality on a raw EMG array.

    Parameters
    ----------
    emg : (C, N) array of raw EMG samples (µV).
    fs : Sample rate in Hz (not used currently, reserved for future freq checks).
    min_rms_uv : Channels with RMS below this are flagged as dead/disconnected.
    max_rms_uv : Channels with RMS above this are flagged as saturated/artefact.
    flat_fraction : Fraction of consecutive identical sample pairs that triggers
                    a "flat/stuck ADC" flag (default 1 %).

    Returns
    -------
    dict with keys:
      ``good``      – list of channel indices that passed all checks
      ``dead``      – list of indices with RMS < min_rms_uv
      ``saturated`` – list of indices with RMS > max_rms_uv
      ``flat``      – list of indices with too many consecutive identical samples
      ``rms``       – (C,) float32 array of per-channel RMS values (µV)
      ``summary``   – human-readable one-line string
    """
    C, N = emg.shape
    rms = np.sqrt(np.mean(emg.astype(np.float64) ** 2, axis=1)).astype(np.float32)

    dead      = np.where(rms < min_rms_uv)[0].tolist()
    saturated = np.where(rms > max_rms_uv)[0].tolist()

    flat = []
    if N > 1:
        for ch in range(C):
            frac = float(np.sum(np.diff(emg[ch]) == 0)) / (N - 1)
            if frac > flat_fraction:
                flat.append(ch)

    bad  = set(dead) | set(saturated) | set(flat)
    good = [i for i in range(C) if i not in bad]

    summary = (
        f"Channel QC: {C} total | {len(good)} good | "
        f"{len(dead)} dead | {len(saturated)} saturated | {len(flat)} flat/stuck"
    )

    return {
        "good":      good,
        "dead":      dead,
        "saturated": saturated,
        "flat":      flat,
        "rms":       rms,
        "summary":   summary,
    }


def process_recording(
    data: dict,
    file_path: str,
    root_dir: str,
    events_file: str | None,
    window_ms: int,
    step_ms: int,
    paper_style: bool = False,
    channels: List[int] | None = None,
    ignore_labels: List[str] | None = None,
    ignore_case: bool = False,
    keep_trial: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Process single recording: preprocess -> feature extract -> label.

    Notes
    -----
    ``window_starts`` are expressed in **absolute hardware sample numbers**
    (same space as the ``Sample Index`` column in transition-style label files).
    For CSV data that starts at t=0 the offset is 0; for Open Ephys recordings
    that begin mid-session the offset is derived from ``t_amplifier[0] * fs``.
    """

    fs = float(data.get("frequency_parameters", {}).get("amplifier_sample_rate") or data.get("sample_rate", 2000))
    emg = data["amplifier_data"]  # (C, N)
    t   = data["t_amplifier"]
    raw_names = data.get("channel_names", [f"CH{i}" for i in range(emg.shape[0])])

    if channels:
        emg = emg[channels, :]

    if paper_style:
        pre = EMGPreprocessor(fs=fs, band=(120.0, fs/2-1), envelope_cutoff=None, feature_fns=["rms"])
    else:
        pre = EMGPreprocessor(fs=fs, envelope_cutoff=5.0)

    emg_pp = pre.preprocess(emg)

    feature_fns = ["rms"] if paper_style else None
    X = pre.extract_emg_features(emg_pp, window_ms, step_ms, feature_fns=feature_fns, progress=False)

    # ── Absolute sample offset ─────────────────────────────────────────────
    # t_amplifier stores seconds since Open Ephys session start.  For a
    # recording that started part-way through a session t[0] can be e.g. 80.6 s
    # ≡ sample 161 253 at 2000 Hz.  Label files (e.g. emg.txt) use the same
    # absolute sample-counter space, so we must offset window_starts to match.
    step_samples = int(step_ms * fs / 1000)
    start_sample = int(round(float(t[0]) * fs)) if len(t) > 0 and float(t[0]) > 0.5 else 0
    window_starts = np.arange(X.shape[0]) * step_samples + start_sample

    if events_file is None:
        events_file = find_event_for_file(root_dir, file_path)
        if not events_file:
            logging.warning(f"No event file found for {file_path}. Using folder name label.")
            label = os.path.basename(os.path.dirname(file_path))
            y = np.full(X.shape[0], label)
            meta = {"fs": fs, "selected_channels": channels, "channel_names": raw_names}
            return X, y, meta

    y = labels_from_events(events_file, window_starts)

    # ── Build ignore mask ──────────────────────────────────────────────────
    # Always exclude windows that fall outside any labelled epoch ("Unknown").
    # Also exclude any caller-specified labels (e.g. "Start", "Rest").
    base_ignore = {"unknown"}
    if ignore_labels:
        base_ignore |= {l.lower() for l in ignore_labels}

    mask = np.ones(len(y), dtype=bool)
    mask &= ~np.array([str(lbl).lower() in base_ignore for lbl in y])

    if not keep_trial:
        y = np.array([re.sub(r'_\d+$', '', l) for l in y])

    X = X[mask]
    y = y[mask]

    metadata = {
        "fs": fs,
        "selected_channels": channels,
        "channel_names": raw_names if channels is None else [raw_names[i] for i in channels],
        "envelope_cutoff_hz": getattr(pre, "env_cut", 5.0) or 5.0,
    }

    return X, y, metadata


def process_recordings(
    paths: list,
    root_dir: str,
    window_ms: int = 200,
    step_ms: int = 50,
    events_file: str | None = None,
    **kwargs,
) -> tuple:
    """Process multiple recordings and concatenate the results into a single dataset.

    Each path is loaded with :func:`load_open_ephys_session` and then passed
    to :func:`process_recording`.  All keyword arguments are forwarded to
    :func:`process_recording` (e.g. ``channels``, ``ignore_labels``,
    ``paper_style``).

    Parameters
    ----------
    paths : list[str]
        Ordered list of file or folder paths accepted by
        :func:`load_open_ephys_session`.
    root_dir : str
        Root directory used for event-file auto-discovery.
    window_ms : int
        Sliding-window length in milliseconds.
    step_ms : int
        Window step in milliseconds.
    events_file : str or None
        Shared event file.  When None, each recording uses auto-discovery.

    Returns
    -------
    X : np.ndarray, shape (total_windows, n_features)
    y : np.ndarray, shape (total_windows,)
    meta : dict
        Metadata from the *last* processed recording (fs, channel_names, …).
    """
    if not paths:
        raise ValueError("paths must be a non-empty list.")

    X_parts, y_parts = [], []
    meta: dict = {}
    for path in paths:
        data = load_open_ephys_session(path)
        X, y, meta = process_recording(
            data, path, root_dir, events_file, window_ms, step_ms, **kwargs
        )
        X_parts.append(X)
        y_parts.append(y)
        logging.info(f"process_recordings: {path!r} → {X.shape[0]} windows")

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), meta


def save_dataset(save_path, X, y, metadata, window_ms, step_ms, channel_map=None, channel_map_file=None, modality='emg', ignore_labels=None):
    class_names = sorted(set(y))
    label_to_id = {c: i for i, c in enumerate(class_names)}

    # Guard against selected_channels being None (channels=None means "all channels")
    sel_ch = metadata.get("selected_channels") or []
    feat_names = metadata.get("feature_names") or []
    ign = list(ignore_labels) if ignore_labels else []

    np.savez(
        save_path,
        X=X, y=y,
        fs=metadata["fs"],
        emg_fs=metadata["fs"],
        class_names=np.array(class_names, dtype=object),
        window_ms=window_ms,
        step_ms=step_ms,
        envelope_cutoff_hz=float(metadata.get("envelope_cutoff_hz", 5.0)),
        selected_channels=np.array(sel_ch, dtype=int),
        channel_names=np.array(metadata.get("channel_names", []), dtype=object),
        feature_names=np.array(feat_names, dtype=object),
        ignore_labels=np.array(ign, dtype=object),
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
        modality=np.array(modality, dtype=object)
    )
    print(f"Saved dataset to {save_path} ({X.shape[0]} windows, {len(class_names)} classes)")


def load_dataset(path: str) -> tuple:
    """
    Load a dataset saved by :func:`save_dataset`.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file produced by ``save_dataset``.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_features)
    y : np.ndarray, shape (n_windows,)
    meta : dict
        Keys: ``fs``, ``window_ms``, ``step_ms``, ``envelope_cutoff_hz``,
        ``channel_names``, ``selected_channels``, ``feature_names``.
    """
    ds = np.load(path, allow_pickle=True)
    meta = {
        "fs":                float(ds["fs"])               if "fs"               in ds.files else 200.0,
        "window_ms":         int(ds["window_ms"])          if "window_ms"        in ds.files else 200,
        "step_ms":           int(ds["step_ms"])            if "step_ms"          in ds.files else 50,
        "envelope_cutoff_hz": float(ds["envelope_cutoff_hz"]) if "envelope_cutoff_hz" in ds.files else 5.0,
        "channel_names":     ds["channel_names"].tolist()  if "channel_names"    in ds.files else None,
        "selected_channels": ds["selected_channels"].tolist() if "selected_channels" in ds.files else None,
        "feature_names":     ds["feature_names"].tolist()  if "feature_names"    in ds.files else None,
        "ignore_labels":     ds["ignore_labels"].tolist()  if "ignore_labels"    in ds.files else [],
    }
    return ds["X"], ds["y"], meta
