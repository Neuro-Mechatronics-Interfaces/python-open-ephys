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
    """
    
    fs = float(data.get("frequency_parameters", {}).get("amplifier_sample_rate") or data.get("sample_rate", 2000))
    emg = data["amplifier_data"] # (C, N)
    t = data["t_amplifier"]
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
    
    start_idx = 0
    step_samples = int(step_ms * fs / 1000)
    window_starts = np.arange(X.shape[0]) * step_samples + start_idx
    
    if events_file is None:
        events_file = find_event_for_file(root_dir, file_path)
        if not events_file:
             logging.warning(f"No event file found for {file_path}. Using folder name label.")
             label = os.path.basename(os.path.dirname(file_path))
             y = np.full(X.shape[0], label)
             meta = {"fs": fs, "selected_channels": channels, "channel_names": raw_names}
             return X, y, meta

    y = labels_from_events(events_file, window_starts)
    
    mask = np.ones(len(y), dtype=bool)
    if ignore_labels:
        if ignore_case:
            ign = {l.lower() for l in ignore_labels}
            mask &= ~np.array([l.lower() in ign for l in y])
        else:
            mask &= ~np.isin(y, ignore_labels)
            
    if not keep_trial:
        y = np.array([re.sub(r'_\d+$', '', l) for l in y])
        
    X = X[mask]
    y = y[mask]
    
    metadata = {
        "fs": fs,
        "selected_channels": channels,
        "channel_names": raw_names if channels is None else [raw_names[i] for i in channels]
    }
    
    return X, y, metadata


def save_dataset(save_path, X, y, metadata, window_ms, step_ms, channel_map=None, channel_map_file=None, modality='emg'):
    class_names = sorted(set(y))
    label_to_id = {c: i for i, c in enumerate(class_names)}
    
    np.savez(
        save_path,
        X=X, y=y,
        fs=metadata["fs"],
        emg_fs=metadata["fs"],
        class_names=np.array(class_names, dtype=object),
        window_ms=window_ms,
        step_ms=step_ms,
        selected_channels=np.array(metadata.get("selected_channels", []), dtype=int),
        channel_names=np.array(metadata.get("channel_names", []), dtype=object),
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
        modality=np.array(modality, dtype=object)
    )
    print(f"Saved dataset to {save_path} ({X.shape[0]} windows, {len(class_names)} classes)")
