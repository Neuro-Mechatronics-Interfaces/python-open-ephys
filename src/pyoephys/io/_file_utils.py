"""
pyoephys.io._file_utils
Utility functions for file and path handling.
"""

import sys
import os
import json
import glob
import numpy as np
from pathlib import Path
import platform
import pathlib
import yaml

# Standard Typing Imports
from typing import Optional, Tuple, Dict, Any

# Optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tkinter import filedialog
except Exception:
    filedialog = None

def adjust_path(path):
    system = platform.system()
    if "microsoft" in platform.uname().release.lower():
        system = "WSL"
    if system == "Windows":
        return path
    elif system == "WSL":
        if len(path) > 1 and path[1] == ":":
            drive_letter = path[0].lower()
            linux_path = path[2:].replace("\\", "/")
            return f"/mnt/{drive_letter}{linux_path}"
        return path
    return path

def check_file_present(file, metrics_file, verbose=False):
    filename = pathlib.Path(file).name
    if filename not in metrics_file['File Name'].tolist():
        if verbose:
            print(f"File {filename} not found in metrics file")
        return filename, False
    return filename, True

def print_progress(i, target, print_step, percent_done, bar_length=40):
    fraction_done = 100 * (1.0 * i / target)
    if fraction_done >= percent_done:
        fraction_bar = i / target
        arrow = '=' * int(fraction_bar * bar_length - 1) + '>' if fraction_bar > 0 else ''
        padding = ' ' * (bar_length - len(arrow))
        ending = '\n' if i == target - 1 else '\r'
        print(f'Progress: [{arrow}{padding}] {int(fraction_bar * 100)}%', end=ending)
        sys.stdout.flush()
        percent_done += print_step
    return percent_done

def load_simple_config(file_path=None, verbose=False):
    if file_path is None and filedialog:
        # Pass
        pass
    if not file_path: return {}
    config_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config_data[key.strip()] = value.strip()
    return config_data

def load_yaml_file(file_path=None):
    if not file_path: return None
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_json_file(file_path=None):
    if not file_path: return None
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def load_config_file(file_path=None, verbose=False):
    if not file_path: return None
    ext = Path(file_path).suffix.lower()
    if ext == '.txt': return load_simple_config(file_path, verbose)
    elif ext in ['.yaml', '.yml']: return load_yaml_file(file_path)
    elif ext == '.json': return load_json_file(file_path)
    return None

def labels_from_events(event_path, window_starts, *, strict_segment=False, fs=2000):
    if pd is None: raise ImportError("pandas required")
    df = pd.read_csv(event_path)
    if 'Sample Index' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Event file must have 'Sample Index' and 'Label' columns")
    ev_idx = np.asarray(df['Sample Index'].values, dtype=np.int64)
    ev_lab = df['Label'].astype(str).values
    ev_lab = np.array([lab.split('#')[0].strip() for lab in ev_lab], dtype=str)
    order = np.argsort(ev_idx)
    ev_idx = ev_idx[order]
    ev_lab = ev_lab[order]
    pos = np.searchsorted(ev_idx, window_starts, side='right') - 1
    y = np.where(pos >= 0, ev_lab[pos], 'Unknown')
    if strict_segment:
        next_pos = pos + 1
        next_change = np.where(next_pos < ev_idx.size, ev_idx[next_pos], np.iinfo(np.int64).max)
        ok = (next_change > window_starts)
        y = np.where(ok, y, 'Unknown')
    return y

def parse_event_file(event_path, verbose=False):
    """
    Load an event file (.event or .txt) as a pandas DataFrame.
    Normalizes columns to 'sample_index' and 'label'.
    """
    if pd is None:
        raise ImportError("pandas is required to parse event files")
    
    df = pd.read_csv(event_path)
    
    # Normalization map
    rename_map = {
        "Sample Index": "sample_index",
        "Label": "label",
        "timestamp": "sample_index", # Fallback for some formats
        "event": "label"
    }
    
    current_cols = df.columns.tolist()
    to_rename = {k: v for k, v in rename_map.items() if k in current_cols and v not in current_cols}
    if to_rename:
        df = df.rename(columns=to_rename)
        
    if "sample_index" not in df.columns or "label" not in df.columns:
        if verbose:
            print(f"[Warning] Event file {event_path} missing required columns. Cols: {df.columns.tolist()}")
            
    return df


def stem_without_timestamp(path):
    import re
    if hasattr(path, "item"):
        try: path = path.item()
        except: pass
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r'_\d{6}_\d{6}$', '', stem)
    stem = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{6}$', '', stem)
    stem = re.sub(r'_\d{8}_\d{6}$', '', stem)
    stem = re.sub(r'_\d{6}$', '', stem)  # YYMMDD
    stem = re.sub(r'_\d+$', '', stem)    # Trial number
    return stem

def find_event_for_file(events_dir, data_file_path):
    stem = stem_without_timestamp(data_file_path)
    base_name = os.path.splitext(os.path.basename(data_file_path))[0]
    candidates = [
        os.path.join(events_dir, f"{base_name}_emg.event"),
        os.path.join(events_dir, f"{base_name}.event"),
        os.path.join(events_dir, f"{stem}_emg.event"),
        os.path.join(events_dir, f"{stem}.event"),
        os.path.join(events_dir, f"{base_name}_emg.txt"),
        os.path.join(events_dir, f"{base_name}.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c): return c
    parent = os.path.dirname(data_file_path)
    if os.path.isdir(parent):
        siblings = glob.glob(os.path.join(parent, f"{stem}*.event")) + glob.glob(os.path.join(parent, f"{stem}*.txt"))
        if siblings: return siblings[0]
    return None

def discover_and_group_files(root_dir, file_type="rhd", file_names=None, exclude_pattern=None, merge_pattern=None):
    search_pattern = f"**/*.{file_type}"
    files = glob.glob(os.path.join(root_dir, search_pattern), recursive=True)
    if file_names:
        files = [f for f in files if os.path.basename(f) in file_names]
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in os.path.basename(f)]
    files = sorted(files)
    groups = {}
    for f in files:
        stem = stem_without_timestamp(f)
        if stem not in groups: groups[stem] = []
        groups[stem].append(f)
    return groups

def load_files_merged(file_type, files, root_dir, verbose=False):
    pass

def find_oebin_files(directory: Path | str) -> list[Path]:
    """Recursively find .oebin files."""
    d = Path(directory)
    if not d.is_dir():
        d = d.parent
    return list(d.rglob("*.oebin"))