import os
import yaml
import json
import time
from typing import Union, Optional
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd
from open_ephys.analysis import Session


def find_session_root(path) -> Path:
    """
    Return a directory suitable for open_ephys.analysis.Session(...):
      - a directory named "Record Node *", OR
      - a directory that directly contains one or more "Record Node *" folders.

    Accepts:
      - a path to ANY directory above/below the recording, OR
      - a direct path to a *.oebin file.

    Strategy:
      1) If a file path is given and it's *.oebin -> start from its parent.
      2) Walk UP: if any ancestor is a "Record Node *" dir -> return that.
      3) If not found, search DOWN from the starting directory for a "Record Node *" dir.
      4) If still not found, but the starting directory directly contains any "Record Node *",
         return the starting directory (Session accepts a parent that contains nodes).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # If a file was passed (e.g., structure.oebin), treat its parent as the start
    if p.is_file():
        if p.suffix.lower() == ".oebin":
            p = p.parent
        else:
            p = p.parent

    # 2) Walk UP to find an ancestor named "Record Node *"
    for anc in [p] + list(p.parents):
        name = anc.name.lower()
        if name.startswith("record node"):
            # return the parent directory
            if anc.parent.is_dir():
                return anc.parent

    # 3) Search DOWN for a "Record Node *" directory
    rn = next((d for d in p.rglob("*") if d.is_dir() and d.name.lower().startswith("record node")), None)
    if rn is not None:
        return rn.parent

    # 4) If current dir directly contains any "Record Node *" children, return the current dir
    children = [d for d in p.iterdir() if d.is_dir() and d.name.lower().startswith("record node")]
    if children:
        return p.parent

    raise FileNotFoundError(f"Could not find a 'Record Node *' directory under or above: {p}")


def find_oebin_files(start_path: str) -> list:
    """
    Recursively search for Open Ephys session files (.oebin) starting from the given path.
    Returns a list of paths to valid .oebin files.
    """
    oebin_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith('.oebin'):
                oebin_files.append(os.path.join(root, file))
    return oebin_files


def load_oebin_file(path: str | None = None, verbose: bool = False) -> dict:
    if path is None:
        path = filedialog.askopenfilename(filetypes=[("OEBIN Files", "*.oebin")])
        if not path:
            return

    # Load OEBIN file
    if os.path.isdir(path):
        # if verbose:
        #    print(f"Searching for OEBIN files in directory: {path}")
        oebin_files = find_oebin_files(path)
        if not oebin_files:
            print("No OEBIN files found in the specified directory.")
            return {}
        path = oebin_files[0]

    if os.path.isfile(path):
        # print(f"|  Loading OEBIN file from: {path}")
        result = load_open_ephys_session(os.path.dirname(path), verbose=verbose)
        if verbose:
            pass
            # print(f"|  Data shape: {result['amplifier_data'].shape}")
            # print(f"|  Sample rate: {result['sample_rate']} Hz")
        return result
    else:
        print(f"File not found: {path}")
        return {}


def load_open_ephys_session(path: str | None = None, verbose: bool = False) -> dict:
    """
    Load Open Ephys session data from any depth: root directory, Record Node, or subfolder.

    Parameters:
        path (str | None): Path to file or folder. If None, opens a dialog.
        verbose (bool): Print session info.

    Returns:
        dict with keys:
        - amplifier_data: np.ndarray (n_channels, n_samples)
        - t_amplifier: np.ndarray (n_samples,)
        - sample_rate: float
        - recording_name: str
        - num_channels: int
        - file_type: 'open-ephys'
        - source_path: str
        - info: raw session metadata
    """
    if path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select Open Ephys Session Folder")
        if not path:
            print("No session selected.")
            return {}

    session_root = str(find_session_root(os.path.abspath(path)))
    print(f"Searching for Open Ephys session in: {session_root}")
    if verbose:
        print(f"Session root found: {session_root}")

    if session_root is None:
        print("No valid Open Ephys session found.")
        raise FileNotFoundError(f"No valid Open Ephys session found in {path}")


    session = Session(session_root)
    recording = session.recordnodes[0].recordings[0]
    stream = recording.continuous[0]

    if verbose:
        print(f"Format: {recording.format}")
        print(f"Directory: {recording.directory}")
        print(f"Number of continuous streams: {len(recording.continuous)}")
        print(f"Number of events: {len(recording.events) if recording.events is not None else 0}")
        print(f"Number of spikes: {len(recording.spikes) if recording.spikes is not None else 0}")

    # Get number of samples and channels
    num_samples = stream.samples.shape[0]
    num_channels = stream.samples.shape[1]

    # Get channel indices of all channels names starting with CH
    ch_indices = []
    adc_indices = []
    channel_names = stream.metadata.get('channel_names')
    if verbose:
        print(f"Channel names: {channel_names}")
    if channel_names:
        ch_indices = [i for i, name in enumerate(channel_names) if name.startswith('CH')]
        if not ch_indices:
            print("No amplifier channels found ")

        adc_indices = [i for i, name in enumerate(channel_names) if name.startswith('ADC')]
        if not adc_indices:
            print("No ADC channels found")

    # Load samples and timestamps
    samples = np.ascontiguousarray(
        stream.get_samples(start_sample_index=0, end_sample_index=num_samples).T,
        dtype=np.float32,
    ) # (C, S), C-order

    timestamps = stream.timestamps[:num_samples]
    sample_rate = recording.info['continuous'][0]['sample_rate']

    return {
        "amplifier_data": samples[ch_indices, :] if ch_indices else [],
        "board_adc_data": samples[adc_indices, :] if adc_indices else [],
        "t_amplifier": timestamps,
        "sample_rate": sample_rate,
        "recording_name": os.path.basename(recording.directory),
        "num_channels": num_channels,
        "file_type": "open-ephys",
        "source_path": session_root,
        "info": recording,
        "channel_names": channel_names,
    }


def find_valid_session_paths(base_dir, extension=".oebin"):
    """
    Recursively search for Open Ephys sessions that contain a metadata file (e.g., structure.oebin).

    Returns a list of valid session directories.
    """
    valid_paths = []
    for root, dirs, files in os.walk(base_dir):
        if any(file.endswith(extension) for file in files):
            valid_paths.append(root)
    return valid_paths


def parse_event_file(event_files, verbose=False):
    """
    Extract all events from event file(s) and return a combined DataFrame.

    Args:
        event_files (str or list): Path(s) to the event file(s).
        verbose (bool): If True, print debug information.

    Returns:
        pd.DataFrame: DataFrame containing all events with columns:
            - 'sample_index': Sample index of the event (int)
            - 'timestamp': Timestamp string (str or None if missing)
            - 'label': Cleaned label string (str)
    """
    if isinstance(event_files, str):
        event_files = [event_files]

    all_events = []

    for file_path in event_files:
        if verbose:
            print(f"> Parsing event file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Check for header
            first_line = lines[0].strip().lower()
            has_header = any(h in first_line for h in ['timestamp', 'label', 'sample'])

            if has_header:
                lines = lines[1:]

            for line_num, line in enumerate(lines, start=2 if has_header else 1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                try:
                    parts = [p.strip() for p in line.split(',')]

                    if len(parts) < 2:
                        if verbose:
                            print(f"  Skipping malformed line {line_num}: {line}")
                        continue

                    sample_index = int(parts[0])
                    if len(parts) == 2:
                        timestamp = None
                        label = parts[1].split('#')[0].strip()
                    else:
                        timestamp = parts[1]
                        label = parts[2].split('#')[0].strip()

                    all_events.append({
                        'sample_index': sample_index,
                        'timestamp': timestamp,
                        'label': label
                    })

                except Exception as e:
                    if verbose:
                        print(f"  Error parsing line {line_num} in {file_path}: {e}")

        except Exception as e:
            if verbose:
                print(f"Failed to read {file_path}: {e}")

    return pd.DataFrame(all_events)


def get_start_indices_from_events_file(file_path):
    """
    Extract Start and End indices from a single event file, assumes .txt ending
    and a specific format with 'Start' and 'End' labels.

    Parameters:
        file_path (str): Path to the event file.

    Returns:
        start_idx (int): Start index for the EMG data.
        end_idx (int): End index for the EMG data.

    """
    try:
        with open(file_path, 'r') as f:
            start_idx, end_idx = None, None
            # Read each line and extract indices
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue  # Skip malformed lines
                sample_index_str, _, label = parts
                label = label.strip().lower()
                if label == 'start' and start_idx is None:
                    start_idx = int(sample_index_str)
                elif label == 'end' and end_idx is None:
                    end_idx = int(sample_index_str)

            return start_idx, end_idx

    except Exception as e:
        print(f"[ERROR] Failed to parse events file {file_path}: {e}")
        return None, None


def load_txt_config(file_path=None, verbose=False):
    """
    Parse a simple key=value style configuration file (e.g. config.txt).

    Parameters:
        file_path (str): Path to the config file.
        verbose (bool): If True, print warnings and info messages.

    Returns:
        dict: Dictionary of key-value settings.

    """

    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File", filetypes=[("Text files", "*.txt")])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    # Dictionary to store the key-value pairs
    config_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value at the first '='
            key, value = line.split('=', 1)
            config_data[key.strip()] = value.strip()
    return config_data


def load_yaml_file(file_path=None):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed config dictionary.

    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File",
                                               filetypes=[("Text files", ["*.yaml", "*.yml"])])
        if not file_path:
            return None

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_json_file(file_path=None):
    """
    Load configuration from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed config dictionary.

    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Notes File",
                                               filetypes=[("Text files", ["*.json"])])
        if not file_path:
            return None

    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def load_config_file(file_path=None, verbose=False):
    """
    Load configuration from a file, supporting .txt, .yaml, and .json formats.

    Args:
        file_path (str): Path to the config file.
        verbose (bool): If True, print debug information.

    Returns:
        dict: Parsed config dictionary or None if loading failed.
    """
    if file_path is None:
        file_path = filedialog.askopenfilename(title="Select Config File",
                                               filetypes=[("Text files", ["*.txt", "*.yaml", "*.yml", "*.json"])])
        if not file_path:
            if verbose:
                print("Cancelled selection")
            return None

    ext = Path(file_path).suffix.lower()
    if ext == '.txt':
        return load_txt_config(file_path, verbose)
    elif ext in ['.yaml', '.yml']:
        return load_yaml_file(file_path)
    elif ext == '.json':
        return load_json_file(file_path)
    else:
        print(f"Unsupported config file format: {ext}")
        return None


def labels_from_events(event_path, window_starts, *, strict_segment=False, fs=2000):
    """
    Map each window start (absolute sample index) to a label using an events CSV
    with columns: 'Sample Index', 'Timestamp', 'Label'. The Timestamp text is ignored.

    strict_segment=True: drops any window whose [start, start+step) crosses an event boundary.
    """
    df = pd.read_csv(event_path)
    if 'Sample Index' not in df.columns or 'Label' not in df.columns:
        raise ValueError("Event file must have 'Sample Index' and 'Label' columns")
    ev_idx = np.asarray(df['Sample Index'].values, dtype=np.int64)
    ev_lab = df['Label'].astype(str).values

    # Trim any '#' comments from labels
    ev_lab = np.array([lab.split('#')[0].strip() for lab in ev_lab], dtype=str)

    # sort by sample index
    order = np.argsort(ev_idx)
    ev_idx = ev_idx[order]
    ev_lab = ev_lab[order]

    # for each window start, pick the last event with idx <= start
    # searchsorted returns insertion position; subtract 1 to get the last <=
    pos = np.searchsorted(ev_idx, window_starts, side='right') - 1
    # anything before the first event is Unknown
    y = np.where(pos >= 0, ev_lab[pos], 'Unknown')

    if strict_segment:
        # drop windows that cross an event boundary
        # boundary after this window's start is at ev_idx[pos+1]
        next_pos = pos + 1
        next_change = np.where(next_pos < ev_idx.size, ev_idx[next_pos], np.iinfo(np.int64).max)
        # if your step in samples is known, ensure start+step <= next_change
        # (pass it in or compute the exact end you want). As a simple guard:
        # keep only windows whose label doesn't change immediately after start.
        ok = (next_change > window_starts)
        y = np.where(ok, y, 'Unknown')
    return y


def last_event_index(path: str) -> Optional[int]:
    if not os.path.isfile(path): return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.lower().startswith("sample"):
                continue
            try:
                idx = int(ln.split()[0].replace(",", ""))
                last = idx if last is None else max(last, idx)
            except Exception:
                pass
    return last


def load_npz_file(file_path: str | None = None, verbose: bool = False) -> dict:
    """
    Load data from a .npz file.

    Parameters:
        file_path (str | None): Path to the .npz file. If None, opens a file dialog.
        verbose (bool): If True, print debug information.

    Returns:
        dict: Dictionary containing the loaded data.
    """
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select NPZ File", filetypes=[("NPZ Files", "*.npz")])
        if not file_path:
            print("No file selected.")
            return {}

    if verbose:
        print(f"Loading NPZ file from: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=True)
        result = {key: data[key] for key in data.files}
        if verbose:
            print(f"Loaded keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"Failed to load NPZ file: {e}")
        return {}