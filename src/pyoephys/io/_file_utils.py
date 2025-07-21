import os
import time
from typing import Union
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from open_ephys.analysis import Session


def find_session_root(start_path: str) -> Union[str, None]:
    """
    Walk upward from the given path to find the session root directory
    that contains at least one 'Record Node XXX' folder.
    """
    current = os.path.abspath(start_path)
    while current != os.path.dirname(current):
        if any(name.startswith("Record Node") and os.path.isdir(os.path.join(current, name))
               for name in os.listdir(current)):
            return current
        current = os.path.dirname(current)
    return None


def load_oebin_file(path: str | None = None, verbose: bool = False) -> dict:

    if path is None:
        path = filedialog.askopenfilename(filetypes=[("OEBIN Files", "*.oebin")])
        if not path:
            return

    # Load OEBIN file
    if os.path.isfile(path):
        print(f"Loading OEBIN file from: {path}")
        result = load_open_ephys_session(os.path.dirname(path), verbose=verbose)
        print(f"OEBIN file loaded")
        if verbose:
            print(f"Data shape: {result['amplifier_data'].shape}")
            print(f"Sample rate: {result['sample_rate']} Hz")
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

    path = os.path.abspath(path)
    session_root = find_session_root(path)

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
    samples = stream.get_samples(start_sample_index=0, end_sample_index=num_samples).T  # (n_channels, n_samples)

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
        "channel_names": channel_names if channel_names else [],
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
