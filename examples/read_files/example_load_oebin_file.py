"""
Demo that shows how to load data from a .oebin file using pyoephys.
"""
import argparse
import os
import re
import sys
from datetime import datetime
from math import gcd

import numpy as np
from pyoephys.io import load_oebin_file, prompt_file
from pyoephys.io._npz_utils import save_as_npz


def _as_matlab_cell_row(values: list[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object).reshape(1, -1)


def _parse_recording_datetime(recording_name: str) -> tuple[str, str, str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", recording_name)
    if not match:
        return "", "", ""

    date_part, time_part = match.groups()
    iso_time = time_part.replace("-", ":")
    iso_value = f"{date_part}T{iso_time}"
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return date_part, iso_time, ""
    return parsed.date().isoformat(), parsed.time().isoformat(), parsed.isoformat()


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an Open Ephys recording and export NPZ / MAT files.")
    parser.add_argument("path", nargs="?", help="Path to a .oebin file or recording folder.")
    parser.add_argument(
        "--downsample-to",
        type=float,
        default=None,
        help="Optional target sample rate in Hz for exported data, for example 4000.",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def resolve_oebin_path(path_arg: str | None = None) -> str:
    if path_arg:
        return path_arg

    path = prompt_file(
        title="Select a .oebin file",
        filetypes=[("Open Ephys metadata", "*.oebin"), ("All files", "*.*")],
    )
    if not path:
        raise SystemExit("No .oebin file selected.")
    return path


def resolve_export_location(input_path: str, result: dict) -> tuple[str, str]:
    recording_root = result.get("source_path")
    if not recording_root:
        recording_root = os.path.dirname(input_path) if input_path.lower().endswith(".oebin") else input_path

    recording_root = os.path.normpath(recording_root)
    export_root = os.path.dirname(recording_root) or recording_root
    label = os.path.basename(recording_root) or str(result.get("recording_name", "open_ephys_recording"))
    return export_root, label


def prepare_result_for_export(input_path: str, result: dict) -> tuple[dict, str, str]:
    export_root, label = resolve_export_location(input_path, result)
    export_result = dict(result)
    export_result.setdefault("source_path", export_root)
    export_result.setdefault("recording_name", label)

    timestamps = np.asarray(export_result.get("t_amplifier"), dtype=np.float64).reshape(-1)
    if timestamps.size:
        clock_start_offset_sec = float(timestamps[0])
        export_result["t_amplifier_original"] = timestamps.copy()
        export_result["t_amplifier"] = timestamps - clock_start_offset_sec
    else:
        clock_start_offset_sec = 0.0

    clock_start_date, clock_start_time, clock_start_iso = _parse_recording_datetime(export_result["recording_name"])
    export_result["clock_start_offset_sec"] = np.float64(clock_start_offset_sec)
    export_result["clock_start_date"] = clock_start_date
    export_result["clock_start_time"] = clock_start_time
    export_result["clock_start_iso"] = clock_start_iso
    return export_result, export_root, label


def downsample_result(result: dict, target_sample_rate: float | None) -> dict:
    if target_sample_rate is None:
        return result

    if not isinstance(result, dict):
        raise ValueError("Input must be a dictionary containing Open Ephys session data.")
    if "amplifier_data" not in result or "sample_rate" not in result:
        raise KeyError("Result must contain amplifier_data and sample_rate to downsample.")

    original_rate = float(result["sample_rate"])
    target_rate = float(target_sample_rate)
    if target_rate <= 0:
        raise ValueError("--downsample-to must be greater than 0.")
    if target_rate > original_rate:
        raise ValueError("--downsample-to must be less than or equal to the original sample rate.")
    if np.isclose(target_rate, original_rate):
        return result

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("scipy is required to downsample recordings.") from exc

    rate_scale = 1000
    up = int(round(target_rate * rate_scale))
    down = int(round(original_rate * rate_scale))
    factor = gcd(up, down)
    up //= factor
    down //= factor

    downsampled = dict(result)
    emg = np.asarray(result["amplifier_data"], dtype=np.float32)
    downsampled["amplifier_data"] = resample_poly(emg, up, down, axis=1).astype(np.float32, copy=False)
    downsampled["sample_rate"] = target_rate

    timestamps = np.asarray(result.get("t_amplifier"), dtype=np.float64).reshape(-1)
    if timestamps.size:
        start_time = float(timestamps[0])
    else:
        start_time = 0.0
    downsampled["t_amplifier"] = start_time + (np.arange(downsampled["amplifier_data"].shape[1], dtype=np.float64) / target_rate)

    board_adc = result.get("board_adc_data")
    if board_adc is not None and np.size(board_adc) > 0:
        board_adc_arr = np.asarray(board_adc, dtype=np.float32)
        downsampled["board_adc_data"] = resample_poly(board_adc_arr, up, down, axis=1).astype(np.float32, copy=False)

    return downsampled


def save_as_demuse_mat(
    result: dict,
    file_path: str | None = None,
    *,
    grid_name: str = "OpenEphysGrid",
    muscle_name: str = "Unspecified",
) -> str:
    """Save Open Ephys EMG data in a MATLAB struct layout used by MU decomposition tools."""
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError("scipy is required to export DEMUSE MATLAB files.") from exc

    required_keys = ["amplifier_data", "sample_rate", "recording_name"]
    if not isinstance(result, dict):
        raise ValueError("Input must be a dictionary containing Open Ephys session data.")
    if not all(key in result for key in required_keys):
        raise KeyError(f"Missing one of the required keys: {required_keys}")

    emg = np.asarray(result["amplifier_data"], dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError("amplifier_data must be a 2D array with shape (n_channels, n_samples).")

    fs_hz = float(result["sample_rate"])
    timestamps = np.asarray(result.get("t_amplifier"), dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        timestamps = np.arange(emg.shape[1], dtype=np.float64) / fs_hz

    raw_channel_names = result.get("channel_names")
    if raw_channel_names is None:
        channel_names = [f"CH{index + 1}" for index in range(emg.shape[0])]
    else:
        channel_names = [str(name) for name in raw_channel_names]
    channel_names_cell = _as_matlab_cell_row(channel_names)

    signal_struct = {
        "data": emg,
        "fsamp": fs_hz,
        "nChan": int(emg.shape[0]),
        "ngrid": 1,
        "gridname": _as_matlab_cell_row([grid_name]),
        "muscle": _as_matlab_cell_row([muscle_name]),
        "channel_names": channel_names_cell,
        "timestamps": timestamps,
        "source_path": str(result.get("source_path", "")),
    }

    board_adc = result.get("board_adc_data")
    if board_adc is not None and np.size(board_adc) > 0:
        reference = np.asarray(board_adc, dtype=np.float32)
        if reference.ndim == 2:
            reference = reference[0]
        signal_struct["target"] = reference.reshape(-1)
        signal_struct["path"] = reference.reshape(-1)

    payload = {
        "signal": signal_struct,
        "amplifier_data": emg,
        "sample_rate": fs_hz,
        "channel_names": channel_names_cell,
        "t_amplifier": timestamps,
        "t_amplifier_original": np.asarray(result.get("t_amplifier_original", timestamps), dtype=np.float64).reshape(-1),
        "clock_start_offset_sec": np.float64(result.get("clock_start_offset_sec", 0.0)),
        "clock_start_date": str(result.get("clock_start_date", "")),
        "clock_start_time": str(result.get("clock_start_time", "")),
        "clock_start_iso": str(result.get("clock_start_iso", "")),
        "source_path": str(result.get("source_path", "")),
        "recording_name": str(result["recording_name"]),
        "SIG": emg,
        "fsamp": fs_hz,
        "timestamps": timestamps,
    }

    if board_adc is not None and np.size(board_adc) > 0:
        payload["board_adc_data"] = np.asarray(board_adc, dtype=np.float32)

    if file_path is None:
        file_path = result["recording_name"] + "_demuse.mat"
    elif not file_path.endswith(".mat"):
        file_path += ".mat"

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    print(f" Saving DEMUSE MAT data to {file_path}...")
    savemat(file_path, payload, do_compression=True, long_field_names=True)
    print(f"DEMUSE MAT data saved to {file_path}")
    return file_path

if __name__ == "__main__":

    save_npz = True
    save_demuse = True

    # ================ Load the data ================
    #path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\Dynamic5kHz\Record Node 101\experiment2\recording1\structure.oebin'
    #path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\raw\DynamicFingers\Record Node 105\experiment1\recording1\structure.oebin'
    #path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\Dynamic1kHz\Record Node 101\experiment1\recording1\structure.oebin"
    #path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw"
    
    args = parse_cli_args()
    path = resolve_oebin_path(args.path)
    result = load_oebin_file(path)
    result = downsample_result(result, args.downsample_to)

    # ================ Print some info ================
    print(result.keys())
    print(f"Shape of emg_data: {result['amplifier_data'].shape}")
    print(f"Sampling frequency: {result['sample_rate']} Hz")
    if 'board_adc_data' in result and len(result['board_adc_data']) > 0:
        print(f"Shape of board_adc_data: {result['board_adc_data'].shape}")
    print(f"Time vector: {result.get('t_amplifier')[:10]}...")  # Print first 10 timestamps

    # =========== Save the pyoephys data to a numpy format ===========
    export_result, export_root, label = prepare_result_for_export(path, result)
    if save_npz:
        save_as_npz(export_result, os.path.join(export_root, f"{label}_emg_data.npz"))
    if save_demuse:
        save_as_demuse_mat(export_result, os.path.join(export_root, f"{label}_demuse.mat"))
