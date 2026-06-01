"""
Convert an Open Ephys ``.oebin`` recording to NPZ and/or DEMUSE MAT files.

Workflow
--------
1. Select a ``.oebin`` file (CLI argument or file-picker dialog).
2. Answer a few dialog boxes:
     * Split the unipolar EMG channels into separate files? (default: yes,
       64 EMG channels per file). ADC/AUX channels are kept in *every* file.
     * Save a ``.npz`` file?
     * Save a DEMUSE ``.mat`` file?
3. Files are named after the *custom* recording folder (the folder that
   contains ``Record Node NNN/...``) and written into ``npz/`` and ``mat/``
   sub-folders next to that custom folder.

Example layout::

    <base>/fdi-sweep/Record Node 101/experiment1/recording1/structure.oebin
      -> <base>/npz/fdi-sweep_emg_data.npz
      -> <base>/mat/fdi-sweep_demuse.mat

Every dialog can be pre-answered from the command line (handy for scripting or
headless runs); supplying a flag skips the corresponding dialog.
"""
import argparse
import os
import re
import sys
from datetime import datetime
from math import gcd

import numpy as np
from pyoephys.io import load_oebin_file, prompt_file, prompt_text, prompt_yes_no
from pyoephys.io._npz_utils import save_as_npz

DEFAULT_EMG_PER_FILE = 64


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
    parser = argparse.ArgumentParser(
        description="Convert an Open Ephys recording into NPZ / DEMUSE MAT files."
    )
    parser.add_argument("path", nargs="?", help="Path to a .oebin file or recording folder.")
    parser.add_argument(
        "--downsample-to",
        type=float,
        default=None,
        help="Optional target sample rate in Hz for exported data, for example 4000.",
    )
    parser.add_argument(
        "--channels-per-file",
        type=int,
        default=None,
        help=(
            "Number of unipolar EMG channels per output file (ADC/AUX channels are "
            "always included). Use 0 to keep all channels in a single file. "
            "If omitted, a dialog asks (default 64)."
        ),
    )
    parser.add_argument("--npz", dest="npz", action="store_true", default=None, help="Save a .npz file (skips the dialog).")
    parser.add_argument("--no-npz", dest="npz", action="store_false", help="Do not save a .npz file (skips the dialog).")
    parser.add_argument("--mat", dest="mat", action="store_true", default=None, help="Save a DEMUSE .mat file (skips the dialog).")
    parser.add_argument("--no-mat", dest="mat", action="store_false", help="Do not save a DEMUSE .mat file (skips the dialog).")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never show dialogs; use CLI flags and defaults instead.",
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


def find_recording_layout(input_path: str) -> tuple[str, str]:
    """
    Resolve the *custom* recording folder name and the directory the export
    folders should live in.

    The Open Ephys directory layout is::

        <export_base>/<custom_name>/Record Node NNN/experimentX/recordingY/structure.oebin

    so the custom folder is the parent of the ``Record Node NNN`` directory and
    the export base is that custom folder's parent. If no ``Record Node`` folder
    is found, fall back to the folder that directly contains ``structure.oebin``.
    """
    if input_path.lower().endswith(".oebin"):
        recording_dir = os.path.dirname(input_path)
    else:
        recording_dir = input_path
    recording_dir = os.path.normpath(os.path.abspath(recording_dir))

    parts = recording_dir.split(os.sep)
    record_node_idx = None
    for idx, part in enumerate(parts):
        if re.match(r"(?i)record\s*node", part):
            record_node_idx = idx
            break

    if record_node_idx is not None and record_node_idx >= 1:
        custom_name = parts[record_node_idx - 1]
        export_base = os.sep.join(parts[:record_node_idx - 1]) or recording_dir
    else:
        custom_name = os.path.basename(recording_dir)
        export_base = os.path.dirname(recording_dir) or recording_dir

    return custom_name, export_base


def prepare_result_for_export(input_path: str, result: dict) -> tuple[dict, str, str]:
    custom_name, export_base = find_recording_layout(input_path)
    export_result = dict(result)
    export_result["source_path"] = export_base
    export_result["recording_name"] = custom_name

    timestamps = np.asarray(export_result.get("t_amplifier"), dtype=np.float64).reshape(-1)
    if timestamps.size:
        clock_start_offset_sec = float(timestamps[0])
        export_result["t_amplifier_original"] = timestamps.copy()
        export_result["t_amplifier"] = timestamps - clock_start_offset_sec
    else:
        clock_start_offset_sec = 0.0

    clock_start_date, clock_start_time, clock_start_iso = _parse_recording_datetime(custom_name)
    export_result["clock_start_offset_sec"] = np.float64(clock_start_offset_sec)
    export_result["clock_start_date"] = clock_start_date
    export_result["clock_start_time"] = clock_start_time
    export_result["clock_start_iso"] = clock_start_iso
    return export_result, export_base, custom_name


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


def _is_auxiliary_channel(name: str) -> bool:
    """ADC/AUX board channels are auxiliary (not unipolar EMG)."""
    upper = str(name).strip().upper()
    return upper.startswith("ADC") or upper.startswith("AUX")


def build_channel_subsets(result: dict, channels_per_file: int | None) -> list[tuple[str, dict]]:
    """
    Split the unipolar EMG channels into groups of ``channels_per_file`` while
    keeping every ADC/AUX channel in each group. Returns a list of
    ``(filename_suffix, subset_result)`` tuples.

    ``channels_per_file`` of ``None`` / ``0`` (or a value that covers all EMG
    channels) yields a single, unsplit result.
    """
    emg = np.asarray(result["amplifier_data"])
    n_channels = emg.shape[0]

    names = [str(n) for n in result.get("channel_names", [])]
    if len(names) != n_channels:
        names = [f"CH{i + 1}" for i in range(n_channels)]

    emg_idx = [i for i, name in enumerate(names) if not _is_auxiliary_channel(name)]
    aux_idx = [i for i, name in enumerate(names) if _is_auxiliary_channel(name)]

    if not channels_per_file or channels_per_file <= 0 or channels_per_file >= len(emg_idx):
        return [("", result)]

    base_name = str(result.get("recording_name", "recording"))
    subsets: list[tuple[str, dict]] = []
    for start in range(0, len(emg_idx), channels_per_file):
        chunk = emg_idx[start:start + channels_per_file]
        rows = chunk + aux_idx
        suffix = f"_emg{start + 1:03d}-{start + len(chunk):03d}"

        subset = dict(result)
        subset["amplifier_data"] = emg[rows, :]
        subset["channel_names"] = [names[i] for i in rows]
        subset["recording_name"] = base_name + suffix
        subsets.append((suffix, subset))

    return subsets


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


def _ask_yes_no(title: str, message: str, cli_value: bool | None, interactive: bool, default: bool) -> bool:
    if cli_value is not None:
        return cli_value
    if not interactive:
        return default
    try:
        return prompt_yes_no(title, message)
    except Exception:
        return default


def _ask_channels_per_file(cli_value: int | None, interactive: bool) -> int:
    if cli_value is not None:
        return max(0, cli_value)
    if not interactive:
        return DEFAULT_EMG_PER_FILE

    try:
        split = prompt_yes_no(
            "Split EMG channels",
            "Save the unipolar EMG channels in separate files?\n"
            f"(Default: {DEFAULT_EMG_PER_FILE} EMG channels per file. "
            "ADC/AUX channels are kept in every file.)",
        )
    except Exception:
        return DEFAULT_EMG_PER_FILE

    if not split:
        return 0

    try:
        text = prompt_text(
            "Channels per file",
            "Number of unipolar EMG channels per file:",
            str(DEFAULT_EMG_PER_FILE),
        )
    except Exception:
        return DEFAULT_EMG_PER_FILE

    if text and text.strip().lstrip("-").isdigit():
        return max(0, int(text.strip()))
    return DEFAULT_EMG_PER_FILE


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)
    interactive = not args.non_interactive

    path = resolve_oebin_path(args.path)
    result = load_oebin_file(path)
    result = downsample_result(result, args.downsample_to)

    print(result.keys())
    print(f"Shape of emg_data: {result['amplifier_data'].shape}")
    print(f"Sampling frequency: {result['sample_rate']} Hz")
    if 'board_adc_data' in result and len(result['board_adc_data']) > 0:
        print(f"Shape of board_adc_data: {result['board_adc_data'].shape}")
    print(f"Time vector: {result.get('t_amplifier')[:10]}...")

    export_result, export_base, custom_name = prepare_result_for_export(path, result)

    channels_per_file = _ask_channels_per_file(args.channels_per_file, interactive)
    save_npz = _ask_yes_no("Save NPZ", "Save a .npz file?", args.npz, interactive, default=True)
    save_mat = _ask_yes_no("Save MAT", "Save a DEMUSE .mat file?", args.mat, interactive, default=True)

    if not (save_npz or save_mat):
        print("No output format selected; nothing to save.")
        return

    subsets = build_channel_subsets(export_result, channels_per_file)

    npz_dir = os.path.join(export_base, "npz")
    mat_dir = os.path.join(export_base, "mat")

    print(
        f"Recording name: {custom_name} | export base: {export_base} | "
        f"{len(subsets)} file(s) | EMG/file: "
        f"{'all' if channels_per_file in (0, None) else channels_per_file}"
    )

    for suffix, subset in subsets:
        base = custom_name + suffix
        if save_npz:
            os.makedirs(npz_dir, exist_ok=True)
            save_as_npz(subset, os.path.join(npz_dir, f"{base}_emg_data.npz"))
        if save_mat:
            os.makedirs(mat_dir, exist_ok=True)
            save_as_demuse_mat(subset, os.path.join(mat_dir, f"{base}_demuse.mat"))


if __name__ == "__main__":
    main()
