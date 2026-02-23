#!/usr/bin/env python3
"""
1_build_dataset.py — EMG Gesture Dataset Builder
=================================================
Reads raw EMG data, extracts windowed features, and saves a labelled
training dataset as a .npz file ready for 2_train_model.py.

Supported input formats
-----------------------
  CSV   — timestamp + channel columns (+ companion labels.csv)
  .npz  — keys: emg (C×S float32), timestamps, fs_hz
  .oebin — Open Ephys binary recording

Default behaviour (zero arguments)
-----------------------------------
  Loads  ./data/gestures/   (Open Ephys recording folder)
  Auto-discovers a label file within it (see below)
  Saves  ./data/gestures/training_dataset.npz

Usage
-----
  # Point at your Open Ephys recording:
  python 1_build_dataset.py --data_path data/gestures

  # Custom CSV + labels:
  python 1_build_dataset.py \\
      --data_path  data/my_emg.csv \\
      --labels_path data/my_labels.csv

  # Open Ephys recording:
  python 1_build_dataset.py \\
      --data_path  data/session_1/structure.oebin \\
      --labels_path data/session_1/labels.csv

  # Override window/step:
  python 1_build_dataset.py --window_ms 200 --step_ms 50

  # Ignore rest segments:
  python 1_build_dataset.py --ignore_labels rest

  # Auto-drop bad channels (quiet-segment QC):
  python 1_build_dataset.py --auto_select_channels

  # Explicit quiet/rest window for QC (first 10 s):
  python 1_build_dataset.py --auto_select_channels --qc_quiet_sec 0 10

  # Tune the noise threshold (µV RMS during rest):
  python 1_build_dataset.py --auto_select_channels --noise_threshold 20

  # Auto-select channels, overwrite any existing dataset, and ignore start labels:
  python 1_build_dataset.py --auto_select_channels --overwrite --ignore_labels Start

Labels file format (CSV / TXT)
------------------------------
  Two columns: "Sample Index" and "Label".
  Each row marks the start of a new epoch (transition format).

  Sample Index,Label
  0,rest
  400,fist
  800,rest
  ...

Label file auto-discovery
-------------------------
  If --labels_path is not provided the script searches for a label file
  next to (or inside) the data folder in this priority order:

    {recording_name}_emg.txt   {recording_name}.txt
    {recording_name}_emg.event {recording_name}.event
    labels.csv                 events.csv
    emg.txt                    labels.txt

  Both "labels" and "emg" variants are accepted so that files named
  ``emg.txt``, ``labels.csv``, ``session1_emg.txt`` etc. are all found
  without any extra flags.
  (Implemented in pyoephys.io.find_event_for_file)
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from scipy.signal import iirnotch, butter, sosfiltfilt, tf2sos

from pyoephys.io import (
    load_simple_config,
    load_open_ephys_session,
    process_recording,
    save_dataset,
)

# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_dataset(
    data_path: str,
    labels_path: str | None = None,
    save_path: str | None = None,
    window_ms: int = 200,
    step_ms: int = 50,
    channels: list | None = None,
    ignore_labels: list | None = None,
    auto_select_channels: bool = False,
    noise_threshold_uv: float = 30.0,
    qc_quiet_sec: tuple[float, float] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """
    Full dataset-build pipeline.

    Parameters
    ----------
    data_path            : Path to EMG data (.csv, .npz, or .oebin).
    labels_path          : Path to a labels/events file (CSV or TXT, transition
                           format).  Auto-detected if None — the script searches
                           for labels.csv, events.csv, emg.txt, labels.txt, or
                           {recording}_emg.txt next to the data folder.
    save_path            : Output .npz path (auto-derived if None).
    window_ms            : Feature window length in milliseconds.
    step_ms              : Window step size in milliseconds.
    channels             : List of channel indices to keep (None = all).
    ignore_labels        : Label names to exclude (e.g. ['rest']).
    auto_select_channels : If True, automatically drop channels flagged bad by QC.
                           Ignored when ``channels`` is set explicitly.
    noise_threshold_uv   : RMS threshold (µV, after bandpass+notch) applied to
                           the quiet segment.  During rest a good channel should
                           be near-zero; anything above this is artifact/noise.
                           Default: 30 µV.
    qc_quiet_sec         : Optional (start_sec, end_sec) tuple defining an explicit
                           quiet/rest window to use for QC instead of auto-detecting
                           the bottom 20 % of the recording.  Example: (0, 10) uses
                           the first 10 seconds.  Default: None (auto-detect).
    overwrite            : Overwrite existing output if True.
    verbose              : Enable DEBUG logging.
    """
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )

    data_path = str(data_path)

    # Derive output path from data path when not specified
    if save_path is None:
        data_dir = (
            Path(data_path).parent if Path(data_path).is_file() else Path(data_path)
        )
        save_path = str(data_dir / "training_dataset.npz")

    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists: {save_path}  (use --overwrite to rebuild)")
        return

    # ── Load ──────────────────────────────────────────────────────────────
    logging.info(f"Loading data from: {data_path}")
    data = load_open_ephys_session(data_path)

    fs = data["sample_rate"]
    n_ch = data["amplifier_data"].shape[0]
    n_samples = data["amplifier_data"].shape[1]
    logging.info(f"  {n_ch} channels | {n_samples} samples | fs={fs:.0f} Hz")

    # ── Channel QC (quiet-segment method) ──────────────────────────────
    # 1. Bandpass (10–500 Hz) + 60 Hz notch on the full recording.
    # 2. Slide 500 ms non-overlapping windows; rank by mean-across-channels
    #    RMS.  Bottom 20 % = rest-like / quiet segments.
    # 3. Per-channel RMS on those quiet windows:
    #      dead  : RMS < 0.5 µV  (open-circuit, no signal anywhere)
    #      noisy : RMS > noise_threshold_uv  (artifact during rest – should
    #              be near-zero on a good electrode)
    # Also catches dead channels via a whole-recording floor check.
    nyq = fs / 2.0
    b_notch, a_notch = iirnotch(w0=60.0, Q=30.0, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    sos_bp    = butter(4, [10.0 / nyq, min(500.0, nyq * 0.98) / nyq],
                       btype="band", output="sos")
    emg_filt  = sosfiltfilt(
        sos_bp, sosfiltfilt(sos_notch, data["amplifier_data"], axis=1), axis=1
    )  # shape: (n_ch, n_samples)

    win_samp = int(fs * 0.5)   # 500 ms windows
    n_wins   = n_samples // win_samp
    if n_wins < 5:
        # fallback: not enough data to segment — use whole-recording RMS
        rms_quiet = np.sqrt(np.mean(emg_filt ** 2, axis=1))
        logging.warning("  QC: recording too short for quiet-segment detection; using full RMS.")
    else:
        if qc_quiet_sec is not None:
            # explicit quiet window provided by the caller
            t_start, t_end = qc_quiet_sec
            s_start = max(0, int(t_start * fs))
            s_end   = min(n_samples, int(t_end * fs))
            if s_end <= s_start:
                logging.warning("  QC: --qc_quiet_sec range is empty; falling back to auto-detect.")
                qc_quiet_sec = None
            else:
                quiet_seg  = emg_filt[:, s_start:s_end]
                rms_quiet  = np.sqrt(np.mean(quiet_seg ** 2, axis=1))
                dur = (s_end - s_start) / fs
                logging.info(f"  QC: using explicit quiet window {t_start:.1f}–{t_end:.1f}s "
                             f"({dur:.1f}s, {s_end - s_start} samples)")

        if qc_quiet_sec is None:
            # auto-detect bottom 20 % of windows by global RMS
            wins = emg_filt[:, : n_wins * win_samp].reshape(n_ch, n_wins, win_samp)
            global_rms = np.sqrt(np.mean(wins ** 2, axis=(0, 2)))  # (n_wins,)
            thresh_pct = np.percentile(global_rms, 20)
            quiet_mask = global_rms <= thresh_pct                   # bottom 20 %
            n_quiet    = int(quiet_mask.sum())
            logging.info(f"  QC: using {n_quiet}/{n_wins} quiet windows "
                         f"(≤20th pct global RMS {thresh_pct:.1f} µV) for channel assessment")
            quiet_data = wins[:, quiet_mask, :]          # (n_ch, n_quiet, win_samp)
            rms_quiet  = np.sqrt(np.mean(quiet_data ** 2, axis=(1, 2)))  # (n_ch,)

    dead_rms_uv = 0.5
    bad_channels = {
        i for i in range(n_ch)
        if rms_quiet[i] < dead_rms_uv or rms_quiet[i] > noise_threshold_uv
    }
    good_channels = sorted(set(range(n_ch)) - bad_channels)
    logging.info(
        f"  QC: {len(good_channels)} good, {len(bad_channels)} bad  "
        f"(median quiet-RMS {np.median(rms_quiet):.1f} µV, "
        f"thresholds: dead<{dead_rms_uv} µV, noisy>{noise_threshold_uv:.0f} µV)"
    )
    if bad_channels:
        logging.warning(f"  Bad channels: {sorted(bad_channels)}")
        if not auto_select_channels:
            logging.warning(
                "  Pass --auto_select_channels to drop them automatically."
            )
    if auto_select_channels and channels is None:
        channels = good_channels
        logging.info(f"  Auto-selected {len(channels)} good channels.")

    # ── Labels ────────────────────────────────────────────────────────────
    # Honour an explicit path; otherwise process_recording auto-discovers
    # labels.csv / events.csv next to the data file via find_event_for_file.
    events_file = labels_path if (labels_path and Path(labels_path).is_file()) else None
    if events_file:
        logging.info(f"Using labels: {events_file}")

    # ── Build features ────────────────────────────────────────────────────
    root_dir = str(Path(data_path).parent if Path(data_path).is_file() else Path(data_path))

    X, y, meta = process_recording(
        data=data,
        file_path=data_path,
        root_dir=root_dir,
        events_file=events_file,
        window_ms=window_ms,
        step_ms=step_ms,
        channels=channels,
        ignore_labels=ignore_labels,
        ignore_case=True,
        keep_trial=False,
    )

    if len(X) == 0:
        logging.error("No windows extracted — check your data and labels paths.")
        return

    class_counts = {c: int(np.sum(y == c)) for c in sorted(set(y))}
    logging.info(f"Extracted {len(X)} windows — classes: {class_counts}")

    # ── Save ──────────────────────────────────────────────────────────────
    save_dataset(save_path, X, y, meta, window_ms, step_ms, ignore_labels=ignore_labels)
    logging.info(f"Dataset saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_DEFAULT_DATA   = _HERE / "data" / "gestures"
_DEFAULT_LABELS = _HERE / "data" / "labels.csv"
_DEFAULT_SAVE   = _HERE / "data" / "training_dataset.npz"
_CONFIG_FILE    = _HERE / ".gesture_config"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build a windowed-feature EMG dataset from raw EMG data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data_path", default=None,
        help=f"Path to EMG data (CSV / .npz / .oebin).  Default: {_DEFAULT_DATA}",
    )
    p.add_argument(
        "--labels_path", default=None,
        help="Path to a labels/events file (CSV or TXT, transition format).  "
             "If omitted the script auto-discovers a file named labels.csv, "
             "events.csv, emg.txt, labels.txt, or {recording}_emg.txt next to "
             "the data folder.",
    )
    p.add_argument(
        "--save_path", default=None,
        help=f"Output .npz file.  Default: {_DEFAULT_SAVE}",
    )
    p.add_argument("--window_ms", type=int, default=None, help="Feature window in ms (default: 200)")
    p.add_argument("--step_ms",   type=int, default=None, help="Window step in ms (default: 50)")
    p.add_argument(
        "--channels", type=int, nargs="+", default=None,
        help="Channel indices to keep (default: all).  E.g. --channels 0 1 2 3",
    )
    p.add_argument(
        "--ignore_labels", nargs="+", default=None,
        help="Labels to exclude (e.g. --ignore_labels rest unknown)",
    )
    p.add_argument(
        "--noise_threshold", type=float, default=30.0,
        help="RMS threshold (µV, after bandpass+notch) applied to the quietest 20%% "
             "of the recording.  Good channels should be near-zero during rest; "
             "anything above this is flagged as noisy/artifact. (default: 30)",
    )
    p.add_argument(
        "--qc_quiet_sec", type=float, nargs=2, default=None, metavar=("START", "END"),
        help="Explicit quiet/rest window (seconds) to use for channel QC instead of "
             "auto-detecting the bottom 20%% of the recording.  "
             "Example: --qc_quiet_sec 0 10  uses the first 10 s.",
    )
    p.add_argument(
        "--auto_select_channels", action="store_true",
        help="Automatically drop channels flagged bad by QC (robust Z-score ±3 SD "
             "or high powerline ratio).  Flatline checks are disabled in batch mode "
             "since quiet rest-period channels are healthy.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.add_argument("--config_file", default=None, help="Optional .gesture_config file path")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()

    # Load optional config for persistent defaults
    cfg = {}
    config_path = args.config_file or _CONFIG_FILE
    if Path(config_path).is_file():
        cfg = load_simple_config(str(config_path))

    # Resolve final values: CLI → config → hard default
    data_path   = args.data_path   or cfg.get("data_path")   or str(_DEFAULT_DATA)
    labels_path = args.labels_path or cfg.get("labels_path") or None
    save_path   = args.save_path   or cfg.get("save_path")   or None
    window_ms   = args.window_ms   or int(cfg.get("window_ms", 200))
    step_ms     = args.step_ms     or int(cfg.get("step_ms",   50))
    verbose     = args.verbose     or (cfg.get("verbose", "false").lower() == "true")

    # Validate that data file exists before we try to load it
    if not Path(data_path).exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print("        Pass --data_path to point at your Open Ephys recording.")
        return 1

    build_dataset(
        data_path=data_path,
        labels_path=labels_path,
        save_path=save_path,
        window_ms=window_ms,
        step_ms=step_ms,
        channels=args.channels,
        ignore_labels=args.ignore_labels,
        auto_select_channels=args.auto_select_channels,
        noise_threshold_uv=args.noise_threshold,
        qc_quiet_sec=tuple(args.qc_quiet_sec) if args.qc_quiet_sec else None,
        overwrite=args.overwrite,
        verbose=verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
