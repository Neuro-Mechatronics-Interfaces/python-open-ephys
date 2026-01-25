#!/usr/bin/env python3
"""
3bv2_predict_npz.py
Offline gesture prediction from an NPZ that contains:
  - EMG array (C, N)
  - timestamps vector (N) in seconds
  - sample rate (Hz)
Optionally channel names.

It locks window/step/envelope cutoff and training channel order from metadata,
exactly like the OEBin predictor.

Expected NPZ keys (auto-detected, multiple aliases supported):
  emg:          'z_emg' OR 'amplifier_data'
  timestamps:   'o_ts'  OR 't_amplifier'
  sample rate:  'fs'    OR 'fs_hz' OR 'sample_rate'
  channel names:'ch_names' OR 'channel_names' (optional; falls back to CH1..CH{C})

Usage:
  python 3bv2_predict_npz.py --npz <path> --root_dir <root> --label <label> --verbose
"""

import matplotlib.pyplot as plt
import argparse
import logging
import numpy as np

from pyoephys.io import (
    load_config_file,
    lock_params_to_meta,
    load_metadata_json,
    load_npz_file,
    select_training_channels_by_name
)
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events
from pyoephys.processing import EMGPreprocessor, fix_missing_emg


def run(npz_path: str, root_dir: str, label: str = "",
        window_ms: int | None = None, step_ms: int | None = None,
        fill_missing: bool = True,
        fill_constant: float = 0.0,
        fill_figures: bool = False,
        fill_fig_dir: str | None = None,
        fill_overlay_channels: tuple[int, ...] = (0, 1, 2),
        verbose: bool = False):

    # Logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load training metadata (locks timing & preprocessing)
    meta = load_metadata_json(root_dir, label=label)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Load NPZ
    data = load_npz_file(npz_path, verbose=verbose)
    fs = data["sample_rate"]
    t = data["t_amplifier"]
    emg = data["amplifier_data"]
    ch_names = list(data.get("channel_names", []))

    # timestamp offset added of [80.627003 seconds] to the t_amplifier vector
    #t += 80.627003
    print(f"First 10 timestamps: {t[:10]}")

    C, N = emg.shape
    if ch_names is None:
        ch_names = [f"CH{i+1}" for i in range(C)]
    else:
        ch_names = [str(x) for x in np.asarray(ch_names).tolist()]
        if len(ch_names) != C:
            raise ValueError(f"Channel names length {len(ch_names)} != EMG channels {C}")

    # Log basic info
    dur_s = N / fs
    t0 = float(t[0])
    tN = float(t[-1])
    logging.info(f"NPZ: fs={fs:.3f} Hz  emg shape={emg.shape}  channels={len(ch_names)}")
    logging.info(f"NPZ time: t0={t0:.6f}s  tN={tN:.6f}s  duration≈{dur_s:.2f}s")

    # Reorder/select channels by the *training channel names* (canonical order)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    emg, sel_idx = select_training_channels_by_name(emg, ch_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # --- Fill NaN/Inf (optional) ---
    if fill_missing and fill_missing.lower() != "off":
        # Figure output dir if user asked for figures but didn't provide a path
        if fill_figures and not fill_fig_dir:
            from pathlib import Path
            fill_fig_dir = str(Path(npz_path).with_suffix("")) + "_nanfix"

        emg, nan_stats = fix_missing_emg(
            emg,
            method=fill_missing.lower(),
            constant_value=fill_constant,
            make_figures=fill_figures,
            out_dir=fill_fig_dir,
            fs=float(fs),
            overlay_channels=fill_overlay_channels,
            verbose=True,
        )
        logging.info(
            f"[nanfix] method={nan_stats.method} "
            f"missing_before={nan_stats.total_missing} "
            f"remaining_after={nan_stats.filled_remaining_missing} "
            f"(~{nan_stats.pct_missing:.2f}% of samples were missing pre-fill)"
        )

    # Plot raw EMG of channel 5 as debug
    if verbose:
        plt.figure(figsize=(10, 4))
        plt.plot(t, emg[4], label='Channel 5 Raw EMG')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Raw EMG Channel 5')
        plt.legend()
        plt.grid()
        plt.show()

    # Preprocessing + features (matching training)
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    print(f"Shape of preprocessed EMG: {emg_pp.shape}")

    # As debug just plot channel 5 with the timetamp
    if verbose:
        plt.figure(figsize=(10, 4))
        plt.plot(t, emg_pp[4], label='Channel 5')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('EMG Channel 5')
        plt.legend()
        plt.grid()
        plt.show()

    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )
    logging.info(f"Extracted feature matrix X with shape {X.shape}")

    # Build window start indices in absolute sample units (no lag)
    step_samples = int(round(step_ms / 1000.0 * fs))
    start_index = int(round(t[0] * fs))  # important: uses provided timestamp origin directly
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    logging.info(
        "Alignment debug (NPZ):\n"
        f"  start_index={start_index:+d} samp (t[0]*fs)\n"
        f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={dur_s:.2f}s\n"
        f"  window_starts [min..max]=[{window_starts.min()} .. {window_starts.max()}]"
    )

    # Load model/scaler, predict, evaluate
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)
    evaluate_against_events(root_dir, window_starts, y_pred)


if __name__ == "__main__":
    p = argparse.ArgumentParser("3b: Offline EMG gesture prediction from NPZ (training-locked)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--npz_file",        type=str, required=True, help="Path to NPZ with EMG/timestamps/fs")
    p.add_argument("--root_dir",   type=str, required=True)
    p.add_argument("--label",      type=str, default="")
    p.add_argument("--window_ms",  type=int, default=None)
    p.add_argument("--step_ms",    type=int, default=None)
    p.add_argument("--fill_missing",
                   choices=["off", "repeat", "interp", "ffill", "bfill", "median", "constant"],
                   default="off",
                   help="How to fill NaN/Inf before preprocessing (default: off)")
    p.add_argument("--fill_constant", type=float, default=0.0,
                   help="Value for --fill_missing constant")
    p.add_argument("--fill_figures", action="store_true",
                   help="Emit NaN map and before/after overlays")
    p.add_argument("--fill_fig_dir", type=str, default=None,
                   help="Directory to save fill QC figures/CSV")
    p.add_argument("--fill_overlay", type=str, default="0,1,2",
                   help="Comma-separated channel indices for the overlay figure")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    overlay = tuple(int(s) for s in args.fill_overlay.split(",") if s.strip() != "")
    cfg.update({
        "npz_path":  args.npz_file,
        "root_dir":  args.root_dir or cfg.get("root_dir", ""),
        "label":     args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", 200),
        "step_ms":   args.step_ms   or cfg.get("step_ms", 50),
        "fill_missing": args.fill_missing,
        "fill_constant": args.fill_constant,
        "fill_figures": args.fill_figures,
        "fill_fig_dir": args.fill_fig_dir,
        "fill_overlay_channels": overlay if overlay else (0, 1, 2),
        "verbose":   args.verbose or cfg.get("verbose", False),
    })

    run(**cfg)
