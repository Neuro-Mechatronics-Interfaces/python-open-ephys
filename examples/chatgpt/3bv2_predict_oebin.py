#!/usr/bin/env python3
"""
3b_predict_oebin.py
Offline gesture prediction from OEBin using the *exact* training settings:
- window/step/envelope cutoff from metadata
- channel selection by NAME in the original training order
"""

import os
import glob
import argparse
import logging
import numpy as np

from pyoephys.io import (
    load_oebin_file,
    load_config_file,
    lock_params_to_meta,
    load_metadata_json,
    select_training_channels_by_name
)
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events
from pyoephys.processing import EMGPreprocessor



#
# def _peek_event_bounds(root_dir: str) -> tuple[int | None, int | None, str]:
#     """Best-effort scan for event files and return (min_idx, max_idx, source_path)."""
#     candidates = []
#     patterns = [
#         os.path.join(root_dir, "events", "*.csv"),
#         os.path.join(root_dir, "events", "*.npz"),
#         os.path.join(root_dir, "events", "*.json"),
#         os.path.join(root_dir, "model", "*events*.csv"),
#         os.path.join(root_dir, "*events*.csv"),
#         os.path.join(root_dir, "events.csv"),
#         os.path.join(root_dir, "events.npz"),
#         os.path.join(root_dir, "events.json"),
#     ]
#     for pat in patterns: candidates.extend(glob.glob(pat))
#     candidates = sorted(set(candidates), key=lambda p: (len(p), p.lower()))
#
#     for path in candidates:
#         try:
#             if path.lower().endswith(".csv"):
#                 with open(path, "r", newline="") as f:
#                     r = csv.DictReader(f)
#                     cols = [c.strip().lower() for c in (r.fieldnames or [])]
#                     sample_cols = [c for c in cols if "sample" in c or c in ("index","idx","start")]
#                     if not sample_cols: continue
#                     smin, smax = None, None
#                     for row in r:
#                         for c in sample_cols:
#                             val = row.get(c)
#                             if not val: continue
#                             try: k = int(round(float(val)))
#                             except: continue
#                             smin = k if smin is None else min(smin, k)
#                             smax = k if smax is None else max(smax, k)
#                     if smin is not None and smax is not None:
#                         return smin, smax, path
#
#             elif path.lower().endswith(".npz"):
#                 with np.load(path, allow_pickle=True) as F:
#                     for key in ("start_sample","sample","samples","indices","idx","start_idx"):
#                         if key in F:
#                             arr = np.asarray(F[key]).astype(int)
#                             if arr.size: return int(arr.min()), int(arr.max()), path
#
#             elif path.lower().endswith(".json"):
#                 with open(path,"r") as f: J = json.load(f)
#                 if isinstance(J, dict):
#                     for key in ("start_sample","sample","samples","indices","idx","start_idx"):
#                         if key in J:
#                             arr = np.asarray(J[key]).astype(int)
#                             if arr.size: return int(arr.min()), int(arr.max()), path
#                     if "events" in J and isinstance(J["events"], list) and J["events"]:
#                         vals = []
#                         for ev in J["events"]:
#                             for key in ("start_sample","sample","idx","index"):
#                                 if key in ev:
#                                     try: vals.append(int(round(float(ev[key]))))
#                                     except: pass
#                         if vals: return min(vals), max(vals), path
#         except Exception:
#             continue
#     return None, None, ""


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
        selected_channels=None, peek_events: bool = False, verbose: bool = False):

    # Logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load metadata (label-specific if available)
    meta = load_metadata_json(root_dir, label=label)

    # Lock timing & preprocessing from metadata (CLI can override window/step)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Load raw EMG
    raw_dir = os.path.join(root_dir, "raw")
    data = load_oebin_file(raw_dir, verbose=verbose)
    file_name = data['source_path'].split(os.sep)[-1]
    emg_fs = float(data["sample_rate"])
    emg_t = data["t_amplifier"]
    emg = data["amplifier_data"]            # (C, N)
    raw_channel_names = list(data.get("channel_names", []))
    dur_s = emg.shape[1] / emg_fs
    t0 = float(emg_t[0]) if emg_t.size else 0.0
    tN = float(emg_t[-1]) if emg_t.size else (emg.shape[1] - 1) / emg_fs
    logging.info(f"OEBin load: fs={emg_fs:.3f} Hz  emg shape={emg.shape}  channels={len(raw_channel_names)}")
    logging.info(f"OEBin time: t0={t0:.6f}s  tN={tN:.6f}s  duration≈{dur_s:.2f}s")

    # Reorder/select channels by the *training channel names* (canonical order)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    emg, sel_idx = select_training_channels_by_name(emg, raw_channel_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # Preprocess + features (matching training)
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)

    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )
    logging.info(f"Extracted feature matrix X with shape {X.shape}")

    # Build window start indices in absolute sample units
    start_index = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    # Alignment debug (matches NPZ debug script style)
    logging.info(
        "Alignment debug (OEBin):\n"
        f"  start_index={start_index:+d} samp (t0*fs)\n"
        f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={dur_s:.2f}s\n"
        f"  window_starts [min..max]=[{window_starts.min()} .. {window_starts.max()}]"
    )

    # # Optional: peek event bounds and quick overlap check
    # if peek_events:
    #     emin, emax, esrc = _peek_event_bounds(root_dir)
    #     if emin is not None:
    #         wmin, wmax = int(window_starts.min()), int(window_starts.max())
    #         overlaps = not (wmax < emin or wmin > emax)
    #         logging.info(f"Event bounds from {esrc}: min_sample={emin}  max_sample={emax}")
    #         logging.info(f"Windows overlap events? {overlaps}  (windows[{wmin}..{wmax}] vs events[{emin}..{emax}])")
    #     else:
    #         logging.info("Event bounds: not found (peek skipped).")

    # Load model/scaler for this label, check feature dim, predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)

    # Evaluate against events on disk
    file_path = os.path.join(root_dir, 'events', f"{os.path.basename(file_name).split('.')[0]}_emg.event")
    evaluate_against_events(file_path, window_starts, y_pred, verbose=verbose)


if __name__ == "__main__":
    p = argparse.ArgumentParser("3b: Offline EMG gesture prediction from OEBin (training-locked)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    # --channels kept for backward-compatibility but ignored (training order rules)
    p.add_argument("--channels",    nargs="+", type=int, default=None)
    p.add_argument("--no_peek_events", action="store_true", help="Skip event bounds search/logs")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", 200),
        "step_ms": args.step_ms or cfg.get("step_ms", 50),
        "selected_channels": None,  # ignored; we lock to training names
        "peek_events": not args.no_peek_events,
        "verbose": args.verbose or cfg.get("verbose", False),
    })

    run(**cfg)
