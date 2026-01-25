#!/usr/bin/env python3
"""
3c_predict_npz_cache_v2.py
Offline gesture prediction from streamed NPZ, with explicit OEBin↔ZMQ timebase alignment.

Fixes:
- Start indices are now derived from (z_ts[0] - o_ts[0]) * fs, not z_ts[0]*fs.
- Optional lag_samples correction applied afterward.
- Alignment debug printout to diagnose "No valid windows to evaluate" issues.
"""

import os, json, argparse, logging
import numpy as np

from pyoephys.io import load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events
from pyoephys.processing import EMGPreprocessor


def _select_training_channels_by_name(emg: np.ndarray,
                                      raw_names: list[str],
                                      trained_names: list[str]) -> tuple[np.ndarray, list[int]]:
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording is missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx


def _load_npz(cache_npz: str):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs_hz = float(F["fs_hz"])
        ch_names_z = list(F["ch_names_z"])
        # Prefer window-stitched stream if available
        emg = F["recon_z"] if "recon_z" in F and F["recon_z"].size else F["z_emg"]
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(emg.shape[1]) / fs_hz
        o_ts = F["o_ts"] if "o_ts" in F else np.array([], dtype=np.float64)
        meta = {}
        if "meta_json" in F:
            meta = json.loads(str(F["meta_json"].item()))
    return dict(emg=emg, fs_hz=fs_hz, z_ts=z_ts, o_ts=o_ts, ch_names_z=ch_names_z, meta=meta)


def run(cache_npz: str,
        root_dir: str,
        label: str = "",
        window_ms: int | None = None,
        step_ms: int | None = None,
        apply_lag_alignment: bool = True,
        manual_start_offset_samples: int | None = None,
        use_raw_z: bool = False,
        verbose: bool = False):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # ---- Load cache (.npz) ----
    C = _load_npz(cache_npz)
    emg_fs = float(C["fs_hz"])
    raw_channel_names = list(C["ch_names_z"])
    lag_samples = int(round(C["meta"].get("lag_samples", 0))) if C.get("meta") else 0

    if use_raw_z and "z_emg" in C:
        # If you want the raw ZMQ concatenation instead of recon_z
        with np.load(cache_npz, allow_pickle=True) as F:
            emg = F["z_emg"]
    else:
        emg = C["emg"]

    z_ts = np.asarray(C["z_ts"])
    o_ts = np.asarray(C["o_ts"]) if C["o_ts"] is not None else np.array([], dtype=np.float64)

    logging.info(f"Loaded cache: {cache_npz} | fs={emg_fs:.3f} Hz | shape={emg.shape} | lag={lag_samples} samp")

    # ---- Load training metadata ----
    meta_train = load_metadata_json(root_dir, label=label)
    trained_names = meta_train.get("data", {}).get("channel_names") or meta_train.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta_train['data'], window_ms, step_ms, selected_channels=None)

    # ---- Channel order by training names ----
    emg, sel_idx = _select_training_channels_by_name(emg, raw_channel_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # ---- Preprocess + features ----
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )
    logging.info(f"Extracted feature matrix X with shape {X.shape}")

    # ---- Build window start indices on the OEBin timebase ----
    z_t0 = float(z_ts[0]) if z_ts.size else 0.0
    o_t0 = float(o_ts[0]) if o_ts.size else 0.0
    time_offset_samples = int(round((z_t0 - o_t0) * emg_fs))  # <— key fix

    if manual_start_offset_samples is not None:
        logging.warning(f"Overriding computed offset ({time_offset_samples}) with manual value: {manual_start_offset_samples}")
        time_offset_samples = int(manual_start_offset_samples)

    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + time_offset_samples

    if apply_lag_alignment and lag_samples:
        window_starts = window_starts - lag_samples
        logging.info(f"Applied lag alignment: shifted starts by {-lag_samples} samples")

    # ---- Debug: print alignment info ----
    approx_duration_s = emg.shape[1] / emg_fs
    logging.info(
        f"Alignment debug:\n"
        f"  o_t0={o_t0:.6f}s  z_t0={z_t0:.6f}s  Δt={z_t0 - o_t0:+.6f}s  => offset={time_offset_samples:+d} samp\n"
        f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={approx_duration_s:.2f}s\n"
        f"  starts[min..max]=[{window_starts.min() if window_starts.size else 'NA'} .. "
        f"{window_starts.max() if window_starts.size else 'NA'}]"
    )

    # ---- Model + predict ----
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)

    # ---- Evaluate against events (if present) ----
    try:
        evaluate_against_events(root_dir, window_starts, y_pred)
    except Exception as e:
        logging.info(f"Event evaluation skipped or failed: {e}")


def parse_args():
    p = argparse.ArgumentParser("3c v2: Offline EMG prediction from NPZ with robust timebase alignment")
    p.add_argument("--config_file", type=str)
    p.add_argument("--cache_npz",   type=str, required=True, help="Path to NPZ saved by streaming capture")
    p.add_argument("--root_dir",    type=str, required=True, help="Run directory with model/metadata/events")
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--no_align",    action="store_true", help="Do NOT apply cached lag_samples alignment")
    p.add_argument("--manual_start_offset_samples", type=int, default=None,
                   help="Override computed (z_t0 - o_t0)*fs with this value (for quick debugging)")
    p.add_argument("--use_raw_z",   action="store_true", help="Use raw z_emg instead of recon_z if present")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        cache_npz=cfg["cache_npz"],
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        apply_lag_alignment=not cfg.get("no_align", False),
        manual_start_offset_samples=cfg.get("manual_start_offset_samples", None),
        use_raw_z=cfg.get("use_raw_z", False),
        verbose=cfg.get("verbose", False),
    )
