#!/usr/bin/env python3
"""
predict_npz_cache.py

Run the trained EMG classifier on an NPZ cache instead of raw OEBin,
while repairing non-finite samples and applying a known/estimated lag.

What it does
------------
1) Loads training metadata to lock window/step/envelope cutoff and channel order.
2) Loads NPZ (prefers 'recon_z' then 'z_emg'), channel-names, timestamps.
3) Repairs NaN/Inf per channel:
   - interp  : linear between previous/next finite sample (default)
   - noise1k : 1 kHz jitter about a linear envelope between boundaries (clipped to boundary range)
   - zero    : set to 0
   - none    : leave as-is (not recommended)
4) Applies a circular lag shift to NPZ (positive = shift right); if not given,
   auto-derives lag from NPZ 'o_ts[0]' vs 'z_ts[0]' when both exist.
5) Preprocesses with training-locked EMGPreprocessor and extracts features.
6) Builds window starts on the OEBin time base (when 'o_ts' present) for event eval.
7) Loads model, predicts, and evaluates against events on disk.

Examples
--------
python predict_npz_cache.py ^
  --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" ^
  --root_dir  "G:/.../2025_07_31" ^
  --label "sleeve_15ch_ring" ^
  --lag_samples 306 ^
  --fix_mode interp --verbose

python predict_npz_cache.py ^
  --cache_npz "...winverify.npz" --root_dir ".../2025_07_31" --label sleeve_15ch_ring ^
  --fix_mode noise1k --noise_scale 0.5  --verbose

"""

import os, argparse, logging, json
import numpy as np

from pyoephys.io import load_metadata_json, lock_params_to_meta, load_config_file, normalize_name
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


# ---------------------------
# small helpers
# ---------------------------

def _select_training_channels_by_name(emg: np.ndarray,
                                      raw_names: list[str],
                                      trained_names: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]

    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")

    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx, [raw_names[i] for i in idx]


def _load_npz(cache_npz: str, prefer_recon=True):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs_hz = float(F["fs_hz"]) if "fs_hz" in F else float(F["fs"])
        ch_names = list(F["ch_names_z"]) if "ch_names_z" in F else list(F["ch_names"])
        stream_used = None
        emg = None
        if prefer_recon and "recon_z" in F and F["recon_z"].size:
            emg = F["recon_z"]; stream_used = "recon_z"
        elif "z_emg" in F:
            emg = F["z_emg"];   stream_used = "z_emg"
        else:
            # some caches use generic 'emg'
            emg = F["emg"];     stream_used = "emg"

        z_ts = np.asarray(F["z_ts"]) if "z_ts" in F else np.arange(emg.shape[1]) / fs_hz
        o_ts = np.asarray(F["o_ts"]) if "o_ts" in F else np.array([], dtype=np.float64)
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    return dict(emg=emg, fs=fs_hz, z_ts=z_ts, o_ts=o_ts, ch_names=ch_names, meta=meta, stream_used=stream_used)


def _runs_from_bool(b: np.ndarray):
    """Return list of (start, end_exclusive) for True-runs in 1D bool array."""
    if b.ndim != 1: b = b.ravel()
    if b.size == 0: return []
    db = np.diff(b.astype(np.int8))
    starts = np.where(db == 1)[0] + 1
    ends   = np.where(db == -1)[0] + 1
    if b[0]:  starts = np.r_[0, starts]
    if b[-1]: ends   = np.r_[ends, b.size]
    return list(zip(starts, ends))


def _fill_nonfinite_channel(x: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    """In-place repair of a 1D channel vector."""
    bad = ~np.isfinite(x)
    if not bad.any() or mode == "none":
        return x

    runs = _runs_from_bool(bad)
    n = x.size
    for a, b in runs:
        L = b - a
        # find boundary samples
        left  = a - 1
        right = b
        # nearest finite left
        while left >= 0 and not np.isfinite(x[left]):
            left -= 1
        # nearest finite right
        while right < n and not np.isfinite(x[right]):
            right += 1

        # boundary values (fallback to zero if all bad)
        vL = float(x[left]) if left >= 0 else 0.0
        vR = float(x[right]) if right < n else vL

        if mode == "zero":
            x[a:b] = 0.0
            continue

        # base: linear envelope from vL to vR
        env = np.linspace(vL, vR, L, dtype=np.float64)

        if mode == "interp":
            x[a:b] = env
            continue

        if mode == "noise1k":
            # 1 kHz “jitter” at fs ≈ 2 kHz → alternate sign pattern
            alt = 1.0 - 2.0 * ((np.arange(L) & 1).astype(np.float64))  # +1, -1, +1, ...
            # jitter amplitude based on boundary spread
            spread = max(1e-6, abs(vR - vL))
            jitter = noise_scale * 0.25 * spread * alt
            fill = env + jitter
            # clip to boundary range (avoid wild excursions)
            lo, hi = (vL, vR) if vL <= vR else (vR, vL)
            fill = np.clip(fill, lo, hi)
            x[a:b] = fill
            continue

        # default safety
        x[a:b] = env
    return x


def _repair_nonfinite(arr: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    A = np.array(arr, copy=True, dtype=np.float64)
    C = A.shape[0]
    total_bad = 0
    for c in range(C):
        badc = ~np.isfinite(A[c])
        total_bad += int(badc.sum())
        if badc.any():
            _fill_nonfinite_channel(A[c], mode=mode, noise_scale=noise_scale)
    return A, total_bad


def _apply_lag_roll(emg: np.ndarray, lag_samples: int) -> np.ndarray:
    """Circularly roll the time axis by lag_samples (positive -> shift right)."""
    if lag_samples == 0:
        return emg
    return np.roll(emg, shift=lag_samples, axis=1)


# ---------------------------
# main
# ---------------------------

def run(cache_npz: str,
        root_dir: str,
        label: str = "",
        window_ms: int | None = None,
        step_ms: int | None = None,
        fix_mode: str = "interp",
        noise_scale: float = 1.0,
        lag_samples: int | None = None,
        prefer_recon: bool = True,
        save_cleaned_npz: str | None = None,
        verbose: bool = False):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # ---- training-locked params
    meta = load_metadata_json(root_dir, label=label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # ---- load NPZ cache
    C = _load_npz(cache_npz, prefer_recon=prefer_recon)
    z_emg_all = C["emg"]; z_names_all = list(C["ch_names"])
    fs = float(C["fs"]);  z_ts = np.asarray(C["z_ts"]); o_ts = np.asarray(C["o_ts"])
    logging.info(f"NPZ: stream={C['stream_used']}  fs={fs:.3f} Hz  emg={z_emg_all.shape}  ch={len(z_names_all)}")

    # ---- training channel order
    z_emg, z_idx, z_names_sel = _select_training_channels_by_name(z_emg_all, z_names_all, trained_names)
    logging.info(f"NPZ channel order (training-locked): {list(zip(z_idx, z_names_sel))[:8]}{' ...' if len(z_idx)>8 else ''}")

    # ---- repair non-finite
    bad_before = int((~np.isfinite(z_emg)).sum())
    z_emg_fixed, total_bad = _repair_nonfinite(z_emg, mode=fix_mode, noise_scale=noise_scale)
    logging.info(f"Non-finite repaired: {total_bad} samples (before={bad_before})")

    # ---- lag (auto if possible)
    auto_lag = None
    if lag_samples is None and o_ts.size and z_ts.size:
        # positive lag means z needs to shift right to align with OEBin
        start_o = int(round(o_ts[0] * fs))
        start_z = int(round(z_ts[0] * fs))
        auto_lag = start_o - start_z
        lag_samples = auto_lag
    lag_samples = int(lag_samples or 0)
    logging.info(f"Lag to apply (samples): {lag_samples}{'  [auto from o_ts/z_ts]' if auto_lag is not None else ''}")

    z_emg_fixed = _apply_lag_roll(z_emg_fixed, lag_samples)

    # ---- preprocess + features (training-locked)
    prep = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    z_pp = prep.preprocess(z_emg_fixed)
    X = prep.extract_emg_features(
        z_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "NPZ features", "unit": "win", "leave": False}
    )
    logging.info(f"Features: X={X.shape}")

    # ---- window starts on OEBin time base if present
    if o_ts.size:
        start_index = int(round(o_ts[0] * fs))
    else:
        start_index = int(round(z_ts[0] * fs)) if z_ts.size else 0
    step_samples = int(round(step_ms / 1000.0 * fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index
    logging.info(f"Window index range: [{window_starts.min()} .. {window_starts.max()}]  step_samp={step_samples}")

    # ---- model & predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()

    nfeat = len(manager.scaler.mean_)
    if X.shape[1] != nfeat:
        raise ValueError(f"Feature dim mismatch (expected {nfeat}) → NPZ={X.shape[1]}")

    y_pred = manager.predict(X)

    # ---- evaluate against events on disk
    try:
        evaluate_against_events(root_dir, window_starts, y_pred)
    except Exception as e:
        logging.info(f"Evaluation skipped/failed: {e}")

    # ---- optional save of cleaned, shifted stream (for debugging/reuse)
    if save_cleaned_npz:
        os.makedirs(os.path.dirname(save_cleaned_npz), exist_ok=True)
        np.savez_compressed(
            save_cleaned_npz,
            fs_hz=fs, emg=z_emg_fixed, ch_names=np.array(z_names_sel, dtype=object),
            z_ts=z_ts, o_ts=o_ts, meta_json=json.dumps(C["meta"]),
            note=f"cleaned={fix_mode}, noise_scale={noise_scale}, lag_samples={lag_samples}, stream={C['stream_used']}"
        )
        logging.info(f"[saved] cleaned NPZ → {save_cleaned_npz}")


def parse_args():
    p = argparse.ArgumentParser("Predict from NPZ cache with gap-fix + lag alignment (training-locked)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--cache_npz",   type=str, required=True)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)

    p.add_argument("--fix_mode",    choices=["interp", "noise1k", "zero", "none"], default="interp")
    p.add_argument("--noise_scale", type=float, default=1.0, help="Scale for noise1k jitter (default 1.0)")
    p.add_argument("--lag_samples", type=int, default=None, help="Circular lag to apply to NPZ (samples). If omitted and o_ts+z_ts exist, auto from start times.")
    p.add_argument("--no_prefer_recon", action="store_true", help="Use z_emg even if recon_z exists")
    p.add_argument("--save_cleaned_npz", type=str, default=None)

    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update({
        "cache_npz": args.cache_npz,
        "root_dir": args.root_dir,
        "label": args.label or cfg.get("label",""),
        "window_ms": args.window_ms or cfg.get("window_ms", None),
        "step_ms": args.step_ms or cfg.get("step_ms", None),
        "fix_mode": args.fix_mode or cfg.get("fix_mode","interp"),
        "noise_scale": float(args.noise_scale if args.noise_scale is not None else cfg.get("noise_scale", 1.0)),
        "lag_samples": args.lag_samples if args.lag_samples is not None else cfg.get("lag_samples", None),
        "prefer_recon": not args.no_prefer_recon,
        "save_cleaned_npz": args.save_cleaned_npz or cfg.get("save_cleaned_npz", None),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    run(**cfg)
