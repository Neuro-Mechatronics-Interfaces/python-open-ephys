#!/usr/bin/env python3
"""
predict_npz_cache_refactored.py

Run the trained EMG classifier on an NPZ cache (captured from Open Ephys GUI),
with strong parity to the OEBin path and rich diagnostics.

Key features
------------
- Training-locked params (window/step/envelope) and channel order.
- Defaults to z_emg (raw ZMQ) for parity; opt-in to recon_z via --prefer_recon.
- Lag priority: CLI > cache meta > auto (o_ts - z_ts) > 0.
- Non-finite repair with per-channel stats (percent, longest run).
- Optional window masking to DROP windows that overlap repaired samples (--skip_bad_windows).
- Optional lag refinement around the chosen lag (--lag_refine_search).
- Optional z_emg vs recon_z NCC diagnostics when both are present.
- Prints first 10 samples & timestamps for the first trained channel (raw/fixed/lagged).
- Optionally saves a cleaned, lag-shifted NPZ.

CLI examples
------------
python predict_npz_cache_refactored.py ^
  --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" ^
  --root_dir  "G:/.../2025_07_31" ^
  --label "sleeve_15ch_ring" ^
  --fix_mode interp --skip_bad_windows 0.10 --verbose

python predict_npz_cache_refactored.py ^
  --cache_npz "...winverify.npz" --root_dir ".../2025_07_31" --label sleeve_15ch_ring ^
  --prefer_recon --fix_mode noise1k --noise_scale 0.25 --skip_bad_windows 0.10 --verbose
"""

from __future__ import annotations
import os, argparse, logging, json
from typing import Optional, Tuple, List, Dict

import numpy as np

from pyoephys.io import (
    load_metadata_json,
    lock_params_to_meta,
    load_config_file,
    normalize_name,
)
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


# ---------------------------
# helpers
# ---------------------------

def _select_training_channels_by_name(
    emg: np.ndarray, raw_names: List[str], trained_names: List[str]
) -> Tuple[np.ndarray, List[int], List[str]]:
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(
            f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}"
        )
    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx, [raw_names[i] for i in idx]


def _npz_get(F: np.lib.npyio.NpzFile, *keys, default=None):
    for k in keys:
        if k in F:
            return F[k]
    return default


def _as_1d_float(arr):
    if arr is None:
        return np.array([], dtype=np.float64)
    a = np.asarray(arr)
    if a.ndim == 0:     # scalars/objects -> treat as missing
        return np.array([], dtype=np.float64)
    return np.asarray(a, dtype=np.float64).ravel()


def _load_npz(cache_npz: str, prefer_recon: bool) -> Dict:
    with np.load(cache_npz, allow_pickle=True) as F:
        print(f"keys: {list(F.keys())}")
        fs = float(_npz_get(F, "fs_hz", "fs"))
        ch_names = list(_npz_get(F, "ch_names_z", "ch_names"))

        # candidate streams (we may use both for diagnostics)
        recon_z = F["recon_z"] if ("recon_z" in F and F["recon_z"].size) else None
        z_emg   = F["z_emg"]   if ("z_emg"   in F) else None
        emg_fallback = F["emg"] if "emg" in F else None

        # choose main stream
        stream_used, emg = None, None
        if prefer_recon and (recon_z is not None):
            emg, stream_used = recon_z, "recon_z"
        elif z_emg is not None:
            emg, stream_used = z_emg, "z_emg"
        else:
            emg, stream_used = emg_fallback, "emg"

        # timestamps
        #z_ts = np.asarray(_npz_get(F, "z_ts"), dtype=float) if "z_ts" in F else np.arange(emg.shape[1]) / fs
        #o_ts = np.asarray(_npz_get(F, "o_ts", default=np.array([], dtype=np.float64)), dtype=float)
        o_ts = _as_1d_float(_npz_get(F, "o_ts", "t0_seconds"))
        z_ts = _as_1d_float(_npz_get(F, "z_ts"))
        if z_ts.size == 0:
            z_ts = np.arange(emg.shape[1], dtype=np.float64) / fs
        # sanity: lengths must match samples
        if o_ts.size not in (0, emg.shape[1]):
            logging.warning(f"o_ts length {o_ts.size} != samples {emg.shape[1]} → ignoring o_ts")
            o_ts = np.array([], dtype=np.float64)

        # meta_json
        if "meta_json" in F:
            try:
                meta = json.loads(str(F["meta_json"].item()))
            except Exception:
                meta = {}
        else:
            meta = {}

    return dict(
        emg=emg, fs=fs, z_ts=z_ts, o_ts=o_ts, ch_names=ch_names, meta=meta,
        stream_used=stream_used, recon_z=recon_z, z_emg=z_emg
    )


def _runs_from_bool(b: np.ndarray):
    if b.ndim != 1: b = b.ravel()
    if b.size == 0: return []
    db = np.diff(b.astype(np.int8))
    starts = np.where(db == 1)[0] + 1
    ends   = np.where(db == -1)[0] + 1
    if b[0]:  starts = np.r_[0, starts]
    if b[-1]: ends   = np.r_[ends, b.size]
    return list(zip(starts, ends))


def _fill_nonfinite_channel(x: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    bad = ~np.isfinite(x)
    if not bad.any() or mode == "none":
        return x

    runs = _runs_from_bool(bad)
    n = x.size
    for a, b in runs:
        L = b - a
        left, right = a - 1, b
        while left >= 0 and not np.isfinite(x[left]): left -= 1
        while right < n and not np.isfinite(x[right]): right += 1
        vL = float(x[left]) if left >= 0 else 0.0
        vR = float(x[right]) if right < n else vL

        if mode == "zero":
            x[a:b] = 0.0
            continue

        # linear envelope
        env = np.linspace(vL, vR, L, dtype=np.float64)

        if mode == "interp":
            x[a:b] = env
            continue

        if mode == "noise1k":
            # tiny texture to avoid perfectly flat regions
            alt = 1.0 - 2.0 * ((np.arange(L) & 1).astype(np.float64))
            spread = max(1e-6, abs(vR - vL))
            jitter = noise_scale * 0.25 * spread * alt
            fill = env + jitter
            lo, hi = (vL, vR) if vL <= vR else (vR, vL)
            x[a:b] = np.clip(fill, lo, hi)
            continue

        x[a:b] = env
    return x


def _repair_nonfinite(arr: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    A = np.array(arr, copy=True, dtype=np.float64)
    total_bad = 0
    per_ch_bad = []
    per_ch_longest = []
    masks = []
    for c in range(A.shape[0]):
        badc = ~np.isfinite(A[c])
        total_bad += int(badc.sum())
        per_ch_bad.append(int(badc.sum()))
        runs = _runs_from_bool(badc)
        longest = 0 if not runs else max(b - a for a, b in runs)
        per_ch_longest.append(int(longest))
        masks.append(badc)
        if badc.any():
            _fill_nonfinite_channel(A[c], mode=mode, noise_scale=noise_scale)
    return A, total_bad, np.vstack(masks), np.array(per_ch_bad), np.array(per_ch_longest)


def _apply_lag_roll(emg: np.ndarray, lag_samples: int) -> np.ndarray:
    if lag_samples == 0:
        return emg
    return np.roll(emg, shift=int(lag_samples), axis=1)


def _ncc(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom > 0 else 0.0


def refine_lag_by_ncc(emg: np.ndarray, approx_lag: int, search: int = 800, ch: int = 0) -> Tuple[int, float]:
    """
    Refine lag by maximizing NCC on one channel around approx_lag ± search.
    Use AFTER non-finite repair, BEFORE applying lag.
    """
    x = emg[ch].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    best_lag, best_corr = approx_lag, -1.0
    for d in range(approx_lag - search, approx_lag + search + 1):
        y = np.roll(x, d)
        r = float(np.dot(x, y) / len(x))  # NCC since x is normalized
        if r > best_corr:
            best_corr, best_lag = r, d
    return best_lag, best_corr


# ---------------------------
# main
# ---------------------------

def run(
    cache_npz: str,
    root_dir: str,
    label: str = "",
    window_ms: Optional[int] = None,
    step_ms: Optional[int] = None,
    fix_mode: str = "interp",
    noise_scale: float = 1.0,
    lag_samples: Optional[int] = None,
    prefer_recon: bool = False,     # default False for parity with OEBin
    save_cleaned_npz: Optional[str] = None,
    skip_bad_windows: Optional[float] = None,  # e.g., 0.10 = drop windows with >=10% repaired samples in ANY selected channel
    lag_refine_search: int = 0,     # set >0 to refine lag within ±search samples
    verbose: bool = False,
    diagnose: bool = True,
):
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # ---- training-locked params
    meta_train = load_metadata_json(root_dir, label=label)
    trained_names = meta_train.get("data", {}).get("channel_names") or meta_train.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta_train["data"], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # ---- load NPZ cache
    C = _load_npz(cache_npz, prefer_recon=prefer_recon)
    z_emg_all = C["emg"]; fs = float(C["fs"])
    z_names_all = list(C["ch_names"])
    z_ts = np.asarray(C["z_ts"], dtype=float)
    o_ts = np.asarray(C["o_ts"], dtype=float)
    meta_cache = C["meta"] if isinstance(C["meta"], dict) else {}
    logging.info(f"NPZ: stream={C['stream_used']}  fs={fs:.3f} Hz  emg={z_emg_all.shape}  ch={len(z_names_all)}  prefer_recon={prefer_recon}")

    # optional: NCC diag between streams (first 15 channels)
    if diagnose and (C["recon_z"] is not None) and (C["z_emg"] is not None) and (C["recon_z"].shape == C["z_emg"].shape):
        z = C["z_emg"]; r = np.roll(C["recon_z"], int(lag_samples or 0), axis=1)
        nccs = [_ncc(z[i], r[i]) for i in range(min(15, z.shape[0]))]
        logging.info(f"[diag] NCC(z_emg vs recon_z, first 15 ch): mean={np.mean(nccs):.3f}  min={np.min(nccs):.3f}  max={np.max(nccs):.3f}")

    # ---- training channel order
    z_emg, z_idx, z_names_sel = _select_training_channels_by_name(z_emg_all, z_names_all, trained_names)
    logging.info(f"NPZ channel order (training-locked): {list(zip(z_idx, z_names_sel))[:8]}{' ...' if len(z_idx)>8 else ''}")

    # ---- non-finite repair (with stats & mask)
    bad_before = int((~np.isfinite(z_emg)).sum())
    z_emg_fixed, total_bad, bad_mask, per_ch_bad, per_ch_longest = _repair_nonfinite(
        z_emg, mode=fix_mode, noise_scale=noise_scale
    )
    logging.info(f"Non-finite repaired: {total_bad} samples (before={bad_before})  mode={fix_mode}")
    if diagnose:
        n = z_emg.shape[1]
        pct = per_ch_bad / n * 100.0
        logging.info(f"[diag] per-channel non-finite % (first 15): {np.round(pct[:15], 2)}")
        logging.info(f"[diag] per-channel longest non-finite run (samples, first 15): {per_ch_longest[:15]}")

    # ---- lag (priority: CLI > cache meta > auto from o_ts/z_ts > 0)
    auto_lag = None
    lag_source = "default(0)"
    if lag_samples is None:
        if "lag_samples" in meta_cache:
            try:
                lag_samples = int(meta_cache["lag_samples"])
                lag_source = "cache_meta"
            except Exception:
                lag_samples = None
        if lag_samples is None and o_ts.size and z_ts.size:
            start_o = int(round(o_ts[0] * fs))
            start_z = int(round(z_ts[0] * fs))
            auto_lag = start_o - start_z
            lag_samples = int(auto_lag)
            lag_source = "auto(o_ts - z_ts)"
    else:
        lag_source = "cli"
    lag_samples = int(lag_samples or 0)
    logging.info(f"Lag to apply (samples): {lag_samples}  source={lag_source}")

    # optional: refine lag locally by NCC on one channel
    if lag_refine_search and lag_refine_search > 0:
        ref_lag, ref_corr = refine_lag_by_ncc(z_emg_fixed, lag_samples, search=int(lag_refine_search), ch=0)
        logging.info(f"[diag] lag refine: approx={lag_samples} -> best={ref_lag} (NCC~{ref_corr:.3f})")
        if abs(ref_lag - lag_samples) >= 10 and ref_corr > 0:
            lag_samples = int(ref_lag)

    # ---- show first 10 samples & times (pre/post fix & lag)
    if diagnose:
        ch0 = 0
        t_base = o_ts if o_ts.size else (z_ts if z_ts.size else np.arange(z_emg.shape[1]) / fs)
        logging.info(f"[diag] First trained channel: {z_names_sel[ch0]}  (idx={z_idx[ch0]})")
        logging.info(f"[diag] raw first 10:  {np.array2string(z_emg[ch0, :10], precision=4)}")
        logging.info(f"[diag] time first 10: {np.array2string(t_base[:10], precision=6)}")
        logging.info(f"[diag] fixed first 10:{np.array2string(z_emg_fixed[ch0, :10], precision=4)}")

    z_emg_fixed = _apply_lag_roll(z_emg_fixed, lag_samples)

    if diagnose:
        logging.info(f"[diag] lagged first 10:{np.array2string(z_emg_fixed[0, :10], precision=4)}")

    # ---- preprocess
    prep = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    z_pp = prep.preprocess(z_emg_fixed)

    # ---- choose time-base for absolute indexing
    if o_ts.size:
        start_index = int(round(o_ts[0] * fs)); base = "o_ts"
    elif z_ts.size:
        start_index = int(round(z_ts[0] * fs)); base = "z_ts"
    else:
        start_index = 0; base = "zero"

    # ---- window geometry (match extractor)
    win = int(round(window_ms / 1000.0 * fs))
    step_samples = int(round(step_ms / 1000.0 * fs))
    n_total = z_emg_fixed.shape[1]
    n_wins = (n_total - win) // step_samples + 1
    if n_wins <= 0:
        raise ValueError(f"Not enough samples for one window: N={n_total}, win={win}, step={step_samples}")

    # ---- optional: build a mask of windows to keep (based on repaired samples)
    mask_windows = None
    if skip_bad_windows is not None:
        # rolling count of repaired samples per channel using convolution
        kernel = np.ones(win, dtype=np.int32)
        bad_per_ch = np.array([
            np.convolve(bad_mask[c].astype(np.int32), kernel, mode="valid")
            for c in range(bad_mask.shape[0])
        ])  # shape: (C, n_wins_candidate)
        starts = np.arange(n_wins) * step_samples
        counts = bad_per_ch[:, starts]  # shape: (C, n_wins)
        frac = counts / float(win)
        mask_windows = ~(frac >= float(skip_bad_windows)).any(axis=0)  # keep if every channel < threshold
        kept = int(mask_windows.sum())
        logging.info(f"[mask] skip_bad_windows={float(skip_bad_windows):.3f} → keeping {kept}/{n_wins} windows ({kept/n_wins*100:.1f}%)")

    # ---- features for ALL windows (then rowslice by mask if present)
    X_full = prep.extract_emg_features(
        z_pp,
        window_ms=window_ms,
        step_ms=step_ms,
        progress=True,
        tqdm_kwargs={"desc": "NPZ features", "unit": "win", "leave": False},
    )
    if X_full.shape[0] != n_wins:
        logging.warning(f"Expected {n_wins} windows, extractor returned {X_full.shape[0]}")

    if mask_windows is not None:
        X = X_full[mask_windows]
        window_starts = (np.arange(n_wins, dtype=int)[mask_windows] * step_samples) + start_index
    else:
        X = X_full
        window_starts = (np.arange(X.shape[0], dtype=int) * step_samples) + start_index

    logging.info(f"Features: X={X.shape}")
    logging.info(f"Window index range: [{window_starts.min()} .. {window_starts.max()}]  "
                 f"step_samp={step_samples}  start_index={start_index}  base={base}")

    # ---- model & predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()

    nfeat = len(manager.scaler.mean_)
    if X.shape[1] != nfeat:
        raise ValueError(f"Feature dim mismatch (expected {nfeat}) → NPZ={X.shape[1]}")

    y_pred = manager.predict(X)

    # ---- evaluate against events
    try:
        evaluate_against_events(root_dir, window_starts, y_pred)
    except Exception as e:
        logging.info(f"Evaluation skipped/failed: {e}")

    # ---- optional save of cleaned/shifted stream
    if save_cleaned_npz:
        os.makedirs(os.path.dirname(save_cleaned_npz), exist_ok=True)
        np.savez_compressed(
            save_cleaned_npz,
            fs_hz=fs,
            emg=z_emg_fixed,
            ch_names=np.array(z_names_sel, dtype=object),
            z_ts=z_ts,
            o_ts=o_ts,
            meta_json=json.dumps(meta_cache),
            note=f"cleaned={fix_mode}, noise_scale={noise_scale}, lag_samples={lag_samples}, from_stream={C['stream_used']}",
        )
        logging.info(f"[saved] cleaned NPZ → {save_cleaned_npz}")


def parse_args():
    p = argparse.ArgumentParser("Predict from NPZ cache with gap-fix + lag alignment (training-locked) + diagnostics")
    p.add_argument("--config_file", type=str)
    p.add_argument("--cache_npz", type=str, required=True)
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--window_ms", type=int, default=None)
    p.add_argument("--step_ms", type=int, default=None)

    p.add_argument("--fix_mode", choices=["interp", "noise1k", "zero", "none"], default="interp")
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--lag_samples", type=int, default=None)
    p.add_argument("--prefer_recon", action="store_true", help="Use recon_z if available (default False).")
    p.add_argument("--save_cleaned_npz", type=str, default=None)
    p.add_argument("--skip_bad_windows", type=float, default=None, help="Drop windows where ANY selected channel has >= this fraction repaired (e.g., 0.10).")
    p.add_argument("--lag_refine_search", type=int, default=0, help="If >0, refine lag within ±N samples on ch0 by NCC.")
    p.add_argument("--no_diagnose", action="store_true")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update({
        "cache_npz": args.cache_npz,
        "root_dir": args.root_dir,
        "label": args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", None),
        "step_ms": args.step_ms or cfg.get("step_ms", None),
        "fix_mode": args.fix_mode or cfg.get("fix_mode", "interp"),
        "noise_scale": float(args.noise_scale if args.noise_scale is not None else cfg.get("noise_scale", 1.0)),
        "lag_samples": args.lag_samples if args.lag_samples is not None else cfg.get("lag_samples", None),
        "prefer_recon": bool(args.prefer_recon if args.prefer_recon is not None else cfg.get("prefer_recon", False)),
        "save_cleaned_npz": args.save_cleaned_npz or cfg.get("save_cleaned_npz", None),
        "skip_bad_windows": args.skip_bad_windows if args.skip_bad_windows is not None else cfg.get("skip_bad_windows", None),
        "lag_refine_search": int(args.lag_refine_search if args.lag_refine_search is not None else cfg.get("lag_refine_search", 0)),
        "verbose": args.verbose or cfg.get("verbose", False),
        "diagnose": not args.no_diagnose,
    })
    run(**cfg)
    # Working!