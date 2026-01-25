#!/usr/bin/env python
# predict_npz_cache_refactored.py
"""
Refactored NPZ prediction script.

Highlights
- Loads the *_capture_winverify.npz* (or any cache NPZ) and picks z_emg or recon_z.
- Repairs NaN/Inf per channel with several modes: interp | repeat | zero | none.
  * repeat = for each contiguous NaN run of length L, copy the *previous L samples*
    (if not enough samples exist, it falls back to repeating the last finite value).
- Applies the lag (in samples) stored in NPZ meta (or 0 if missing).
- Extracts windows locked to training params (window_ms=200, step_ms=50, env_cut=5).
- Optionally drops windows that overlap repaired samples ( --skip_bad_windows ).
- Evaluates with the trained model and prints a Classification Report + Confusion Matrix.
"""

from __future__ import annotations
import argparse
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ---- Project-local imports (same modules you’re already using) ----------------
# If your package layout differs, adjust these two imports only.
from pyoephys.io import load_metadata_json, lock_params_to_meta, load_config_file, normalize_name
from pyoephys.prep import EMGPreprocessor            # feature extraction (produces 105 features)
from pyoephys.modeling import load_locked_model      # loads model + label map locked to training
# ------------------------------------------------------------------------------


# ---------------------------
# Utilities
# ---------------------------

def _runs_from_bool(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end_exclusive) indices for contiguous True runs in a 1D boolean array."""
    if mask.ndim != 1:
        mask = mask.ravel()
    runs: List[Tuple[int, int]] = []
    if mask.size == 0:
        return runs
    on = False
    start = 0
    for i, v in enumerate(mask):
        if v and not on:
            on, start = True, i
        elif not v and on:
            on = False
            runs.append((start, i))
    if on:
        runs.append((start, mask.size))
    return runs


def _fill_nonfinite_channel(x: np.ndarray, mode: str = "interp") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Repair a single 1D channel. Returns (fixed, diag).
      mode:
        - "interp": linear interpolate over NaN runs, forward/backward fill at edges
        - "repeat": for each NaN run of length L, copy the previous L samples (or the last finite value)
        - "zero": set NaNs/Infs to zero
        - "none": do nothing
    """
    y = x.astype(np.float64, copy=True)
    bad = ~np.isfinite(y)

    diag = {
        "pct_bad": 100.0 * bad.mean() if bad.size else 0.0,
        "longest_run": 0.0,
        "n_runs": 0,
    }

    if not bad.any() or mode == "none":
        return y, diag

    runs = _runs_from_bool(bad)
    diag["n_runs"] = float(len(runs))
    if runs:
        diag["longest_run"] = float(max(e - s for s, e in runs))

    if mode == "zero":
        y[bad] = 0.0
        return y, diag

    if mode == "interp":
        good = ~bad
        idx = np.arange(y.size, dtype=np.int64)
        if good.any():
            # forward/backward fill edges
            first_good = np.argmax(good)
            last_good = len(good) - 1 - np.argmax(good[::-1])
            y[:first_good] = y[first_good]
            y[last_good + 1:] = y[last_good]
            # linear interpolate interior gaps
            y[bad] = np.interp(idx[bad], idx[good], y[good])
        else:
            y[:] = 0.0
        return y, diag

    if mode == "repeat":
        # For each bad run of length L, paste the previous L samples (or last finite value if not enough history)
        for s, e in runs:
            L = e - s
            if s == 0:
                # No history at all
                fill_val = 0.0
                # if there is a first finite after the run, copy that instead of zero
                after = e if e < y.size else y.size - 1
                nxt = np.argmax(np.isfinite(y[after:])) if after < y.size else 0
                if after + nxt < y.size and np.isfinite(y[after + nxt]):
                    fill_val = y[after + nxt]
                y[s:e] = fill_val
            else:
                have = min(L, s)
                block = y[s - have:s].copy()
                if block.size == 0 or not np.isfinite(block).any():
                    # fall back to last finite scalar
                    k = s - 1
                    while k >= 0 and not np.isfinite(y[k]):
                        k -= 1
                    fill_val = y[k] if k >= 0 else 0.0
                    y[s:e] = fill_val
                else:
                    # tile the block to match length L
                    reps = int(np.ceil(L / block.size))
                    tiled = np.tile(block, reps)[:L]
                    y[s:e] = tiled
        return y, diag

    raise ValueError(f"Unknown fix mode: {mode}")


def _repair_nonfinite(emg: np.ndarray, mode: str = "interp") -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Repair all channels independently.
    Returns (fixed_emg, bad_any_mask, per_channel_diag)
      - bad_any_mask: True wherever *any* channel had non-finite before the repair
    """
    C, S = emg.shape
    fixed = np.empty_like(emg, dtype=np.float64)
    bad_any = np.zeros(S, dtype=bool)
    diag: Dict[int, Dict[str, float]] = {}
    for c in range(C):
        y, d = _fill_nonfinite_channel(emg[c], mode=mode)
        fixed[c] = y
        diag[c] = d
        bad_any |= ~np.isfinite(emg[c])
    return fixed, bad_any, diag


def _apply_lag_roll(emg: np.ndarray, lag: int) -> np.ndarray:
    """Positive lag means np.roll to the RIGHT (delay)."""
    if lag == 0:
        return emg
    return np.roll(emg, shift=lag, axis=1)


def _select_training_channels_by_name(
    emg: np.ndarray, raw_names: List[str], trained_names: List[str]
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Match trained channel names to the current recording (order locked to training)."""
    norm_raw = [normalize_name(n) for n in raw_names]
    index_of = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in index_of]
    if missing:
        raise RuntimeError(f"Recording missing trained channels: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    idx = [index_of[n] for n in want_norm]
    return emg[idx, :], idx, [raw_names[i] for i in idx]


def _mask_windows_from_bad(any_bad: np.ndarray, window: int, step: int, tol: float) -> np.ndarray:
    """
    Return keep_mask over window starts, dropping windows whose fraction of
    repaired samples exceeds tol (0..1).
    """
    S = any_bad.size
    starts = np.arange(0, S - window + 1, step, dtype=np.int64)
    keep = np.ones(starts.size, dtype=bool)
    if tol <= 0:
        return keep
    for i, s in enumerate(starts):
        e = s + window
        frac = any_bad[s:e].mean()
        if frac > tol:
            keep[i] = False
    return keep


# ---------------------------
# Core runner
# ---------------------------

def run(
    cache_npz: str,
    root_dir: str,
    label: str,
    fix_mode: str = "interp",
    skip_bad_windows: float = 0.10,
    prefer_recon: bool = False,
    lag_refine_search: int = 0,
    verbose: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="[%(levelname)s] %(message)s")

    # --- Load cache NPZ ---
    z = np.load(cache_npz, allow_pickle=True)
    meta = z.get("meta", {}).item() if "meta" in z.files else {}
    fs = float(meta.get("fs", z.get("fs", 2000.0)))
    ch_names = list(meta.get("ch_names", [f"CH{i+1}" for i in range(z["z_emg"].shape[0])]))

    # Pick stream: z_emg or recon_z
    have_z = "z_emg" in z.files
    have_recon = "recon_z" in z.files
    if prefer_recon and have_recon:
        emg = np.asarray(z["recon_z"], dtype=np.float64)
        stream_name = "recon_z"
    elif have_z:
        emg = np.asarray(z["z_emg"], dtype=np.float64)
        stream_name = "z_emg"
    elif have_recon:
        emg = np.asarray(z["recon_z"], dtype=np.float64)
        stream_name = "recon_z"
    else:
        raise RuntimeError("Neither 'z_emg' nor 'recon_z' were found in the NPZ.")

    C, S = emg.shape
    logging.info(f"NPZ: stream={stream_name}  fs={fs:.3f} Hz  emg={emg.shape}  ch={C}")

    # --- Training-locked params (window/step/env_cut + trained channel names) ---
    # These are what your model was trained with.
    tlock = load_config_file(root_dir, label)               # returns dict with window_ms, step_ms, env_cut, ch_names, etc.
    window_ms = int(tlock.get("window_ms", 200))
    step_ms   = int(tlock.get("step_ms", 50))
    env_cut   = float(tlock.get("env_cut", 5.0))
    trained_ch_names: List[str] = list(tlock["trained_channel_names"])  # MUST exist in the config

    window = int(round(window_ms * fs / 1000.0))
    step   = int(round(step_ms   * fs / 1000.0))

    # Lock channels to training order
    emg_tr, idx_tr, used_names = _select_training_channels_by_name(emg, ch_names, trained_ch_names)
    logging.info(f"NPZ channel order (training-locked): {list(zip(idx_tr[:8], used_names[:8]))} ...")

    # --- Repair non-finite ---
    fixed, any_bad, diag = _repair_nonfinite(emg_tr, mode=fix_mode)
    total_bad = int(any_bad.sum())
    logging.info(f"Non-finite repaired: {total_bad} samples (mode={fix_mode})")
    logging.info(f"[diag] per-channel non-finite % (first 15): {np.array([diag[i]['pct_bad'] for i in range(min(15, len(diag)))])}")
    logging.info(f"[diag] per-channel longest non-finite run (samples, first 15): {np.array([diag[i]['longest_run'] for i in range(min(15, len(diag)))])}")

    # --- Apply cached lag (and optional small refinement) ---
    lag = int(meta.get("lag_samples", 0))
    logging.info(f"Lag to apply (samples): {lag}  source=cache_meta")

    if lag_refine_search and lag_refine_search > 0 and have_z and have_recon:
        # You can add cross-correlation refinement here if desired.
        pass

    fixed = _apply_lag_roll(fixed, lag)

    # Debug peek
    ch0 = 0
    t0 = meta.get("t0_seconds", 0.0)
    ts = t0 + np.arange(S) / fs
    logging.info(f"[diag] First trained channel: {used_names[ch0]}  (idx={idx_tr[ch0]})")
    logging.info(f"[diag] raw first 10:  {np.round(emg_tr[ch0, :10], 3)}")
    logging.info(f"[diag] time first 10: {np.round(ts[:10], 6)}")
    logging.info(f"[diag] fixed first 10:{np.round(fixed[ch0, :10], 3)}")

    # --- Mask bad windows (optional) ---
    keep_mask = _mask_windows_from_bad(any_bad, window, step, tol=float(skip_bad_windows))
    starts = np.arange(0, fixed.shape[1] - window + 1, step, dtype=np.int64)
    kept_starts = starts[keep_mask]
    logging.info(f"[mask] skip_bad_windows={skip_bad_windows:.3f} → keeping {kept_starts.size}/{starts.size} windows ({100.0*kept_starts.size/starts.size:.1f}%)")

    # --- Features ---
    prep = EMGPreprocessor(fs=fs, window_samples=window, step_samples=step, env_cut=env_cut)
    X = prep.extract_emg_features(fixed, window_starts=kept_starts)  # shape (N, 105)
    logging.info(f"Features: X={X.shape}")

    # --- Load model + labels (locked to training) ---
    model, class_names = load_locked_model(root_dir, label)  # returns a fitted model and an ordered list of class names

    # --- Predict ---
    y_pred = model.predict(X)               # integer indices
    y_pred_names = [class_names[i] for i in y_pred]

    # If truth labels are available via events in NPZ meta, map them to the same window_starts:
    y_true_names = None
    if "window_labels" in meta:
        # meta["window_labels"] should be a list of (start_index, label) aligned to training params,
        # or it could be per-window already. Adjust to your exact structure as needed.
        # Here we assume meta["window_labels"] is a dict {int start_idx: "ClassName"}.
        wl = meta["window_labels"]
        y_true_names = [wl.get(int(s), "Rest") for s in kept_starts]

    # --- Report ---
    if y_true_names is not None:
        print("\n=== Classification Report ===")
        print(classification_report(y_true_names, y_pred_names, labels=class_names, zero_division=0))
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_true_names, y_pred_names, labels=class_names))
    else:
        print("Predictions (first 50):", y_pred_names[:50])
        print("Note: no ground-truth labels in NPZ meta; printed predictions only.")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict from NPZ cache with non-finite repair + lag + window masking")
    p.add_argument("--cache_npz", required=True, help="Path to *_capture_winverify.npz")
    p.add_argument("--root_dir",  required=True, help="Recording root (used to locate model + training lock)")
    p.add_argument("--label",     required=True, help="Model label (folder under root_dir/model)")
    p.add_argument("--fix_mode",  default="interp", choices=["interp", "repeat", "zero", "none"], help="Repair mode for NaN/Inf")
    p.add_argument("--skip_bad_windows", type=float, default=0.10, help="Drop windows with >tol fraction of repaired samples")
    p.add_argument("--prefer_recon", action="store_true", help="Use recon_z over z_emg if both present")
    p.add_argument("--lag_refine_search", type=int, default=0, help="Optional small NCC search around cached lag")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
