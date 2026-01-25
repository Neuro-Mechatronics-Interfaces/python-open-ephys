#!/usr/bin/env python3
"""
compare_oebin_vs_npz_features_preds.py

What it does
------------
1) Loads OEBin and NPZ (uses recon_z by default), locks channels by the *training name order*.
2) Uses the training-locked preprocessing/feature params (window/step/env_cut).
3) Builds window indices on the OEBin sample axis (same as your working OEBin script).
4) Extracts features, runs the SAME model on both X matrices, evaluates each vs events.
5) Compares features (mean/median/max |Δ|, corrcoef) and per-window prediction agreement.
6) Prints & plots the first N raw samples and the first N timestamps (+ UNIQUES) for
   **multiple preview channels** (by names and/or indices).
7) Dumps the **first feature window** vectors for both OEBin and NPZ and prints their diffs.

Typical usage
-------------
python compare_oebin_vs_npz_features_preds.py ^
  --root_dir "G:/.../2025_07_31" ^
  --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" ^
  --label "sleeve_15ch_ring" ^
  --preview_channel_names CH17 CH43 ^
  --preview_channel_idxs 0 1 ^
  --preview_n 20 --verbose --plot_small_diff
"""

import os, json, argparse, logging, csv
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

from pyoephys.io import load_oebin_file, load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


# ---------------------------
# Helpers
# ---------------------------

def _shift(arr, k):
    """Shift 1D array by k (right if k>0), zero-filling."""
    if k == 0:
        return arr
    y = np.zeros_like(arr)
    if k > 0:
        y[k:] = arr[:-k]
    else:
        y[:k] = arr[-k:]
    return y

def compare_streams_raw(o_emg: np.ndarray,
                        z_emg: np.ndarray,
                        fs: float,
                        ch_names: list[str],
                        max_lag: int = 5,
                        csv_path: str | None = None):
    """
    Per-channel sample-by-sample comparison of RAW OEBin vs NPZ:
      - RMS (each), RMS of difference
      - correlation at lag=0
      - best correlation over lags in [-max_lag, max_lag] and the lag that achieves it
      - linear fit o ≈ a*z + b (slope, intercept)
    Crops both streams to the overlapping length if lengths differ.
    Optionally write a CSV with one row per channel.
    """
    C = min(o_emg.shape[0], z_emg.shape[0])
    S = min(o_emg.shape[1], z_emg.shape[1])
    if (o_emg.shape[1] != z_emg.shape[1]):
        print(f"[warn] length mismatch (OEBin={o_emg.shape[1]}, NPZ={z_emg.shape[1]}) → "
              f"cropping both to overlap S={S} samples")

    rows = []
    worst = []

    for c in range(C):
        a = o_emg[c, :S].astype(np.float64, copy=False)
        b = z_emg[c, :S].astype(np.float64, copy=False)

        # basic stats
        mu_a, mu_b = a.mean(), b.mean()
        sd_a, sd_b = a.std(ddof=0), b.std(ddof=0)
        rms_a = np.sqrt((a*a).mean())
        rms_b = np.sqrt((b*b).mean())

        # corr at lag 0 (guard zero variance)
        corr0 = np.nan
        if sd_a > 0 and sd_b > 0:
            corr0 = float(np.corrcoef(a, b)[0,1])

        # small-lag search
        best_corr, best_lag = np.nan, 0
        if sd_a > 0 and sd_b > 0:
            bc = -1e9
            bl = 0
            for lag in range(-max_lag, max_lag+1):
                b_shift = _shift(b, lag)
                cc = np.corrcoef(a, b_shift)[0,1]
                if cc > bc:
                    bc, bl = float(cc), lag
            best_corr, best_lag = bc, bl

        # linear fit o ≈ a*z + b (only if variance in b)
        slope, intercept = np.nan, np.nan
        if sd_b > 0:
            A = np.vstack([b, np.ones_like(b)]).T
            slope, intercept = np.linalg.lstsq(A, a, rcond=None)[0]
            slope, intercept = float(slope), float(intercept)

        # RMS difference at lag 0
        rms_diff0 = np.sqrt(((a - b)**2).mean())

        rows.append(dict(
            channel_index=c,
            channel_name=ch_names[c] if ch_names and c < len(ch_names) else f"ch{c}",
            rms_o=rms_a, rms_z=rms_b, rms_diff=rms_diff0,
            corr0=corr0, best_corr=best_corr, best_lag=best_lag,
            slope=slope, intercept=intercept,
        ))

        # track "worst" by low corr or large diff
        score = (np.isnan(corr0) and 1.0) or (1.0 - max(corr0, -1.0))
        worst.append((score, c))

    # print quick summary
    import math
    corr0_vals = [r["corr0"] for r in rows if not math.isnan(r["corr0"])]
    best_corr_vals = [r["best_corr"] for r in rows if not math.isnan(r["best_corr"])]
    rms_ratios = [ (r["rms_o"]/(r["rms_z"]+1e-9)) for r in rows ]
    lags = [r["best_lag"] for r in rows if not math.isnan(r["best_corr"])]

    def q(x, p):
        return float(np.percentile(np.asarray(x), p)) if x else np.nan

    print("\n=== RAW NPZ vs OEBin — sample-by-sample ===")
    print(f"channels={C}  samples/channel={S}  fs={fs:.3f} Hz  lag_search=±{max_lag} samples")
    print(f"corr0: median={q(corr0_vals,50):.3f}  p10={q(corr0_vals,10):.3f}  p90={q(corr0_vals,90):.3f}")
    print(f"best_corr (over lags): median={q(best_corr_vals,50):.3f}  p90={q(best_corr_vals,90):.3f}")
    if lags:
        print(f"best_lag: mode≈{max(set(lags), key=lags.count)}  median={q(lags,50):.1f}")
    print(f"RMS ratio (O/Z): median={q(rms_ratios,50):.3f}  p10={q(rms_ratios,10):.3f}  p90={q(rms_ratios,90):.3f}")

    worst.sort(reverse=True)
    print("\nWorst 5 channels by low corr0 / large diff:")
    for _, c in worst[:5]:
        r = rows[c]
        print(f"  {r['channel_name']:>8s}  corr0={r['corr0']:.3f}  best_corr={r['best_corr']:.3f} @lag={r['best_lag']:+d}  "
              f"RMS(O/Z)={r['rms_o']:.2f}/{r['rms_z']:.2f}  slope={r['slope']:.3f}  intercept={r['intercept']:.3f}")

    if csv_path:
        import csv, os
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[raw compare CSV] saved → {csv_path}")



def envelope_npz_preprocess(x, fs, env_cut_hz=5.0):
    """
    NPZ stream already looks like an envelope (slow, mostly positive).
    Do a light clean + (optional) smoothing to env_cut.
    """
    x = x.astype(np.float64, copy=False)
    # Make sure it's non-negative (some slight negative offsets show up)
    x = np.abs(x)
    # Optional gentle smoothing to env_cut to match OEBin envelope bandwidth
    sos_lp = butter(2, env_cut_hz, btype="low", fs=fs, output="sos")
    # Use the same safe filtfilt helper you already have:
    def _sosfiltfilt_safe(sos, sig):
        padlen = min(3 * (sos.shape[0] * 2), sig.shape[1] - 1)
        if padlen < 1:
            return sosfilt(sos, sig, axis=1)
        try:
            return sosfiltfilt(sos, sig, axis=1, padlen=padlen)
        except Exception:
            return sosfilt(sos, sig, axis=1)
    x = _sosfiltfilt_safe(sos_lp, x)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def safe_npz_preprocess(x, fs, env_cut_hz=5.0):
    """
    Robust EMG preprocessing that avoids NaNs:
    - cast to float64
    - per-channel DC removal
    - 20–450 Hz bandpass (4th-order, zero-phase; fall back to causal if needed)
    - full-wave rectify
    - envelope low-pass at env_cut_hz (2nd-order, zero-phase; causal fallback)
    """
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=1, keepdims=True)

    def _sosfiltfilt_safe(sos, sig):
        padlen = min(3 * (sos.shape[0] * 2), sig.shape[1] - 1)
        if padlen < 1:
            # not enough points for filtfilt – causal fallback
            return sosfilt(sos, sig, axis=1)
        try:
            return sosfiltfilt(sos, sig, axis=1, padlen=padlen)
        except Exception:
            return sosfilt(sos, sig, axis=1)

    # bandpass
    sos_bp = butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")
    x = _sosfiltfilt_safe(sos_bp, x)

    # rectify
    x = np.abs(x)

    # envelope lowpass
    sos_lp = butter(2, env_cut_hz, btype="low", fs=fs, output="sos")
    x = _sosfiltfilt_safe(sos_lp, x)

    # guard against any residual non-finite
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# --- add near the top (imports already there) ---
def summarize_nonfinite(arr: np.ndarray, fs: float, label: str):
    """Print per-channel counts and first/last bad sample index/time."""
    bad = ~np.isfinite(arr)
    C, S = arr.shape
    total_bad = int(bad.sum())
    print(f"{label}: non-finite total={total_bad} of {C*S}")
    if total_bad == 0:
        return None
    first_bad_t = None
    for c in range(C):
        bc = bad[c]
        if bc.any():
            idx = np.where(bc)[0]
            i0, i1, cnt = int(idx[0]), int(idx[-1]), int(idx.size)
            if first_bad_t is None or i0 < first_bad_t:
                first_bad_t = i0
            t0, t1 = i0/fs, i1/fs
            print(f"  ch {c:3d}: bad_count={cnt:6d}  first={i0:6d} ({t0:8.3f}s)  last={i1:6d} ({t1:8.3f}s)")
    return int(first_bad_t) if first_bad_t is not None else None

def fix_nonfinite_per_channel(arr: np.ndarray, mode: str = "interp") -> np.ndarray:
    """
    Replace NaN/Inf per channel.
    mode = 'interp' (linear), 'zero', or 'none'
    """
    if mode == "none":
        return arr
    A = np.array(arr, copy=True)
    C, S = A.shape
    for c in range(C):
        x = A[c]
        bad = ~np.isfinite(x)
        if not bad.any():
            continue
        if mode == "zero":
            x[bad] = 0.0
            continue
        # interp
        good = ~bad
        if good.any():
            xi = np.arange(S)
            # Edge handling: extend edges with nearest good sample
            if not good[0]:
                first = np.where(good)[0][0]
                x[:first] = x[first]
                good[:first] = True
            if not good[-1]:
                last = np.where(good)[0][-1]
                x[last+1:] = x[last]
                good[last+1:] = True
            x[bad] = np.interp(xi[bad], xi[good], x[good])
        else:
            # all bad: just zeros
            x[:] = 0.0
    return A

def match_dc_rms(ref: np.ndarray, tgt: np.ndarray, do_dc: bool, do_rms: bool, fs: float, secs: float = 2.0):
    """Return tgt' matched to ref over the first 'secs' seconds (per-channel)."""
    if not (do_dc or do_rms):
        return tgt
    S = tgt.shape[1]
    k = min(S, int(round(secs*fs)))
    ref0 = ref[:, :k]
    tgt0 = tgt[:, :k]
    out = tgt.copy()
    if do_dc:
        out -= out.mean(axis=1, keepdims=True)
        ref_mu = ref0.mean(axis=1, keepdims=True)
        out += ref_mu
    if do_rms:
        eps = 1e-8
        ref_r = np.sqrt((ref0**2).mean(axis=1, keepdims=True) + eps)
        tgt_r = np.sqrt((tgt0**2).mean(axis=1, keepdims=True) + eps)
        scale = (ref_r + eps) / (tgt_r + eps)
        out = (out.T * scale.ravel()).T
    return out


def _select_training_channels_by_name(emg: np.ndarray,
                                      raw_names: list[str],
                                      trained_names: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (emg_reordered, idx_list, names_in_training_order)."""
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    names_ordered = [raw_names[i] for i in idx]
    return emg[idx, :], idx, names_ordered


def _load_npz(cache_npz: str, use_raw_z: bool):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs_hz = float(F["fs_hz"])
        ch_names_z = list(F["ch_names_z"])
        if (not use_raw_z) and ("recon_z" in F) and F["recon_z"].size:
            emg = F["recon_z"]; stream_used = "recon_z"
        else:
            emg = F["z_emg"];   stream_used = "z_emg"
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(emg.shape[1]) / fs_hz
        o_ts = F["o_ts"] if "o_ts" in F else np.array([], dtype=np.float64)
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    return dict(emg=emg, fs=fs_hz, z_ts=z_ts, o_ts=o_ts, ch_names=ch_names_z, meta=meta, stream_used=stream_used)


def _format_arr(a, maxlen=20, precision=6):
    """Pretty print small vectors with limited length."""
    a = np.asarray(a)
    if a.size > maxlen:
        a = a[:maxlen]
    return np.array2string(a, precision=precision, separator=", ", suppress_small=False)


def _resolve_preview_indices(o_names_sel: list[str],
                             preview_channel_names: list[str] | None,
                             preview_channel_idxs: list[int] | None) -> list[int]:
    """Return a de-duplicated list of indices (in training-locked order) to preview."""
    ind = []
    if preview_channel_idxs:
        for i in preview_channel_idxs:
            if i < 0 or i >= len(o_names_sel):
                raise RuntimeError(f"--preview_channel_idxs contains out-of-range index {i} (0..{len(o_names_sel)-1})")
            ind.append(i)
    if preview_channel_names:
        want_norms = [normalize_name(nm) for nm in preview_channel_names]
        norm_sel = [normalize_name(n) for n in o_names_sel]
        for nm, nm_norm in zip(preview_channel_names, want_norms):
            if nm_norm not in norm_sel:
                raise RuntimeError(f"Preview channel name '{nm}' not found in training-locked set.")
            ind.append(norm_sel.index(nm_norm))
    # De-duplicate preserving order
    seen = set()
    out = []
    for i in ind:
        if i not in seen:
            out.append(i); seen.add(i)
    if not out:
        out = [0]  # default: first channel in training-locked order
    return out


# ---------------------------
# Main
# ---------------------------
def run(root_dir: str,
        cache_npz: str,
        label: str = "",
        window_ms: int | None = None,
        step_ms: int | None = None,
        use_raw_z: bool = False,
        verbose: bool = False,
        plot_small_diff: bool = False,
        preview_channel_names: list[str] | None = None,
        preview_channel_idxs: list[int] | None = None,
        preview_n: int = 20,
        dump_preview_csv: str | None = None,
        nonfinite_strategy: str = "interp",
        dc_match: bool = False,
        rms_match: bool = False,
        plot_nan_region_secs: float = 0.0):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # ---- Load training metadata / params ----
    meta = load_metadata_json(root_dir, label=label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # ---- Load OEBin ----
    raw_dir = os.path.join(root_dir, "raw")
    D = load_oebin_file(raw_dir, verbose=verbose)
    o_fs = float(D["sample_rate"])
    o_t  = D["t_amplifier"]
    o_emg_all = D["amplifier_data"]
    o_names_all = list(D.get("channel_names", []))
    logging.info(f"OEBin: fs={o_fs:.3f} Hz  emg={o_emg_all.shape}  ch={len(o_names_all)}  t0={o_t[0]:.6f}s")

    # Lock channels by training names (OEBin)
    o_emg, o_idx, o_names_sel = _select_training_channels_by_name(o_emg_all, o_names_all, trained_names)
    logging.info(f"OEBin channel order (training-locked): {list(zip(o_idx, o_names_sel))}")

    # ---- Load NPZ ----
    C = _load_npz(cache_npz, use_raw_z=use_raw_z)
    z_fs = float(C["fs"]); z_ts = np.asarray(C["z_ts"]); o_ts_from_npz = np.asarray(C["o_ts"])
    z_emg_all = C["emg"]; z_names_all = list(C["ch_names"])
    logging.info(f"NPZ: stream={C['stream_used']}  fs={z_fs:.3f} Hz  emg={z_emg_all.shape}  ch={len(z_names_all)}")

    # Lock channels by training names (NPZ)
    z_emg, z_idx, z_names_sel = _select_training_channels_by_name(z_emg_all, z_names_all, trained_names)
    logging.info(f"NPZ channel order (training-locked): {list(zip(z_idx, z_names_sel))}")

    # ---- Resolve preview channels (in training-locked space) ----
    pv_indices = _resolve_preview_indices(o_names_sel, preview_channel_names, preview_channel_idxs)
    pv_names   = [o_names_sel[i] for i in pv_indices]
    logging.info(f"Preview channels (training-locked): {list(zip(pv_indices, pv_names))}")

    # ---- RAW previews (BEFORE preprocessing) + timestamp previews ----
    Np = int(preview_n)
    # Print and (optionally) CSV
    if dump_preview_csv:
        outp = Path(dump_preview_csv); outp.parent.mkdir(parents=True, exist_ok=True)
        fcsv = open(outp, "w", newline=""); wcsv = csv.writer(fcsv)
        wcsv.writerow(["channel_idx","channel_name","i","o_raw","z_raw","o_ts","z_ts"])

    print("\n=== RAW + Timestamp previews (first N per preview channel) ===")
    for j, (pidx, pname) in enumerate(zip(pv_indices, pv_names)):
        raw_o = o_emg[pidx]; raw_z = z_emg[pidx]
        n_o = min(Np, raw_o.size); n_z = min(Np, raw_z.size)
        n_to = min(Np, o_t.size);  n_tz = min(Np, z_ts.size)

        print(f"\n-- Channel {pidx} ({pname}) --")
        print(f"OEBin raw[{pname}][:{n_o}] = { _format_arr(raw_o[:n_o], maxlen=n_o, precision=6) }")
        print(f"NPZ   raw[{pname}][:{n_z}] = { _format_arr(raw_z[:n_z], maxlen=n_z, precision=6) }")

        # Timestamps + uniques
        uo = np.unique(o_t[:n_to]); uz = np.unique(z_ts[:n_tz])
        print(f"OEBin t[:{n_to}] = { _format_arr(o_t[:n_to], maxlen=n_to, precision=6) }")
        print(f"NPZ   t[:{n_tz}] = { _format_arr(z_ts[:n_tz], maxlen=n_tz, precision=6) }")
        print(f"Unique OEBin t (first {n_to}) → count={uo.size} values: { _format_arr(uo, maxlen=uo.size, precision=6) }")
        print(f"Unique NPZ   t (first {n_tz}) → count={uz.size} values: { _format_arr(uz, maxlen=uz.size, precision=6) }")
        if n_to > 1:
            d_o = np.diff(o_t[:n_to]); print(f"OEBin Δt stats (first {n_to-1}): mean={d_o.mean():.9f}, std={d_o.std():.9f}, min={d_o.min():.9f}, max={d_o.max():.9f}")
        if n_tz > 1:
            d_z = np.diff(z_ts[:n_tz]); print(f"NPZ   Δt stats (first {n_tz-1}): mean={d_z.mean():.9f}, std={d_z.std():.9f}, min={d_z.min():.9f}, max={d_z.max():.9f}")

        # CSV rows
        if dump_preview_csv:
            m = max(n_o, n_z, n_to, n_tz)
            for i in range(m):
                wcsv.writerow([
                    pidx, pname,
                    i,
                    float(raw_o[i]) if i < raw_o.size else "",
                    float(raw_z[i]) if i < raw_z.size else "",
                    float(o_t[i]) if i < o_t.size else "",
                    float(z_ts[i]) if i < z_ts.size else "",
                ])

    if dump_preview_csv:
        fcsv.close()
        print(f"[preview CSV] saved → {dump_preview_csv}")

    # ---- After locking channels (o_emg, z_emg) and before any preprocessing ----
    print()
    print("== RAW non-finite summary BEFORE fix ==")
    summarize_nonfinite(o_emg, o_fs, "o_emg")
    summarize_nonfinite(z_emg, z_fs, "z_emg")

    # Fix non-finite on RAW (interpolate by default)
    o_emg = fix_nonfinite_per_channel(o_emg, mode=cfg.get("raw_nonfinite_strategy", "interp"))
    z_emg = fix_nonfinite_per_channel(z_emg, mode=cfg.get("raw_nonfinite_strategy", "interp"))

    print("== RAW non-finite summary AFTER fix ==")
    summarize_nonfinite(o_emg, o_fs, "o_emg")
    summarize_nonfinite(z_emg, z_fs, "z_emg")

    # Sample-by-sample raw comparison
    compare_streams_raw(
        o_emg=o_emg,
        z_emg=z_emg,
        fs=o_fs,
        ch_names=o_names_sel,
        max_lag=cfg.get("raw_max_lag", 5),
        csv_path=cfg.get("raw_compare_csv", None),
    )

    # ---- Plot first N raw samples for preview channels ----
    if len(pv_indices) > 0:
        rows = len(pv_indices)
        fig, axes = plt.subplots(rows, 1, figsize=(12, max(3, 2.5*rows)), constrained_layout=True)
        if rows == 1: axes = [axes]
        for ax, pidx, pname in zip(axes, pv_indices, pv_names):
            raw_o = o_emg[pidx]; raw_z = z_emg[pidx]
            n = min(Np, raw_o.size, raw_z.size)
            # plot vs sample index (0..N-1) for simple visual comparison
            idx = np.arange(n)
            ax.plot(idx, raw_o[:n], marker="o", lw=1.0, label=f"OEBin {pname}")
            ax.plot(idx, raw_z[:n], marker="x", lw=1.0, label=f"NPZ {pname}", alpha=0.8)
            ax.set_title(f"First {n} raw samples — {pname}")
            ax.set_xlabel("sample #")
            ax.set_ylabel("amplitude")
            ax.legend(loc="best")
        plt.show()

    # ---- Preprocessing + features (same settings) ----
    pre_o = EMGPreprocessor(fs=o_fs, envelope_cutoff=env_cut, verbose=verbose)
    o_pp = pre_o.preprocess(o_emg)
    pre_z = EMGPreprocessor(fs=z_fs, envelope_cutoff=env_cut, verbose=verbose)

    def _is_envelope_like(x, fs):
        # Heuristic: how much energy lives above 20 Hz?
        sos_hp = butter(2, 20, btype="high", fs=fs, output="sos")
        hp = sosfiltfilt(sos_hp, x, axis=1)
        rms = lambda a: np.sqrt((a * a).mean())
        return (rms(hp) / (rms(x) + 1e-12)) < 0.05

    mode = cfg.get("npz_preproc", "safe")
    if mode == "emgpreproc":
        z_pp = pre_z.preprocess(z_emg)
    elif mode == "envelope":
        z_pp = envelope_npz_preprocess(z_emg, fs=z_fs, env_cut_hz=env_cut)
    elif mode == "auto":
        if _is_envelope_like(z_emg, z_fs):
            logging.info("NPZ looks envelope-like → using envelope_npz_preprocess()")
            z_pp = envelope_npz_preprocess(z_emg, fs=z_fs, env_cut_hz=env_cut)
        else:
            z_pp = safe_npz_preprocess(z_emg, fs=z_fs, env_cut_hz=env_cut)
    else:  # "safe"
        z_pp = safe_npz_preprocess(z_emg, fs=z_fs, env_cut_hz=env_cut)

    # 1) Summarize NaNs in z_pp (and o_pp for symmetry)
    print()
    t_bad = summarize_nonfinite(o_pp, o_fs, "o_pp")
    t_bad = summarize_nonfinite(z_pp, z_fs, "z_pp")

    # 2) Optional plot around first bad time (raw & preprocessed)
    if (t_bad is not None) and (plot_nan_region_secs > 0):
        w = int(round(plot_nan_region_secs * z_fs))
        a = max(0, t_bad - w)
        b = min(z_pp.shape[1], t_bad + w)
        ch0 = 0
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True)
        ax[0].plot(z_emg[ch0, a:b]);
        ax[0].set_title(f"z_emg ch{ch0} [{a}:{b}]")
        ax[1].plot(z_pp[ch0, a:b]);
        ax[1].set_title(f"z_pp  ch{ch0} [{a}:{b}]  (bad @ {t_bad})")
        for axy in ax: axy.axvline(t_bad - a, ls="--")
        plt.show()

    # 3) Fix non-finite in z_pp before feature extraction
    z_pp_fixed = fix_nonfinite_per_channel(z_pp, mode=nonfinite_strategy)
    summarize_nonfinite(z_pp_fixed, z_fs, "z_pp (after fix)")

    # 4) Optional DC/RMS matching (helps if distributions differ)
    z_pp_fixed = match_dc_rms(o_pp, z_pp_fixed, do_dc=dc_match, do_rms=rms_match, fs=z_fs, secs=2.0)

    # 5) Recompute features from *fixed* z_pp
    X_o = pre_o.extract_emg_features(o_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
                                     tqdm_kwargs={"desc": "OEBin features", "unit": "win", "leave": False})
    X_z = pre_z.extract_emg_features(z_pp_fixed, window_ms=window_ms, step_ms=step_ms, progress=True,
                                     tqdm_kwargs={"desc": "NPZ   features", "unit": "win", "leave": False})
    logging.info(f"Features: OEBin X={X_o.shape}  NPZ X={X_z.shape}")

    def sig_stats(name, x):
        print(f"{name}: rms={np.sqrt((x ** 2).mean()):.3f}  p95={np.percentile(x, 95):.3f}  max={x.max():.3f}")

    sig_stats("o_pp", o_pp)
    sig_stats("z_pp (pre-match)", z_pp)
    # after match_dc_rms(...)
    sig_stats("z_pp (post-match)", z_pp_fixed)

    def report_finite(name, arr):
        bad = ~np.isfinite(arr)
        print(f"{name}: finite={np.isfinite(arr).all()}  NaNs={np.isnan(arr).sum()}  Infs={np.isinf(arr).sum()}")
        if arr.ndim == 2 and bad.any():
            bad_rows = np.where(bad.any(axis=1))[0][:10]
            bad_cols = np.where(bad.any(axis=0))[0][:10]
            print(f"  first bad window idx: {bad_rows}")
            print(f"  first bad feature idx: {bad_cols}")

    z_emg = np.nan_to_num(z_emg, nan=0.0, posinf=0.0, neginf=0.0)

    report_finite("z_emg", z_emg)
    report_finite("z_pp", z_pp)
    report_finite("X_z", X_z)


    # ---- Dump first feature window vectors & diffs ----
    if X_o.shape[0] > 0 and X_z.shape[0] > 0:
        f0_o = X_o[0].astype(float, copy=False)
        f0_z = X_z[0].astype(float, copy=False)
        d0   = f0_o - f0_z
        print("\n=== First feature window (full vectors) ===")
        np.set_printoptions(precision=6, suppress=False, linewidth=220)
        print(f"X_o[0] ({f0_o.size}):\n{f0_o}")
        print(f"X_z[0] ({f0_z.size}):\n{f0_z}")
        print(f"Δ = X_o[0] - X_z[0]:\n{d0}")
        print(f"Δ stats: mean={d0.mean():.6g}  median={np.median(d0):.6g}  min={d0.min():.6g}  max={d0.max():.6g}  L2={np.linalg.norm(d0):.6g}")

    # ---- Window starts on OEBin sample axis (both flows) ----
    start_index_o = int(round(float(o_t[0]) * o_fs))
    step_samples_o = int(round(step_ms / 1000.0 * o_fs))
    ws_o = np.arange(X_o.shape[0], dtype=int) * step_samples_o + start_index_o

    # NPZ mirrors OEBin indexing using OEBin t0 from NPZ if present, else from the OEBin we just loaded
    o_t0_npz = float(o_ts_from_npz[0]) if o_ts_from_npz.size else float(o_t[0])
    start_index_z = int(round(o_t0_npz * z_fs))
    step_samples_z = int(round(step_ms / 1000.0 * z_fs))
    ws_z = np.arange(X_z.shape[0], dtype=int) * step_samples_z + start_index_z

    logging.info(
        "Index ranges:\n"
        f"  OEBin ws[min..max]=[{ws_o.min()} .. {ws_o.max()}]\n"
        f"  NPZ   ws[min..max]=[{ws_z.min()} .. {ws_z.max()}]  (start via OEBin t0)"
    )

    # ---- Model / predict ----
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    nfeat = len(manager.scaler.mean_)
    if X_o.shape[1] != nfeat or X_z.shape[1] != nfeat:
        raise ValueError(f"Feature dim mismatch (expected {nfeat}) → OEBin={X_o.shape[1]} NPZ={X_z.shape[1]}")

    y_o = manager.predict(X_o)
    y_z = manager.predict(X_z)

    # ---- Evaluate each vs events ----
    print("\n=== OEBin evaluation ===")
    try:
        evaluate_against_events(root_dir, ws_o, y_o)
    except Exception as e:
        logging.info(f"OEBin evaluation skipped/failed: {e}")

    print("\n=== NPZ evaluation ===")
    try:
        evaluate_against_events(root_dir, ws_z, y_z)
    except Exception as e:
        logging.info(f"NPZ evaluation skipped/failed: {e}")

    # ---- Direct comparisons (without events) ----
    n = min(X_o.shape[0], X_z.shape[0])
    d = np.abs(X_o[:n] - X_z[:n])
    print("\n=== Feature diff stats (first n windows, OEBin vs NPZ) ===")
    print(f"n_windows_compared = {n}")
    print(f"mean(|Δ|) = {d.mean():.6g}   median(|Δ|) = {np.median(d):.6g}   max(|Δ|) = {d.max():.6g}")
    try:
        r = np.corrcoef(X_o[:n].ravel(), X_z[:n].ravel())[0,1]
        print(f"corrcoef(X_o, X_z) = {r:.6f}")
    except Exception:
        pass

    print("\n=== Prediction agreement (window-by-window) ===")
    agree = (y_o[:n] == y_z[:n])
    print(f"agree_count = {int(agree.sum())} / {n}  ({100.0*agree.mean():.2f}%)")

    # Optional tiny plot of per-window |Δ| (mean over features)
    if plot_small_diff:
        m = d.mean(axis=1)
        plt.figure(figsize=(10,3))
        plt.plot(m)
        plt.title("Per-window mean |Δfeature| (OEBin vs NPZ)")
        plt.xlabel("window #")
        plt.ylabel("mean |Δ|")
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser("Compare OEBin vs NPZ features & predictions + multi-channel raw/timestamp previews")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--cache_npz",   type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--use_raw_z",   action="store_true", help="Use z_emg instead of recon_z from NPZ")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--plot_small_diff", action="store_true")

    # Multi-channel previews
    p.add_argument("--preview_channel_names", nargs="+", default=None,
                   help="One or more channel NAMES to preview (in training-locked order).")
    p.add_argument("--preview_channel_idxs",  nargs="+", type=int, default=None,
                   help="One or more channel indices to preview (in training-locked order).")
    p.add_argument("--preview_n",            type=int, default=20,
                   help="How many initial samples/timestamps to print/plot (default 20).")
    p.add_argument("--dump_preview_csv",     type=str, default=None,
                   help="Optional path to save a CSV of preview rows (i, o_raw, z_raw, o_ts, z_ts) per channel.")
    p.add_argument("--nonfinite_strategy", choices=["none", "interp", "zero"], default="interp",
                   help="How to fix non-finite values in NPZ preprocessed signal before features.")
    p.add_argument("--dc_match", action="store_true", help="Match NPZ DC to OEBin (per-channel).")
    p.add_argument("--rms_match", action="store_true", help="Match NPZ RMS to OEBin over first ~2s (per-channel).")
    p.add_argument("--plot_nan_region_secs", type=float, default=0.0,
                   help="If >0, plot a small window around the first non-finite time in z_pp.")
    p.add_argument("--npz_preproc", choices=["emgpreproc", "safe", "envelope", "auto"], default="safe",
                   help="NPZ pipeline: raw-EMG (emgpreproc/safe), envelope, or auto-detect.")
    p.add_argument("--raw_nonfinite_strategy", choices=["none", "interp", "zero"], default="interp",
                   help="Fix non-finite on RAW o_emg and z_emg before any processing.")
    p.add_argument("--raw_compare_csv", type=str, default=None,
                   help="Optional CSV path to write per-channel raw comparison metrics.")
    p.add_argument("--raw_max_lag", type=int, default=5,
                   help="Max samples of ±lag to search when aligning for best raw correlation.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        root_dir=cfg["root_dir"],
        cache_npz=cfg["cache_npz"],
        label=cfg.get("label",""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        use_raw_z=cfg.get("use_raw_z", False),
        verbose=cfg.get("verbose", False),
        plot_small_diff=cfg.get("plot_small_diff", False),
        preview_channel_names=cfg.get("preview_channel_names", None),
        preview_channel_idxs=cfg.get("preview_channel_idxs", None),
        preview_n=cfg.get("preview_n", 20),
        dump_preview_csv=cfg.get("dump_preview_csv", None),
        nonfinite_strategy=cfg.get("nonfinite_strategy", "interp"),
        dc_match=cfg.get("dc_match", False),
        rms_match=cfg.get("rms_match", False),
        plot_nan_region_secs=cfg.get("plot_nan_region_secs", 0.0),
    )
