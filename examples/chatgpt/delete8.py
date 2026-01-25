#!/usr/bin/env python3
"""
oebin_npz_align_compare.py

- Load OEBin + NPZ (auto-pick recon_z if present, else z_emg).
- Reorder channels using training metadata (label).
- Fill NPZ NaN/Inf spans with "~1 kHz" noise bounded by prev/next real values.
- Find best lag in ±max_lag samples (per-channel), optionally use global median lag.
- Roll (circular shift) NPZ by chosen lag(s), trim to common length and verify shape.
- Compute RMSE per channel (overall & excluding filled samples).
- Plot top-5 worst channels (by chosen RMSE metric) with shaded dropout spans.
- Optionally save residual plots.

Usage (example):
python oebin_npz_align_compare.py ^
  --root_dir "G:/.../2025_07_31" ^
  --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" ^
  --label "sleeve_15ch_ring" ^
  --max_lag 800 --use_global_lag ^
  --out_dir "./_align_out" --rmse_metric overall --plot_secs 5 --zoom_secs 1.0
"""

import os, json, argparse, logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

import matplotlib.pyplot as plt

from pyoephys.io import load_oebin_file, load_metadata_json, lock_params_to_meta, normalize_name

# ------------------------
# Utils
# ------------------------

def _load_npz(cache_npz: str):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs = float(F["fs_hz"])
        ch_names = list(F["ch_names_z"])
        stream = "recon_z" if ("recon_z" in F and F["recon_z"].size) else "z_emg"
        emg = F[stream]
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(emg.shape[1]) / fs
        o_ts = F["o_ts"] if "o_ts" in F else np.array([], dtype=np.float64)
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    return {"emg": emg, "fs": fs, "z_ts": z_ts, "o_ts": o_ts, "ch_names": ch_names, "stream": stream, "meta": meta}

def _select_training_channels_by_name(emg: np.ndarray, raw_names, trained_names):
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    names_ordered = [raw_names[i] for i in idx]
    return emg[idx, :], idx, names_ordered

def contiguous_runs(mask: np.ndarray):
    """Yield (start, end_inclusive) for True runs in a boolean mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    # split where gaps >1
    splits = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, splits)
    return [(int(c[0]), int(c[-1])) for c in chunks]

def fill_nonfinite_with_1khz_noise(x: np.ndarray, fs: float):
    """
    For each channel (row), replace NaN/Inf runs with a high-frequency sequence:
    - base = alternating ±1 (≈ Nyquist / ~1kHz for fs=2kHz)
    - small random jitter added
    - value range bounded by prev & next finite samples
    - cosine fade at edges to avoid clicks
    Returns filled array and boolean mask of filled samples per channel.
    """
    X = np.array(x, dtype=np.float64, copy=True)
    C, S = X.shape
    filled_mask = np.zeros_like(X, dtype=bool)

    for c in range(C):
        v = X[c]
        bad = ~np.isfinite(v)
        if not bad.any():
            continue
        runs = contiguous_runs(bad)
        for i0, i1 in runs:
            L = i1 - i0 + 1
            # prev/next finite
            prev_idx = i0 - 1
            while prev_idx >= 0 and not np.isfinite(v[prev_idx]):
                prev_idx -= 1
            next_idx = i1 + 1
            while next_idx < S and not np.isfinite(v[next_idx]):
                next_idx += 1
            prev_val = v[prev_idx] if prev_idx >= 0 else 0.0
            next_val = v[next_idx] if next_idx < S else 0.0

            lo, hi = (prev_val, next_val) if prev_val <= next_val else (next_val, prev_val)
            mid = 0.5 * (lo + hi)
            amp = 0.5 * (hi - lo)
            if amp == 0:
                amp = max(1.0, abs(mid) * 0.05)

            # 1 kHz-ish alternating pattern + small noise
            base = np.where(np.arange(L) % 2 == 0, 1.0, -1.0)  # square wave @ fs/2
            jitter = 0.1 * (2 * np.random.rand(L) - 1.0)
            y = mid + amp * (0.9 * base + 0.1 * jitter)

            # short cosine fade at edges
            w = min(16, L // 2)
            if w > 0:
                ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w)))
                # blend from prev_val to filled, then to next_val
                y[:w] = (1 - ramp) * prev_val + ramp * y[:w]
                y[-w:] = ramp[::-1] * y[-w:] + (1 - ramp[::-1]) * next_val

            v[i0:i1+1] = y
            filled_mask[c, i0:i1+1] = True

    return X, filled_mask

def best_lag_smallsearch(a: np.ndarray, b: np.ndarray, max_lag: int):
    """
    Search correlation over integer lags in [-max_lag, +max_lag].
    Returns (best_lag, best_corr).
    """
    assert a.shape == b.shape
    n = a.size
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a -= a.mean(); b -= b.mean()
    sa = a.std() + 1e-12; sb = b.std() + 1e-12

    def corr_at(lag):
        if lag > 0:
            x, y = a[lag:], b[:-lag]
        elif lag < 0:
            x, y = a[:lag], b[-lag:]
        else:
            x, y = a, b
        if x.size == 0:
            return -np.inf
        return float(np.dot((x / sa), (y / sb)) / x.size)

    best_r = -np.inf; best_l = 0
    for L in range(-max_lag, max_lag + 1):
        r = corr_at(L)
        if r > best_r:
            best_r, best_l = r, L
    return best_l, best_r

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def rmse_excluding(mask, a, b):
    """mask=True where samples are to be excluded."""
    keep = ~mask
    if keep.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a[keep] - b[keep]) ** 2)))

def shade_mask_runs(ax, t, mask, color="red", alpha=0.15):
    runs = contiguous_runs(mask)
    for i0, i1 in runs:
        ax.axvspan(t[i0], t[i1], color=color, alpha=alpha, linewidth=0)

# ------------------------
# Main
# ------------------------

def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Load training meta for channel names
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")

    # --- Load OEBin
    raw_dir = os.path.join(args.root_dir, "raw")
    D = load_oebin_file(raw_dir, verbose=args.verbose)
    o_fs = float(D["sample_rate"])
    o_t = D["t_amplifier"]
    o_all = D["amplifier_data"]
    o_names_all = list(D.get("channel_names", []))
    logging.info(f"OEBin: fs={o_fs:.3f}  emg={o_all.shape}")

    o_emg, o_idx, o_names = _select_training_channels_by_name(o_all, o_names_all, trained_names)

    # --- Load NPZ
    C = _load_npz(args.cache_npz)
    z_fs = float(C["fs"]); z_ts = np.asarray(C["z_ts"]); z_all = C["emg"]; z_names_all = list(C["ch_names"])
    logging.info(f"NPZ: stream={C['stream']}  fs={z_fs:.3f}  emg={z_all.shape}")

    z_emg, z_idx, z_names = _select_training_channels_by_name(z_all, z_names_all, trained_names)

    # --- Verify fs match
    if abs(o_fs - z_fs) > 1e-6:
        raise RuntimeError(f"Sample rates differ: OEBin={o_fs} vs NPZ={z_fs}")

    fs = o_fs
    # Trim to common length before filling/alignment (we'll roll later and trim again)
    S = min(o_emg.shape[1], z_emg.shape[1])
    o_emg = o_emg[:, :S]
    z_emg = z_emg[:, :S]
    t = np.arange(S) / fs

    # --- Fill NPZ non-finite with ~1kHz noise
    z_filled, filled_mask = fill_nonfinite_with_1khz_noise(z_emg, fs)
    filled_counts = filled_mask.sum(axis=1)

    # --- Find lag per channel
    lags = np.zeros(o_emg.shape[0], dtype=int)
    cors = np.zeros(o_emg.shape[0], dtype=float)
    for c in range(o_emg.shape[0]):
        # use a robust slice (ignore first/last 0.5 s) to avoid edge effects
        a = o_emg[c, :]
        b = z_filled[c, :]
        l, r = best_lag_smallsearch(a, b, args.max_lag)
        lags[c] = l; cors[c] = r
    if args.use_global_lag:
        global_lag = int(np.median(lags))
        logging.info(f"Using GLOBAL median lag = {global_lag} samples ({global_lag/fs:.6f} s)")
        lags[:] = global_lag

    # --- Roll, then trim to common length again (circular as requested)
    S2 = S
    z_shifted = np.vstack([np.roll(z_filled[c], lags[c]) for c in range(z_filled.shape[0])])
    filled_mask_shifted = np.vstack([np.roll(filled_mask[c], lags[c]) for c in range(filled_mask.shape[0])])

    assert o_emg.shape == z_shifted.shape == filled_mask_shifted.shape

    # --- Metrics
    rows = []
    rmse_vals = []
    rmse_ex_vals = []

    for c, name in enumerate(o_names):
        r_all = rmse(o_emg[c], z_shifted[c])
        r_ex = rmse_excluding(filled_mask_shifted[c], o_emg[c], z_shifted[c])
        rmse_vals.append(r_all); rmse_ex_vals.append(r_ex)
        rows.append(dict(
            channel_index=c, channel_name=name, lag=lags[c], lag_secs=lags[c]/fs,
            corr_after=cors[c], rmse_overall=r_all, rmse_excl_filled=r_ex,
            filled_samples=int(filled_counts[c])
        ))

    # --- Save CSV
    import csv
    csv_path = outdir / "oebin_npz_alignment_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logging.info(f"[ok] metrics CSV → {csv_path}")

    # --- Pick top-5 by requested metric
    metric_array = np.array(rmse_vals if args.rmse_metric == "overall" else rmse_ex_vals)
    order = np.argsort(metric_array)[::-1]
    topk = order[:min(5, len(order))]

    # --- Plot overlays + shaded dropout regions
    def _plot_panel(ax, ch_idx, t0=0.0, t1=None, with_legend=False):
        if t1 is None: t1 = t[-1]
        a = int(max(0, np.floor(t0 * fs))); b = int(min(S2, np.ceil(t1 * fs)))
        tt = t[a:b]
        ax.plot(tt, o_emg[ch_idx, a:b], lw=0.8, label="OEBin")
        ax.plot(tt, z_shifted[ch_idx, a:b], lw=0.8, alpha=0.85, label="NPZ (filled+shift)")
        shade_mask_runs(ax, tt, filled_mask_shifted[ch_idx, a:b], color="red", alpha=0.15)
        nm = o_names[ch_idx]
        ax.set_title(f"{nm}  RMSE={rmse_vals[ch_idx]:.2f}  r={cors[ch_idx]:.3f}  lag={lags[ch_idx]}")
        if with_legend:
            ax.legend(loc="upper right")
        ax.set_xlim(tt[0], tt[-1])

    # Full-duration (or first N seconds) plot
    Tplot = args.plot_secs if args.plot_secs > 0 else t[-1]
    fig_h = max(3.0, 2.5 * len(topk))
    fig, axes = plt.subplots(len(topk), 1, figsize=(12, fig_h), sharex=True, constrained_layout=True)
    if len(topk) == 1: axes = [axes]
    for ax, ch in zip(axes, topk):
        _plot_panel(ax, ch, 0.0, min(Tplot, t[-1]), with_legend=True)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(outdir / "topk_worst_rmse.png", dpi=150)
    logging.info(f"[ok] figure → {outdir / 'topk_worst_rmse.png'}")

    # Optional zoom around a middle segment
    if args.zoom_secs > 0:
        tmid = 0.5 * t[-1]
        t0 = max(0.0, tmid - 0.5 * args.zoom_secs)
        t1 = min(t[-1], tmid + 0.5 * args.zoom_secs)
        fig2, axes2 = plt.subplots(len(topk), 1, figsize=(12, fig_h), sharex=True, constrained_layout=True)
        if len(topk) == 1: axes2 = [axes2]
        for ax, ch in zip(axes2, topk):
            _plot_panel(ax, ch, t0, t1, with_legend=(ch == topk[0]))
        axes2[-1].set_xlabel("Time (s)")
        fig2.savefig(outdir / "topk_worst_rmse_zoom.png", dpi=150)
        logging.info(f"[ok] figure → {outdir / 'topk_worst_rmse_zoom.png'}")

    # Optional residual plots
    if args.save_residuals:
        for ch in topk:
            resid = o_emg[ch] - z_shifted[ch]
            np.save(outdir / f"residual_{o_names[ch]}.npy", resid)
        logging.info("[ok] saved residual .npy for top-k channels")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Align NPZ to OEBin, fill dropouts, compare, and plot top-5 worst channels.")
    p.add_argument("--root_dir", required=True, type=str)
    p.add_argument("--cache_npz", required=True, type=str)
    p.add_argument("--label", default="", type=str)
    p.add_argument("--max_lag", type=int, default=800, help="Search window ±samples for best lag.")
    p.add_argument("--use_global_lag", action="store_true", help="Use median lag across channels.")
    p.add_argument("--rmse_metric", choices=["overall","excl_filled"], default="overall",
                   help="Which RMSE to rank channels by for plots.")
    p.add_argument("--out_dir", type=str, default="./_align_out")
    p.add_argument("--plot_secs", type=float, default=5.0, help="Duration to show in top-k plot (0 = full).")
    p.add_argument("--zoom_secs", type=float, default=1.0, help="Zoom panel duration around mid recording (0=off).")
    p.add_argument("--save_residuals", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)
    # Working!
