#!/usr/bin/env python3
"""
plot_oebin_vs_npz.py
Visual sanity-check for OEBin vs cached NPZ (ZMQ capture). Plots:
  1) Raw overlay vs time (unaligned)
  2) Normalized cross-correlation vs lag
  3) Aligned overlay (using cached or computed lag)

Usage (typical):
  python plot_oebin_vs_npz.py \
    --root_dir "G:/.../2025_07_31" \
    --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" \
    --channel 0 --use_recon_z --match_by_name --verbose
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Your package
from pyoephys.io import normalize_name, load_open_ephys_session


# ----------------------------
# Helpers
# ----------------------------
def ncc_lag(a: np.ndarray, b: np.ndarray, max_lag: int | None = None):
    """Normalized cross-correlation peak; +lag means B later; to align, shift B left by lag."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n = min(a.size, b.size)
    a = a[:n]; b = b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        raise RuntimeError("No finite overlap for NCC.")
    a = (a[finite] - a[finite].mean()) / (a[finite].std() + 1e-12)
    b = (b[finite] - b[finite].mean()) / (b[finite].std() + 1e-12)

    N = len(a)
    L = 1 << (2*N - 1).bit_length()
    fa = np.fft.rfft(a, L); fb = np.fft.rfft(b, L)
    corr = np.fft.irfft(fa * np.conj(fb), L)
    corr = np.concatenate([corr[-(N-1):], corr[:N]])
    lags = np.arange(-N+1, N, dtype=int)

    if max_lag is not None:
        keep = (lags >= -max_lag) & (lags <= max_lag)
        lags, corr = lags[keep], corr[keep]

    k = int(np.argmax(corr))
    return int(lags[k]), float(corr[k]), lags, corr


def align_1d_by_lag(a: np.ndarray, b: np.ndarray, lag: int):
    """
    Align using the NCC convention above.
    +lag means B is later; align by shifting B left by lag → return overlap.
    """
    n = min(a.size, b.size)
    if lag > 0:
        L = n - lag
        return a[:L], b[lag:lag+L]
    elif lag < 0:
        k = -lag; L = n - k
        return a[k:k+L], b[:L]
    else:
        return a[:n], b[:n]


def pick_channel_by_name(emg: np.ndarray, names: list[str], target_name: str) -> tuple[np.ndarray, int]:
    """Return (1D channel array, idx) by matching a channel NAME (case/space-insensitive)."""
    nn = [normalize_name(n) for n in names]
    t = normalize_name(target_name)
    if t not in nn:
        raise ValueError(f"Channel name '{target_name}' not found in stream.")
    idx = nn.index(t)
    return emg[idx], idx


# ----------------------------
# Main plotting
# ----------------------------
def run(root_dir: str,
        cache_npz: str,
        channel: int = 0,
        channel_name: str | None = None,
        use_recon_z: bool = True,
        match_by_name: bool = False,
        max_lag_ms: float | None = None,
        start_s: float | None = None,
        duration_s: float | None = None,
        decimate: int = 1,
        prefer_cached_lag: bool = True,
        verbose: bool = False,
        save_fig: str | None = None):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[% (levelname)s] %(message)s".replace(" ", ""), level=lvl)

    # ---- Load OEBin (from root_dir/raw)
    sess = load_open_ephys_session(root_dir, verbose=verbose)
    o_emg = sess["amplifier_data"]
    o_ts = sess["t_amplifier"]
    o_fs = float(sess["sample_rate"])
    o_names = sess["channel_names"]
    logging.info(f"OEBin: fs={o_fs:.3f} Hz  emg shape={o_emg.shape}  ch={len(o_names)}  t0={o_ts[0]:.6f}s")

    # ---- Load NPZ
    cache_npz = Path(cache_npz)
    with np.load(cache_npz, allow_pickle=True) as F:
        present = sorted(F.files)
        z_fs = float(F["fs_hz"])
        ch_names_z = list(F["ch_names_z"])
        z_emg_all = F["recon_z"] if ("recon_z" in F and F["recon_z"].size and use_recon_z) else F["z_emg"]
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(z_emg_all.shape[1]) / z_fs
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    logging.info(f"NPZ: fields={present} | fs={z_fs:.3f} Hz  emg shape={z_emg_all.shape}  ch_names_z={len(ch_names_z)}")

    # ---- Choose channel (index or name), for BOTH sources
    if match_by_name:
        # Prefer provided name, otherwise use the OEBin's channel name at 'channel'
        target_name = channel_name or (o_names[channel] if o_names else f"CH{channel+1}")
        o_ch, o_idx = pick_channel_by_name(o_emg, o_names, target_name)
        z_ch, z_idx = pick_channel_by_name(z_emg_all, ch_names_z, target_name)
        logging.info(f"Matched by name '{target_name}': OEBin idx={o_idx}, NPZ idx={z_idx}")
    else:
        o_ch = o_emg[channel]
        z_ch = z_emg_all[channel]
        logging.info(f"Matched by index channel={channel}")

    # ---- Optional windowing (zoom)
    if start_s is not None and duration_s is not None:
        o_start = int(max(0, round(start_s * o_fs)))
        o_stop  = int(min(o_emg.shape[1], o_start + round(duration_s * o_fs)))
        z_start = int(max(0, round(start_s * z_fs)))
        z_stop  = int(min(z_emg_all.shape[1], z_start + round(duration_s * z_fs)))
        o_ch = o_ch[o_start:o_stop]
        z_ch = z_ch[z_start:z_stop]
        o_ts_plot = np.arange(o_ch.size) / o_fs
        z_ts_plot = np.arange(z_ch.size) / z_fs
    else:
        o_ts_plot = np.arange(o_ch.size) / o_fs
        z_ts_plot = np.arange(z_ch.size) / z_fs

    # ---- Decimate for plotting speed (no filtering — just for visualization)
    if decimate > 1:
        o_ch = o_ch[::decimate]
        z_ch = z_ch[::decimate]
        o_ts_plot = o_ts_plot[::decimate]
        z_ts_plot = z_ts_plot[::decimate]

    # ---- Compute lag (and optionally use cached lag)
    max_lag = None
    if max_lag_ms is not None and max_lag_ms > 0:
        max_lag = int(round((max_lag_ms / 1000.0) * float(min(o_fs, z_fs))))
    lag_ncc, score, lags, corr = ncc_lag(o_ch, z_ch, max_lag=max_lag)

    lag_cached = int(meta.get("lag_samples", 0))
    lag_to_apply = lag_cached if prefer_cached_lag else lag_ncc

    logging.info(
        f"Lag (NCC) = {lag_ncc} samp ({lag_ncc/float(z_fs):.6f}s), score={score:.4f} | "
        f"cached_lag={lag_cached} → using lag_to_apply={lag_to_apply}"
    )

    # ---- Align and make plots
    a_al, b_al = align_1d_by_lag(o_ch, z_ch, -lag_to_apply)  # shift sign to align as we define it
    L = min(a_al.size, b_al.size)
    t_shared = np.arange(L, dtype=np.float64) / float(z_fs)

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)

    # 1) Raw (unaligned) overlay
    axes[0].plot(o_ts_plot, o_ch, label="OEBin", linewidth=0.8)
    axes[0].plot(z_ts_plot, z_ch, label="NPZ (recon_z)" if use_recon_z else "NPZ (z_emg)", alpha=0.7, linewidth=0.8)
    axes[0].set_title(f"Raw overlay (channel {channel if not match_by_name else target_name})")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")

    # 2) NCC curve
    axes[1].plot(lags/float(z_fs), corr, linewidth=0.8)
    axes[1].axvline(lag_ncc/float(z_fs), ls="--", label=f"NCC max {lag_ncc} samp")
    if prefer_cached_lag:
        axes[1].axvline(lag_cached/float(z_fs), ls="--", color='k', label=f"cached {lag_cached} samp")
    axes[1].set_title("Normalized cross-correlation (OEBin vs NPZ)")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper right")

    # 3) Aligned overlay (overlap only)
    axes[2].plot(t_shared, a_al[:L], label="OEBin (aligned)", linewidth=0.8)
    axes[2].plot(t_shared, b_al[:L], label="NPZ (aligned)", alpha=0.7, linewidth=0.8)
    axes[2].set_title(f"Aligned overlay using {'cached' if prefer_cached_lag else 'NCC'} lag = {lag_to_apply} samp")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")

    if save_fig:
        Path(save_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, dpi=150)
        print(f"[plot] saved → {save_fig}")
    else:
        plt.show()


def parse_args():
    ap = argparse.ArgumentParser("Plot OEBin vs NPZ (ZMQ cache) channel overlay + NCC + aligned overlay")
    ap.add_argument("--root_dir", type=str, required=True, help="Run directory containing raw/.. OEBin files")
    ap.add_argument("--cache_npz", type=str, required=True, help="NPZ path saved by streaming capture script")
    ap.add_argument("--channel", type=int, default=0, help="Channel index (ignored if --match_by_name with --channel_name)")
    ap.add_argument("--channel_name", type=str, default=None, help="Channel NAME to match (if --match_by_name)")
    ap.add_argument("--use_recon_z", action="store_true", help="Use recon_z (window-stitched) if available")
    ap.add_argument("--match_by_name", action="store_true", help="Match channel by NAME across OEBin and NPZ")
    ap.add_argument("--max_lag_ms", type=float, default=None, help="Optional limit for NCC lag search (ms)")
    ap.add_argument("--start_s", type=float, default=None, help="Optional zoom: start time (s)")
    ap.add_argument("--duration_s", type=float, default=None, help="Optional zoom: duration (s)")
    ap.add_argument("--decimate", type=int, default=1, help="Plot every Nth sample for speed")
    ap.add_argument("--prefer_cached_lag", action="store_true", help="Apply cached lag from NPZ meta instead of NCC")
    ap.add_argument("--save_fig", type=str, default=None, help="Path to save figure (png/pdf)")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        root_dir=args.root_dir,
        cache_npz=args.cache_npz,
        channel=args.channel,
        channel_name=args.channel_name,
        use_recon_z=args.use_recon_z,
        match_by_name=args.match_by_name,
        max_lag_ms=args.max_lag_ms,
        start_s=args.start_s,
        duration_s=args.duration_s,
        decimate=args.decimate,
        prefer_cached_lag=args.prefer_cached_lag,
        verbose=args.verbose,
        save_fig=args.save_fig,
    )
