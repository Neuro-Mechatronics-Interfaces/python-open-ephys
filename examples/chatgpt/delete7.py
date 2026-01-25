#!/usr/bin/env python3
"""
plot_npz_channel.py

Plot one channel from an NPZ file (recon_z preferred; falls back to z_emg).
Highlights all NaN/Inf regions with a transparent red band.

Usage examples
--------------
python plot_npz_channel.py --npz "path/to/cache.npz" --channel_name CH17
python plot_npz_channel.py --npz "...npz" --channel_idx 0 --start_secs 10 --duration_secs 5
python plot_npz_channel.py --npz "...npz" --channel_name CH43 --stream z_emg --ds 4 --save_png out.png
python plot_npz_channel.py --npz "...npz" --channel_name CH17 --no_nan_shade
"""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_npz_stream(npz_path: str, stream: str = "auto"):
    with np.load(npz_path, allow_pickle=True) as F:
        fs = float(F["fs_hz"]) if "fs_hz" in F else float(F.get("fs", 2000.0))
        ch_names = list(F["ch_names_z"]) if "ch_names_z" in F else list(F.get("ch_names", []))

        if stream == "auto":
            if ("recon_z" in F) and F["recon_z"].size:
                data = F["recon_z"]; used = "recon_z"
            else:
                data = F["z_emg"];   used = "z_emg"
        elif stream in ("recon_z", "z_emg"):
            if stream not in F: raise RuntimeError(f"Stream '{stream}' not in NPZ.")
            data = F[stream]; used = stream
        else:
            raise ValueError("--stream must be 'auto', 'recon_z', or 'z_emg'")

        if "z_ts" in F:
            t = np.asarray(F["z_ts"], dtype=float)
            if t.size != data.shape[1]:
                t = np.arange(data.shape[1]) / fs
        else:
            t = np.arange(data.shape[1]) / fs

        meta = None
        if "meta_json" in F:
            try: meta = json.loads(str(F["meta_json"].item()))
            except Exception: meta = None

    return dict(data=np.asarray(data), fs=fs, t=t, ch_names=ch_names, stream_used=used, meta=meta)


def resolve_channel(ch_names, channel_idx: int | None, channel_name: str | None) -> int:
    if channel_idx is not None:
        if channel_idx < 0 or channel_idx >= len(ch_names):
            raise IndexError(f"--channel_idx out of range 0..{len(ch_names)-1}")
        return channel_idx
    if channel_name is not None:
        try:
            return ch_names.index(channel_name)
        except ValueError:
            def norm(s): return s.strip().lower().replace(" ", "")
            lookup = {norm(n): i for i, n in enumerate(ch_names)}
            key = norm(channel_name)
            if key in lookup: return lookup[key]
            raise ValueError(f"--channel_name '{channel_name}' not found. Available: {ch_names[:10]}{'...' if len(ch_names)>10 else ''}")
    return 0


def fix_nonfinite_1d(x: np.ndarray, mode: str = "interp") -> np.ndarray:
    if mode == "none": return x
    y = x.astype(np.float64, copy=True)
    bad = ~np.isfinite(y)
    if not bad.any(): return y
    if mode == "zero":
        y[bad] = 0.0; return y
    # linear interpolation with edge hold
    n = y.size; xi = np.arange(n)
    good = ~bad
    if not good.any(): return np.zeros_like(y)
    first = np.where(good)[0][0]; last = np.where(good)[0][-1]
    y[:first] = y[first]; y[last+1:] = y[last]
    y[bad] = np.interp(xi[bad], xi[good], y[good])
    return y


def nan_segments(mask: np.ndarray) -> list[tuple[int,int]]:
    """Return list of [start_idx, end_idx) for contiguous True in mask."""
    if mask.ndim != 1: mask = mask.ravel()
    if mask.size == 0 or not mask.any(): return []
    m = mask.astype(np.int8)
    dm = np.diff(np.r_[0, m, 0])
    starts = np.where(dm == 1)[0]
    ends   = np.where(dm == -1)[0]
    return list(zip(starts, ends))  # half-open ranges [s, e)


def main():
    ap = argparse.ArgumentParser("Plot a single NPZ channel with NaN shading.")
    ap.add_argument("--npz", required=True, help="Path to NPZ cache file.")
    ap.add_argument("--channel_idx", type=int, default=None, help="Channel index (0-based).")
    ap.add_argument("--channel_name", type=str, default=None, help="Channel name (e.g., CH17).")
    ap.add_argument("--stream", choices=["auto", "recon_z", "z_emg"], default="auto",
                    help="Which NPZ array to plot.")
    ap.add_argument("--start_secs", type=float, default=0.0, help="Start time (s).")
    ap.add_argument("--duration_secs", type=float, default=None, help="Duration (s). If omitted, plot to end.")
    ap.add_argument("--nonfinite", choices=["interp", "zero", "none"], default="interp",
                    help="How to repair NaN/Inf before plotting (line only; shading always uses original mask).")
    ap.add_argument("--ds", type=int, default=1, help="Downsample (plot every Nth sample).")
    ap.add_argument("--save_png", type=str, default=None, help="If set, save figure to this path.")
    ap.add_argument("--no_nan_shade", action="store_true", help="Disable shading of NaN/Inf regions.")
    ap.add_argument("--nan_alpha", type=float, default=0.25, help="Alpha for NaN/Inf shaded bands.")
    args = ap.parse_args()

    pack = load_npz_stream(args.npz, stream=args.stream)
    Z = pack["data"]; fs = pack["fs"]; t_full = pack["t"]; names = pack["ch_names"]; used = pack["stream_used"]

    if not names or len(names) != Z.shape[0]:
        names = [f"CH{i+1}" for i in range(Z.shape[0])]

    ch = resolve_channel(names, args.channel_idx, args.channel_name)
    x_full = Z[ch].astype(np.float64, copy=False)
    name = names[ch]

    # time slicing
    n = x_full.size
    a = max(0, int(round(args.start_secs * fs)))
    b = n if args.duration_secs is None else min(n, a + int(round(args.duration_secs * fs)))
    if a >= b: raise ValueError("Empty slice after applying start/duration.")
    x = x_full[a:b]
    tt = t_full[a:b] if t_full.size == n else (np.arange(a, b) / fs)

    # mask BEFORE any repair (for shading)
    bad_mask = ~np.isfinite(x)
    bad_cnt = int(bad_mask.sum())
    if bad_cnt > 0:
        print(f"[info] non-finite samples in slice: {bad_cnt} / {x.size}")

    # fix for plotting the trace
    x_fixed = fix_nonfinite_1d(x, mode=args.nonfinite)

    # optional downsample (trace only)
    ds = max(1, int(args.ds))
    x_plot = x_fixed[::ds]
    tt_plot = tt[::ds]

    # stats
    rms = np.sqrt((x_plot * x_plot).mean())
    print(f"[ok] NPZ='{Path(args.npz).name}' stream={used} fs={fs:.3f} Hz  channel={name} (idx={ch})")
    print(f"[ok] window=[{args.start_secs:.3f}s, {args.start_secs + (b-a)/fs:.3f}s] samples={x.size} ds={ds} RMS={rms:.3f}")

    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tt_plot, x_plot, lw=1.0)
    ax.set_title(f"{Path(args.npz).name} — {used} — {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # draw shaded bands for every contiguous NaN/Inf segment (use original timebase, not downsampled)
    if bad_cnt > 0 and not args.no_nan_shade:
        segs = nan_segments(bad_mask)
        # choose a minimal visible width for single-sample gaps
        min_w = (1.0 / fs) if fs > 0 else (tt[-1] - tt[0]) / max(1, len(tt))
        for s, e in segs:
            t0 = tt[s]
            t1 = tt[e-1] + min_w if e > s else (tt[s] + min_w)
            ax.axvspan(t0, t1, color="red", alpha=args.nan_alpha, linewidth=0, zorder=0)
        ax.text(0.01, 0.98, f"NaN/Inf segments: {len(segs)}  (samples: {bad_cnt})",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="red", alpha=0.8)

    fig.tight_layout()
    if args.save_png:
        outp = Path(args.save_png); outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=150)
        print(f"[saved] {outp}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
