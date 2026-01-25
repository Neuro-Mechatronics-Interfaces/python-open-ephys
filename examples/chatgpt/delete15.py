#!/usr/bin/env python
import argparse, os, json, math
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def contiguous_runs(bad):
    """Return start,end indices (half-open) of True-runs in boolean array."""
    if bad.ndim != 1: bad = bad.ravel()
    diff = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
    starts = np.nonzero(diff == 1)[0]
    ends   = np.nonzero(diff == -1)[0]
    return starts, ends

def repeat_last_fill_1d(x):
    """Fill non-finites by repeating the last L samples before each L-length gap.
    If no previous samples exist, copy the next L; if neither, fill zeros."""
    y = x.copy()
    bad = ~np.isfinite(y)
    if not bad.any():
        return y, bad
    starts, ends = contiguous_runs(bad)
    for s, e in zip(starts, ends):
        L = e - s
        if s > 0:
            # try to take the last L real samples before the gap
            src_start = max(0, s - L)
            src = y[src_start:s]
            if np.isfinite(src).all() and src.size > 0:
                if src.size >= L:
                    y[s:e] = src[-L:]
                else:
                    # pad by repeating the first available value
                    pad = np.full(L - src.size, src[0])
                    y[s:e] = np.concatenate([pad, src])
                continue
            # if those include NaNs (edge case), fall through
        # no usable history -> use the next L real samples
        if e < y.size and np.isfinite(y[e:min(y.size, e + L)]).all() and (min(y.size, e + L) - e) > 0:
            nxt = y[e:min(y.size, e + L)]
            if nxt.size >= L: y[s:e] = nxt[:L]
            else:
                pad = np.full(L - nxt.size, nxt[-1])
                y[s:e] = np.concatenate([nxt, pad])
        else:
            y[s:e] = 0.0
    return y, bad

def safe_corr(a, b):
    if a.size < 2 or b.size < 2: return np.nan
    a = a - np.mean(a); b = b - np.mean(b)
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0 or db == 0: return np.nan
    return float(np.dot(a, b) / (da * db))

def infer_fs(npz):
    if "fs_hz" in npz.files: return float(npz["fs_hz"])
    if "z_ts" in npz.files and npz["z_ts"].size > 1:
        dt = np.median(np.diff(npz["z_ts"]))
        return 1.0 / dt
    if "o_ts" in npz.files and npz["o_ts"].size > 1:
        dt = np.median(np.diff(npz["o_ts"]))
        return 1.0 / dt
    raise RuntimeError("Could not infer sample rate.")

def compute_lag_samples(fs, o_ts, z_ts, cli_lag=None):
    if cli_lag is not None: return int(cli_lag), "cli"
    if o_ts.size and z_ts.size:
        auto = int(round((o_ts[0] - z_ts[0]) * fs))
        return auto, "auto(o_ts - z_ts)"
    return 0, "default0"

def apply_lag(x, lag):
    if lag == 0: return x
    y = np.full_like(x, np.nan)
    if lag > 0:
        y[:, lag:] = x[:, :-lag]
    else:
        y[:, :lag] = x[:, -lag:]
    return y

# ---------- plotting ----------
def plot_nanmap_and_overlay(all_t, o_all, z_all, z_bad_all, ch_names, out_dir, page_size=16, zoom=None):
    """
    For each channel: OEBin (blue), NPZ after fill+shift (orange), red vertical lines at original NaN indices.
    Produces multiple PNGs if channels > page_size.
    """
    C, S = z_all.shape
    pages = int(math.ceil(C / page_size))
    for p in range(pages):
        i0 = p * page_size
        i1 = min(C, (p + 1) * page_size)
        fig_h = 2.2 * (i1 - i0)
        fig = plt.figure(figsize=(12, fig_h), constrained_layout=True)
        for row, ch in enumerate(range(i0, i1), start=1):
            ax = fig.add_subplot(i1 - i0, 1, row)
            # plot
            ax.plot(all_t, o_all[ch], lw=0.8, label="OEBin", alpha=0.9)
            ax.plot(all_t, z_all[ch], lw=0.8, label="NPZ (filled+shift)", alpha=0.9)
            # shade NaN runs from ORIGINAL z
            starts, ends = contiguous_runs(z_bad_all[ch])
            for s, e in zip(starts, ends):
                ax.axvspan(all_t[s], all_t[e-1], color="red", alpha=0.10, lw=0)
            ax.set_ylabel(ch_names[ch] if ch_names else f"CH{ch+1}")
            if row == 1: ax.legend(loc="upper right", fontsize=8)
            if zoom:
                ax.set_xlim(zoom[0], zoom[1])
            else:
                ax.set_xlim(all_t[0], all_t[-1])
        ax.set_xlabel("Time (s)")
        fname = os.path.join(out_dir, f"allch_overlay_nanmap_page{p+1}.png")
        fig.suptitle("OEBin vs NPZ (filled+shift) with missing segments highlighted", y=0.998, fontsize=11)
        fig.savefig(fname, dpi=130)
        plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare OEBin vs NPZ across all channels, compute similarity, and plot NaN maps.")
    ap.add_argument("--cache_npz", required=True, help="Path to the .npz produced by delete3.py")
    ap.add_argument("--out_dir", required=False, default=".", help="Output directory for figures/CSV")
    ap.add_argument("--lag_samples", type=int, default=None, help="Lag (samples). If omitted, computed from o_ts - z_ts.")
    ap.add_argument("--refine_lag", type=int, default=0, help="(Optional) +/- samples to refine via NCC on one channel (0=off).")
    ap.add_argument("--page_size", type=int, default=16, help="Channels per figure page.")
    ap.add_argument("--zoom", type=float, nargs=2, default=None, help="Optional [t_start t_end] seconds for xlim.")
    ap.add_argument("--no_plots", action="store_true", help="Skip figure generation.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with np.load(args.cache_npz, allow_pickle=True) as npz:
        # arrays present in caches created by delete3.py
        z_raw = np.array(npz["z_emg"])        # (C, S)
        o_raw = np.array(npz["o_emg"])        # (C, S)
        z_ts  = np.array(npz.get("z_ts", []))
        o_ts  = np.array(npz.get("o_ts", []))
        chz   = npz.get("ch_names_z", None)
        chz   = [str(x) for x in (chz.tolist() if chz is not None else [])]
        fs    = infer_fs(npz)

    C, S = z_raw.shape
    t0 = float(z_ts[0]) if z_ts.size else 0.0
    all_t = np.arange(S) / fs + (t0 if not args.zoom else 0.0)

    # fill and lag NPZ:
    z_filled = np.empty_like(z_raw, dtype=float)
    z_bad    = ~np.isfinite(z_raw)
    for c in range(C):
        z_filled[c], _ = repeat_last_fill_1d(z_raw[c])

    lag, lag_src = compute_lag_samples(fs, o_ts, z_ts, args.lag_samples)
    z_shift = apply_lag(z_filled, lag)

    # (optional) refine lag via naive NCC on channel with least NaNs
    if args.refine_lag and args.refine_lag > 0:
        nan_fracs = z_bad.mean(axis=1)
        ref_ch = int(np.argmin(nan_fracs))
        rng = np.arange(-args.refine_lag, args.refine_lag + 1)
        best_lag, best_r = lag, -np.inf
        ref_o = o_raw[ref_ch]
        for dl in rng:
            cand = apply_lag(z_filled[[ref_ch], :], lag + dl)[0]
            mask = np.isfinite(ref_o) & np.isfinite(cand)
            if mask.sum() < 10: continue
            r = safe_corr(ref_o[mask], cand[mask])
            if r > best_r:
                best_r, best_lag = r, lag + dl
        lag = int(best_lag)
        z_shift = apply_lag(z_filled, lag)

    # metrics per channel
    rows = []
    for c in range(C):
        o = o_raw[c]
        z = z_shift[c]
        # original NaN mask from z_raw (before fill & shift, then shifted to align)
        bad_shifted = apply_lag(z_bad[[c], :].astype(float), lag)[0]
        bad_shifted = bad_shifted == 1.0  # convert back to bool
        # masks
        mask_all  = np.isfinite(o) & np.isfinite(z)
        mask_excl = mask_all & (~bad_shifted)
        # RMSE
        def rmse(a,b,m):
            if m.sum() == 0: return np.nan
            d = a[m] - b[m]
            return float(np.sqrt(np.mean(d*d)))
        rmse_all  = rmse(o, z, mask_all)
        rmse_excl = rmse(o, z, mask_excl)
        r_all  = safe_corr(o[mask_all], z[mask_all]) if mask_all.any() else np.nan
        r_excl = safe_corr(o[mask_excl], z[mask_excl]) if mask_excl.any() else np.nan
        nan_pct = float(z_bad[c].mean() * 100.0)
        starts, ends = contiguous_runs(z_bad[c])
        longest = int(0 if len(starts)==0 else np.max(ends - starts))
        rows.append([c, chz[c] if c < len(chz) else f"CH{c+1}", r_all, r_excl, rmse_all, rmse_excl, nan_pct, longest])

    # save CSV
    import csv
    csv_path = os.path.join(args.out_dir, "oebin_npz_similarity_all_channels.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ch_idx","ch_name","r_all","r_excl_bad","rmse_all","rmse_excl_bad","nan_pct","longest_nan_run_samples","lag_applied_samples","lag_source"])
        for r in rows:
            w.writerow(r + [lag, lag_src])
    print(f"[metrics] saved → {csv_path}")
    print(f"[metrics] lag applied (samples): {lag}  source={lag_src}")

    # figures
    if not args.no_plots:
        # time base for plotting after lag: keep same abscissa to show shaded NaNs correctly.
        plot_nanmap_and_overlay(all_t, o_raw, z_shift, z_bad, chz, args.out_dir, page_size=args.page_size, zoom=args.zoom)
        print(f"[fig] saved → {os.path.join(args.out_dir, 'allch_overlay_nanmap_page*.png')}")

if __name__ == "__main__":
    main()
