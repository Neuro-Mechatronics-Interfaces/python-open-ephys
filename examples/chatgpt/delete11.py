#!/usr/bin/env python3
"""
verify_cache_vs_oebin.py
Compare NPZ cache vs OEBin *after applying lag* to confirm similarity.

- Uses training channel order from metadata.
- Reads lag from NPZ meta_json by default; optional --lag_samples override.
- Repairs NaN/Inf (interp/zero/none) and builds a filled mask.
- Applies lag (shift NPZ LEFT by +lag) and trims to overlapping region.
- Computes per-channel RMSE/corr/slope/intercept/RMS ratio/%filled (optionally excluding filled).
- Writes CSV + "top-k worst RMSE" overlays (full + zoom).

Usage:
  python verify_cache_vs_oebin.py ^
    --root_dir ".../2025_07_31" ^
    --cache_npz ".../_oe_cache/..._winverify.npz" ^
    --label "sleeve_15ch_ring" ^
    --outdir ".../_oe_cache/verify" --plot_topk 5 --exclude_filled
"""

import os, json, argparse, logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pyoephys.io import load_oebin_file, load_metadata_json, lock_params_to_meta, normalize_name

def load_npz(cache_npz: str):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs = float(F["fs_hz"])
        ch_names = list(F["ch_names_z"])
        stream = "recon_z" if ("recon_z" in F and F["recon_z"].size) else "z_emg"
        Z = F[stream]
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(Z.shape[1]) / fs
        meta = {}
        if "meta_json" in F:
            try:
                meta = json.loads(str(F["meta_json"].item()))
            except Exception:
                pass
    lag_samples = None
    # Look for lag stored earlier
    for key in ("lag_samples", "npz_lag_samples", "align_lag_samples"):
        if key in meta:
            try: lag_samples = int(meta[key]); break
            except Exception: pass
    return {"emg": Z, "fs": fs, "ts": np.asarray(z_ts), "names": ch_names,
            "stream": stream, "meta": meta, "lag_samples": lag_samples}

def select_by_training(emg, raw_names, trained_names):
    rn = [normalize_name(n) for n in raw_names]
    idx = []
    for nm in trained_names:
        n = normalize_name(nm)
        if n not in rn: raise RuntimeError(f"Missing channel '{nm}' in recording.")
        idx.append(rn.index(n))
    names = [raw_names[i] for i in idx]
    return emg[idx, :], idx, names

def summarize_nonfinite(A):
    bad = ~np.isfinite(A)
    return int(bad.sum()), bad

def fix_nonfinite_interp(A, return_mask=False):
    A = np.array(A, dtype=np.float64, copy=True)
    C, S = A.shape
    bad_total, mask = summarize_nonfinite(A)
    for c in range(C):
        x = A[c]; bad = ~np.isfinite(x)
        if not bad.any(): continue
        good = ~bad
        xi = np.arange(S)
        # extend edges with nearest good
        if not good[0]:
            i0 = np.where(good)[0][0]
            x[:i0] = x[i0]; good[:i0] = True
        if not good[-1]:
            i1 = np.where(good)[0][-1]
            x[i1+1:] = x[i1]; good[i1+1:] = True
        x[bad] = np.interp(xi[bad], xi[good], x[good])
    return (A, mask) if return_mask else A

def apply_lag_pair(O, Z, lag):
    """
    Positive lag means NPZ is later: shift NPZ LEFT by 'lag'.
    Returns (O_trim, Z_shifted_trim).
    """
    if lag >= 0:
        Zs = Z[:, lag:]
        Oa = O[:, :Zs.shape[1]]
        Za = Zs[:, :Oa.shape[1]]
    else:
        # NPZ earlier: shift right (equivalent to drop from O)
        lag = -lag
        Oa = O[:, lag:]
        Za = Z[:, :Oa.shape[1]]
    n = min(Oa.shape[1], Za.shape[1])
    return Oa[:, :n], Za[:, :n]

def metrics_per_channel(O, Z, filled_mask=None, exclude_filled=False):
    """
    Returns list of dicts with rmse, corr, slope, intercept, rms_o, rms_z, pct_filled.
    If exclude_filled: compute metrics on ~filled==False positions only.
    """
    C, S = O.shape
    rows = []
    for c in range(C):
        o = O[c].astype(np.float64, copy=False)
        z = Z[c].astype(np.float64, copy=False)
        if exclude_filled and filled_mask is not None:
            valid = ~filled_mask[c, :S]
            if valid.any():
                o = o[valid]; z = z[valid]
        # guard
        if o.size == 0 or z.size == 0:
            rmse = np.nan; r = np.nan; slope = np.nan; intercept = np.nan
            r_o = np.nan; r_z = np.nan
        else:
            rmse = float(np.sqrt(np.mean((o - z) ** 2)))
            if np.std(o) > 0 and np.std(z) > 0:
                r = float(np.corrcoef(o, z)[0, 1])
                A = np.vstack([z, np.ones_like(z)]).T
                slope, intercept = np.linalg.lstsq(A, o, rcond=None)[0]
                slope, intercept = float(slope), float(intercept)
            else:
                r = np.nan; slope = np.nan; intercept = np.nan
            r_o = float(np.sqrt(np.mean(o ** 2)))
            r_z = float(np.sqrt(np.mean(z ** 2)))
        pct_fill = 0.0
        if filled_mask is not None:
            pct_fill = 100.0 * float(np.mean(filled_mask[c, :S]))
        rows.append(dict(rmse=rmse, corr=r, slope=slope, intercept=intercept,
                         rms_o=r_o, rms_z=r_z, pct_filled=pct_fill))
    return rows


def make_topk_plots(O, Z, sr, names, top_idx, filled_mask=None, out_png="topk_worst_rmse.png", title_note="", zoom=None):
    k = len(top_idx)
    t = np.arange(O.shape[1]) / sr
    fig, axes = plt.subplots(k, 1, figsize=(12, 2.2*k), sharex=True, constrained_layout=True)
    if k == 1: axes = [axes]
    for ax, ci in zip(axes, top_idx):
        ax.plot(t, O[ci], lw=0.9, label="OEBin")
        ax.plot(t, Z[ci], lw=0.9, alpha=0.9, label="NPZ (filled+shift)")
        if filled_mask is not None:
            bad = filled_mask[ci, :O.shape[1]]
            # add translucent red patches at runs of True
            if bad.any():
                idx = np.flatnonzero(bad)
                # group consecutive indices
                starts = [idx[0]]
                ends = []
                for i in range(1, idx.size):
                    if idx[i] != idx[i-1] + 1:
                        ends.append(idx[i-1]); starts.append(idx[i])
                ends.append(idx[-1])
                for s, e in zip(starts, ends):
                    ax.axvspan(t[s], t[e], color="red", alpha=0.15, linewidth=0)
        ax.set_title(f"{names[ci]}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    if zoom:
        for ax in axes:
            ax.set_xlim(zoom)
    fig.suptitle(title_note)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(cache_npz, root_dir, label="", lag_samples=None, exclude_filled=True,
        nonfinite="interp", topk_count=5, outdir=None, zoom_secs= (36.2, 37.0)):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    meta = load_metadata_json(root_dir, label=label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing channel names (training order).")

    # OEBin
    raw_dir = os.path.join(root_dir, "raw")
    D = load_oebin_file(raw_dir, verbose=False)
    o_fs = float(D["sample_rate"])
    O_all = D["amplifier_data"]; O_names = list(D.get("channel_names", []))

    # NPZ
    C = load_npz(cache_npz)
    z_fs = float(C["fs"]); Z_all = C["emg"]; Z_names = C["names"]
    cache_lag = C["lag_samples"]

    if abs(o_fs - z_fs) > 1e-6:
        raise RuntimeError(f"fs mismatch: OEBin {o_fs} vs NPZ {z_fs}")

    O, o_idx, names = select_by_training(O_all, O_names, trained_names)
    Z, z_idx, _ = select_by_training(Z_all, Z_names, trained_names)

    # Build filled mask (before fix)
    _, filled_mask = summarize_nonfinite(Z)
    if nonfinite == "interp":
        Z_fixed, filled_mask = fix_nonfinite_interp(Z, return_mask=True)
    elif nonfinite == "zero":
        Z_fixed = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Z_fixed = Z.copy()
    logging.info(f"Non-finite in NPZ before fix: {int((~np.isfinite(Z)).sum())} samples")

    # Choose lag
    lag = lag_samples if lag_samples is not None else (cache_lag or 0)
    logging.info(f"Using lag_samples={lag} (positive → shift NPZ left)")

    # Apply lag and trim to common region
    Oa, Za = apply_lag_pair(O, Z_fixed, lag)
    filled_a = filled_mask[:, :Za.shape[1]] if filled_mask is not None else None

    # Metrics
    rows = metrics_per_channel(Oa, Za, filled_mask=filled_a, exclude_filled=exclude_filled)
    rmse = np.array([r["rmse"] for r in rows])
    order = np.argsort(-rmse)  # worst first
    names_sel = names

    # Output dir
    if outdir is None:
        outdir = os.path.join(os.path.dirname(cache_npz), "verify")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(outdir, "oebin_npz_afterlag_metrics.csv")

    # Write CSV
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel","rmse","corr","slope","intercept","rms_o","rms_z","pct_filled"])
        w.writeheader()
        for i, r in enumerate(rows):
            r2 = dict(r); r2["channel"] = names_sel[i]; w.writerow(r2)
    logging.info(f"[ok] metrics → {csv_path}")

    # Plots
    top = order[:max(1, topk_count)]
    full_png = os.path.join(outdir, "topk_worst_rmse.png")
    make_topk_plots(Oa, Za, o_fs, names_sel, top, filled_mask=filled_a,
              out_png=full_png, title_note=f"lag={lag}  exclude_filled={exclude_filled}")
    zoom_png = os.path.join(outdir, "topk_worst_rmse_zoom.png")
    if zoom_secs:
        make_topk_plots(Oa, Za, o_fs, names_sel, top, filled_mask=filled_a,
                  out_png=zoom_png, title_note=f"lag={lag}  exclude_filled={exclude_filled}",
                  zoom=zoom_secs)
    logging.info(f"[ok] figs → {full_png}, {zoom_png}")

    # Summary to console
    valid_rmse = rmse[np.isfinite(rmse)]
    med_corr = np.nanmedian([r["corr"] for r in rows])
    msg = (f"channels={len(rows)}  overlap_samples={Oa.shape[1]}  "
           f"RMSE median={np.nanmedian(valid_rmse):.2f}  p90={np.nanpercentile(valid_rmse,90):.2f}  "
           f"corr median={med_corr:.3f}")
    logging.info(msg)

def parse_args():
    p = argparse.ArgumentParser("Verify NPZ cache vs OEBin after applying lag.")
    p.add_argument("--root_dir", required=True, type=str)
    p.add_argument("--cache_npz", required=True, type=str)
    p.add_argument("--label", default="", type=str)
    p.add_argument("--lag_samples", default=None, type=int,
                   help="Override lag; by default reads from NPZ meta_json.")
    p.add_argument("--exclude_filled", action="store_true",
                   help="Exclude filled samples from metric calculations.")
    p.add_argument("--nonfinite", choices=["interp","zero","none"], default="interp")
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--zoom_start", type=float, default=36.2)
    p.add_argument("--zoom_end", type=float, default=37.0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.cache_npz, a.root_dir, label=a.label,
        lag_samples=a.lag_samples, exclude_filled=a.exclude_filled,
        nonfinite=a.nonfinite, topk_count=a.plot_topk, outdir=a.outdir,
        zoom_secs=(a.zoom_start, a.zoom_end))
