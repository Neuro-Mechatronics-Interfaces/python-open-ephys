#!/usr/bin/env python3
import argparse, json, logging
from pathlib import Path
import numpy as np

def contiguous_runs(mask: np.ndarray):
    """Yield (start, end_inclusive) for True runs in a boolean mask."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, splits)
    return [(int(c[0]), int(c[-1])) for c in chunks]

def repeat_segment_fill_all_channels(Z: np.ndarray):
    """
    Fill NaN/Inf spans by copying the immediately preceding segment
    of the SAME length within the channel. If not enough history
    exists (or the file starts with NaNs), we tile whatever finite
    prefix exists; if none, we forward-fill from the first finite value.
    Returns (filled_array, filled_mask, per_channel_counts).
    """
    X = np.array(Z, dtype=np.float64, copy=True)
    C, S = X.shape
    filled_mask = np.zeros_like(X, dtype=bool)
    counts = np.zeros(C, dtype=int)

    for c in range(C):
        v = X[c]
        bad = ~np.isfinite(v)
        if not bad.any():
            continue

        # Process left-to-right so earlier fills can serve as history
        for i0, i1 in contiguous_runs(bad):
            L = i1 - i0 + 1
            # Preferred source: the L samples immediately before the gap
            src_start = i0 - L
            src_end   = i0

            if src_start >= 0:
                src = v[src_start:src_end]
                # if the source includes non-finite (overlapping earlier gap),
                # it’s already been filled (we process left→right), so it’s safe
            else:
                # not enough history — use whatever finite prefix exists
                finite_prefix = v[:i0][np.isfinite(v[:i0])]
                if finite_prefix.size > 0:
                    # tile to required length
                    reps = int(np.ceil(L / finite_prefix.size))
                    src = np.tile(finite_prefix, reps)[:L]
                else:
                    # channel starts with NaNs — forward-fill from first finite value
                    j = i1 + 1
                    while j < S and not np.isfinite(v[j]):
                        j += 1
                    fill_val = v[j] if j < S else 0.0
                    src = np.full(L, fill_val, dtype=np.float64)

            # If we had a valid historical block but its length < L (can happen when src_start < 0)
            if src.shape[0] != L:
                reps = int(np.ceil(L / src.shape[0]))
                src = np.tile(src, reps)[:L]

            v[i0:i1+1] = src
            filled_mask[c, i0:i1+1] = True
            counts[c] += L

    return X, filled_mask, counts

def main():
    ap = argparse.ArgumentParser("Fill NPZ non-finite spans by repeating preceding segment (per-channel).")
    ap.add_argument("--cache_npz", required=True, help="Path to existing cache NPZ.")
    ap.add_argument("--out_npz", default="", help="Optional output path (default: adds _filledhold.npz).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(levelname)s] %(message)s")

    in_path = Path(args.cache_npz)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.out_npz) if args.out_npz else in_path.with_name(in_path.stem + "_filledhold.npz")

    logging.info(f"Loading NPZ: {in_path}")
    with np.load(in_path, allow_pickle=True) as F:
        # load everything so we can pass-through unchanged
        payload = {k: F[k] for k in F.files}

    # choose stream (same logic as your other scripts)
    stream = "recon_z" if ("recon_z" in payload and payload["recon_z"].size) else "z_emg"
    Z = np.array(payload[stream], dtype=np.float64, copy=True)  # (C, S)
    ch_names = list(payload.get("ch_names_z", []))
    fs = float(payload.get("fs_hz", 0.0))
    logging.info(f"Stream={stream}  fs={fs:.3f} Hz  shape={Z.shape}  channels={len(ch_names)}")

    filled, filled_mask, counts = repeat_segment_fill_all_channels(Z)

    total_filled = int(filled_mask.sum())
    per_ch_pct = 100.0 * counts / filled.shape[1]
    logging.info(f"Filled samples (total): {total_filled}")
    logging.info(f"Per-channel filled %% (first 15): {np.round(per_ch_pct[:15], 2)}")

    # write out a new NPZ; keep everything else identical, but store the filled data
    payload[stream] = filled
    # annotate meta so downstream logs show the fill mode
    meta = {}
    try:
        meta = json.loads(str(payload["meta_json"].item()))
    except Exception:
        pass
    meta = meta or {}
    meta.setdefault("cache", {})["fill_mode"] = "repeat_segment"
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)

    # optional: store a mask for debugging
    payload["filled_mask"] = filled_mask
    payload["filled_counts_per_channel"] = counts

    np.savez_compressed(out_path, **payload)
    logging.info(f"[ok] wrote → {out_path}")

    # quick sanity peek: print a couple of channels with most filling
    worst = np.argsort(counts)[::-1][:5]
    for idx in worst:
        cname = ch_names[idx] if idx < len(ch_names) else f"CH{idx+1}"
        logging.info(f"[diag] {cname}: filled {counts[idx]} samples "
                     f"({per_ch_pct[idx]:.2f}%)")

if __name__ == "__main__":
    main()
