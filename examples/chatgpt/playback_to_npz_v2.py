#!/usr/bin/env python
"""
Stream EMG from Open Ephys via ZMQClient and save a minimal OEBin-compatible NPZ:

  amplifier_data : (C, S) float32
  sample_rate    : float
  t_amplifier    : (S,) float64
  channel_names  : (C,) object

Also writes a sidecar "<out>.quality.json" with NaN/Inf metrics (unless --no_quality_json).

Examples
--------
# Match the reference recording length & sample rate from an OEBin path/dir:
python playback_to_npz.py \
  --file_path "G:\\...\\raw\\gestures" \
  --out "G:\\...\\_oe_cache\\zmq_out.npz"

# Or capture by explicit duration or samples:
python playback_to_npz.py --file_path "..." --out out.npz --duration_s 75
python playback_to_npz.py --file_path "..." --out out.npz --samples 150000 --sample_rate 2000
"""
from __future__ import annotations

import argparse, os, time, json
from pathlib import Path
import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.io import load_oebin_file


# ------------------------------
# Utilities
# ------------------------------
def _ensure_out_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _map_zmq_to_oebin_order(client: ZMQClient, oebin_channel_names: list[str]) -> list[str]:
    """Return ZMQ names re-ordered to match OEBin's channel order and apply selection on the client."""
    # wait for channel names to appear
    for _ in range(200):
        zmq_names = list(client.channel_names or [])
        if zmq_names:
            break
        time.sleep(0.02)
    if not zmq_names:
        raise RuntimeError("ZMQ reported no channel names.")

    name_to_idx = {nm: i for i, nm in enumerate(zmq_names)}
    missing = [nm for nm in oebin_channel_names if nm not in name_to_idx]
    if missing:
        raise RuntimeError(f"ZMQ is missing required channels (showing up to 10): {missing[:10]}")

    idx = [name_to_idx[nm] for nm in oebin_channel_names]
    client.set_channel_index(idx)
    return [zmq_names[i] for i in idx]


def _longest_run(mask_1d: np.ndarray) -> int:
    """Longest consecutive True run in a boolean 1-D array."""
    if mask_1d.size == 0:
        return 0
    v = mask_1d.view(np.int8)
    d = np.diff(np.concatenate(([0], v, [0])))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return 0 if starts.size == 0 else int((ends - starts).max())


def compute_nan_metrics(emg: np.ndarray, fs: float, ch_names: list[str]) -> dict:
    bad = ~np.isfinite(emg)
    per_ch_missing = bad.sum(axis=1)
    per_ch_pct = per_ch_missing / emg.shape[1] * 100.0
    per_ch_longest = np.array([_longest_run(bad[i]) for i in range(emg.shape[0])], dtype=int)

    return {
        "channels": len(ch_names),
        "samples": int(emg.shape[1]),
        "total_missing": int(bad.sum()),
        "pct_missing": float(bad.sum() / emg.size * 100.0),
        "per_channel": [
            {
                "name": ch_names[i],
                "missing": int(per_ch_missing[i]),
                "pct": float(per_ch_pct[i]),
                "longest_run_samples": int(per_ch_longest[i]),
                "longest_run_ms": float(per_ch_longest[i] * 1000.0 / fs),
            }
            for i in range(emg.shape[0])
        ],
    }


def bounded_fill_small_gaps(emg: np.ndarray, max_run_samples: int) -> tuple[np.ndarray, int]:
    """
    Repeat-last-value fill only for runs <= max_run_samples. Longer runs are kept as NaN/Inf.
    Returns (filled_emg, total_samples_filled).
    """
    if max_run_samples <= 0:
        return emg, 0

    X = emg.copy()
    total_filled = 0
    C, S = X.shape
    for c in range(C):
        row = X[c]
        bad = ~np.isfinite(row)
        if not bad.any():
            continue

        last_val = np.nan
        i = 0
        while i < S:
            if not bad[i]:
                last_val = row[i]
                i += 1
            else:
                j = i
                while j < S and bad[j]:
                    j += 1
                run = j - i
                if run <= max_run_samples and np.isfinite(last_val):
                    row[i:j] = last_val
                    total_filled += run
                # else: leave as missing for downstream QC/skip
                i = j
    return X, total_filled


# ------------------------------
# Core capture
# ------------------------------
def capture_and_save(
    *,
    out_npz: str,
    oebin_names: list[str],
    host_ip: str = "127.0.0.1",
    data_port: str = "5556",
    heartbeat_port: str | None = None,
    buffer_seconds: float = 180.0,
    ready_timeout_s: float = 5.0,
    duration_s: float | None = None,
    samples:   int   | None = None,
    sample_rate_hint: float | None = None,
    start_acquisition: bool = True,
    align_to_header_index: bool = True,
    fill_small_gaps_ms: float = 0.0,
    write_quality_json: bool = True,
    verbose: bool = True,
):
    """
    Capture streaming EMG + timestamps from OE GUI via ZMQ and save an NPZ with minimal OEBin-like keys.
    * If `samples` provided: capture until that many samples acquired (truncate if slightly over).
    * Else if `duration_s` provided: capture until ≈ duration * fs (fs inferred from timestamps if needed).
    """
    # Create client
    client = ZMQClient(
        host_ip=host_ip,
        data_port=data_port,
        heartbeat_port=heartbeat_port,
        buffer_seconds=buffer_seconds,
        auto_start=False,
        expected_channel_count=None,
        set_index_looping=False,
        align_to_header_index=align_to_header_index,
        verbose=verbose,
    )

    # Try to control GUI
    try:
        if start_acquisition:
            client.gui.idle()
            time.sleep(0.1)
            client.gui.start_acquisition()
    except Exception:
        if verbose:
            print("[info] GUI control not available; assuming acquisition is already running.")

    # Start streaming
    client.start()
    if not client.ready_event.wait(ready_timeout_s):
        client.stop()
        raise TimeoutError("ZMQClient not ready; no data received.")

    # Accumulate chunks
    last_seen = int(getattr(client, "global_sample_index", 0))
    chunks, ts_chunks = [], []
    total = 0
    target_samples = samples
    inferred_fs = None

    if verbose:
        print("[stream] capturing… Ctrl+C to stop.")

    try:
        t0 = time.time()
        while True:
            gsi = int(getattr(client, "global_sample_index", 0))
            new_n = gsi - last_seen
            if new_n <= 0:
                time.sleep(0.003)
                # duration-based stop if fs known
                if (target_samples is None) and (inferred_fs is not None) and (duration_s is not None):
                    if total >= int(duration_s * inferred_fs):
                        break
                continue

            try:
                Y_new, T_new = client.get_latest(new_n)
            except Exception:
                time.sleep(0.002)
                continue

            if Y_new is None or Y_new.size == 0:
                time.sleep(0.002)
                continue

            # Normalize timestamps -> 1-D (S,)
            ts = None
            if T_new is not None and np.size(T_new) > 0:
                Tarr = np.asarray(T_new)
                if Tarr.ndim == 2:
                    ts = np.asarray(Tarr[0], dtype=np.float64, order="C")
                elif Tarr.ndim == 1:
                    ts = np.asarray(Tarr, dtype=np.float64, order="C")

                if ts is not None and ts.ndim == 1 and ts.size >= 3 and inferred_fs is None:
                    diffs = np.diff(ts)
                    finite = np.isfinite(diffs)
                    if np.any(finite):
                        m = float(np.nanmedian(diffs[finite]))
                        if m > 0 and np.isfinite(m):
                            inferred_fs = float(round(1.0 / m))
                            if verbose:
                                print(f"[stream] inferred sample rate ≈ {inferred_fs:.3f} Hz from timestamps.")

            chunks.append(np.asarray(Y_new, dtype=np.float32, order="C"))
            if ts is not None:
                ts_chunks.append(ts)

            total += Y_new.shape[1]
            last_seen = gsi

            # stopping criteria
            if target_samples is not None and total >= target_samples:
                break
            if (target_samples is None) and (inferred_fs is not None) and (duration_s is not None):
                if total >= int(duration_s * inferred_fs):
                    break

            # guard if fs can't be inferred
            if (duration_s is not None) and (time.time() - t0 > duration_s * 3.0):
                if verbose:
                    print("[warn] duration guard tripped; stopping capture.")
                break

    except KeyboardInterrupt:
        if verbose:
            print("[stream] interrupted by user.")
    finally:
        try:
            if start_acquisition:
                client.gui.idle()
        except Exception:
            pass
        client.stop()

    if not chunks:
        raise RuntimeError("No data captured from ZMQ stream.")

    # Lock ZMQ channel order to OEBin order (so shapes/names match training)
    locked_names = _map_zmq_to_oebin_order(client, oebin_names)

    # Concatenate & trim
    amplifier_data = np.concatenate(chunks, axis=1)
    if ts_chunks:
        t_concat = np.concatenate(ts_chunks, axis=0)
        if target_samples is not None and t_concat.shape[0] > target_samples:
            t_concat = t_concat[:target_samples]
        t_amplifier = np.asarray(t_concat[:amplifier_data.shape[1]], dtype=np.float64)
    else:
        # fallback: synthesize timestamps if needed
        fs_for_ts = (sample_rate_hint if sample_rate_hint is not None else inferred_fs)
        if fs_for_ts is None:
            raise RuntimeError("No timestamps and sample rate couldn't be inferred; pass --sample_rate.")
        S = amplifier_data.shape[1]
        t_amplifier = np.arange(S, dtype=np.float64) / float(fs_for_ts)

    # Decide sample rate to save
    if sample_rate_hint is not None:
        sample_rate = float(sample_rate_hint)
    elif inferred_fs is not None:
        sample_rate = float(inferred_fs)
    else:
        diffs = np.diff(t_amplifier)
        m = np.nanmedian(diffs[np.isfinite(diffs)]) if diffs.size else 0.0
        if m <= 0:
            raise RuntimeError("Could not determine sample rate.")
        sample_rate = float(1.0 / m)

    # Strict length align
    Smin = min(amplifier_data.shape[1], t_amplifier.shape[0])
    amplifier_data = amplifier_data[:, :Smin].astype(np.float32, copy=False)
    t_amplifier = t_amplifier[:Smin].astype(np.float64, copy=False)

    # --- Quality metrics BEFORE fill
    metrics_before = compute_nan_metrics(amplifier_data, sample_rate, locked_names)

    # --- Optional: bounded small-gap fill (repeat-last) ---
    max_run = int(round((fill_small_gaps_ms or 0.0) * 1e-3 * sample_rate))
    filled_samples = 0
    if max_run > 0:
        amplifier_data, filled_samples = bounded_fill_small_gaps(amplifier_data, max_run)

    # --- Quality metrics AFTER fill
    metrics_after = compute_nan_metrics(amplifier_data, sample_rate, locked_names)
    quality = {
        "fs_hz": float(sample_rate),
        "fill_small_gaps_ms": float(fill_small_gaps_ms or 0.0),
        "fill_method": "repeat_last" if max_run > 0 else "off",
        "filled_samples": int(filled_samples),
        "before": metrics_before,
        "after":  metrics_after,
    }

    # Save NPZ
    out_path = _ensure_out_path(out_npz)
    np.savez_compressed(
        out_path,
        amplifier_data=amplifier_data,
        sample_rate=np.array(sample_rate, dtype=np.float64),
        t_amplifier=t_amplifier,
        channel_names=np.array(locked_names, dtype=object),
    )

    # Save quality JSON
    if write_quality_json:
        with open(str(out_path) + ".quality.json", "w") as f:
            json.dump(quality, f, indent=2)

    if verbose:
        print(
            f"[done] wrote {out_path}\n"
            f"       amplifier_data shape: {amplifier_data.shape}\n"
            f"       sample_rate: {sample_rate:.3f} Hz\n"
            f"       t_amplifier len: {t_amplifier.shape[0]}\n"
            f"       channels: {len(locked_names)}\n"
            f"       quality json: {'saved' if write_quality_json else 'skipped'}"
        )


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Capture EMG via ZMQ and save OEBin-compatible NPZ (+quality JSON).")
    ap.add_argument("--file_path", type=str, required=True,
                    help="Path to an Open Ephys .oebin file or its directory (for channel order & defaults)")
    ap.add_argument("--out", required=True, help="Output NPZ path.")
    ap.add_argument("--host_ip", default="127.0.0.1")
    ap.add_argument("--data_port", default="5556")
    ap.add_argument("--heartbeat_port", default=None)
    ap.add_argument("--buffer_seconds", type=float, default=180.0)
    ap.add_argument("--ready_timeout", type=float, default=5.0)

    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--duration_s", type=float, help="Capture duration in seconds.")
    grp.add_argument("--samples", type=int, help="Capture exact number of samples.")

    ap.add_argument("--sample_rate", type=float, default=None,
                    help="Optional hint if timestamps are missing; also used to synthesize timestamps.")
    ap.add_argument("--fill_small_gaps_ms", type=float, default=0.0,
                    help="Repeat-last fill for NaN/Inf runs ≤ this many ms (0 = off).")
    ap.add_argument("--no_quality_json", action="store_true",
                    help="Do not write the sidecar <out>.quality.json")

    ap.add_argument("--no_start_gui", action="store_true",
                    help="Do not attempt to control OE GUI acquisition.")
    ap.add_argument("--no_align_to_header_index", action="store_true",
                    help="Disable aligning to header index (advanced).")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Pull defaults + channel order from the reference OEBin
    sess = load_oebin_file(args.file_path, verbose=False)
    oebin_names = list(sess["channel_names"])
    S_ref = int(sess["amplifier_data"].shape[1])
    fs_ref = float(sess["sample_rate"])
    print(f"[oebin] reference length: {S_ref} samples @ {fs_ref:.3f} Hz with {len(oebin_names)} channels")

    # If neither duration nor samples is provided, default to the OEBin sample count
    if args.samples is None and args.duration_s is None:
        args.samples = S_ref
    # Default sample rate hint to OEBin's fs if unspecified
    if args.sample_rate is None:
        args.sample_rate = fs_ref

    capture_and_save(
        out_npz=args.out,
        oebin_names=oebin_names,
        host_ip=args.host_ip,
        data_port=args.data_port,
        heartbeat_port=args.heartbeat_port,
        buffer_seconds=args.buffer_seconds,
        ready_timeout_s=args.ready_timeout,
        duration_s=args.duration_s,
        samples=args.samples,
        sample_rate_hint=args.sample_rate,
        start_acquisition=(not args.no_start_gui),
        align_to_header_index=(not args.no_align_to_header_index),
        fill_small_gaps_ms=float(args.fill_small_gaps_ms or 0.0),
        write_quality_json=(not args.no_quality_json),
        verbose=args.verbose or True,
    )


if __name__ == "__main__":
    main()
