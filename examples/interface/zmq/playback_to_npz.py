#!/usr/bin/env python
"""
Stream EMG from Open Ephys via ZMQClient and save a minimal OEBin-compatible NPZ:

  amplifier_data : (C, S) float32
  sample_rate    : float
  t_amplifier    : (S,) float64
  channel_names  : (C,) object

Usage example:
  python zmq_capture_to_oebin_npz.py \
    --out "G:\\path\\to\\capture_emg_oebin_compat.npz" \
    --duration_s 75 \
    --host_ip 127.0.0.1 --data_port 5556 --ready_timeout 5

If you prefer exact samples instead of duration:
  python zmq_capture_to_oebin_npz.py --samples 150000 --sample_rate 2000 --out out.npz
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import numpy as np

# pyoephys
from pyoephys.interface import ZMQClient
from pyoephys.io import load_oebin_file


def _ensure_out_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# --- NEW: make ZMQ channel order match the OEBin order exactly ---
def _map_zmq_to_oebin_order(client, oebin_channel_names):
    # wait until names appear
    for _ in range(100):
        zmq_names = list(client.channel_names or [])
        if zmq_names:
            break
        time.sleep(0.02)
    if not zmq_names:
        raise RuntimeError("ZMQ reported no channel names.")

    # exact-name map
    idx = []
    zmap = {nm: i for i, nm in enumerate(zmq_names)}
    missing = [nm for nm in oebin_channel_names if nm not in zmap]
    if missing:
        raise RuntimeError(f"ZMQ missing channels: {missing[:10]}{'...' if len(missing)>10 else ''}")
    for nm in oebin_channel_names:
        idx.append(zmap[nm])

    client.set_channel_index(idx)
    return [zmq_names[i] for i in idx]



def capture_and_save(
    out_npz: str,
    host_ip: str = "127.0.0.1",
    data_port: str = "5556",
    heartbeat_port: str | None = None,
    buffer_seconds: float = 180.0,
    ready_timeout_s: float = 5.0,
    duration_s: float | None = 60.0,
    samples: int | None = None,
    sample_rate_hint: float | None = None,
    start_acquisition: bool = True,
    align_to_header_index: bool = True,
    verbose: bool = True,
):
    """
    Capture streaming EMG + timestamps from OE GUI via ZMQ and save an NPZ with the
    minimal OEBin-like keys.

    If `samples` is provided, we capture that many samples. Otherwise we capture
    approximately `duration_s * fs` samples (fs inferred from timestamps if needed).
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
            #client.wait_for_expected_channels(timeout=5.0)

    except Exception:
        if verbose:
            print("[info] GUI control not available; assuming acquisition is already running.")

    # Start streaming
    client.start()
    if not client.ready_event.wait(ready_timeout_s):
        client.stop()
        raise TimeoutError("ZMQClient not ready; no data received.")

    # Accumulate chunks until we hit the target
    last_seen = int(getattr(client, "global_sample_index", 0))
    chunks = []
    ts_chunks = []
    total = 0
    target_samples = samples  # may be None initially
    inferred_fs = None

    if verbose:
        print("[stream] capturing… Ctrl+C to stop.")

    try:
        # Keep fetching new samples
        t0 = time.time()
        while True:
            gsi = int(getattr(client, "global_sample_index", 0))
            new_n = gsi - last_seen
            if new_n <= 0:
                time.sleep(0.003)
                # If target is duration-based and fs is known, we can also break by time
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

            # timestamps: Open Ephys typically gives identical per-channel timestamps.
            # Accept shapes (C, S) or (S,) and normalize to 1-D ts of length S.
            ts = None
            if T_new is not None and np.size(T_new) > 0:
                T_arr = np.asarray(T_new)
                if T_arr.ndim == 2:
                    # Expect (C, S). Use row 0.
                    ts = np.asarray(T_arr[0], dtype=np.float64, order="C")
                elif T_arr.ndim == 1:
                    # Already (S,)
                    ts = np.asarray(T_arr, dtype=np.float64, order="C")
                else:
                    ts = None

                # If we have a plausible 1-D vector, optionally infer fs once.
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

            # Decide stopping condition
            if target_samples is not None:
                if total >= target_samples:
                    break
            else:
                # duration-based
                if (inferred_fs is not None) and (duration_s is not None):
                    if total >= int(duration_s * inferred_fs):
                        break

            # Guardrail against extremely long run if fs couldn't be inferred
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

    # Concatenate
    amplifier_data = np.concatenate(chunks, axis=1)
    if ts_chunks:
        t_concat = np.concatenate(ts_chunks, axis=0)
        # If we overshot target_samples for sample-based capture, trim timestamps
        if target_samples is not None and t_concat.shape[0] > target_samples:
            t_concat = t_concat[:target_samples]
        # If we overshot for duration-based with inferred fs, trim to EMG length
        t_amplifier = np.asarray(t_concat[:amplifier_data.shape[1]], dtype=np.float64)
    else:
        # fallback: synthesize a timestamp vector if needed
        if inferred_fs is None:
            if sample_rate_hint is None:
                raise RuntimeError(
                    "No timestamps and sample rate couldn't be inferred. "
                    "Provide --sample_rate (Hz) to synthesize timestamps."
                )
            inferred_fs = float(sample_rate_hint)
        t0 = 0.0
        S = amplifier_data.shape[1]
        t_amplifier = t0 + np.arange(S, dtype=np.float64) / float(inferred_fs)

    # Pick the sample rate we'll save
    if sample_rate_hint is not None:
        sample_rate = float(sample_rate_hint)
    elif inferred_fs is not None:
        sample_rate = float(inferred_fs)
    else:
        # last resort: estimate from timestamps we just built (should not happen)
        diffs = np.diff(t_amplifier)
        m = np.nanmedian(diffs[np.isfinite(diffs)]) if diffs.size else 0.0
        if m <= 0:
            raise RuntimeError("Could not determine sample rate.")
        sample_rate = float(1.0 / m)

    # Align lengths strictly (truncate to match)
    Smin = min(amplifier_data.shape[1], t_amplifier.shape[0])
    amplifier_data = amplifier_data[:, :Smin].astype(np.float32, copy=False)
    t_amplifier = t_amplifier[:Smin].astype(np.float64, copy=False)
    channel_names = client.channel_names

    out_path = _ensure_out_path(out_npz)
    np.savez_compressed(
        out_path,
        amplifier_data=amplifier_data,
        sample_rate=np.array(sample_rate, dtype=np.float64),
        t_amplifier=t_amplifier,
        channel_names=channel_names,
    )

    if verbose:
        print(
            f"[done] wrote {out_path}\n"
            f"       amplifier_data shape: {amplifier_data.shape}\n"
            f"       sample_rate: {sample_rate:.3f} Hz\n"
            f"       t_amplifier len: {t_amplifier.shape[0]}\n"
            f"       channels: {len(channel_names)}"
        )


def main():
    ap = argparse.ArgumentParser(description="Capture EMG via ZMQ and save OEBin-compatible NPZ.")
    ap.add_argument("--file_path", type=str, required=True, help="Path to an Open Ephys .oebin file or its directory")
    ap.add_argument("--out", required=True, help="Output NPZ path.")
    ap.add_argument("--host_ip", default="127.0.0.1")
    ap.add_argument("--data_port", default="5556")
    ap.add_argument("--heartbeat_port", default=None)
    ap.add_argument("--buffer_seconds", type=float, default=180.0)
    ap.add_argument("--ready_timeout", type=float, default=5.0)

    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--duration_s", type=float, help="Capture duration in seconds.")
    group.add_argument("--samples", type=int, help="Capture exact number of samples.")

    ap.add_argument("--sample_rate", type=float, default=None,
                    help="Optional hint if timestamps are missing; also used to synthesize timestamps.")
    ap.add_argument("--no_start_gui", action="store_true",
                    help="Do not attempt to control OE GUI acquisition.")
    ap.add_argument("--no_align_to_header_index", action="store_true",
                    help="Disable aligning to header index (advanced).")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.file_path:
        sess = load_oebin_file(args.file_path, verbose=False)
        S_ref = int(sess["amplifier_data"].shape[1])
        fs_ref = float(sess["sample_rate"])
        # Prefer OEBIN counts unless user explicitly provided overrides
        if args.samples is None and args.duration_s is None:
            args.samples = S_ref
        if args.sample_rate is None:
            args.sample_rate = fs_ref
        # (Optional) print a small notice
        print(f"[oebin] reference length: {S_ref} samples @ {fs_ref:.3f} Hz")

    capture_and_save(
        out_npz=args.out,
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
        verbose=args.verbose or True,
    )


if __name__ == "__main__":
    main()
