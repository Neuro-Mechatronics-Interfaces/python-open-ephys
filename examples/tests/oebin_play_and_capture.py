#!/usr/bin/env python3
"""
Play back an Open Ephys .oebin session over LSL and simultaneously record the
LSL stream to an .npz. The saved .npz will have the same number of samples as
the source recording (assuming no loopback).

Usage
-----
python examples/oebin_play_and_capture.py \
  --path /data/session_or_oebin \
  --stream_name EMG \
  --out ./exports/capture.npz \
  --recorded_ts \
  --block_ms 10 \
  --verbose

Then visualize:
python examples/lsl_viewer_subplots.py --stream_name EMG --channels all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pyoephys.io import load_open_ephys_session
from pyoephys.interface._playback_client import OEBinPlaybackClient
from pyoephys.interface._lsl_client import LSLClient, NotReadyError
from pyoephys.logging import configure


def _find_oebin(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".oebin":
        return path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    for p in sorted(path.glob("*.oebin")):
        return p.resolve()
    for p in sorted(path.rglob("*.oebin")):
        return p.resolve()
    raise FileNotFoundError(f"No .oebin file found under: {path}")


def _read_fs(oebin: Path) -> Optional[float]:
    try:
        meta = json.loads(oebin.read_text(encoding="utf-8"))
    except Exception:
        return None

    def _listify(x):
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            return list(x.values())
        return []

    cands = []
    if "recordings" in meta:
        for rec in _listify(meta["recordings"]):
            streams = rec.get("streams") or rec
            if "continuous" in streams:
                cands.extend(_listify(streams["continuous"]))
    if "streams" in meta and "continuous" in meta["streams"]:
        cands.extend(_listify(meta["streams"]["continuous"]))

    for s in cands:
        fs = s.get("sample_rate") or s.get("sampleRate") or s.get("rate")
        if fs:
            try:
                return float(fs)
            except Exception:
                pass
    return None


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Play back an .oebin over LSL and record to NPZ with matching sample count.")
    ap.add_argument("--path", required=True, help="Path to a .oebin file OR a directory containing it")
    ap.add_argument("--stream_name", default="EMG", help="LSL stream name (publisher & recorder)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--block_size", type=int, default=None, help="Samples per push (overrides --block_ms)")
    g.add_argument("--block_ms", type=float, default=32.0, help="Chunk duration in ms (converted using fs)")
    ap.add_argument("--recorded_ts", action="store_true", help="Publish recorded timestamps from file")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed factor")
    ap.add_argument("--out", required=True, help="Output .npz path for captured LSL stream")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    configure("INFO" if args.verbose else "WARNING")

    # Load session once to know exact shapes (and to validate input)
    sess = load_open_ephys_session(Path(args.path))
    y_src = np.asarray(sess["amplifier_data"], dtype=np.float32)  # (C, S)
    t_src = np.asarray(sess["t_amplifier"], dtype=np.float64)  # (S,)
    fs = float(sess["sample_rate"])
    ch_names = list(sess.get("channel_names", [f"ch{i}" for i in range(y_src.shape[0])]))

    C, S = y_src.shape
    dur = S / fs

    # Compute playback block size
    bs = args.block_size
    if bs is None:
        bs = max(1, int(round((args.block_ms / 1000.0) * fs)))

    print(f"[setup] channels={C} samples={S} fs={fs:.2f}Hz duration={dur:.3f}s")
    print(f"[playback] block_size={bs} recorded_ts={args.recorded_ts} speed={args.speed}x stream='{args.stream_name}'")
    print(f"[capture] ring buffer >= duration ({dur:.1f}s) to hold the entire run")

    # Start the LSL recorder FIRST (in its own thread internally), with a big buffer
    # so we can fetch the whole recording at the end in one shot.
    buffer_seconds = dur + 5.0  # margin
    lsl_rec = LSLClient(
        stream_name=args.stream_name,
        stream_type="EMG",
        timeout_s=30.0,  # generous resolve time in case outlet is late
        buffer_seconds=buffer_seconds,  # keep entire run
        verbose=args.verbose,
    )
    # start() will resolve then spawn its own pull thread
    # To avoid blocking our main thread (which needs to start playback), just call start() normally:
    # It resolves quickly once playback starts.
    # (If your environment resolves slowly, you can thread this; not necessary on most machines.)
    # Start playback now:
    pb = OEBinPlaybackClient(
        oebin_path=str(args.path),
        stream_name=args.stream_name,
        block_size=int(bs),
        loopback=False,  # We want exactly one pass for a 1:1 sample match
        enable_lsl=True,
        use_recorded_timestamps=bool(args.recorded_ts),
        speed_factor=float(args.speed),
        verbose=args.verbose,
    )
    pb.start()

    # Now start the recorder; it will block resolving until the outlet exists.
    lsl_rec.start()

    # Wait until recorder reports ready (received first chunk)
    t0 = time.time()
    while not lsl_rec.ready_event.is_set():
        if time.time() - t0 > 30:
            raise TimeoutError("Recorder did not receive any LSL data within 30s.")
        time.sleep(0.05)

    # Wait for playback to finish (end of file)
    while pb.is_running():
        time.sleep(0.1)

    # Give a tiny grace period for the last chunk to land
    time.sleep(0.2)

    # Pull exactly S samples out of the recorder's ring buffer
    try:
        y_cap, t_cap = lsl_rec.get_latest(int(S))
    except NotReadyError:
        lsl_rec.stop()
        pb.stop()
        raise SystemExit("Recorder not ready; no data captured.")

    # Clean up
    lsl_rec.stop()
    pb.stop()

    # Sanity checks
    if y_cap.shape[1] != S:
        print(f"[warn] captured samples ({y_cap.shape[1]}) != source samples ({S}). "
              f"Increase buffer or verify resolve timing.")
    if y_cap.shape[0] != C:
        print(f"[warn] captured channels ({y_cap.shape[0]}) != source channels ({C}).")

    # Save NPZ
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        emg=y_cap.astype(np.float32),
        timestamps=t_cap.astype(np.float64),
        fs_hz=fs,
        channel_names=np.array(ch_names, dtype=object),
        source_oebin=str(args.path),
        stream_name=args.stream_name,
    )
    print(f"[done] wrote NPZ: {out}  ({y_cap.shape[1]} samples × {y_cap.shape[0]} channels)")
    print("[note] If counts differ, set a larger recorder buffer_seconds or reduce --block_ms.")


if __name__ == "__main__":
    main()
