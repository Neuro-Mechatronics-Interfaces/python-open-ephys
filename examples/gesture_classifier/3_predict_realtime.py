#!/usr/bin/env python3
"""
3_predict_realtime.py  -  Real-time EMG gesture prediction via ZMQ
====================================================================
Connects to a running Open Ephys GUI session, streams EMG windows, and
prints the predicted gesture on every step.

Requires Open Ephys GUI to be recording / playing back with the ZMQ
Interface plugin active.

Usage (defaults work after training on the example data):
    python 3_predict_realtime.py

Common overrides:
    python 3_predict_realtime.py --model_dir ./data/model --host 127.0.0.1 --port 5556
    python 3_predict_realtime.py --duration 30          # stop after 30 s
    python 3_predict_realtime.py --n_windows 200        # stop after 200 predictions

Before running, complete the pipeline:
    python 1_build_dataset.py
    python 2_train_model.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, load_training_metadata


# ── helpers ──────────────────────────────────────────────────────────────────


def _majority_vote(history: "deque[str]") -> str:
    """Return the most common label in a sliding window."""
    counts: dict[str, int] = {}
    for lbl in history:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)  # type: ignore[arg-type]


# ── core ──────────────────────────────────────────────────────────────────────

def predict_realtime(
    root_dir: str = "./data",
    host: str = "127.0.0.1",
    port: str = "5556",
    smooth_k: int = 5,
    duration: Optional[float] = None,
    n_windows: Optional[int] = None,
    connect_timeout: float = 15.0,
    verbose: bool = False,
) -> None:
    """
    Stream EMG from an Open Ephys GUI ZMQ plugin and print predictions.

    Parameters
    ----------
    root_dir         : Data root directory containing the ``model/`` sub-folder
                       (same directory passed to 2_train_model.py, e.g. ``./data``).
    host             : IP address of the Open Ephys machine.
    port             : ZMQ data port (string, e.g. "5556").
    smooth_k         : Number of consecutive windows to majority-vote over.
    duration         : Stop automatically after this many seconds (None = run forever).
    n_windows        : Stop automatically after this many predictions (None = run forever).
    connect_timeout  : Seconds to wait for first data frame before aborting.
    verbose          : Print extra debug information.
    """

    # ── load model metadata ──────────────────────────────────────────────────
    print(f"Loading model metadata from {root_dir!r} ...")
    meta = load_training_metadata(root_dir)

    window_ms: int = int(meta.get("window_ms", 200))
    step_ms: int = int(meta.get("step_ms", 50))
    env_cut: float = float(meta.get("envelope_cutoff_hz", 5.0))
    # normalise [] → None so channel slicing is skipped when all channels are used
    channels = meta.get("selected_channels") or None

    if verbose:
        print(f"  window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")
        print(f"  channels={channels}")

    # ── load model ───────────────────────────────────────────────────────────
    manager = ModelManager(root_dir=root_dir, label="", model_cls=EMGClassifier)
    manager.load_model()
    print("Model loaded.")

    # ── connect to ZMQ stream ────────────────────────────────────────────────
    print(f"Connecting to Open Ephys ZMQ stream at {host}:{port} ...")
    client = ZMQClient(
        host_ip=host,
        data_port=port,
        auto_start=True,
        verbose=verbose,
    )

    print(f"Waiting for data (timeout={connect_timeout:.0f}s) ...")
    if not client.ready_event.wait(timeout=connect_timeout):
        print(
            "ERROR: No data received from Open Ephys within "
            f"{connect_timeout:.0f}s.\n"
            "Check that the ZMQ Interface plugin is active and the GUI is\n"
            "in Record or Playback mode."
        )
        sys.exit(1)

    fs: float = client.fs
    print(f"Stream ready -- fs={fs:.1f} Hz")

    # ── channel count sanity check ───────────────────────────────────────────
    # N_channels is lazy-initialised on the first get_latest() call.
    # Wait briefly so all channels have had a chance to send at least one packet.
    print("Checking channel count ...")
    time.sleep(1.0)
    client.get_latest(1)          # triggers channel_index / N_channels lazy init
    expected_ch = int(meta.get("input_dim", 0))
    received_ch = int(client.N_channels)
    if expected_ch > 0 and received_ch != expected_ch:
        print(
            f"ERROR: ZMQ stream is sending {received_ch} channel(s), "
            f"but the model was trained on {expected_ch}.\n"
            "  → In Open Ephys GUI, open the ZMQ Interface plugin settings\n"
            "    and make sure ALL channels are enabled (not just a sub-selection).\n"
            "  → Alternatively, retrain the model on data recorded with the\n"
            f"    same {received_ch}-channel configuration."
        )
        sys.exit(1)
    print(f"  {received_ch} channels confirmed.")

    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)

    # ── prediction loop ──────────────────────────────────────────────────────
    history: deque = deque(maxlen=max(1, smooth_k))
    step_secs = step_ms / 1000.0
    t_start = time.time()
    n_done = 0

    print()
    print("-" * 50)
    print("  Gesture prediction  (Ctrl-C to stop)")
    print("-" * 50)

    try:
        while True:
            # ── stop conditions ──────────────────────────────────────────────
            if duration is not None and (time.time() - t_start) >= duration:
                print(f"\nReached duration limit ({duration:.1f}s).")
                break
            if n_windows is not None and n_done >= n_windows:
                print(f"\nReached window limit ({n_windows}).")
                break

            # ── get latest window ────────────────────────────────────────────
            try:
                window = client.get_latest_window(window_ms)  # (C, N)
            except Exception:
                time.sleep(0.01)
                continue

            if window is None or window.size == 0:
                time.sleep(0.01)
                continue

            # ── channel selection ────────────────────────────────────────────
            if channels is not None:
                try:
                    window = window[channels, :]
                except (IndexError, TypeError):
                    pass  # use all channels if index OOB

            # ── preprocess + features ────────────────────────────────────────
            envelope = pre.preprocess(window)                                    # (C, N)
            feat_2d = pre.extract_emg_features(envelope, window_ms, window_ms)  # (1, C*F)

            if feat_2d.ndim != 2 or feat_2d.shape[0] == 0:
                time.sleep(step_secs)
                continue

            # ── predict ──────────────────────────────────────────────────────
            try:
                raw_pred = manager.predict(feat_2d)
                label = str(raw_pred[0]) if hasattr(raw_pred, "__len__") else str(raw_pred)
            except Exception as exc:
                if verbose:
                    print(f"  [predict error] {exc}")
                time.sleep(step_secs)
                continue

            history.append(label)
            smoothed = _majority_vote(history)

            n_done += 1
            elapsed = time.time() - t_start
            print(
                f"  [{elapsed:6.1f}s | win {n_done:5d}]  "
                f"raw={label:<15s}  smoothed={smoothed}",
                flush=True,
            )

            time.sleep(step_secs)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print()
    print(f"Total windows predicted: {n_done}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Real-time EMG gesture prediction from Open Ephys ZMQ stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root_dir",
        default="./data/gesture_model",
        help="Data root directory containing the model/ sub-folder (same as 2_train_model.py).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP address of the Open Ephys host machine.",
    )
    p.add_argument(
        "--port",
        default="5556",
        help="ZMQ data port of the Open Ephys ZMQ Interface plugin.",
    )
    p.add_argument(
        "--smooth_k",
        type=int,
        default=5,
        help="Majority-vote window size (number of consecutive predictions).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds.",
    )
    p.add_argument(
        "--n_windows",
        type=int,
        default=None,
        help="Stop automatically after this many predictions.",
    )
    p.add_argument(
        "--connect_timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the first ZMQ data frame.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = p.parse_args()

    predict_realtime(
        root_dir=args.root_dir,
        host=args.host,
        port=args.port,
        smooth_k=args.smooth_k,
        duration=args.duration,
        n_windows=args.n_windows,
        connect_timeout=args.connect_timeout,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
