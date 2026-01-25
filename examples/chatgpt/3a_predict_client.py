#!/usr/bin/env python3
"""
Real-time (playback) gesture prediction that strictly pulls the latest samples,
updates a sliding window (ring buffer), extracts features for the current
window only, and predicts immediately. No pre-buffering of the full file.
"""
from __future__ import annotations

import os
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm

from pyoephys.interface import OEBinPlaybackClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.io import load_config_file, lock_params_to_meta
from pyoephys.ml import (
    ModelManager,
    EMGClassifier,
    load_training_metadata,
    evaluate_against_events,
)


def run(
    root_dir: str,
    label: str = "",
    window_ms: int | None = None,
    step_ms: int | None = None,
    warmup_ms: int = 500,
    selected_channels=None,
    verbose: bool = False,
    show_progress: bool = True,
):
    # --------------------------
    # Logging
    # --------------------------
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s")

    # --------------------------
    # Lock params to training meta
    # --------------------------
    meta = load_training_metadata(root_dir)
    window_ms, step_ms, selected_channels, env_cut = lock_params_to_meta(
        meta, window_ms, step_ms, selected_channels
    )

    # --------------------------
    # Playback client
    # --------------------------
    file_path = os.path.join(root_dir, "raw", "gestures")
    client = OEBinPlaybackClient(file_path, loopback=False, enable_lsl=False, verbose=verbose)
    client.start()

    fs = float(client.sampling_rate)
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))

    base_index = int(round(client.t_file[0] * fs)) if getattr(client, "t_file", None) is not None else 0
    total_samples_declared = int(getattr(client, "n_samples", 0))
    pbar = tqdm(total=total_samples_declared, desc="Playback", unit="samples", dynamic_ncols=True, disable=not show_progress)

    # --------------------------
    # Preprocessor + warmup
    # --------------------------
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)

    # Wait until some data is available, then warm up filters
    warm = client.get_latest_window(max(1, warmup_ms))
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = client.get_latest_window(max(1, warmup_ms))

    if selected_channels is not None:
        warm = warm[selected_channels, :]
    if warm.ndim == 1:
        warm = warm[None, :]

    # Prime state
    _ = pre.preprocess(warm)
    C = warm.shape[0]
    ring = np.zeros((C, W), dtype=np.float32)  # sliding window buffer
    carry = None  # leftover filtered tail < S

    # --------------------------
    # Model
    # --------------------------
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # --------------------------
    # Real-time loop
    # --------------------------
    preds, starts = [], []
    emitted_end = 0
    last_seen = 0

    try:
        while not getattr(client, "is_done", lambda: False)():
            # How many *new* raw samples exist?
            total = int(getattr(client, "total_samples", 0))
            new_n = total - last_seen
            if new_n <= 0:
                time.sleep(0.005)
                continue

            need_ms = max(1, int(round(1000.0 * new_n / fs)))
            chunk = client.get_latest_window(need_ms)
            if chunk is None or chunk.size == 0:
                time.sleep(0.002)
                continue

            if selected_channels is not None:
                chunk = chunk[selected_channels, :]
            if chunk.ndim == 1:
                chunk = chunk[None, :]

            # Stateful preprocessing of ONLY the new samples
            y_new = pre.preprocess(chunk)

            # Prepend any leftover tail (<S) from prior iteration so we can emit fixed S steps
            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                carry = y_new
                last_seen = total
                continue

            n_steps = Nf // S
            for k in range(n_steps):
                y_step = y_new[:, k * S : (k + 1) * S]

                # Slide the window
                ring = np.concatenate([ring[:, S:], y_step], axis=1)

                # Extract one feature vector for the *current* window only
                feats = pre.extract_emg_features(ring, window_ms=window_ms, step_ms=window_ms, return_windows=False)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)

                if feats.shape[1] != n_features_expected:
                    raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")

                pred = manager.predict(feats)[0]
                preds.append(pred)

                # Left-edge index for this window
                emitted_end += S
                starts.append(base_index + emitted_end - W)

            # Keep the tail (remainder < S) for the next iteration
            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            # Progress is based on raw playback position
            if pbar is not None:
                pbar.update(new_n)
                pbar.set_postfix_str(f"t={total / fs:.2f}s  pred={pred}")
            last_seen = total

            # Small sleep to avoid tight-spin
            time.sleep(0.001)

    finally:
        client.stop()
        if pbar is not None:
            pbar.close()
        logging.info("Streaming stopped.")

    # --------------------------
    # Optional evaluation against events file
    # --------------------------
    evaluate_against_events(root_dir, np.asarray(starts, dtype=int), preds)


def parse_args():
    p = argparse.ArgumentParser("3a: Real-time gesture prediction from playback client (strict latest-sample mode)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--warmup_ms",   type=int, default=500)
    p.add_argument("--selected_channels", nargs="+", type=int, default=None)
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--no_progress", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update(vars(args))
    run(
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        warmup_ms=cfg.get("warmup_ms", 500),
        selected_channels=cfg.get("selected_channels", None),
        verbose=cfg.get("verbose", False),
        show_progress=not cfg.get("no_progress", False),
    )
