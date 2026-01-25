"""
emg_helpers.py — Lightweight helpers to keep example scripts short & readable.

These utilities centralize common logic used across the example CLI scripts:
- Loading & enforcing training metadata (window/step/channels)
- Creating an EMG preprocessor with the right settings
- Streaming features from OEBinPlaybackClient in fixed steps
- Offline evaluation against an events file
"""
from __future__ import annotations

import os
import json
import time
import logging
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from pyoephys.processing import EMGPreprocessor
from pyoephys.io import labels_from_events



# ------------------------------
# Streaming feature extraction
# ------------------------------

def feature_stream_from_client(
    client,
    pre: EMGPreprocessor,
    window_ms: int,
    step_ms: int,
    selected_channels: Optional[List[int]] = None,
    warmup_ms: int = 500,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Yield (feature_row, window_start_sample) from a live/loopback OEBinPlaybackClient.

    The stream emits one feature vector per `step_ms` using a ring buffer of size `window_ms`.
    """
    fs = float(client.sampling_rate)
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))

    # derive absolute index base to align with events
    base_index = int(round(client.t_file[0] * fs)) if getattr(client, "t_file", None) is not None else 0

    # warmup so filters reach steady state
    warm = client.get_latest_window(max(1, warmup_ms))
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = client.get_latest_window(max(1, warmup_ms))

    if selected_channels is not None:
        warm = warm[selected_channels, :]

    _ = pre.preprocess(warm)
    C = warm.shape[0]
    ring = np.zeros((C, W), dtype=np.float32)

    emitted_end = 0
    carry = None  # filtered tail < S carried to next loop
    last_seen = 0

    def _is_done():
        return getattr(client, "is_done", lambda: False)()

    while not _is_done():
        total = int(getattr(client, "total_samples", 0))
        new_n = total - last_seen
        if new_n <= 0:
            time.sleep(0.005)
            continue

        need_ms = max(1, int(round(1000.0 * new_n / fs)))
        chunk = client.get_latest_window(need_ms)
        if chunk is None or chunk.size == 0:
            time.sleep(0.005)
            continue

        if selected_channels is not None:
            chunk = chunk[selected_channels, :]
        if chunk.ndim == 1:
            chunk = chunk[None, :]

        y_new = pre.preprocess(chunk)
        if carry is not None and carry.shape[1] > 0:
            y_new = np.concatenate([carry, y_new], axis=1)

        Nf = y_new.shape[1]
        if Nf < S:
            carry = y_new
            last_seen = total
            continue

        n_steps = Nf // S
        for k in range(n_steps):
            y_step = y_new[:, k * S:(k + 1) * S]
            ring = np.concatenate([ring[:, S:], y_step], axis=1)

            # Use the same class method as training; step==window => 1 row
            feats = pre.extract_emg_features(
                ring, window_ms=window_ms, step_ms=window_ms, return_windows=False
            )
            emitted_end += S
            start_idx = base_index + emitted_end - W
            yield feats, start_idx

        rem = Nf - n_steps * S
        carry = y_new[:, -rem:] if rem > 0 else None
        last_seen = total

