#!/usr/bin/env python3
"""
3cv2_predict_gui_realtime.py
----------------------------
Live EMG prediction from Open Ephys GUI via ZMQ:
- Locks preprocessing + timing to training metadata
- Warmup for filter state
- Incremental updates with drain_new(), ring buffer of size W
- Predict every step_ms (training stride)
"""
import argparse
import logging
import signal
import time
from typing import List, Dict

import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.io import load_metadata_json, normalize_name, lock_params_to_meta
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier


def _map_training_names_to_indices(client: ZMQClient, trained_names: List[str]) -> List[int]:
    name_by_idx: Dict[int, str] = getattr(client, "_name_by_index", {})
    norm_to_idx = {normalize_name(nm): idx for idx, nm in name_by_idx.items()}
    indices, missing = [], []
    for nm in trained_names:
        nrm = normalize_name(nm)
        if nrm in norm_to_idx:
            indices.append(norm_to_idx[nrm])
        else:
            missing.append(nm)
    if missing:
        logging.warning("Some training channels not (yet) present on the live stream: %s", missing)
    if not indices:
        raise RuntimeError("No training channels were found on the live stream yet.")
    return indices


def main():
    ap = argparse.ArgumentParser("Realtime EMG prediction from Open Ephys ZMQ stream (training-locked).")
    ap.add_argument("--root_dir", required=True, help="Root with training artifacts & metadata.json.")
    ap.add_argument("--label", default="", help="Optional training label variant.")
    ap.add_argument("--zmq", default="tcp://127.0.0.1", help="ZMQ ip/prefix (e.g., tcp://127.0.0.1).")
    ap.add_argument("--data-port", type=int, default=5556)
    ap.add_argument("--heartbeat-port", type=int, default=5557)
    ap.add_argument("--wait", type=float, default=15.0, help="Seconds to wait for required channels.")
    ap.add_argument("--warmup-ms", type=int, default=500, help="Initial filter warmup duration.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Proceed if only a subset of training channels are present.")
    args = ap.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # 1) Lock to training settings
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    window_ms, step_ms, env_cut, _ = lock_params_to_meta(
        meta, None, None,  # use training window/step
        selected_channels=None
    )
    #logging.info("Training-locked params: window_ms=%d, step_ms=%d, envelope_cutoff=%.3f Hz",
    #             window_ms, step_ms, env_cut)

    # 2) Model
    manager = ModelManager(
        root_dir=args.root_dir, label=args.label,
        model_cls=EMGClassifier, config={"verbose": args.verbose},
    )
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # 3) ZMQ client (ring big enough to hold > W)
    window_secs = max(1.0, window_ms / 1000.0 * 5)
    client = ZMQClient(
        zqm_ip=args.zmq,
        http_ip="127.0.0.1",
        data_port=args.data_port,
        heartbeat_port=args.heartbeat_port,
        window_secs=window_secs,
        channels=None,
        auto_start=True,
        verbose=args.verbose,
        expected_channel_names=trained_names,
        expected_channel_count=None,
        require_complete=not args.allow_partial,
        required_fraction=1.0,
        max_channels=512,
    )

    stop = False
    def _sigint(_s, _f):
        nonlocal stop
        stop = True
        print("\n[Info] Ctrl-C received; closing...")
    signal.signal(signal.SIGINT, _sigint)

    # 4) Wait for channels & map names -> indices, in training order
    if not client.wait_for_channels(timeout_sec=args.wait):
        if args.allow_partial:
            logging.warning("Proceeding with subset of training channels (timeout reached).")
        else:
            raise RuntimeError("Required training channels did not appear within --wait seconds.")
    indices = _map_training_names_to_indices(client, trained_names)
    client.set_channel_index(indices)
    logging.info("Selected %d/%d channels in training order.", len(indices), len(trained_names))

    # 5) Status and preprocessor
    stat = client.get_connection_status()
    fs = float(client.fs)
    logging.info("Connected: %s | fs≈%.1f Hz | seen=%s | n_total=%d",
                 stat["connected"], fs, stat["seen_channels"], stat["n_channels_total"])
    if fs <= 0:
        raise RuntimeError("Client sampling rate unavailable/invalid.")

    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=args.verbose)

    # 6) Derived sizes
    W = int(round(window_ms / 1000.0 * fs))  # samples per window
    S = int(round(step_ms   / 1000.0 * fs))  # samples per step
    C = len(indices)
    ring = np.zeros((C, W), dtype=np.float32)
    carry = None

    # 7) Warmup (establish filter state)
    warm = client.get_latest_window(max(1, args.warmup_ms))
    t0 = time.time()
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = client.get_latest_window(max(1, args.warmup_ms))
        if time.time() - t0 > 5.0:
            logging.warning("Still waiting for warmup samples...")
            t0 = time.time()
    if warm.ndim == 1:
        warm = warm[np.array(indices), None]
    else:
        warm = warm  # already selected by set_channel_index
    _ = pre.preprocess(warm)  # warm filters

    prints = 0
    try:
        while not stop:
            # 8) Pull only NEW samples since last call
            _t_new, y_new = client.drain_new()
            if y_new is None or y_new.size == 0:
                time.sleep(0.002)
                continue
            # y_new is already channel-selected: shape (C, K)

            # 9) Preprocess increment (preserves filter state continuity)
            y_new = pre.preprocess(y_new)

            # 10) Concatenate any carried remainder from last tick
            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                carry = y_new
                continue

            # 11) Advance in strides of S and emit a prediction per step
            n_steps = Nf // S
            for k in range(n_steps):
                step_chunk = y_new[:, k*S:(k+1)*S]
                # Slide ring buffer by S
                if S >= W:
                    # degenerate: just keep latest W
                    ring = step_chunk[:, -W:]
                else:
                    ring = np.concatenate([ring[:, S:], step_chunk], axis=1)

                # Extract features on the FULL ring window (W) with step==window
                feats = pre.extract_emg_features(ring, window_ms=window_ms, step_ms=window_ms, progress=False)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)

                if feats.shape[1] != n_features_expected:
                    raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")

                y_pred = manager.predict(feats)   # (1,)
                y = str(y_pred[0])

                prints += 1
                print(f"[{prints:04d}] step={S} samp | ring={C}x{W} -> y={y}")

            # 12) Keep remainder for next tick
            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            time.sleep(0.001)

    finally:
        try:
            client.stop()
            client.close()
        except Exception:
            pass
        print("[Info] Closed. Bye.]")

if __name__ == "__main__":
    main()
