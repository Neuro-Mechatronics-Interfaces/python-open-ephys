
#!/usr/bin/env python3
"""
3cv2_predict_gui_window.py
-------------------------
Connect to a running Open Ephys GUI via ZMQ and, on each loop, grab a fixed
window of recent EMG samples for the *training channels* (default: 200 ms).

Key differences vs. 3cv2_predict_gui.py:
- No draining in the tight loop (avoids discarding data).
- Uses get_latest_window(window_ms) to fetch (C, N) per print interval.

Example:
    python 3cv2_predict_gui_window_predict.py \
      --root_dir "G:\\Shared drives\\NML_shared\\DataShare\\HDEMG Human Healthy\\HD-EMG_Cuff\\Jonathan\\2025_07_31" \
      --label sleeve_15ch_ring \
      --verbose
"""
import argparse
import logging
import signal
import time
from typing import List, Dict

import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.io import load_metadata_json, normalize_name, lock_params_to_meta
from pyoephys.ml import ModelManager, EMGClassifier
from pyoephys.processing import EMGPreprocessor


def _map_training_names_to_indices(client: ZMQClient, trained_names: List[str]) -> List[int]:
    """Build channel indices in the *training order* by matching names seen on the live stream."""
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
    ap = argparse.ArgumentParser("Grab fixed windows from Open Ephys via ZMQ.")
    ap.add_argument("--root_dir", required=True, help="Root with metadata.json from training.")
    ap.add_argument("--label", default="", help="Optional label used during training.")
    ap.add_argument("--zmq", default="tcp://127.0.0.1", help="ZMQ ip/prefix (e.g., tcp://127.0.0.1).")
    ap.add_argument("--data-port", type=int, default=5556, help="ZMQ data port.")
    ap.add_argument("--heartbeat-port", type=int, default=5557, help="ZMQ heartbeat port.")
    ap.add_argument("--interval", type=float, default=0.5, help="Seconds between window fetches.")
    ap.add_argument("--window-ms", type=int, default=200, help="Window length in milliseconds.")
    ap.add_argument("--wait", type=float, default=15.0, help="Seconds to wait for required channels.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Proceed if only a subset of training channels are present.")
    args = ap.parse_args()

    # Logging
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load training channel names (canonical order)
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    logging.info("Expecting %d training channels (by name).", len(trained_names))

    # Lock timing/preprocessing to training; allow override of window_ms via CLI
    locked_window_ms, locked_step_ms, _, env_cut = lock_params_to_meta(
        meta,
        args.window_ms,
        None,  # step not needed for single-window
        selected_channels=None  # we use name-based mapping instead
    )
    window_ms = locked_window_ms
    logging.info("Expecting %d training channels; using window_ms=%d (from training), envelope_cutoff=%.3f Hz",
                 len(trained_names), window_ms, env_cut)

    # Model + scaler
    manager = ModelManager(
        root_dir=args.root_dir,
        label=args.label,
        model_cls=EMGClassifier,
        config={"verbose": args.verbose},
    )
    manager.load_model()
    # We'll verify feature dimensionality after we build the first feature vector.

    # Connect ZMQ client with a ring buffer big enough for our window
    window_secs = max(1.0, args.window_ms / 1000.0 * 5)  # generous buffer (≥5× window)
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
        max_channels=128,
    )

    # Graceful shutdown
    stop = False
    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
        print("\n[Info] Ctrl-C received; closing...")
    signal.signal(signal.SIGINT, _sigint)

    # Wait for channels
    if not client.wait_for_channels(timeout_sec=args.wait):
        if args.allow_partial:
            logging.warning("Proceeding with subset of training channels (timeout reached).")
        else:
            raise RuntimeError("Required training channels did not appear within --wait seconds.")

    # Map names -> indices, in training order, then select
    indices = _map_training_names_to_indices(client, trained_names)
    client.set_channel_index(indices)
    logging.info("Selected %d/%d channels in training order.", len(indices), len(trained_names))

    # Initial status
    stat = client.get_connection_status()
    fs = client.fs
    logging.info("Connected: %s | fs≈%.1f Hz | seen_channels=%s | n_total=%d",
                 stat["connected"], fs, stat["seen_channels"], stat["n_channels_total"])

    # Preprocessor (for feature extraction)
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=args.verbose)

    # Compute required samples for a full window
    win_samples = int(round(window_ms / 1000.0 * fs))

    # Loop: fetch fixed-length window and print shape
    prints = 0
    feature_dim_checked = False
    next_t = time.time()

    while not stop:
        now = time.time()
        if now >= next_t:
            next_t = now + args.interval

            # Latest fixed-length window (C, N)
            Y_win = client.get_latest_window(window_ms)
            C, N = Y_win.shape
            if N < win_samples:
                # Initial warmup: ring buffer not yet filled
                logging.debug("Warmup: received %d/%d samples for the window; skipping this tick.", N, win_samples)
                continue

            # Preprocess to match training (bandpass/notch/rectify/envelope as defined in EMGPreprocessor)
            Y_pp = pre.preprocess(Y_win)  # expects (C, N)

            # Extract features for exactly one window by setting step == window_ms
            X = pre.extract_emg_features(Y_pp, window_ms=window_ms, step_ms=window_ms, progress=False)
            # Expect a single row (1, F)
            if X.ndim != 2 or X.shape[0] < 1:
                logging.warning("Unexpected feature shape from a single window: %s", X.shape)
                continue
            x_row = X[0:1, :]  # keep 2D for predict()

            # First pass: verify feature dimension matches scaler/model expectation
            if not feature_dim_checked:
                n_features_expected = len(manager.scaler.mean_)
                if x_row.shape[1] != n_features_expected:
                    raise ValueError(
                        f"Feature dim {x_row.shape[1]} != scaler expectation {n_features_expected}. "
                        f"Check preprocessing/feature settings against training metadata."
                    )
                feature_dim_checked = True

            # Predict
            y_pred = manager.predict(x_row)  # should return shape (1,) outputs np.str_('class')

            # Convert the class output to a native string
            y = str(y_pred[0])

            prints += 1
            eff_ms = (N / fs) * 1000.0
            print(f"[{prints:04d}] win({C}x{N} ≈ {eff_ms:.1f} ms) -> y={y}")

        time.sleep(0.005)

    try:
        client.stop()
        client.close()
    except Exception:
        pass
    print("[Info] Closed. Bye.]")


if __name__ == "__main__":
    main()
