#!/usr/bin/env python3
"""
3av2_predict_zmq.py
Realtime (strict) EMG prediction from Open Ephys GUI via ZMQClient, training-locked.
- Incremental preprocessing with filter state
- Predict every step_ms (matches training)
- Align window start indices to the original recording for evaluation
"""

import argparse, logging, signal, time, os
from typing import List, Dict, Optional

import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.io import load_metadata_json, lock_params_to_meta, normalize_name, load_oebin_file
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


def _map_training_names_to_indices(client: ZMQClient, trained_names: List[str]) -> List[int]:
    name_by_idx: Dict[int, str] = getattr(client, "_name_by_index", {})
    norm_to_idx = {normalize_name(nm): i for i, nm in name_by_idx.items()}
    idx, missing = [], []
    for nm in trained_names:
        nrm = normalize_name(nm)
        if nrm in norm_to_idx:
            idx.append(norm_to_idx[nrm])
        else:
            missing.append(nm)
    if missing:
        logging.warning("Live stream missing some training channels (yet?): %s", missing)
    if not idx:
        raise RuntimeError("No training channels found on the live stream.")
    return idx


def _samples_from_timestamps(t_new: np.ndarray, fs: float) -> np.ndarray:
    if t_new is None or t_new.size == 0:
        return np.array([], dtype=np.int64)
    t_new = np.asarray(t_new).ravel()
    frac = np.mean((t_new - np.floor(t_new)) != 0)
    if frac > 0.1 or np.nanmax(t_new) < 1e6:
        return np.asarray(np.round(t_new * fs), dtype=np.int64)
    return np.asarray(t_new, dtype=np.int64)


def _align_base_index(align_from: str, root_dir: str, fs: float, t_new_first: Optional[np.ndarray]) -> int:
    align_from = (align_from or "oebin").lower()
    if align_from == "oebin":
        raw_dir = os.path.join(root_dir, "raw")
        d = load_oebin_file(raw_dir, verbose=False)
        t0 = float(np.asarray(d["t_amplifier"])[0])
        fs_file = float(d["sample_rate"])
        if abs(fs_file - fs) > 1e-3:
            logging.warning("Sample rate mismatch: GUI fs=%.3f vs OEBin fs=%.3f", fs, fs_file)
        base_idx = int(round(t0 * fs_file))
        logging.info("Alignment: OEBin t0=%.6f s -> base_index=%d", t0, base_idx)
        return base_idx
    elif align_from == "timestamps":
        s_idx = _samples_from_timestamps(np.asarray(t_new_first), fs)
        base_idx = int(s_idx[0]) if s_idx.size else 0
        logging.info("Alignment: ZMQ timestamps -> base_index=%d", base_idx)
        return base_idx
    else:
        logging.info("Alignment: zero-based")
        return 0


def _infer_target_samples(meta: dict, fs: float,
                          cli_target_samples: Optional[int],
                          cli_target_seconds: Optional[float]) -> Optional[int]:
    if cli_target_samples and cli_target_samples > 0:
        return int(cli_target_samples)
    if cli_target_seconds and cli_target_seconds > 0:
        return int(round(cli_target_seconds * fs))
    # Try a few metadata fields
    for path in [("data", "n_samples"), ("n_samples",), ("data", "duration_sec"), ("duration_sec",), ("data", "duration"), ("duration",)]:
        try:
            val = meta
            for k in path:
                val = val[k]
            if "duration" in path or "duration_sec" in path:
                return int(round(float(val) * fs))
            return int(val)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser("Realtime, step-locked prediction from ZMQClient (training-matched).")
    ap.add_argument("--root_dir", required=True, help="Root with training artifacts & metadata.json.")
    ap.add_argument("--label", default="", help="Training label variant (if used).")
    ap.add_argument("--zmq", default="tcp://127.0.0.1")
    ap.add_argument("--data-port", type=int, default=5556)
    ap.add_argument("--heartbeat-port", type=int, default=5557)
    ap.add_argument("--warmup-ms", type=int, default=500)
    ap.add_argument("--target-samples", type=int, default=None)
    ap.add_argument("--target-seconds", type=float, default=None)
    ap.add_argument("--align-from", choices=["oebin", "timestamps", "zero"], default="oebin",
                    help="How to set the base sample index for evaluation.")
    ap.add_argument("--wait", type=float, default=15.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--print-proba", action="store_true")
    args = ap.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # 1) Training-locked params (NOTE: correct unpack order)
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta, None, None, selected_channels=None)
    logging.info("Training-locked params: window_ms=%d, step_ms=%d, envelope_cutoff=%.3f Hz",
                 window_ms, step_ms, float(env_cut))

    # 2) Model
    manager = ModelManager(root_dir=args.root_dir, label=args.label,
                           model_cls=EMGClassifier, config={"verbose": args.verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # 3) ZMQ client
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
        require_complete=not args.allow_partially if hasattr(args, "allow_partially") else not args.allow_partial,
        required_fraction=1.0,
        max_channels=512,
    )

    stop = False
    def _sigint(_s, _f):
        nonlocal stop
        stop = True
        print("\n[Info] Ctrl-C received; closing...")
    signal.signal(signal.SIGINT, _sigint)

    # 4) Wait & map channels
    if not client.wait_for_channels(timeout_sec=args.wait):
        if args.allow_partial:
            logging.warning("Proceeding with subset of training channels (timeout).")
        else:
            raise RuntimeError("Required channels not present in time.")
    indices = _map_training_names_to_indices(client, trained_names)
    client.set_channel_index(indices)
    logging.info("Selected %d/%d channels.", len(indices), len(trained_names))

    # 5) Status + preprocessor
    stat = client.get_connection_status()
    fs = float(client.fs)
    logging.info("Connected: %s | fs≈%.1f Hz | seen=%s | n_total=%d",
                 stat["connected"], fs, stat["seen_channels"], stat["n_channels_total"])
    if fs <= 0:
        raise RuntimeError("Invalid sampling rate from client.")

    pre = EMGPreprocessor(fs=fs, envelope_cutoff=float(env_cut), verbose=args.verbose)

    # 6) Sizes
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))
    C = len(indices)

    # 7) Warmup
    warm = client.get_latest_window(max(1, args.warmup_ms))
    t0 = time.time()
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = client.get_latest_window(max(1, args.warmup_ms))
        if time.time() - t0 > 5.0:
            logging.warning("Still waiting for warmup samples...")
            t0 = time.time()
    _ = pre.preprocess(warm)

    # 8) Ring + bookkeeping
    ring = np.zeros((C, W), dtype=np.float32)
    carry = None
    preds: List[str] = []
    starts: List[int] = []

    target_samples = _infer_target_samples(meta, fs, args.target_samples, args.target_seconds)
    emitted_end = 0
    base_index: Optional[int] = None

    # 9) Loop
    try:
        while not stop:
            t_new, y_new = client.drain_new()
            if y_new is None or y_new.size == 0:
                time.sleep(0.002)
                continue

            # Set base_index on first batch using chosen align mode
            if base_index is None:
                base_index = _align_base_index(args.align_from, args.root_dir, fs, t_new)

            # Incremental preprocessing
            y_new = pre.preprocess(y_new)

            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                carry = y_new
                continue

            n_steps = Nf // S
            for k in range(n_steps):
                step_chunk = y_new[:, k*S:(k+1)*S]
                ring = step_chunk[:, -W:] if S >= W else np.concatenate([ring[:, S:], step_chunk], axis=1)

                feats = pre.extract_emg_features(ring, window_ms=window_ms, step_ms=window_ms, progress=False)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                if feats.shape[1] != n_features_expected:
                    raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")

                pred = manager.predict(feats)[0]
                preds.append(str(pred))  # ensure string label

                emitted_end += S
                starts.append(int(base_index + emitted_end - W))

                if target_samples is not None and emitted_end >= target_samples:
                    stop = True
                    break

            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None
            time.sleep(0.001)

    finally:
        try:
            client.stop()
            client.close()
        except Exception:
            pass

    # 10) Evaluate
    if not starts:
        logging.warning("No predictions emitted; nothing to evaluate.")
        return
    evaluate_against_events(args.root_dir, np.asarray(starts, dtype=np.int64), preds)
    print(f"[Info] Done and evaluated. total_preds={len(preds)}")


if __name__ == "__main__":
    main()
