#!/usr/bin/env python3
"""
3b_predict_zmq_client.py
Real-time gesture prediction from Open Ephys GUI (ZMQ):
- Locks channel selection by NAME to the exact training order from metadata
- Streams strictly from the latest samples using ZMQClient
- Maintains a sliding ring window (W) and advances by step (S) for each prediction
- Optional progress and graceful exit
"""

import os, time, argparse, logging
import numpy as np
from tqdm import tqdm

from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.io import load_config_file, lock_params_to_meta, load_metadata_json, normalize_name, load_open_ephys_session
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


def _fetch_latest_ms(client: ZMQClient, ms: int, fs: float):
    """
    Fetch the newest ~ms of data from the client.
    Tries get_latest_window(ms) (if available), otherwise falls back to get_latest(n_samples).
    Returns ndarray with shape (C, N) or None if not available yet.
    """
    ms = max(1, int(ms))
    # Try ms-based API (present on some clients)
    try:
        y = client.get_latest_window(ms)
        if y is not None and y.size:
            return y
    except Exception:
        pass

    # Fallback: request by samples
    n = max(1, int(round(ms / 1000.0 * fs)))
    try:
        y, _t = client.get_latest(n)
        return y
    except Exception:
        return None


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None, warmup_ms: int = 500,
        host_ip: str = "127.0.0.1", data_port: str = "5556", heartbeat_port: str | None = None,
        buffer_seconds: float = 120.0, ready_timeout_s: float = 5.0, idle_on_exit: bool = True,
        selected_channels=None,  # ignored; order comes from training metadata
        verbose: bool = False, show_progress: bool = True):

    logging.basicConfig(level=(logging.DEBUG if verbose else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ---- Load label-specific training metadata & lock inference params ----
    meta = load_metadata_json(root_dir, label)
    window_ms, step_ms, trained_sel, env_cut = lock_params_to_meta(
        meta['data'], window_ms, step_ms, selected_channels
    )
    if not trained_sel:
        raise RuntimeError("metadata missing selected_channels (training channel order).")

    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    # --- Load OEBin just to know (fs, S_ref) and for progress/end conditions ---
    #     We point at the same gesture folder used during playback.
    oedir = os.path.join(root_dir, "raw", "gestures")
    data = load_open_ephys_session(oedir, verbose=verbose)
    fs = float(data["sample_rate"])
    S_ref = int(data["amplifier_data"].shape[1])  # expected total samples
    C_ref = int(data["amplifier_data"].shape[0])
    emg_t = data["t_amplifier"]

    # ---- Connect ZMQClient (GUI) ----
    logging.info(f"Starting ZMQClient(host={host_ip}, data_port={data_port}) …")
    client = ZMQClient(
        host_ip=host_ip,
        data_port=data_port,
        heartbeat_port=heartbeat_port,
        buffer_seconds=buffer_seconds,
        #expected_channel_count=None,
        expected_channel_names=trained_names,
        auto_start=False,
        set_index_looping=True,
        align_to_header_index=True,
        verbose=verbose,
    )

    # Start GUI acquisition if possible
    client.gui.idle()
    time.sleep(0.1)
    client.gui.start_acquisition()
    client.start()
    client.wait_for_expected_channels(timeout=5.0)

    # Wait for first data to arrive
    if not client.ready_event.wait(ready_timeout_s):
        client.stop()
        raise TimeoutError("ZMQClient not ready; no data received within timeout.")

    # fs from client (fallback to metadata if needed)
    fs = float(getattr(client, "fs", meta.get("data", {}).get("fs_hz", 0.0)))
    if fs <= 0:
        raise RuntimeError("Could not determine sample rate (fs) from ZMQClient or metadata.")
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))
    logging.info(f"Streaming: fs={fs:.3f} Hz, window W={W} samp (~{window_ms} ms), step S={S} samp (~{step_ms} ms)")

    # ---- Channel map: training NAMES -> incoming ZMQ indices (exact order) ----
    zmq_names = client.channel_names or []
    if not zmq_names:
        # Give it a moment on slower setups
        for _ in range(50):
            time.sleep(0.02)
            zmq_names = client.channel_names or []
            if zmq_names:
                break
    if not zmq_names:
        client.stop()
        raise RuntimeError("ZMQ stream has no channel names yet.")

    name_to_idx = {normalize_name(n): i for i, n in enumerate(zmq_names)}
    want_norm = [normalize_name(nm) for nm in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in name_to_idx]
    if missing:
        client.stop()
        raise RuntimeError(f"ZMQ stream missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")

    selected_idx = [name_to_idx[n] for n in want_norm]
    client.set_channel_index(selected_idx)
    logging.info(f"Locked {len(selected_idx)} training channels to ZMQ stream order.")

    # ---- Progress tracker (uses a rolling delta of the global index) ----
    base_index = int(getattr(client, "global_sample_index", 0))
    last_seen = base_index
    pbar = tqdm(total=0, desc="ZMQ", unit="samples", dynamic_ncols=True, disable=not show_progress)

    # ---- Preprocessor + warmup ----
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    warm = _fetch_latest_ms(client, warmup_ms, fs)
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = _fetch_latest_ms(client, warmup_ms, fs)
    if warm.ndim == 1:
        warm = warm[np.array(range(len(selected_idx))), None]
    _ = pre.preprocess(warm)

    # ---- Model ----
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # ---- Online loop ----
    preds, starts = [], []
    ring = np.zeros((len(selected_idx), W), dtype=np.float32)
    carry = None
    emitted_end = 0
    last_pred = None

    try:
        while True:
            total = int(getattr(client, "global_sample_index", 0))
            new_n = max(0, total - last_seen)

            if new_n <= 0:
                time.sleep(0.003)
                continue

            # Convert "new_n" samples into time, then fetch exactly that many newest samples
            need_ms = max(1, int(round(1000.0 * new_n / fs)))
            chunk = _fetch_latest_ms(client, need_ms, fs)
            if chunk is None or chunk.size == 0:
                time.sleep(0.002)
                continue

            # exact training-channel order is already set via set_channel_index()
            if chunk.ndim == 1:
                chunk = chunk[np.array(range(len(selected_idx))), None]

            y_new = pre.preprocess(chunk)

            # Concatenate with carry from the previous loop
            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                carry = y_new
                last_seen = total
                continue

            n_steps = Nf // S
            for k in range(n_steps):
                y_step = y_new[:, k*S:(k+1)*S]
                # Slide ring window forward by S and append the new samples
                ring = np.concatenate([ring[:, S:], y_step], axis=1)

                feats = pre.extract_emg_features(ring, window_ms=window_ms, step_ms=window_ms, return_windows=False)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                if feats.shape[1] != n_features_expected:
                    raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")

                pred = manager.predict(feats)[0]
                preds.append(pred)
                emitted_end += S
                starts.append(base_index + emitted_end - W)
                last_pred = pred

            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            last_seen = total
            if pbar is not None:
                pbar.update(new_n)
                if last_pred is not None:
                    pbar.set_postfix_str(f"t={total / fs:.2f}s  pred={last_pred}")

            # short sleep prevents spin when GUI is paused but connection still alive
            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        try:
            if idle_on_exit:
                client.gui.idle()
        except Exception:
            pass
        client.stop()
        if pbar is not None:
            pbar.close()
        logging.info("Streaming stopped.")

    # Optional: evaluate against on-disk events if present
    try:
        evaluate_against_events(root_dir, np.asarray(starts, dtype=int), preds)
    except Exception as e:
        logging.info(f"Event evaluation skipped or failed: {e}")


def parse_args():
    p = argparse.ArgumentParser("3b: Real-time gesture prediction from ZMQClient (Open Ephys GUI)")
    p.add_argument("--config_file",   type=str)
    p.add_argument("--root_dir",      type=str, required=True, help="Folder containing model + metadata + events")
    p.add_argument("--label",         type=str, default="",   help="Training label namespace")
    p.add_argument("--window_ms",     type=int, default=None)
    p.add_argument("--step_ms",       type=int, default=None)
    p.add_argument("--warmup_ms",     type=int, default=500)

    # ZMQ / GUI params
    p.add_argument("--host_ip",       type=str, default="127.0.0.1")
    p.add_argument("--data_port",     type=str, default="5556")
    p.add_argument("--heartbeat_port",type=str, default=None)
    p.add_argument("--buffer_seconds",type=float, default=120.0)
    p.add_argument("--ready_timeout", type=float, default=5.0)
    p.add_argument("--idle_on_exit",  action="store_true")

    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--no_progress",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        warmup_ms=cfg.get("warmup_ms", 500),
        host_ip=cfg.get("host_ip", "127.0.0.1"),
        data_port=str(cfg.get("data_port", "5556")),
        heartbeat_port=cfg.get("heartbeat_port", None),
        buffer_seconds=float(cfg.get("buffer_seconds", 120.0)),
        ready_timeout_s=float(cfg.get("ready_timeout", 5.0)),
        idle_on_exit=bool(cfg.get("idle_on_exit", False)),
        selected_channels=None,  # ignored; we lock by training names
        verbose=cfg.get("verbose", False),
        show_progress=not cfg.get("no_progress", False),
    )
