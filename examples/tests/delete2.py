#!/usr/bin/env python3
"""
3a_predict_client_zmq.py
Strict real-time (ZMQ) gesture prediction from Open Ephys GUI:
- Locks channel selection by NAME to the exact training order from metadata
- No pre-buffering; pulls latest samples only; updates a ring buffer; predicts per step
"""
import os, time, argparse, logging
import numpy as np
from tqdm import tqdm

from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.io import load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


def _fetch_latest_ms(client: ZMQClient, ms: int, fs: float):
    """
    Fetch the newest ~ms of data from the client.
    Tries get_latest_window(ms) if available; falls back to get_latest(n_samples).
    Returns ndarray with shape (C, N) or None if not available yet.
    """
    ms = max(1, int(ms))
    # Prefer ms-based window if client exposes it
    try:
        y = client.get_latest_window(ms)
        if y is not None and y.size:
            return y
    except Exception:
        pass

    # Fallback by samples
    n = max(1, int(round(ms / 1000.0 * fs)))
    try:
        y, _t = client.get_latest(n)
        return y
    except Exception:
        return None


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
        warmup_ms: int = 500, selected_channels=None, verbose: bool = False,
        show_progress: bool = True, host_ip: str = "127.0.0.1", data_port: str = "5556",
        heartbeat_port: str | None = None, buffer_seconds: float = 120.0,
        ready_timeout_s: float = 5.0, idle_on_exit: bool = False):

    logging.basicConfig(level=(logging.DEBUG if verbose else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load label-specific metadata & lock params ----
    meta = load_metadata_json(root_dir, label)
    window_ms, step_ms, _selected_channels_ignored, env_cut = lock_params_to_meta(
        meta['data'], window_ms, step_ms, selected_channels
    )
    if not meta.get('data', {}).get('selected_channels'):
        raise RuntimeError("metadata missing selected_channels (training channel order).")

    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    C_ref = len(meta['data']['selected_channels'])

    # ---- ZMQ client (GUI) ----
    logging.info(f"Starting ZMQClient(host={host_ip}, data_port={data_port}) with label '{label}'")
    client = ZMQClient(
        host_ip=host_ip,
        data_port=data_port,
        heartbeat_port=heartbeat_port,
        buffer_seconds=buffer_seconds,
        auto_start=False,
        expected_channel_count=None,
        expected_channel_names=trained_names,
        set_index_looping=False,
        align_to_header_index=True,
        verbose=verbose,
    )

    # Attempt to control GUI; if not available, continue anyway
    try:
        client.gui.idle()
        time.sleep(0.1)
        client.gui.start_acquisition()
    except Exception:
        logging.debug("GUI control not available; assuming acquisition is already running.")

    client.start()
    client.wait_for_expected_channels(timeout=5.0)

    # Wait for first data
    if not client.ready_event.wait(ready_timeout_s):
        client.stop()
        raise TimeoutError("ZMQClient not ready; no data received within timeout.")

    # Sample rate
    fs = float(getattr(client, "fs", meta.get("data", {}).get("fs_hz", 0.0)))
    if fs <= 0:
        client.stop()
        raise RuntimeError("Could not determine sample rate (fs) from ZMQClient or metadata.")
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))
    logging.info(f"ZMQClient started: fs={fs} Hz, W={W} samples, S={S} samples.")

    # ---- Map training channel NAMES -> incoming ZMQ indices (exact order) ----
    zmq_names = client.channel_names or []
    if not zmq_names:
        # give it a moment to populate
        for _ in range(50):
            time.sleep(0.02)
            zmq_names = client.channel_names or []
            if zmq_names:
                break
    if not zmq_names:
        client.stop()
        raise RuntimeError("ZMQ stream has no channel names yet.")

    print(f"meta selected channels: {meta['data']['selected_channels']}")
    name_to_idx = {normalize_name(n): i for i, n in enumerate(zmq_names)}
    want_norm = [normalize_name(nm) for nm in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in name_to_idx]
    if missing:
        client.stop()
        raise RuntimeError(f"ZMQ stream missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    selected_idx = [name_to_idx[n] for n in want_norm]
    client.set_channel_index(selected_idx)
    logging.info(f"Locked {len(selected_idx)} training channels (of {C_ref} trained) to incoming ZMQ names.")

    # ---- Progress (unknown total for live GUI; show rolling sample count) ----
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

    # ---- Strict real-time loop (latest-sample increments only) ----
    preds, starts = [], []
    ring = np.zeros((len(selected_idx), W), dtype=np.float32)
    carry = None
    emitted_end = 0
    last_pred = None

    try:
        while True:
            total = int(getattr(client, "global_sample_index", 0))
            new_n = total - last_seen
            if new_n <= 0:
                time.sleep(0.005); continue

            need_ms = max(1, int(round(1000.0 * new_n / fs)))
            chunk = _fetch_latest_ms(client, need_ms, fs)
            if chunk is None or chunk.size == 0:
                time.sleep(0.002); continue

            # exact training-channel order set via set_channel_index()
            if chunk.ndim == 1:
                chunk = chunk[np.array(range(len(selected_idx))), None]

            y_new = pre.preprocess(chunk)

            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                carry = y_new; last_seen = total; continue

            n_steps = Nf // S
            for k in range(n_steps):
                y_step = y_new[:, k*S:(k+1)*S]
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

            if pbar is not None:
                pbar.update(new_n)
                if last_pred is not None:
                    pbar.set_postfix_str(f"t={total / fs:.2f}s  pred={last_pred}")
            last_seen = total
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

    # Optional: evaluate against event markers if your root_dir has them
    try:
        evaluate_against_events(root_dir, np.asarray(starts, dtype=int), preds)
    except Exception as e:
        logging.info(f"Event evaluation skipped or failed: {e}")


def parse_args():
    p = argparse.ArgumentParser("3a (ZMQ): Strict real-time gesture prediction from Open Ephys GUI")
    p.add_argument("--config_file",   type=str)
    p.add_argument("--root_dir",      type=str, required=True)
    p.add_argument("--label",         type=str, default="")
    p.add_argument("--window_ms",     type=int, default=None)
    p.add_argument("--step_ms",       type=int, default=None)
    p.add_argument("--warmup_ms",     type=int, default=500)
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--no_progress",   action="store_true")

    # ZMQ params (kept here so CLI mirrors the playback script style + minimal additions)
    p.add_argument("--host_ip",       type=str, default="127.0.0.1")
    p.add_argument("--data_port",     type=str, default="5556")
    p.add_argument("--heartbeat_port",type=str, default=None)
    p.add_argument("--buffer_seconds",type=float, default=120.0)
    p.add_argument("--ready_timeout", type=float, default=5.0)
    p.add_argument("--idle_on_exit",  action="store_true")
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
        selected_channels=None,  # always ignored; we use training names
        verbose=cfg.get("verbose", False),
        show_progress=not cfg.get("no_progress", False),
        host_ip=cfg.get("host_ip", "127.0.0.1"),
        data_port=str(cfg.get("data_port", "5556")),
        heartbeat_port=cfg.get("heartbeat_port", None),
        buffer_seconds=float(cfg.get("buffer_seconds", 120.0)),
        ready_timeout_s=float(cfg.get("ready_timeout", 5.0)),
        idle_on_exit=bool(cfg.get("idle_on_exit", False)),
    )
