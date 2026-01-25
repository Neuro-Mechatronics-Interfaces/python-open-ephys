#!/usr/bin/env python3
"""
3av2_predict_zmq.py
Real-time gesture prediction from Open Ephys GUI via ZMQClient.

Behavior:
- Locks channel selection by NAME to the exact training order from metadata
- Uses header-aligned global index (no loopbacks), drains NEW samples
- Stops when (a) collected >= OEBin sample length OR (b) header index stalls ~1s
"""

import os, time, argparse, logging
import numpy as np
from tqdm import tqdm

from pyoephys.io import load_open_ephys_session, load_config_file, load_metadata_json, lock_params_to_meta, \
    normalize_name
from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None, warmup_ms: int = 500,
        http_ip: str = "127.0.0.1", data_port: str = "5556", heartbeat_port: str | None = None,
        stall_timeout_s: float = 1.0, safety_margin_s: float = 2.0, verbose: bool = False, show_progress: bool = True):

    logging.basicConfig(level=(logging.DEBUG if verbose else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load label-specific metadata & lock params
    meta = load_metadata_json(root_dir, label)
    window_ms, step_ms, selected_channels, env_cut = lock_params_to_meta(
        meta['data'], window_ms, step_ms, None
    )
    if not selected_channels:
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

    # ZMQ client: with large enough buffer and header-index alignment
    logging.info("Connecting to Open Ephys GUI…")
    need_seconds = (S_ref / fs) + safety_margin_s
    client = ZMQClient(
        host_ip=http_ip,
        data_port=data_port,
        heartbeat_port=heartbeat_port,
        buffer_seconds=need_seconds,
        expected_channel_names=trained_names,
        auto_start=False,
        set_index_looping=True,
        align_to_header_index=True,  # write by header index so holes/loops are handled
        verbose=verbose,
    )

    # Make sure GUI is acquiring; if your environment requires it:
    client.gui.idle()
    time.sleep(0.1)
    client.gui.start_acquisition()
    client.start()
    client.wait_for_expected_channels(timeout=5.0)

    fs = float(client.fs)
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms / 1000.0 * fs))
    logging.info(f"ZMQClient started: fs={fs} Hz, W={W} samples, S={S} samples.")

    print(f"meta selected channels: {meta['data']['selected_channels']}")
    name_to_idx = {normalize_name(n): i for i, n in enumerate(client.channel_names)}
    want_norm = [normalize_name(nm) for nm in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in name_to_idx]
    if missing:
        client.stop()
        raise RuntimeError(f"ZMQ stream missing channels required by model: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    selected_idx = [name_to_idx[n] for n in want_norm]
    logging.info(f"Locked {len(selected_idx)} training channels (of {C_ref} file channels) to incoming ZMQ names.")

    # progress
    pbar = tqdm(total=S_ref, desc="ZMQ", unit="samples",
                dynamic_ncols=True, disable=not show_progress)

    # Preprocessor + warmup
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)

    # warm = client.get_latest_window(max(1, warmup_ms))
    # while warm is None or warm.size == 0:
    #     time.sleep(0.01)
    #     warm = client.get_latest(max(1, warmup_ms))
    # warm = warm[selected_idx, :] if warm.ndim == 2 else warm[np.array(selected_idx), None]
    warmup_samp = max(1, int(round(warmup_ms / 1000.0 * fs)))
    warm = None
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm, _ = client.get_latest(warmup_samp)  # (Y, t)
    warm = warm[selected_idx, :] if warm.ndim == 2 else warm[np.array(selected_idx), None]
    _ = pre.preprocess(warm)

    # --- Model ---
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # Streaming loop (drain_new + stall/end logic)
    preds, starts = [], []
    ring = np.zeros((len(selected_idx), W), dtype=np.float32)
    carry = None

    # --- progress & stall bookkeeping ---
    t0_sec = float(emg_t[0])  # e.g. ~80.6 s
    offset_samples = int(round(t0_sec * fs))  # convert to samples
    base_index = offset_samples  # where we started counting
    base0 = int(client.global_sample_index)  # where we started counting
    last_total = base0
    last_change = time.time()

    #pbar.total = S_ref
    #pbar.n = 0
    #pbar.refresh()

    emitted_end = 0
    last_seen = 0
    try:
        while True:
            total = int(client.global_sample_index)
            collected = max(0, total - base0)

            # end conditions
            if collected >= S_ref:
                logging.info("Collected target sample count (>= OEBin).")
                break
            if total == last_total:
                # stalled? (header index didn't advance)
                if (time.time() - last_change) >= stall_timeout_s:
                    logging.info("Header index stalled; stopping capture.")
                    break
                time.sleep(0.01)
                continue

            # we have new samples
            n_new = total - last_total
            last_change = time.time()

            # pull exactly the NEW tail from the header-aligned ring
            # (get_latest is aligned to header index in your ZMQClient)
            Y_new, _t = client.get_latest(n_new)  # (C_all, n_new)
            Y_new = Y_new[selected_idx, :]  # keep training channels

            # preprocess (rectify + envelope)
            y_new = pre.preprocess(Y_new)  # (C_sel, n_new)

            # stitch with carry and make windows of length W stepping by S
            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            #    padded = np.concatenate([carry, Yp], axis=1)
            #    abs0 = last_total - carry.shape[1]  # absolute index of padded[:, 0]
            #else:
            #    padded = Yp
            #    abs0 = last_total

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

            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            if pbar is not None:
                pbar.update(n_new); pbar.set_postfix_str(f"t={total / fs:.2f}s  pred={pred}")
            last_total = total
            # small breather to avoid tight loop when data is already complete
            time.sleep(0.001)

    finally:
        pbar.close()
        # stop GUI first so the header index stops advancing; then stop client
        client.gui.idle()
        client.stop()

    # --- Evaluate against emg.event ---
    if len(starts) == 0:
        logging.warning("No windows were produced; nothing to evaluate.")
        return
    starts = np.asarray(starts, dtype=np.int64)
    evaluate_against_events(root_dir, starts, preds)


def parse_args():
    p = argparse.ArgumentParser("3a (ZMQ): Real-time gesture prediction from Open Ephys GUI")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--window_ms", type=int, default=None)
    p.add_argument("--step_ms", type=int, default=None)
    p.add_argument("--warmup_ms", type=int, default=500)
    p.add_argument("--http_ip", type=str, default="127.0.0.1")
    p.add_argument("--data_port", type=str, default="5556")
    p.add_argument("--heartbeat_port", type=str, default=None)
    p.add_argument("--stall_timeout_s", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no_progress", action="store_true")
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
        http_ip=cfg.get("http_ip", "127.0.0.1"),
        data_port=cfg.get("data_port", "5556"),
        heartbeat_port=cfg.get("heartbeat_port", None),
        stall_timeout_s=cfg.get("stall_timeout_s", 1.0),
        verbose=cfg.get("verbose", False),
        show_progress=not cfg.get("no_progress", False),
    )
