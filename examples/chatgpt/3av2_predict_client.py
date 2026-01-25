#!/usr/bin/env python3
"""
3a_predict_client.py
Strict real-time (playback) gesture prediction:
- Locks channel selection by NAME to the exact training order from metadata
- No pre-buffering; pulls latest samples only; updates a ring buffer; predicts per step
"""
import os, time, argparse, logging
import numpy as np
from tqdm import tqdm

from pyoephys.interface import OEBinPlaybackClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.io import load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
        warmup_ms: int = 500, selected_channels=None, verbose: bool = False, show_progress: bool = True):

    logging.basicConfig(level=(logging.DEBUG if verbose else logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ---- Load label-specific metadata & lock params
    meta = load_metadata_json(root_dir, label)
    window_ms, step_ms, selected_channels, env_cut = lock_params_to_meta(
        meta['data'], window_ms, step_ms, selected_channels
    )
    if not selected_channels:
        raise RuntimeError("metadata missing selected_channels (training channel order).")

    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    C_ref = len(meta['data']['selected_channels'])

    # Playback client
    file_path = os.path.join(root_dir, "raw", "gestures")
    logging.info(f"Starting OEBinPlaybackClient for {file_path} with label '{label}'")
    client = OEBinPlaybackClient(file_path, loopback=False, enable_lsl=False, verbose=verbose)
    client.start()

    fs = float(client.fs)
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))
    logging.info(f"ZMQClient started: fs={fs} Hz, W={W} samples, S={S} samples.")

    # ---- Map training channel NAMES -> playback indices (exact order)
    print(f"meta selected channels: {meta['data']['selected_channels']}")
    name_to_idx = {normalize_name(n): i for i, n in enumerate(client.channel_names)}
    want_norm = [normalize_name(nm) for nm in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in name_to_idx]
    if missing:
        client.stop()
        raise RuntimeError(f"Playback stream missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    selected_idx = [name_to_idx[n] for n in want_norm]
    logging.info(f"Locked {len(selected_idx)} training channels (of {C_ref} file channels) to incoming ZMQ names.")

    # Progress
    base_index = int(round(getattr(client, "t_file", [0])[0] * fs)) if getattr(client, "t_file", None) is not None else 0
    total_samples_declared = int(getattr(client, "n_samples", 0))
    pbar = tqdm(total=total_samples_declared, desc="Playback", unit="samples",
                dynamic_ncols=True, disable=not show_progress)

    # Preprocessor + warmup
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)

    warm = client.get_latest_window(max(1, warmup_ms))
    while warm is None or warm.size == 0:
        time.sleep(0.01)
        warm = client.get_latest_window(max(1, warmup_ms))
    warm = warm[selected_idx, :] if warm.ndim == 2 else warm[np.array(selected_idx), None]
    _ = pre.preprocess(warm)

    # Model
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # Loop
    preds, starts = [], []
    ring = np.zeros((len(selected_idx), W), dtype=np.float32)
    carry = None

    emitted_end = 0
    last_seen = 0
    try:
        while not getattr(client, "is_done", lambda: False)():
            total = int(getattr(client, "total_samples", 0))
            new_n = total - last_seen
            if new_n <= 0:
                time.sleep(0.005); continue

            need_ms = max(1, int(round(1000.0 * new_n / fs)))
            chunk = client.get_latest_window(need_ms)
            if chunk is None or chunk.size == 0:
                time.sleep(0.002); continue

            # Select training channels, exact order
            if chunk.ndim == 1:
                chunk = chunk[np.array(selected_idx), None]
            else:
                chunk = chunk[selected_idx, :]

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

            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            if pbar is not None:
                pbar.update(new_n); pbar.set_postfix_str(f"t={total / fs:.2f}s  pred={pred}")
            last_seen = total
            # small breather to avoid tight loop when data is already complete
            time.sleep(0.001)

    finally:
        client.stop()
        if pbar is not None:
            pbar.close()
        logging.info("Streaming stopped.")

    evaluate_against_events(root_dir, np.asarray(starts, dtype=int), preds)


def parse_args():
    p = argparse.ArgumentParser("3a: Real-time gesture prediction from playback client (strict latest-sample mode)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--warmup_ms",   type=int, default=500)
    # deprecated: channel indices now ignored; channel order comes from metadata
    p.add_argument("--selected_channels", nargs="+", type=int, default=None)
    p.add_argument("--verbose",     action="store_true")
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
        selected_channels=None,  # always ignored; we use training names
        verbose=cfg.get("verbose", False),
        show_progress=not cfg.get("no_progress", False),
    )
