#!/usr/bin/env python3
import os, time, argparse, logging, json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from pyoephys.interface import OEBinPlaybackClient
from pyoephys.processing import EMGPreprocessor, extract_features  # adjust if your module path differs
from pyoephys.ml import ModelManager, EMGClassifier
from pyoephys.io import labels_from_events, load_config_file


def predict_from_playback(
    root_dir: str,
    label: str = "",
    window_ms: int = 200,
    step_ms: int = 50,
    warmup_ms: int = 500,
    selected_channels=None,
    verbose: bool = False,
    progress: bool = True,
):
    # ---- logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    # ---- load training metadata to FORCE identical settings
    meta_path = os.path.join(root_dir, "model", "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata.json at {meta_path} (needed to match training settings)")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # lock to training values (fallback to args if not present)
    window_ms = int(meta.get("window_ms", window_ms or 200))
    step_ms = int(meta.get("step_ms", step_ms or 50))
    env_cut = float(meta.get("envelope_cutoff_hz", 5.0))
    tr_channels = meta.get("selected_channels", None)
    if tr_channels is not None:
        selected_channels = tr_channels

    # ---- playback client
    oebin_path = os.path.join(root_dir, "raw", "gestures")
    client = OEBinPlaybackClient(oebin_path, loopback=False, enable_lsl=False, verbose=verbose)
    client.start()

    fs = float(client.sampling_rate)
    W = int(round(window_ms / 1000.0 * fs))
    S = int(round(step_ms   / 1000.0 * fs))

    # Absolute sample-index offset that matches events file
    base_index = int(round(client.t_file[0] * fs)) if getattr(client, "t_file", None) is not None else 0

    # total samples in file for ETA
    total_samples = int(getattr(client, "n_samples", 0))  # OEBinPlaybackClient sets this
    pbar = None
    #prev_n = 0
    last_seen = 0
    if tqdm is not None and total_samples > 0:
        pbar = tqdm(total=total_samples, desc="Playback", unit="samples", dynamic_ncols=True)

    # ---- preprocessor: MUST match training (envelope_cutoff=5.0 used in your build)
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=5.0, verbose=verbose)

    # ---- warmup: prime filter states with a short window
    warm = client.get_latest_window(max(1, warmup_ms))
    if warm is None or warm.size == 0:
        logging.info("Waiting for playback to start producing data...")
        while True:
            warm = client.get_latest_window(max(1, warmup_ms))
            if warm is not None and warm.size:
                break
            time.sleep(0.01)

    if selected_channels is not None:
        warm = warm[selected_channels, :]
    # process warmup once
    _ = pre.preprocess(warm)

    # ring buffer (C, W)
    C = warm.shape[0]
    ring = np.zeros((C, W), dtype=np.float32)

    # ---- model manager
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()  # loads model + scaler + encoder (+ pca if any)
    n_features_expected = len(manager.scaler.mean_)

    preds, starts = [], []

    def _is_done():
        return getattr(client, "is_done", lambda: False)()

    # streaming stepper: emit 1 prediction per S samples, even if big chunks arrive
    emitted_end = 0  # samples we have emitted predictions for
    carry = None  # < S filtered samples carried to next loop

    try:
        while not _is_done():
            total = int(client.total_samples)  # how many samples playback has produced
            new_n = total - last_seen  # new samples since last loop
            if new_n <= 0:
                time.sleep(0.005)
                continue

            # pull a step-sized chunk (in ms). May be short early on—guard it.
            need_ms = max(1, int(round(1000.0 * new_n / fs)))
            chunk = client.get_latest_window(need_ms)  # (C, ~new_n) tail, may overlap with previous
            if chunk is None or chunk.size == 0:
                time.sleep(0.005)
                continue

            if selected_channels is not None:
                chunk = chunk[selected_channels, :]

            # ensure (C, N)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            #if chunk.shape[0] != C:
            #    # channel count changed? re-init ring & preprocessor
            #    C = chunk.shape[0]
            #    ring = np.zeros((C, W), dtype=np.float32)
            #    pre.reset_states()

            # If playback hasn't advanced enough yet, wait
            #if chunk.shape[1] < S:
            #    time.sleep(0.001)
            #    continue

            # preprocess just the new tail (stateful)
            y_new = pre.preprocess(chunk)  # (C, Nnew)
            #y_step = y_new[:, -S:]         # use last S samples for the step

            # prepend any leftover filtered tail from previous iteration
            if carry is not None and carry.shape[1] > 0:
                y_new = np.concatenate([carry, y_new], axis=1)

            Nf = y_new.shape[1]
            if Nf < S:
                # not enough to emit one step yet; keep as carry
                carry = y_new
                last_seen = total
                if pbar is not None:
                    pbar.update(new_n)
                    pbar.set_postfix_str(f"t={total / fs:.1f}s  pred=…")
                continue

            # how many step-sized predictions can we emit now?
            n_steps = Nf // S
            for k in range(n_steps):
                # take exactly S filtered samples for this step
                y_step = y_new[:, k * S:(k + 1) * S]  # (C, S)

                # advance the ring by one step
                ring = np.concatenate([ring[:, S:], y_step], axis=1)

                # === FEATURES via the SAME CLASS METHOD as training ===
                # pass the ring as a full trace with window==step==W → 1 row
                feats = pre.extract_emg_features(
                    ring, window_ms=window_ms, step_ms=window_ms, return_windows=False
                )  # shape (1, D)

                # guardrail: scaler feature dim must match training
                if feats.shape[1] != n_features_expected:
                    raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")

                # predict (manager scales + pca + decode)
                pred = manager.predict(feats)[0]
                preds.append(pred)

                # LEFT-EDGE absolute index for this window
                emitted_end += S
                starts.append(base_index + emitted_end - W)

            # keep remainder (<S) for the next iteration
            rem = Nf - n_steps * S
            carry = y_new[:, -rem:] if rem > 0 else None

            # progress: mark the raw playback progress (not emitted_end)
            if pbar is not None:
                pbar.update(new_n)
                pbar.set_postfix_str(f"t={total / fs:.1f}s  pred={pred}")
            last_seen = total

            time.sleep(0.001)

            # push into ring
            # ring = np.concatenate([ring[:, S:], y_step], axis=1)
            #
            # # feature vector for the current window
            # feats = extract_features(ring)           # (D,)
            # if feats.ndim == 1:
            #     feats = feats.reshape(1, -1)
            # if feats.shape[1] != n_features_expected:
            #     raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")
            #
            # # predict (manager scales + decodes internally)
            # pred = manager.predict(feats)[0]
            # preds.append(pred)
            #
            # # LEFT EDGE, absolute index = base_index + (played_so_far - W)
            # abs_end = int(client.total_samples)
            # starts.append(base_index + abs_end - W)
            #
            # # progress update
            # if pbar is not None:
            #     delta = abs_end - prev_n
            #     if delta > 0:
            #         pbar.update(delta)
            #         # show current playback time and last prediction
            #         pbar.set_postfix_str(f"t={abs_end / fs:.1f}s  pred={pred}")
            #         prev_n = abs_end
            #
            # # throttle a bit to avoid busy-looping
            # time.sleep(max(0.0, (step_ms / 1000.0) * 0.25))

    except KeyboardInterrupt:
        pass

    finally:
        if pbar is not None:
            # ensure the bar completes even if we ended early
            if last_seen < total_samples:
                pbar.update(total_samples - last_seen)
            pbar.close()
        client.stop()
        logging.info("Streaming stopped.")

    # ---- offline evaluation (optional)
    ev_path = os.path.join(root_dir, "events", "emg.event")
    if os.path.isfile(ev_path) and len(starts) > 0:
        y_true = labels_from_events(ev_path, np.asarray(starts, dtype=int))
        mask = ~np.isin(y_true, ["Unknown", "Start"])
        y_true = np.asarray(y_true)[mask]
        y_pred = np.asarray(preds)[mask]

        if y_true.size == 0:
            logging.warning("No valid windows to evaluate (all Unknown/Start).")
        else:
            print("\n=== Classification Report ===")
            print(classification_report(y_true, y_pred))
            print("=== Confusion Matrix ===")
            print(confusion_matrix(y_true, y_pred))
    else:
        logging.info("Skipped offline evaluation (no events file or no predictions).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("3a_v5: Real-time gesture prediction from playback client")
    ap.add_argument("--config_file", type=str)
    ap.add_argument("--root_dir",    type=str, required=True)
    ap.add_argument("--label",       type=str, default="")
    ap.add_argument("--window_ms",   type=int, default=200)
    ap.add_argument("--step_ms",     type=int, default=50)
    ap.add_argument("--warmup_ms",   type=int, default=500)
    ap.add_argument("--selected_channels", nargs="+", type=int, default=None)
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update(vars(args))
    #predict_from_playback(**cfg)
    predict_from_playback(
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        warmup_ms=cfg.get("warmup_ms", 500),
        selected_channels=cfg.get("selected_channels", None),
        verbose=cfg.get("verbose", False),
        progress=not cfg.get("no_progress", False),
    )
