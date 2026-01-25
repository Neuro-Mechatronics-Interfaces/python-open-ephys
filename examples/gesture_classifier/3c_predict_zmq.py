#!/usr/bin/env python3
import os, json, time, argparse, logging
import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor, extract_features
from pyoephys.ml import ModelManager, EMGClassifier
from pyoephys.io import load_config_file

from tqdm.auto import tqdm


# -------------------- helpers --------------------

def load_metadata(meta_path: str) -> dict:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f) or {}
    # sanity
    needed = ["window_ms", "step_ms", "envelope_cutoff_hz"]
    for k in needed:
        if k not in meta:
            raise RuntimeError(f"metadata.json missing '{k}'")
    return meta


def merge_cfg(meta: dict, args: argparse.Namespace) -> dict:
    """
    Merge precedence: CLI > metadata.json defaults.
    Only include keys the predictor actually uses.
    """
    cfg = {}

    # required core
    cfg["root_dir"]  = args.root_dir
    cfg["label"]     = args.label
    cfg["verbose"]   = args.verbose

    # window/step/filter from metadata (can be overridden by CLI, if provided)
    cfg["window_ms"] = int(args.window_ms) if args.window_ms is not None else int(meta["window_ms"])
    cfg["step_ms"]   = int(args.step_ms)   if args.step_ms   is not None else int(meta["step_ms"])
    cfg["envelope_cutoff_hz"] = float(args.envelope_cutoff_hz) if args.envelope_cutoff_hz is not None else float(meta.get("envelope_cutoff_hz", 5.0))

    # channel order expectations
    # prefer names for robust mapping; fall back to indices
    trained_names = meta.get("channel_names", None)
    trained_idx   = meta.get("selected_channels", None)
    if trained_names is None and trained_idx is None:
        raise RuntimeError("metadata.json is missing 'channel_names' and 'selected_channels'. Please re-run training to write channel_names so we can enforce channel count.")
    cfg["trained_channel_names"] = trained_names
    cfg["trained_selected_indices"] = trained_idx

    # fs (optional in metadata; if missing, use ZMQ-reported fs)
    cfg["sample_rate_hz"] = float(meta["sample_rate_hz"]) if "sample_rate_hz" in meta else None

    # feature dimension (used for a strong check post-scaler load)
    cfg["n_features"] = int(meta["n_features"]) if "n_features" in meta else None

    # ZMQ readiness knobs (metadata can carry these; CLI can override)
    cfg["require_complete"] = bool(meta.get("require_complete", True))
    if args.require_complete is not None:
        cfg["require_complete"] = bool(args.require_complete)

    cfg["required_fraction"] = float(meta.get("required_fraction", 1.0))
    if args.required_fraction is not None:
        cfg["required_fraction"] = float(args.required_fraction)

    cfg["channel_wait_timeout_sec"] = float(meta.get("channel_wait_timeout_sec", 15.0))
    if args.channel_wait_timeout_sec is not None:
        cfg["channel_wait_timeout_sec"] = float(args.channel_wait_timeout_sec)

    # stop conditions
    cfg["duration_sec"]          = args.duration_sec
    cfg["n_windows"]             = args.n_windows
    cfg["inactivity_timeout_sec"]= args.inactivity_timeout_sec

    # ZMQ connection
    cfg["zmq_ip"]         = args.zmq_ip
    cfg["data_port"]      = int(args.data_port)
    cfg["heartbeat_port"] = int(args.heartbeat_port)

    return cfg


def compute_target_windows(duration_sec, window_ms, step_ms):
    if duration_sec is None:
        return None
    T_ms = int(round(float(duration_sec) * 1000.0))
    if T_ms < window_ms:
        return 1
    return 1 + (T_ms - window_ms) // step_ms


# -------------------- main --------------------

def predict_from_zmq(cfg: dict):
    root_dir = cfg["root_dir"]
    label    = cfg.get("label", "")
    verbose  = bool(cfg.get("verbose", False))

    # logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    # metadata
    meta_path = os.path.join(root_dir, "model", "metadata.json")
    meta = load_metadata(meta_path)
    # recompute merged config with any CLI overrides
    cfg = merge_cfg(meta, argparse.Namespace(**cfg))

    window_ms = int(cfg["window_ms"])
    step_ms   = int(cfg["step_ms"])
    env_cut   = float(cfg["envelope_cutoff_hz"])
    require_complete       = bool(cfg["require_complete"])
    required_fraction      = float(cfg["required_fraction"])
    channel_wait_timeout_s = float(cfg["channel_wait_timeout_sec"])

    # connect ZMQ
    client = ZMQClient(
        zqm_ip=cfg["zmq_ip"],
        http_ip="127.0.0.1",
        data_port=cfg["data_port"],
        heartbeat_port=cfg["heartbeat_port"],
        window_secs=max(5.0, window_ms/1000.0*3.0),  # keep a few windows in buffer
        channels=None,       # we'll map after we see channel_names
        auto_start=True,
        verbose=verbose,
    )

    # Use metadata fs if present, else trust client.fs
    fs = float(cfg["sample_rate_hz"]) if cfg["sample_rate_hz"] is not None else float(client.fs)
    W  = int(round(window_ms * fs / 1000.0))
    S  = int(round(step_ms   * fs / 1000.0))

    # map ZMQ channels to training order (prefer names)
    trained_names = cfg["trained_channel_names"]
    trained_idx   = cfg["trained_selected_indices"]
    if trained_names is not None:
        # build mapping by name
        name_to_zidx = {nm: i for i, nm in enumerate(client.channel_names)}
        present = [nm for nm in trained_names if nm in name_to_zidx]
        frac = len(present) / float(len(trained_names))
        # wait for missing channels if required
        t0 = time.time()
        while frac < required_fraction:
            if require_complete and (time.time() - t0) > channel_wait_timeout_s:
                missing = [nm for nm in trained_names if nm not in name_to_zidx]
                raise TimeoutError(f"Timed out waiting for required channels via ZMQ. Missing: {missing}")
            # refresh mapping
            name_to_zidx = {nm: i for i, nm in enumerate(client.channel_names)}
            present = [nm for nm in trained_names if nm in name_to_zidx]
            frac = len(present) / float(len(trained_names))
            time.sleep(0.05)

        # finalized per-training order mapping (-1 for missing)
        map_zmq_idx = [name_to_zidx.get(nm, -1) for nm in trained_names]
        # set client to pull only channels that actually exist (for aligned drains)
        client.channel_index = [i for i in map_zmq_idx if i >= 0]
        # we might be missing some; we will zero-fill them when constructing feature window
        trained_C = len(trained_names)

    else:
        # fallback: indices from metadata (less robust than names)
        trained_C = len(trained_idx)
        # ensure ZMQ stream has at least max index
        max_idx = max(trained_idx)
        if max_idx >= len(client.channel_names):
            raise RuntimeError(
                f"ZMQ stream has {len(client.channel_names)} chans but model expects index up to {max_idx}."
            )
        client.channel_index = list(trained_idx)
        # identity mapping (all present)
        map_zmq_idx = list(trained_idx)

    # wait for an aligned full window across the client-selected channels
    deadline = time.time() + channel_wait_timeout_s
    while time.time() < deadline:
        if client.complete_count() >= W:
            break
        time.sleep(0.01)
    else:
        raise TimeoutError("Timed out waiting for W aligned samples across selected ZMQ channels.")

    # model manager (loads model, scaler, encoder, etc.)
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)

    # preprocessor same as training
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)

    # warm-up filters with ~1s worth of aligned data
    warm_steps = max(1, int((max(1000, window_ms) / step_ms)))
    for _ in range(warm_steps):
        idx, step_raw = client.drain_aligned(S)
        if step_raw is None:
            time.sleep(0.005)
            continue
        # reorder + zero-fill to training order/size
        if trained_names is not None:
            # construct (trained_C, S)
            step_ord = np.zeros((trained_C, step_raw.shape[1]), dtype=np.float32)
            write_r = 0
            for r, zidx in enumerate(map_zmq_idx):
                if zidx >= 0:
                    # find position of zidx within client.channel_index
                    try:
                        pos = client.channel_index.index(zidx)
                        step_ord[r, :] = step_raw[pos, :]
                    except ValueError:
                        # rare race: mapping changed mid-warmup; just leave zeros
                        pass
            y_new = pre.preprocess(step_ord)
        else:
            y_new = pre.preprocess(step_raw)

    # ring buffer in training channel order
    ring = np.zeros((trained_C, W), dtype=np.float32)

    # stopping conditions
    duration_sec  = cfg.get("duration_sec", None)
    n_windows_max = cfg.get("n_windows", None)
    target_windows = n_windows_max if n_windows_max is not None else compute_target_windows(duration_sec, window_ms, step_ms)

    inactivity_timeout = float(cfg.get("inactivity_timeout_sec", 10.0))
    last_data_time = time.time()

    if tqdm is not None:
        pbar = tqdm(total=target_windows, desc="ZMQ (preds)", unit="win", dynamic_ncols=True)

    windows_done = 0
    preds = []

    try:
        while True:
            idx, step_raw = client.drain_aligned(S)
            if step_raw is None:
                if (time.time() - last_data_time) > inactivity_timeout:
                    logging.info("No new aligned data; stopping.")
                    break
                time.sleep(0.005)
                continue

            last_data_time = time.time()

            # reorder + zero-fill to match training channel order
            if trained_names is not None:
                step_ord = np.zeros((trained_C, step_raw.shape[1]), dtype=np.float32)
                for r, zidx in enumerate(map_zmq_idx):
                    if zidx >= 0:
                        try:
                            pos = client.channel_index.index(zidx)
                            step_ord[r, :] = step_raw[pos, :]
                        except ValueError:
                            # mapping drift; keep zeros this step
                            pass
            else:
                # indices already align with training order
                step_ord = step_raw

            # preprocess only the new step (stateful) and roll into window
            y_new = pre.preprocess(step_ord)        # (trained_C, S)
            ring  = np.concatenate([ring[:, S:], y_new[:, -S:]], axis=1)

            # features and predict
            feats = extract_features(ring).reshape(1, -1)
            if feats.shape[1] != n_features_expected:
                raise ValueError(f"Feature dim {feats.shape[1]} != scaler expectation {n_features_expected}")
            pred = manager.predict(feats)[0]
            preds.append(pred)

            # sample-derived timestamp
            t_end_s = float(idx[-1] + 1) / fs

            # UI/progress
            if tqdm is not None:
                windows_done += 1
                pbar.update(1)
                pbar.set_postfix_str(f"t={t_end_s:0.1f}s  pred={pred}")

            # stop?
            if target_windows is not None and windows_done >= target_windows:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if tqdm is not None:
            pbar.close()
        client.stop()
        logging.info("ZMQ streaming stopped.")


if __name__ == "__main__":

    p = argparse.ArgumentParser("3c_predict_zmq: Real-time EMG gesture prediction from Open Ephys GUI via ZMQ")
    p.add_argument("--config_file", type=str, default=None, help="Optional JSON with overrides")
    p.add_argument("--root_dir", type=str, required=True, help="Session root with model/metadata.json")
    p.add_argument("--label", type=str, default="", help="Optional label for model variant")
    p.add_argument("--window_ms", type=int, default=None)
    p.add_argument("--step_ms", type=int, default=None)
    p.add_argument("--envelope_cutoff_hz", type=float, default=None)
    p.add_argument("--duration_sec", type=float, default=None, help="Run for N seconds")
    p.add_argument("--n_windows", type=int, default=None, help="Run for N windows")
    p.add_argument("--inactivity_timeout_sec", type=float, default=10.0)

    # ZMQ connection
    p.add_argument("--zmq_ip", type=str, default="tcp://localhost")
    p.add_argument("--data_port", type=int, default=5556)
    p.add_argument("--heartbeat_port", type=int, default=5557)

    # Readiness gating
    p.add_argument("--require_complete", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--required_fraction", type=float, default=None)
    p.add_argument("--channel_wait_timeout_sec", type=float, default=None)    args = ap.parse_args()

    p.add_argument("--verbose", action="store_true")

    # start from config file if provided
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}

    # inject CLI args (argparse.Namespace -> dict)
    for k, v in vars(args).items():
        cfg[k] = v

    predict_from_zmq(cfg)
