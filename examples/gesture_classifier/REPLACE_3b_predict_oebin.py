#!/usr/bin/env python3
import os, json, argparse, logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from pyoephys.io import load_oebin_file, load_config_file, labels_from_events
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier


def predict_offline_from_oebin(
    root_dir: str,
    label: str = "",
    window_ms: int | None = None,
    step_ms: int | None = None,
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
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    window_ms = int(meta.get("window_ms", window_ms or 200))
    step_ms   = int(meta.get("step_ms",   step_ms   or 50))
    env_cut   = float(meta.get("envelope_cutoff_hz", 5.0))
    tr_channels = meta.get("selected_channels", None)
    if tr_channels is not None:
        selected_channels = tr_channels

    # ---- load raw EMG (same source as training)
    raw_dir = os.path.join(root_dir, "raw")
    data    = load_oebin_file(raw_dir, verbose=verbose)
    emg_fs  = float(data["sample_rate"])
    emg_t   = data["t_amplifier"]          # seconds, (N,)
    emg     = data["amplifier_data"]       # (C, N)
    if selected_channels is not None:
        emg = emg[selected_channels, :]

    # ---- preprocess (MUST match training)
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)

    # ---- features (via the SAME class method)
    X = pre.extract_emg_features(
        emg_pp,
        window_ms=window_ms,
        step_ms=step_ms,
        progress=(progress and tqdm is not None),
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )
    print(f"[INFO] Features: X.shape = {X.shape}")

    # ---- window start indices (LEFT EDGE; absolute sample indices)
    start_index  = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index
    print(f"[INFO] Window starts: {len(window_starts)} "
          f"({window_starts[0]} .. {window_starts[-1]})")

    # ---- model (scaler + encoder applied inside manager.predict)
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)  # returns class names

    # ---- offline evaluation against events (same mapping as training)
    ev_path = os.path.join(root_dir, "events", "emg.event")
    if not os.path.isfile(ev_path):
        logging.warning(f"No events file at {ev_path}; skipping evaluation.")
        return

    y_true = labels_from_events(ev_path, window_starts)
    mask   = ~np.isin(y_true, ["Unknown", "Start"])
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]

    if y_true.size == 0:
        logging.warning("No valid windows to evaluate (all Unknown/Start).")
        return

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("3b_v6: Offline EMG gesture prediction from OEBin (training-identical)")
    ap.add_argument("--config_file", type=str)
    ap.add_argument("--root_dir",    type=str, required=True)
    ap.add_argument("--label",       type=str, default="")
    ap.add_argument("--window_ms",   type=int, default=None, help="If omitted, use training metadata")
    ap.add_argument("--step_ms",     type=int, default=None, help="If omitted, use training metadata")
    ap.add_argument("--selected_channels", nargs="+", type=int, default=None, help="If omitted, use training metadata")
    ap.add_argument("--verbose",     action="store_true")
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update(vars(args))

    predict_offline_from_oebin(
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        selected_channels=cfg.get("selected_channels", None),
        verbose=cfg.get("verbose", False),
        progress=not cfg.get("no_progress", False),
    )
