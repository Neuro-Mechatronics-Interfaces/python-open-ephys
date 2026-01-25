#!/usr/bin/env python3
"""
Offline gesture prediction from OEBin (same settings as training).
"""

import os
import argparse
import logging
import numpy as np
from pyoephys.io import load_oebin_file, load_config_file, lock_params_to_meta
from pyoephys.ml import ModelManager, EMGClassifier, load_training_metadata, evaluate_against_events
from pyoephys.processing import EMGPreprocessor


def run(root_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
        selected_channels=None, verbose: bool = False):

    # Set up logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load training metadata to lock parameters
    meta = load_training_metadata(root_dir)
    window_ms, step_ms, selected_channels, env_cut = lock_params_to_meta(
        meta, window_ms, step_ms, selected_channels
    )

    # Load raw EMG data from oebin file, will use this as reference
    raw_dir = os.path.join(root_dir, "raw")
    data = load_oebin_file(raw_dir, verbose=verbose)
    emg_fs = float(data["sample_rate"])
    emg_t = data["t_amplifier"]
    emg = data["amplifier_data"]
    if selected_channels is not None:
        emg = emg[selected_channels, :]

    # Make preprocessor
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)

    # Extract features, same as the built dataset
    X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )

    logging.info(f"Extracted feature matrix X with shape {X.shape}")
    start_index = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)

    evaluate_against_events(root_dir, window_starts, y_pred)


if __name__ == "__main__":

    p = argparse.ArgumentParser("3b: Offline EMG gesture prediction from OEBin")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--channels", nargs="+", type=int, default=None)
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", 200),
        "step_ms": args.step_ms or cfg.get("step_ms", 50),
        "selected_channels": args.channels or cfg.get("channels", None),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    lvl = logging.DEBUG if cfg["verbose"] else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)
    run(**cfg)