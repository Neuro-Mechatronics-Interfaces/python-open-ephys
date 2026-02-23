#!/usr/bin/env python3
"""
2_train_model.py — EMG Gesture Model Trainer
=============================================
Loads the .npz dataset produced by 1_build_dataset.py, trains an EMG gesture
classifier, and writes a model + metadata.json that 3_predict_realtime.py and
predict.py both read.

Default behaviour (zero arguments)
------------------------------------
  Reads  ./data/gestures/training_dataset.npz
  Saves model artefacts to  ./data/model/

Usage
-----
  # Train on example data (after running 1_build_dataset.py):
  python 2_train_model.py

  # Custom dataset path:
  python 2_train_model.py --dataset_path data/my_training.npz

  # K-fold cross-validation instead of a single train/val split:
  python 2_train_model.py --kfold

  # Retrain even if a model already exists:
  python 2_train_model.py --overwrite
"""

import argparse
import os
from pathlib import Path

import numpy as np

from pyoephys.io import load_simple_config, load_dataset
from pyoephys.ml import ModelManager, EMGClassifier, write_training_metadata


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_DEFAULT_DATASET = _HERE / "data" / "gestures" / "training_dataset.npz"
_DEFAULT_ROOT    = _HERE / "data" / "gesture_model"
_CONFIG_FILE     = _HERE / ".gesture_config"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    dataset_path: str | None = None,
    root_dir: str | None = None,
    label: str = "",
    kfold: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
):
    """
    Train an EMG gesture classifier from a pre-built .npz dataset.

    Parameters
    ----------
    dataset_path : Path to the .npz dataset file.  When None, looks for
                   ``training_dataset.npz`` inside *root_dir*.
    root_dir     : Directory where model artefacts are saved (``model/``
                   sub-folder).  Defaults to the directory that contains
                   *dataset_path*.
    label        : Optional tag for output file names.
    kfold        : Run k-fold cross-validation instead of a single split.
    overwrite    : Retrain even if a model already exists.
    verbose      : Print extra diagnostic information.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────
    if dataset_path is None:
        dataset_path = str(_DEFAULT_DATASET)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run  python 1_build_dataset.py  first."
        )

    if root_dir is None:
        root_dir = str(_DEFAULT_ROOT)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset: {dataset_path}")
    X, y, ds_meta = load_dataset(dataset_path)
    print(f"  X: {X.shape}  |  classes: {sorted(set(y.tolist()))}")

    fs         = ds_meta["fs"]
    chan_names = ds_meta["channel_names"]
    sel_ch     = ds_meta["selected_channels"]
    window_ms  = ds_meta["window_ms"]
    step_ms    = ds_meta["step_ms"]
    env_cut    = ds_meta["envelope_cutoff_hz"]
    feat_names = ds_meta["feature_names"]

    # ── ModelManager ──────────────────────────────────────────────────────
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier)

    if kfold:
        print("Running k-fold cross-validation …")
        cv_result = manager.cross_validate(X, y)
        for i, m in enumerate(cv_result["folds"], start=1):
            print(f"  Fold {i}: {m}")
        print(f"  Summary: {cv_result['summary']}")
    elif not os.path.isfile(manager.model_path) or overwrite:
        print("Training …")
        manager.train(X, y)
        print(f"  Done.  Validation metrics: {manager.eval_metrics}")
    else:
        print("Model already exists — loading.  (Pass --overwrite to retrain.)")
        manager.load_model()

    # ── Write metadata.json (single source of truth for prediction scripts) ──
    n_features = len(manager.scaler.mean_) if manager.scaler is not None else X.shape[1]
    write_training_metadata(
        root_dir=root_dir,
        window_ms=window_ms,
        step_ms=step_ms,
        envelope_cutoff_hz=env_cut,
        channel_names=chan_names,
        selected_channels=sel_ch,
        sample_rate_hz=fs,
        n_features=n_features,
        feature_set=feat_names,
        ignore_labels=ds_meta.get("ignore_labels") or [],
        require_complete=True,
        required_fraction=1.0,
        channel_wait_timeout_sec=15.0,
    )
    return manager.model, manager.scaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train an EMG gesture classifier from a .npz dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset_path", default=None,
        help=f"Path to .npz dataset.  Default: {_DEFAULT_DATASET}",
    )
    p.add_argument(
        "--root_dir", default=None,
        help=f"Directory for model output (default: {_DEFAULT_ROOT}).",
    )
    p.add_argument("--label",     default="",    help="Model label / output file tag.")
    p.add_argument("--kfold",     action="store_true", help="K-fold cross-validation mode.")
    p.add_argument("--overwrite", action="store_true", help="Retrain even if model exists.")
    p.add_argument("--config_file", default=None)
    p.add_argument("--verbose",   action="store_true")
    args = p.parse_args()

    # Optional config file
    cfg = {}
    config_path = args.config_file or _CONFIG_FILE
    if Path(config_path).is_file():
        cfg = load_simple_config(str(config_path))

    train_model(
        dataset_path=args.dataset_path or cfg.get("dataset_path") or cfg.get("save_path"),
        root_dir=args.root_dir or cfg.get("root_dir"),
        label=args.label or cfg.get("label", ""),
        kfold=args.kfold,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
