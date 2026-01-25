#!/usr/bin/env python3
"""
Train an EMG gesture classifier from a pre-built dataset (npz).
"""

import os
import json
import argparse
import numpy as np
from pyoephys.io import load_config_file
from pyoephys.ml import ModelManager, EMGClassifier


def train_model(cfg: dict):
    label = cfg.get("label", "")
    data_path = os.path.join(cfg["root_dir"], f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    # Load dataset
    data = np.load(data_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    emg_fs = float(data["emg_fs"])
    window_ms = int(data["window_ms"])
    step_ms = int(data["step_ms"])

    feature_spec = None
    if "feature_spec" in data.files:
        feature_spec = json.loads(str(data["feature_spec"]))

    manager = ModelManager(root_dir=cfg["root_dir"], label=label, model_cls=EMGClassifier, config=cfg)

    if cfg.get("kfold", False):
        cv_metrics = manager.cross_validate(X, y)
        print("Cross-validation metrics:")
        for i, m in enumerate(cv_metrics, 1):
            print(f"Fold {i}: {m}")
    elif not os.path.isfile(manager.model_path) or cfg.get("overwrite", False):
        print("Training new model...")
        manager.train(X, y)
    else:
        print("Model exists; loading.")
        manager.load_model()

    # Label classes (don’t assume LabelEncoder exists)
    if hasattr(manager, "label_encoder"):
        label_classes = [str(c) for c in manager.label_encoder.classes_]
    elif hasattr(manager, "classes_"):
        label_classes = [str(c) for c in manager.classes_]
    else:
        label_classes = sorted([str(c) for c in np.unique(y)])

    # Scalar info
    scaler_mean = getattr(getattr(manager, "scaler", None), "mean_", None)
    scaler_scale = getattr(getattr(manager, "scaler", None), "scale_", None)
    if scaler_mean is not None: scaler_mean = scaler_mean.tolist()
    if scaler_scale is not None: scaler_scale = scaler_scale.tolist()

    meta = manager.build_metadata(
        sample_rate_hz=emg_fs,
        window_ms=window_ms,
        step_ms=step_ms,
        envelope_cutoff_hz=cfg.get("envelope_cutoff_hz", 5.0),
        selected_channels=data.get("selected_channels", []),  # from dataset npz
        channel_names=data["channel_names"].tolist() if "channel_names" in data.files else [],
        feature_spec=feature_spec,
        n_features=X.shape[1],
        label_classes=label_classes,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        extra=manager.eval_metrics,
    )
    manager.save_metadata(meta)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Train an EMG gesture classification model.")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",   type=str, required=True)
    p.add_argument("--label",      type=str, default="")
    p.add_argument("--kfold",      action="store_true")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "kfold": args.kfold or cfg.get("kfold", False),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    train_model(cfg)
