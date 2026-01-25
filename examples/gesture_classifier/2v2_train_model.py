#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

from pyoephys.io import load_config_file
from pyoephys.ml import ModelManager, EMGClassifier


# -------------------- metadata helper --------------------

def write_training_metadata(
    root_dir: str,
    *,
    window_ms: int,
    step_ms: int,
    envelope_cutoff_hz: float,
    channel_names=None,
    selected_channels=None,
    sample_rate_hz: float | None = None,
    n_features: int | None = None,
    feature_set=None,
    # ZMQ readiness knobs (what the ZMQ script reads)
    require_complete: bool = True,
    required_fraction: float = 1.0,
    channel_wait_timeout_sec: float = 15.0,
) -> None:
    """
    Merge/update fields in model/metadata.json so real-time ZMQ prediction has a single source of truth.
    Non-destructive: preserves any existing keys unless overridden here.
    """
    meta_dir = os.path.join(root_dir, "model")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "metadata.json")

    # Load existing metadata if present
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f) or {}
        except Exception:
            meta = {}

    # Core pipeline params (training-time truth)
    meta.update({
        "window_ms": int(window_ms),
        "step_ms": int(step_ms),
        "envelope_cutoff_hz": float(envelope_cutoff_hz),
        "require_complete": bool(require_complete),
        "required_fraction": float(required_fraction),
        "channel_wait_timeout_sec": float(channel_wait_timeout_sec),
    })

    if channel_names is not None:
        meta["channel_names"] = list(channel_names)
    if selected_channels is not None:
        meta["selected_channels"] = [int(i) for i in selected_channels]
    if sample_rate_hz is not None:
        meta["sample_rate_hz"] = float(sample_rate_hz)
    if n_features is not None:
        meta["n_features"] = int(n_features)
    if feature_set is not None:
        meta["feature_set"] = list(feature_set)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated at {meta_path}")


# -------------------- training script --------------------

def train_model(cfg):
    root_dir = cfg["root_dir"]
    label    = cfg.get("label", "")
    kfold    = cfg.get("kfold", False)
    overwrite = cfg.get("overwrite", False)
    verbose   = cfg.get("verbose", False)

    data_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    # Load features and labels (+ any helpful metadata if present)
    print(f"Loading dataset from {data_path}")
    ds = np.load(data_path, allow_pickle=True)
    X, y = ds["X"], ds["y"]
    print(f"Data shape - X: {X.shape}, y: {y.shape}")

    # Optional items saved by your dataset builder (defensive defaults if absent)
    fs   = float(ds["fs"]) if "fs" in ds.files else 2000.0
    chan_names = ds["channel_names"].tolist() if "channel_names" in ds.files else None
    sel_ch     = ds["selected_channels"].tolist() if "selected_channels" in ds.files else None

    # Use dataset-stored preprocessing params if present; else sane defaults
    window_ms  = int(ds["window_ms"]) if "window_ms" in ds.files else 200
    step_ms    = int(ds["step_ms"]) if "step_ms" in ds.files else 50
    env_cut    = float(ds["envelope_cutoff_hz"]) if "envelope_cutoff_hz" in ds.files else 5.0

    # Optional feature names (purely informational)
    feat_names = ds["feature_names"].tolist() if "feature_names" in ds.files else None

    # Model manager
    output_dim = len(np.unique(y)) if y.ndim == 1 else y.shape[1]
    print(f"Output dimension for classification: {output_dim}, {np.unique(y)}")

    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config=cfg)

    if kfold:
        print("Running k-fold cross-validation...")
        cv_metrics = manager.cross_validate(X, y)
        print("Cross-validation metrics:")
        for i, m in enumerate(cv_metrics, start=1):
            print(f"Fold {i}: {m}")
        # Even after CV, keep metadata in sync with dataset so predictors know parameters
        n_features = X.shape[1]
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
            # ZMQ readiness knobs (adjust if you prefer different defaults)
            require_complete=True,
            required_fraction=1.0,
            channel_wait_timeout_sec=15.0,
        )
        return None, None

    elif not os.path.isfile(manager.model_path) or overwrite:
        print("Training new model...")
        manager.train(X, y)
        print("Training complete. Validation metrics:")
        print(manager.eval_metrics)
    else:
        print("Model already exists, loading existing model + scaler/encoder.")
        manager.load_model()

    # After train or load, write/update metadata for prediction scripts
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
        # ZMQ readiness knobs (what your ZMQ predictor reads)
        require_complete=True,     # block until all trained channels present
        required_fraction=1.0,     # set <1.0 to allow small dropouts (pads missing chans with zeros)
        channel_wait_timeout_sec=15.0,
    )

    return manager.model, manager.scaler


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train an EMG gesture classification model.")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",   type=str, default="")
    p.add_argument("--label",      type=str, default="")
    p.add_argument("--kfold",      action="store_true")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)

    cfg["root_dir"]  = args.root_dir if args.root_dir is not None else cfg.get("root_dir", "")
    cfg["label"]     = args.label if args.label is not None else cfg.get("label", "")
    cfg["kfold"]     = args.kfold
    cfg["overwrite"] = args.overwrite
    cfg["verbose"]   = args.verbose

    model, scaler = train_model(cfg)
