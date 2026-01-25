import os
import argparse
import numpy as np

from pyoephys.io import load_config_file
from pyoephys.ml import ModelManager, EMGClassifier


def train_model(cfg):
    root_dir = cfg["root_dir"]
    label    = cfg.get("label", "")
    kfold    = cfg.get("kfold", False)
    overwrite = cfg.get("overwrite", False)

    data_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    # Load features and labels
    print(f"Loading dataset from {data_path}")
    data = np.load(data_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"Data shape - X: {X.shape}, y: {y.shape}")

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
    elif not os.path.isfile(manager.model_path) or overwrite:
        print("Training new model...")
        manager.train(X, y)
        print("Training complete. Validation metrics:")
        print(manager.eval_metrics)
    else:
        print("Model already exists, loading existing model + scaler/encoder.")
        manager.load_model()

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
