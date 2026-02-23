#!/usr/bin/env python3
"""
predict.py — Offline EMG Gesture Prediction
============================================
Loads a trained model (from 2_train_model.py) and evaluates it against a
recorded EMG file, printing per-class accuracy.

Default behaviour (zero arguments)
------------------------------------
  Data  : ./data/gestures/
  Labels: emg.txt alongside the recording (auto-detected)
  Model : ./data/model/

Usage
-----
  # Evaluate using data in data/gestures/:
  python predict.py

  # Custom file:
  python predict.py --data_path data/my_session/ --model_dir data/model

  # Suppress accuracy output (just get predictions):
  python predict.py --no_eval

Note
----
For real-time ZMQ prediction from a live Open Ephys GUI stream, use
3_predict_realtime.py instead.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pyoephys.io import load_simple_config, load_open_ephys_session, process_recording
from pyoephys.io._file_utils import find_event_for_file
from pyoephys.ml import ModelManager, EMGClassifier

# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_file(
    data_path: str,
    model_dir: str,
    labels_path: str | None = None,
    label: str = "",
    evaluate: bool = True,
    save_predictions: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Run offline gesture prediction on a single EMG file.

    Parameters
    ----------
    data_path       : Path to EMG data (.csv / .npz / .oebin).
    model_dir       : Directory that contains ``metadata.json`` and model
                      artefacts (the ``model/`` sub-folder produced by
                      2_train_model.py, e.g. ``./data/model``).
    labels_path     : Path to labels CSV.  Auto-detected if None.
    label           : Model label used during training (empty = default).
    evaluate        : Print accuracy + classification report when True and
                      ground-truth labels are available.
    save_predictions: Save predictions alongside the input file.
    verbose         : Extra logging.

    Returns
    -------
    y_pred : (N,) array of predicted gesture labels.
    """
    # ── Load metadata ──────────────────────────────────────────────────────
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        # 2_train_model.py saves into a model/ subdirectory; check there too
        meta_path = os.path.join(model_dir, "model", "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"metadata.json not found in {model_dir!r}.\n"
            "Run  python 2_train_model.py  first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    window_ms     = int(meta.get("window_ms", 200))
    step_ms       = int(meta.get("step_ms",   50))
    # channels=[] means "all" — normalise to None so process_recording
    # does not slice emg with an empty index list.
    channels_raw  = meta.get("selected_channels")
    channels      = channels_raw if channels_raw else None
    # Reproduce the same label filter used during dataset building so the
    # evaluation is apples-to-apples (e.g. "Start" excluded from both).
    ignore_labels = meta.get("ignore_labels") or []

    # ── Load data ──────────────────────────────────────────────────────────
    data = load_open_ephys_session(data_path)
    fs = data["sample_rate"]
    if verbose:
        print(f"Loaded: {data['amplifier_data'].shape[1]} samples @ {fs:.0f} Hz")

    # ── Events / labels ───────────────────────────────────────────────────
    data_dir = str(Path(data_path).parent if Path(data_path).is_file() else Path(data_path))
    if labels_path and Path(labels_path).is_file():
        events_file = labels_path
    else:
        # Auto-discover labels.csv / events.csv next to the data file
        events_file = find_event_for_file(data_dir, str(data_path))
    if verbose and events_file:
        print(f"Labels: {events_file}")

    # ── Feature extraction (identical pipeline to training) ────────────────
    X, y_true, _ = process_recording(
        data=data,
        file_path=data_path,
        root_dir=data_dir,
        events_file=events_file,
        window_ms=window_ms,
        step_ms=step_ms,
        channels=channels,
        ignore_labels=ignore_labels,
        ignore_case=True,
        keep_trial=False,
    )

    if len(X) == 0:
        print("[WARN] No windows extracted — check data and labels paths.")
        return np.array([])

    # ── Predict ───────────────────────────────────────────────────────────
    # meta_path is at <root>/model/metadata.json; ModelManager wants root_dir=<root>
    manager_root = str(Path(meta_path).parent.parent)
    manager = ModelManager(root_dir=manager_root, label=label, model_cls=EMGClassifier)
    manager.load_model()
    y_pred = manager.predict(X)

    print(f"\nPredictions on {len(y_pred)} windows:")

    # ── Evaluate ──────────────────────────────────────────────────────────
    if evaluate and events_file is not None:
        unique_true = set(y_true)
        if len(unique_true) > 1 or (len(unique_true) == 1 and "unknown" not in unique_true):
            try:
                from sklearn.metrics import accuracy_score, classification_report
                acc = accuracy_score(y_true, y_pred)
                print(f"Accuracy: {acc:.3f}\n")
                print(classification_report(y_true, y_pred))
            except ImportError:
                # Fallback: simple per-class accuracy
                classes = sorted(unique_true)
                print(f"{'Class':<18} {'Correct':>7} {'Total':>7} {'Acc':>6}")
                print("-" * 44)
                for cls in classes:
                    mask = y_true == cls
                    correct = int(np.sum(y_pred[mask] == cls))
                    total   = int(np.sum(mask))
                    print(f"{cls:<18} {correct:>7} {total:>7} {correct/total:>6.3f}")
        else:
            print("(No ground-truth labels — showing raw predictions)")
            from collections import Counter
            for cls, cnt in sorted(Counter(y_pred).items()):
                print(f"  {cls}: {cnt} windows")

    # ── Save ──────────────────────────────────────────────────────────────
    if save_predictions:
        out = str(Path(data_path).with_suffix(".pred.npz"))
        np.savez(out, y_pred=y_pred, y_true=y_true)
        print(f"\nPredictions saved → {out}")

    return y_pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).parent
_DEFAULT_DATA  = _HERE / "data" / "gestures"
_DEFAULT_MODEL = _HERE / "data" / "gesture_model"
_CONFIG_FILE   = _HERE / ".gesture_config"


def main():
    p = argparse.ArgumentParser(
        description="Offline EMG gesture prediction from a recorded file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data_path", default=None,
        help=f"Path to EMG data (.csv / .npz / .oebin).  Default: {_DEFAULT_DATA}",
    )
    p.add_argument(
        "--model_dir", default=None,
        help=f"Directory containing model artefacts.  Default: {_DEFAULT_MODEL / 'model'}",
    )
    p.add_argument("--labels_path",       default=None, help="Labels CSV (auto-detected if omitted).")
    p.add_argument("--label",             default="",   help="Model label used in 2_train_model.py.")
    p.add_argument("--no_eval",           action="store_true", help="Skip accuracy evaluation.")
    p.add_argument("--save_predictions",  action="store_true", help="Save .pred.npz alongside input.")
    p.add_argument("--config_file",       default=None)
    p.add_argument("--verbose",           action="store_true")
    args = p.parse_args()

    cfg = {}
    config_path = args.config_file or _CONFIG_FILE
    if Path(config_path).is_file():
        cfg = load_simple_config(str(config_path))

    data_path = args.data_path or cfg.get("data_path") or str(_DEFAULT_DATA)
    model_dir = args.model_dir or cfg.get("model_dir") or str(_DEFAULT_MODEL / "model")
    label     = args.label     or cfg.get("label", "")

    if not Path(data_path).exists():
        print(f"[ERROR] Data not found: {data_path}")
        print("        Pass --data_path to point at your recording.")
        return 1

    if not Path(model_dir).exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print("        Run  python 2_train_model.py  first.")
        return 1

    predict_file(
        data_path=data_path,
        model_dir=model_dir,
        labels_path=args.labels_path or cfg.get("labels_path"),
        label=label,
        evaluate=not args.no_eval,
        save_predictions=args.save_predictions,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


