import os
import logging
import argparse
import numpy as np

from pyoephys.io import load_oebin_file, load_config_file, labels_from_events
from pyoephys.processing import EMGPreprocessor


def build_training_dataset(
    root_dir,
    save_path=None,
    label=None,
    window_ms=200,
    step_ms=50,
    selected_channels=None,
    overwrite=False,
    verbose=False,
):
    # — Save path
    if save_path is None:
        name = f"{label}_training_dataset.npz" if label else "training_dataset.npz"
        save_path = os.path.join(root_dir, name)

    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists at {save_path}. Use --overwrite to regenerate.")
        return np.load(save_path, allow_pickle=True)

    # — Load EMG
    data   = load_oebin_file(os.path.join(root_dir, "raw"), verbose=verbose)
    emg_fs = float(data["sample_rate"])
    emg    = data["amplifier_data"]          # shape (C, N)
    emg_t  = data["t_amplifier"]             # seconds, shape (N,)

    # Channels
    if selected_channels is not None:
        logging.info(f"Using selected channels: {selected_channels}")
        emg = emg[selected_channels, :]

    # — Preprocess: BP 20–498 + 60 Hz notch, rectify, envelope at 5 Hz
    logging.info("Preprocessing EMG signals...")
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0, verbose=verbose)
    emg_pp = pre.preprocess(emg)

    # — Features
    logging.info(f"Extracting features: window={window_ms}ms, step={step_ms}ms")
    X = pre.extract_emg_features(
        emg_pp,
        window_ms=window_ms,
        step_ms=step_ms,
        progress=True,
        tqdm_kwargs={"desc": "Building dataset", "leave": False},
    )
    logging.info(f"[INFO] Extracted feature matrix X with shape {X.shape}")

    # — Window start indices (LEFT EDGE; absolute sample idx)
    start_index  = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index
    print(f"Number of window starts: {len(window_starts)}, "
          f"from {window_starts[0]} to {window_starts[-1]}")

    # — Labels from events (robust to timestamp text; sorts by Sample Index)
    ev_path = os.path.join(root_dir, "events", "emg.event")
    logging.info(f"Parsing event file: {ev_path}")
    y = labels_from_events(ev_path, window_starts)

    # — Filter Unknown/Start
    mask = ~np.isin(y, ["Unknown", "Start"])
    removed = int((~mask).sum())
    X, y = X[mask], y[mask]
    logging.info(f"Filtered out {removed} windows; remaining = {len(y)}")

    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch windows ({X.shape[0]}) vs labels ({len(y)})")

    # — Save
    logging.info(f"Saving feature dataset to {save_path}...")
    np.savez(
        save_path,
        X=X,
        y=y,
        emg_fs=emg_fs,
        window_ms=window_ms,
        step_ms=step_ms,
        selected_channels=np.array(selected_channels) if selected_channels is not None else None,
        channel_names=data.get("channel_names", None),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build EMG training dataset from OEBin + events.")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",   type=str, default="")
    p.add_argument("--label",      type=str, default="")
    p.add_argument("--channels",   nargs="+", type=int, default=None)
    p.add_argument("--window_ms",  type=int, default=200)
    p.add_argument("--step_ms",    type=int, default=50)
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--save_path",  type=str, default=None)
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)

    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "label": args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", 200),
        "step_ms": args.step_ms or cfg.get("step_ms", 50),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "save_path": args.save_path or cfg.get("save_path", None),
        "selected_channels": args.channels or cfg.get("selected_channels", None),
        "verbose": args.verbose or cfg.get("verbose", False),
    })

    lvl = logging.DEBUG if cfg["verbose"] else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    build_training_dataset(**cfg)
