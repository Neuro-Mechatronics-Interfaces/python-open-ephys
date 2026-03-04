import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    if "emg" not in data or "angles" not in data:
        raise KeyError(f"{path} missing required keys: 'emg' and 'angles'")
    return data


def _normalize_emg(emg):
    # Expect (N, 8, window_len, 1). Accept common variants.
    arr = np.asarray(emg)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Could be (N, 8, window_len) or (N, window_len, 8)
        if arr.shape[1] == 8:
            arr = arr[:, :, :, None]
            return arr.astype(np.float32)
        if arr.shape[2] == 8:
            arr = arr.transpose(0, 2, 1)[:, :, :, None]
            return arr.astype(np.float32)
    raise ValueError(f"Unsupported EMG shape: {arr.shape}")


def _normalize_angles(angles):
    arr = np.asarray(angles)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported angles shape: {arr.shape}")


def _load_datasets(paths):
    emg_list = []
    angles_list = []
    meta = {"datasets": []}
    ref_shape = None
    ref_angle_dim = None

    for p in paths:
        data = _load_npz(p)
        emg = _normalize_emg(data["emg"])
        angles = _normalize_angles(data["angles"])

        if ref_shape is None:
            ref_shape = emg.shape[1:3]
        if emg.shape[1:3] != ref_shape:
            raise ValueError(
                f"{p} EMG shape mismatch. Expected (*, {ref_shape[0]}, {ref_shape[1]}, 1) got {emg.shape}"
            )

        if ref_angle_dim is None:
            ref_angle_dim = angles.shape[1]
        if angles.shape[1] != ref_angle_dim:
            raise ValueError(
                f"{p} angle dim mismatch. Expected {ref_angle_dim}, got {angles.shape[1]}"
            )

        if emg.shape[0] != angles.shape[0]:
            raise ValueError(
                f"{p} sample count mismatch: emg {emg.shape[0]} vs angles {angles.shape[0]}"
            )

        emg_list.append(emg)
        angles_list.append(angles)
        meta["datasets"].append(
            {
                "path": str(p),
                "samples": int(emg.shape[0]),
                "emg_shape": list(emg.shape),
                "angle_shape": list(angles.shape),
            }
        )

    emg_all = np.concatenate(emg_list, axis=0)
    angles_all = np.concatenate(angles_list, axis=0)
    meta["total_samples"] = int(emg_all.shape[0])
    meta["emg_shape"] = list(emg_all.shape)
    meta["angle_shape"] = list(angles_all.shape)
    return emg_all, angles_all, meta


def _load_index(index_path):
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    base = path.parent
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            files = data.get("files") or data.get("datasets") or data.get("paths")
            side = data.get("side")
        else:
            files = data
            side = None
        if not isinstance(files, list):
            raise ValueError("JSON index must be a list or contain a 'files' list.")
        return [Path(p) if Path(p).is_absolute() else base / p for p in files], side
    if path.suffix.lower() in (".csv", ".tsv"):
        files = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(
                f, delimiter="," if path.suffix.lower() == ".csv" else "\t"
            )
            for row in reader:
                if not row:
                    continue
                p = Path(row[0])
                files.append(p if p.is_absolute() else base / p)
        return files, None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    files = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        files.append(p if p.is_absolute() else base / p)
    return files, None


def _build_model(window_len, angle_dim, feature_dim, lr):
    from tensorflow import keras

    inputs = keras.Input(shape=(8, window_len, 1), name="emg_input")
    x = keras.layers.Reshape((8, window_len), name="squeeze_channel")(inputs)
    x = keras.layers.Permute((2, 1), name="time_major")(x)  # (window_len, 8)
    x = keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(feature_dim, activation="relu", name="dense_8")(x)
    outputs = keras.layers.Dense(angle_dim, activation="linear", name="angle_out")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a feature extractor (DL backbone) for joint angle regression.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data", nargs="+", help="One or more .npz datasets")
    parser.add_argument("--data_index", help="Index file (.json/.csv/.tsv/.txt) listing dataset paths")
    parser.add_argument("--make_index", help="Write a dataset index JSON from --index_root/--index_glob and exit")
    parser.add_argument("--index_root", help="Root folder for --make_index")
    parser.add_argument("--index_glob", default="**/*.npz", help="Glob pattern for --make_index (default: **/*.npz)")
    parser.add_argument("--index_side", choices=("left", "right"), default=None, help="Side label for --make_index")
    parser.add_argument("--index_absolute", action="store_true", help="Write absolute paths in --make_index")
    parser.add_argument("--out_model", required=True, help="Output .h5 model path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension (dense_8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--side", choices=("left", "right"), default=None, help="Optional side label for metadata")
    args = parser.parse_args()

    if args.make_index:
        if not args.index_root:
            raise SystemExit("--make_index requires --index_root")
        root = Path(args.index_root)
        if args.index_absolute:
            files = sorted(str(p.resolve()) for p in root.glob(args.index_glob) if p.is_file())
        else:
            files = sorted(p.relative_to(root).as_posix() for p in root.glob(args.index_glob) if p.is_file())
        payload = {"side": args.index_side, "files": files}
        out_path = Path(args.make_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote {out_path} with {len(files)} files")
        return

    if not args.data and not args.data_index:
        raise SystemExit("Provide --data or --data_index")

    paths = []
    side_from_index = None
    if args.data_index:
        paths, side_from_index = _load_index(args.data_index)
    if args.data:
        paths.extend([Path(p) for p in args.data])
    if not paths:
        raise SystemExit("No dataset paths found.")

    emg, angles, meta = _load_datasets(paths)
    if args.side is None and side_from_index:
        args.side = side_from_index
    window_len = emg.shape[2]
    angle_dim = angles.shape[1]

    model = _build_model(window_len, angle_dim, args.feature_dim, args.lr)

    callbacks = []
    try:
        from tensorflow import keras

        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            )
        )
    except Exception:
        pass

    history = model.fit(
        emg,
        angles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    metrics = {
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "final_mae": float(history.history["mae"][-1]) if "mae" in history.history else None,
        "feature_dim": args.feature_dim,
        "window_len": int(window_len),
        "angle_dim": int(angle_dim),
        "side": args.side,
    }
    meta.update(metrics)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Saved model: {out_path}")
    print(f"[META] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
