import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import train_regressor as tr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _parse_kernels(value: str):
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = []
    for p in parts:
        k = int(p)
        if k <= 0:
            continue
        out.append(k)
    if not out:
        out = [3, 5, 7]
    return tuple(out)


def _rms_subframe_sequence(emg_4d: np.ndarray, n_subframes: int) -> np.ndarray:
    arr = np.asarray(emg_4d, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 1:
        raise ValueError(f"Expected EMG shape (N,C,T,1), got {arr.shape}")
    n, c, t, _ = arr.shape
    n_subframes = int(max(1, n_subframes))
    frame_len = int(max(1, t // n_subframes))
    out = np.zeros((n, c, n_subframes), dtype=np.float32)
    base = arr[..., 0]
    for i in range(n_subframes):
        start = i * frame_len
        end = t if i == (n_subframes - 1) else min(t, start + frame_len)
        chunk = base[:, :, start:end]
        if chunk.shape[2] == 0:
            chunk = base[:, :, -1:]
        out[:, :, i] = np.sqrt(np.mean(chunk**2, axis=2) + 1e-8)
    return out[..., None]


def _build_model(
    window_len,
    channel_dim,
    angle_dim,
    use_imu=False,
    imu_dim=0,
    lr=1e-3,
    arch="baseline",
    ms_kernels=(3, 5, 7),
    attn_blocks=3,
    attn_heads=3,
    attn_key_dim=64,
    attn_value_dim=128,
    use_posenc=True,
):
    import tensorflow as tf
    from tensorflow import keras

    class SinusoidalPositionalEncoding(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, x):
            length = tf.shape(x)[1]
            dim = tf.shape(x)[2]
            pos = tf.cast(tf.range(length)[:, None], tf.float32)
            i = tf.cast(tf.range(dim)[None, :], tf.float32)
            angle_rates = 1.0 / tf.pow(
                10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(dim, tf.float32)
            )
            angle_rads = pos * angle_rates
            sin_terms = tf.sin(angle_rads[:, 0::2])
            cos_terms = tf.cos(angle_rads[:, 1::2])
            pe = tf.reshape(tf.stack([sin_terms, cos_terms], axis=-1), (length, -1))[
                :, :dim
            ]
            return x + pe[None, :, :]

    def _ms_conv_module(x, filters, kernels):
        branches = []
        for k in kernels:
            b = keras.layers.Conv1D(filters, int(k), padding="same", activation="relu")(
                x
            )
            branches.append(b)
        y = keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)
        return y

    def _self_attention_block(x, heads, key_dim, value_dim):
        attn = keras.layers.MultiHeadAttention(
            num_heads=int(heads),
            key_dim=int(key_dim),
            value_dim=int(value_dim),
            dropout=0.1,
        )(x, x)
        x = keras.layers.Add()([x, attn])
        x = keras.layers.LayerNormalization()(x)
        ff = keras.layers.Dense(256, activation="relu")(x)
        ff = keras.layers.Dense(int(x.shape[-1]))(ff)
        x = keras.layers.Add()([x, ff])
        x = keras.layers.LayerNormalization()(x)
        return x

    emg_in = keras.Input(shape=(channel_dim, window_len, 1), name="emg")
    x = keras.layers.Reshape((channel_dim, window_len))(emg_in)
    x = keras.layers.Permute((2, 1), name="time_major")(x)  # (T, C)

    if str(arch).lower() == "paper_msattn":
        x = _ms_conv_module(x, filters=64, kernels=ms_kernels)
        x = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)
        x = _ms_conv_module(x, filters=64, kernels=ms_kernels)
        if bool(use_posenc):
            x = SinusoidalPositionalEncoding(name="posenc")(x)
        for _ in range(int(attn_blocks)):
            x = _self_attention_block(
                x,
                heads=int(attn_heads),
                key_dim=int(attn_key_dim),
                value_dim=int(attn_value_dim),
            )
    else:
        x = keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(
            x, x
        )
        x = keras.layers.Add()([x, attn])
        x = keras.layers.LayerNormalization()(x)

        ff = keras.layers.Dense(128, activation="relu")(x)
        ff = keras.layers.Dense(128)(ff)
        x = keras.layers.Add()([x, ff])
        x = keras.layers.LayerNormalization()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)

    inputs = [emg_in]
    if use_imu and imu_dim > 0:
        imu_in = keras.Input(shape=(imu_dim,), name="imu_feat")
        x = keras.layers.Concatenate()([x, imu_in])
        inputs.append(imu_in)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(angle_dim, activation="linear", name="angle_out")(x)

    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _mask_finite(emg, angles, imu=None):
    mask = np.isfinite(angles).all(axis=1) & np.isfinite(emg).all(axis=(1, 2, 3))
    if imu is not None:
        mask = mask & np.isfinite(imu).all(axis=(1, 2))
    emg = emg[mask]
    angles = angles[mask]
    if imu is not None:
        imu = imu[mask]
    return emg, angles, imu


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN-Attention regressor on train sessions and evaluate on held-out test session"
    )
    parser.add_argument("--train_data", nargs="+", required=True)
    parser.add_argument("--test_data", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emg_transform", default="log1p", choices=["none", "log1p"])
    parser.add_argument("--use_imu", default="auto", choices=["auto", "on", "off"])
    parser.add_argument(
        "--angle_scaler", default="minmax", choices=["none", "minmax", "standard"]
    )
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_test_windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_mode", default="raw", choices=["raw", "rms_subframes"])
    parser.add_argument("--rms_subframes", type=int, default=25)
    parser.add_argument("--paper_replication", action="store_true")
    parser.add_argument(
        "--arch", default="baseline", choices=["baseline", "paper_msattn"]
    )
    parser.add_argument("--ms_kernels", default="3,5,7")
    parser.add_argument("--attn_blocks", type=int, default=3)
    parser.add_argument("--attn_heads", type=int, default=3)
    parser.add_argument("--attn_key_dim", type=int, default=64)
    parser.add_argument("--attn_value_dim", type=int, default=128)
    parser.add_argument("--use_posenc", default="on", choices=["on", "off"])
    args = parser.parse_args()

    ms_kernels = _parse_kernels(args.ms_kernels)
    use_posenc = str(args.use_posenc).lower() == "on"

    if args.paper_replication:
        args.arch = "paper_msattn"
        args.input_mode = "rms_subframes"
        args.use_imu = "off"
        args.emg_transform = "none"
        args.angle_scaler = "minmax"
        args.ms_kernels = "3,5,7"
        args.attn_blocks = 3
        args.attn_heads = 3
        args.attn_key_dim = 64
        args.attn_value_dim = 128
        args.use_posenc = "on"
        ms_kernels = _parse_kernels(args.ms_kernels)
        use_posenc = True

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        import tensorflow as tf

        tf.random.set_seed(int(args.seed))
    except Exception:
        pass

    tr_emg, tr_angles, tr_imu, tr_meta = tr._load_datasets(
        [Path(p) for p in args.train_data]
    )
    te_emg, te_angles, te_imu, te_meta = tr._load_datasets(
        [Path(p) for p in args.test_data]
    )

    tr_emg = tr._normalize_emg(tr_emg)
    te_emg = tr._normalize_emg(te_emg)
    tr_angles = np.asarray(tr_angles, dtype=np.float32)
    te_angles = np.asarray(te_angles, dtype=np.float32)

    if args.use_imu == "off":
        tr_imu = None
        te_imu = None
    elif args.use_imu == "on" and (tr_imu is None or te_imu is None):
        print("[WARN] IMU requested but missing in train or test; disabling IMU")
        tr_imu = None
        te_imu = None

    if tr_imu is not None:
        tr_imu = tr._normalize_imu(tr_imu)
    if te_imu is not None:
        te_imu = tr._normalize_imu(te_imu)

    tr_emg, tr_angles, tr_imu = _mask_finite(tr_emg, tr_angles, tr_imu)
    te_emg, te_angles, te_imu = _mask_finite(te_emg, te_angles, te_imu)

    if args.max_train_windows > 0:
        tr_emg = tr_emg[: args.max_train_windows]
        tr_angles = tr_angles[: args.max_train_windows]
        if tr_imu is not None:
            tr_imu = tr_imu[: args.max_train_windows]
    if args.max_test_windows > 0:
        te_emg = te_emg[: args.max_test_windows]
        te_angles = te_angles[: args.max_test_windows]
        if te_imu is not None:
            te_imu = te_imu[: args.max_test_windows]

    tr_arr = tr._apply_emg_transform(tr_emg[..., 0], args.emg_transform)[..., None]
    te_arr = tr._apply_emg_transform(te_emg[..., 0], args.emg_transform)[..., None]
    if args.input_mode == "rms_subframes":
        tr_arr = _rms_subframe_sequence(tr_arr, n_subframes=int(args.rms_subframes))
        te_arr = _rms_subframe_sequence(te_arr, n_subframes=int(args.rms_subframes))

    # --- Feature Normalization (Critical for Convergence) ---
    # Flatten to (N, Features) where Features = C * Subframes
    n_tr, n_ch, n_sub, _ = tr_arr.shape
    n_te = te_arr.shape[0]

    tr_flat = tr_arr.reshape(n_tr, -1)
    te_flat = te_arr.reshape(n_te, -1)

    # Use StandardScaler (Z-score)
    x_scaler = StandardScaler()
    tr_flat = x_scaler.fit_transform(tr_flat)
    te_flat = x_scaler.transform(te_flat)

    # Reshape back to (N, C, Subframes, 1)
    tr_arr = tr_flat.reshape(n_tr, n_ch, n_sub, 1)
    te_arr = te_flat.reshape(n_te, n_ch, n_sub, 1)

    print(
        f"[INFO] Applied StandardScaler to Inputs. Mean: {x_scaler.mean_.mean():.4f}, Scale: {x_scaler.scale_.mean():.4f}"
    )
    # --------------------------------------------------------

    tr_imu_feat = tr._imu_features(tr_imu) if tr_imu is not None else None
    te_imu_feat = tr._imu_features(te_imu) if te_imu is not None else None
    use_imu = tr_imu_feat is not None and te_imu_feat is not None

    y_scaler = None
    y_train = tr_angles
    if args.angle_scaler == "minmax":
        y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    elif args.angle_scaler == "standard":
        y_scaler = StandardScaler()
    if y_scaler is not None:
        y_train = y_scaler.fit_transform(tr_angles)

    model = _build_model(
        window_len=tr_arr.shape[2],
        channel_dim=tr_arr.shape[1],
        angle_dim=tr_angles.shape[1],
        use_imu=use_imu,
        imu_dim=int(tr_imu_feat.shape[1]) if use_imu else 0,
        lr=float(args.lr),
        arch=args.arch,
        ms_kernels=ms_kernels,
        attn_blocks=int(args.attn_blocks),
        attn_heads=int(args.attn_heads),
        attn_key_dim=int(args.attn_key_dim),
        attn_value_dim=int(args.attn_value_dim),
        use_posenc=use_posenc,
    )

    x_train = [tr_arr, tr_imu_feat] if use_imu else tr_arr
    x_test = [te_arr, te_imu_feat] if use_imu else te_arr

    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    pred_scaled = model.predict(x_test, verbose=0)
    pred = (
        y_scaler.inverse_transform(pred_scaled) if y_scaler is not None else pred_scaled
    )

    mae = mean_absolute_error(te_angles, pred)
    r2 = r2_score(te_angles, pred, multioutput="variance_weighted")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cnn_attention_regressor.h5"
    model.save(str(model_path))

    if y_scaler is not None:
        with open(out_dir / "target_scaler.pkl", "wb") as f:
            pickle.dump(y_scaler, f)

    # Save Input Scaler (Added for normalization fix)
    # We used x_scaler (StandardScaler)
    # But x_scaler is only defined if we ran the normalization block.
    # Check if 'x_scaler' is in locals
    if "x_scaler" in locals():
        with open(out_dir / "input_scaler.pkl", "wb") as f:
            pickle.dump(x_scaler, f)
            print(f"[SAVED] {out_dir / 'input_scaler.pkl'}")

    meta = {
        "model_type": "cnn_attention_regressor",
        "arch": str(args.arch),
        "paper_replication": bool(args.paper_replication),
        "input_mode": str(args.input_mode),
        "rms_subframes": int(args.rms_subframes),
        "feature_mode": "cnn_attention",
        "emg_feature_mode": "cnn_attention",
        "emg_transform": args.emg_transform,
        "angle_scaler": args.angle_scaler,
        "ms_kernels": list(ms_kernels),
        "attn_blocks": int(args.attn_blocks),
        "attn_heads": int(args.attn_heads),
        "attn_key_dim": int(args.attn_key_dim),
        "attn_value_dim": int(args.attn_value_dim),
        "use_posenc": bool(use_posenc),
        "use_imu": bool(use_imu),
        "imu_features": int(tr_imu_feat.shape[1]) if use_imu else 0,
        "train_data": [str(p) for p in args.train_data],
        "test_data": [str(p) for p in args.test_data],
        "n_train": int(tr_arr.shape[0]),
        "n_test": int(te_arr.shape[0]),
        "window_len": int(tr_arr.shape[2]),
        "angle_dim": int(tr_angles.shape[1]),
        "mae_test": float(mae),
        "r2_test": float(r2),
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        "final_train_loss": float(history.history["loss"][-1]),
        "seed": int(args.seed),
        "angle_keys": tr_meta.get("angle_keys") or te_meta.get("angle_keys"),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] CNN-Attention test MAE={mae:.4f} R2={r2:.4f}")
    print(f"[SAVED] {model_path}")


if __name__ == "__main__":
    main()
