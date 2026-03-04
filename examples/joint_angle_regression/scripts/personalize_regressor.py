import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import train_regressor as tr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _fit_diagonal_identity_calibrator(x_train, y_train, alpha):
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    out_dim = int(y_train.shape[1])

    coef = np.zeros((out_dim, out_dim), dtype=np.float32)
    intercept = np.zeros((out_dim,), dtype=np.float32)

    # Fit per-joint delta model: y = x + (c*x + b) = (1+c)*x + b
    for j in range(out_dim):
        xj = x_train[:, j].reshape(-1, 1)
        delta = (y_train[:, j] - x_train[:, j]).reshape(-1)
        reg = Ridge(alpha=float(alpha), fit_intercept=True)
        reg.fit(xj, delta)
        coef[j, j] = 1.0 + float(reg.coef_.reshape(-1)[0])
        intercept[j] = float(reg.intercept_)

    return coef, intercept


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _load_base_bundle(model_dir: Path):
    metrics_path = model_dir / "metrics.json"
    reg_path = model_dir / "mlp_regressor.pkl"
    scaler_path = model_dir / "scaler.pkl"
    cnn_path = model_dir / "cnn_attention_regressor.h5"

    model_kind = None
    regressor = None
    scaler = None
    cnn_model = None

    if reg_path.exists() and scaler_path.exists():
        model_kind = "mlp"
        with open(reg_path, "rb") as f:
            regressor = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    elif cnn_path.exists():
        model_kind = "cnn"
        from tensorflow import keras

        cnn_model = keras.models.load_model(str(cnn_path), compile=False)
    else:
        raise FileNotFoundError(
            "Model dir must contain either (mlp_regressor.pkl + scaler.pkl) or cnn_attention_regressor.h5"
        )

    target_scaler = None
    ts_path = model_dir / "target_scaler.pkl"
    if ts_path.exists():
        with open(ts_path, "rb") as f:
            target_scaler = pickle.load(f)

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    return model_kind, regressor, scaler, cnn_model, target_scaler, metrics


def _build_base_predictions_cnn(npz_paths, cnn_model, target_scaler, metrics):
    emg, angles, imu, _ = tr._load_datasets([Path(p) for p in npz_paths])

    emg = tr._normalize_emg(emg)
    angles = np.asarray(angles, dtype=np.float32)
    if imu is not None:
        imu = tr._normalize_imu(imu)

    mask = np.isfinite(angles).all(axis=1) & np.isfinite(emg).all(axis=(1, 2, 3))
    if imu is not None:
        mask = mask & np.isfinite(imu).all(axis=(1, 2))

    emg = emg[mask]
    angles = angles[mask]
    if imu is not None:
        imu = imu[mask]

    emg_transform = metrics.get("emg_transform", "none")
    emg_arr = tr._apply_emg_transform(emg[..., 0], emg_transform)[..., None]

    use_imu = bool(metrics.get("use_imu", False) or metrics.get("imu_included", False))
    if use_imu and imu is not None:
        imu_feat = tr._imu_features(imu)
        pred_scaled = cnn_model.predict([emg_arr, imu_feat], verbose=0)
    else:
        pred_scaled = cnn_model.predict(emg_arr, verbose=0)

    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred_scaled)
    else:
        pred = pred_scaled
    return np.asarray(pred, dtype=np.float32), np.asarray(angles, dtype=np.float32)


def _build_base_predictions(
    npz_paths,
    regressor,
    scaler,
    target_scaler,
    metrics,
    feature_extractor_override=None,
):
    emg, angles, imu, _ = tr._load_datasets([Path(p) for p in npz_paths])

    mask = np.isfinite(angles).all(axis=1)
    emg_mask = np.isfinite(emg).all(axis=(1, 2, 3))
    mask = mask & emg_mask
    if imu is not None:
        imu_mask = np.isfinite(imu).all(axis=(1, 2))
        mask = mask & imu_mask

    if not np.all(mask):
        emg = emg[mask]
        angles = angles[mask]
        if imu is not None:
            imu = imu[mask]

    feature_mode = metrics.get(
        "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
    )
    emg_transform = metrics.get("emg_transform", "none")
    emd_max_imfs = int(metrics.get("emd_max_imfs", 3))

    extractor = None
    if feature_mode == "extractor":
        fx = feature_extractor_override or metrics.get("feature_extractor")
        extractor = tr.load_feature_extractor(fx)
        if extractor is None:
            raise RuntimeError(
                "Base model expects extractor features, but extractor could not be loaded"
            )

    feats = tr.extract_features(
        emg,
        extractor,
        feature_mode=feature_mode,
        emg_transform=emg_transform,
        emd_max_imfs=emd_max_imfs,
    )

    use_imu = bool(metrics.get("imu_included", False))
    if use_imu and imu is not None:
        feats = np.concatenate([feats, tr._imu_features(imu)], axis=1)

    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and int(feats.shape[1]) != int(n_in):
        raise ValueError(
            f"Feature size mismatch: built {feats.shape[1]} but scaler expects {n_in}"
        )

    feats_scaled = scaler.transform(feats)
    pred_scaled = regressor.predict(feats_scaled)
    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred_scaled)
    else:
        pred = pred_scaled
    return np.asarray(pred, dtype=np.float32), np.asarray(angles, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a fast personalization adaptor on top of an existing regressor"
    )
    parser.add_argument(
        "--base_model_dir",
        required=True,
        help="Directory containing base mlp_regressor/scaler/metrics",
    )
    parser.add_argument(
        "--calib_data",
        nargs="+",
        required=True,
        help="One or more calibration .npz datasets",
    )
    parser.add_argument(
        "--out_path",
        default="",
        help="Output personalization .pkl path (default: <base_model_dir>/personalization.pkl)",
    )
    parser.add_argument(
        "--feature_extractor",
        default="",
        help="Optional override path for extractor model when base model uses extractor features",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength"
    )
    parser.add_argument(
        "--adaptor_mode",
        default="full_ridge",
        choices=["full_ridge", "diagonal_identity", "bias_only"],
        help="Personalization adaptor type",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split ratio (0 disables split)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.base_model_dir)
    out_path = (
        Path(args.out_path) if args.out_path else model_dir / "personalization.pkl"
    )

    model_kind, regressor, scaler, cnn_model, target_scaler, metrics = (
        _load_base_bundle(model_dir)
    )
    if model_kind == "mlp":
        pred, y_true = _build_base_predictions(
            npz_paths=args.calib_data,
            regressor=regressor,
            scaler=scaler,
            target_scaler=target_scaler,
            metrics=metrics,
            feature_extractor_override=(args.feature_extractor or None),
        )
    else:
        pred, y_true = _build_base_predictions_cnn(
            npz_paths=args.calib_data,
            cnn_model=cnn_model,
            target_scaler=target_scaler,
            metrics=metrics,
        )

    if pred.shape[0] < 8:
        raise ValueError("Need at least 8 calibration windows for personalization")

    if args.val_size > 0.0 and pred.shape[0] >= 16:
        x_train, x_val, y_train, y_val = train_test_split(
            pred, y_true, test_size=float(args.val_size), random_state=args.random_state
        )
    else:
        x_train, y_train = pred, y_true
        x_val, y_val = pred, y_true

    if args.adaptor_mode == "full_ridge":
        adaptor = Ridge(alpha=float(args.alpha), fit_intercept=True)
        adaptor.fit(x_train, y_train)
        coef = np.asarray(adaptor.coef_, dtype=np.float32)
        intercept = np.asarray(adaptor.intercept_, dtype=np.float32)
    elif args.adaptor_mode == "diagonal_identity":
        coef, intercept = _fit_diagonal_identity_calibrator(
            x_train=x_train,
            y_train=y_train,
            alpha=float(args.alpha),
        )
    else:
        # bias_only: y = x + b
        out_dim = int(y_train.shape[1])
        coef = np.eye(out_dim, dtype=np.float32)
        intercept = np.mean(y_train - x_train, axis=0).astype(np.float32)

    base_mae = mean_absolute_error(y_val, x_val)
    base_r2 = r2_score(y_val, x_val, multioutput="variance_weighted")

    y_val_adapt = x_val @ coef.T + intercept.reshape(1, -1)
    adapt_mae = mean_absolute_error(y_val, y_val_adapt)
    adapt_r2 = r2_score(y_val, y_val_adapt, multioutput="variance_weighted")

    payload = {
        "type": "ridge_output_calibration",
        "base_model_dir": str(model_dir),
        "base_model_kind": model_kind,
        "adaptor_mode": str(args.adaptor_mode),
        "coef": coef,
        "intercept": intercept,
        "alpha": float(args.alpha),
        "fit_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "base_mae": float(base_mae),
        "base_r2": float(base_r2),
        "adapted_mae": float(adapt_mae),
        "adapted_r2": float(adapt_r2),
        "improvement_mae": float(base_mae - adapt_mae),
        "improvement_r2": float(adapt_r2 - base_r2),
        "angle_keys": metrics.get("angle_keys"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    summary_path = out_path.with_suffix(".json")
    summary = {
        "type": payload["type"],
        "base_model_dir": payload["base_model_dir"],
        "base_model_kind": payload["base_model_kind"],
        "adaptor_mode": payload["adaptor_mode"],
        "alpha": payload["alpha"],
        "fit_samples": payload["fit_samples"],
        "val_samples": payload["val_samples"],
        "base_mae": payload["base_mae"],
        "base_r2": payload["base_r2"],
        "adapted_mae": payload["adapted_mae"],
        "adapted_r2": payload["adapted_r2"],
        "improvement_mae": payload["improvement_mae"],
        "improvement_r2": payload["improvement_r2"],
        "angle_keys": payload.get("angle_keys"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[DONE] Personalization saved: "
        f"{out_path} | MAE {base_mae:.4f}->{adapt_mae:.4f} | R2 {base_r2:.4f}->{adapt_r2:.4f}"
    )


if __name__ == "__main__":
    main()
