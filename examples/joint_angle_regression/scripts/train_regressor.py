import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_EMD_CLASS = None
_EMD_IMPORT_ERROR = None


class KerasFeatureExtractorLocal:
    """Minimal .h5 feature extractor wrapper.

    Uses the penultimate layer by default (or `dense_8` if present) so the
    extracted representation can be used as regressor input features.
    """

    def __init__(self, model_path: str):
        from tensorflow import keras

        base_model = keras.models.load_model(model_path, compile=False)
        layer_names = [layer.name for layer in base_model.layers]
        if "dense_8" in layer_names:
            feat_layer = base_model.get_layer("dense_8")
        elif len(base_model.layers) >= 2:
            feat_layer = base_model.layers[-2]
        else:
            feat_layer = base_model.layers[-1]
        self.model = keras.Model(inputs=base_model.input, outputs=feat_layer.output)

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)


def load_feature_extractor(path):
    if not path or str(path).lower() in ("none", "raw", "flat"):
        return None
    if not os.path.exists(path):
        print(
            f"[WARN] Feature extractor not found: {path}. Falling back to raw features."
        )
        return None
    try:
        if path.endswith(".h5"):
            return KerasFeatureExtractorLocal(path)
        print(
            f"[WARN] Unsupported feature extractor format: {path}. Falling back to raw features."
        )
        return None
    except Exception as exc:
        print(
            f"[WARN] Failed to load feature extractor: {exc}. Falling back to raw features."
        )
        return None


def _normalize_emg(emg):
    arr = np.asarray(emg, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr
    if arr.ndim == 3:
        # (N, C, T) or (N, T, C)
        # Try to infer based on shape
        # Heuristic: T is usually larger than C.
        s1, s2 = arr.shape[1], arr.shape[2]
        if s1 < s2:  # s1 is C, s2 is T
            return arr[:, :, :, None]
        else:  # s2 is C, s1 is T
            return arr.transpose(0, 2, 1)[:, :, :, None]
    raise ValueError(f"Unexpected EMG shape: {arr.shape}")


def _normalize_imu(imu):
    arr = np.asarray(imu, dtype=np.float32)
    if arr.ndim == 3:
        # (N, C, T) or (N, T, C)
        if arr.shape[1] in (6, 9):
            return arr
        if arr.shape[2] in (6, 9):
            return arr.transpose(0, 2, 1)
    raise ValueError(f"Unexpected IMU shape: {arr.shape}")


def _imu_features(imu):
    imu = _normalize_imu(imu)
    mean = imu.mean(axis=2)
    std = imu.std(axis=2)
    return np.concatenate([mean, std], axis=1)


def _apply_emg_transform(emg, mode):
    if mode == "none":
        return emg
    if mode == "log1p":
        return np.log1p(np.maximum(emg, 0.0))
    raise ValueError(f"Unknown EMG transform: {mode}")


def _bandpower_features(emg):
    # emg: (N, C, T)
    if emg.size == 0:
        raise ValueError("No samples left after filtering; cannot train.")
    mean = emg.mean(axis=2)
    std = emg.std(axis=2)
    minv = emg.min(axis=2)
    maxv = emg.max(axis=2)
    if emg.shape[2] > 1:
        diff = np.diff(emg, axis=2)
        diff_energy = np.mean(diff**2, axis=2)
    else:
        diff_energy = np.zeros_like(mean)
    t = np.linspace(-0.5, 0.5, emg.shape[2], dtype=np.float32)
    denom = float(np.sum(t**2)) if emg.shape[2] > 1 else 1.0
    slope = np.tensordot(emg, t, axes=([2], [0])) / denom
    return np.concatenate([mean, std, minv, maxv, diff_energy, slope], axis=1)


def _get_emd_class():
    global _EMD_CLASS, _EMD_IMPORT_ERROR
    if _EMD_CLASS is not None:
        return _EMD_CLASS
    if _EMD_IMPORT_ERROR is not None:
        raise ImportError(_EMD_IMPORT_ERROR)
    try:
        from PyEMD import EMD

        _EMD_CLASS = EMD
        return _EMD_CLASS
    except Exception as exc:
        _EMD_IMPORT_ERROR = (
            "PyEMD is required for feature_mode='emd_stats'. "
            "Install with: pip install EMD-signal"
        )
        raise ImportError(_EMD_IMPORT_ERROR) from exc


def _spectral_centroid_norm(signal_1d):
    spec = np.abs(np.fft.rfft(signal_1d))
    if spec.size <= 1:
        return 0.0
    denom = float(np.sum(spec))
    if denom <= 1e-12:
        return 0.0
    bins = np.arange(spec.size, dtype=np.float32)
    centroid = float(np.sum(bins * spec) / denom)
    return centroid / float(max(1, spec.size - 1))


def _emd_stats_features(emg, max_imfs=3):
    # emg: (N, C, T)
    if emg.size == 0:
        raise ValueError("No samples left after filtering; cannot train.")
    if max_imfs <= 0:
        raise ValueError("emd_max_imfs must be >= 1")

    EMD = _get_emd_class()
    n, c, t = emg.shape
    per_imf_feats = 6  # rms, std, abs_mean, zcr, energy_ratio, spectral_centroid
    out = np.zeros((n, c * (max_imfs * per_imf_feats + 1)), dtype=np.float32)

    for i in range(n):
        row = []
        for ch in range(c):
            sig = np.asarray(emg[i, ch], dtype=np.float32)
            total_energy = float(np.sum(sig**2) + 1e-8)

            emd = EMD()
            imfs = emd.emd(sig)
            if imfs is None or np.size(imfs) == 0:
                imfs = np.zeros((0, t), dtype=np.float32)
            else:
                imfs = np.asarray(imfs, dtype=np.float32)
                if imfs.ndim == 1:
                    imfs = imfs[None, :]

            use_count = int(min(max_imfs, imfs.shape[0]))
            for k in range(max_imfs):
                if k < use_count:
                    imf = imfs[k]
                    rms = float(np.sqrt(np.mean(imf**2)))
                    std = float(np.std(imf))
                    abs_mean = float(np.mean(np.abs(imf)))
                    if imf.size > 1:
                        zcr = float(np.mean((imf[:-1] * imf[1:]) < 0.0))
                    else:
                        zcr = 0.0
                    energy = float(np.sum(imf**2))
                    energy_ratio = energy / total_energy
                    centroid = _spectral_centroid_norm(imf)
                    row.extend([rms, std, abs_mean, zcr, energy_ratio, centroid])
                else:
                    row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            if use_count > 0:
                residue = sig - np.sum(imfs[:use_count], axis=0)
            else:
                residue = sig
            residue_ratio = float(np.sum(residue**2) / total_energy)
            row.append(residue_ratio)

        out[i] = np.asarray(row, dtype=np.float32)
    return out


def extract_features(emg, extractor, feature_mode, emg_transform, emd_max_imfs=3):
    emg = _normalize_emg(emg)
    if emg.shape[0] == 0:
        raise ValueError("No samples left after filtering; cannot train.")
    arr = emg[..., 0]
    arr = _apply_emg_transform(arr, emg_transform)
    if extractor is not None:
        return extractor.predict(arr[..., None], verbose=0)
    if feature_mode == "bandpower_stats":
        return _bandpower_features(arr)
    if feature_mode == "emd_stats":
        return _emd_stats_features(arr, max_imfs=int(emd_max_imfs))
    if feature_mode == "raw_flat":
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Unknown EMG feature mode: {feature_mode}")


def load_personalization(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        adaptor = pickle.load(f)
    return adaptor if isinstance(adaptor, dict) else None


def apply_personalization(pred, adaptor):
    if adaptor is None:
        return pred
    y = np.asarray(pred, dtype=np.float32)
    coef = np.asarray(adaptor.get("coef"), dtype=np.float32)
    intercept = np.asarray(adaptor.get("intercept"), dtype=np.float32)
    if coef.ndim != 2:
        raise ValueError("Invalid personalization adaptor: coef must be 2D")
    if y.shape[1] != coef.shape[1]:
        raise ValueError(
            f"Personalization dim mismatch: pred has {y.shape[1]}, adaptor expects {coef.shape[1]}"
        )
    return y @ coef.T + intercept.reshape(1, -1)


def _safe_scalar(data, key):
    if key not in data:
        return None
    val = data[key]
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        return float(val.reshape(-1)[0])
    try:
        return float(val)
    except Exception:
        return None


def _safe_get(data, key):
    try:
        if hasattr(data, "files") and key not in data.files:
            return None
        return data[key]
    except Exception:
        return None


def _load_datasets(paths):
    emg_list = []
    angles_list = []
    imu_list = []
    angle_keys_ref = None
    target_spec_ref = None
    window_len_ref = None
    angle_dim_ref = None
    imu_expected = None
    used_imu = True
    meta = {
        "datasets": [],
        "wrist_orientations": [],
    }

    for p in paths:
        data = np.load(p, allow_pickle=True)
        emg = _normalize_emg(data["emg"])
        angles = np.asarray(data["angles"], dtype=np.float32)
        if window_len_ref is None:
            window_len_ref = emg.shape[2]
        if emg.shape[2] != window_len_ref:
            raise ValueError(
                f"{p} window_len mismatch: {emg.shape[2]} vs {window_len_ref}"
            )
        if angle_dim_ref is None:
            angle_dim_ref = angles.shape[1]
        if angles.shape[1] != angle_dim_ref:
            raise ValueError(
                f"{p} angle dim mismatch: {angles.shape[1]} vs {angle_dim_ref}"
            )

        angle_keys = _safe_get(data, "angle_keys")
        if angle_keys is not None:
            angle_keys = (
                angle_keys.tolist()
                if isinstance(angle_keys, np.ndarray)
                else angle_keys
            )
            if angle_keys_ref is None:
                angle_keys_ref = angle_keys
            elif angle_keys != angle_keys_ref:
                raise ValueError(f"{p} angle_keys mismatch")

        target_spec = _safe_get(data, "target_spec")
        if isinstance(target_spec, np.ndarray):
            try:
                target_spec = target_spec.item()
            except Exception:
                target_spec = None
        if target_spec_ref is None:
            target_spec_ref = target_spec
        elif target_spec is not None and target_spec != target_spec_ref:
            raise ValueError(
                f"{p} target_spec mismatch: {target_spec} vs {target_spec_ref}"
            )

        imu = _safe_get(data, "imu")
        if imu is None:
            used_imu = False
        else:
            imu = _normalize_imu(imu)
            if imu_expected is None:
                imu_expected = imu.shape[1]
            elif imu.shape[1] != imu_expected:
                raise ValueError(
                    f"{p} IMU channel mismatch: {imu.shape[1]} vs {imu_expected}"
                )
            imu_list.append(imu)

        wrist = _safe_get(data, "wrist_orientation")
        if isinstance(wrist, np.ndarray):
            try:
                wrist = wrist.item()
            except Exception:
                wrist = None
        if wrist:
            meta["wrist_orientations"].append(str(wrist))

        emg_list.append(emg)
        angles_list.append(angles)
        meta["datasets"].append(str(p))

    emg_all = (
        np.concatenate(emg_list, axis=0)
        if emg_list
        else np.zeros((0, 8, 1, 1), dtype=np.float32)
    )
    angles_all = (
        np.concatenate(angles_list, axis=0)
        if angles_list
        else np.zeros((0, angle_dim_ref or 0), dtype=np.float32)
    )
    imu_all = None
    if used_imu and imu_list:
        imu_all = np.concatenate(imu_list, axis=0)
    else:
        used_imu = False

    meta["window_len"] = int(window_len_ref or emg_all.shape[2])
    meta["angle_dim"] = int(angle_dim_ref or angles_all.shape[1])
    meta["angle_keys"] = angle_keys_ref
    meta["target_spec"] = target_spec_ref
    meta["imu_included"] = bool(used_imu)
    return emg_all, angles_all, imu_all, meta


def main():
    parser = argparse.ArgumentParser(
        description="Train joint-angle regressor on EMG/IMU features."
    )
    parser.add_argument(
        "--data", nargs="+", required=True, help="One or more .npz datasets"
    )
    parser.add_argument(
        "--feature_extractor",
        default="none",
        help="Path to pretrained feature extractor (.h5) or 'none'",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for model/scaler/metadata"
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--max_iter", type=int, default=500, help="MLP max iterations")
    parser.add_argument(
        "--emg_transform",
        default="log1p",
        choices=["none", "log1p"],
        help="EMG transform",
    )
    parser.add_argument(
        "--use_imu",
        default="auto",
        choices=["auto", "on", "off"],
        help="Whether to include IMU features (auto=use if present).",
    )
    parser.add_argument(
        "--emg_features",
        default="bandpower_stats",
        choices=["bandpower_stats", "emd_stats", "raw_flat"],
        help="EMG feature mode when no extractor is provided",
    )
    parser.add_argument(
        "--emd_max_imfs",
        type=int,
        default=3,
        help="Max IMFs per channel for emd_stats feature mode",
    )
    parser.add_argument(
        "--angle_scaler",
        default="minmax",
        choices=["none", "minmax", "standard"],
        help="Scale output angles before training",
    )
    args = parser.parse_args()

    emg, angles, imu, meta_in = _load_datasets([Path(p) for p in args.data])
    if args.use_imu == "off":
        imu = None
        meta_in["imu_included"] = False
    elif args.use_imu == "on" and imu is None:
        print("[WARN] IMU requested but not found in datasets; continuing without IMU.")
    angle_keys = meta_in.get("angle_keys")
    target_spec = meta_in.get("target_spec")
    emg = _normalize_emg(emg)
    angles = np.asarray(angles, dtype=np.float32)

    mask = np.isfinite(angles).all(axis=1)
    emg_mask = np.isfinite(emg).all(axis=(1, 2, 3))
    mask = mask & emg_mask
    if imu is not None:
        imu = _normalize_imu(imu)
        imu_mask = np.isfinite(imu).all(axis=(1, 2))
        mask = mask & imu_mask
    if not np.all(mask):
        dropped = int(np.size(mask) - np.count_nonzero(mask))
        print(f"[WARN] Dropping {dropped} samples with NaN/Inf values")
        emg = emg[mask]
        angles = angles[mask]
        if imu is not None:
            imu = imu[mask]

    extractor = load_feature_extractor(args.feature_extractor)
    features = extract_features(
        emg,
        extractor,
        args.emg_features,
        args.emg_transform,
        emd_max_imfs=args.emd_max_imfs,
    )
    imu_feat = None
    if imu is not None:
        imu_feat = _imu_features(imu)
        features = np.concatenate([features, imu_feat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, angles, test_size=args.test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    target_scaler = None
    y_train_scaled = y_train
    if args.angle_scaler == "minmax":
        target_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    elif args.angle_scaler == "standard":
        target_scaler = StandardScaler()
    if target_scaler is not None:
        y_train_scaled = target_scaler.fit_transform(y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=args.max_iter,
        random_state=42,
    )
    mlp.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = mlp.predict(X_test_scaled)
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput="variance_weighted")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "mlp_regressor.pkl"), "wb") as f:
        pickle.dump(mlp, f)
    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    if target_scaler is not None:
        with open(os.path.join(args.out_dir, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)

    first_data = np.load(args.data[0], allow_pickle=True)
    meta = {
        "mae": float(mae),
        "r2": float(r2),
        "mae_cutoff": None,
        "r2_cutoff": None,
        "feature_extractor": args.feature_extractor if extractor is not None else None,
        "feature_mode": "extractor" if extractor is not None else args.emg_features,
        "emg_feature_mode": args.emg_features if extractor is None else "extractor",
        "emg_transform": args.emg_transform,
        "emd_max_imfs": int(args.emd_max_imfs),
        "angle_scaler": args.angle_scaler,
        "data": [str(p) for p in args.data],
        "imu_features": int(imu_feat.shape[1]) if imu_feat is not None else 0,
        "imu_mode": args.use_imu,
        "target_spec": target_spec,
        "angle_keys": angle_keys.tolist()
        if hasattr(angle_keys, "tolist")
        else angle_keys,
        "window_ms": _safe_scalar(first_data, "window_ms"),
        "overlap_ms": _safe_scalar(first_data, "overlap_ms"),
        "fs": _safe_scalar(first_data, "fs"),
    }
    meta.update(
        {
            "wrist_orientations": meta_in.get("wrist_orientations"),
            "imu_included": meta_in.get("imu_included"),
        }
    )
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] MAE={mae:.4f}  R2={r2:.4f}")
    print(f"[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()
