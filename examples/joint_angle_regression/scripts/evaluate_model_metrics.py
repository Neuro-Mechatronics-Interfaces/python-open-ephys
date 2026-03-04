import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import train_regressor as tr
from sklearn.metrics import mean_absolute_error, r2_score


def _cnn_custom_objects():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return {}

    class SinusoidalPositionalEncoding(keras.layers.Layer):
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

    return {"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding}


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


def _causal_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or arr.size == 0:
        return arr
    k = int(max(1, k))
    out = np.empty_like(arr, dtype=np.float32)
    csum = np.cumsum(arr.astype(np.float32), axis=0)
    for i in range(arr.shape[0]):
        i0 = max(0, i - k + 1)
        if i0 == 0:
            out[i] = csum[i] / float(i + 1)
        else:
            out[i] = (csum[i] - csum[i0 - 1]) / float(i - i0 + 1)
    return out


def _load_personalization(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _extract_features_chunked(
    emg,
    imu,
    metrics,
    chunk_size=0,
    cache_path: Path | None = None,
):
    feature_mode = metrics.get(
        "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
    )
    emg_transform = metrics.get("emg_transform", "none")
    emd_max_imfs = int(metrics.get("emd_max_imfs", 3))
    use_imu = bool(metrics.get("imu_included", False)) and imu is not None

    if cache_path is not None and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        feats = np.asarray(payload["features"], dtype=np.float32)
        cached_mode = str(payload.get("feature_mode", ""))
        cached_transform = str(payload.get("emg_transform", ""))
        cached_imfs = int(payload.get("emd_max_imfs", -1))
        cached_n = int(payload.get("n", -1))
        if (
            cached_mode == feature_mode
            and cached_transform == emg_transform
            and cached_imfs == emd_max_imfs
            and cached_n == int(emg.shape[0])
        ):
            return feats, True

    n = int(emg.shape[0])
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = n
    chunk_size = max(1, int(chunk_size))

    all_feats = []
    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        feats = tr.extract_features(
            emg[i0:i1],
            None,
            feature_mode=feature_mode,
            emg_transform=emg_transform,
            emd_max_imfs=emd_max_imfs,
        )
        if use_imu:
            feats = np.concatenate([feats, tr._imu_features(imu[i0:i1])], axis=1)
        all_feats.append(np.asarray(feats, dtype=np.float32))

    features = (
        np.concatenate(all_feats, axis=0)
        if all_feats
        else np.zeros((0, 0), dtype=np.float32)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features,
            feature_mode=feature_mode,
            emg_transform=emg_transform,
            emd_max_imfs=emd_max_imfs,
            n=int(n),
        )

    return features, False


def evaluate(
    model_dir: Path,
    npz_path: Path,
    max_windows: int | None = None,
    chunk_size: int = 0,
    feature_cache: Path | None = None,
    personalization_path: Path | None = None,
    smooth_samples: int = 0,
):
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    emg = tr._normalize_emg(np.asarray(data["emg"], dtype=np.float32))
    angles = np.asarray(data["angles"], dtype=np.float32)
    imu = None
    if "imu" in data.files:
        imu = tr._normalize_imu(np.asarray(data["imu"], dtype=np.float32))

    if max_windows is not None and max_windows > 0:
        emg = emg[:max_windows]
        angles = angles[:max_windows]
        if imu is not None:
            imu = imu[:max_windows]

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    model_type = str(metrics.get("model_type", "")).lower()
    is_cnn = (model_dir / "cnn_attention_regressor.h5").exists() or (
        "cnn_attention" in model_type
    )
    feature_mode = metrics.get(
        "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
    )
    emg_transform = metrics.get("emg_transform", "none")
    emd_max_imfs = int(metrics.get("emd_max_imfs", 3))
    use_imu = bool(metrics.get("use_imu", False))

    target_scaler = None
    if (model_dir / "target_scaler.pkl").exists():
        with open(model_dir / "target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)

    if is_cnn:
        from tensorflow.keras.models import load_model

        model = load_model(
            str(model_dir / "cnn_attention_regressor.h5"),
            compile=False,
            custom_objects=_cnn_custom_objects(),
        )
        emg_in = tr._apply_emg_transform(emg[..., 0], emg_transform)[..., None]
        input_mode = str(metrics.get("input_mode", "raw")).lower()
        if input_mode == "rms_subframes":
            emg_in = _rms_subframe_sequence(
                emg_in,
                n_subframes=int(metrics.get("rms_subframes", 25)),
            )
        model_input = emg_in
        if use_imu:
            if imu is None:
                raise ValueError(
                    "metrics.json indicates use_imu=true but NPZ has no IMU data"
                )
            imu_feat = tr._imu_features(imu)
            model_input = [emg_in, imu_feat]
        pred_scaled = model.predict(model_input, verbose=0)
    else:
        with open(model_dir / "mlp_regressor.pkl", "rb") as f:
            reg = pickle.load(f)
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        feats, used_cache = _extract_features_chunked(
            emg=emg,
            imu=imu,
            metrics=metrics,
            chunk_size=chunk_size,
            cache_path=feature_cache,
        )

        pred_scaled = reg.predict(scaler.transform(feats))

    used_cache = bool(
        (not is_cnn) and feature_cache is not None and feature_cache.exists()
    )
    pred = (
        target_scaler.inverse_transform(pred_scaled)
        if target_scaler is not None
        else pred_scaled
    )

    if personalization_path is None:
        p_diag = model_dir / "personalization_diagonal_identity.pkl"
        p_std = model_dir / "personalization.pkl"
        personalization_path = (
            p_diag if p_diag.exists() else (p_std if p_std.exists() else None)
        )
    personalization = _load_personalization(personalization_path)
    if personalization is not None:
        pred = tr.apply_personalization(pred, personalization)

    pred = _causal_smooth(np.asarray(pred, dtype=np.float32), int(smooth_samples))

    mae = mean_absolute_error(angles, pred)
    r2 = r2_score(angles, pred, multioutput="variance_weighted")

    return {
        "model_dir": str(model_dir),
        "npz": str(npz_path),
        "n": int(angles.shape[0]),
        "feature_mode": feature_mode,
        "model_type": ("cnn_attention_regressor" if is_cnn else "mlp_regressor"),
        "use_imu": bool(use_imu),
        "emg_transform": emg_transform,
        "emd_max_imfs": emd_max_imfs,
        "chunk_size": int(chunk_size) if int(chunk_size) > 0 else int(angles.shape[0]),
        "feature_cache": str(feature_cache) if feature_cache is not None else None,
        "used_cache": bool(used_cache),
        "personalization": str(personalization_path)
        if personalization_path is not None
        else None,
        "smooth_samples": int(smooth_samples),
        "elapsed_sec": float(time.time() - t0),
        "mae": float(mae),
        "r2": float(r2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one trained model directory on one NPZ"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Chunk size for feature extraction (0=all at once)",
    )
    parser.add_argument(
        "--feature_cache",
        default="",
        help="Optional .npz cache path for extracted features",
    )
    parser.add_argument(
        "--personalization",
        default="",
        help="Optional personalization .pkl path (default: auto-detect in model_dir)",
    )
    parser.add_argument(
        "--smooth_samples",
        type=int,
        default=0,
        help="Causal moving-average window in prediction samples",
    )
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    res = evaluate(
        model_dir=Path(args.model_dir),
        npz_path=Path(args.npz),
        max_windows=(args.max_windows if args.max_windows > 0 else None),
        chunk_size=args.chunk_size,
        feature_cache=(Path(args.feature_cache) if args.feature_cache else None),
        personalization_path=(
            Path(args.personalization) if args.personalization else None
        ),
        smooth_samples=args.smooth_samples,
    )

    print(
        f"[{Path(args.model_dir).name}] n={res['n']} mode={res['feature_mode']} "
        f"MAE={res['mae']:.4f} R2={res['r2']:.4f} t={res['elapsed_sec']:.1f}s"
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
