import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import train_regressor as tr


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_gesture_segments(data_npz):
    if "plan_json" not in data_npz.files:
        return []
    try:
        items = json.loads(str(data_npz["plan_json"]))
    except Exception:
        return []
    if not isinstance(items, list):
        return []

    segments = []
    t = 0.0
    for item in items:
        try:
            label = str(item.get("label", ""))
            dur = float(item.get("duration", 0.0))
        except Exception:
            continue
        dur = max(0.0, dur)
        if label:
            segments.append({"label": label, "start": t, "end": t + dur})
        t += dur
    return segments


def _align_actual_angles(all_angles, all_keys, target_keys):
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    idx = [key_to_idx[k] for k in target_keys]
    return all_angles[:, idx]


def _load_model_bundle(model_dir: Path, feature_extractor_path: Path):
    with open(model_dir / "mlp_regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    target_scaler = None
    target_scaler_path = model_dir / "target_scaler.pkl"
    if target_scaler_path.exists():
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

    metrics = _load_json(model_dir / "metrics.json")
    extractor = tr.load_feature_extractor(str(feature_extractor_path))

    return {
        "regressor": regressor,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "metrics": metrics,
        "extractor": extractor,
    }


def _predict_for_bundle(emg, actual_angles, bundle):
    feature_mode = bundle["metrics"].get("emg_feature_mode", "raw_flat")
    emg_transform = bundle["metrics"].get("emg_transform", "none")

    emg_arr = tr._normalize_emg(emg)
    y_arr = np.asarray(actual_angles, dtype=np.float32)

    mask = np.isfinite(y_arr).all(axis=1) & np.isfinite(emg_arr).all(axis=(1, 2, 3))
    emg_arr = emg_arr[mask]
    y_arr = y_arr[mask]

    feats = tr.extract_features(
        emg_arr,
        bundle["extractor"],
        feature_mode=feature_mode,
        emg_transform=emg_transform,
    )
    feats = bundle["scaler"].transform(feats)
    pred_scaled = bundle["regressor"].predict(feats)
    pred = (
        bundle["target_scaler"].inverse_transform(pred_scaled)
        if bundle["target_scaler"] is not None
        else pred_scaled
    )

    return {"mask": mask, "actual": y_arr, "pred": pred}


def _pearson_per_joint(y_true, y_pred):
    vals = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i]
        b = y_pred[:, i]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            vals.append(float("nan"))
        else:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return vals


def _lag_samples_per_joint(y_true, y_pred, max_lag=120):
    lags = []
    n = y_true.shape[0]
    for i in range(y_true.shape[1]):
        a = y_true[:, i] - np.mean(y_true[:, i])
        b = y_pred[:, i] - np.mean(y_pred[:, i])
        best_lag = 0
        best_corr = -np.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a_seg = a[-lag:]
                b_seg = b[: n + lag]
            elif lag > 0:
                a_seg = a[: n - lag]
                b_seg = b[lag:]
            else:
                a_seg = a
                b_seg = b
            if len(a_seg) < 10:
                continue
            denom = np.linalg.norm(a_seg) * np.linalg.norm(b_seg)
            if denom < 1e-12:
                continue
            corr = float(np.dot(a_seg, b_seg) / denom)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        lags.append(int(best_lag))
    return lags


def _velocity_mae_per_joint(y_true, y_pred):
    if y_true.shape[0] < 2:
        return [float("nan")] * y_true.shape[1]
    v_true = np.diff(y_true, axis=0)
    v_pred = np.diff(y_pred, axis=0)
    return [float(np.mean(np.abs(v_true[:, i] - v_pred[:, i]))) for i in range(y_true.shape[1])]


def _compute_joint_metrics(y_true, y_pred, joint_keys):
    abs_err = np.abs(y_true - y_pred)
    err = y_pred - y_true
    sq_err = (y_true - y_pred) ** 2

    mae_per_joint = np.mean(abs_err, axis=0)
    rmse_per_joint = np.sqrt(np.mean(sq_err, axis=0))
    bias_per_joint = np.mean(err, axis=0)

    p50 = np.percentile(abs_err, 50, axis=0)
    p90 = np.percentile(abs_err, 90, axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)

    rom = np.maximum(np.max(y_true, axis=0) - np.min(y_true, axis=0), 1e-6)
    nrmse = rmse_per_joint / rom

    r2_j = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    pearson_j = _pearson_per_joint(y_true, y_pred)
    lag_j = _lag_samples_per_joint(y_true, y_pred, max_lag=120)
    vel_mae_j = _velocity_mae_per_joint(y_true, y_pred)

    rows = []
    for i, k in enumerate(joint_keys):
        rows.append(
            {
                "joint": k,
                "mae": float(mae_per_joint[i]),
                "rmse": float(rmse_per_joint[i]),
                "nrmse": float(nrmse[i]),
                "bias": float(bias_per_joint[i]),
                "p50_abs_err": float(p50[i]),
                "p90_abs_err": float(p90[i]),
                "p95_abs_err": float(p95[i]),
                "r2": float(r2_j[i]),
                "pearson_r": float(pearson_j[i]) if np.isfinite(pearson_j[i]) else None,
                "lag_samples": int(lag_j[i]),
                "vel_mae": float(vel_mae_j[i]) if np.isfinite(vel_mae_j[i]) else None,
            }
        )

    pooled = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "r2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "bias": float(np.mean(err)),
        "p50_abs_err": float(np.percentile(abs_err, 50)),
        "p90_abs_err": float(np.percentile(abs_err, 90)),
        "p95_abs_err": float(np.percentile(abs_err, 95)),
    }

    return pooled, rows


def _compute_per_gesture_metrics(y_true, y_pred, segments):
    if not segments:
        return []

    rows = []
    for seg in segments:
        s = int(max(0, np.floor(seg["start"])))
        e = int(min(y_true.shape[0], np.ceil(seg["end"])))
        if e <= s:
            continue
        yt = y_true[s:e]
        yp = y_pred[s:e]
        abs_err = np.abs(yt - yp)
        rows.append(
            {
                "gesture": seg["label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "n_samples": int(e - s),
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "bias": float(np.mean(yp - yt)),
                "p90_abs_err": float(np.percentile(abs_err, 90)),
            }
        )
    return rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _evaluate_model(
    model_name,
    bundle,
    emg,
    all_angles,
    all_keys,
    gesture_segments,
):
    target_keys = bundle["metrics"].get("angle_keys") or all_keys
    actual = _align_actual_angles(all_angles, all_keys, target_keys)
    pred_pack = _predict_for_bundle(emg, actual, bundle)

    y_true = pred_pack["actual"]
    y_pred = pred_pack["pred"]

    pooled, per_joint = _compute_joint_metrics(y_true, y_pred, target_keys)
    per_gesture = _compute_per_gesture_metrics(y_true, y_pred, gesture_segments)

    return {
        "model": model_name,
        "n_samples": int(y_true.shape[0]),
        "n_joints": int(y_true.shape[1]),
        "pooled": pooled,
        "per_joint": per_joint,
        "per_gesture": per_gesture,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute expanded session6 metrics for full14/finger5/index models.")
    parser.add_argument(
        "--session6_npz",
        default="data/sub-001/ses-006/sub-001_ses-006_task-jointangles_wrist-neutral.npz",
    )
    parser.add_argument("--full14_model_dir", default="models/sub-001/ses-001to005_regressor")
    parser.add_argument("--full14_feature_extractor", default="models/sub-001/ses-001to005_feature_extractor.h5")
    parser.add_argument("--finger5_model_dir", default="models/sub-001/finger5_ses-001to005_regressor")
    parser.add_argument("--finger5_feature_extractor", default="models/sub-001/finger5_ses-001to005_feature_extractor.h5")
    parser.add_argument("--index_model_dir", default="models/sub-001/index-only_ses-001to005_regressor")
    parser.add_argument("--index_feature_extractor", default="models/sub-001/index-only_ses-001to005_feature_extractor.h5")
    parser.add_argument("--out_dir", default="models/sub-001/session6_metrics_report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.session6_npz, allow_pickle=True)
    emg = d["emg"]
    all_angles = np.asarray(d["angles"], dtype=np.float32)
    all_keys = d["angle_keys"].tolist() if "angle_keys" in d.files else []
    gesture_segments = _extract_gesture_segments(d)

    bundles = {
        "full14": _load_model_bundle(Path(args.full14_model_dir), Path(args.full14_feature_extractor)),
        "finger5": _load_model_bundle(Path(args.finger5_model_dir), Path(args.finger5_feature_extractor)),
        "index_only": _load_model_bundle(Path(args.index_model_dir), Path(args.index_feature_extractor)),
    }

    report = {
        "session6_npz": str(args.session6_npz),
        "gesture_segment_count": len(gesture_segments),
        "models": {},
    }

    pooled_rows = []
    for name, bundle in bundles.items():
        res = _evaluate_model(name, bundle, emg, all_angles, all_keys, gesture_segments)
        report["models"][name] = res

        pooled_rows.append({"model": name, **res["pooled"], "n_samples": res["n_samples"], "n_joints": res["n_joints"]})

        _write_csv(out_dir / f"{name}_per_joint_metrics.csv", res["per_joint"])
        _write_csv(out_dir / f"{name}_per_gesture_metrics.csv", res["per_gesture"])

    _write_csv(out_dir / "pooled_metrics.csv", pooled_rows)
    (out_dir / "metrics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[DONE] Expanded metrics computed")
    for row in pooled_rows:
        print(f"[{row['model']}] MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f} P90={row['p90_abs_err']:.4f}")
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
