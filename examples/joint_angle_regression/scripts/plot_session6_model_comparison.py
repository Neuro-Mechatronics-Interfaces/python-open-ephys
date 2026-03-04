import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
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


def _align_actual_angles(all_angles, all_keys, target_keys):
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    idx = [key_to_idx[k] for k in target_keys]
    return all_angles[:, idx]


def _predict(emg, actual_angles, bundle):
    metrics = bundle["metrics"]
    feature_mode = metrics.get("emg_feature_mode", "raw_flat")
    emg_transform = metrics.get("emg_transform", "none")

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
    if bundle["target_scaler"] is not None:
        pred = bundle["target_scaler"].inverse_transform(pred_scaled)
    else:
        pred = pred_scaled

    mae = float(mean_absolute_error(y_arr, pred))
    r2 = float(r2_score(y_arr, pred, multioutput="variance_weighted"))
    mae_per_joint = np.mean(np.abs(y_arr - pred), axis=0)

    return {
        "mask": mask,
        "actual": y_arr,
        "pred": pred,
        "mae": mae,
        "r2": r2,
        "mae_per_joint": mae_per_joint,
    }


def _extract_gesture_events(data_npz):
    if "plan_json" not in data_npz.files:
        return []
    try:
        plan_raw = str(data_npz["plan_json"])
        items = json.loads(plan_raw)
    except Exception:
        return []
    if not isinstance(items, list):
        return []

    events = []
    t = 0.0
    for item in items:
        try:
            label = str(item.get("label", ""))
            dur = float(item.get("duration", 0.0))
        except Exception:
            continue
        if label:
            events.append((t, label))
        t += max(0.0, dur)
    return events


def _extract_gesture_segments(data_npz):
    if "plan_json" not in data_npz.files:
        return []
    try:
        plan_raw = str(data_npz["plan_json"])
        items = json.loads(plan_raw)
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


def _plot_timeseries_grid(
    time_axis,
    actual,
    pred,
    joint_keys,
    title,
    out_path,
    gesture_events=None,
):
    n = len(joint_keys)
    cols = 2 if n > 6 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16 if cols == 2 else 14, 2.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, key in enumerate(joint_keys):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.plot(time_axis, actual[:, idx], color="#1f77b4", linewidth=1.0, label="Actual")
        ax.plot(time_axis, pred[:, idx], color="#d62728", linewidth=1.0, alpha=0.9, linestyle="--", label="Pred")
        ax.set_title(key)
        ax.grid(alpha=0.25)
        if gesture_events:
            y0, y1 = ax.get_ylim()
            ytxt = y0 + 0.92 * (y1 - y0)
            for g_t, g_label in gesture_events:
                if g_t < float(time_axis[0]) or g_t > float(time_axis[-1]):
                    continue
                ax.axvline(g_t, color="#7f7f7f", linestyle=":", linewidth=0.7, alpha=0.7)
                if r == 0 and c == 0:
                    ax.text(
                        g_t,
                        ytxt,
                        g_label,
                        rotation=90,
                        va="top",
                        ha="center",
                        fontsize=7,
                        alpha=0.85,
                    )

    # Hide any unused axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Time")
    fig.supylabel("Angle")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _rolling_mae_trace(actual, pred, window=200):
    err = np.mean(np.abs(actual - pred), axis=1)
    if len(err) < window:
        return np.arange(len(err)), err
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(err, kernel, mode="valid")
    x = np.arange(len(smooth)) + (window // 2)
    return x, smooth


def _plot_error_trace(
    title,
    actual,
    pred,
    time_axis,
    out_path,
    window=200,
    color="#f58518",
    gesture_events=None,
):
    x, y = _rolling_mae_trace(actual, pred, window=window)
    t = np.asarray(time_axis, dtype=np.float64).reshape(-1)
    x_t = t[np.clip(x, 0, max(0, t.size - 1))] if t.size else x
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(x_t, y, color=color, linewidth=1.5)
    if gesture_events:
        y0, y1 = ax.get_ylim()
        ytxt = y0 + 0.96 * (y1 - y0)
        for g_t, g_label in gesture_events:
            if x_t.size and (g_t < float(x_t[0]) or g_t > float(x_t[-1])):
                continue
            ax.axvline(g_t, color="#7f7f7f", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(
                g_t,
                ytxt,
                g_label,
                rotation=90,
                va="top",
                ha="center",
                fontsize=7,
                alpha=0.85,
            )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rolling MAE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_performance_summary(
    full14,
    full14_keys,
    finger5,
    finger5_keys,
    index_only,
    index_only_keys,
    t14,
    t5,
    ti,
    out_path,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Full14 MAE per joint
    ax = axes[0, 0]
    ax.bar(np.arange(len(full14_keys)), full14["mae_per_joint"], color="#4c78a8")
    ax.set_title(f"Full14 MAE per joint (overall MAE={full14['mae']:.2f}, R²={full14['r2']:.3f})")
    ax.set_xticks(np.arange(len(full14_keys)))
    ax.set_xticklabels(full14_keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)

    # Finger5 MAE per joint
    ax = axes[0, 1]
    ax.bar(np.arange(len(finger5_keys)), finger5["mae_per_joint"], color="#f58518")
    ax.set_title(f"Finger5 MAE per joint (overall MAE={finger5['mae']:.2f}, R²={finger5['r2']:.3f})")
    ax.set_xticks(np.arange(len(finger5_keys)))
    ax.set_xticklabels(finger5_keys, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)

    # Pooled scatter (actual vs predicted)
    ax = axes[1, 0]
    y_a = full14["actual"].reshape(-1)
    y_p = full14["pred"].reshape(-1)
    f_a = finger5["actual"].reshape(-1)
    f_p = finger5["pred"].reshape(-1)
    i_a = index_only["actual"].reshape(-1)
    i_p = index_only["pred"].reshape(-1)
    ax.scatter(y_a, y_p, s=3, alpha=0.15, label="Full14", color="#4c78a8")
    ax.scatter(f_a, f_p, s=3, alpha=0.15, label="Finger5", color="#f58518")
    ax.scatter(i_a, i_p, s=3, alpha=0.15, label="Index", color="#54a24b")
    low = 0.0
    high = float(max(np.max(y_a), np.max(y_p), np.max(f_a), np.max(f_p), np.max(i_a), np.max(i_p)))
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_title("Pooled Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    ax.grid(alpha=0.25)

    # Rolling MAE over time for all models
    ax = axes[1, 1]
    window = 200
    x14, y14 = _rolling_mae_trace(full14["actual"], full14["pred"], window=window)
    x5, y5 = _rolling_mae_trace(finger5["actual"], finger5["pred"], window=window)
    xi, yi = _rolling_mae_trace(index_only["actual"], index_only["pred"], window=window)
    t14 = np.asarray(t14, dtype=np.float64).reshape(-1)
    t5 = np.asarray(t5, dtype=np.float64).reshape(-1)
    ti = np.asarray(ti, dtype=np.float64).reshape(-1)
    x14_t = t14[np.clip(x14, 0, max(0, t14.size - 1))] if t14.size else x14
    x5_t = t5[np.clip(x5, 0, max(0, t5.size - 1))] if t5.size else x5
    xi_t = ti[np.clip(xi, 0, max(0, ti.size - 1))] if ti.size else xi
    ax.plot(x14_t, y14, label="Full14 rolling MAE", color="#4c78a8")
    ax.plot(x5_t, y5, label="Finger5 rolling MAE", color="#f58518")
    ax.plot(xi_t, yi, label="Index rolling MAE", color="#54a24b")
    ax.set_title("Error over time (rolling MAE)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle("Session 6 Prediction Performance Summary", fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_task_alignment_confidence(
    full14,
    finger5,
    index_only,
    gesture_segments,
    t14,
    t5,
    ti,
    out_path,
):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)

    # Top: rolling MAE traces with gesture shading
    ax = axes[0]
    x14, y14 = _rolling_mae_trace(full14["actual"], full14["pred"], window=200)
    x5, y5 = _rolling_mae_trace(finger5["actual"], finger5["pred"], window=200)
    xi, yi = _rolling_mae_trace(index_only["actual"], index_only["pred"], window=200)

    t14 = np.asarray(t14, dtype=np.float64).reshape(-1)
    t5 = np.asarray(t5, dtype=np.float64).reshape(-1)
    ti = np.asarray(ti, dtype=np.float64).reshape(-1)

    x14_t = t14[np.clip(x14, 0, max(0, t14.size - 1))] if t14.size else x14
    x5_t = t5[np.clip(x5, 0, max(0, t5.size - 1))] if t5.size else x5
    xi_t = ti[np.clip(xi, 0, max(0, ti.size - 1))] if ti.size else xi

    ax.plot(x14_t, y14, label="Full14", color="#4c78a8", linewidth=1.4)
    ax.plot(x5_t, y5, label="Finger5", color="#f58518", linewidth=1.4)
    ax.plot(xi_t, yi, label="Index", color="#54a24b", linewidth=1.4)

    if gesture_segments:
        colors = ["#e8eef6", "#f8efe2"]
        y0, y1 = ax.get_ylim()
        ytxt = y0 + 0.96 * (y1 - y0)
        for idx, seg in enumerate(gesture_segments):
            start = float(seg["start"])
            end = float(seg["end"])
            label = str(seg["label"])
            ax.axvspan(start, end, color=colors[idx % 2], alpha=0.35, zorder=0)
            mid = (start + end) * 0.5
            if x14_t.size and mid >= float(x14_t[0]) and mid <= float(x14_t[-1]):
                ax.text(mid, ytxt, label, ha="center", va="top", fontsize=7, rotation=90, alpha=0.8)

    ax.set_title("Task Alignment Confidence: Rolling Error vs Gesture Blocks")
    if gesture_segments:
        max_t = max(float(seg["end"]) for seg in gesture_segments)
        ax.set_xlim(0.0, max_t)
    ax.set_ylabel("Rolling MAE")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    # Bottom: per-gesture mean error bars (categorical)
    ax = axes[1]
    if gesture_segments:
        labels = []
        full_vals = []
        finger_vals = []
        index_vals = []

        def segment_mae(arr_actual, arr_pred, time_axis, start, end):
            ta = np.asarray(time_axis, dtype=np.float64).reshape(-1)
            if ta.size == 0:
                return np.nan
            mask = (ta >= start) & (ta < end)
            if not np.any(mask):
                return np.nan
            return float(np.mean(np.abs(arr_actual[mask] - arr_pred[mask])))

        for seg in gesture_segments:
            label = str(seg["label"])
            start = float(seg["start"])
            end = float(seg["end"])
            labels.append(label)
            full_vals.append(segment_mae(full14["actual"], full14["pred"], t14, start, end))
            finger_vals.append(segment_mae(finger5["actual"], finger5["pred"], t5, start, end))
            index_vals.append(segment_mae(index_only["actual"], index_only["pred"], ti, start, end))

        x = np.arange(len(labels))
        w = 0.28
        ax.bar(x - w, full_vals, width=w, label="Full14", color="#4c78a8")
        ax.bar(x, finger_vals, width=w, label="Finger5", color="#f58518")
        ax.bar(x + w, index_vals, width=w, label="Index", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean absolute error")
        ax.set_title("Per-Gesture Mean Error")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No gesture segments available", ha="center", va="center")
        ax.set_axis_off()

    axes[1].set_xlabel("Gesture")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot session-6 actual vs predicted joint angles for full14 and finger5 models."
    )
    parser.add_argument(
        "--session6_npz",
        default="data/sub-001/ses-006/sub-001_ses-006_task-jointangles_wrist-neutral.npz",
    )
    parser.add_argument(
        "--full14_model_dir",
        default="models/sub-001/ses-001to005_regressor",
    )
    parser.add_argument(
        "--full14_feature_extractor",
        default="models/sub-001/ses-001to005_feature_extractor.h5",
    )
    parser.add_argument(
        "--finger5_model_dir",
        default="models/sub-001/finger5_ses-001to005_regressor",
    )
    parser.add_argument(
        "--finger5_feature_extractor",
        default="models/sub-001/finger5_ses-001to005_feature_extractor.h5",
    )
    parser.add_argument(
        "--index_model_dir",
        default="models/sub-001/index-only_ses-001to005_regressor",
    )
    parser.add_argument(
        "--index_feature_extractor",
        default="models/sub-001/index-only_ses-001to005_feature_extractor.h5",
    )
    parser.add_argument(
        "--out_dir",
        default="models/sub-001/session6_comparison_figs",
    )
    args = parser.parse_args()

    session6_npz = Path(args.session6_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(session6_npz, allow_pickle=True)
    emg = data["emg"]
    all_angles = np.asarray(data["angles"], dtype=np.float32)
    all_keys = data["angle_keys"].tolist() if "angle_keys" in data else []
    timestamps = np.asarray(data["timestamps"], dtype=np.float64) if "timestamps" in data else None
    gesture_events = _extract_gesture_events(data)
    gesture_segments = _extract_gesture_segments(data)

    full14_bundle = _load_model_bundle(
        Path(args.full14_model_dir), Path(args.full14_feature_extractor)
    )
    finger5_bundle = _load_model_bundle(
        Path(args.finger5_model_dir), Path(args.finger5_feature_extractor)
    )
    index_bundle = _load_model_bundle(
        Path(args.index_model_dir), Path(args.index_feature_extractor)
    )

    full14_keys = full14_bundle["metrics"].get("angle_keys") or all_keys
    finger5_keys = finger5_bundle["metrics"].get("angle_keys") or [
        "thumb_cmc_mcp",
        "index_mcp",
        "middle_mcp",
        "ring_mcp",
        "pinky_mcp",
    ]
    index_keys = index_bundle["metrics"].get("angle_keys") or [
        "index_mcp",
        "index_pip",
        "index_dip",
    ]

    full14_actual = _align_actual_angles(all_angles, all_keys, full14_keys)
    finger5_actual = _align_actual_angles(all_angles, all_keys, finger5_keys)
    index_actual = _align_actual_angles(all_angles, all_keys, index_keys)

    full14_pred = _predict(emg, full14_actual, full14_bundle)
    finger5_pred = _predict(emg, finger5_actual, finger5_bundle)
    index_pred = _predict(emg, index_actual, index_bundle)

    # Time axis aligned to masks
    if timestamps is not None and len(timestamps) == len(all_angles):
        t14 = timestamps[full14_pred["mask"]]
        t5 = timestamps[finger5_pred["mask"]]
        ti = timestamps[index_pred["mask"]]

        # Anchor to session_start when available so gesture plan timings (which
        # begin at 0 s) align to recorded windows.
        session_start = None
        if "session_start" in data.files:
            try:
                session_start = float(np.asarray(data["session_start"]).reshape(-1)[0])
            except Exception:
                session_start = None

        if session_start is not None and np.isfinite(session_start):
            t14 = t14 - session_start
            t5 = t5 - session_start
            ti = ti - session_start
        else:
            t14 = t14 - t14[0]
            t5 = t5 - t5[0]
            ti = ti - ti[0]
    else:
        t14 = np.arange(full14_pred["actual"].shape[0])
        t5 = np.arange(finger5_pred["actual"].shape[0])
        ti = np.arange(index_pred["actual"].shape[0])

    _plot_timeseries_grid(
        t14,
        full14_pred["actual"],
        full14_pred["pred"],
        full14_keys,
        "Session 6: Full14 Actual vs Predicted Time Series",
        out_dir / "session6_full14_timeseries.png",
    )
    _plot_timeseries_grid(
        t14,
        full14_pred["actual"],
        full14_pred["pred"],
        full14_keys,
        "Session 6: Full14 Actual vs Predicted Time Series (Annotated)",
        out_dir / "session6_full14_timeseries_annotated.png",
        gesture_events=gesture_events,
    )
    _plot_timeseries_grid(
        t5,
        finger5_pred["actual"],
        finger5_pred["pred"],
        finger5_keys,
        "Session 6: Finger5 Actual vs Predicted Time Series",
        out_dir / "session6_finger5_timeseries.png",
    )
    _plot_timeseries_grid(
        t5,
        finger5_pred["actual"],
        finger5_pred["pred"],
        finger5_keys,
        "Session 6: Finger5 Actual vs Predicted Time Series (Annotated)",
        out_dir / "session6_finger5_timeseries_annotated.png",
        gesture_events=gesture_events,
    )
    _plot_timeseries_grid(
        ti,
        index_pred["actual"],
        index_pred["pred"],
        index_keys,
        "Session 6: Index-only Actual vs Predicted Time Series",
        out_dir / "session6_index_timeseries.png",
    )
    _plot_timeseries_grid(
        ti,
        index_pred["actual"],
        index_pred["pred"],
        index_keys,
        "Session 6: Index-only Actual vs Predicted Time Series (Annotated)",
        out_dir / "session6_index_timeseries_annotated.png",
        gesture_events=gesture_events,
    )

    _plot_error_trace(
        "Session 6: Full14 Error Trace (rolling MAE)",
        full14_pred["actual"],
        full14_pred["pred"],
        t14,
        out_dir / "session6_full14_error_trace.png",
        window=200,
        color="#4c78a8",
    )
    _plot_error_trace(
        "Session 6: Full14 Error Trace (rolling MAE, Annotated)",
        full14_pred["actual"],
        full14_pred["pred"],
        t14,
        out_dir / "session6_full14_error_trace_annotated.png",
        window=200,
        color="#4c78a8",
        gesture_events=gesture_events,
    )

    _plot_error_trace(
        "Session 6: Finger5 Error Trace (rolling MAE)",
        finger5_pred["actual"],
        finger5_pred["pred"],
        t5,
        out_dir / "session6_finger5_error_trace.png",
        window=200,
        color="#f58518",
    )
    _plot_error_trace(
        "Session 6: Finger5 Error Trace (rolling MAE, Annotated)",
        finger5_pred["actual"],
        finger5_pred["pred"],
        t5,
        out_dir / "session6_finger5_error_trace_annotated.png",
        window=200,
        color="#f58518",
        gesture_events=gesture_events,
    )
    _plot_error_trace(
        "Session 6: Index-only Error Trace (rolling MAE)",
        index_pred["actual"],
        index_pred["pred"],
        ti,
        out_dir / "session6_index_error_trace.png",
        window=200,
        color="#54a24b",
    )
    _plot_error_trace(
        "Session 6: Index-only Error Trace (rolling MAE, Annotated)",
        index_pred["actual"],
        index_pred["pred"],
        ti,
        out_dir / "session6_index_error_trace_annotated.png",
        window=200,
        color="#54a24b",
        gesture_events=gesture_events,
    )
    _plot_performance_summary(
        full14_pred,
        full14_keys,
        finger5_pred,
        finger5_keys,
        index_pred,
        index_keys,
        t14,
        t5,
        ti,
        out_dir / "session6_performance_summary.png",
    )
    _plot_task_alignment_confidence(
        full14_pred,
        finger5_pred,
        index_pred,
        gesture_segments,
        t14,
        t5,
        ti,
        out_dir / "session6_task_alignment_confidence.png",
    )

    summary = {
        "session6_npz": str(session6_npz),
        "full14": {
            "mae": full14_pred["mae"],
            "r2": full14_pred["r2"],
            "mae_per_joint": [float(v) for v in full14_pred["mae_per_joint"]],
            "angle_keys": full14_keys,
        },
        "finger5": {
            "mae": finger5_pred["mae"],
            "r2": finger5_pred["r2"],
            "mae_per_joint": [float(v) for v in finger5_pred["mae_per_joint"]],
            "angle_keys": finger5_keys,
        },
        "index_only": {
            "mae": index_pred["mae"],
            "r2": index_pred["r2"],
            "mae_per_joint": [float(v) for v in index_pred["mae_per_joint"]],
            "angle_keys": index_keys,
        },
        "figures": {
            "full14_timeseries": str(out_dir / "session6_full14_timeseries.png"),
            "full14_timeseries_annotated": str(out_dir / "session6_full14_timeseries_annotated.png"),
            "full14_error_trace": str(out_dir / "session6_full14_error_trace.png"),
            "full14_error_trace_annotated": str(out_dir / "session6_full14_error_trace_annotated.png"),
            "finger5_timeseries": str(out_dir / "session6_finger5_timeseries.png"),
            "finger5_timeseries_annotated": str(out_dir / "session6_finger5_timeseries_annotated.png"),
            "index_timeseries": str(out_dir / "session6_index_timeseries.png"),
            "index_timeseries_annotated": str(out_dir / "session6_index_timeseries_annotated.png"),
            "finger5_error_trace": str(out_dir / "session6_finger5_error_trace.png"),
            "finger5_error_trace_annotated": str(out_dir / "session6_finger5_error_trace_annotated.png"),
            "index_error_trace": str(out_dir / "session6_index_error_trace.png"),
            "index_error_trace_annotated": str(out_dir / "session6_index_error_trace_annotated.png"),
            "performance_summary": str(out_dir / "session6_performance_summary.png"),
            "task_alignment_confidence": str(out_dir / "session6_task_alignment_confidence.png"),
        },
        "gesture_event_count": len(gesture_events),
        "gesture_segment_count": len(gesture_segments),
    }
    (out_dir / "session6_comparison_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"[DONE] Full14 MAE={full14_pred['mae']:.4f}, R2={full14_pred['r2']:.4f} | "
        f"Finger5 MAE={finger5_pred['mae']:.4f}, R2={finger5_pred['r2']:.4f} | "
        f"Index MAE={index_pred['mae']:.4f}, R2={index_pred['r2']:.4f}"
    )
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
