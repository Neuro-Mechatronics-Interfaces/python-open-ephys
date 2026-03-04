import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_session6_model_comparison as cmp


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _plot_model_error_bars(pooled_rows, out_path: Path):
    models = [r["model"] for r in pooled_rows]
    mae = np.array([_to_float(r.get("mae")) for r in pooled_rows], dtype=np.float64)
    rmse = np.array([_to_float(r.get("rmse")) for r in pooled_rows], dtype=np.float64)

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, mae, width=w, label="MAE")
    b2 = ax.bar(x + w / 2, rmse, width=w, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error")
    ax.set_title("Session-6 model error comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_tail_error_bars(pooled_rows, out_path: Path):
    models = [r["model"] for r in pooled_rows]
    p90 = np.array([_to_float(r.get("p90_abs_err")) for r in pooled_rows], dtype=np.float64)
    p95 = np.array([_to_float(r.get("p95_abs_err")) for r in pooled_rows], dtype=np.float64)

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, p90, width=w, label="P90 abs error")
    b2 = ax.bar(x + w / 2, p95, width=w, label="P95 abs error")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Absolute error")
    ax.set_title("Session-6 tail-error comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _aggregate_gesture_mae(rows):
    totals = {}
    counts = {}
    for r in rows:
        gesture = str(r.get("gesture", ""))
        if not gesture or gesture.lower() == "rest":
            continue
        mae = _to_float(r.get("mae"), default=np.nan)
        n = int(_to_float(r.get("n_samples"), default=0))
        if not np.isfinite(mae) or n <= 0:
            continue
        totals[gesture] = totals.get(gesture, 0.0) + mae * n
        counts[gesture] = counts.get(gesture, 0) + n
    out = {}
    for g, total in totals.items():
        c = counts.get(g, 0)
        if c > 0:
            out[g] = total / c
    return out


def _plot_gesture_mae_by_model(metrics_dir: Path, out_path: Path, max_gestures=10):
    files = {
        "full14": metrics_dir / "full14_per_gesture_metrics.csv",
        "finger5": metrics_dir / "finger5_per_gesture_metrics.csv",
        "index_only": metrics_dir / "index_only_per_gesture_metrics.csv",
    }

    per_model = {}
    for model, fpath in files.items():
        rows = _read_csv_rows(fpath)
        per_model[model] = _aggregate_gesture_mae(rows)

    all_gestures = sorted(set().union(*[set(v.keys()) for v in per_model.values()]))
    if not all_gestures:
        return

    full14_vals = per_model["full14"]
    ranked = sorted(all_gestures, key=lambda g: full14_vals.get(g, 0.0), reverse=True)
    selected = ranked[:max_gestures]

    x = np.arange(len(selected))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(11, 1.1 * len(selected)), 5.5))
    colors = {"full14": "#4c78a8", "finger5": "#f58518", "index_only": "#54a24b"}

    for idx, model in enumerate(["full14", "finger5", "index_only"]):
        vals = [per_model[model].get(g, np.nan) for g in selected]
        offset = (idx - 1) * w
        ax.bar(x + offset, vals, width=w, label=model, color=colors[model])

    ax.set_xticks(x)
    ax.set_xticklabels(selected, rotation=35, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Gesture-level MAE by model (top full14-error gestures)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_full14_joint_mae(metrics_dir: Path, out_path: Path, top_k=10):
    rows = _read_csv_rows(metrics_dir / "full14_per_joint_metrics.csv")
    tuples = []
    for r in rows:
        joint = str(r.get("joint", ""))
        mae = _to_float(r.get("mae"), default=np.nan)
        if joint and np.isfinite(mae):
            tuples.append((joint, mae))
    tuples.sort(key=lambda t: t[1], reverse=True)
    tuples = tuples[:top_k]
    if not tuples:
        return

    joints = [t[0] for t in tuples][::-1]
    maes = [t[1] for t in tuples][::-1]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(joints) + 2.2))
    bars = ax.barh(joints, maes, color="#e45756")
    ax.set_xlabel("MAE")
    ax.set_title("Full14 hardest joints (highest MAE)")
    ax.grid(axis="x", alpha=0.25)
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.15, b.get_y() + b.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_full14_lag_vs_mae(metrics_dir: Path, out_path: Path):
    rows = _read_csv_rows(metrics_dir / "full14_per_joint_metrics.csv")
    joints, mae, lag = [], [], []
    for r in rows:
        joint = str(r.get("joint", ""))
        m = _to_float(r.get("mae"), default=np.nan)
        l = _to_float(r.get("lag_samples"), default=np.nan)
        if joint and np.isfinite(m) and np.isfinite(l):
            joints.append(joint)
            mae.append(m)
            lag.append(abs(l))

    if not joints:
        return

    mae = np.asarray(mae, dtype=np.float64)
    lag = np.asarray(lag, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(lag, mae, s=45, alpha=0.85)
    for x, y, j in zip(lag, mae, joints):
        ax.text(x + 0.2, y + 0.05, j, fontsize=8)
    ax.set_xlabel("Absolute lag (samples)")
    ax.set_ylabel("MAE")
    ax.set_title("Full14 per-joint timing lag vs error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _resolve_time_axis(timestamps, mask, session_start):
    t = np.asarray(timestamps, dtype=np.float64)[mask]
    if session_start is not None and np.isfinite(session_start):
        return t - session_start
    return t - t[0]


def _prepare_predictions(session6_npz: Path, full14_model_dir: Path, full14_feature_extractor: Path, finger5_model_dir: Path, finger5_feature_extractor: Path, index_model_dir: Path, index_feature_extractor: Path):
    data = np.load(session6_npz, allow_pickle=True)
    emg = data["emg"]
    all_angles = np.asarray(data["angles"], dtype=np.float32)
    all_keys = data["angle_keys"].tolist() if "angle_keys" in data else []
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)

    session_start = None
    if "session_start" in data.files:
        try:
            session_start = float(np.asarray(data["session_start"]).reshape(-1)[0])
        except Exception:
            session_start = None

    full14_bundle = cmp._load_model_bundle(full14_model_dir, full14_feature_extractor)
    finger5_bundle = cmp._load_model_bundle(finger5_model_dir, finger5_feature_extractor)
    index_bundle = cmp._load_model_bundle(index_model_dir, index_feature_extractor)

    full14_keys = full14_bundle["metrics"].get("angle_keys") or all_keys
    finger5_keys = finger5_bundle["metrics"].get("angle_keys") or ["thumb_cmc_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
    index_keys = index_bundle["metrics"].get("angle_keys") or ["index_mcp", "index_pip", "index_dip"]

    full14_actual = cmp._align_actual_angles(all_angles, all_keys, full14_keys)
    finger5_actual = cmp._align_actual_angles(all_angles, all_keys, finger5_keys)
    index_actual = cmp._align_actual_angles(all_angles, all_keys, index_keys)

    full14_pred = cmp._predict(emg, full14_actual, full14_bundle)
    finger5_pred = cmp._predict(emg, finger5_actual, finger5_bundle)
    index_pred = cmp._predict(emg, index_actual, index_bundle)

    t14 = _resolve_time_axis(timestamps, full14_pred["mask"], session_start)
    t5 = _resolve_time_axis(timestamps, finger5_pred["mask"], session_start)
    ti = _resolve_time_axis(timestamps, index_pred["mask"], session_start)

    return {
        "full14": {"pred": full14_pred, "t": t14},
        "finger5": {"pred": finger5_pred, "t": t5},
        "index_only": {"pred": index_pred, "t": ti},
    }


def _plot_error_cdf(predictions, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"full14": "#4c78a8", "finger5": "#f58518", "index_only": "#54a24b"}
    labels = {"full14": "Full14", "finger5": "Finger5", "index_only": "Index"}
    for key in ["full14", "finger5", "index_only"]:
        p = predictions[key]["pred"]
        abs_err = np.abs(p["actual"] - p["pred"]).reshape(-1)
        abs_err = abs_err[np.isfinite(abs_err)]
        if abs_err.size == 0:
            continue
        x = np.sort(abs_err)
        y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
        ax.plot(x, y, linewidth=2.0, color=colors[key], label=labels[key])
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("CDF")
    ax.set_title("Error distribution (CDF)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_bland_altman(predictions, out_path: Path, max_points=5000):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = {"full14": "#4c78a8", "finger5": "#f58518", "index_only": "#54a24b"}
    titles = {"full14": "Full14", "finger5": "Finger5", "index_only": "Index"}
    rng = np.random.default_rng(7)

    for ax, key in zip(axes, ["full14", "finger5", "index_only"]):
        p = predictions[key]["pred"]
        actual = p["actual"].reshape(-1)
        pred = p["pred"].reshape(-1)
        mask = np.isfinite(actual) & np.isfinite(pred)
        actual = actual[mask]
        pred = pred[mask]
        if actual.size == 0:
            ax.set_title(f"{titles[key]} (no data)")
            continue
        mean_v = 0.5 * (actual + pred)
        diff = pred - actual
        if mean_v.size > max_points:
            idx = rng.choice(mean_v.size, size=max_points, replace=False)
            mean_v = mean_v[idx]
            diff = diff[idx]
        mu = float(np.mean(diff))
        sd = float(np.std(diff))
        loa_hi = mu + 1.96 * sd
        loa_lo = mu - 1.96 * sd
        ax.scatter(mean_v, diff, s=6, alpha=0.2, color=colors[key])
        ax.axhline(mu, color="black", linestyle="-", linewidth=1)
        ax.axhline(loa_hi, color="black", linestyle="--", linewidth=1)
        ax.axhline(loa_lo, color="black", linestyle="--", linewidth=1)
        ax.set_title(titles[key])
        ax.set_xlabel("Mean(actual, pred)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Pred - Actual")
    fig.suptitle("Bland–Altman agreement plots")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_error_drift(predictions, out_path: Path, window=200):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"full14": "#4c78a8", "finger5": "#f58518", "index_only": "#54a24b"}
    labels = {"full14": "Full14", "finger5": "Finger5", "index_only": "Index"}
    for key in ["full14", "finger5", "index_only"]:
        p = predictions[key]["pred"]
        t = predictions[key]["t"]
        x, y = cmp._rolling_mae_trace(p["actual"], p["pred"], window=window)
        t_roll = t[np.clip(x, 0, max(0, t.size - 1))] if t.size else x
        ax.plot(t_roll, y, color=colors[key], linewidth=1.6, alpha=0.9, label=labels[key])
        if np.asarray(t_roll).size >= 2:
            coeff = np.polyfit(t_roll, y, deg=1)
            yhat = np.polyval(coeff, t_roll)
            ax.plot(t_roll, yhat, color=colors[key], linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rolling MAE")
    ax.set_title("Error drift over time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build session-6 Q&A figure pack from metrics CSV files.")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="models/sub-001/session6_metrics_report",
        help="Directory containing pooled/per-joint/per-gesture metrics CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="models/sub-001/session6_comparison_figs/qa_figure_pack",
        help="Output directory for generated QA figures.",
    )
    parser.add_argument("--max_gestures", type=int, default=10, help="Max gestures shown in gesture MAE chart.")
    parser.add_argument("--top_joints", type=int, default=10, help="Top full14 joints in hardest-joints chart.")
    parser.add_argument("--session6_npz", type=str, default="data/sub-001/ses-006/sub-001_ses-006_task-jointangles_wrist-neutral.npz")
    parser.add_argument("--full14_model_dir", type=str, default="models/sub-001/ses-001to005_regressor")
    parser.add_argument("--full14_feature_extractor", type=str, default="models/sub-001/ses-001to005_feature_extractor.h5")
    parser.add_argument("--finger5_model_dir", type=str, default="models/sub-001/finger5_ses-001to005_regressor")
    parser.add_argument("--finger5_feature_extractor", type=str, default="models/sub-001/finger5_ses-001to005_feature_extractor.h5")
    parser.add_argument("--index_model_dir", type=str, default="models/sub-001/index-only_ses-001to005_regressor")
    parser.add_argument("--index_feature_extractor", type=str, default="models/sub-001/index-only_ses-001to005_feature_extractor.h5")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows = _read_csv_rows(metrics_dir / "pooled_metrics.csv")

    _plot_model_error_bars(pooled_rows, out_dir / "qa_model_mae_rmse_bar.png")
    _plot_tail_error_bars(pooled_rows, out_dir / "qa_model_p90_p95_bar.png")
    _plot_gesture_mae_by_model(metrics_dir, out_dir / "qa_gesture_mae_by_model.png", max_gestures=args.max_gestures)
    _plot_full14_joint_mae(metrics_dir, out_dir / "qa_full14_hardest_joints_barh.png", top_k=args.top_joints)
    _plot_full14_lag_vs_mae(metrics_dir, out_dir / "qa_full14_lag_vs_mae_scatter.png")

    preds = _prepare_predictions(
        session6_npz=Path(args.session6_npz),
        full14_model_dir=Path(args.full14_model_dir),
        full14_feature_extractor=Path(args.full14_feature_extractor),
        finger5_model_dir=Path(args.finger5_model_dir),
        finger5_feature_extractor=Path(args.finger5_feature_extractor),
        index_model_dir=Path(args.index_model_dir),
        index_feature_extractor=Path(args.index_feature_extractor),
    )
    _plot_error_cdf(preds, out_dir / "qa_error_cdf.png")
    _plot_bland_altman(preds, out_dir / "qa_bland_altman.png")
    _plot_error_drift(preds, out_dir / "qa_error_drift_over_time.png")

    print("[DONE] QA figure pack generated")
    print(f"[OUT]  {out_dir}")


if __name__ == "__main__":
    main()