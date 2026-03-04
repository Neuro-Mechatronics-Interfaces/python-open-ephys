import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import train_regressor as tr


def _session_id_from_path(path: Path):
    parts = path.as_posix().split("/")
    for part in parts:
        if part.startswith("ses-"):
            return part
    return path.stem


def _label_to_phase(label: str):
    l = str(label).lower()
    if "rest" in l:
        return "rest"
    if "flex" in l:
        return "flexion"
    if "extend" in l:
        return "extension"
    if "trajectory" in l:
        return "trajectory"
    if "coordinated_grasp" in l or "grasp" in l:
        return "grasp"
    if "free_movement" in l or "free" in l:
        return "free"
    return "other"


def _extract_phase_vector(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    n = int(np.asarray(data["angles"]).shape[0])
    if "plan_json" not in data.files:
        return np.asarray(["unknown"] * n, dtype=object)
    try:
        plan = json.loads(str(data["plan_json"]))
    except Exception:
        return np.asarray(["unknown"] * n, dtype=object)
    if not isinstance(plan, list):
        return np.asarray(["unknown"] * n, dtype=object)

    durations = []
    labels = []
    for item in plan:
        labels.append(str(item.get("label", "unknown")))
        try:
            durations.append(float(item.get("duration", 0.0)))
        except Exception:
            durations.append(0.0)

    total = float(np.sum(np.maximum(0.0, np.asarray(durations, dtype=np.float64))))
    if total <= 0:
        return np.asarray(["unknown"] * n, dtype=object)

    t = np.linspace(0.0, total, n, endpoint=False)
    edges = np.cumsum(np.maximum(0.0, np.asarray(durations, dtype=np.float64)))
    starts = np.concatenate([[0.0], edges[:-1]])

    out = np.empty(n, dtype=object)
    for i, ti in enumerate(t):
        idx = int(np.searchsorted(edges, ti, side="right"))
        idx = min(max(idx, 0), len(labels) - 1)
        out[i] = _label_to_phase(labels[idx])
    return out


def _load_single(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    emg = np.asarray(d["emg"], dtype=np.float32)
    angles = np.asarray(d["angles"], dtype=np.float32)
    return emg, angles


def _fit_regressor(train_paths, feature_extractor, emg_transform="none", emg_features="raw_flat", angle_scaler="minmax", max_iter=500):
    emg, angles, _, _ = tr._load_datasets([Path(p) for p in train_paths])
    emg = tr._normalize_emg(emg)
    angles = np.asarray(angles, dtype=np.float32)
    mask = np.isfinite(angles).all(axis=1) & np.isfinite(emg).all(axis=(1, 2, 3))
    emg = emg[mask]
    angles = angles[mask]

    extractor = tr.load_feature_extractor(str(feature_extractor))
    feats = tr.extract_features(emg, extractor, feature_mode=emg_features, emg_transform=emg_transform)
    scaler = tr.StandardScaler()
    feats_s = scaler.fit_transform(feats)

    target_scaler = None
    y = angles
    if angle_scaler == "minmax":
        target_scaler = tr.MinMaxScaler()
        y = target_scaler.fit_transform(y)
    elif angle_scaler == "standard":
        target_scaler = tr.StandardScaler()
        y = target_scaler.fit_transform(y)

    reg = tr.MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    reg.fit(feats_s, y)
    return {"extractor": extractor, "scaler": scaler, "target_scaler": target_scaler, "reg": reg}


def _predict_bundle(bundle, npz_path: Path):
    emg, angles = _load_single(npz_path)
    emg = tr._normalize_emg(emg)
    mask = np.isfinite(angles).all(axis=1) & np.isfinite(emg).all(axis=(1, 2, 3))
    emg = emg[mask]
    y = angles[mask]

    feats = tr.extract_features(emg, bundle["extractor"], feature_mode="raw_flat", emg_transform="none")
    feats_s = bundle["scaler"].transform(feats)
    pred = bundle["reg"].predict(feats_s)
    if bundle["target_scaler"] is not None:
        pred = bundle["target_scaler"].inverse_transform(pred)

    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred, multioutput="variance_weighted"))
    phase = _extract_phase_vector(npz_path)[mask]
    return {"mae": mae, "r2": r2, "actual": y, "pred": pred, "phase": phase}


def _bootstrap_ci_mae(y_true, y_pred, n_boot=500, seed=7):
    rng = np.random.default_rng(seed)
    err = np.mean(np.abs(y_true - y_pred), axis=1)
    n = err.shape[0]
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(err[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_permutation_p(y_true_a, y_pred_a, y_true_b, y_pred_b, n_perm=2000, seed=11):
    rng = np.random.default_rng(seed)
    ea = np.mean(np.abs(y_true_a - y_pred_a), axis=1)
    eb = np.mean(np.abs(y_true_b - y_pred_b), axis=1)
    n = min(ea.size, eb.size)
    ea = ea[:n]
    eb = eb[:n]
    obs = float(np.mean(ea - eb))
    count = 0
    for _ in range(n_perm):
        sign = rng.choice([-1.0, 1.0], size=n)
        d = (ea - eb) * sign
        if abs(np.mean(d)) >= abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1)), obs


def _save_matrix_heatmap(matrix, labels, out_path: Path, title: str, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test session")
    ax.set_ylabel("Train-heldout session")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if np.isfinite(v) and v > np.nanmean(matrix) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _save_phase_plot(phase_rows, out_path: Path):
    phases = sorted(set(r["phase"] for r in phase_rows))
    models = sorted(set(r["model"] for r in phase_rows))
    x = np.arange(len(phases))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"full14": "#4c78a8", "finger5": "#f58518", "index_only": "#54a24b"}
    for i, m in enumerate(models):
        vals = []
        for p in phases:
            rows = [r for r in phase_rows if r["model"] == m and r["phase"] == p]
            vals.append(np.mean([r["mae"] for r in rows]) if rows else np.nan)
        ax.bar(x + (i - (len(models)-1)/2)*w, vals, width=w, label=m, color=colors.get(m, None))
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Phase-specific error breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _get_session_paths(base_dir: Path):
    out = sorted(base_dir.glob("ses-*/**/*.npz"))
    if not out:
        out = sorted(base_dir.glob("ses-*/*.npz"))
    return out


def main():
    parser = argparse.ArgumentParser(description="Cross-session generalization + CI/significance + phase breakdown report.")
    parser.add_argument("--full14_dir", default="data/sub-001")
    parser.add_argument("--finger5_dir", default="data/sub-001_finger5")
    parser.add_argument("--index_dir", default="data/sub-001_index_only")
    parser.add_argument("--full14_extractor", default="models/sub-001/ses-001to005_feature_extractor.h5")
    parser.add_argument("--finger5_extractor", default="models/sub-001/finger5_ses-001to005_feature_extractor.h5")
    parser.add_argument("--index_extractor", default="models/sub-001/index-only_ses-001to005_feature_extractor.h5")
    parser.add_argument("--out_dir", default="models/sub-001/session_generalization_report")
    parser.add_argument("--max_iter", type=int, default=450)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "full14": (Path(args.full14_dir), Path(args.full14_extractor)),
        "finger5": (Path(args.finger5_dir), Path(args.finger5_extractor)),
        "index_only": (Path(args.index_dir), Path(args.index_extractor)),
    }

    summary = {"models": {}, "significance": []}
    phase_rows = []
    heldout_predictions = {}

    for model_name, (data_dir, extractor_path) in cfg.items():
        paths = _get_session_paths(data_dir)
        sessions = [_session_id_from_path(p) for p in paths]
        n = len(paths)
        mae_mat = np.full((n, n), np.nan, dtype=np.float64)
        r2_mat = np.full((n, n), np.nan, dtype=np.float64)

        heldout_preds = {}

        for i, held in enumerate(paths):
            train = [p for p in paths if p != held]
            bundle = _fit_regressor(
                train_paths=train,
                feature_extractor=extractor_path,
                emg_transform="none",
                emg_features="raw_flat",
                angle_scaler="minmax",
                max_iter=args.max_iter,
            )
            for j, testp in enumerate(paths):
                pred = _predict_bundle(bundle, testp)
                mae_mat[i, j] = pred["mae"]
                r2_mat[i, j] = pred["r2"]
                if i == j:
                    heldout_preds[sessions[i]] = pred
                    phases = pred["phase"]
                    per_samp = np.mean(np.abs(pred["actual"] - pred["pred"]), axis=1)
                    for ph in np.unique(phases):
                        m = phases == ph
                        if np.any(m):
                            phase_rows.append({
                                "model": model_name,
                                "session": sessions[i],
                                "phase": str(ph),
                                "mae": float(np.mean(per_samp[m])),
                                "n": int(np.sum(m)),
                            })

        heldout_predictions[model_name] = heldout_preds

        _save_matrix_heatmap(mae_mat, sessions, out_dir / f"{model_name}_cross_session_mae_heatmap.png", f"{model_name}: Cross-session MAE")
        _save_matrix_heatmap(r2_mat, sessions, out_dir / f"{model_name}_cross_session_r2_heatmap.png", f"{model_name}: Cross-session R²", cmap="coolwarm")

        np.savetxt(out_dir / f"{model_name}_cross_session_mae.csv", mae_mat, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"{model_name}_cross_session_r2.csv", r2_mat, delimiter=",", fmt="%.6f")

        # CI from held-out diagonal pooled samples
        y_true = np.concatenate([heldout_preds[s]["actual"] for s in sessions], axis=0)
        y_pred = np.concatenate([heldout_preds[s]["pred"] for s in sessions], axis=0)
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
        ci_lo, ci_hi = _bootstrap_ci_mae(y_true, y_pred, n_boot=500, seed=13)

        summary["models"][model_name] = {
            "sessions": sessions,
            "pooled_heldout_mae": mae,
            "pooled_heldout_r2": r2,
            "mae_ci95": [ci_lo, ci_hi],
        }

    # paired significance tests on pooled held-out predictions
    pairs = [("full14", "finger5"), ("full14", "index_only"), ("finger5", "index_only")]
    all_sessions = sorted(set(heldout_predictions["full14"].keys()))
    for a, b in pairs:
        ya = np.concatenate([heldout_predictions[a][s]["actual"] for s in all_sessions], axis=0)
        pa = np.concatenate([heldout_predictions[a][s]["pred"] for s in all_sessions], axis=0)
        yb = np.concatenate([heldout_predictions[b][s]["actual"] for s in all_sessions], axis=0)
        pb = np.concatenate([heldout_predictions[b][s]["pred"] for s in all_sessions], axis=0)
        pval, diff = _paired_permutation_p(ya, pa, yb, pb, n_perm=2000, seed=17)
        summary["significance"].append({
            "model_a": a,
            "model_b": b,
            "delta_mae_a_minus_b": float(diff),
            "paired_permutation_p": float(pval),
        })

    _save_phase_plot(phase_rows, out_dir / "phase_specific_mae.png")

    # Save phase CSV
    phase_csv = out_dir / "phase_specific_mae.csv"
    with phase_csv.open("w", encoding="utf-8") as f:
        f.write("model,session,phase,mae,n\n")
        for r in phase_rows:
            f.write(f"{r['model']},{r['session']},{r['phase']},{r['mae']:.6f},{r['n']}\n")

    (out_dir / "session_generalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] Session generalization report complete")
    print(f"[OUT]  {out_dir}")


if __name__ == "__main__":
    main()