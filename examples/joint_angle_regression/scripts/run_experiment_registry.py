import argparse
import csv
import fnmatch
import hashlib
import json
from pathlib import Path

from evaluate_model_metrics import evaluate


def _is_model_dir(path: Path) -> bool:
    has_metrics = (path / "metrics.json").exists()
    has_model = (path / "mlp_regressor.pkl").exists() or (
        path / "cnn_attention_regressor.h5"
    ).exists()
    return bool(has_metrics and has_model)


def _discover_model_dirs(root: Path):
    out = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        model_dir = metrics_path.parent
        if _is_model_dir(model_dir):
            out.append(model_dir)
    return out


def _discover_personalizations(model_dir: Path, include_none: bool = True):
    items = []
    if include_none:
        items.append(None)
    files = sorted(model_dir.glob("personalization*.pkl"))
    seen = set()
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            items.append(p)
    return items


def _approach_id(row: dict):
    signature = "|".join(
        [
            str(row.get("model_dir_name", "")),
            str(row.get("model_type", "")),
            str(row.get("feature_mode", "")),
            str(row.get("emg_transform", "")),
            str(row.get("use_imu", "")),
            str(row.get("personalization_name", "none")),
            str(row.get("smooth_samples", 0)),
            str(row.get("chunk_size", 0)),
        ]
    )
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def _checkpoint_payload(root, npz_path, smooth_values, rows, failures):
    rows_sorted = sorted(rows, key=lambda x: float(x.get("r2", -1e9)), reverse=True)
    best_overall_r2 = rows_sorted[0] if rows_sorted else None
    best_overall_mae = (
        min(rows, key=lambda x: float(x.get("mae", 1e9))) if rows else None
    )

    best_by_model = {}
    for r in rows:
        key = r.get("model_dir_name", "unknown")
        prev = best_by_model.get(key)
        if prev is None or float(r.get("r2", -1e9)) > float(prev.get("r2", -1e9)):
            best_by_model[key] = r

    unique_feature_sets = sorted(
        {
            (
                str(r.get("model_type", "")),
                str(r.get("feature_mode", "")),
                str(r.get("emg_transform", "")),
                bool(r.get("use_imu", False)),
                str(r.get("personalization_name", "none")),
                int(r.get("smooth_samples", 0)),
            )
            for r in rows
        }
    )

    payload = {
        "root": str(root),
        "npz": str(npz_path),
        "smooth_values": smooth_values,
        "rows": rows,
        "failures": failures,
        "best_overall_r2": best_overall_r2,
        "best_overall_mae": best_overall_mae,
        "best_by_model": best_by_model,
        "unique_feature_sets": [
            {
                "model_type": t[0],
                "feature_mode": t[1],
                "emg_transform": t[2],
                "use_imu": t[3],
                "personalization_name": t[4],
                "smooth_samples": t[5],
            }
            for t in unique_feature_sets
        ],
    }
    return payload


def _write_outputs(out_json: Path, out_csv: Path, payload: dict):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload.get("rows", [])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_keys = [
        "approach_id",
        "model_dir_name",
        "model_type",
        "feature_mode",
        "emg_transform",
        "use_imu",
        "personalization_name",
        "smooth_samples",
        "mae",
        "r2",
        "n",
        "elapsed_sec",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in csv_keys})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and log experiment registry with unique configuration + metrics rows"
    )
    parser.add_argument("--root", default="models/sub-001")
    parser.add_argument("--npz", required=True)
    parser.add_argument(
        "--model_glob", default="*", help="Model directory name glob under --root"
    )
    parser.add_argument("--smooth_list", default="0,5,9,15,25")
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_json", default="models/sub-001/experiment_registry.json")
    parser.add_argument("--out_csv", default="models/sub-001/experiment_registry.csv")
    args = parser.parse_args()

    root = Path(args.root)
    npz_path = Path(args.npz)
    smooth_values = [
        int(x.strip()) for x in str(args.smooth_list).split(",") if x.strip()
    ]
    max_windows = args.max_windows if int(args.max_windows) > 0 else None
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None

    model_dirs = [d for d in _discover_model_dirs(root) if d.name]
    if args.model_glob and args.model_glob != "*":
        pattern = str(args.model_glob)
        model_dirs = [
            d
            for d in model_dirs
            if fnmatch.fnmatch(d.name, pattern)
            or fnmatch.fnmatch(d.name, f"*{pattern}*")
        ]

    print(f"[INFO] discovered_model_dirs={len(model_dirs)}")

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    rows = []
    failures = []
    done_keys = set()
    if args.resume and out_json.exists():
        try:
            prior = json.loads(out_json.read_text(encoding="utf-8"))
            rows = list(prior.get("rows", []))
            failures = list(prior.get("failures", []))
            for r in rows:
                done_keys.add(
                    (
                        str(r.get("model_dir_name", "")),
                        str(r.get("personalization_name", "none")),
                        int(r.get("smooth_samples", 0)),
                    )
                )
            print(f"[INFO] resume loaded rows={len(rows)} failures={len(failures)}")
        except Exception as exc:
            print(f"[WARN] resume ignored (failed to parse existing output): {exc}")

    try:
        for model_dir in model_dirs:
            personalization_opts = _discover_personalizations(
                model_dir, include_none=True
            )
            cache_path = None
            if feature_cache_dir is not None:
                feature_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = feature_cache_dir / f"{model_dir.name}_features.npz"
            for personalization_path in personalization_opts:
                p_name = (
                    personalization_path.name
                    if personalization_path is not None
                    else "none"
                )
                for smooth in smooth_values:
                    key = (model_dir.name, p_name, int(smooth))
                    if key in done_keys:
                        continue
                    try:
                        result = evaluate(
                            model_dir=model_dir,
                            npz_path=npz_path,
                            max_windows=max_windows,
                            chunk_size=int(args.chunk_size),
                            feature_cache=cache_path,
                            personalization_path=personalization_path,
                            smooth_samples=int(smooth),
                        )
                        row = dict(result)
                        row["model_dir_name"] = model_dir.name
                        row["personalization_name"] = p_name
                        row["approach_id"] = _approach_id(row)
                        rows.append(row)
                        done_keys.add(key)
                        print(
                            f"[OK] {model_dir.name} | p={row['personalization_name']} | s={smooth} "
                            f"| MAE={row['mae']:.4f} R2={row['r2']:.4f}"
                        )
                    except Exception as exc:
                        failure = {
                            "model_dir": str(model_dir),
                            "personalization": str(personalization_path)
                            if personalization_path
                            else "none",
                            "smooth_samples": int(smooth),
                            "error": str(exc),
                        }
                        failures.append(failure)
                        print(
                            f"[FAIL] {model_dir.name} | p={failure['personalization']} "
                            f"| s={smooth} | {failure['error']}"
                        )

                    if len(rows) % 10 == 0:
                        payload = _checkpoint_payload(
                            root, npz_path, smooth_values, rows, failures
                        )
                        _write_outputs(out_json, out_csv, payload)
    except KeyboardInterrupt:
        print("[WARN] Interrupted. Writing partial registry.")

    payload = _checkpoint_payload(root, npz_path, smooth_values, rows, failures)
    _write_outputs(out_json, out_csv, payload)

    print(f"[DONE] rows={len(rows)} failures={len(failures)}")
    best_overall_r2 = payload.get("best_overall_r2")
    if best_overall_r2 is not None:
        print(
            f"[BEST_R2] {best_overall_r2.get('model_dir_name')} "
            f"p={best_overall_r2.get('personalization_name')} "
            f"s={best_overall_r2.get('smooth_samples')} "
            f"R2={float(best_overall_r2.get('r2')):.4f} MAE={float(best_overall_r2.get('mae')):.4f}"
        )
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_csv}")


if __name__ == "__main__":
    main()
