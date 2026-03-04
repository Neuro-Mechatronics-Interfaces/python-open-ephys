import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _seed_stats(seed_items):
    if not seed_items:
        return None
    maes = [float(x["mae_test"]) for x in seed_items]
    r2s = [float(x["r2_test"]) for x in seed_items]
    n = len(seed_items)
    mean_mae = sum(maes) / n
    mean_r2 = sum(r2s) / n
    std_mae = (sum((m - mean_mae) ** 2 for m in maes) / n) ** 0.5
    std_r2 = (sum((r - mean_r2) ** 2 for r in r2s) / n) ** 0.5
    return {
        "count": n,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build consolidated model comparison report"
    )
    parser.add_argument("--root", default="models/sub-001")
    parser.add_argument(
        "--out_json", default="models/sub-001/model_comparison_report.json"
    )
    parser.add_argument(
        "--out_csv", default="models/sub-001/model_comparison_report.csv"
    )
    args = parser.parse_args()

    root = Path(args.root)

    band = _read_json(root / "eval_bandpower_001to005" / "session6_eval.json")
    emd = _read_json(root / "eval_emd_001to005" / "session6_eval_full_chunked.json")

    cnn_primary = _read_json(root / "eval_cnn_attention_001to005" / "metrics.json")
    cnn_seed_7 = _read_json(root / "eval_cnn_attention_001to005_seed7" / "metrics.json")
    cnn_seed_99 = _read_json(
        root / "eval_cnn_attention_001to005_seed99" / "metrics.json"
    )

    p_band = _read_json(
        root / "eval_bandpower_001to005" / "personalization_calib120.json"
    )
    p_band_diag = _read_json(
        root / "eval_bandpower_001to005" / "personalization_calib120_diagonal.json"
    )
    p_emd = _read_json(root / "eval_emd_001to005" / "personalization_calib120.json")
    p_emd_diag = _read_json(
        root / "eval_emd_001to005" / "personalization_calib120_diagonal.json"
    )
    p_cnn = _read_json(
        root / "eval_cnn_attention_001to005" / "personalization_calib120.json"
    )
    p_cnn_diag = _read_json(
        root / "eval_cnn_attention_001to005" / "personalization_calib120_diagonal.json"
    )

    def _best_personalization(*items):
        valid = [x for x in items if x is not None]
        if not valid:
            return None
        return min(valid, key=lambda x: float(x.get("adapted_mae", 1e9)))

    p_band_best = _best_personalization(p_band, p_band_diag)
    p_emd_best = _best_personalization(p_emd, p_emd_diag)
    p_cnn_best = _best_personalization(p_cnn, p_cnn_diag)

    cnn_seeds = [x for x in [cnn_primary, cnn_seed_7, cnn_seed_99] if x is not None]
    cnn_seed_summary = _seed_stats(cnn_seeds)

    report = {
        "bandpower": {
            "session6": band,
            "personalization_calib120": p_band,
            "personalization_calib120_diagonal": p_band_diag,
            "personalization_best": p_band_best,
        },
        "emd": {
            "session6": emd,
            "personalization_calib120": p_emd,
            "personalization_calib120_diagonal": p_emd_diag,
            "personalization_best": p_emd_best,
        },
        "cnn_attention": {
            "session6_primary": cnn_primary,
            "session6_seed_runs": cnn_seeds,
            "session6_seed_summary": cnn_seed_summary,
            "personalization_calib120": p_cnn,
            "personalization_calib120_diagonal": p_cnn_diag,
            "personalization_best": p_cnn_best,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    if band:
        rows.append(
            {
                "model": "bandpower_mlp",
                "n": band.get("n"),
                "mae": band.get("mae"),
                "r2": band.get("r2"),
            }
        )
    if emd:
        rows.append(
            {
                "model": "emd_mlp",
                "n": emd.get("n"),
                "mae": emd.get("mae"),
                "r2": emd.get("r2"),
            }
        )
    if cnn_primary:
        rows.append(
            {
                "model": "cnn_attention_seed42",
                "n": cnn_primary.get("n_test"),
                "mae": cnn_primary.get("mae_test"),
                "r2": cnn_primary.get("r2_test"),
            }
        )
    if cnn_seed_summary:
        rows.append(
            {
                "model": "cnn_attention_seed_mean",
                "n": cnn_primary.get("n_test") if cnn_primary else None,
                "mae": cnn_seed_summary.get("mean_mae"),
                "r2": cnn_seed_summary.get("mean_r2"),
                "mae_std": cnn_seed_summary.get("std_mae"),
                "r2_std": cnn_seed_summary.get("std_r2"),
            }
        )

    if p_band:
        rows.append(
            {
                "model": "bandpower_personalized_calib120",
                "n": p_band.get("val_samples"),
                "mae": p_band.get("adapted_mae"),
                "r2": p_band.get("adapted_r2"),
            }
        )
    if p_band_diag:
        rows.append(
            {
                "model": "bandpower_personalized_calib120_diagonal",
                "n": p_band_diag.get("val_samples"),
                "mae": p_band_diag.get("adapted_mae"),
                "r2": p_band_diag.get("adapted_r2"),
            }
        )
    if p_emd:
        rows.append(
            {
                "model": "emd_personalized_calib120",
                "n": p_emd.get("val_samples"),
                "mae": p_emd.get("adapted_mae"),
                "r2": p_emd.get("adapted_r2"),
            }
        )
    if p_emd_diag:
        rows.append(
            {
                "model": "emd_personalized_calib120_diagonal",
                "n": p_emd_diag.get("val_samples"),
                "mae": p_emd_diag.get("adapted_mae"),
                "r2": p_emd_diag.get("adapted_r2"),
            }
        )
    if p_cnn:
        rows.append(
            {
                "model": "cnn_attention_personalized_calib120",
                "n": p_cnn.get("val_samples"),
                "mae": p_cnn.get("adapted_mae"),
                "r2": p_cnn.get("adapted_r2"),
            }
        )
    if p_cnn_diag:
        rows.append(
            {
                "model": "cnn_attention_personalized_calib120_diagonal",
                "n": p_cnn_diag.get("val_samples"),
                "mae": p_cnn_diag.get("adapted_mae"),
                "r2": p_cnn_diag.get("adapted_r2"),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "n", "mae", "r2", "mae_std", "r2_std"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")

    print(f"[DONE] Report JSON: {out_json}")
    print(f"[DONE] Report CSV:  {out_csv}")


if __name__ == "__main__":
    main()
