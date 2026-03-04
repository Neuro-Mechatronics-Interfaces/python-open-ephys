import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run full replication benchmark on all available Ninapro subjects"
    )
    parser.add_argument(
        "--data_dir",
        default="data/ninapro_db2/processed_npz",
        help="Path to processed .npz files",
    )
    parser.add_argument(
        "--out_dir",
        default="models/ninapro_replication_benchmark",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--python_exe",
        default="C:/Users/NML/Documents/Github/.venv/Scripts/python.exe",
        help="Python interpreter to use",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per subject")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all subjects available in the data dir
    # Filename format: sub-XXX_task-exerciseE1_train.npz
    files = os.listdir(args.data_dir)
    subjects = sorted(
        list(set([f.split("_")[0] for f in files if f.startswith("sub-")]))
    )

    print(f"Found subjects: {subjects}")

    results = {}

    for sub in subjects:
        print(f"\n[BENCHMARK] Processing {sub}...")

        # Define Train/Test files for E1 + E2
        # Check if files exist
        train_files = []
        test_files = []

        for exercise in ["E1", "E2"]:
            f_train = f"{sub}_task-exercise{exercise}_train.npz"
            f_test = f"{sub}_task-exercise{exercise}_test.npz"

            p_train = os.path.join(args.data_dir, f_train)
            p_test = os.path.join(args.data_dir, f_test)

            if os.path.exists(p_train) and os.path.exists(p_test):
                train_files.append(p_train)
                test_files.append(p_test)
            else:
                print(
                    f"  [WARN] Missing files for {sub} {exercise}, skipping exercise."
                )

        if not train_files:
            print(f"  [SKIP] No valid training files for {sub}")
            continue

        subj_out_dir = out_path / sub

        # Construct command
        # python scripts/train_cnn_attention_regressor.py --paper_replication ...
        cmd = [
            args.python_exe,
            "scripts/train_cnn_attention_regressor.py",
            "--paper_replication",
            "--train_data",
            *train_files,
            "--test_data",
            *test_files,
            "--out_dir",
            str(subj_out_dir),
            "--epochs",
            str(args.epochs),
        ]

        try:
            print(f"  Running training for {sub}...")
            # We capture output to avoid spamming the console too much, but print progress
            # actually let's dump to a log file
            log_file = subj_out_dir / "train.log"
            subj_out_dir.mkdir(parents=True, exist_ok=True)

            with open(log_file, "w") as f_log:
                subprocess.check_call(cmd, stdout=f_log, stderr=subprocess.STDOUT)

            # Read metrics.json
            metrics_path = subj_out_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    meta = json.load(f)

                mae = meta.get("mae_test", -1)
                r2 = meta.get("r2_test", -1)
                results[sub] = {"mae": mae, "r2": r2, "status": "success"}
                print(f"  [DONE] {sub}: R2={r2:.4f}, MAE={mae:.4f}")
            else:
                results[sub] = {"status": "failed_no_metrics"}
                print(f"  [FAIL] {sub}: Metrics file not found.")

        except subprocess.CalledProcessError:
            print(f"  [FAIL] Training crashed for {sub}. Check logs.")
            results[sub] = {"status": "crashed"}

    # Save summary
    summary_path = out_path / "summary_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[BENCHMARK COMPLETED]")
    print(f"Results saved to {summary_path}")

    # Calculate average R2 of successful runs
    r2_vals = [d["r2"] for d in results.values() if d.get("status") == "success"]
    if r2_vals:
        print(f"Average R2: {np.mean(r2_vals):.4f}")


if __name__ == "__main__":
    main()
