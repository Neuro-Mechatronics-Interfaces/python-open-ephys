import glob
import os
import subprocess
import sys


def train_sub001():
    # 1. Find Data Files
    # Assuming standard structure data/sub-001/ses-XXX/...
    base_dir = "data/sub-001"
    train_files = []
    test_files = []

    # Sessions 1-5 for training
    for i in range(1, 6):
        pattern = os.path.join(base_dir, f"ses-{i:03d}", "*wrist-neutral.npz")
        found = glob.glob(pattern)
        if found:
            train_files.extend(found)

    # Session 6 for testing
    test_pattern = os.path.join(base_dir, "ses-006", "*wrist-neutral.npz")
    found_test = glob.glob(test_pattern)
    if found_test:
        test_files.extend(found_test)

    print(
        f"Found {len(train_files)} training files and {len(test_files)} testing files."
    )

    if not train_files or not test_files:
        print("Error: Could not find data files.")
        return

    # 2. Construct Training Command
    # Using parameters found to be compatible: input_mode='raw' (since data is (N, C, T, 1))
    # emg_transform='log1p' (to handle Intan range)
    # epochs=25

    cmd = (
        [sys.executable, "scripts/train_cnn_attention_regressor.py", "--train_data"]
        + train_files
        + ["--test_data"]
        + test_files
        + [
            "--out_dir",
            "models/sub-001/result",
            "--epochs",
            "25",
            "--input_mode",
            "raw",
            "--emg_transform",
            "log1p",
            "--arch",
            "baseline",  # Robust 1D CNN
        ]
    )

    print("\nRunning Training Command:")
    print(" ".join(cmd))

    subprocess.run(cmd)


if __name__ == "__main__":
    train_sub001()
