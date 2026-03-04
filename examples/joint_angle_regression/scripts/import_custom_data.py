import argparse
import os

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def load_data(file_path, variable_name=None):
    """
    Load data from .mat, .csv, or .npy files.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".mat":
        mat = scipy.io.loadmat(file_path)
        if variable_name:
            if variable_name in mat:
                return mat[variable_name]
            else:
                raise ValueError(
                    f"Variable '{variable_name}' not found in {file_path}. Available keys: {list(mat.keys())}"
                )
        else:
            # Try to guess: look for 'emg', 'data', or the largest array
            for key in ["emg", "data", "channels"]:
                if key in mat:
                    print(f"Auto-detected variable '{key}'")
                    return mat[key]
            # Fallback: largest array
            largest_key = max(
                mat.keys(),
                key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0,
            )
            print(f"Auto-detected variable '{largest_key}'")
            return mat[largest_key]

    elif ext == ".csv":
        # Assume generic CSV: Time x Channels
        df = pd.read_csv(file_path)
        return df.values

    elif ext == ".npy":
        return np.load(file_path)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def process_and_save(
    emg_file, kin_file, output_file, fs_emg, fs_kin=None, emg_var=None, kin_var=None
):
    print(f"Loading EMG from {emg_file}...")
    raw_emg = load_data(emg_file, emg_var)

    print(f"Loading Kinematics from {kin_file}...")
    raw_kin = load_data(kin_file, kin_var)

    # Check dimensions
    # Standard: (Time, Channels)
    if raw_emg.shape[0] < raw_emg.shape[1]:
        print("Warning: EMG shape (Channels, Time)? Transposing to (Time, Channels)...")
        raw_emg = raw_emg.T

    if raw_kin.shape[0] < raw_kin.shape[1]:
        print(
            "Warning: Kinematics shape (Channels, Time)? Transposing to (Time, Channels)..."
        )
        raw_kin = raw_kin.T

    n_emg = raw_emg.shape[0]
    n_kin = raw_kin.shape[0]

    # Resample Kinematics to match EMG
    if n_emg != n_kin:
        print(f"Resampling Kinematics from {n_kin} to {n_emg} samples...")
        # Use scipy.signal.resample for Fourier method or interp for linear
        # Linear is safer for trajectories
        x_kin = np.linspace(0, 1, n_kin)
        x_emg = np.linspace(0, 1, n_emg)

        new_kin = np.zeros((n_emg, raw_kin.shape[1]))
        for i in range(raw_kin.shape[1]):
            new_kin[:, i] = np.interp(x_emg, x_kin, raw_kin[:, i])
        kin_data = new_kin
    else:
        kin_data = raw_kin

    # Filter EMG (Bandpass 20-500Hz)
    print("Filtering EMG (20-500Hz Bandpass)...")
    try:
        b, a = butter_bandpass(20, 500, fs_emg, order=4)
        emg_filtered = filtfilt(b, a, raw_emg, axis=0)
    except Exception as e:
        print(f"Filtering failed (fs={fs_emg} might be too low for 500Hz?): {e}")
        emg_filtered = raw_emg  # Fallback

    # Windowing (200ms windows, 10ms step usually, or non-overlapping?)
    # Ninapro setup: 200ms window, step size?
    # Actually, for training regression, usually windows overlap.
    # Let's default to: Window 200ms (0.2 * fs), Step 10ms (0.01 * fs) -> Dense training
    # OR Step 200ms (Non-overlapping) -> Sparse
    # The paper used "10ms increment" (sliding window).

    win_len_sec = 0.2
    step_sec = 0.05  # 50ms step to reduce data size slightly, or 0.01 for dense

    win_len = int(win_len_sec * fs_emg)
    step_len = int(step_sec * fs_emg)

    print(f"Segmenting data: Window={win_len} samples, Step={step_len} samples...")

    n_samples = emg_filtered.shape[0]
    n_windows = (n_samples - win_len) // step_len

    if n_windows <= 0:
        print("Error: Data too short for windowing.")
        return

    n_emg_ch = emg_filtered.shape[1]
    n_kin_ch = kin_data.shape[1]

    EMG = np.zeros((n_windows, n_emg_ch, win_len), dtype=np.float32)
    ANG = np.zeros((n_windows, n_kin_ch), dtype=np.float32)

    for i in range(n_windows):
        start = i * step_len
        end = start + win_len

        # (Channels, Time) format for CNN
        EMG[i] = emg_filtered[start:end].T
        ANG[i] = kin_data[end - 1]  # Last sample of window as target

    print(f"Saving {n_windows} windows to {output_file}...")
    np.savez_compressed(output_file, emg=EMG, angles=ANG, fs=fs_emg)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import custom EMG/Kinematic data for training."
    )
    parser.add_argument(
        "--emg", required=True, help="Path to EMG file (.mat, .csv, .npy)"
    )
    parser.add_argument(
        "--kin", required=True, help="Path to Kinematics file (.mat, .csv, .npy)"
    )
    parser.add_argument("--out", required=True, help="Output .npz file path")
    parser.add_argument(
        "--fs", type=float, default=2000.0, help="EMG Sampling Rate (Hz)"
    )
    parser.add_argument("--emg_var", help="Variable name for EMG in .mat file")
    parser.add_argument("--kin_var", help="Variable name for Kinematics in .mat file")

    args = parser.parse_args()

    process_and_save(
        args.emg,
        args.kin,
        args.out,
        args.fs,
        emg_var=args.emg_var,
        kin_var=args.kin_var,
    )
