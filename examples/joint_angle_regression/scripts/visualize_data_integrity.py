import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_integrity(npz_file, start_idx=0, duration_sec=10):
    if not os.path.exists(npz_file):
        print(f"File not found: {npz_file}")
        return

    data = np.load(npz_file)

    # Extract data
    # Check keys
    if "emg" in data:
        emg = data["emg"]
    elif "data" in data:
        emg = data["data"]
    else:
        print("No 'emg' or 'data' key found.")
        return

    angles = data["angles"]
    fs = float(data["fs"])

    # Unpack if windowed
    # Windowed shape: (N_windows, Channels, Time) or similar
    # We want to flatten it back to continuous for visualization if possible,
    # OR just plot the first few windows side-by-side.
    is_windowed = emg.ndim == 3 or emg.ndim == 4

    if is_windowed:
        print(f"Data is windowed: {emg.shape}. Visualizing concatenated windows...")

        # Calculate mean/RMS for visualization
        if emg.ndim == 4:
            # (N, C, T, 1) -> RMS over T -> (N, C, 1) -> (N, C)
            emg_mean = np.sqrt(np.mean(emg**2, axis=2))
            if emg_mean.ndim == 3:
                emg_mean = emg_mean.squeeze(-1)
        else:
            # (N, C, T) -> RMS over T -> (N, C)
            emg_mean = np.sqrt(np.mean(emg**2, axis=2))

        emg_to_plot = emg_mean
        ang_to_plot = angles

    else:
        print(f"Data is continuous. Shape: {emg.shape}")
        if emg.shape[0] < emg.shape[1]:
            print(
                "Warning: Channels first? Transposing for visualization (Samples, Channels)."
            )
            emg = emg.T

        emg_to_plot = emg
        ang_to_plot = angles

    # Select range
    start_sample = int(start_idx)
    steps = len(emg_to_plot)
    # If windowed, 'fs' is window density. If continuous, fs is sampling rate.
    # Let's just limit by samples
    end_sample = min(start_sample + 2000, steps)  # Plot 2000 points

    # Calculate stats first
    print("\nStatistical Integrity Check:")
    print(f"EMG Original Range: {np.min(emg_to_plot):.4e} to {np.max(emg_to_plot):.4e}")
    emg_mean_val = np.mean(emg_to_plot)
    emg_std_val = np.std(emg_to_plot)
    print(f"EMG Mean: {emg_mean_val:.4e}, Std: {emg_std_val:.4e}")

    # Check per-channel activity
    channel_stds = np.std(emg_to_plot, axis=0)
    print("Channel Standard Deviations:")
    for ch_idx, std in enumerate(channel_stds):
        print(f"  Ch {ch_idx}: {std:.4e}")
        if std < 1e-12:
            print(f"    [WARN] Dead channel {ch_idx}?")

    emg_data_for_plot = emg_to_plot
    if np.max(np.abs(emg_to_plot)) < 1e-2:
        print(
            "  [INFO] EMG values are small (likely raw units). Standardizing for plot visibility..."
        )
        emg_data_for_plot = (emg_to_plot - emg_mean_val) / (emg_std_val + 1e-8)

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # EMG
    axs[0].plot(emg_data_for_plot[start_sample:end_sample])
    axs[0].set_title("EMG Inputs (RMS)" if is_windowed else "EMG Raw")
    axs[0].set_ylabel("Amplitude")

    # Angles
    axs[1].plot(ang_to_plot[start_sample:end_sample])
    axs[1].set_title("Joint Angle Targets")
    axs[1].set_ylabel("Angle (Normalized or Deg)")
    axs[1].set_xlabel("Sample / Window Index")

    plt.tight_layout()
    plt.show()

    print(f"Ang Range: {np.min(ang_to_plot):.4f} to {np.max(ang_to_plot):.4f}")

    print(f"EMG NaNs: {np.isnan(emg_to_plot).sum()}")
    print(f"Ang NaNs: {np.isnan(ang_to_plot).sum()}")

    # Simple correlation check (averaged across channels)
    # Use standardized view for correlation to avoid scale issues
    emg_mag = np.mean(np.abs(emg_data_for_plot), axis=1)
    ang_mag = np.mean(np.abs(ang_to_plot - np.mean(ang_to_plot, axis=0)), axis=1)

    if len(emg_mag) == len(ang_mag):
        corr = np.corrcoef(emg_mag, ang_mag)[0, 1]
        print(f"Global EMG-Kinematic Correlation: {corr:.4f}")
        if corr < 0.1:
            print("  [WARN] Low correlation. Possible sync issue or bad data.")
        else:
            print("  [OK] Positive correlation detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to .npz file")
    args = parser.parse_args()

    visualize_integrity(args.npz_file)
