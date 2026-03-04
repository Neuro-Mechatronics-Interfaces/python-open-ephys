import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags


def analyze_sync(npz_file):
    print(f"Analyzing: {npz_file}")
    data = np.load(npz_file)

    # 1. Check Timestamps
    if "timestamps" in data:
        ts = data["timestamps"]
        diffs = np.diff(ts)
        jitter = np.std(diffs) * 1000
        mean_dt = np.mean(diffs) * 1000
        print("Timing Analysis:")
        print(f"  Mean Interval: {mean_dt:.2f} ms")
        print(f"  Jitter (Std):  {jitter:.2f} ms")

        if jitter > 10:
            print("  [WARN] High jitter detected! GUI thread might be blocking.")
    else:
        print("  [WARN] No timestamps found.")

    # 2. Check Lag via Cross-Correlation
    emg = data["emg"]  # (N, C, T, 1) or (N, C, T)
    angles = data["angles"]  # (N, 14)

    # Flatten EMG to continuous-ish for correlation
    # Actually, we have one angle sample per window.
    # Let's take the mean energy of the window vs the angle sample.
    if emg.ndim == 4:
        emg_energy = np.mean(emg**2, axis=(1, 2, 3))
    else:
        emg_energy = np.mean(emg**2, axis=(1, 2))

    # Use Angle Velocity (change) for correlation with EMG energy
    # (EMG causes *force* -> acceleration/velocity)
    angle_mag = np.linalg.norm(angles, axis=1)  # Magnitude of all joints
    angle_vel = np.gradient(angle_mag)

    # Simple correlation
    print("\ncalculating cross-correlation...")
    xcorr = correlate(
        angle_mag - np.mean(angle_mag), emg_energy - np.mean(emg_energy), mode="full"
    )
    lags = correlation_lags(len(angle_mag), len(emg_energy), mode="full")

    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]

    # Lag is in "samples" (windows). Window step?
    # Inspect fs
    fs = data["fs"] if "fs" in data else 500.0  # This is EMG fs.
    # What is window rate?
    # If timestamps typical diff is 50ms, then fs_window = 20Hz.
    if "timestamps" in data:
        fs_win = 1.0 / np.mean(np.diff(data["timestamps"]))
    else:
        fs_win = 20.0  # Guess based on GUI timer

    lag_sec = peak_lag / fs_win

    print("Sync Analysis:")
    print(f"  Window Rate: {fs_win:.1f} Hz")
    print(f"  Peak Correlation Lag: {peak_lag} windows ({lag_sec * 1000:.1f} ms)")

    print("\nInterpretation:")
    print("  Negative Lag: EMG happens BEFORE Angle (Physiological + System Latency)")
    print(
        "  Positive Lag: Angle happens BEFORE EMG (Major System Lag in Camera pipeline)"
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(emg_energy / np.max(emg_energy), label="EMG Energy")
    plt.plot(angle_mag / np.max(angle_mag), label="Angle Mag")
    plt.legend()
    plt.title("Normalized Signals")

    plt.subplot(2, 1, 2)
    plt.plot(lags / fs_win, xcorr)
    plt.axvline(
        lag_sec, color="r", linestyle="--", label=f"Peak: {lag_sec * 1000:.0f}ms"
    )
    plt.legend()
    plt.title("Cross Correlation")
    plt.xlabel("Lag (seconds)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_sync.py <file.npz>")
    else:
        analyze_sync(sys.argv[1])
