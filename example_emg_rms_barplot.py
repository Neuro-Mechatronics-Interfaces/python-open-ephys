import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt
from matplotlib.animation import FuncAnimation
from utilities.ephys_utilities import OpenEphysClient


# === Filtering Functions ===
def notch_filter(data, fs, freq=60.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)


def bandpass_filter(data, fs, low=10.0, high=500.0, order=4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data)


def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))


# === Parameters ===
fs = 2000  # Hz
window_sec = 0.5
window_samples = int(fs * window_sec)
n_channels = 128
noise_threshold = 100  # RMS threshold for flagging noisy channels

client = OpenEphysClient(data_port=5556)

# === Initial Plot ===
fig, ax = plt.subplots(figsize=(16, 6))
channels = np.arange(1, n_channels + 1)
bars = ax.bar(channels, np.zeros(n_channels), color='dodgerblue')
ax.set_ylim(0, 500)
ax.set_xticks(np.arange(0, n_channels + 1, 8))
ax.set_xlabel("Channel")
ax.set_ylabel("RMS Amplitude")
ax.set_title(f"Live EMG RMS per Channel (0.5s Window) - Noise > {noise_threshold}")
ax.grid(True, axis='y')
plt.tight_layout()


# === Animation Update Function ===
def update(frame):
    rms_values = []

    for ch in range(n_channels):
        samples = client.get_samples(channel=ch, n_samples=window_samples)
        if len(samples) < window_samples:
            samples = np.pad(samples, (0, window_samples - len(samples)), mode='constant')

        samples = np.array(samples, dtype=np.float32)

        try:
            filtered = notch_filter(samples, fs)
            filtered = bandpass_filter(filtered, fs)
            rms_val = compute_rms(filtered)
        except Exception:
            rms_val = 0

        rms_values.append(rms_val)

    # Update bar heights and colors
    for bar, val in zip(bars, rms_values):
        bar.set_height(val)
        bar.set_color('red' if val > noise_threshold else 'dodgerblue')

    return bars


# === Start Animation ===
ani = FuncAnimation(fig, update, interval=500)  # update every 500 ms
plt.show()
