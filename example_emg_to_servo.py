import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import iirnotch, butter, filtfilt
from utilities.ephys_utilities import OpenEphysClient


# === Filters ===
def notch_filter(data, fs, freq=60.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, fs, low=10.0, high=500.0, order=4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data)

def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))

def differential_to_angle(rms_left, rms_right, gain=1.2, bias=90):
    return np.clip((rms_left - rms_right) * gain + bias, 0, 180)


# === Parameters ===
fs = 2000
window_sec = 0.5
window_samples = int(fs * window_sec)
n_channels = 128
threshold = 100

client = OpenEphysClient(data_port=5556)

# === Plot Layout ===
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])

# Rolling angle plot
ax_angle = fig.add_subplot(gs[0])
angle_line, = ax_angle.plot([], [], lw=2, color='purple')
ax_angle.set_ylim(0, 180)
ax_angle.set_xlim(0, 2)
ax_angle.set_xlabel("Time (s)")
ax_angle.set_ylabel("Servo Angle (deg)")
ax_angle.set_title("Rolling Servo Angle")
ax_angle.grid(True)
angle_history = []
x_history = []
frame_count = 0

# RMS Bars
ax_bars = fig.add_subplot(gs[1])
bars = ax_bars.bar(['Left RMS', 'Right RMS'], [0, 0], color=['steelblue', 'salmon'])
ax_bars.set_ylim(0, 300)
ax_bars.set_ylabel("Avg RMS")
ax_bars.set_title("Filtered RMS")

# Servo Dial
ax_dial = fig.add_subplot(gs[2], polar=True)
ax_dial.set_theta_zero_location("W")
ax_dial.set_theta_direction(-1)
dial_line, = ax_dial.plot([], [], lw=3, color='green')
ax_dial.set_title("Servo Angle Dial", pad=20)
ax_dial.set_yticklabels([])
ax_dial.set_xticks(np.radians(np.linspace(0, 180, 7)))
ax_dial.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])

# Angle smoothing buffer
angle_buffer = []

# === Update ===
def update(frame):
    global frame_count
    frame_count += 1

    left_rms_vals = []
    right_rms_vals = []

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

        if rms_val < threshold:
            (left_rms_vals if ch < 64 else right_rms_vals).append(rms_val)

    rms_left = np.mean(left_rms_vals) if left_rms_vals else 0
    rms_right = np.mean(right_rms_vals) if right_rms_vals else 0

    # Raw angle
    angle_raw = differential_to_angle(rms_left, rms_right)

    # === Smooth angle with rolling buffer ===
    angle_buffer.append(angle_raw)
    if len(angle_buffer) > 5:
        angle_buffer.pop(0)
    angle_smooth = np.mean(angle_buffer)

    # === Rolling plot ===
    t = frame_count * 0.25
    x_history.append(t)
    angle_history.append(angle_smooth)
    if len(x_history) > 4:
        x_history.pop(0)
        angle_history.pop(0)
    angle_line.set_data(x_history, angle_history)
    ax_angle.set_xlim(max(0, t - 1), t)

    # === RMS Bars ===
    bars[0].set_height(rms_left)
    bars[1].set_height(rms_right)

    # === Dial Plot ===
    theta = np.radians(angle_smooth)
    dial_line.set_data([theta, theta], [0, 1])

    return [angle_line, *bars, dial_line]


# === Run Animation ===
ani = FuncAnimation(fig, update, interval=250)
plt.show()
