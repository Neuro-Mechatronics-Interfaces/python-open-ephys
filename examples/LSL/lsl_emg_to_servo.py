import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsEllipseItem
import pyqtgraph as pg
from scipy.signal import iirnotch, butter, filtfilt
from pyoephys.interface import LSLClient


# === Filters ===
def notch_filter(data, fs, freq=60.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data)


def bandpass_filter(data, fs, low=10.0, high=400.0, order=4):
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data)


def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))


def differential_to_angle(rms_left, rms_right, gain=1.2, bias=90):
    return np.clip((rms_left - rms_right) * gain + bias, 0, 180)


class EMGWindow(QtWidgets.QMainWindow):
    def __init__(self, client, n_channels=128, threshold=100, window_sec=0.5, time_buffer_sec=2):
        super().__init__()
        self.setWindowTitle("Real-Time EMG to Servo Dashboard")
        self.client = client
        self.fs = client.fs or 2000
        self.window_samples = int(self.fs * window_sec)
        self.angle_history = []
        self.time_history = []
        self.angle_buffer = []
        self.frame_count = 0
        self.threshold = threshold
        self.n_channels = n_channels
        self.time_buffer_sec = time_buffer_sec

        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # === Servo Angle Plot ===
        self.angle_plot = pg.PlotWidget(title="Servo Angle")
        self.angle_plot.setYRange(0, 180)
        self.angle_plot.setLabel('left', "Angle (deg)")
        self.angle_curve = self.angle_plot.plot(pen=pg.mkPen('m', width=2))
        layout.addWidget(self.angle_plot, stretch=2)

        # === RMS Bar Graph ===
        self.rms_plot = pg.PlotWidget(title="Left / Right RMS")
        self.rms_plot.setYRange(0, 300)
        self.rms_bars = pg.BarGraphItem(x=[0, 1], height=[0, 0], width=0.6, brushes=['steelblue', 'salmon'])
        self.rms_plot.addItem(self.rms_bars)
        layout.addWidget(self.rms_plot, stretch=1)

        # === Servo Dial ===
        self.dial_plot = pg.PlotWidget(title="Servo Dial", polar=True)
        self.dial_plot.setAspectLocked(True)
        self.dial_plot.hideAxis('bottom')
        self.dial_plot.hideAxis('left')
        self.dial_angle_line = pg.PlotDataItem(pen=pg.mkPen('g', width=4))
        self.dial_plot.addItem(self.dial_angle_line)
        self.dial_circle = QGraphicsEllipseItem(-1, -1, 2, 2)
        self.dial_circle.setPen(pg.mkPen('gray'))
        self.dial_plot.addItem(self.dial_circle)
        layout.addWidget(self.dial_plot, stretch=1)

    def update(self):
        n = self.client.n_channels
        left_rms_vals = []
        right_rms_vals = []

        for ch in range(n):
            samples = self.client.get_samples(ch, self.window_samples)
            if len(samples) < self.window_samples:
                samples = [0.0] * (self.window_samples - len(samples)) + samples
            signal = np.array(samples, dtype=np.float32)
            try:
                signal = notch_filter(signal, self.fs)
                signal = bandpass_filter(signal, self.fs)
                rms_val = compute_rms(signal)
            except Exception:
                print(f"Error processing channel {ch}: {sys.exc_info()[0]}")
                rms_val = 0

            if rms_val < self.threshold:
                (left_rms_vals if ch < n // 2 else right_rms_vals).append(rms_val)

        rms_left = np.mean(left_rms_vals) if left_rms_vals else 0
        rms_right = np.mean(right_rms_vals) if right_rms_vals else 0
        angle_raw = differential_to_angle(rms_left, rms_right)

        # Smooth angle
        self.angle_buffer.append(angle_raw)
        if len(self.angle_buffer) > self.time_buffer_sec:
            self.angle_buffer.pop(0)
        angle_smooth = np.mean(self.angle_buffer)

        # === Update plots ===
        self.frame_count += 1
        t = self.frame_count * 0.05
        self.time_history.append(t)
        self.angle_history.append(angle_smooth)
        if len(self.time_history) > 100:  #self.fs/self.time_buffer_sec:
            self.time_history.pop(0)
            self.angle_history.pop(0)

        self.angle_curve.setData(self.time_history, self.angle_history)
        #self.rms_bars.setOpts(height=[rms_left, rms_right])
        self.rms_plot.removeItem(self.rms_bars)
        self.rms_bars = pg.BarGraphItem(x=[0, 1], height=[rms_left, rms_right], width=0.6, brushes=['steelblue', 'salmon'])
        self.rms_plot.addItem(self.rms_bars)

        # Dial line
        theta = np.radians(angle_smooth)
        self.dial_angle_line.setData([0, np.cos(theta)], [0, np.sin(theta)])


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    client = LSLClient()
    win = EMGWindow(client, window_sec=0.1)
    win.show()
    sys.exit(app.exec_())
