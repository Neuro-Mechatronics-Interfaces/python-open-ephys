import argparse
import time
from pathlib import Path

import numpy as np

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_QT = True
except Exception:
    HAS_QT = False

try:
    from pyoephys.interface import ZMQClient
except Exception as exc:
    ZMQClient = None
    OEPHYS_IMPORT_ERROR = str(exc)
else:
    OEPHYS_IMPORT_ERROR = None

try:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "interface" / "imu"))
    from sleeveimu import SleeveIMUClient

    SLEEVEIMU_AVAILABLE = True
except Exception:
    SleeveIMUClient = None
    SLEEVEIMU_AVAILABLE = False

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except Exception as exc:
    StreamInfo = None
    StreamOutlet = None
    local_clock = None
    PYLSL_IMPORT_ERROR = str(exc)
else:
    PYLSL_IMPORT_ERROR = None


def _now():
    return local_clock() if local_clock is not None else time.time()


def build_outlets(emg_stream_name: str, imu_stream_name: str, fs: float, emg_channels: int):
    emg_info = StreamInfo(
        emg_stream_name,
        "EMG",
        int(emg_channels),
        float(fs),
        "float32",
        f"{emg_stream_name}_src",
    )
    emg_channels_xml = emg_info.desc().append_child("channels")
    for idx in range(int(emg_channels)):
        ch = emg_channels_xml.append_child("channel")
        ch.append_child_value("label", f"EMG{idx + 1}")
        ch.append_child_value("unit", "uV")
        ch.append_child_value("type", "emg")
    emg_outlet = StreamOutlet(emg_info)

    imu_info = StreamInfo(
        imu_stream_name,
        "IMU",
        9,
        0.0,
        "float32",
        f"{imu_stream_name}_src",
    )
    imu_channels_xml = imu_info.desc().append_child("channels")
    for name in [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ]:
        ch = imu_channels_xml.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("type", "imu")
    imu_outlet = StreamOutlet(imu_info)

    return emg_outlet, imu_outlet


class OpenEphysLSLStreamer:
    def __init__(
        self,
        host="127.0.0.1",
        port=5556,
        expected_fs=5000.0,
        emg_channels=8,
        emg_stream_name="OpenEphys_EMG",
        imu_stream_name="OpenEphys_IMU",
        chunk_size=512,
        imu_enabled=False,
        imu_host="192.168.4.1",
        imu_port=5555,
        imu_transport="UDP",
    ):
        self.host = host
        self.port = int(port)
        self.expected_fs = float(expected_fs)
        self.emg_channels = int(emg_channels)
        self.emg_stream_name = emg_stream_name
        self.imu_stream_name = imu_stream_name
        self.chunk_size = int(chunk_size)
        self.imu_enabled = bool(imu_enabled)
        self.imu_host = imu_host
        self.imu_port = int(imu_port)
        self.imu_transport = imu_transport

        self.client = None
        self.imu_client = None
        self.emg_outlet = None
        self.imu_outlet = None
        self.running = False

        self.total_emg = 0
        self.total_imu = 0
        self.last_poll = 0.0
        self.last_emg_rms = 0.0
        self.last_emg_std = 0.0
        self.last_imu_std = 0.0
        self.last_mag_std = 0.0
        self.last_chunk = 0
        self.last_error = ""

    def start(self):
        if self.running:
            return
        if ZMQClient is None:
            raise RuntimeError(f"pyoephys import failed: {OEPHYS_IMPORT_ERROR}")
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(f"pylsl import failed: {PYLSL_IMPORT_ERROR}")

        self.client = ZMQClient(
            host_ip=self.host,
            data_port=str(self.port),
            buffer_seconds=30.0,
            expected_channel_count=self.emg_channels,
            auto_start=False,
            verbose=False,
        )
        self.client.index_log_interval_s = float("inf")
        self.client.start()
        if not self.client.ready_event.wait(timeout=5.0):
            self.client.stop()
            self.client = None
            raise RuntimeError(
                f"No Open Ephys data from tcp://{self.host}:{self.port} (timeout 5s)."
            )

        fs = float(self.client.fs) if float(self.client.fs) > 0 else self.expected_fs
        self.emg_outlet, self.imu_outlet = build_outlets(
            self.emg_stream_name, self.imu_stream_name, fs, self.emg_channels
        )

        if self.imu_enabled and SLEEVEIMU_AVAILABLE and SleeveIMUClient is not None:
            try:
                self.imu_client = SleeveIMUClient(
                    host=self.imu_host,
                    port=self.imu_port,
                    transport=self.imu_transport,
                    auto_start=True,
                )
                self.imu_client.wait_connected(timeout=3.0)
            except Exception:
                self.imu_client = None

        self.running = True
        self.last_poll = _now()
        self.last_error = ""

    def stop(self):
        self.running = False
        if self.imu_client is not None:
            try:
                self.imu_client.stop()
            except Exception:
                pass
        self.imu_client = None

        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
        self.client = None

        self.emg_outlet = None
        self.imu_outlet = None

    def poll_once(self):
        info = {
            "running": self.running,
            "rate_hz": 0.0,
            "chunk": 0,
            "total_emg": self.total_emg,
            "total_imu": self.total_imu,
            "emg_rms": self.last_emg_rms,
            "emg_std": self.last_emg_std,
            "imu_std": self.last_imu_std,
            "mag_std": self.last_mag_std,
            "error": self.last_error,
        }
        if not self.running or self.client is None:
            return info

        now = _now()
        dt = max(now - self.last_poll, 1e-6)
        self.last_poll = now
        info["rate_hz"] = 1.0 / dt

        t_new, y_new = self.client.drain_new()
        if y_new is None or y_new.size == 0:
            return info

        # y_new shape is (channels, samples)
        emg = np.asarray(y_new[: self.emg_channels, :], dtype=np.float32).T
        n_samples = int(emg.shape[0])
        if n_samples <= 0:
            return info

        fs = float(self.client.fs) if float(self.client.fs) > 0 else self.expected_fs
        ts_end = _now()
        ts = ts_end - (np.arange(n_samples, dtype=np.float64)[::-1] / fs)

        if self.imu_client is not None and hasattr(self.imu_client, "is_running") and self.imu_client.is_running():
            rpy = self.imu_client.get_rpy_deg()
            if rpy:
                r, p, y = rpy
                imu_row = np.array([r, p, y, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                imu = np.tile(imu_row, (n_samples, 1))
            else:
                imu = np.zeros((n_samples, 9), dtype=np.float32)
        else:
            imu = np.zeros((n_samples, 9), dtype=np.float32)

        for idx in range(n_samples):
            self.emg_outlet.push_sample(emg[idx].tolist(), float(ts[idx]))
            self.imu_outlet.push_sample(imu[idx].tolist(), float(ts[idx]))

        self.total_emg += n_samples
        self.total_imu += n_samples
        self.last_chunk = n_samples
        self.last_emg_rms = float(np.sqrt(np.mean(emg * emg)))
        self.last_emg_std = float(np.std(emg))
        self.last_imu_std = float(np.std(imu[:, :6])) if imu.shape[1] >= 6 else 0.0
        self.last_mag_std = float(np.std(imu[:, 6:9])) if imu.shape[1] >= 9 else 0.0

        info.update(
            {
                "chunk": n_samples,
                "total_emg": self.total_emg,
                "total_imu": self.total_imu,
                "emg_rms": self.last_emg_rms,
                "emg_std": self.last_emg_std,
                "imu_std": self.last_imu_std,
                "mag_std": self.last_mag_std,
            }
        )
        return info


class StreamerWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.streamer = OpenEphysLSLStreamer(
            host=args.host,
            port=args.port,
            expected_fs=args.fs,
            emg_channels=args.channels,
            emg_stream_name=args.emg_stream_name,
            imu_stream_name=args.imu_stream_name,
            chunk_size=args.chunk_size,
            imu_enabled=args.imu_enabled,
            imu_host=args.imu_host,
            imu_port=args.imu_port,
            imu_transport=args.imu_transport,
        )

        self.setWindowTitle("Open Ephys -> LSL Streamer")
        self.resize(520, 260)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        cfg_group = QGroupBox("Streams")
        cfg = QGridLayout(cfg_group)
        cfg.addWidget(QLabel("EMG stream"), 0, 0)
        self.emg_name = QLineEdit(args.emg_stream_name)
        cfg.addWidget(self.emg_name, 0, 1)
        cfg.addWidget(QLabel("IMU stream"), 1, 0)
        self.imu_name = QLineEdit(args.imu_stream_name)
        cfg.addWidget(self.imu_name, 1, 1)
        layout.addWidget(cfg_group)

        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.samples = QLabel("Samples: 0")
        self.emg_stats = QLabel("EMG RMS: N/A | σ: N/A")
        self.imu_stats = QLabel("IMU σ: N/A | Mag σ: N/A")
        self.rate = QLabel("Rate: N/A")

        layout.addWidget(self.status)
        layout.addWidget(self.samples)
        layout.addWidget(self.emg_stats)
        layout.addWidget(self.imu_stats)
        layout.addWidget(self.rate)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("Connect + Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.auto_retry = QCheckBox("Auto-retry")
        self.auto_retry.setChecked(True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.auto_retry)
        layout.addLayout(btns)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(50)

    def _start(self):
        self.args.emg_stream_name = self.emg_name.text().strip() or self.args.emg_stream_name
        self.args.imu_stream_name = self.imu_name.text().strip() or self.args.imu_stream_name
        try:
            self.streamer.stop()
        except Exception:
            pass
        self.streamer = OpenEphysLSLStreamer(
            host=self.args.host,
            port=self.args.port,
            expected_fs=self.args.fs,
            emg_channels=self.args.channels,
            emg_stream_name=self.args.emg_stream_name,
            imu_stream_name=self.args.imu_stream_name,
            chunk_size=self.args.chunk_size,
            imu_enabled=self.args.imu_enabled,
            imu_host=self.args.imu_host,
            imu_port=self.args.imu_port,
            imu_transport=self.args.imu_transport,
        )
        self.streamer.start()
        self.status.setText("Connected")
        self.status.setStyleSheet("color: #44ff44; font-weight: bold;")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop(self):
        self.streamer.stop()
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _poll(self):
        if not self.streamer.running:
            if self.auto_retry.isChecked() and not self.btn_start.isEnabled():
                try:
                    self._start()
                except Exception:
                    pass
            return

        try:
            info = self.streamer.poll_once()
            self.samples.setText(
                f"Samples: {info['total_emg']} (+{info['chunk']})"
            )
            self.emg_stats.setText(
                f"EMG RMS: {info['emg_rms']:.3f} | σ: {info['emg_std']:.3f}"
            )
            self.imu_stats.setText(
                f"IMU σ: {info['imu_std']:.3f} | Mag σ: {info['mag_std']:.3f}"
            )
            self.rate.setText(f"Rate: {info['rate_hz']:.1f} Hz")
        except Exception as exc:
            self.status.setText(f"Error: {exc}")
            self.status.setStyleSheet("color: #ff6666; font-weight: bold;")

    def closeEvent(self, event):
        self.streamer.stop()
        event.accept()


def run_cli(args):
    streamer = OpenEphysLSLStreamer(
        host=args.host,
        port=args.port,
        expected_fs=args.fs,
        emg_channels=args.channels,
        emg_stream_name=args.emg_stream_name,
        imu_stream_name=args.imu_stream_name,
        chunk_size=args.chunk_size,
        imu_enabled=args.imu_enabled,
        imu_host=args.imu_host,
        imu_port=args.imu_port,
        imu_transport=args.imu_transport,
    )
    streamer.start()
    print(
        f"Streaming LSL: EMG='{args.emg_stream_name}', IMU='{args.imu_stream_name}'"
    )
    try:
        while True:
            info = streamer.poll_once()
            if info["chunk"] > 0:
                print(
                    f"chunk={info['chunk']} total={info['total_emg']} "
                    f"rate={info['rate_hz']:.1f}Hz rms={info['emg_rms']:.3f}",
                    end="\r",
                    flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()


def build_arg_parser():
    p = argparse.ArgumentParser(description="Open Ephys -> LSL streamer")
    p.add_argument("--host", default="127.0.0.1", help="Open Ephys ZMQ host")
    p.add_argument("--port", type=int, default=5556, help="Open Ephys ZMQ data port")
    p.add_argument("--fs", type=float, default=5000.0, help="Expected EMG sampling rate")
    p.add_argument("--channels", type=int, default=8, help="EMG channel count")
    p.add_argument("--chunk-size", type=int, default=512, help="Max pull chunk size")
    p.add_argument("--emg-stream-name", default="OpenEphys_EMG")
    p.add_argument("--imu-stream-name", default="OpenEphys_IMU")

    p.add_argument("--imu-enabled", action="store_true", help="Enable SleeveIMU bridge")
    p.add_argument("--imu-host", default="192.168.4.1")
    p.add_argument("--imu-port", type=int, default=5555)
    p.add_argument("--imu-transport", default="UDP", choices=["UDP", "TCP"])

    p.add_argument("--gui", action="store_true", help="Run Qt control panel")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.gui:
        if not HAS_QT:
            raise RuntimeError("PyQt5 not available. Install PyQt5 or run without --gui.")
        app = QApplication([])
        win = StreamerWindow(args)
        win.show()
        app.exec_()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
