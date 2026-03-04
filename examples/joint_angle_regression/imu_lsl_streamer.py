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
    import sys

    # Add parent directory to path to find interface.imu.sleeveimu
    sys.path.insert(0, str(Path(__file__).parent.parent / "interface" / "imu"))
    from sleeveimu import SleeveIMUClient

    SLEEVEIMU_AVAILABLE = True
except Exception as exc:
    SleeveIMUClient = None
    SLEEVEIMU_AVAILABLE = False
    SLEEVEIMU_ERROR = str(exc)
else:
    SLEEVEIMU_ERROR = None

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


class IMULSLStreamer:
    def __init__(
        self,
        imu_stream_name="OpenEphys_IMU",
        imu_host="192.168.4.1",
        imu_port=5555,
        imu_transport="UDP",
        fs=60.0,  # Approximate rate for SleeveIMU
    ):
        self.imu_stream_name = imu_stream_name
        self.imu_host = imu_host
        self.imu_port = int(imu_port)
        self.imu_transport = imu_transport
        self.fs = float(fs)

        self.imu_client = None
        self.imu_outlet = None
        self.running = False

        self.total_imu = 0
        self.last_poll = 0.0
        self.last_imu_std = 0.0
        self.last_mag_std = 0.0
        self.last_error = ""

    def start(self):
        if self.running:
            return
        if not SLEEVEIMU_AVAILABLE or SleeveIMUClient is None:
            raise RuntimeError(f"SleeveIMU import failed: {SLEEVEIMU_ERROR}")
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(f"pylsl import failed: {PYLSL_IMPORT_ERROR}")

        # Create LSL Outlet
        info = StreamInfo(
            self.imu_stream_name,
            "IMU",
            9,
            self.fs,
            "float32",
            f"{self.imu_stream_name}_src",
        )
        channels = info.desc().append_child("channels")
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
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("type", "imu")

        self.imu_outlet = StreamOutlet(info)

        # Connect to IMU
        try:
            self.imu_client = SleeveIMUClient(
                host=self.imu_host,
                port=self.imu_port,
                transport=self.imu_transport,
                auto_start=True,
            )
            self.imu_client.wait_connected(timeout=3.0)
        except Exception as e:
            self.imu_client = None
            raise RuntimeError(
                f"Failed to connect to IMU at {self.imu_host}:{self.imu_port}: {e}"
            )

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
        self.imu_outlet = None

    def poll_once(self):
        info = {
            "running": self.running,
            "rate_hz": 0.0,
            "chunk": 0,
            "total_imu": self.total_imu,
            "imu_std": self.last_imu_std,
            "mag_std": self.last_mag_std,
            "error": self.last_error,
        }
        if not self.running or self.imu_client is None:
            return info

        now = _now()
        dt = max(now - self.last_poll, 1e-6)

        rpy = self.imu_client.get_rpy_deg()
        if rpy:
            r, p, y = rpy
            sample = [r, p, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # For rate calculation, we can update last_poll only when we push
            # Limit poll rate roughly to self.fs
            if dt > (0.5 / self.fs):
                self.last_poll = now
                info["rate_hz"] = 1.0 / dt

                self.imu_outlet.push_sample(sample, now)
                self.total_imu += 1
                info["chunk"] = 1

                # Stats
                self.last_imu_std = np.std([r, p, y])  # Rough proxy
        else:
            # No data
            pass

        info.update(
            {
                "total_imu": self.total_imu,
                "imu_std": self.last_imu_std,
                "mag_std": self.last_mag_std,
                "rate_hz": info.get("rate_hz", 0.0),
            }
        )
        return info


_DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #262b33;
    color: #c8ccd4;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3a414b;
    border-radius: 4px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #d0d4db;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLabel { color: #c8ccd4; }
QLineEdit, QSpinBox {
    background-color: #2a3039;
    border: 1px solid #3a414b;
    border-radius: 3px;
    padding: 3px 6px;
    color: #c8ccd4;
}
QPushButton {
    background-color: #3a414b;
    border: 1px solid #4a525e;
    border-radius: 3px;
    padding: 5px 14px;
    color: #c8ccd4;
    font-weight: bold;
}
QPushButton:hover { background-color: #444d59; }
QPushButton:pressed { background-color: #505a68; }
QPushButton:disabled { color: #666; background-color: #2a3039; }
QCheckBox { color: #c8ccd4; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #4a525e;
    border-radius: 2px;
    background: #2a3039;
}
QCheckBox::indicator:checked { background: #5294e2; border-color: #5294e2; }
"""


class IMUStreamerWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.streamer = None
        self.last_retry = 0.0
        self._init_ui()
        self.setStyleSheet(_DARK_STYLE)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(20)  # 50Hz polling for UI

    def _init_ui(self):
        self.setWindowTitle("IMU \u2192 LSL Streamer")
        self.setMinimumSize(400, 300)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(6)

        # -- Connection --
        conn_group = QGroupBox("IMU Connection")
        cg = QGridLayout(conn_group)
        cg.addWidget(QLabel("Host"), 0, 0)
        self.host_edit = QLineEdit(self.args.imu_host)
        cg.addWidget(self.host_edit, 0, 1)

        cg.addWidget(QLabel("Port"), 1, 0)
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(self.args.imu_port)
        cg.addWidget(self.port_edit, 1, 1)
        layout.addWidget(conn_group)

        # -- Stream Name --
        sg = QGroupBox("LSL Settings")
        sl = QGridLayout(sg)
        sl.addWidget(QLabel("Stream Name"), 0, 0)
        self.name_edit = QLineEdit(self.args.imu_stream_name)
        sl.addWidget(self.name_edit, 0, 1)
        layout.addWidget(sg)

        # -- Status --
        stat_group = QGroupBox("Status")
        v = QVBoxLayout(stat_group)
        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.info_lbl = QLabel("Total Samples: 0")
        self.info_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.rate_lbl = QLabel("Rate: 0 Hz")
        v.addWidget(self.status)
        v.addWidget(self.info_lbl)
        v.addWidget(self.rate_lbl)
        layout.addWidget(stat_group)

        # -- Buttons --
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Streaming")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.auto_retry = QCheckBox("Auto-retry")
        self.auto_retry.setChecked(True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.auto_retry)
        layout.addLayout(btns)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        layout.addStretch()

    def _build_streamer(self):
        host = self.host_edit.text().strip()
        port = self.port_edit.value()
        name = self.name_edit.text().strip()
        return IMULSLStreamer(
            imu_stream_name=name,
            imu_host=host,
            imu_port=port,
            imu_transport="UDP",  # Hardcoded or add UI choice
        )

    def _on_start(self):
        # Tear down old
        if self.streamer:
            try:
                self.streamer.stop()
            except:
                pass

        self.status.setText("Connecting...")
        self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        QApplication.processEvents()

        try:
            self.streamer = self._build_streamer()
            self.streamer.start()
        except Exception as e:
            self.status.setText(f"Error: {str(e)[:60]}")
            self.status.setStyleSheet(
                "color: #ff4444; font-weight: bold; font-size: 14px;"
            )
            self.streamer = None
            return

        self.status.setText("Streaming")
        self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_enabled(False)

    def _on_stop(self):
        if self.streamer:
            self.streamer.stop()
        self.streamer = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_enabled(True)

    def _set_enabled(self, val):
        self.host_edit.setEnabled(val)
        self.port_edit.setEnabled(val)
        self.name_edit.setEnabled(val)

    def _tick(self):
        if not self.streamer or not self.streamer.running:
            if self.auto_retry.isChecked() and not self.btn_start.isEnabled():
                now = time.time()
                if now - self.last_retry > 2.0:
                    self.last_retry = now
                    self._on_start()
            return

        try:
            info = self.streamer.poll_once()
            self.info_lbl.setText(f"Total Samples: {info['total_imu']:,}")
            self.rate_lbl.setText(f"Rate: {info.get('rate_hz', 0):.1f} Hz")
        except Exception as e:
            self.status.setText(f"Error: {e}")
            self._on_stop()

    def closeEvent(self, event):
        if self.streamer:
            self.streamer.stop()
        event.accept()


def run_cli(args):
    streamer = IMULSLStreamer(
        imu_stream_name=args.imu_stream_name,
        imu_host=args.imu_host,
        imu_port=args.imu_port,
        imu_transport=args.imu_transport,
    )
    streamer.start()
    print(f"Streaming IMU: {args.imu_stream_name} from {args.imu_host}:{args.imu_port}")
    try:
        while True:
            info = streamer.poll_once()
            if info["chunk"] > 0:
                print(
                    f"IMU Samples: {info['total_imu']} Rate: {info['rate_hz']:.1f}Hz",
                    end="\r",
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()


def main():
    parser = argparse.ArgumentParser(description="IMU -> LSL Streamer")
    parser.add_argument("--imu-stream-name", default="OpenEphys_IMU")
    parser.add_argument("--imu-host", default="192.168.4.1")
    parser.add_argument("--imu-port", type=int, default=5555)
    parser.add_argument("--imu-transport", default="UDP")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    if args.no_gui or not HAS_QT:
        run_cli(args)
    else:
        app = QApplication([])
        win = IMUStreamerWindow(args)
        win.show()
        app.exec_()


if __name__ == "__main__":
    main()
