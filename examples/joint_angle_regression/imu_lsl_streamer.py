import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
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
        imu_serial_port="COM5",
        imu_serial_baudrate=115200,
        fs=100.0,
    ):
        self.imu_stream_name = imu_stream_name
        self.imu_host = imu_host
        self.imu_port = int(imu_port)
        self.imu_transport = str(imu_transport).upper()
        self.imu_serial_port = imu_serial_port
        self.imu_serial_baudrate = int(imu_serial_baudrate)
        self.fs = float(fs)

        self.imu_client = None
        self.imu_outlet = None
        self.running = False
        self.connected_source = None

        self.total_imu = 0
        self.last_seq = None
        self.dropped_imu = 0
        self.last_poll = 0.0
        self.last_imu_std = 0.0
        self.last_mag_std = 0.0
        self.last_error = ""
        self.source_anchor_us = None
        self.source_anchor_local = None
        self.rate_timestamps = deque()
        self.rate_window_s = 1.0
        self.display_rate_hz = 0.0

    def _timestamp_for_packet(self, pkt, fallback_now):
        src_us = pkt.get("src_us")
        if src_us is None:
            return pkt.get("_received_at", fallback_now)

        src_us = int(src_us)
        if self.source_anchor_us is None or self.source_anchor_local is None:
            self.source_anchor_us = src_us
            self.source_anchor_local = pkt.get("_received_at", fallback_now)
            return self.source_anchor_local

        delta_us = (src_us - self.source_anchor_us) & 0xFFFFFFFF
        return self.source_anchor_local + (delta_us / 1_000_000.0)

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
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
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
                serial_port=self.imu_serial_port,
                serial_baudrate=self.imu_serial_baudrate,
                auto_start=True,
            )
            if not self.imu_client.wait_connected(timeout=3.0):
                raise RuntimeError("Timed out waiting for Pico handshake")
            if self.imu_transport == "SERIAL":
                self.connected_source = (
                    self.imu_client.connected_port or self.imu_serial_port
                )
            else:
                self.connected_source = f"{self.imu_host}:{self.imu_port}"
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
        self.connected_source = None
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
            "rate_hz": self.display_rate_hz,
            "chunk": 0,
            "total_imu": self.total_imu,
            "dropped_imu": self.dropped_imu,
            "imu_std": self.last_imu_std,
            "mag_std": self.last_mag_std,
            "error": self.last_error,
        }
        if not self.running or self.imu_client is None:
            return info

        now = _now()

        packets = self.imu_client.get_imu_packets()
        if packets:
            for pkt in packets:
                r, p, y = pkt.get("rpy", (0.0, 0.0, 0.0))
                ax, ay, az = pkt.get("acc", (0.0, 0.0, 0.0))
                gx, gy, gz = pkt.get("gyr", (0.0, 0.0, 0.0))
                sample = [r, p, y, ax, ay, az, gx, gy, gz]
                timestamp = self._timestamp_for_packet(pkt, now)
                self.imu_outlet.push_sample(sample, timestamp)
                self.rate_timestamps.append(timestamp)

                seq = pkt.get("seq")
                if seq is not None and self.last_seq is not None:
                    gap = int(seq) - int(self.last_seq) - 1
                    if gap > 0:
                        self.dropped_imu += gap
                if seq is not None:
                    self.last_seq = int(seq)

            self.total_imu += len(packets)
            info["chunk"] = len(packets)
            self.last_poll = now

            last_rpy = packets[-1].get("rpy", (0.0, 0.0, 0.0))
            self.last_imu_std = np.std(last_rpy)

        cutoff = now - self.rate_window_s
        while self.rate_timestamps and self.rate_timestamps[0] < cutoff:
            self.rate_timestamps.popleft()

        if len(self.rate_timestamps) >= 2:
            span = max(self.rate_timestamps[-1] - self.rate_timestamps[0], 1e-6)
            self.display_rate_hz = (len(self.rate_timestamps) - 1) / span
        elif not self.rate_timestamps:
            self.display_rate_hz = 0.0

        info["rate_hz"] = self.display_rate_hz

        info.update(
            {
                "total_imu": self.total_imu,
                "dropped_imu": self.dropped_imu,
                "imu_std": self.last_imu_std,
                "mag_std": self.last_mag_std,
                "rate_hz": self.display_rate_hz,
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
        self.last_ui_update = 0.0
        self.ui_update_interval = 0.25
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
        cg.addWidget(QLabel("Transport"), 0, 0)
        self.transport_combo = QComboBox()
        self.transport_combo.addItems(["UDP", "SERIAL"])
        idx = max(0, self.transport_combo.findText(self.args.imu_transport.upper()))
        self.transport_combo.setCurrentIndex(idx)
        cg.addWidget(self.transport_combo, 0, 1)

        cg.addWidget(QLabel("Host"), 1, 0)
        self.host_edit = QLineEdit(self.args.imu_host)
        cg.addWidget(self.host_edit, 1, 1)

        cg.addWidget(QLabel("Port"), 2, 0)
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(self.args.imu_port)
        cg.addWidget(self.port_edit, 2, 1)

        cg.addWidget(QLabel("Serial Port"), 3, 0)
        self.serial_port_edit = QLineEdit(self.args.imu_serial_port)
        cg.addWidget(self.serial_port_edit, 3, 1)
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
        self.drop_lbl = QLabel("Dropped: 0")
        self.source_lbl = QLabel("Source: -")
        v.addWidget(self.status)
        v.addWidget(self.info_lbl)
        v.addWidget(self.rate_lbl)
        v.addWidget(self.drop_lbl)
        v.addWidget(self.source_lbl)
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
        self.transport_combo.currentTextChanged.connect(self._update_transport_ui)
        self._update_transport_ui(self.transport_combo.currentText())
        layout.addStretch()

    def _build_streamer(self):
        host = self.host_edit.text().strip()
        port = self.port_edit.value()
        name = self.name_edit.text().strip()
        transport = self.transport_combo.currentText().strip().upper()
        serial_port = self.serial_port_edit.text().strip()
        return IMULSLStreamer(
            imu_stream_name=name,
            imu_host=host,
            imu_port=port,
            imu_transport=transport,
            imu_serial_port=serial_port,
            imu_serial_baudrate=self.args.imu_serial_baudrate,
        )

    def _update_transport_ui(self, transport):
        is_serial = str(transport).upper() == "SERIAL"
        self.host_edit.setEnabled(not is_serial)
        self.port_edit.setEnabled(not is_serial)
        self.serial_port_edit.setEnabled(is_serial)

    def _on_start(self):
        # Tear down old
        if self.streamer:
            try:
                self.streamer.stop()
            except Exception:
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
        self.source_lbl.setText(f"Source: {self.streamer.connected_source or '-'}")
        self.last_ui_update = 0.0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_enabled(False)

    def _on_stop(self):
        if self.streamer:
            self.streamer.stop()
        self.streamer = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.source_lbl.setText("Source: -")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_enabled(True)

    def _set_enabled(self, val):
        self.transport_combo.setEnabled(val)
        self.host_edit.setEnabled(
            val and self.transport_combo.currentText().upper() != "SERIAL"
        )
        self.port_edit.setEnabled(
            val and self.transport_combo.currentText().upper() != "SERIAL"
        )
        self.serial_port_edit.setEnabled(
            val and self.transport_combo.currentText().upper() == "SERIAL"
        )
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
            now = time.time()
            if (now - self.last_ui_update) >= self.ui_update_interval:
                self.info_lbl.setText(f"Total Samples: {info['total_imu']:,}")
                self.rate_lbl.setText(f"Rate: {info.get('rate_hz', 0):.1f} Hz")
                self.drop_lbl.setText(f"Dropped: {info.get('dropped_imu', 0):,}")
                self.last_ui_update = now
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
        imu_serial_port=args.imu_serial_port,
        imu_serial_baudrate=args.imu_serial_baudrate,
    )
    streamer.start()
    if args.imu_transport.upper() == "SERIAL":
        source = args.imu_serial_port
    else:
        source = f"{args.imu_host}:{args.imu_port}"
    print(f"Streaming IMU: {args.imu_stream_name} from {source}")
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
    parser.add_argument("--imu-serial-port", default="AUTO")
    parser.add_argument("--imu-serial-baudrate", type=int, default=115200)
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
