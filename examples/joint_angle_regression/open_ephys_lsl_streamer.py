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


def build_outlets(
    emg_stream_name: str, imu_stream_name: str, fs: float, emg_channels: int
):
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
        expected_fs=0.0,
        emg_channels=0,
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
        self.emg_channels = int(emg_channels)  # 0 = auto-detect
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
        self.detected_fs = 0.0  # filled after connect
        self._prev_written = 0  # track ref-channel total_samples_written

    def _wait_for_channels(self, timeout=3.0):
        """Poll seen_nums until the count stabilises for 0.5 s or *timeout* expires."""
        import time as _t

        start = _t.time()
        prev_count = 0
        stable_since = start
        while (_t.time() - start) < timeout:
            with self.client._lock:
                n = len(self.client.seen_nums)
            if n != prev_count:
                prev_count = n
                stable_since = _t.time()
            elif (_t.time() - stable_since) >= 0.5:
                break
            _t.sleep(0.05)
        return prev_count

    def start(self):
        if self.running:
            return
        if ZMQClient is None:
            raise RuntimeError(f"pyoephys import failed: {OEPHYS_IMPORT_ERROR}")
        if StreamInfo is None or StreamOutlet is None:
            raise RuntimeError(f"pylsl import failed: {PYLSL_IMPORT_ERROR}")

        kw = dict(
            host_ip=self.host,
            data_port=str(self.port),
            buffer_seconds=30.0,
            auto_start=False,
            verbose=False,
        )
        if self.emg_channels > 0:
            kw["expected_channel_count"] = self.emg_channels

        self.client = ZMQClient(**kw)
        self.client.index_log_interval_s = float("inf")
        self.client.start()

        # Wait for first data frame
        if not self.client.ready_event.wait(timeout=5.0):
            self.client.stop()
            self.client = None
            raise RuntimeError(
                f"No Open Ephys data from tcp://{self.host}:{self.port} (timeout 5s)."
            )

        # Wait for channel count to stabilise (auto-detect)
        n_detected = self._wait_for_channels(timeout=3.0)

        with self.client._lock:
            detected = sorted(self.client.seen_nums)
        if self.emg_channels <= 0:
            self.emg_channels = len(detected)
        if self.emg_channels <= 0:
            self.client.stop()
            self.client = None
            raise RuntimeError("No channels detected from Open Ephys stream.")

        # Set channel index to the detected channels
        ch_idx = detected[: self.emg_channels]
        self.client.set_channel_index(ch_idx)

        # Infer sampling rate from the stream (client.fs is updated from ZMQ headers)
        client_fs = float(self.client.fs)
        if client_fs > 0 and (self.expected_fs <= 0 or self.expected_fs == 5000.0):
            self.detected_fs = client_fs
        elif self.expected_fs > 0:
            self.detected_fs = self.expected_fs
        else:
            self.detected_fs = client_fs if client_fs > 0 else 2000.0
        fs = self.detected_fs
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

        # Sync drain cursor to ref channel's total_samples_written
        with self.client._lock:
            self._prev_written = int(self.client.total_samples_written)

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
            "channels": self.emg_channels,
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

        # Use total_samples_written (incremented every ref-channel packet) as cursor.
        # This is a monotonically increasing counter of how many samples the ref
        # channel has written — reliable and independent of header-index math.
        with self.client._lock:
            total_w = int(self.client.total_samples_written)
            n_new = total_w - self._prev_written
            if n_new <= 0:
                return info
            # Cap to buffer length
            max_buf = self.client._deque_len
            if n_new > max_buf:
                n_new = max_buf
            ch_idx = self.client.channel_index or []
            n_ch = len(ch_idx)
            if n_ch == 0:
                return info
            # Read tails from each channel's deque
            emg_arr = np.zeros((n_ch, n_new), dtype=np.float32)
            for i, ch in enumerate(ch_idx):
                buf = self.client.buffers[ch]
                blen = len(buf)
                take = min(blen, n_new)
                if take > 0:
                    # Read from the tail of the deque efficiently
                    start_idx = blen - take
                    for j in range(take):
                        emg_arr[i, n_new - take + j] = buf[start_idx + j]
            self._prev_written = total_w

        # emg_arr: (channels, n_new) → transpose to (n_new, channels)
        emg = emg_arr.T
        n_samples = emg.shape[0]
        n_ch_actual = emg.shape[1]
        info["channels"] = n_ch_actual

        if n_samples <= 0:
            return info

        fs = float(self.client.fs) if float(self.client.fs) > 0 else self.expected_fs
        ts_end = _now()
        ts = ts_end - (np.arange(n_samples, dtype=np.float64)[::-1] / fs)

        if (
            self.imu_client is not None
            and hasattr(self.imu_client, "is_running")
            and self.imu_client.is_running()
        ):
            rpy = self.imu_client.get_rpy_deg()
            if rpy:
                r, p, y = rpy
                imu_row = np.array([r, p, y, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                imu = np.tile(imu_row, (n_samples, 1))
            else:
                imu = np.zeros((n_samples, 9), dtype=np.float32)
        else:
            imu = np.zeros((n_samples, 9), dtype=np.float32)

        # Push entire chunk to LSL at once (non-blocking)
        self.emg_outlet.push_chunk(emg.tolist(), ts.tolist())
        self.imu_outlet.push_chunk(imu.tolist(), ts.tolist())

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


class StreamerWindow(QMainWindow):
    """Open Ephys -> LSL control panel with connection config and auto-retry."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.streamer = None
        self.last_retry = 0.0
        self._init_ui()
        self.setStyleSheet(_DARK_STYLE)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    # ---- UI ----------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("Open Ephys  \u2192  LSL Streamer")
        self.setMinimumSize(460, 420)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(6)

        # -- Connection settings group --
        conn_group = QGroupBox("Connection")
        cg = QGridLayout(conn_group)
        cg.setSpacing(4)

        cg.addWidget(QLabel("Host"), 0, 0)
        self.host_edit = QLineEdit(self.args.host)
        self.host_edit.setFixedWidth(150)
        cg.addWidget(self.host_edit, 0, 1)

        cg.addWidget(QLabel("Port"), 0, 2)
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(self.args.port)
        cg.addWidget(self.port_edit, 0, 3)

        cg.addWidget(QLabel("Channels"), 1, 0)
        self.ch_edit = QSpinBox()
        self.ch_edit.setRange(0, 256)
        self.ch_edit.setSpecialValueText("Auto")
        self.ch_edit.setValue(self.args.channels)
        cg.addWidget(self.ch_edit, 1, 1)

        cg.addWidget(QLabel("Fs (Hz)"), 1, 2)
        self.fs_edit = QSpinBox()
        self.fs_edit.setRange(0, 100000)
        self.fs_edit.setSpecialValueText("Auto")
        self.fs_edit.setValue(int(self.args.fs))
        self.fs_edit.setFixedWidth(80)
        cg.addWidget(self.fs_edit, 1, 3)

        layout.addWidget(conn_group)

        # -- Stream names group --
        stream_group = QGroupBox("LSL Stream Names")
        sg = QGridLayout(stream_group)
        sg.setSpacing(4)

        sg.addWidget(QLabel("EMG stream"), 0, 0)
        self.emg_name = QLineEdit(self.args.emg_stream_name)
        sg.addWidget(self.emg_name, 0, 1)

        sg.addWidget(QLabel("IMU stream"), 1, 0)
        self.imu_name = QLineEdit(self.args.imu_stream_name)
        sg.addWidget(self.imu_name, 1, 1)

        layout.addWidget(stream_group)

        # -- Status labels --
        stat_group = QGroupBox("Status")
        sl = QVBoxLayout(stat_group)
        sl.setSpacing(2)

        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.samples = QLabel("Samples: 0")
        self.emg_stats = QLabel("EMG RMS: N/A  |  \u03c3: N/A")
        self.imu_stats = QLabel("IMU \u03c3: N/A  |  Mag \u03c3: N/A")
        self.rate = QLabel("Rate: N/A")

        sl.addWidget(self.status)
        sl.addWidget(self.samples)
        sl.addWidget(self.emg_stats)
        sl.addWidget(self.imu_stats)
        sl.addWidget(self.rate)
        layout.addWidget(stat_group)

        # -- Buttons --
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Connect + Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.auto_retry = QCheckBox("Auto-retry (2 s)")
        self.auto_retry.setChecked(True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.auto_retry)
        layout.addLayout(btns)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        self.reminder = QLabel(
            "Ensure Open Ephys is running with the ZMQ plugin enabled."
        )
        self.reminder.setStyleSheet("color: #ffaa00; font-size: 11px;")
        self.reminder.setWordWrap(True)
        layout.addWidget(self.reminder)

        layout.addStretch()

    # ---- Actions -----------------------------------------------------------
    def _build_streamer(self):
        """Create a fresh OpenEphysLSLStreamer from current widget values."""
        host = self.host_edit.text().strip() or self.args.host
        port = self.port_edit.value()
        channels = self.ch_edit.value()
        fs = float(self.fs_edit.value())
        emg_name = self.emg_name.text().strip() or self.args.emg_stream_name
        imu_name = self.imu_name.text().strip() or self.args.imu_stream_name
        return OpenEphysLSLStreamer(
            host=host,
            port=port,
            expected_fs=fs,
            emg_channels=channels,
            emg_stream_name=emg_name,
            imu_stream_name=imu_name,
            chunk_size=self.args.chunk_size,
            imu_enabled=self.args.imu_enabled,
            imu_host=self.args.imu_host,
            imu_port=self.args.imu_port,
            imu_transport=self.args.imu_transport,
        )

    def _on_start(self):
        # Tear down any old streamer
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception:
                pass

        self.status.setText("Connecting\u2026")
        self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        QApplication.processEvents()

        try:
            self.streamer = self._build_streamer()
            self.streamer.start()
        except Exception as exc:
            short = str(exc)[:120]
            self.status.setText(f"Error: {short}")
            self.status.setStyleSheet(
                "color: #ff4444; font-weight: bold; font-size: 14px;"
            )
            self.streamer = None
            return

        self.status.setText("Streaming")
        self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")
        # Update channel count and fs from auto-detection
        self.ch_edit.setValue(self.streamer.emg_channels)
        if self.streamer.detected_fs > 0:
            self.fs_edit.setValue(int(self.streamer.detected_fs))
        self.reminder.hide()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_config_enabled(False)

    def _on_stop(self):
        if self.streamer is not None:
            self.streamer.stop()
        self.streamer = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.reminder.show()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_config_enabled(True)

    def _set_config_enabled(self, enabled: bool):
        for w in (
            self.host_edit,
            self.port_edit,
            self.ch_edit,
            self.fs_edit,
            self.emg_name,
            self.imu_name,
        ):
            w.setEnabled(enabled)

    # ---- Poll timer --------------------------------------------------------
    def _tick(self):
        # Auto-retry logic
        if self.streamer is None or not self.streamer.running:
            if self.auto_retry.isChecked() and not self.btn_start.isEnabled():
                now = time.time()
                if now - self.last_retry > 2.0:
                    self.last_retry = now
                    self._on_start()
            return

        try:
            info = self.streamer.poll_once()
            ch = info["channels"]
            fs_str = f"{self.streamer.detected_fs:.0f}" if self.streamer.detected_fs > 0 else "?"
            self.samples.setText(
                f"Samples: {info['total_emg']:,}  |  chunk ({ch}ch, {info['chunk']})  @ {fs_str} Hz"
            )
            if info["chunk"] > 0:
                self.emg_stats.setText(
                    f"EMG RMS: {info['emg_rms']:.3f}  |  \u03c3: {info['emg_std']:.3f}"
                )
                self.imu_stats.setText(
                    f"IMU \u03c3: {info['imu_std']:.3f}  |  Mag \u03c3: {info['mag_std']:.3f}"
                )
            self.rate.setText(f"Rate: {info['rate_hz']:.1f} Hz")
        except Exception as exc:
            self.status.setText(f"Error: {exc}")
            self.status.setStyleSheet(
                "color: #ff4444; font-weight: bold; font-size: 14px;"
            )
            self._on_stop()

    def closeEvent(self, event):
        if self.streamer is not None:
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
        f" | {streamer.emg_channels}ch @ {streamer.detected_fs:.0f} Hz"
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
    p.add_argument(
        "--fs",
        type=float,
        default=0.0,
        help="Sampling rate in Hz (0 = auto-detect from stream)",
    )
    p.add_argument(
        "--channels", type=int, default=0, help="EMG channel count (0 = auto-detect)"
    )
    p.add_argument("--chunk-size", type=int, default=512, help="Max pull chunk size")
    p.add_argument("--emg-stream-name", default="OpenEphys_EMG")
    p.add_argument("--imu-stream-name", default="OpenEphys_IMU")

    p.add_argument("--imu-enabled", action="store_true", help="Enable SleeveIMU bridge")
    p.add_argument("--imu-host", default="192.168.4.1")
    p.add_argument("--imu-port", type=int, default=5555)
    p.add_argument("--imu-transport", default="UDP", choices=["UDP", "TCP"])

    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless CLI mode (default is GUI)",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.no_gui or not HAS_QT:
        run_cli(args)
    else:
        app = QApplication([])
        win = StreamerWindow(args)
        win.show()
        app.exec_()


if __name__ == "__main__":
    main()
