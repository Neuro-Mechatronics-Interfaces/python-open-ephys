import argparse
import re
import time
from collections import deque

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
    emg_stream_name: str,
    fs: float,
    emg_channels: int,
    adc_stream_name: str = "OpenEphys_ADC",
    adc_channels: int = 0,
    emg_labels: list = None,
    adc_labels: list = None,
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
        lbl = (
            emg_labels[idx] if emg_labels and idx < len(emg_labels) else f"EMG{idx + 1}"
        )
        ch.append_child_value("label", lbl)
        ch.append_child_value("unit", "uV")
        ch.append_child_value("type", "emg")
    emg_outlet = StreamOutlet(emg_info)

    # ADC outlet (only if we have ADC channels)
    adc_outlet = None
    if adc_channels > 0:
        adc_info = StreamInfo(
            adc_stream_name,
            "ADC",
            int(adc_channels),
            float(fs),
            "float32",
            f"{adc_stream_name}_src",
        )
        adc_channels_xml = adc_info.desc().append_child("channels")
        for idx in range(int(adc_channels)):
            ch = adc_channels_xml.append_child("channel")
            lbl = (
                adc_labels[idx]
                if adc_labels and idx < len(adc_labels)
                else f"ADC{idx + 1}"
            )
            ch.append_child_value("label", lbl)
            ch.append_child_value("unit", "V")
            ch.append_child_value("type", "adc")
        adc_outlet = StreamOutlet(adc_info)

    return emg_outlet, adc_outlet


class OpenEphysLSLStreamer:
    _ADC_PATTERN = re.compile(r"(?i)^ADC")  # matches channel names starting with "ADC"

    def __init__(
        self,
        host="127.0.0.1",
        port=5556,
        expected_fs=0.0,
        emg_channels=0,
        emg_stream_name="OpenEphys_EMG",
        adc_stream_name="OpenEphys_ADC",
        chunk_size=512,
    ):
        self.host = host
        self.port = int(port)
        self.expected_fs = float(expected_fs)
        self.emg_channels = int(emg_channels)  # 0 = auto-detect
        self.emg_stream_name = emg_stream_name
        self.adc_stream_name = adc_stream_name
        self.chunk_size = int(chunk_size)

        self.client = None
        self.emg_outlet = None
        self.adc_outlet = None
        self.running = False

        # Channel index lists (filled during start)
        self.emg_ch_idx = []  # ZMQ channel indices for EMG
        self.adc_ch_idx = []  # ZMQ channel indices for ADC
        self.emg_labels = []  # names from Open Ephys
        self.adc_labels = []
        self.n_adc = 0

        self.total_emg = 0
        self.total_adc = 0
        self.last_poll = 0.0
        self.last_emg_rms = 0.0
        self.last_emg_std = 0.0
        self.last_chunk = 0
        self.last_error = ""
        self.detected_fs = 0.0  # filled after connect
        self._header_fs = 0.0  # from ZMQ header field
        self._measured_fs = 0  # empirical throughput
        self._prev_idx = 0  # track global_sample_index (per-channel)
        self._rate_points = deque()
        self._rate_window_s = 1.0
        self._display_rate_hz = 0.0
        self._ingest_points = deque()
        self._ingest_window_s = 2.0
        self._ingest_rate_hz = 0.0
        self._sample_time_anchor_idx = 0
        self._sample_time_anchor_lsl = 0.0
        self._pending_emg = None
        self._pending_adc = None
        self._pending_ts = np.empty((0,), dtype=np.float64)

    @staticmethod
    def _round_fs(raw: float) -> float:
        """Round a measured fs to the nearest 'standard' rate."""
        # Common DAQ rates
        standard = [
            1000,
            1250,
            1500,
            2000,
            2500,
            3000,
            3333,
            4000,
            5000,
            6250,
            8000,
            10000,
            12500,
            15000,
            20000,
            25000,
            30000,
            40000,
            50000,
        ]
        best = min(standard, key=lambda s: abs(s - raw))
        # Only snap if within 10 %
        if abs(best - raw) / max(best, 1) < 0.10:
            return float(best)
        return round(raw)

    def _wait_for_channels(self, timeout=3.0):
        """Poll seen_nums until the count stabilises for 0.5 s or *timeout* expires.

        Also measures empirical sample throughput so the caller can
        cross-validate the header-reported ``sample_rate``.

        Returns ``(n_channels, measured_fs)`` where *measured_fs* is
        samples-per-second computed from ``global_sample_index`` (header-
        based per-channel index, not the raw payload byte count).
        """
        import time as _t

        start = _t.time()
        prev_count = 0
        stable_since = start
        channels_stable = False

        # snapshot the per-channel sample index at start
        with self.client._lock:
            idx_t0 = int(self.client.global_sample_index)

        while (_t.time() - start) < timeout:
            with self.client._lock:
                n = len(self.client.seen_nums)
            if n != prev_count:
                prev_count = n
                stable_since = _t.time()
            elif not channels_stable and (_t.time() - stable_since) >= 0.5:
                channels_stable = True
                # Keep looping a bit longer to accumulate a better fs estimate.
                # We want at least 1 s of data total for a reliable rate.
                if _t.time() >= start + 1.5:
                    break
            elif channels_stable:
                if _t.time() >= start + 1.5:
                    break
            _t.sleep(0.05)

        elapsed = max(_t.time() - start, 1e-6)
        with self.client._lock:
            idx_t1 = int(self.client.global_sample_index)
        measured_fs = (idx_t1 - idx_t0) / elapsed
        return prev_count, measured_fs

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
        n_detected, measured_fs = self._wait_for_channels(timeout=3.0)

        with self.client._lock:
            detected = sorted(self.client.seen_nums)
            name_map = dict(
                self.client._name_by_index
            )  # {ch_idx: "CH1" / "ADC1" / ...}

        if not detected:
            self.client.stop()
            self.client = None
            raise RuntimeError("No channels detected from Open Ephys stream.")

        # ---- Partition channels into EMG vs ADC by name ----
        emg_idx, adc_idx = [], []
        emg_lbl, adc_lbl = [], []
        for ch in detected:
            name = name_map.get(ch, f"CH{ch + 1}")
            if self._ADC_PATTERN.match(name):
                adc_idx.append(ch)
                adc_lbl.append(name)
            else:
                emg_idx.append(ch)
                emg_lbl.append(name)

        # Apply user-specified EMG channel cap (0 = all non-ADC)
        if self.emg_channels > 0:
            emg_idx = emg_idx[: self.emg_channels]
            emg_lbl = emg_lbl[: self.emg_channels]
        else:
            self.emg_channels = len(emg_idx)

        if self.emg_channels <= 0 and len(adc_idx) <= 0:
            self.client.stop()
            self.client = None
            raise RuntimeError("No channels detected from Open Ephys stream.")

        self.emg_ch_idx = emg_idx
        self.adc_ch_idx = adc_idx
        self.emg_labels = emg_lbl
        self.adc_labels = adc_lbl
        self.n_adc = len(adc_idx)

        # Tell ZMQ client about all channels we care about (EMG + ADC)
        all_ch = emg_idx + adc_idx
        self.client.set_channel_index(all_ch)

        # ---- Infer sampling rate ----
        # Three possible sources (best → worst):
        #   1. User-supplied expected_fs  (if > 0)
        #   2. Empirical throughput measured during channel stabilisation
        #   3. Header-reported sample_rate (client.fs)
        # The empirical rate is the most trustworthy when available because
        # it reflects actual data throughput rather than a header field that
        # some plugins may set incorrectly (e.g. reporting the hardware Intan
        # chip rate of 30/40 kHz instead of the software-decimated rate).
        header_fs = float(self.client.fs)
        self._header_fs = header_fs
        self._measured_fs = self._round_fs(measured_fs) if measured_fs > 100 else round(measured_fs)

        if self.expected_fs > 0:
            # User explicitly chose a rate – honour it.
            self.detected_fs = self.expected_fs
        elif measured_fs > 100:
            # Round to nearest "nice" rate (multiple of 250 or 1000)
            rounded = self._round_fs(measured_fs)
            self.detected_fs = rounded
            # Warn if header claims something very different
            if header_fs > 0 and abs(header_fs - rounded) / max(header_fs, 1) > 0.15:
                import warnings

                warnings.warn(
                    f"ZMQ header reports sample_rate={header_fs:.0f} Hz but "
                    f"empirical throughput is ~{rounded:.0f} Hz. "
                    f"Using measured rate. Override with --fs if needed."
                )
        elif header_fs > 0:
            self.detected_fs = header_fs
        else:
            self.detected_fs = 2000.0
        fs = self.detected_fs
        self.emg_outlet, self.adc_outlet = build_outlets(
            self.emg_stream_name,
            fs,
            self.emg_channels,
            adc_stream_name=self.adc_stream_name,
            adc_channels=self.n_adc,
            emg_labels=self.emg_labels,
            adc_labels=self.adc_labels,
        )

        # Sync drain cursor to global_sample_index (per-channel header index)
        with self.client._lock:
            self._prev_idx = int(self.client.global_sample_index)
        self._sample_time_anchor_idx = self._prev_idx
        self._sample_time_anchor_lsl = _now()
        self._ingest_points.clear()
        self._ingest_points.append((self._sample_time_anchor_lsl, self._prev_idx))
        self._pending_emg = np.empty((0, len(self.emg_ch_idx)), dtype=np.float32)
        self._pending_adc = np.empty((0, len(self.adc_ch_idx)), dtype=np.float32)
        self._pending_ts = np.empty((0,), dtype=np.float64)

        self.running = True
        self.last_poll = _now()
        self.last_error = ""

    def stop(self):
        self.running = False
        try:
            self._push_pending_chunks(force=True)
        except Exception:
            pass
        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
        self.client = None

        self.emg_outlet = None
        self.adc_outlet = None

    def _push_pending_chunks(self, force=False):
        if self._pending_ts is None or self._pending_ts.size == 0:
            return 0, 0

        emitted = 0
        last_chunk = 0
        chunk_n = max(int(self.chunk_size), 1)

        while self._pending_ts.size >= chunk_n or (force and self._pending_ts.size > 0):
            take = chunk_n if self._pending_ts.size >= chunk_n else int(self._pending_ts.size)
            emg_chunk = self._pending_emg[:take]
            adc_chunk = self._pending_adc[:take]
            ts_chunk = self._pending_ts[:take].tolist()

            if emg_chunk.shape[1] > 0 and self.emg_outlet is not None:
                self.emg_outlet.push_chunk(emg_chunk.tolist(), ts_chunk)
            if adc_chunk.shape[1] > 0 and self.adc_outlet is not None:
                self.adc_outlet.push_chunk(adc_chunk.tolist(), ts_chunk)

            self.total_emg += take
            if adc_chunk.shape[1] > 0:
                self.total_adc += take
            self.last_chunk = take
            self.last_poll = _now()
            if emg_chunk.shape[1] > 0:
                self.last_emg_rms = float(np.sqrt(np.mean(emg_chunk * emg_chunk)))
                self.last_emg_std = float(np.std(emg_chunk))
            if ts_chunk:
                self._rate_points.append((float(ts_chunk[-1]), int(self.total_emg)))

            self._pending_emg = self._pending_emg[take:]
            self._pending_adc = self._pending_adc[take:]
            self._pending_ts = self._pending_ts[take:]
            emitted += take
            last_chunk = take

        return emitted, last_chunk

    def poll_once(self):
        info = {
            "running": self.running,
            "rate_hz": self._display_rate_hz,
            "ingest_rate_hz": self._ingest_rate_hz,
            "nominal_rate_hz": self.detected_fs if self.detected_fs > 0 else self.expected_fs,
            "chunk": 0,
            "ingest_chunk": 0,
            "pending_chunk": int(self._pending_ts.size) if self._pending_ts is not None else 0,
            "channels": self.emg_channels,
            "n_adc": self.n_adc,
            "total_emg": self.total_emg,
            "total_adc": self.total_adc,
            "emg_rms": self.last_emg_rms,
            "emg_std": self.last_emg_std,
            "error": self.last_error,
        }
        if not self.running or self.client is None:
            return info

        now = _now()

        n_emg = len(self.emg_ch_idx)
        n_adc = len(self.adc_ch_idx)

        # Use global_sample_index (header-based per-channel index) as cursor.
        with self.client._lock:
            cur_idx = int(self.client.global_sample_index)
            n_new = cur_idx - self._prev_idx
            if n_new <= 0:
                return info
            chunk_start_idx = self._prev_idx
            max_buf = self.client._deque_len
            if n_new > max_buf:
                n_new = max_buf
                chunk_start_idx = cur_idx - n_new

            # Helper to read tails from a set of channel deques
            def _read_channels(ch_list):
                nc = len(ch_list)
                if nc == 0:
                    return np.zeros((0, n_new), dtype=np.float32)
                arr = np.zeros((nc, n_new), dtype=np.float32)
                for i, ch in enumerate(ch_list):
                    buf = self.client.buffers[ch]
                    blen = len(buf)
                    take = min(blen, n_new)
                    if take > 0:
                        start_idx = blen - take
                        for j in range(take):
                            arr[i, n_new - take + j] = buf[start_idx + j]
                return arr

            emg_arr = _read_channels(self.emg_ch_idx)  # (n_emg, n_new)
            adc_arr = _read_channels(self.adc_ch_idx)  # (n_adc, n_new)
            self._prev_idx = cur_idx

        # Transpose to (n_samples, n_channels)
        emg = emg_arr.T  # (n_new, n_emg)
        adc = adc_arr.T  # (n_new, n_adc)
        n_samples = emg.shape[0]
        info["ingest_chunk"] = n_samples
        info["channels"] = n_emg
        info["emg_shape"] = emg.shape  # (n_samples, n_emg)
        info["adc_shape"] = adc.shape  # (n_samples, n_adc)

        if n_samples <= 0:
            return info

        fs = self.detected_fs if self.detected_fs > 0 else self.expected_fs
        sample_idx = chunk_start_idx + np.arange(n_samples, dtype=np.float64)
        ts = self._sample_time_anchor_lsl + (
            (sample_idx - float(self._sample_time_anchor_idx)) / max(fs, 1e-6)
        )
        if self._pending_emg is None:
            self._pending_emg = np.empty((0, n_emg), dtype=np.float32)
        if self._pending_adc is None:
            self._pending_adc = np.empty((0, n_adc), dtype=np.float32)

        self._pending_emg = np.vstack((self._pending_emg, emg))
        self._pending_adc = np.vstack((self._pending_adc, adc))
        self._pending_ts = np.concatenate((self._pending_ts, ts))

        _, emitted_chunk = self._push_pending_chunks(force=False)
        info["pending_chunk"] = int(self._pending_ts.size)

        self._ingest_points.append((float(now), int(cur_idx)))
        ingest_cutoff = float(now) - self._ingest_window_s
        while len(self._ingest_points) >= 2 and self._ingest_points[0][0] < ingest_cutoff:
            self._ingest_points.popleft()

        if len(self._ingest_points) >= 2:
            t0i, s0i = self._ingest_points[0]
            t1i, s1i = self._ingest_points[-1]
            span_i = max(t1i - t0i, 1e-6)
            self._ingest_rate_hz = (s1i - s0i) / span_i

        latest_ts = (
            float(self._pending_ts[-1])
            if self._pending_ts.size > 0
            else (float(ts[-1]) if ts.size > 0 else now)
        )
        cutoff = latest_ts - self._rate_window_s
        while len(self._rate_points) >= 2 and self._rate_points[0][0] < cutoff:
            self._rate_points.popleft()

        if len(self._rate_points) >= 2:
            t0, s0 = self._rate_points[0]
            t1, s1 = self._rate_points[-1]
            span = max(t1 - t0, 1e-6)
            self._display_rate_hz = (s1 - s0) / span
        elif self._rate_points:
            self._display_rate_hz = 0.0

        info.update(
            {
                "rate_hz": self._display_rate_hz,
                "ingest_rate_hz": self._ingest_rate_hz,
                "nominal_rate_hz": fs,
                "chunk": emitted_chunk,
                "ingest_chunk": n_samples,
                "pending_chunk": int(self._pending_ts.size),
                "total_emg": self.total_emg,
                "total_adc": self.total_adc,
                "emg_rms": self.last_emg_rms,
                "emg_std": self.last_emg_std,
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
        self.last_ui_update = 0.0
        self.ui_update_interval = 0.25
        self._init_ui()
        self.setStyleSheet(_DARK_STYLE)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    # ---- UI ----------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("Open Ephys  \u2192  LSL Streamer")
        self.setMinimumSize(460, 440)

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

        self.sync_checkbox = QCheckBox("Auto-detect from stream")
        self.sync_checkbox.setToolTip(
            "Ignore manual Channels/Fs values and use the detected stream configuration"
        )
        self.sync_checkbox.setChecked(True)
        cg.addWidget(self.sync_checkbox, 2, 0, 1, 2)

        cg.addWidget(QLabel("Chunk size"), 2, 2)
        self.chunk_edit = QSpinBox()
        self.chunk_edit.setRange(1, 8192)
        self.chunk_edit.setSingleStep(16)
        self.chunk_edit.setValue(int(self.args.chunk_size))
        self.chunk_edit.setToolTip("Number of samples to emit per LSL chunk")
        cg.addWidget(self.chunk_edit, 2, 3)

        layout.addWidget(conn_group)

        # -- Stream names group --
        stream_group = QGroupBox("LSL Stream Names")
        sg = QGridLayout(stream_group)
        sg.setSpacing(4)

        sg.addWidget(QLabel("EMG stream"), 0, 0)
        self.emg_name = QLineEdit(self.args.emg_stream_name)
        sg.addWidget(self.emg_name, 0, 1)

        sg.addWidget(QLabel("ADC stream"), 1, 0)
        self.adc_name = QLineEdit(self.args.adc_stream_name)
        sg.addWidget(self.adc_name, 1, 1)

        layout.addWidget(stream_group)

        # -- Status labels --
        stat_group = QGroupBox("Status")
        sl = QVBoxLayout(stat_group)
        sl.setSpacing(2)

        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info = QLabel("Channels: EMG=0  ADC=0")
        self.emg_shape = QLabel("EMG total: —  |  output chunk: —")
        self.rate = QLabel("Output fs: N/A  |  Input fs: N/A")

        sl.addWidget(self.status)
        sl.addWidget(self.ch_info)
        sl.addWidget(self.emg_shape)
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
        if self.sync_checkbox.isChecked():
            channels = 0
            fs = 0.0
        else:
            channels = self.ch_edit.value()
            fs = float(self.fs_edit.value())
        chunk_size = int(self.chunk_edit.value())
        emg_name = self.emg_name.text().strip() or self.args.emg_stream_name
        adc_name = self.adc_name.text().strip() or self.args.adc_stream_name
        return OpenEphysLSLStreamer(
            host=host,
            port=port,
            expected_fs=fs,
            emg_channels=channels,
            emg_stream_name=emg_name,
            adc_stream_name=adc_name,
            chunk_size=chunk_size,
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

        # Update channel count and fs from auto-detection if enabled
        if self.sync_checkbox.isChecked():
            if self.streamer.emg_channels > 0:
                self.ch_edit.setValue(self.streamer.emg_channels)
            if self.streamer.detected_fs > 0:
                self.fs_edit.setValue(int(self.streamer.detected_fs))

        n_emg = len(self.streamer.emg_ch_idx)
        n_adc = len(self.streamer.adc_ch_idx)
        self.ch_info.setText(
            f"Channels: EMG={n_emg} ({', '.join(self.streamer.emg_labels[:4])}{'...' if n_emg > 4 else ''})  "
            f"ADC={n_adc}{' (' + ', '.join(self.streamer.adc_labels) + ')' if n_adc else ''}"
        )
        self.reminder.hide()
        self.last_ui_update = 0.0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_config_enabled(False)

    def _on_stop(self):
        if self.streamer is not None:
            self.streamer.stop()
        self.streamer = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info.setText("Channels: EMG=0  ADC=0")
        self.emg_shape.setText("EMG total: —  |  output chunk: —")
        self.rate.setText("Output fs: N/A  |  Input fs: N/A")
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
            self.chunk_edit,
            self.emg_name,
            self.adc_name,
            self.sync_checkbox,
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
            now = time.time()
            if (now - self.last_ui_update) >= self.ui_update_interval:
                emitted_chunk = int(info.get("chunk", 0))
                ingest_chunk = int(info.get("ingest_chunk", 0))
                pending_chunk = int(info.get("pending_chunk", 0))
                output_chunk_text = (
                    f"{emitted_chunk} samples"
                    if emitted_chunk > 0
                    else "waiting"
                )
                self.emg_shape.setText(
                    f"EMG total: {info['total_emg']:,} samples  |  output chunk: {output_chunk_text}  |  ingest: {ingest_chunk}  |  queued: {pending_chunk}"
                )
                self.rate.setText(
                    f"Output fs: {info.get('nominal_rate_hz', 0):.1f} Hz  |  Input fs: {info.get('ingest_rate_hz', 0):.1f} Hz"
                )
                self.last_ui_update = now
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
        adc_stream_name=args.adc_stream_name,
        chunk_size=args.chunk_size,
    )
    streamer.start()
    n_emg = len(streamer.emg_ch_idx)
    n_adc = len(streamer.adc_ch_idx)
    print(
        f"Streaming LSL: EMG='{args.emg_stream_name}' ({n_emg}ch)"
        f", ADC='{args.adc_stream_name}' ({n_adc}ch)"
        f" @ {streamer.detected_fs:.0f} Hz"
        f"  (header={streamer._header_fs:.0f}, measured={streamer._measured_fs})"
    )
    if n_emg:
        print(f"  EMG channels: {streamer.emg_labels}")
    if n_adc:
        print(f"  ADC channels: {streamer.adc_labels}")
    try:
        while True:
            info = streamer.poll_once()
            if info["chunk"] > 0:
                print(
                    f"chunk={info['chunk']} emg={info['total_emg']} adc={info.get('total_adc', 0)} "
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
    p.add_argument("--adc-stream-name", default="OpenEphys_ADC")

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
