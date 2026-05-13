import argparse
import re
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


def _format_rate_value(value: float, *, auto_for_zero: bool = False) -> str:
    if value is None:
        return "?"
    value = float(value)
    if value <= 0.0:
        return "Auto" if auto_for_zero else "?"
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f}"
    return f"{value:.1f}"


def _fs_summary(
    requested_fs: float,
    source_fs: float,
    emitted_fs: float,
    *,
    header_fs: float = 0.0,
    measured_fs: float = 0.0,
) -> str:
    return (
        "LSL fs: "
        f"requested={_format_rate_value(requested_fs, auto_for_zero=True)}  |  "
        f"source={_format_rate_value(source_fs)}  |  "
        f"emitted={_format_rate_value(emitted_fs)}  |  "
        f"header={_format_rate_value(header_fs)}  |  "
        f"measured={_format_rate_value(measured_fs)}"
    )


def build_outlets(
    emg_stream_name: str, imu_stream_name: str, fs: float, emg_channels: int,
    adc_stream_name: str = "OpenEphys_ADC", adc_channels: int = 0,
    emg_labels: list = None, adc_labels: list = None,
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
        lbl = emg_labels[idx] if emg_labels and idx < len(emg_labels) else f"EMG{idx + 1}"
        ch.append_child_value("label", lbl)
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
            lbl = adc_labels[idx] if adc_labels and idx < len(adc_labels) else f"ADC{idx + 1}"
            ch.append_child_value("label", lbl)
            ch.append_child_value("unit", "V")
            ch.append_child_value("type", "adc")
        adc_outlet = StreamOutlet(adc_info)

    return emg_outlet, imu_outlet, adc_outlet


class OpenEphysLSLStreamer:
    _ADC_PATTERN = re.compile(r"(?i)^ADC")  # matches channel names starting with "ADC"

    def __init__(
        self,
        host="127.0.0.1",
        port=5556,
        expected_fs=0.0,
        emg_channels=0,
        emg_stream_name="OpenEphys_EMG",
        imu_stream_name="OpenEphys_IMU",
        adc_stream_name="OpenEphys_ADC",
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
        self.adc_stream_name = adc_stream_name
        self.chunk_size = int(chunk_size)
        self.imu_enabled = bool(imu_enabled)
        self.imu_host = imu_host
        self.imu_port = int(imu_port)
        self.imu_transport = imu_transport

        self.client = None
        self.imu_client = None
        self.emg_outlet = None
        self.imu_outlet = None
        self.adc_outlet = None
        self.running = False

        # Channel index lists (filled during start)
        self.emg_ch_idx = []  # ZMQ channel indices for EMG
        self.adc_ch_idx = []  # ZMQ channel indices for ADC
        self.emg_labels = []  # names from Open Ephys
        self.adc_labels = []
        self.n_adc = 0

        self.total_emg = 0
        self.total_imu = 0
        self.total_adc = 0
        self.last_poll = 0.0
        self.last_emg_rms = 0.0
        self.last_emg_std = 0.0
        self.last_imu_std = 0.0
        self.last_mag_std = 0.0
        self.last_chunk = 0
        self.last_error = ""
        self.detected_fs = 0.0  # filled after connect
        self._header_fs = 0.0  # from ZMQ header field
        self._measured_fs = 0  # empirical throughput
        self._prev_idx = 0  # track payload sample count (or header index fallback)
        self._cursor_source = "header"
        self._source_fs = 0.0
        self._pending_resample_idx = np.empty(0, dtype=np.int64)
        self._pending_resample_rows = None

    def _current_sample_cursor_locked(self) -> int:
        """Return the monotonic sample cursor for new-data detection.

        Prefer payload-derived sample counts because some ZMQ broadcasters
        report header indices and sample_rate in the hardware clock domain
        even when the delivered payload has already been decimated.
        """
        if self._cursor_source == "payload":
            return int(getattr(self.client, "total_samples_written", 0))
        return int(self.client.global_sample_index)

    def _maybe_downsample_chunk(self, data: np.ndarray) -> np.ndarray:
        """Reduce a raw source-rate chunk to the requested LSL output rate.

        When Open Ephys broadcasts payloads at the hardware clock rate but the
        user requests a lower LSL rate, average source samples into output-rate
        bins based on absolute sample index. This preserves per-channel alignment
        and supports non-integer ratios such as 30 kHz -> 4 kHz.
        """
        if data.size == 0:
            return data

        source_fs = float(self._source_fs)
        target_fs = float(self.detected_fs if self.detected_fs > 0 else self.expected_fs)
        if source_fs <= 0.0 or target_fs <= 0.0 or source_fs <= (target_fs * 1.01):
            return data

        start_idx = self._prev_idx - data.shape[0]
        abs_idx = start_idx + np.arange(data.shape[0], dtype=np.int64)

        if self._pending_resample_rows is not None and self._pending_resample_rows.size:
            data = np.vstack((self._pending_resample_rows, data))
            abs_idx = np.concatenate((self._pending_resample_idx, abs_idx))

        bin_ids = np.floor((abs_idx.astype(np.float64) * target_fs) / source_fs).astype(np.int64)
        if bin_ids.size == 0:
            return data[:0]

        next_bin = int(np.floor(((int(abs_idx[-1]) + 1) * target_fs) / source_fs))
        if next_bin == int(bin_ids[-1]):
            keep_mask = bin_ids == bin_ids[-1]
            emit_mask = ~keep_mask
            self._pending_resample_idx = abs_idx[keep_mask].copy()
            self._pending_resample_rows = data[keep_mask].copy()
        else:
            emit_mask = np.ones(bin_ids.shape, dtype=bool)
            self._pending_resample_idx = np.empty(0, dtype=np.int64)
            self._pending_resample_rows = None

        emit_bin_ids = bin_ids[emit_mask]
        emit_rows = data[emit_mask]
        if emit_rows.size == 0:
            return data[:0]

        unique_bins, start_pos, counts = np.unique(
            emit_bin_ids, return_index=True, return_counts=True
        )
        reduced = np.zeros((unique_bins.size, emit_rows.shape[1]), dtype=np.float32)
        for i, (pos, count) in enumerate(zip(start_pos, counts)):
            reduced[i, :] = emit_rows[pos:pos + count].mean(axis=0, dtype=np.float64)
        return reduced

    @staticmethod
    def _round_fs(raw: float) -> float:
        """Round a measured fs to the nearest 'standard' rate."""
        # Common DAQ rates
        standard = [
            1000, 1250, 1500, 2000, 2500, 3000, 3333, 4000, 5000,
            6250, 8000, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000,
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

        Returns ``(n_channels, measured_fs, cursor_source)`` where
        *measured_fs* is samples-per-second computed from actual payload
        samples on the reference channel when available, falling back to
        the header-based global index only if payload counters are not yet
        advancing.
        """
        import time as _t

        start = _t.time()
        prev_count = 0
        stable_since = start
        channels_stable = False

        # Snapshot both payload and header counters. Payload samples reflect
        # what was actually delivered over ZMQ; header indices may reflect a
        # higher-rate hardware clock domain.
        with self.client._lock:
            payload_t0 = int(getattr(self.client, "total_samples_written", 0))
            header_t0 = int(self.client.global_sample_index)

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
            payload_t1 = int(getattr(self.client, "total_samples_written", 0))
            header_t1 = int(self.client.global_sample_index)
        payload_delta = max(payload_t1 - payload_t0, 0)
        header_delta = max(header_t1 - header_t0, 0)
        if payload_delta > 0:
            measured_fs = payload_delta / elapsed
            cursor_source = "payload"
        else:
            measured_fs = header_delta / elapsed
            cursor_source = "header"
        return prev_count, measured_fs, cursor_source

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
            set_index_looping=False,
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
        n_detected, measured_fs, cursor_source = self._wait_for_channels(timeout=3.0)

        with self.client._lock:
            detected = sorted(self.client.seen_nums)
            name_map = dict(self.client._name_by_index)  # {ch_idx: "CH1" / "ADC1" / ...}

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
        self._measured_fs = round(measured_fs)
        self._cursor_source = cursor_source
        self._source_fs = measured_fs if measured_fs > 0 else header_fs
        self._pending_resample_idx = np.empty(0, dtype=np.int64)
        self._pending_resample_rows = None

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
        self.emg_outlet, self.imu_outlet, self.adc_outlet = build_outlets(
            self.emg_stream_name, self.imu_stream_name, fs, self.emg_channels,
            adc_stream_name=self.adc_stream_name, adc_channels=self.n_adc,
            emg_labels=self.emg_labels, adc_labels=self.adc_labels,
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

        # Sync drain cursor to global_sample_index (per-channel header index)
        with self.client._lock:
            self._prev_idx = self._current_sample_cursor_locked()

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
        self.adc_outlet = None

    def poll_once(self):
        info = {
            "running": self.running,
            "rate_hz": 0.0,
            "chunk": 0,
            "channels": self.emg_channels,
            "n_adc": self.n_adc,
            "total_emg": self.total_emg,
            "total_imu": self.total_imu,
            "total_adc": self.total_adc,
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

        n_emg = len(self.emg_ch_idx)
        n_adc = len(self.adc_ch_idx)

        # Use payload sample counts as the primary cursor when available.
        with self.client._lock:
            cur_idx = self._current_sample_cursor_locked()
            if cur_idx < self._prev_idx:
                self._prev_idx = cur_idx
                info["error"] = "Source sample index reset; resynchronized to continuing playback."
                self.last_error = info["error"]
                return info
            n_new = cur_idx - self._prev_idx
            if n_new <= 0:
                return info
            max_buf = self.client._deque_len
            if n_new > max_buf:
                n_new = max_buf

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
        if n_emg > 0 or n_adc > 0:
            combined = np.concatenate((emg, adc), axis=1) if n_adc > 0 else emg
            reduced = self._maybe_downsample_chunk(combined)
            if n_adc > 0:
                emg = reduced[:, :n_emg]
                adc = reduced[:, n_emg:]
            else:
                emg = reduced
                adc = np.zeros((reduced.shape[0], 0), dtype=np.float32)

        n_samples = emg.shape[0]
        info["channels"] = n_emg
        info["emg_shape"] = emg.shape  # (n_samples, n_emg)
        info["adc_shape"] = adc.shape  # (n_samples, n_adc)

        if n_samples <= 0:
            return info

        fs = self.detected_fs if self.detected_fs > 0 else self.expected_fs
        ts_end = _now()
        ts = ts_end - (np.arange(n_samples, dtype=np.float64)[::-1] / fs)
        ts_list = ts.tolist()

        # ---- IMU ----
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

        # ---- Push to LSL ----
        if n_emg > 0 and self.emg_outlet is not None:
            self.emg_outlet.push_chunk(emg.tolist(), ts_list)
        self.imu_outlet.push_chunk(imu.tolist(), ts_list)
        if n_adc > 0 and self.adc_outlet is not None:
            self.adc_outlet.push_chunk(adc.tolist(), ts_list)

        self.total_emg += n_samples
        self.total_imu += n_samples
        if n_adc > 0:
            self.total_adc += n_samples
        self.last_chunk = n_samples
        if n_emg > 0:
            self.last_emg_rms = float(np.sqrt(np.mean(emg * emg)))
            self.last_emg_std = float(np.std(emg))
        self.last_imu_std = float(np.std(imu[:, :6])) if imu.shape[1] >= 6 else 0.0
        self.last_mag_std = float(np.std(imu[:, 6:9])) if imu.shape[1] >= 9 else 0.0

        info.update(
            {
                "chunk": n_samples,
                "total_emg": self.total_emg,
                "total_imu": self.total_imu,
                "total_adc": self.total_adc,
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

        cg.addWidget(QLabel("LSL Out (Hz)"), 1, 2)
        self.fs_edit = QSpinBox()
        self.fs_edit.setRange(0, 100000)
        self.fs_edit.setSpecialValueText("Auto")
        self.fs_edit.setValue(int(self.args.fs))
        self.fs_edit.setFixedWidth(80)
        cg.addWidget(self.fs_edit, 1, 3)

        self.fs_hint = QLabel("0 = follow the incoming source rate")
        self.fs_hint.setStyleSheet("color: #8f98a4; font-size: 11px;")
        cg.addWidget(self.fs_hint, 2, 2, 1, 2)

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

        sg.addWidget(QLabel("IMU stream"), 2, 0)
        self.imu_name = QLineEdit(self.args.imu_stream_name)
        sg.addWidget(self.imu_name, 2, 1)

        layout.addWidget(stream_group)

        # -- Status labels --
        stat_group = QGroupBox("Status")
        sl = QVBoxLayout(stat_group)
        sl.setSpacing(2)

        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info = QLabel("Channels: EMG=0  ADC=0")
        self.emg_shape = QLabel("EMG: —")
        self.adc_shape = QLabel("ADC: —")
        self.fs_info = QLabel(
            _fs_summary(self.args.fs, 0.0, 0.0, header_fs=0.0, measured_fs=0.0)
        )
        self.emg_stats = QLabel("EMG RMS: N/A  |  \u03c3: N/A")
        self.imu_stats = QLabel("IMU \u03c3: N/A  |  Mag \u03c3: N/A")
        self.rate = QLabel("Loop: N/A")

        sl.addWidget(self.status)
        sl.addWidget(self.ch_info)
        sl.addWidget(self.emg_shape)
        sl.addWidget(self.adc_shape)
        sl.addWidget(self.fs_info)
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
        adc_name = self.adc_name.text().strip() or self.args.adc_stream_name
        imu_name = self.imu_name.text().strip() or self.args.imu_stream_name
        return OpenEphysLSLStreamer(
            host=host,
            port=port,
            expected_fs=fs,
            emg_channels=channels,
            emg_stream_name=emg_name,
            imu_stream_name=imu_name,
            adc_stream_name=adc_name,
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
        # Update channel count but keep the requested LSL output rate control unchanged.
        self.ch_edit.setValue(self.streamer.emg_channels)
        n_emg = len(self.streamer.emg_ch_idx)
        n_adc = len(self.streamer.adc_ch_idx)
        self.ch_info.setText(
            f"Channels: EMG={n_emg} ({', '.join(self.streamer.emg_labels[:4])}{'...' if n_emg > 4 else ''})  "
            f"ADC={n_adc}{' (' + ', '.join(self.streamer.adc_labels) + ')' if n_adc else ''}"
        )
        self.fs_info.setText(
            _fs_summary(
                self.streamer.expected_fs,
                self.streamer._source_fs,
                self.streamer.detected_fs,
                header_fs=self.streamer._header_fs,
                measured_fs=self.streamer._measured_fs,
            )
        )
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
        self.fs_info.setText(
            _fs_summary(self.fs_edit.value(), 0.0, 0.0, header_fs=0.0, measured_fs=0.0)
        )
        self.rate.setText("Loop: N/A")
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
            self.adc_name,
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
            fs_str = _format_rate_value(self.streamer.detected_fs)
            eshape = info.get("emg_shape", (0, 0))
            ashape = info.get("adc_shape", (0, 0))
            self.emg_shape.setText(
                f"EMG: {info['total_emg']:,} samples  |  chunk {eshape}  |  LSL out @ {fs_str} Hz"
            )
            self.adc_shape.setText(
                f"ADC: {info.get('total_adc', 0):,} samples  |  chunk {ashape}"
                if ashape[1] > 0 else "ADC: none"
            )
            self.fs_info.setText(
                _fs_summary(
                    self.streamer.expected_fs,
                    self.streamer._source_fs,
                    self.streamer.detected_fs,
                    header_fs=self.streamer._header_fs,
                    measured_fs=self.streamer._measured_fs,
                )
            )
            self.rate.setText(
                f"Loop: {info['rate_hz']:.1f} Hz  |  source={_format_rate_value(self.streamer._source_fs)} Hz  |  emitted={fs_str} Hz"
            )
            if info["chunk"] > 0:
                self.emg_stats.setText(
                    f"EMG RMS: {info['emg_rms']:.3f}  |  \u03c3: {info['emg_std']:.3f}"
                )
                self.imu_stats.setText(
                    f"IMU \u03c3: {info['imu_std']:.3f}  |  Mag \u03c3: {info['mag_std']:.3f}"
                )
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
        adc_stream_name=args.adc_stream_name,
        chunk_size=args.chunk_size,
        imu_enabled=args.imu_enabled,
        imu_host=args.imu_host,
        imu_port=args.imu_port,
        imu_transport=args.imu_transport,
    )
    streamer.start()
    n_emg = len(streamer.emg_ch_idx)
    n_adc = len(streamer.adc_ch_idx)
    print(
        f"Streaming LSL: EMG='{args.emg_stream_name}' ({n_emg}ch)"
        f", ADC='{args.adc_stream_name}' ({n_adc}ch)"
        f", IMU='{args.imu_stream_name}'"
        f"  [{_fs_summary(args.fs, streamer._source_fs, streamer.detected_fs, header_fs=streamer._header_fs, measured_fs=streamer._measured_fs)}]"
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
                    f"chunk={info['chunk']} emg={info['total_emg']} adc={info.get('total_adc',0)} "
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
        help="Requested LSL output rate in Hz (0 = follow the incoming source stream rate)",
    )
    p.add_argument(
        "--channels", type=int, default=0, help="EMG channel count (0 = auto-detect)"
    )
    p.add_argument("--chunk-size", type=int, default=512, help="Max pull chunk size")
    p.add_argument("--emg-stream-name", default="OpenEphys_EMG")
    p.add_argument("--adc-stream-name", default="OpenEphys_ADC")
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
