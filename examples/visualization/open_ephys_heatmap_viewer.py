import argparse
from collections import deque
import os
import re
import time

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_API", "pyqt5")

try:
    from PyQt5.QtCore import QPointF, QRectF, QTimer
    from PyQt5.QtGui import QBrush, QColor, QPainterPath, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGraphicsPathItem,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QGraphicsRectItem,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_QT = True
except Exception:
    HAS_QT = False


_QT_MAIN_WINDOW_BASE = QMainWindow if HAS_QT else object

try:
    if HAS_QT:
        import pyqtgraph as pg
    else:
        pg = None
except Exception:
    pg = None

try:
    from pyoephys.interface import ZMQClient, NotReadyError
except Exception as exc:
    ZMQClient = None
    NotReadyError = RuntimeError
    OEPHYS_IMPORT_ERROR = str(exc)
else:
    OEPHYS_IMPORT_ERROR = None

try:
    from pyoephys.io import infer_grid_dimensions, apply_grid_permutation
except Exception as exc:
    infer_grid_dimensions = None
    apply_grid_permutation = None
    GRID_IMPORT_ERROR = str(exc)
else:
    GRID_IMPORT_ERROR = None

try:
    from pyoephys.processing import RealtimeFilter
except Exception as exc:
    RealtimeFilter = None
    FILTER_IMPORT_ERROR = str(exc)
else:
    FILTER_IMPORT_ERROR = None


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
QSpinBox, QDoubleSpinBox, QComboBox {
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


class OpenEphysHeatmapSource:
    _ADC_PATTERN = re.compile(r"(?i)^ADC")

    def __init__(
        self,
        host="127.0.0.1",
        port=5556,
        expected_fs=0.0,
        emg_channels=0,
        history_seconds=10.0,
        enable_bandpass=True,
        bp_low=20.0,
        bp_high=500.0,
        enable_notch=True,
        notch_freq=60.0,
    ):
        self.host = host
        self.port = int(port)
        self.expected_fs = float(expected_fs)
        self.emg_channels = int(emg_channels)
        self.history_seconds = float(history_seconds)
        self.enable_bandpass = bool(enable_bandpass)
        self.bp_low = float(bp_low)
        self.bp_high = float(bp_high)
        self.enable_notch = bool(enable_notch)
        self.notch_freq = float(notch_freq)

        self.client = None
        self.running = False
        self.emg_ch_idx = []
        self.adc_ch_idx = []
        self.emg_labels = []
        self.adc_labels = []
        self.detected_fs = 0.0
        self.header_fs = 0.0
        self.measured_fs = 0.0
        self._prev_idx = 0
        self._history = deque()
        self._history_samples = 0
        self._max_history_samples = 1
        self._filter = None

    @staticmethod
    def _round_fs(raw):
        standard = [
            1000, 1250, 1500, 2000, 2500, 3000, 3333, 4000, 5000,
            6250, 8000, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000,
        ]
        best = min(standard, key=lambda rate: abs(rate - raw))
        if abs(best - raw) / max(best, 1) < 0.10:
            return float(best)
        return float(round(raw))

    def _wait_for_channels(self, timeout=3.0):
        start = time.time()
        prev_count = 0
        stable_since = start
        channels_stable = False

        with self.client._lock:
            idx_t0 = int(self.client.global_sample_index)

        while (time.time() - start) < timeout:
            with self.client._lock:
                n_channels = len(self.client.seen_nums)
            if n_channels != prev_count:
                prev_count = n_channels
                stable_since = time.time()
            elif not channels_stable and (time.time() - stable_since) >= 0.5:
                channels_stable = True
                if time.time() >= start + 1.5:
                    break
            elif channels_stable and time.time() >= start + 1.5:
                break
            time.sleep(0.05)

        elapsed = max(time.time() - start, 1e-6)
        with self.client._lock:
            idx_t1 = int(self.client.global_sample_index)
        measured_fs = (idx_t1 - idx_t0) / elapsed
        return prev_count, measured_fs

    def start(self):
        if self.running:
            return
        if ZMQClient is None:
            raise RuntimeError(f"pyoephys import failed: {OEPHYS_IMPORT_ERROR}")
        if infer_grid_dimensions is None or apply_grid_permutation is None:
            raise RuntimeError(f"grid utility import failed: {GRID_IMPORT_ERROR}")
        if RealtimeFilter is None:
            raise RuntimeError(f"filter import failed: {FILTER_IMPORT_ERROR}")

        kw = {
            "host_ip": self.host,
            "data_port": str(self.port),
            "buffer_seconds": 30.0,
            "auto_start": False,
            "set_index_looping": False,
            "verbose": False,
        }
        if self.emg_channels > 0:
            kw["expected_channel_count"] = self.emg_channels

        self.client = ZMQClient(**kw)
        self.client.index_log_interval_s = float("inf")
        self.client.start()

        if not self.client.ready_event.wait(timeout=5.0):
            self.client.stop()
            self.client = None
            raise RuntimeError(
                f"No Open Ephys data from tcp://{self.host}:{self.port} (timeout 5s)."
            )

        n_detected, measured_fs = self._wait_for_channels(timeout=3.0)

        with self.client._lock:
            detected = sorted(self.client.seen_nums)
            name_map = dict(self.client._name_by_index)

        if not detected:
            self.client.stop()
            self.client = None
            raise RuntimeError("No channels detected from Open Ephys stream.")

        emg_idx = []
        adc_idx = []
        emg_labels = []
        adc_labels = []
        for ch in detected:
            name = name_map.get(ch, f"CH{ch + 1}")
            if self._ADC_PATTERN.match(name):
                adc_idx.append(ch)
                adc_labels.append(name)
            else:
                emg_idx.append(ch)
                emg_labels.append(name)

        if self.emg_channels > 0:
            emg_idx = emg_idx[: self.emg_channels]
            emg_labels = emg_labels[: self.emg_channels]
        else:
            self.emg_channels = len(emg_idx)

        if self.emg_channels <= 0:
            self.client.stop()
            self.client = None
            raise RuntimeError(
                f"No EMG channels detected. Total detected={n_detected}, ADC={len(adc_idx)}."
            )

        self.emg_ch_idx = emg_idx
        self.adc_ch_idx = adc_idx
        self.emg_labels = emg_labels
        self.adc_labels = adc_labels
        self.client.set_channel_index(self.emg_ch_idx)

        self.header_fs = float(self.client.fs)
        self.measured_fs = measured_fs
        if self.expected_fs > 0:
            self.detected_fs = self.expected_fs
        elif measured_fs > 100:
            self.detected_fs = self._round_fs(measured_fs)
        elif self.header_fs > 0:
            self.detected_fs = self.header_fs
        else:
            self.detected_fs = 2000.0

        self._max_history_samples = max(1, int(round(self.detected_fs * self.history_seconds)))
        self._history.clear()
        self._history_samples = 0
        self._filter = RealtimeFilter(
            fs=float(self.detected_fs),
            n_channels=len(self.emg_ch_idx),
            bp_low=float(self.bp_low),
            bp_high=min(float(self.bp_high), max(float(self.detected_fs * 0.45), float(self.bp_low) + 1.0)),
            enable_bandpass=bool(self.enable_bandpass),
            notch_freqs=(float(self.notch_freq),),
            enable_notch=bool(self.enable_notch),
        )

        with self.client._lock:
            self._prev_idx = int(self.client.global_sample_index)

        self.running = True

    def stop(self):
        self.running = False
        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
        self.client = None
        self._history.clear()
        self._history_samples = 0
        self._filter = None

    def _read_new_chunk(self):
        with self.client._lock:
            cur_idx = int(self.client.global_sample_index)
            if cur_idx < self._prev_idx:
                self._prev_idx = cur_idx
                self._history.clear()
                self._history_samples = 0
                if self._filter is not None:
                    self._filter.reset()
                return None
            n_new = cur_idx - self._prev_idx
            if n_new <= 0:
                return None

            max_buf = self.client._deque_len
            if n_new > max_buf:
                n_new = max_buf

            n_channels = len(self.emg_ch_idx)
            arr = np.zeros((n_channels, n_new), dtype=np.float32)
            for i, ch in enumerate(self.emg_ch_idx):
                buf = self.client.buffers[ch]
                blen = len(buf)
                take = min(blen, n_new)
                if take > 0:
                    start_idx = blen - take
                    for j in range(take):
                        arr[i, n_new - take + j] = buf[start_idx + j]
            self._prev_idx = cur_idx
        return arr

    def _append_history(self, block):
        if block is None or block.size == 0:
            return
        self._history.append(block)
        self._history_samples += int(block.shape[1])
        while self._history and self._history_samples > self._max_history_samples:
            old = self._history.popleft()
            self._history_samples -= int(old.shape[1])

    def _update_history(self):
        if not self.running or self.client is None:
            return
        block = self._read_new_chunk()
        if block is None or block.size == 0:
            return
        if self._filter is not None:
            block = self._filter.process(block)
        self._append_history(block)

    def _latest_filtered_window(self, window_ms):
        self._update_history()
        n_required = max(1, int(round(self.detected_fs * float(window_ms) / 1000.0)))
        if self._history_samples < 1:
            raise NotReadyError("No filtered samples available for heatmap.")

        chunks = []
        remaining = n_required
        for chunk in reversed(self._history):
            if remaining <= 0:
                break
            take = min(int(chunk.shape[1]), remaining)
            chunks.append(chunk[:, -take:])
            remaining -= take

        if not chunks:
            raise NotReadyError("No filtered samples available for heatmap.")

        return np.concatenate(list(reversed(chunks)), axis=1)

    def latest_rms(self, window_ms=250):
        if not self.running or self.client is None:
            raise RuntimeError("Viewer is not connected.")

        data = self._latest_filtered_window(window_ms=int(window_ms))
        if data is None or data.size == 0:
            raise NotReadyError("No samples available for heatmap.")
        return np.sqrt(np.mean(np.square(data, dtype=np.float32), axis=1, dtype=np.float32))


class HeatmapWindow(_QT_MAIN_WINDOW_BASE):
    def __init__(self, args):
        super().__init__()
        if not HAS_QT:
            raise RuntimeError("PyQt5 is required for the heatmap viewer.")

        self.args = args
        self.source = None
        self.levels = None
        self.last_retry = 0.0
        self.silhouette_item = None
        self.forearm_overlay_enabled = bool(self.args.forearm_overlay)
        self.selection_item = None
        self.index_grid = None
        self.selected_channel_idx = None
        self.current_grid = None

        self._init_ui()
        self.setStyleSheet(_DARK_STYLE)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def _init_ui(self):
        self.setWindowTitle("Open Ephys Realtime Heatmap Viewer")
        self.setMinimumSize(1100, 700)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setSpacing(8)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(6)
        controls.setMaximumWidth(430)

        conn_group = QGroupBox("Connection")
        cg = QGridLayout(conn_group)
        cg.setSpacing(4)

        cg.addWidget(QLabel("Host"), 0, 0)
        self.host_edit = QLineEdit(self.args.host)
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
        cg.addWidget(self.fs_edit, 1, 3)

        controls_layout.addWidget(conn_group)

        heat_group = QGroupBox("Heatmap")
        hg = QGridLayout(heat_group)
        hg.setSpacing(4)

        hg.addWidget(QLabel("Rows"), 0, 0)
        self.rows_edit = QSpinBox()
        self.rows_edit.setRange(0, 64)
        self.rows_edit.setSpecialValueText("Auto")
        self.rows_edit.setValue(self.args.rows)
        hg.addWidget(self.rows_edit, 0, 1)

        hg.addWidget(QLabel("Cols"), 0, 2)
        self.cols_edit = QSpinBox()
        self.cols_edit.setRange(0, 64)
        self.cols_edit.setSpecialValueText("Auto")
        self.cols_edit.setValue(self.args.cols)
        hg.addWidget(self.cols_edit, 0, 3)

        hg.addWidget(QLabel("RMS window (ms)"), 1, 0)
        self.window_edit = QSpinBox()
        self.window_edit.setRange(10, 5000)
        self.window_edit.setValue(self.args.window_ms)
        hg.addWidget(self.window_edit, 1, 1)

        hg.addWidget(QLabel("Update (ms)"), 1, 2)
        self.update_edit = QSpinBox()
        self.update_edit.setRange(20, 2000)
        self.update_edit.setValue(self.args.update_ms)
        hg.addWidget(self.update_edit, 1, 3)

        hg.addWidget(QLabel("Orientation"), 2, 0)
        self.orientation_edit = QComboBox()
        self.orientation_edit.addItems([
            "none",
            "rot90",
            "rot180",
            "rot270",
            "fliph",
            "flipv",
            "transpose",
        ])
        self.orientation_edit.setCurrentText(self.args.orientation)
        hg.addWidget(self.orientation_edit, 2, 1, 1, 3)

        hg.addWidget(QLabel("Layout"), 3, 0)
        self.layout_edit = QComboBox()
        self.layout_edit.addItems([
            "hdemg128_vertical_columns",
            "auto",
            "linear",
        ])
        self.layout_edit.setCurrentText(self.args.layout)
        hg.addWidget(self.layout_edit, 3, 1, 1, 3)

        self.auto_scale = QCheckBox("Auto color scale")
        self.auto_scale.setChecked(True)
        hg.addWidget(self.auto_scale, 4, 0, 1, 2)

        hg.addWidget(QLabel("Labels"), 4, 2)
        self.label_mode = QComboBox()
        self.label_mode.addItems([
            "none",
            "channel numbers",
            "channel names",
            "RMS values",
        ])
        initial_label_mode = self.args.labels
        if initial_label_mode is None:
            initial_label_mode = "RMS values" if self.args.show_labels else "none"
        self.label_mode.setCurrentText(initial_label_mode)
        hg.addWidget(self.label_mode, 4, 3)

        hg.addWidget(QLabel("Scale min"), 5, 0)
        self.scale_min_edit = QDoubleSpinBox()
        self.scale_min_edit.setRange(-1e9, 1e9)
        self.scale_min_edit.setDecimals(6)
        self.scale_min_edit.setSingleStep(1.0)
        self.scale_min_edit.setValue(float(self.args.scale_min))
        hg.addWidget(self.scale_min_edit, 5, 1)

        hg.addWidget(QLabel("Scale max"), 5, 2)
        self.scale_max_edit = QDoubleSpinBox()
        self.scale_max_edit.setRange(-1e9, 1e9)
        self.scale_max_edit.setDecimals(6)
        self.scale_max_edit.setSingleStep(1.0)
        self.scale_max_edit.setValue(float(self.args.scale_max))
        hg.addWidget(self.scale_max_edit, 5, 3)
        self._on_auto_scale_changed(self.auto_scale.isChecked())

        controls_layout.addWidget(heat_group)

        filter_group = QGroupBox("Filtering")
        fg = QGridLayout(filter_group)
        fg.setSpacing(4)

        self.bandpass_check = QCheckBox("Bandpass")
        self.bandpass_check.setChecked(self.args.bandpass)
        fg.addWidget(self.bandpass_check, 0, 0, 1, 2)

        fg.addWidget(QLabel("Low (Hz)"), 1, 0)
        self.bp_low_edit = QSpinBox()
        self.bp_low_edit.setRange(1, 5000)
        self.bp_low_edit.setValue(int(self.args.bp_low))
        fg.addWidget(self.bp_low_edit, 1, 1)

        fg.addWidget(QLabel("High (Hz)"), 1, 2)
        self.bp_high_edit = QSpinBox()
        self.bp_high_edit.setRange(2, 10000)
        self.bp_high_edit.setValue(int(self.args.bp_high))
        fg.addWidget(self.bp_high_edit, 1, 3)

        self.notch_check = QCheckBox("Notch")
        self.notch_check.setChecked(self.args.notch)
        fg.addWidget(self.notch_check, 2, 0, 1, 2)

        fg.addWidget(QLabel("Freq (Hz)"), 2, 2)
        self.notch_freq_edit = QSpinBox()
        self.notch_freq_edit.setRange(1, 1000)
        self.notch_freq_edit.setValue(int(self.args.notch_freq))
        fg.addWidget(self.notch_freq_edit, 2, 3)

        controls_layout.addWidget(filter_group)

        stat_group = QGroupBox("Status")
        sl = QVBoxLayout(stat_group)
        sl.setSpacing(2)

        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info = QLabel("Channels: EMG=0 ADC=0")
        self.fs_info = QLabel("Fs: ?")
        self.grid_info = QLabel("Grid: ?")
        self.rms_stats = QLabel("RMS min/max/mean: N/A")
        self.scale_info = QLabel("Color scale: N/A")
        self.pick_info = QLabel("Selected: none")

        sl.addWidget(self.status)
        sl.addWidget(self.ch_info)
        sl.addWidget(self.fs_info)
        sl.addWidget(self.grid_info)
        sl.addWidget(self.rms_stats)
        sl.addWidget(self.scale_info)
        sl.addWidget(self.pick_info)
        controls_layout.addWidget(stat_group)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("Connect")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.auto_retry = QCheckBox("Auto-retry (2 s)")
        self.auto_retry.setChecked(True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.auto_retry)
        controls_layout.addLayout(btns)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.update_edit.valueChanged.connect(self._on_update_interval_changed)
        self.auto_scale.toggled.connect(self._on_auto_scale_changed)
        self.scale_min_edit.valueChanged.connect(self._on_manual_scale_changed)
        self.scale_max_edit.valueChanged.connect(self._on_manual_scale_changed)
        self.label_mode.currentTextChanged.connect(self._on_label_mode_changed)

        reminder = QLabel("Ensure Open Ephys is running with the ZMQ plugin enabled.")
        reminder.setStyleSheet("color: #ffaa00; font-size: 11px;")
        reminder.setWordWrap(True)
        controls_layout.addWidget(reminder)
        controls_layout.addStretch()

        outer.addWidget(controls, stretch=0)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#1f2329")
        self.plot = self.graphics.addPlot()
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setMenuEnabled(False)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setZValue(2)
        self.plot.addItem(self.image_item)
        cmap = pg.colormap.get("inferno")
        self.image_item.setLookupTable(cmap.getLookupTable(nPts=256))
        self.text_items = []
        self.selection_item = QGraphicsRectItem()
        self.selection_item.setPen(QPen(QColor(240, 245, 250, 230), 0.15))
        self.selection_item.setBrush(QBrush(QColor(255, 255, 255, 0)))
        self.selection_item.setZValue(5)
        self.selection_item.hide()
        self.plot.addItem(self.selection_item)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        outer.addWidget(self.graphics, stretch=1)

    def _on_update_interval_changed(self, value):
        self.timer.setInterval(max(20, int(value)))

    def _on_auto_scale_changed(self, checked):
        self.scale_min_edit.setEnabled(not checked)
        self.scale_max_edit.setEnabled(not checked)
        self.levels = None

    def _on_manual_scale_changed(self, value):
        if not self.auto_scale.isChecked():
            self.levels = None

    def _on_label_mode_changed(self, value):
        if self.current_grid is not None:
            self._draw_cell_labels(self.current_grid)

    def _build_source(self):
        return OpenEphysHeatmapSource(
            host=self.host_edit.text().strip() or self.args.host,
            port=self.port_edit.value(),
            expected_fs=float(self.fs_edit.value()),
            emg_channels=self.ch_edit.value(),
            history_seconds=max(10.0, float(self.window_edit.value()) / 1000.0 + 2.0),
            enable_bandpass=self.bandpass_check.isChecked(),
            bp_low=float(self.bp_low_edit.value()),
            bp_high=float(self.bp_high_edit.value()),
            enable_notch=self.notch_check.isChecked(),
            notch_freq=float(self.notch_freq_edit.value()),
        )

    def _grid_shape(self, n_channels):
        if self.layout_edit.currentText() == "hdemg128_vertical_columns" and n_channels == 128:
            if self.rows_edit.value() == 0:
                self.rows_edit.setValue(13)
            if self.cols_edit.value() == 0:
                self.cols_edit.setValue(10)
            return 13, 10

        rows = self.rows_edit.value()
        cols = self.cols_edit.value()
        if rows > 0 and cols > 0:
            return rows, cols

        inferred_rows = None
        inferred_cols = None
        if infer_grid_dimensions is not None:
            inferred_rows, inferred_cols = infer_grid_dimensions(self.source.emg_labels)
            if not inferred_rows or not inferred_cols:
                inferred_rows, inferred_cols = infer_grid_dimensions([None] * n_channels)

        if inferred_rows and inferred_cols:
            if rows == 0:
                self.rows_edit.setValue(int(inferred_rows))
            if cols == 0:
                self.cols_edit.setValue(int(inferred_cols))
            return int(inferred_rows), int(inferred_cols)

        side = int(np.ceil(np.sqrt(max(n_channels, 1))))
        rows = side if rows == 0 else rows
        cols = int(np.ceil(float(n_channels) / rows)) if cols == 0 else cols
        if self.rows_edit.value() == 0:
            self.rows_edit.setValue(int(rows))
        if self.cols_edit.value() == 0:
            self.cols_edit.setValue(int(cols))
        return int(rows), int(cols)

    def _apply_orientation(self, grid, orientation):
        if orientation == "none":
            return grid
        if orientation == "rot90":
            return np.rot90(grid, k=-1)
        if orientation == "rot180":
            return np.rot90(grid, k=2)
        if orientation == "rot270":
            return np.rot90(grid, k=1)
        if orientation == "fliph":
            return np.fliplr(grid)
        if orientation == "flipv":
            return np.flipud(grid)
        if orientation == "transpose":
            return grid.T
        return grid

    def _make_grid(self, values, rows, cols, orientation):
        layout = self.layout_edit.currentText()
        if layout == "hdemg128_vertical_columns" and values.size == 128:
            grid = np.full((13, 10), np.nan, dtype=np.float32)
            col_lengths = [13, 13, 13, 13, 12, 12, 13, 13, 13, 13]
            start = 0
            for col, length in enumerate(col_lengths):
                stop = start + length
                grid[:length, col] = values[start:stop]
                start = stop
            return self._apply_orientation(grid, orientation)

        grid = np.full((rows, cols), np.nan, dtype=np.float32)
        cell_count = min(values.size, rows * cols)
        ordered = np.array(values[:cell_count], dtype=np.float32, copy=True)
        grid.flat[:cell_count] = ordered
        return self._apply_orientation(grid, orientation)

    def _make_index_grid(self, n_channels, rows, cols, orientation):
        layout = self.layout_edit.currentText()
        if layout == "hdemg128_vertical_columns" and n_channels == 128:
            grid = np.full((13, 10), -1, dtype=np.int32)
            col_lengths = [13, 13, 13, 13, 12, 12, 13, 13, 13, 13]
            start = 0
            for col, length in enumerate(col_lengths):
                stop = start + length
                grid[:length, col] = np.arange(start, stop, dtype=np.int32)
                start = stop
            return self._apply_orientation(grid, orientation)

        grid = np.full((rows, cols), -1, dtype=np.int32)
        cell_count = min(n_channels, rows * cols)
        grid.flat[:cell_count] = np.arange(cell_count, dtype=np.int32)
        return self._apply_orientation(grid, orientation)

    def _channel_details(self, channel_idx):
        if self.source is None or channel_idx is None or channel_idx < 0:
            return None
        if channel_idx >= len(self.source.emg_ch_idx):
            return None
        zmq_idx = self.source.emg_ch_idx[channel_idx]
        label = self.source.emg_labels[channel_idx] if channel_idx < len(self.source.emg_labels) else f"CH{zmq_idx + 1}"
        return {
            "local_index": int(channel_idx),
            "channel_number": int(channel_idx + 1),
            "zmq_index": int(zmq_idx),
            "label": label,
        }

    def _set_selected_channel(self, channel_idx, row=None, col=None):
        details = self._channel_details(channel_idx)
        if details is None:
            self.selected_channel_idx = None
            self.pick_info.setText("Selected: none")
            self.selection_item.hide()
            return

        self.selected_channel_idx = int(channel_idx)
        self.pick_info.setText(
            f"Selected: ch {details['channel_number']} | label={details['label']} | zmq_idx={details['zmq_index']}"
        )
        if row is None or col is None:
            self.selection_item.hide()
            return

        self.selection_item.setRect(QRectF(float(col), float(row), 1.0, 1.0))
        self.selection_item.show()

    def _on_plot_clicked(self, event):
        if self.index_grid is None:
            return
        if event.button() != 1:
            return

        pos = self.plot.vb.mapSceneToView(event.scenePos())
        col = int(np.floor(pos.x()))
        row = int(np.floor(pos.y()))
        if row < 0 or col < 0 or row >= self.index_grid.shape[0] or col >= self.index_grid.shape[1]:
            self._set_selected_channel(None)
            return

        channel_idx = int(self.index_grid[row, col])
        if channel_idx < 0:
            self._set_selected_channel(None)
            return
        self._set_selected_channel(channel_idx, row=row, col=col)

    def _clear_silhouette(self):
        if self.silhouette_item is not None:
            self.plot.removeItem(self.silhouette_item)
            self.silhouette_item = None

    def _draw_forearm_silhouette(self, grid, orientation):
        self._clear_silhouette()
        if not self.forearm_overlay_enabled:
            return
        if self.layout_edit.currentText() != "hdemg128_vertical_columns":
            return
        if orientation != "none":
            return

        rows, cols = grid.shape
        cx = cols / 2.0
        points = [
            QPointF(cx - 1.1, -0.8),
            QPointF(cx + 1.1, -0.8),
            QPointF(cx + 1.7, 0.1),
            QPointF(cx + 1.6, 1.5),
            QPointF(cols + 0.4, 4.1),
            QPointF(cols + 0.7, rows - 0.8),
            QPointF(cols - 0.1, rows + 0.9),
            QPointF(0.1, rows + 0.9),
            QPointF(-0.7, rows - 0.8),
            QPointF(-0.4, 4.1),
            QPointF(cx - 1.6, 1.5),
            QPointF(cx - 1.7, 0.1),
        ]
        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()

        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(223, 202, 184, 52)))
        item.setPen(QPen(QColor(244, 235, 226, 120), 0.12))
        item.setZValue(0)
        self.plot.addItem(item)
        self.silhouette_item = item

    def _clear_cell_labels(self):
        while self.text_items:
            item = self.text_items.pop()
            self.plot.removeItem(item)

    def _draw_cell_labels(self, grid):
        self._clear_cell_labels()
        label_mode = self.label_mode.currentText()
        if label_mode == "none":
            return
        if self.index_grid is None:
            return

        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = grid[row, col]
                if np.isnan(value):
                    continue
                channel_idx = int(self.index_grid[row, col])
                details = self._channel_details(channel_idx)
                if details is None:
                    continue
                if label_mode == "channel numbers":
                    label = str(details["channel_number"])
                elif label_mode == "channel names":
                    label = str(details["label"])
                else:
                    label = f"{value:.1f}"
                text = pg.TextItem(text=label, color="#f5f7fa", anchor=(0.5, 0.5))
                text.setZValue(4)
                text.setPos(col + 0.5, row + 0.5)
                self.plot.addItem(text)
                self.text_items.append(text)

    def _on_start(self):
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass

        self.status.setText("Connecting...")
        self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        QApplication.processEvents()

        try:
            self.source = self._build_source()
            self.source.start()
        except Exception as exc:
            self.status.setText(f"Error: {str(exc)[:140]}")
            self.status.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px;")
            self.source = None
            return

        n_emg = len(self.source.emg_ch_idx)
        n_adc = len(self.source.adc_ch_idx)
        rows, cols = self._grid_shape(n_emg)
        self.status.setText("Streaming")
        self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")
        self.ch_edit.setValue(n_emg)
        if self.source.detected_fs > 0:
            self.fs_edit.setValue(int(self.source.detected_fs))
        self.ch_info.setText(
            f"Channels: EMG={n_emg} ({', '.join(self.source.emg_labels[:4])}{'...' if n_emg > 4 else ''}) ADC={n_adc}"
        )
        self.fs_info.setText(
            f"Fs: using {self.source.detected_fs:.0f} Hz | header={self.source.header_fs:.0f} | measured={self.source.measured_fs:.0f}"
        )
        self.grid_info.setText(
            f"Grid: {rows} x {cols} | layout={self.layout_edit.currentText()} | orientation={self.orientation_edit.currentText()}"
        )
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_config_enabled(False)
        self.levels = None
        self._tick()

    def _on_stop(self):
        if self.source is not None:
            self.source.stop()
        self.source = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info.setText("Channels: EMG=0 ADC=0")
        self.fs_info.setText("Fs: ?")
        self.grid_info.setText("Grid: ?")
        self.rms_stats.setText("RMS min/max/mean: N/A")
        self.scale_info.setText("Color scale: N/A")
        self.pick_info.setText("Selected: none")
        self.image_item.clear()
        self._clear_silhouette()
        self._clear_cell_labels()
        self.selection_item.hide()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_config_enabled(True)
        self.levels = None
        self.index_grid = None
        self.selected_channel_idx = None
        self.current_grid = None

    def _set_config_enabled(self, enabled):
        for widget in (
            self.host_edit,
            self.port_edit,
            self.ch_edit,
            self.fs_edit,
            self.rows_edit,
            self.cols_edit,
            self.window_edit,
            self.update_edit,
            self.orientation_edit,
            self.layout_edit,
            self.bandpass_check,
            self.bp_low_edit,
            self.bp_high_edit,
            self.notch_check,
            self.notch_freq_edit,
        ):
            widget.setEnabled(enabled)

    def _tick(self):
        if self.source is None or not self.source.running:
            if self.auto_retry.isChecked() and not self.btn_start.isEnabled():
                now = time.time()
                if now - self.last_retry > 2.0:
                    self.last_retry = now
                    self._on_start()
            return

        try:
            rms = self.source.latest_rms(window_ms=self.window_edit.value())
            rows, cols = self._grid_shape(rms.size)
            orientation = self.orientation_edit.currentText()
            grid = self._make_grid(rms, rows, cols, orientation)
            self.index_grid = self._make_index_grid(rms.size, rows, cols, orientation)
            self.current_grid = grid

            self.status.setText("Streaming")
            self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")

            finite = grid[np.isfinite(grid)]
            if finite.size == 0:
                return

            if self.auto_scale.isChecked() or self.levels is None:
                if self.auto_scale.isChecked():
                    lo = float(np.nanpercentile(finite, 5))
                    hi = float(np.nanpercentile(finite, 95))
                    if hi <= lo:
                        hi = lo + 1e-6
                    self.scale_min_edit.blockSignals(True)
                    self.scale_max_edit.blockSignals(True)
                    self.scale_min_edit.setValue(lo)
                    self.scale_max_edit.setValue(hi)
                    self.scale_min_edit.blockSignals(False)
                    self.scale_max_edit.blockSignals(False)
                else:
                    lo = float(self.scale_min_edit.value())
                    hi = float(self.scale_max_edit.value())
                    if hi <= lo:
                        hi = lo + 1e-6
                if hi <= lo:
                    hi = lo + 1e-6
                self.levels = (lo, hi)

            self._draw_forearm_silhouette(grid, orientation)
            self.image_item.setImage(grid, autoLevels=False)
            self.image_item.setLevels(self.levels)
            self.plot.setRange(xRange=(0, cols), yRange=(0, rows), padding=0.02)
            self._draw_cell_labels(grid)

            if self.selected_channel_idx is not None and self.index_grid is not None:
                matches = np.argwhere(self.index_grid == self.selected_channel_idx)
                if matches.size > 0:
                    row, col = matches[0]
                    self._set_selected_channel(self.selected_channel_idx, row=int(row), col=int(col))
                else:
                    self._set_selected_channel(None)

            self.grid_info.setText(
                f"Grid: {grid.shape[0]} x {grid.shape[1]} | layout={self.layout_edit.currentText()} | orientation={orientation} | filled={int(np.isfinite(grid).sum())}/{grid.size}"
            )
            self.rms_stats.setText(
                f"RMS min/max/mean: {float(np.nanmin(finite)):.3f} / {float(np.nanmax(finite)):.3f} / {float(np.nanmean(finite)):.3f}"
            )
            self.scale_info.setText(
                f"Color scale: {self.levels[0]:.3f} to {self.levels[1]:.3f} ({'auto' if self.auto_scale.isChecked() else 'manual'}) | BP={self.bandpass_check.isChecked()} {self.bp_low_edit.value()}-{self.bp_high_edit.value()} Hz | Notch={self.notch_check.isChecked()} @{self.notch_freq_edit.value()} Hz"
            )
        except NotReadyError:
            self.status.setText("Waiting for data stream...")
            self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        except Exception as exc:
            self.status.setText(f"Error: {str(exc)[:140]}")
            self.status.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px;")
            self._on_stop()

    def closeEvent(self, event):
        if self.source is not None:
            self.source.stop()
        event.accept()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Open Ephys realtime heatmap viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys ZMQ host")
    parser.add_argument("--port", type=int, default=5556, help="Open Ephys ZMQ data port")
    parser.add_argument(
        "--fs",
        type=float,
        default=0.0,
        help="Sampling rate in Hz (0 = auto-detect from stream)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=0,
        help="EMG channel count to use (0 = auto-detect non-ADC channels)",
    )
    parser.add_argument("--rows", type=int, default=0, help="Grid rows (0 = auto)")
    parser.add_argument("--cols", type=int, default=0, help="Grid cols (0 = auto)")
    parser.add_argument("--window-ms", type=int, default=250, help="RMS window in milliseconds")
    parser.add_argument("--update-ms", type=int, default=100, help="Heatmap refresh period in milliseconds")
    parser.add_argument(
        "--layout",
        default="hdemg128_vertical_columns",
        choices=["hdemg128_vertical_columns", "auto", "linear"],
        help="Grid layout mode",
    )
    parser.add_argument(
        "--orientation",
        default="none",
        choices=["none", "rot90", "rot180", "rot270", "fliph", "flipv", "transpose"],
        help="Optional grid remapping mode",
    )
    parser.add_argument("--bp-low", type=float, default=20.0, help="Bandpass low cutoff in Hz")
    parser.add_argument("--bp-high", type=float, default=500.0, help="Bandpass high cutoff in Hz")
    parser.add_argument("--no-bandpass", dest="bandpass", action="store_false", help="Disable bandpass filtering")
    parser.add_argument("--notch-freq", type=float, default=60.0, help="Notch frequency in Hz")
    parser.add_argument("--no-notch", dest="notch", action="store_false", help="Disable notch filtering")
    parser.add_argument("--show-labels", action="store_true", help="Overlay RMS values on each cell")
    parser.add_argument(
        "--labels",
        choices=["none", "channel numbers", "channel names", "RMS values"],
        default=None,
        help="Cell label overlay mode",
    )
    parser.add_argument("--scale-min", type=float, default=0.0, help="Manual color scale minimum")
    parser.add_argument("--scale-max", type=float, default=1.0, help="Manual color scale maximum")
    parser.add_argument("--forearm-overlay", action="store_true", help="Enable the stylized forearm/hand silhouette overlay")
    parser.set_defaults(bandpass=True, notch=True, forearm_overlay=False)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not HAS_QT:
        raise RuntimeError("PyQt5 is required for the realtime heatmap viewer.")

    app = QApplication([])
    win = HeatmapWindow(args)
    win.timer.setInterval(max(20, int(args.update_ms)))
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
