import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QProcess, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Ensure repo root on path for nml imports when running from examples/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hardware/device streaming is handled by external LSL streamer scripts.

# Feature extractors are imported lazily to avoid hard dependency issues at GUI start.
KerasFeatureExtractor = None
PyTorchFeatureExtractor = None

try:
    from pylsl import StreamInlet, local_clock, resolve_byprop, resolve_streams

    HAS_LSL = True
except Exception:
    resolve_byprop = None
    resolve_streams = None
    StreamInlet = None
    local_clock = None
    HAS_LSL = False

try:
    from lsl_utils import ANGLE_KEYS, DEFAULT_TARGET_SPEC, get_target_keys
except Exception:
    ANGLE_KEYS = []
    get_target_keys = None
    DEFAULT_TARGET_SPEC = "full14"

FRIENDLY_JOINT_LABELS = {
    "thumb_cmc_mcp": "Thumb base",
    "thumb_ip": "Thumb tip",
    "index_mcp": "Index knuckle",
    "index_pip": "Index mid",
    "index_dip": "Index tip",
    "middle_mcp": "Middle knuckle",
    "middle_pip": "Middle mid",
    "middle_dip": "Middle tip",
    "ring_mcp": "Ring knuckle",
    "ring_pip": "Ring mid",
    "ring_dip": "Ring tip",
    "pinky_mcp": "Pinky knuckle",
    "pinky_pip": "Pinky mid",
    "pinky_dip": "Pinky tip",
}


def _friendly_joint(name):
    return FRIENDLY_JOINT_LABELS.get(name, name.replace("_", " "))


def _clock():
    if local_clock is not None:
        return local_clock()
    return time.time()


# print("GUI python:", sys.executable)


class LSLMonitor:
    def __init__(self, labels, emg_stream_input, imu_stream_input):
        self.labels = labels
        self.emg_stream_input = emg_stream_input
        self.imu_stream_input = imu_stream_input
        self.last_update = 0.0
        self.last_resolve = 0.0
        self.last_emg_chunk = None
        self.last_emg_ts = None
        self.last_imu_chunk = None
        self.last_imu_ts = None
        self.last_chunk_id = 0
        self.emg_inlet = None
        self.imu_inlet = None
        self.emg_connected = False
        self.imu_connected = False
        self.last_emg_data_time = 0.0
        self.last_imu_data_time = 0.0

    def _read_stream_name(self, widget):
        try:
            return widget.text().strip() if widget is not None else ""
        except Exception:
            return ""

    def _ensure_inlet(self, stream_name, current_inlet):
        if current_inlet is not None:
            return current_inlet
        if not stream_name or resolve_byprop is None or StreamInlet is None:
            return None
        try:
            streams = resolve_byprop("name", stream_name, timeout=0.01)
            if streams:
                return StreamInlet(streams[0])
        except Exception:
            return None
        return None

    def poll(self):
        try:
            now = _clock()
            dt = max(now - self.last_update, 1e-6)
            poll_rate = 1.0 / dt

            self.last_update = now

            emg_stream_name = self._read_stream_name(self.emg_stream_input)
            imu_stream_name = self._read_stream_name(self.imu_stream_input)

            if now - self.last_resolve >= 0.5:
                self.last_resolve = now
                self.emg_inlet = self._ensure_inlet(emg_stream_name, self.emg_inlet)
                self.imu_inlet = self._ensure_inlet(imu_stream_name, self.imu_inlet)

            emg_chunk_count = 0
            if self.emg_inlet is not None:
                try:
                    samples, timestamps = self.emg_inlet.pull_chunk(
                        timeout=0.0, max_samples=512
                    )
                    if samples:
                        emg_arr = np.asarray(samples, dtype=np.float32)
                        if emg_arr.ndim == 1:
                            emg_arr = emg_arr.reshape(1, -1)
                        emg_arr = emg_arr.T
                        if emg_arr.shape[0] > 8:
                            emg_arr = emg_arr[:8, :]
                        elif emg_arr.shape[0] < 8:
                            pad_rows = 8 - emg_arr.shape[0]
                            emg_arr = np.vstack(
                                [
                                    emg_arr,
                                    np.zeros(
                                        (pad_rows, emg_arr.shape[1]), dtype=np.float32
                                    ),
                                ]
                            )

                        emg_chunk_count = int(emg_arr.shape[1])
                        self.last_emg_chunk = emg_arr.T
                        if timestamps and len(timestamps) == emg_chunk_count:
                            self.last_emg_ts = np.asarray(timestamps, dtype=np.float64)
                        else:
                            fs = 500.0
                            t_end = _clock()
                            t0 = t_end - (self.last_emg_chunk.shape[0] - 1) / fs
                            self.last_emg_ts = (
                                t0 + np.arange(self.last_emg_chunk.shape[0]) / fs
                            )
                        self.last_chunk_id += 1
                        self.labels["shape"].setText(
                            f"Shape: {self.last_emg_chunk.shape}"
                        )
                        self.emg_connected = True
                        self.last_emg_data_time = now
                except Exception:
                    self.emg_inlet = None
                    self.emg_connected = False

            imu_chunk_count = 0
            if self.imu_inlet is not None:
                try:
                    samples, timestamps = self.imu_inlet.pull_chunk(
                        timeout=0.0, max_samples=512
                    )
                    if samples:
                        imu_arr = np.asarray(samples, dtype=np.float32)
                        if imu_arr.ndim == 1:
                            imu_arr = imu_arr.reshape(1, -1)
                        imu_chunk_count = int(imu_arr.shape[0])
                        self.last_imu_chunk = imu_arr
                        if timestamps and len(timestamps) == imu_chunk_count:
                            self.last_imu_ts = np.asarray(timestamps, dtype=np.float64)
                        elif self.last_emg_ts is not None and len(self.last_emg_ts):
                            self.last_imu_ts = self.last_emg_ts[-imu_chunk_count:]
                        else:
                            self.last_imu_ts = np.full(
                                (imu_chunk_count,), _clock(), dtype=np.float64
                            )

                        self.labels["imu_shape"].setText(
                            f"Shape: {self.last_imu_chunk.shape}"
                        )
                        self.imu_connected = True
                        self.last_imu_data_time = now
                except Exception:
                    self.imu_inlet = None
                    self.imu_connected = False

            if self.emg_inlet is None:
                self.emg_connected = False
            if self.imu_inlet is None:
                self.imu_connected = False

            # Check for stale data (no new data for >1 second)
            emg_stale = False
            imu_stale = False
            if self.emg_connected and self.last_emg_data_time > 0:
                emg_age = now - self.last_emg_data_time
                if emg_age > 1.0:
                    emg_stale = True
                    self.emg_connected = False
            if self.imu_connected and self.last_imu_data_time > 0:
                imu_age = now - self.last_imu_data_time
                if imu_age > 1.0:
                    imu_stale = True
                    self.imu_connected = False

            # Update EMG status with stale detection
            if self.emg_connected:
                status_text = "● Connected"
                status_color = "#44ff44"
            elif emg_stale:
                status_text = "● Stale"
                status_color = "#d9b44a"
            else:
                status_text = "● Waiting..."
                status_color = "#ffaa00"
            self.labels["status"].setText(status_text)
            self.labels["status"].setStyleSheet(
                f"color: {status_color}; font-weight: bold;"
            )

            # Update IMU status with stale detection
            if "imu_status" in self.labels:
                if self.imu_connected:
                    imu_status_text = "● Connected"
                    imu_status_color = "#44ff44"
                elif imu_stale:
                    imu_status_text = "● Stale"
                    imu_status_color = "#d9b44a"
                else:
                    imu_status_text = "● Waiting..."
                    imu_status_color = "#ffaa00"
                self.labels["imu_status"].setText(imu_status_text)
                self.labels["imu_status"].setStyleSheet(
                    f"color: {imu_status_color}; font-weight: bold;"
                )

            # Show polling rate only when stream is actively receiving data.
            self.labels["rate"].setText(
                f"Rate: {poll_rate:.1f} Hz" if self.emg_connected else "Rate: N/A Hz"
            )
            if "imu_rate" in self.labels:
                self.labels["imu_rate"].setText(
                    f"Rate: {poll_rate:.1f} Hz"
                    if self.imu_connected
                    else "Rate: N/A Hz"
                )
        except Exception:
            self.labels["status"].setText("● Error")
            self.labels["status"].setStyleSheet("color: #ff4444; font-weight: bold;")
            self.labels["rate"].setText("Rate: N/A Hz")
            if "imu_rate" in self.labels:
                self.labels["imu_rate"].setText("Rate: N/A Hz")


class CollapsibleSection(QFrame):
    def __init__(
        self, title, content_widget, parent=None, default_open=True, header_widget=None
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.toggle_btn = QToolButton(text=title, checkable=True, checked=default_open)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow if default_open else Qt.RightArrow)
        self.toggle_btn.clicked.connect(self._on_toggled)

        header = QHBoxLayout()
        header.addWidget(self.toggle_btn)
        header.addStretch(1)
        if header_widget is not None:
            header.addWidget(header_widget)

        self.content = content_widget
        self.content.setVisible(default_open)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(header)
        layout.addWidget(self.content)

    def _on_toggled(self, checked):
        self.content.setVisible(checked)
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)


class FlowBlock(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("flow_block")
        self.compact = True
        self.compact_width = 105
        self.full_width = 140
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold;")
        self.title.setWordWrap(True)
        self.title.setAlignment(Qt.AlignCenter)
        self.status = QLabel("Status: idle")
        self.status.setAlignment(Qt.AlignCenter)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addWidget(self.detail)
        self.set_compact(True)
        self._apply_size()

    def set_state(self, ok, status_text, detail_text=""):
        self.status.setText(status_text)
        self.detail.setText(detail_text)
        if ok == "warn":
            self.setProperty("state", "warn")
        elif ok == "active":
            self.setProperty("state", "active")
        elif ok is True:
            self.setProperty("state", "ok")
        elif ok is False:
            self.setProperty("state", "bad")
        else:
            self.setProperty("state", "idle")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_compact(self, compact=True):
        self.compact = compact
        self.status.setVisible(not compact)
        self.detail.setVisible(not compact)
        self._apply_size()

    def _apply_size(self):
        width = self.compact_width if self.compact else self.full_width
        self.setFixedWidth(width)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

    def set_size_profile(self, compact_width=None, full_width=None, min_height=None):
        if compact_width is not None:
            self.compact_width = compact_width
        if full_width is not None:
            self.full_width = full_width
        if min_height is not None:
            self.setMinimumHeight(min_height)
        self._apply_size()


class FlowOverlay(QWidget):
    def __init__(self, diagram):
        super().__init__(diagram)
        self.diagram = diagram
        self.show_lines = True
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def set_show_lines(self, enabled):
        self.show_lines = enabled
        self.update()

    def paintEvent(self, event):
        if not self.show_lines:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        base_pen = QPen(QColor("#6f7b86"), 2)
        base_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(base_pen)
        painter.setBrush(Qt.NoBrush)

        for start, end, ports, style in self.diagram.get_connections():
            pen = QPen(base_pen)
            if style == "dashed":
                pen.setStyle(Qt.DashLine)
                pen.setDashPattern([6, 6])
            painter.setPen(pen)

            path = QPainterPath(start)
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            if abs(dx) >= abs(dy):
                bend = max(18, abs(dx) * 0.35)
                c1 = QPointF(start.x() + (bend if dx >= 0 else -bend), start.y())
                c2 = QPointF(end.x() - (bend if dx >= 0 else -bend), end.y())
            else:
                bend = max(18, abs(dy) * 0.35)
                c1 = QPointF(start.x(), start.y() + (bend if dy >= 0 else -bend))
                c2 = QPointF(end.x(), end.y() - (bend if dy >= 0 else -bend))
            path.cubicTo(c1, c2, end)
            painter.drawPath(path)

            painter.setPen(base_pen)
            painter.setBrush(QColor("#6f7b86"))
            for port in ports:
                painter.drawEllipse(port, 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)

        painter.end()


class FlowDiagram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocks = {
            "emg": FlowBlock("EMG Acquisition"),
            "filters": FlowBlock("Filters"),
            "window": FlowBlock("Windowing"),
            "hand": FlowBlock("Hand LSL"),
            "target": FlowBlock("Angle QC"),
            "imu": FlowBlock("IMU (opt)"),
            "sync": FlowBlock("Sync + Match"),
            "buffer": FlowBlock("Buffer/Record"),
            "features": FlowBlock("Features + Scale"),
            "train": FlowBlock("Train Reg"),
            "saved": FlowBlock("Saved"),
            "infer": FlowBlock("Infer"),
            "compare": FlowBlock("Live Compare"),
        }
        self.compact = True
        self.blocks["sync"].set_size_profile(
            compact_width=65, full_width=85, min_height=50
        )
        self.blocks["buffer"].set_size_profile(
            compact_width=75, full_width=110, min_height=50
        )
        self.blocks["features"].set_size_profile(
            compact_width=80, full_width=115, min_height=50
        )
        self.blocks["infer"].set_size_profile(
            compact_width=65, full_width=90, min_height=48
        )
        self.blocks["compare"].set_size_profile(
            compact_width=75, full_width=105, min_height=50
        )
        self.blocks["train"].set_size_profile(
            compact_width=70, full_width=95, min_height=45
        )
        self.blocks["saved"].set_size_profile(
            compact_width=70, full_width=95, min_height=45
        )
        self.blocks["filters"].set_size_profile(compact_width=80, full_width=110)
        self.blocks["target"].set_size_profile(compact_width=80, full_width=110)
        self.blocks["imu"].set_size_profile(compact_width=70, full_width=95)
        self._build_layout()
        self.overlay = FlowOverlay(self)
        self.overlay.raise_()
        self.overlay.setGeometry(self.rect())

    def _hspacer(self):
        spacer = QWidget()
        spacer.setMinimumWidth(8)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spacer

    def _vspacer(self):
        spacer = QWidget()
        spacer.setMinimumHeight(12)
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        return spacer

    def _ports(self, count):
        ports = QWidget()
        layout = QVBoxLayout(ports)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for _ in range(count):
            dot = QLabel("●")
            dot.setStyleSheet("color: #6f7b86; font-size: 8px;")
            layout.addWidget(dot, alignment=Qt.AlignLeft)
        return ports

    def _wrap_with_ports(self, block, left_ports=1):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        if left_ports > 1:
            layout.addWidget(self._ports(left_ports))
        layout.addWidget(block)
        return container

    def _build_layout(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(12)

        sync_wrap = self._wrap_with_ports(self.blocks["sync"], left_ports=3)

        layout.addWidget(self.blocks["emg"], 0, 0)
        layout.addWidget(self._hspacer(), 0, 1)
        layout.addWidget(self.blocks["filters"], 0, 2)
        layout.addWidget(self._hspacer(), 0, 3)
        layout.addWidget(self.blocks["window"], 0, 4)

        layout.addWidget(self._hspacer(), 0, 5)
        layout.addWidget(self.blocks["imu"], 1, 0)

        layout.addWidget(sync_wrap, 1, 6)
        layout.addWidget(self._hspacer(), 1, 7)
        layout.addWidget(self.blocks["buffer"], 1, 8)
        layout.addWidget(self._hspacer(), 1, 9)
        layout.addWidget(self.blocks["features"], 1, 10)
        layout.addWidget(self._hspacer(), 1, 11)
        layout.addWidget(self.blocks["infer"], 1, 12)
        layout.addWidget(self._hspacer(), 1, 13)
        layout.addWidget(self.blocks["compare"], 1, 14)

        layout.addWidget(self.blocks["hand"], 2, 0)
        layout.addWidget(self._hspacer(), 2, 1)
        layout.addWidget(self.blocks["target"], 2, 2)
        layout.addWidget(self._hspacer(), 2, 3, 1, 1)

        layout.addWidget(self.blocks["train"], 2, 10)
        layout.addWidget(self._hspacer(), 2, 11)
        layout.addWidget(self.blocks["saved"], 2, 12)

        for col in range(1, 15, 2):
            layout.setColumnMinimumWidth(col, 8)
            layout.setColumnStretch(col, 1)
        for col in range(0, 15, 2):
            layout.setColumnStretch(col, 0)

    def set_compact(self, compact):
        self.compact = compact
        for block in self.blocks.values():
            block.set_compact(compact)
        if self.overlay:
            self.overlay.update()

    def set_show_lines(self, enabled):
        self.overlay.set_show_lines(enabled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.overlay:
            self.overlay.setGeometry(self.rect())
            self.overlay.update()

    def _anchor_point(self, block_key, side, port_index=0, port_count=1):
        block = self.blocks[block_key]
        rect = block.rect()
        port_count = max(1, port_count)
        frac = (port_index + 1) / (port_count + 1)
        if side == "left":
            x = 0
            y = rect.height() * frac
        elif side == "right":
            x = rect.width()
            y = rect.height() * frac
        elif side == "top":
            x = rect.width() / 2
            y = 0
        else:
            x = rect.width() / 2
            y = rect.height()
        point = block.mapTo(self, QPoint(int(x), int(y)))
        return QPointF(point)

    def get_connections(self):
        conns = [
            ("emg", "filters", "right", "left", 0, 1, 0, 1, "solid"),
            ("filters", "window", "right", "left", 0, 1, 0, 1, "solid"),
            ("window", "sync", "right", "left", 0, 1, 0, 3, "solid"),
            ("imu", "sync", "right", "left", 0, 1, 1, 3, "dashed"),
            ("hand", "target", "right", "left", 0, 1, 0, 1, "solid"),
            ("target", "sync", "right", "left", 0, 1, 2, 3, "solid"),
            ("sync", "buffer", "right", "left", 0, 1, 0, 1, "solid"),
            ("buffer", "features", "right", "left", 0, 1, 0, 1, "solid"),
            ("features", "infer", "right", "left", 0, 2, 0, 1, "solid"),
            ("features", "train", "bottom", "top", 1, 2, 0, 1, "solid"),
            ("train", "saved", "right", "left", 0, 1, 0, 1, "solid"),
            ("saved", "infer", "top", "bottom", 0, 1, 0, 1, "solid"),
            ("infer", "compare", "right", "left", 0, 1, 0, 1, "solid"),
        ]

        rendered = []
        for (
            from_key,
            to_key,
            from_side,
            to_side,
            from_port,
            from_count,
            to_port,
            to_count,
            style,
        ) in conns:
            start = self._anchor_point(from_key, from_side, from_port, from_count)
            end = self._anchor_point(to_key, to_side, to_port, to_count)
            ports = [start, end]
            rendered.append((start, end, ports, style))
        return rendered


class SessionConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Joint Angle Regression Console (Flow)")
        self.setMinimumSize(980, 760)
        self._closing = False

        self.proc_prompts = None
        self.proc_record = None
        self.proc_train = None
        self.proc_compare = None
        self.proc_feat = None
        self._feat_out_auto = None
        self._train_out_auto = None
        self.prompt_timer = None
        self.prompt_plan_data = []
        self.prompt_idx = 0
        self.prompt_end_time = 0.0
        self.prompt_outlet = None
        self.feat_last_epoch = 0
        self.feat_max_epochs = 0
        self.feat_early_stopped = False

        self.log_buffer = []
        self.angle_inlet = None
        self.last_angle_inlet_resolve = 0.0
        self.last_angle_seen = 0.0
        self.last_angle_rate_hz = 0.0
        self.angle_count = 0
        self.prompt_mtime = 0.0
        self.angle_buffer = []
        self.last_valid_angles = [None, None]
        self.last_angle_match_ok = None
        self._snapshot_done = False
        self.snapshot_on_open = False
        self.snapshot_on_open_path = None
        self.snapshot_delay_ms = 200
        self.marker_inlet = None
        self.last_marker_inlet_resolve = 0.0
        self.last_marker_seen = 0.0
        self.marker_buffer = []
        self.last_lsl_status_check = 0.0
        self.lsl_status_check_interval_s = 0.5
        self.inlet_resolve_interval_s = 0.25

        # In-GUI recording state
        self.recording_active = False
        self.record_start_time = 0.0
        self.record_duration = 0
        self.record_fs = 500.0
        self.record_window_ms = 150
        self.record_overlap_ms = 75
        self.record_window_len = self._ms_to_samples(self.record_window_ms)
        self.record_overlap = self._ms_to_samples(self.record_overlap_ms)
        self.record_max_age = 0.5
        self.angle_lag_ms = 75.0
        self.angle_lag_s = self.angle_lag_ms / 1000.0
        self.record_hand_idx = 0
        self.record_out_path = ""
        self.record_angle_stream = ""
        self.record_marker_stream = ""
        self.record_plan_file = ""
        self.record_plan_json = ""
        self.record_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.record_ts_buf = np.zeros((0,), dtype=np.float64)
        self.record_imu_buf = np.zeros((0, 9), dtype=np.float32)
        self.record_emg_windows = []
        self.record_angle_targets = []
        self.record_window_ts = []
        self.record_marker_labels = []
        self.record_imu_windows = []
        self.record_skip_no_angle = 0
        self.record_skip_nan_angle = 0
        self.record_skip_old_angle = 0
        self.record_fill_nan_angle = 0
        self.record_last_chunk_id = -1
        self._record_out_auto = None

        # In-GUI compare state (no external process)
        self.compare_active = False
        self.compare_extractor = None
        self.compare_regressor = None
        self.compare_scaler = None
        self.compare_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.compare_ts_buf = np.zeros((0,), dtype=np.float64)
        self.compare_imu_buf = np.zeros((0, 9), dtype=np.float32)
        self.compare_last_chunk_id = -1
        self.compare_window_len = self.record_window_len
        self.compare_overlap = self.record_overlap
        self.compare_max_age = max(
            0.2, (self.record_window_ms / 1000.0) + abs(self.angle_lag_s)
        )
        self.compare_smooth_ms = 150.0
        self.compare_pred_history = []
        self.compare_emg_transform = "log1p"
        self.compare_feature_mode = "bandpower_stats"
        self.compare_target_scaler = None
        self.emg_transform = "log1p"
        self.emg_feature_mode = "bandpower_stats"
        self.angle_scaler_mode = "minmax"
        self.target_spec = DEFAULT_TARGET_SPEC
        self.target_keys = (
            get_target_keys(self.target_spec) if get_target_keys else ANGLE_KEYS
        )
        self.compare_joints = list(self.target_keys)
        self.compare_joints_idx = []

        # Optional EMG filters (load directly to avoid heavy imports at GUI startup)
        try:
            filters_path = (
                Path(__file__).resolve().parents[2]
                / "nml"
                / "gesture_classifier"
                / "filters.py"
            )
            if filters_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "nml_gesture_filters", str(filters_path)
                )
                filters_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(filters_mod)
                self._filter_class = getattr(filters_mod, "BiquadMultiChan", None)
                self._filter_types = getattr(filters_mod, "FilterTypes", None)
            else:
                self._filter_class = None
                self._filter_types = None
            if self._filter_class is None or self._filter_types is None:
                raise RuntimeError("Filter classes not available.")
            self.filter_hp_freq = 4.5
            self.filter_hp_q = 0.5
            self.filter_notch_freq = 50.0
            self.filter_notch_q = 4.0
            self.filter_lp_freq = 100.0
            self.filter_lp_q = 0.5
            self.record_filters = self._build_filters()
        except Exception as exc:
            self._filter_class = None
            self._filter_types = None
            self.record_filters = None
            self.filter_import_error = str(exc)

        self.flow_blocks = {}
        self._init_ui()

        self.monitor = LSLMonitor(
            self.mr_labels, self.emg_stream_name, self.imu_stream_name
        )
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_all)
        self.timer.start(50)

    def _default_feature_extractor_path(self):
        local_path = Path(__file__).parent / "models" / "pretrained_transformer.h5"
        if local_path.exists():
            return local_path
        repo_root = Path(__file__).resolve().parents[3]
        naviflame_path = (
            repo_root / "NaviFlame" / "naviflame" / "models" / "og_fine_tune.h5"
        )
        if naviflame_path.exists():
            return naviflame_path
        return local_path

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(8)

        content_layout.addWidget(self._build_flow_panel())
        content_layout.addWidget(self._build_experiment_panel())

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(self._build_emg_panel(), stretch=0)
        top_row.addWidget(self._build_imu_panel(), stretch=0)
        top_row.addWidget(self._build_lsl_panel(), stretch=0)
        top_row.addWidget(self._build_live_panel(), stretch=1)
        content_layout.addLayout(top_row)
        mid_row = QHBoxLayout()
        mid_row.setSpacing(8)
        mid_row.addWidget(self._build_record_panel(), stretch=1)
        mid_row.addWidget(self._build_prompt_panel(), stretch=1)
        mid_row.addWidget(self._build_filter_panel(), stretch=1)
        content_layout.addLayout(mid_row)

        content_layout.addWidget(self._build_train_panel())

        compare_row = QHBoxLayout()
        compare_row.setSpacing(8)
        compare_row.addWidget(self._build_compare_panel(), stretch=1)
        content_layout.addLayout(compare_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)
        root.addWidget(self._build_log_panel(), stretch=0)

        self._apply_styles()
        self._load_prompt_preview()

    def _build_flow_panel(self):
        group = QGroupBox("Pipeline Flow")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(QLabel("Method"))
        self.flow_method_combo = QComboBox()
        self.flow_method_combo.addItems(
            [
                "Bandpower + MLP (current)",
                "Raw flat + MLP (legacy)",
                "Transformer + MLP (feature extractor)",
            ]
        )
        self.flow_method_combo.currentTextChanged.connect(
            self._on_pipeline_preset_changed
        )
        header.addWidget(self.flow_method_combo)

        self.flow_toggle_details = QCheckBox("Show details")
        self.flow_toggle_details.setChecked(False)
        self.flow_toggle_details.toggled.connect(self._on_flow_details_toggled)
        self.flow_toggle_details.setToolTip("Expand nodes to show inline status text.")
        header.addWidget(self.flow_toggle_details)

        self.flow_toggle_compact = QCheckBox("Compact nodes")
        self.flow_toggle_compact.setChecked(True)
        self.flow_toggle_compact.toggled.connect(self._on_flow_compact_toggled)
        self.flow_toggle_compact.setToolTip("Toggle compact node height.")
        header.addWidget(self.flow_toggle_compact)

        self.flow_toggle_lines = QCheckBox("Show lines")
        self.flow_toggle_lines.setChecked(True)
        self.flow_toggle_lines.toggled.connect(self._on_flow_lines_toggled)
        self.flow_toggle_lines.setToolTip("Show connector lines between nodes.")
        header.addWidget(self.flow_toggle_lines)

        self.flow_snapshot_btn = QPushButton("Save Snapshot")
        self.flow_snapshot_btn.clicked.connect(lambda: self._save_snapshot())
        header.addWidget(self.flow_snapshot_btn)

        self.flow_snapshot_auto = QCheckBox("Auto snapshot")
        self.flow_snapshot_auto.setToolTip(
            "Capture a PNG automatically when the GUI opens."
        )
        self.flow_snapshot_auto.toggled.connect(self._on_snapshot_auto_toggled)
        header.addWidget(self.flow_snapshot_auto)

        legend = QLabel("● OK   ● Error   ● Idle")
        legend.setStyleSheet("color: #9aa3ad;")
        legend.setToolTip("Node border colors reflect status.")
        header.addWidget(legend)
        header.addStretch(1)
        layout.addLayout(header)

        self.flow_canvas = FlowDiagram()
        layout.addWidget(self.flow_canvas)
        self.flow_blocks = self.flow_canvas.blocks
        return group

    def _build_experiment_panel(self):
        group = QGroupBox("Experiment")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 6, 8, 6)

        self.exp_subject = QLineEdit("")
        self.exp_subject.setPlaceholderText("Subject ID")
        self.exp_subject.setMaximumWidth(160)
        self.exp_session = QLineEdit("")
        self.exp_session.setPlaceholderText("Session ID")
        self.exp_session.setMaximumWidth(160)
        self.exp_notes = QLineEdit("")
        self.exp_notes.setPlaceholderText("Notes / run label")
        self.exp_subject.textChanged.connect(self._update_record_path_from_experiment)
        self.exp_session.textChanged.connect(self._update_record_path_from_experiment)
        self.exp_notes.textChanged.connect(self._update_record_path_from_experiment)
        self.exp_wrist = QComboBox()
        self.exp_wrist.addItems(["neutral", "pronated", "supinated", "free", "dynamic"])
        self.exp_wrist.setToolTip("Wrist orientation during the session.")

        layout.addWidget(QLabel("Subject"))
        layout.addWidget(self.exp_subject)
        layout.addWidget(QLabel("Session"))
        layout.addWidget(self.exp_session)
        layout.addWidget(QLabel("Notes"))
        layout.addWidget(self.exp_notes, stretch=1)
        layout.addWidget(QLabel("Wrist"))
        layout.addWidget(self.exp_wrist)

        task_row = QHBoxLayout()
        self.task_start_btn = QPushButton("Start Task")
        self.task_stop_btn = QPushButton("Stop Task")
        self.task_stop_btn.setEnabled(False)
        self.task_start_btn.clicked.connect(self._start_task)
        self.task_stop_btn.clicked.connect(self._stop_task)
        task_row.addWidget(self.task_start_btn)
        task_row.addWidget(self.task_stop_btn)
        layout.addLayout(task_row)
        return group

    def _update_record_path_from_experiment(self):
        if not hasattr(self, "record_out"):
            return
        subject_raw = self.exp_subject.text().strip()
        session_raw = self.exp_session.text().strip()
        notes_raw = self.exp_notes.text().strip()
        wrist_raw = (
            self.exp_wrist.currentText().strip() if hasattr(self, "exp_wrist") else ""
        )
        if not subject_raw and not session_raw:
            return
        safe_subject = subject_raw.replace(" ", "_")
        safe_session = session_raw.replace(" ", "_")
        safe_notes = notes_raw.replace(" ", "_")
        safe_wrist = wrist_raw.replace(" ", "_").lower()
        if safe_subject and not safe_subject.startswith("sub-"):
            safe_subject = f"sub-{safe_subject.zfill(3) if safe_subject.isdigit() else safe_subject}"
        if safe_session and not safe_session.startswith("ses-"):
            safe_session = f"ses-{safe_session.zfill(3) if safe_session.isdigit() else safe_session}"
        base = Path(__file__).parent / "data"
        if safe_subject:
            base = base / safe_subject
        if safe_session:
            base = base / safe_session
        filename_bits = []
        if safe_subject:
            filename_bits.append(safe_subject)
        if safe_session:
            filename_bits.append(safe_session)
        if not filename_bits:
            filename_bits.append("session")
        filename_bits.append("task-jointangles")
        if safe_wrist and safe_wrist not in ("any", "any_wrist"):
            filename_bits.append(f"wrist-{safe_wrist}")
        if safe_notes:
            if not safe_notes.startswith("run-"):
                safe_notes = f"run-{safe_notes}"
            filename_bits.append(safe_notes)
        filename = "_".join(filename_bits) + ".npz"
        new_path = str(base / filename)

        current = self.record_out.text().strip()
        if current and self._record_out_auto and current != self._record_out_auto:
            # Respect manual edits if the user already changed the path.
            return
        self._record_out_auto = new_path
        self.record_out.setText(new_path)

        if hasattr(self, "feat_out_model") and self.feat_out_model:
            feat_name_bits = []
            if safe_subject:
                feat_name_bits.append(safe_subject)
            if safe_session:
                feat_name_bits.append(safe_session)
            if not feat_name_bits:
                feat_name_bits.append("session")
            if safe_wrist and safe_wrist not in ("any", "any_wrist"):
                feat_name_bits.append(f"wrist-{safe_wrist}")
            feat_name_bits.append("feature_extractor")
            feat_name = "_".join(feat_name_bits) + ".h5"
            feat_path = str(base / "models" / feat_name)
            current_feat = self.feat_out_model.text().strip()
            if (
                current_feat
                and self._feat_out_auto
                and current_feat != self._feat_out_auto
            ):
                return
            self._feat_out_auto = feat_path
            self.feat_out_model.setText(feat_path)

        if hasattr(self, "train_out") and self.train_out:
            train_bits = []
            if safe_subject:
                train_bits.append(safe_subject)
            if safe_session:
                train_bits.append(safe_session)
            if not train_bits:
                train_bits.append("session")
            if safe_wrist and safe_wrist not in ("any", "any_wrist"):
                train_bits.append(f"wrist-{safe_wrist}")
            train_bits.append("joint_regressor")
            train_dir = str(base / "models" / "_".join(train_bits))
            current_train = self.train_out.text().strip()
            if (
                current_train
                and self._train_out_auto
                and current_train != self._train_out_auto
            ):
                return
            self._train_out_auto = train_dir
            self.train_out.setText(train_dir)

    def _build_emg_panel(self):
        group = QGroupBox("EMG (LSL)")
        layout = QVBoxLayout(group)

        emg_row = QHBoxLayout()
        emg_row.addWidget(QLabel("Stream"))
        self.emg_stream_name = QLineEdit("OpenEphys_EMG")
        self.emg_stream_name.setMinimumWidth(80)
        self.emg_stream_name.setMaximumWidth(130)
        emg_row.addWidget(self.emg_stream_name, stretch=1)

        status = QLabel("● Disconnected")
        status.setStyleSheet("color: #ff6666; font-weight: bold;")
        shape = QLabel("Shape: N/A")
        rate = QLabel("Rate: N/A Hz")

        layout.addLayout(emg_row)
        layout.addWidget(status)
        layout.addWidget(shape)
        layout.addWidget(rate)
        group.setMinimumWidth(140)
        group.setMaximumWidth(200)

        self.mr_labels = {
            "status": status,
            "shape": shape,
            "rate": rate,
        }
        return group

    def _build_imu_panel(self):
        group = QGroupBox("IMU (LSL)")
        layout = QVBoxLayout(group)

        imu_row = QHBoxLayout()
        imu_row.addWidget(QLabel("Stream"))
        self.imu_stream_name = QLineEdit("OpenEphys_IMU")
        self.imu_stream_name.setMinimumWidth(80)
        self.imu_stream_name.setMaximumWidth(130)
        imu_row.addWidget(self.imu_stream_name, stretch=1)

        imu_status = QLabel("● Disconnected")
        imu_status.setStyleSheet("color: #ff6666; font-weight: bold;")
        imu_shape = QLabel("Shape: N/A")
        imu_rate = QLabel("Rate: N/A Hz")

        layout.addLayout(imu_row)
        layout.addWidget(imu_status)
        layout.addWidget(imu_shape)
        layout.addWidget(imu_rate)
        group.setMinimumWidth(140)
        group.setMaximumWidth(200)

        self.mr_labels.update(
            {
                "imu_status": imu_status,
                "imu_shape": imu_shape,
                "imu_rate": imu_rate,
            }
        )
        return group

    def _build_lsl_panel(self):
        group = QGroupBox("Hand Tracking (LSL)")
        layout = QVBoxLayout(group)

        self.angle_stream_name = QLineEdit("StereoHandTracker_Angles")
        self.marker_stream_name = QLineEdit(
            "StereoHandTracker_Landmarks"
        )  # Receives hand landmarks from mocap broadcaster
        self.angle_stream_name.setMinimumWidth(120)
        self.angle_stream_name.setMaximumWidth(180)
        self.marker_stream_name.setMinimumWidth(120)
        self.marker_stream_name.setMaximumWidth(180)

        angle_row = QHBoxLayout()
        angle_row.addWidget(QLabel("Angles"))
        angle_row.addWidget(self.angle_stream_name, stretch=1)
        layout.addLayout(angle_row)

        self.angle_status = QLabel("● Waiting...")
        self.angle_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        layout.addWidget(self.angle_status)

        marker_row = QHBoxLayout()
        marker_row.addWidget(QLabel("Markers"))
        marker_row.addWidget(self.marker_stream_name, stretch=1)
        layout.addLayout(marker_row)

        self.marker_status = QLabel("● Waiting...")
        self.marker_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        layout.addWidget(self.marker_status)

        self.hand_rate_label = QLabel("Rate: N/A Hz")
        layout.addWidget(self.hand_rate_label)

        self.arm_button = QPushButton("Arm Recording")
        self.arm_button.setCheckable(True)
        self.arm_button.setChecked(True)
        self.arm_button.toggled.connect(self._on_arm_toggled)
        self.arm_button.setVisible(False)
        layout.addWidget(self.arm_button)

        group.setMinimumWidth(180)
        group.setMaximumWidth(240)

        return group

    def _build_prompt_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(6)

        self.prompt_plan = QLineEdit(
            str(Path(__file__).parent / "prompts_default.json")
        )
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self.prompt_plan)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(
            lambda: self._pick_file(self.prompt_plan, "Select prompt plan")
        )
        prompt_row.addWidget(btn_browse)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_prompt_preview)
        prompt_row.addWidget(btn_load)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setPlaceholderText(
            "Prompt plan preview will appear here..."
        )

        layout.addWidget(QLabel("Plan"))
        layout.addLayout(prompt_row)
        layout.addWidget(QLabel("Preview"))
        layout.addWidget(self.prompt_preview)

        advanced = QGroupBox("Advanced controls")
        adv_layout = QVBoxLayout(advanced)
        prompts_row = QHBoxLayout()
        btn_prompts_start = QPushButton("Start Prompts")
        btn_prompts_stop = QPushButton("Stop Prompts")
        btn_prompts_stop.setEnabled(False)
        btn_prompts_start.clicked.connect(
            lambda: self._start_prompts(btn_prompts_start, btn_prompts_stop)
        )
        btn_prompts_stop.clicked.connect(
            lambda: self._stop_process("prompts", btn_prompts_start, btn_prompts_stop)
        )
        prompts_row.addWidget(btn_prompts_start)
        prompts_row.addWidget(btn_prompts_stop)
        adv_layout.addLayout(prompts_row)

        record_row = QHBoxLayout()
        btn_record_start = QPushButton("Start Recording")
        btn_record_stop = QPushButton("Stop Recording")
        btn_record_stop.setEnabled(False)
        btn_record_start.clicked.connect(
            lambda: self._start_record(btn_record_start, btn_record_stop)
        )
        btn_record_stop.clicked.connect(
            lambda: self._stop_process("record", btn_record_start, btn_record_stop)
        )
        record_row.addWidget(btn_record_start)
        record_row.addWidget(btn_record_stop)
        adv_layout.addLayout(record_row)
        adv_layout.addWidget(
            QLabel("Use Start Task for one-click recording + prompts.")
        )

        self.prompts_start_btn = btn_prompts_start
        self.prompts_stop_btn = btn_prompts_stop
        self.record_start_btn = btn_record_start
        self.record_stop_btn = btn_record_stop
        layout.addWidget(advanced)

        section = CollapsibleSection("Task Prompter", content, default_open=False)
        return section

    def _build_record_panel(self):
        content = QWidget()
        layout = QGridLayout(content)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        default_out = self._next_session_path()
        self.record_out = QLineEdit(str(default_out))
        out_row = QHBoxLayout()
        out_row.addWidget(self.record_out)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(
            lambda: self._pick_save(self.record_out, "Save recording")
        )
        out_row.addWidget(btn_out)
        self.record_duration_spin = QSpinBox()
        self.record_duration_spin.setRange(10, 3600)
        self.record_duration_spin.setValue(300)
        self.record_window_spin = QSpinBox()
        self.record_window_spin.setRange(20, 1000)
        self.record_window_spin.setValue(int(self.record_window_ms))
        self.record_window_spin.valueChanged.connect(self._update_window_labels)
        self.record_overlap_spin = QSpinBox()
        self.record_overlap_spin.setRange(0, 900)
        self.record_overlap_spin.setValue(int(self.record_overlap_ms))
        self.record_overlap_spin.valueChanged.connect(self._update_window_labels)
        self.record_window_info = QLabel("")
        self._update_window_labels()
        self.angle_lag_spin = QDoubleSpinBox()
        self.angle_lag_spin.setRange(-500.0, 500.0)
        self.angle_lag_spin.setDecimals(1)
        self.angle_lag_spin.setSingleStep(5.0)
        self.angle_lag_spin.setValue(self.angle_lag_ms)
        self.record_hand = QComboBox()
        self.record_hand.addItems(["Slot 0 (remapped)", "Slot 1"])

        self.target_angle_combo = QComboBox()
        self.target_angle_combo.addItem("All 14 joints (full14)", "full14")
        self.target_angle_combo.addItem("5 MCP flexion (finger5)", "finger5")
        self.target_angle_combo.addItem("Index only (3 DOF)", "index_only")
        # Set initial selection to match current target_spec
        for i in range(self.target_angle_combo.count()):
            if self.target_angle_combo.itemData(i) == self.target_spec:
                self.target_angle_combo.setCurrentIndex(i)
                break
        self.target_angle_combo.currentIndexChanged.connect(self._on_target_spec_changed)

        row = 0
        layout.addWidget(QLabel("Output"), row, 0)
        layout.addLayout(out_row, row, 1)
        row += 1
        layout.addWidget(QLabel("Duration (s)"), row, 0)
        layout.addWidget(self.record_duration_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("Window (ms)"), row, 0)
        layout.addWidget(self.record_window_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("Overlap (ms)"), row, 0)
        layout.addWidget(self.record_overlap_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("Window info"), row, 0)
        layout.addWidget(self.record_window_info, row, 1)
        row += 1
        layout.addWidget(QLabel("Angle lag (ms)"), row, 0)
        layout.addWidget(self.angle_lag_spin, row, 1)
        row += 1
        layout.addWidget(QLabel("Hand slot"), row, 0)
        layout.addWidget(self.record_hand, row, 1)
        row += 1
        layout.addWidget(QLabel("Target angles"), row, 0)
        layout.addWidget(self.target_angle_combo, row, 1)
        section = CollapsibleSection("Recording", content, default_open=False)
        return section

    def _build_train_panel(self):
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setSpacing(10)

        self.train_data = QLineEdit(
            str(Path(__file__).parent / "data" / "session_001.npz")
        )
        self.train_feat = QLineEdit("none")
        self.train_out = QLineEdit(
            str(Path(__file__).parent / "models" / "joint_regressor")
        )
        data_row = QHBoxLayout()
        data_row.addWidget(self.train_data)
        btn_data = QPushButton("Browse")
        btn_data.clicked.connect(
            lambda: self._pick_file(self.train_data, "Select dataset")
        )
        data_row.addWidget(btn_data)
        feat_row = QHBoxLayout()
        feat_row.addWidget(self.train_feat)
        btn_feat = QPushButton("Browse")
        btn_feat.clicked.connect(
            lambda: self._pick_file(self.train_feat, "Select feature extractor")
        )
        feat_row.addWidget(btn_feat)
        out_row = QHBoxLayout()
        out_row.addWidget(self.train_out)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(
            lambda: self._pick_dir(self.train_out, "Select output directory")
        )
        out_row.addWidget(btn_out)

        left_col = QWidget()
        left_layout = QGridLayout(left_col)
        left_layout.setHorizontalSpacing(8)
        left_layout.setVerticalSpacing(6)

        feat_group = QGroupBox("Feature Extraction")
        feat_layout = QVBoxLayout(feat_group)
        feat_layout.setSpacing(6)

        self.feature_source_combo = QComboBox()
        self.feature_source_combo.addItems(
            [
                "None (raw/bandpower only)",
                "Transformer (.h5 feature extractor)",
                "CNN (train below)",
            ]
        )
        self.feature_source_combo.setToolTip(
            "Select how features are produced:\n"
            "- None: use bandpower/raw features directly\n"
            "- Transformer: load a pretrained feature extractor (.h5)\n"
            "- CNN: use the CNN trained below as the feature extractor"
        )
        self.feature_source_combo.currentTextChanged.connect(
            self._on_feature_source_changed
        )

        self.pipeline_preset_combo = QComboBox()
        self.pipeline_preset_combo.addItems(
            [
                "Bandpower + MLP (current)",
                "Raw flat + MLP (legacy)",
                "Transformer + MLP (feature extractor)",
            ]
        )
        self.pipeline_preset_combo.currentTextChanged.connect(
            self._on_pipeline_preset_changed
        )

        if hasattr(self, "flow_method_combo"):
            self.flow_method_combo.blockSignals(True)
            self.flow_method_combo.setCurrentText(
                self.pipeline_preset_combo.currentText()
            )
            self.flow_method_combo.blockSignals(False)

        self.train_emg_transform_combo = QComboBox()
        self.train_emg_transform_combo.addItems(["log1p", "none"])
        self.train_emg_transform_combo.setToolTip(
            "Transform applied to EMG features.\nlog1p compresses dynamic range."
        )
        self.train_emg_transform_combo.setCurrentText(self.emg_transform)
        self.train_emg_transform_combo.currentTextChanged.connect(
            self._on_pipeline_param_changed
        )

        self.train_feature_combo = QComboBox()
        self.train_feature_combo.addItems(["bandpower_stats", "raw_flat"])
        self.train_feature_combo.setToolTip(
            "EMG feature extraction.\nbandpower_stats = mean/var per band;\nraw_flat = flatten window."
        )
        self.train_feature_combo.setCurrentText(self.emg_feature_mode)
        self.train_feature_combo.currentTextChanged.connect(
            self._on_pipeline_param_changed
        )

        self.train_angle_scaler_combo = QComboBox()
        self.train_angle_scaler_combo.addItems(["minmax", "standard", "none"])
        self.train_angle_scaler_combo.setToolTip("Scale target angles before training.")
        self.train_angle_scaler_combo.setCurrentText(self.angle_scaler_mode)
        self.train_angle_scaler_combo.currentTextChanged.connect(
            self._on_pipeline_param_changed
        )

        self.train_use_imu = QCheckBox("Use IMU features")
        self.train_use_imu.setChecked(True)
        self.train_use_imu.setToolTip("Include IMU stats if present in dataset.")

        self.pipeline_help = QLabel(
            "Bandpower: stats per EMG band → MLP\n"
            "Raw flat: flatten window → MLP\n"
            "Transformer: external feature extractor → MLP"
        )
        self.pipeline_help.setStyleSheet("color: #9aa3ad;")
        self.pipeline_help.setWordWrap(True)
        self.pipeline_expected = QLabel("Expected EMG shape: (N, 8, window, 1)")
        self.pipeline_expected.setStyleSheet("color: #8f99a6;")

        feat_layout.addWidget(QLabel("Feature source"))
        feat_layout.addWidget(self.feature_source_combo)
        feat_layout.addWidget(QLabel("Preset"))
        feat_layout.addWidget(self.pipeline_preset_combo)
        feat_layout.addWidget(QLabel("EMG transform"))
        feat_layout.addWidget(self.train_emg_transform_combo)
        feat_layout.addWidget(QLabel("EMG feature mode"))
        feat_layout.addWidget(self.train_feature_combo)
        feat_layout.addWidget(QLabel("Angle scaler"))
        feat_layout.addWidget(self.train_angle_scaler_combo)
        feat_layout.addWidget(self.train_use_imu)
        feat_layout.addWidget(self.pipeline_expected)
        feat_layout.addWidget(self.pipeline_help)

        header_actions = QWidget()
        btn_row = QHBoxLayout(header_actions)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        btn_start = QPushButton("Train Regressor")
        btn_stop = QPushButton("Stop Train")
        btn_stop.setEnabled(False)
        btn_start.clicked.connect(lambda: self._start_train(btn_start, btn_stop))
        btn_stop.clicked.connect(
            lambda: self._stop_process("train", btn_start, btn_stop)
        )
        btn_row.addWidget(btn_start)
        btn_row.addWidget(btn_stop)

        row = 0
        left_layout.addWidget(QLabel("Dataset (.npz)"), row, 0)
        left_layout.addLayout(data_row, row, 1)
        row += 1
        left_layout.addWidget(QLabel("Feature extractor"), row, 0)
        left_layout.addLayout(feat_row, row, 1)
        row += 1
        left_layout.addWidget(QLabel("Output dir"), row, 0)
        left_layout.addLayout(out_row, row, 1)
        row += 1
        self.train_mae_thresh = QDoubleSpinBox()
        self.train_mae_thresh.setRange(0.1, 30.0)
        self.train_mae_thresh.setDecimals(1)
        self.train_mae_thresh.setSingleStep(0.5)
        self.train_mae_thresh.setValue(5.0)
        self.train_r2_thresh = QDoubleSpinBox()
        self.train_r2_thresh.setRange(0.0, 0.99)
        self.train_r2_thresh.setDecimals(2)
        self.train_r2_thresh.setSingleStep(0.05)
        self.train_r2_thresh.setValue(0.85)
        left_layout.addWidget(QLabel("MAE cutoff (deg)"), row, 0)
        left_layout.addWidget(self.train_mae_thresh, row, 1)
        row += 1
        left_layout.addWidget(QLabel("R² cutoff"), row, 0)
        left_layout.addWidget(self.train_r2_thresh, row, 1)
        row += 1
        self.train_pass_label = QLabel("Result: N/A")
        self.train_pass_label.setStyleSheet("color: #9aa3ad;")
        left_layout.addWidget(self.train_pass_label, row, 0, 1, 2)
        row += 1
        left_layout.addWidget(feat_group, row, 0, 1, 2)

        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setSpacing(6)

        feat_train_group = QGroupBox("Feature Extractor (CNN)")
        feat_train_layout = QGridLayout(feat_train_group)
        feat_train_layout.setHorizontalSpacing(8)
        feat_train_layout.setVerticalSpacing(6)

        self.feat_data_paths = []
        self.feat_data_display = QLineEdit("")
        self.feat_data_display.setReadOnly(True)
        self.feat_data_display.setPlaceholderText("Select one or more .npz files")
        btn_pick_feat = QPushButton("Select datasets")
        btn_pick_feat.clicked.connect(self._pick_feature_datasets)
        btn_add_subject = QPushButton("Add subject")
        btn_add_subject.clicked.connect(self._add_subject_sessions)
        btn_add_all = QPushButton("Add all subjects")
        btn_add_all.clicked.connect(self._add_all_sessions)
        btn_clear_feat = QPushButton("Clear")
        btn_clear_feat.clicked.connect(self._clear_feature_datasets)

        feat_data_row = QHBoxLayout()
        feat_data_row.addWidget(self.feat_data_display, stretch=1)
        feat_data_row.addWidget(btn_pick_feat)
        feat_data_row.addWidget(btn_add_subject)
        feat_data_row.addWidget(btn_add_all)
        feat_data_row.addWidget(btn_clear_feat)

        self.feat_out_model = QLineEdit(
            str(Path(__file__).parent / "models" / "feature_extractor.h5")
        )
        feat_out_row = QHBoxLayout()
        feat_out_row.addWidget(self.feat_out_model)
        btn_feat_out = QPushButton("Browse")
        btn_feat_out.clicked.connect(
            lambda: self._pick_save(self.feat_out_model, "Save feature extractor")
        )
        feat_out_row.addWidget(btn_feat_out)
        self.feat_out_model.textChanged.connect(
            self._on_feature_extractor_output_changed
        )
        self.feat_data_summary = QLabel("No datasets selected.")
        self.feat_data_summary.setStyleSheet("color: #8f99a6;")
        self.feat_wrist_filter = QComboBox()
        self.feat_wrist_filter.addItems(
            ["Any wrist", "neutral", "pronated", "supinated", "free", "dynamic"]
        )
        self.feat_wrist_filter.setToolTip(
            "Filter datasets by recorded wrist orientation metadata."
        )
        self.feat_status_label = QLabel("Status: idle")
        self.feat_status_label.setStyleSheet("color: #9aa3ad;")

        self.feat_epochs = QSpinBox()
        self.feat_epochs.setRange(1, 500)
        self.feat_epochs.setValue(50)
        self.feat_batch = QSpinBox()
        self.feat_batch.setRange(1, 1024)
        self.feat_batch.setValue(64)
        self.feat_val_split = QDoubleSpinBox()
        self.feat_val_split.setRange(0.05, 0.5)
        self.feat_val_split.setDecimals(2)
        self.feat_val_split.setSingleStep(0.05)
        self.feat_val_split.setValue(0.2)
        self.feat_dim = QSpinBox()
        self.feat_dim.setRange(8, 512)
        self.feat_dim.setValue(128)
        self.feat_lr = QDoubleSpinBox()
        self.feat_lr.setRange(1e-5, 1e-2)
        self.feat_lr.setDecimals(5)
        self.feat_lr.setSingleStep(0.0001)
        self.feat_lr.setValue(0.001)
        self.feat_side = QComboBox()
        self.feat_side.addItems(["", "left", "right"])

        feat_row = 0
        feat_train_layout.addWidget(QLabel("Datasets (.npz)"), feat_row, 0)
        feat_train_layout.addLayout(feat_data_row, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Dataset check"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_data_summary, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Wrist filter"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_wrist_filter, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Train status"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_status_label, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Output model"), feat_row, 0)
        feat_train_layout.addLayout(feat_out_row, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Epochs"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_epochs, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Batch size"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_batch, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Val split"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_val_split, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Feature dim"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_dim, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Learning rate"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_lr, feat_row, 1)
        feat_row += 1
        feat_train_layout.addWidget(QLabel("Side label"), feat_row, 0)
        feat_train_layout.addWidget(self.feat_side, feat_row, 1)

        self.feat_start_btn = QPushButton("Train Feature Extractor")
        self.feat_stop_btn = QPushButton("Stop")
        self.feat_stop_btn.setEnabled(False)
        self.feat_start_btn.clicked.connect(
            lambda: self._start_feature_extractor(
                self.feat_start_btn, self.feat_stop_btn
            )
        )
        self.feat_stop_btn.clicked.connect(
            lambda: self._stop_process("feat", self.feat_start_btn, self.feat_stop_btn)
        )
        feat_btn_row = QHBoxLayout()
        feat_btn_row.addWidget(self.feat_start_btn)
        feat_btn_row.addWidget(self.feat_stop_btn)
        feat_train_layout.addLayout(feat_btn_row, feat_row + 1, 0, 1, 2)

        right_layout.addWidget(feat_train_group)

        layout.addWidget(left_col, stretch=1)
        layout.addWidget(right_col, stretch=1)
        section = CollapsibleSection(
            "Training", content, default_open=False, header_widget=header_actions
        )
        self._on_feature_source_changed(self.feature_source_combo.currentText())
        return section

    def _build_compare_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(6)

        self.compare_feat = QLineEdit("none")
        self.compare_reg = QLineEdit(
            str(
                Path(__file__).parent
                / "models"
                / "joint_regressor"
                / "mlp_regressor.pkl"
            )
        )
        self.compare_scaler = QLineEdit(
            str(Path(__file__).parent / "models" / "joint_regressor" / "scaler.pkl")
        )
        self.compare_smooth_spin = QSpinBox()
        self.compare_smooth_spin.setRange(0, 1000)
        self.compare_smooth_spin.setValue(int(self.compare_smooth_ms))
        feat_row = QHBoxLayout()
        feat_row.addWidget(self.compare_feat)
        btn_feat = QPushButton("Browse")
        btn_feat.clicked.connect(
            lambda: self._pick_file(self.compare_feat, "Select feature extractor")
        )
        feat_row.addWidget(btn_feat)
        reg_row = QHBoxLayout()
        reg_row.addWidget(self.compare_reg)
        btn_reg = QPushButton("Browse")
        btn_reg.clicked.connect(
            lambda: self._pick_file(self.compare_reg, "Select regressor")
        )
        reg_row.addWidget(btn_reg)
        scaler_row = QHBoxLayout()
        scaler_row.addWidget(self.compare_scaler)
        btn_scaler = QPushButton("Browse")
        btn_scaler.clicked.connect(
            lambda: self._pick_file(self.compare_scaler, "Select scaler")
        )
        scaler_row.addWidget(btn_scaler)

        header_actions = QWidget()
        btn_row = QHBoxLayout(header_actions)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        btn_start = QPushButton("Start Compare")
        btn_stop = QPushButton("Stop Compare")
        btn_stop.setEnabled(False)
        btn_start.clicked.connect(lambda: self._start_compare_gui(btn_start, btn_stop))
        btn_stop.clicked.connect(lambda: self._stop_compare_gui(btn_start, btn_stop))
        btn_row.addWidget(btn_start)
        btn_row.addWidget(btn_stop)

        layout.addWidget(QLabel("Feature extractor"))
        layout.addLayout(feat_row)
        layout.addWidget(QLabel("Regressor"))
        layout.addLayout(reg_row)
        layout.addWidget(QLabel("Scaler"))
        layout.addLayout(scaler_row)
        layout.addWidget(QLabel("Smoothing (ms)"))
        layout.addWidget(self.compare_smooth_spin)
        note = QLabel("Uses metrics.json if present, otherwise GUI settings.")
        note.setStyleSheet("color: #9aa3ad;")
        layout.addWidget(note)
        layout.addWidget(QLabel("Output: in-GUI Compare status"))
        section = CollapsibleSection(
            "Live Compare", content, default_open=False, header_widget=header_actions
        )
        return section

    def _build_live_panel(self):
        group = QGroupBox("Live Metrics")
        layout = QVBoxLayout(group)

        # Connection status indicators
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(8)
        status_grid.setVerticalSpacing(4)

        self.mr_led = QLabel("●")
        self.mr_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        self.angle_led = QLabel("●")
        self.angle_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        self.record_led = QLabel("●")
        self.record_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        self.prompts_led = QLabel("●")
        self.prompts_led.setStyleSheet("color: #ff6666; font-size: 16px;")

        emg_widget = QWidget()
        emg_row = QHBoxLayout(emg_widget)
        emg_row.setContentsMargins(0, 0, 0, 0)
        emg_row.setSpacing(4)
        emg_row.addWidget(self.mr_led)
        emg_row.addWidget(QLabel("EMG (LSL)"))

        angle_widget = QWidget()
        angle_row = QHBoxLayout(angle_widget)
        angle_row.setContentsMargins(0, 0, 0, 0)
        angle_row.setSpacing(4)
        angle_row.addWidget(self.angle_led)
        angle_row.addWidget(QLabel("Angles (LSL)"))

        rec_widget = QWidget()
        rec_row = QHBoxLayout(rec_widget)
        rec_row.setContentsMargins(0, 0, 0, 0)
        rec_row.setSpacing(4)
        rec_row.addWidget(self.record_led)
        rec_row.addWidget(QLabel("Rec"))

        prompts_widget = QWidget()
        prompts_row = QHBoxLayout(prompts_widget)
        prompts_row.setContentsMargins(0, 0, 0, 0)
        prompts_row.setSpacing(4)
        prompts_row.addWidget(self.prompts_led)
        prompts_row.addWidget(QLabel("Prompts"))

        status_grid.addWidget(emg_widget, 0, 0)
        status_grid.addWidget(angle_widget, 0, 1)
        status_grid.addWidget(rec_widget, 1, 0)
        status_grid.addWidget(prompts_widget, 1, 1)
        layout.addLayout(status_grid)

        # Metrics
        self.last_angle_label = QLabel("Angles: 0")
        self.angle_age_label = QLabel("Age: N/A")
        self.ready_status = QLabel("Ready: NO")
        self.ready_status.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.record_windows_label = QLabel("Windows: 0")
        self.emg_buf_label = QLabel("EMG buf: 0")

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(4)
        metrics_grid.addWidget(self.last_angle_label, 0, 0)
        metrics_grid.addWidget(self.angle_age_label, 0, 1)
        metrics_grid.addWidget(self.ready_status, 1, 0)
        metrics_grid.addWidget(self.record_windows_label, 1, 1)
        metrics_grid.addWidget(self.emg_buf_label, 2, 0, 1, 2)
        layout.addLayout(metrics_grid)

        self.angle_match_label = QLabel("Angle match: N/A")
        layout.addWidget(self.angle_match_label)

        self.compare_label = QLabel("Compare: N/A")
        self.compare_label.setWordWrap(True)
        layout.addWidget(self.compare_label)

        self.compare_bar_label = QLabel("")
        self.compare_bar_label.setStyleSheet("font-family: Consolas, monospace;")
        self.compare_bar_label.setWordWrap(True)
        self.compare_bar_label.setVisible(False)
        layout.addWidget(self.compare_bar_label)

        group.setMinimumWidth(210)
        return group

    def _build_filter_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(6)

        if self._filter_class is None:
            msg = "Filters unavailable (missing nml.gesture_classifier.filters)"
            if hasattr(self, "filter_import_error"):
                msg += f": {self.filter_import_error}"
            layout.addWidget(QLabel(msg))
            return CollapsibleSection("EMG Filters", content, default_open=False)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("HPF Hz"))
        self.hp_freq = QDoubleSpinBox()
        self.hp_freq.setRange(0.5, 20.0)
        self.hp_freq.setDecimals(2)
        self.hp_freq.setValue(self.filter_hp_freq)
        self.hp_freq.valueChanged.connect(self._on_filter_params_changed)
        row1.addWidget(self.hp_freq)
        row1.addWidget(QLabel("Q"))
        self.hp_q = QDoubleSpinBox()
        self.hp_q.setRange(0.1, 10.0)
        self.hp_q.setDecimals(2)
        self.hp_q.setValue(self.filter_hp_q)
        self.hp_q.valueChanged.connect(self._on_filter_params_changed)
        row1.addWidget(self.hp_q)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Notch Hz"))
        self.notch_freq = QDoubleSpinBox()
        self.notch_freq.setRange(40.0, 70.0)
        self.notch_freq.setDecimals(2)
        self.notch_freq.setValue(self.filter_notch_freq)
        self.notch_freq.valueChanged.connect(self._on_filter_params_changed)
        row2.addWidget(self.notch_freq)
        row2.addWidget(QLabel("Q"))
        self.notch_q = QDoubleSpinBox()
        self.notch_q.setRange(0.5, 10.0)
        self.notch_q.setDecimals(2)
        self.notch_q.setValue(self.filter_notch_q)
        self.notch_q.valueChanged.connect(self._on_filter_params_changed)
        row2.addWidget(self.notch_q)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("LPF Hz"))
        self.lp_freq = QDoubleSpinBox()
        self.lp_freq.setRange(20.0, 300.0)
        self.lp_freq.setDecimals(1)
        self.lp_freq.setValue(self.filter_lp_freq)
        self.lp_freq.valueChanged.connect(self._on_filter_params_changed)
        row3.addWidget(self.lp_freq)
        row3.addWidget(QLabel("Q"))
        self.lp_q = QDoubleSpinBox()
        self.lp_q.setRange(0.1, 10.0)
        self.lp_q.setDecimals(2)
        self.lp_q.setValue(self.filter_lp_q)
        self.lp_q.valueChanged.connect(self._on_filter_params_changed)
        row3.addWidget(self.lp_q)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addWidget(QLabel(f"Fs: {self.record_fs:.0f} Hz"))
        return CollapsibleSection("EMG Filters", content, default_open=False)

    def _build_log_panel(self):
        group = QGroupBox("Console Output")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Process output will appear here...")
        self.log_view.setMinimumHeight(100)
        self.log_view.setMaximumHeight(150)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.log_view)
        return group

    def _poll_all(self):
        self.monitor.poll()
        self._poll_angle_stream()
        self._poll_marker_stream()
        self._update_ready_state()
        # QProcess output handled via signals
        self._watch_prompt_plan()
        self._record_tick()
        self._compare_tick()

    def _check_lsl(self, stream_name, label, kind):
        if not HAS_LSL or resolve_byprop is None:
            label.setText("● LSL unavailable")
            label.setStyleSheet("color: #ff6666; font-weight: bold;")
            return
        if not stream_name:
            label.setText("● Waiting...")
            label.setStyleSheet("color: #ffaa00; font-weight: bold;")
            return
        try:
            # Keep status checks fast to avoid UI hitching.
            streams = resolve_byprop("name", stream_name, timeout=0.01)
            if streams:
                # Don't overwrite "Stale" – data-flow staleness takes priority
                # over the LSL resolver cache which lingers after disconnect.
                if "Stale" not in label.text():
                    label.setText("● Connected")
                    label.setStyleSheet("color: #44ff44; font-weight: bold;")
            else:
                label.setText("● Waiting...")
                label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        except Exception:
            label.setText("● Error")
            label.setStyleSheet("color: #ff6666; font-weight: bold;")

    def _rescan_lsl(self):
        if not HAS_LSL or resolve_byprop is None:
            self._append_log("[LSL] pylsl not available")
            return
        try:
            # Check if all input fields are empty
            angle_name = self.angle_stream_name.text().strip()
            marker_name = self.marker_stream_name.text().strip()
            emg_name = self.emg_stream_name.text().strip()
            imu_name = self.imu_stream_name.text().strip()

            if not angle_name and not marker_name and not emg_name and not imu_name:
                # Show all available LSL streams
                if resolve_streams is None:
                    self._append_log("[LSL] resolve_streams not available")
                    return
                streams = resolve_streams(wait_time=1.0)
                if streams:
                    self._append_log(f"[LSL] Found {len(streams)} stream(s):")
                    for s in streams:
                        stream_type = s.type() if hasattr(s, "type") else "N/A"
                        stream_name = s.name() if hasattr(s, "name") else "N/A"
                        stream_channels = (
                            s.channel_count() if hasattr(s, "channel_count") else "N/A"
                        )
                        self._append_log(
                            f"  - {stream_name} (type: {stream_type}, channels: {stream_channels})"
                        )
                else:
                    self._append_log("[LSL] No streams found")
            else:
                checks = [
                    ("Angles", angle_name),
                    ("Markers", marker_name),
                    ("EMG", emg_name),
                    ("IMU", imu_name),
                ]
                for kind, stream_name in checks:
                    if not stream_name:
                        continue
                    streams = resolve_byprop("name", stream_name, timeout=0.2)
                    if streams:
                        info = streams[0]
                        self._append_log(
                            f"[LSL] {kind}: OK ({stream_name}, type={info.type()}, ch={info.channel_count()})"
                        )
                    else:
                        self._append_log(f"[LSL] {kind}: Not found ({stream_name})")
        except Exception as exc:
            self._append_log(f"[LSL] Rescan error: {exc}")

    def _poll_angle_stream(self):
        if not HAS_LSL or StreamInlet is None:
            return
        name = self.angle_stream_name.text().strip()
        if not name:
            self.angle_inlet = None
            self.angle_status.setText("\u25cf Waiting...")
            self.angle_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
            self.hand_rate_label.setText("Rate: N/A Hz")
            return
        # Resolve inlet if we don't have one (throttled to reduce UI blocking)
        if self.angle_inlet is None:
            now = _clock()
            if now - self.last_angle_inlet_resolve >= self.inlet_resolve_interval_s:
                self.last_angle_inlet_resolve = now
                try:
                    streams = resolve_byprop("name", name, timeout=0.01)
                    if streams:
                        self.angle_inlet = StreamInlet(streams[0])
                except Exception:
                    self.angle_inlet = None
        # If data has been stale for >2 s, consider the stream disconnected:
        # null out the inlet so the system transitions to "waiting".
        if self.angle_inlet and self.last_angle_seen:
            stale_age = _clock() - self.last_angle_seen
            if stale_age > 2.0:
                self.angle_inlet = None
                self.angle_count = 0
                self.last_angle_seen = 0.0
                self.last_angle_rate_hz = 0.0
        # Pull data
        if self.angle_inlet:
            try:
                samples, timestamps = self.angle_inlet.pull_chunk(
                    timeout=0.0, max_samples=32
                )
            except Exception:
                self.angle_inlet = None
                samples, timestamps = [], []
            if samples:
                now = _clock()
                chunk_count = len(samples)
                if timestamps and len(timestamps) >= 2:
                    span = max(float(timestamps[-1] - timestamps[0]), 1e-6)
                    self.last_angle_rate_hz = float((len(timestamps) - 1) / span)
                elif self.last_angle_seen:
                    dt = max(now - self.last_angle_seen, 1e-6)
                    self.last_angle_rate_hz = float(chunk_count / dt)
                for idx, sample in enumerate(samples):
                    ts = None
                    if timestamps and idx < len(timestamps):
                        ts = timestamps[idx]
                    tstamp = ts if ts else now
                    self.angle_count += 1
                    self.angle_buffer.append((tstamp, sample))
                self.last_angle_seen = now
                if len(self.angle_buffer) > 2000:
                    self.angle_buffer = self.angle_buffer[-2000:]
                # Provide live angle-match status even when not recording
                if (
                    self.monitor
                    and self.monitor.last_emg_ts is not None
                    and self.monitor.last_emg_ts.size
                ):
                    self._update_angle_match_status(
                        self.monitor.last_emg_ts[-1], window_ready=True
                    )
                else:
                    if self.angle_match_label:
                        self.angle_match_label.setText("Angle match: waiting for EMG")
        # --- Drive status label from actual data flow ---
        age = _clock() - self.last_angle_seen if self.last_angle_seen else None
        if age is not None and age < 1.0:
            self.angle_status.setText("\u25cf Connected")
            self.angle_status.setStyleSheet("color: #44ff44; font-weight: bold;")
        elif age is not None and age < 2.0:
            self.angle_status.setText("\u25cf Stale")
            self.angle_status.setStyleSheet("color: #d9b44a; font-weight: bold;")
        elif self.angle_inlet is not None and self.angle_count == 0:
            # Inlet resolved but no data yet
            self.angle_status.setText("\u25cf Connected")
            self.angle_status.setStyleSheet("color: #44ff44; font-weight: bold;")
        else:
            self.angle_status.setText("\u25cf Waiting...")
            self.angle_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.last_angle_label.setText(f"Angles: {self.angle_count}")
        self.angle_age_label.setText(
            f"Age: {age:.2f}s" if age is not None else "Age: N/A"
        )
        self.hand_rate_label.setText(
            f"Rate: {self.last_angle_rate_hz:.1f} Hz"
            if self.last_angle_rate_hz > 0
            else "Rate: N/A Hz"
        )

    def _poll_marker_stream(self):
        if not HAS_LSL or StreamInlet is None:
            return
        name = self.marker_stream_name.text().strip()
        if not name:
            self.marker_inlet = None
            self.marker_status.setText("\u25cf Waiting...")
            self.marker_status.setStyleSheet("color: #ffaa00; font-weight: bold;")
            return
        # Resolve inlet if we don't have one (throttled)
        if self.marker_inlet is None:
            now = _clock()
            if now - self.last_marker_inlet_resolve >= self.inlet_resolve_interval_s:
                self.last_marker_inlet_resolve = now
                try:
                    streams = resolve_byprop(
                        "name", name, timeout=0.01
                    )
                    if streams:
                        self.marker_inlet = StreamInlet(streams[0])
                except Exception:
                    self.marker_inlet = None
        if self.marker_inlet:
            try:
                samples, timestamps = self.marker_inlet.pull_chunk(
                    timeout=0.0, max_samples=32
                )
            except Exception:
                self.marker_inlet = None
                samples, timestamps = [], []
            if samples:
                now = _clock()
                self.last_marker_seen = now
                for idx, sample in enumerate(samples):
                    ts = None
                    if timestamps and idx < len(timestamps):
                        ts = timestamps[idx]
                    tstamp = ts if ts else now
                    self.marker_buffer.append((tstamp, sample[0]))
                if len(self.marker_buffer) > 2000:
                    self.marker_buffer = self.marker_buffer[-2000:]
        # Drive status label from data flow
        marker_age = (
            _clock() - self.last_marker_seen if self.last_marker_seen else None
        )
        if marker_age is not None and marker_age < 2.0:
            self.marker_status.setText("\u25cf Connected")
            self.marker_status.setStyleSheet("color: #44ff44; font-weight: bold;")
        elif self.marker_inlet is not None and not self.marker_buffer:
            self.marker_status.setText("\u25cf Connected")
            self.marker_status.setStyleSheet("color: #44ff44; font-weight: bold;")
        else:
            self.marker_status.setText("\u25cf Waiting...")
            self.marker_status.setStyleSheet("color: #ffaa00; font-weight: bold;")

    def _update_ready_state(self):
        mr_ok = self.monitor.emg_connected
        ang_ok = "Connected" in self.angle_status.text()
        ready = mr_ok and ang_ok and self.arm_button.isChecked()
        self.ready_status.setText(f"Ready: {'YES' if ready else 'NO'}")
        self.ready_status.setStyleSheet(
            "color: #44ff44; font-weight: bold;"
            if ready
            else "color: #ff6666; font-weight: bold;"
        )
        self.mr_led.setStyleSheet(
            "color: #44ff44; font-size: 16px;"
            if mr_ok
            else "color: #ff6666; font-size: 16px;"
        )
        self.angle_led.setStyleSheet(
            "color: #44ff44; font-size: 16px;"
            if ang_ok
            else "color: #ff6666; font-size: 16px;"
        )
        if self.record_start_btn:
            self.record_start_btn.setEnabled(ready)
        self._update_flow_status()

    def _update_flow_status(self):
        if not self.flow_blocks:
            return
        emg_ok = self.monitor.emg_connected and self.monitor.last_emg_chunk is not None
        emg_detail = ""
        if emg_ok and self.monitor.last_emg_chunk is not None:
            emg_detail = f"buf {self.monitor.last_emg_chunk.shape[0]} samples"
        self.flow_blocks["emg"].set_state(
            emg_ok,
            "EMG: streaming" if emg_ok else "EMG: disconnected",
            emg_detail or "Waiting for data",
        )
        self.flow_blocks["emg"].setToolTip(
            f"Raw EMG stream\n{emg_detail or 'Waiting for data'}"
        )

        filters_ok = self.record_filters is not None
        filt_detail = (
            f"HP {self.filter_hp_freq:.1f} / Notch {self.filter_notch_freq:.1f} / LP {self.filter_lp_freq:.1f}"
            if filters_ok
            else "Filters unavailable"
        )
        filters_active = bool(filters_ok and emg_ok)
        self.flow_blocks["filters"].set_state(
            True if filters_active else None,
            "Filters: on" if filters_active else "Filters: idle",
            filt_detail,
        )
        self.flow_blocks["filters"].setToolTip(f"Filter chain\n{filt_detail}")

        buffer_count = len(self.record_emg_windows)
        buffer_active = buffer_count > 0
        window_detail = (
            f"{self.record_window_ms:.0f} ms / {self.record_overlap_ms:.0f} ms overlap"
        )
        window_ready = emg_ok and filters_ok
        window_state = True if window_ready else None
        window_status = "Windowing: active" if window_ready else "Windowing: idle"
        self.flow_blocks["window"].set_state(
            window_state,
            window_status,
            window_detail,
        )
        self.flow_blocks["window"].setToolTip(f"Windowing\n{window_detail}")

        hand_ok = self.angle_count > 0 and self.last_angle_seen > 0
        hand_age = _clock() - self.last_angle_seen if self.last_angle_seen else None
        hand_stale = hand_age is not None and hand_age > 1.0
        hand_detail = f"Angles: {self.angle_stream_name.text().strip()}"
        if hand_ok and not hand_stale:
            hand_state = True
            hand_status = "Hand tracking: OK"
        elif hand_ok and hand_stale:
            hand_state = "warn"
            hand_status = "Hand tracking: stale"
        else:
            hand_state = False
            hand_status = "Hand tracking: waiting"
        self.flow_blocks["hand"].set_state(
            hand_state,
            hand_status,
            hand_detail,
        )
        self.flow_blocks["hand"].setToolTip(f"LSL angles stream\n{hand_detail}")

        target_ok = hand_ok
        target_detail = f"QC: age<{self.record_max_age * 1000:.0f}ms | NaN drop"
        if hand_ok and not hand_stale:
            target_state = True
            target_status = "Angle prep: OK"
        elif hand_ok and hand_stale:
            target_state = "warn"
            target_status = "Angle prep: stale"
        else:
            target_state = None
            target_status = "Angle prep: idle"
        self.flow_blocks["target"].set_state(
            target_state,
            target_status,
            target_detail,
        )
        self.flow_blocks["target"].setToolTip(f"Angle prep/QC\n{target_detail}")

        if self.last_angle_match_ok is None:
            sync_ok = None
            sync_status = "Sync: waiting"
            sync_detail = "Waiting for windows"
        else:
            sync_ok = bool(self.last_angle_match_ok)
            sync_status = "Sync: OK" if sync_ok else "Sync: MISS"
            delta = getattr(self, "last_angle_match_delta_ms", None)
            sync_detail = (
                f"Δ {delta:.0f} ms | lag {self.angle_lag_ms:.0f} ms"
                if delta is not None
                else ""
            )
        
        # Keep sync yellow until all inputs are green
        if not emg_ok or not (hand_ok and not hand_stale):
            sync_state = "warn"
            if not emg_ok and not hand_ok:
                sync_status = "Sync: waiting (no inputs)"
            elif not emg_ok:
                sync_status = "Sync: waiting (no EMG)"
            elif not hand_ok or hand_stale:
                sync_status = "Sync: waiting (no angles)"
        else:
            # All inputs ready, show actual sync state
            sync_state = sync_ok if sync_ok is not None else "warn"
        self.flow_blocks["sync"].set_state(sync_state, sync_status, sync_detail)
        self.flow_blocks["sync"].setToolTip(
            f"Sync/match\n{sync_detail or 'Waiting for windows'}"
        )

        buffer_detail = f"Buffered {buffer_count} windows"
        if self.recording_active:
            buffer_state = "active"
            buffer_status = "Buffer: recording"
        elif buffer_active:
            buffer_state = True
            buffer_status = "Buffer: active"
        else:
            buffer_state = "warn"
            buffer_status = "Buffer: idle"
        self.flow_blocks["buffer"].set_state(
            buffer_state,
            buffer_status,
            buffer_detail,
        )
        self.flow_blocks["buffer"].setToolTip(
            f"Window buffer / recording\n{buffer_detail}"
        )

        feat_detail = f"{self.emg_feature_mode} | {self.emg_transform}"
        if self.recording_active:
            feat_state = "active"
            feat_status = "Features: recording"
        elif buffer_active:
            feat_state = True
            feat_status = "Features: active"
        else:
            feat_state = None
            feat_status = "Features: idle"
        self.flow_blocks["features"].set_state(
            feat_state,
            feat_status,
            feat_detail,
        )
        self.flow_blocks["features"].setToolTip(f"Features + scaling\n{feat_detail}")

        reg_path = self.compare_reg.text().strip() if self.compare_reg else ""
        scaler_path = self.compare_scaler.text().strip() if self.compare_scaler else ""
        saved_ok = bool(
            reg_path
            and scaler_path
            and os.path.exists(reg_path)
            and os.path.exists(scaler_path)
        )
        train_active = self.proc_train is not None
        train_data_path = (
            self.train_data.text().strip() if hasattr(self, "train_data") else ""
        )
        train_ready = bool(train_data_path and os.path.exists(train_data_path))
        train_detail = (
            "Running"
            if train_active
            else ("Done" if saved_ok else ("Ready" if train_ready else "Idle"))
        )
        train_state = (
            "active"
            if train_active
            else (True if saved_ok else ("warn" if train_ready else None))
        )
        self.flow_blocks["train"].set_state(
            train_state,
            "Train: active"
            if train_active
            else ("Train: ready" if train_ready else "Train: idle"),
            train_detail,
        )
        self.flow_blocks["train"].setToolTip(f"Training path\n{train_detail}")

        if "imu" in self.flow_blocks:
            imu_ok = (
                self.monitor.emg_connected and self.monitor.last_imu_chunk is not None
            )
            imu_detail = "Streaming" if imu_ok else "No data"
            self.flow_blocks["imu"].set_state(
                True if imu_ok else False,
                "IMU: OK" if imu_ok else "IMU: no data",
                imu_detail,
            )
            self.flow_blocks["imu"].setToolTip("IMU from LSL stream\n(dashed input)")

        saved_detail = "Ready" if saved_ok else "Missing model/scaler"
        self.flow_blocks["saved"].set_state(
            True if saved_ok else "warn",
            "Saved: OK" if saved_ok else "Saved: pending",
            saved_detail,
        )
        self.flow_blocks["saved"].setToolTip(f"Saved model + scaler\n{saved_detail}")

        infer_ok = self.compare_active
        infer_state = "active" if infer_ok else ("warn" if saved_ok else None)
        infer_status = (
            "Infer: active"
            if infer_ok
            else ("Infer: ready" if saved_ok else "Infer: idle")
        )
        self.flow_blocks["infer"].set_state(
            infer_state,
            infer_status,
            "",
        )
        self.flow_blocks["infer"].setToolTip("Inference path")

        compare_ok = self.compare_active
        compare_state = "active" if compare_ok else ("warn" if saved_ok else None)
        compare_status = (
            "Compare: active"
            if compare_ok
            else ("Compare: ready" if saved_ok else "Compare: idle")
        )
        self.flow_blocks["compare"].set_state(
            compare_state,
            compare_status,
            "",
        )
        self.flow_blocks["compare"].setToolTip("Predicted vs LSL angles")

    def _on_arm_toggled(self, checked):
        self.arm_button.setText("Armed" if checked else "Arm Recording")

    def _on_target_spec_changed(self, idx):
        spec = self.target_angle_combo.itemData(idx)
        if spec:
            self.target_spec = spec
            self.target_keys = (
                get_target_keys(spec) if get_target_keys else ANGLE_KEYS
            )
            self.compare_joints = list(self.target_keys)
            self.compare_joints_idx = []
            n = len(self.target_keys)
            self._append_log(f"[config] Target angles: {spec} ({n} joints)")

    def _start_prompts(self, btn_start, btn_stop):
        if self.proc_prompts:
            return
        plan = self.prompt_plan.text().strip()
        script_path = Path(__file__).parent / "task_prompter_lsl.py"
        if script_path.exists():
            self.proc_prompts = self._run_script(
                "task_prompter_lsl.py", ["--plan_file", plan], "prompts"
            )
            btn_start.setEnabled(False)
            btn_stop.setEnabled(True)
            if self.prompts_led:
                self.prompts_led.setStyleSheet("color: #44ff44; font-size: 16px;")
            self._append_log(f"[prompts] Using plan: {plan}")
            return

        if not self._load_prompt_plan_data(plan):
            self._append_log(f"[prompts] Failed to load plan: {plan}")
            return
        self._start_prompt_sequence()
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)
        if self.prompts_led:
            self.prompts_led.setStyleSheet("color: #44ff44; font-size: 16px;")
        self._append_log(f"[prompts] Using plan (in-GUI): {plan}")

    def _load_prompt_plan_data(self, path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("Prompt plan must be a list.")
            self.prompt_plan_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                duration = float(item.get("duration", 0))
                if label and duration > 0:
                    self.prompt_plan_data.append({"label": label, "duration": duration})
            return bool(self.prompt_plan_data)
        except Exception as exc:
            self._append_log(f"[prompts] Plan parse error: {exc}")
            self.prompt_plan_data = []
            return False

    def _start_prompt_sequence(self):
        if not self.prompt_plan_data:
            return
        if self.prompt_timer is None:
            self.prompt_timer = QTimer()
            self.prompt_timer.timeout.connect(self._advance_prompt_sequence)
        self.prompt_idx = 0
        self.prompt_end_time = 0.0
        try:
            from lsl_utils import HAS_LSL, make_marker_outlet

            if HAS_LSL and self.prompt_outlet is None:
                self.prompt_outlet = make_marker_outlet()
        except Exception:
            self.prompt_outlet = None
        self._emit_prompt()
        self.prompt_timer.start(200)

    def _emit_prompt(self):
        if self.prompt_idx >= len(self.prompt_plan_data):
            self._stop_prompt_sequence()
            return
        item = self.prompt_plan_data[self.prompt_idx]
        label = item["label"]
        duration = float(item["duration"])
        self.prompt_end_time = _clock() + duration
        self._append_log(f"[prompts] [PROMPT] {label} ({duration:.1f}s)")
        if self.prompt_outlet is not None:
            try:
                self.prompt_outlet.push_sample([label])
            except Exception:
                pass

    def _advance_prompt_sequence(self):
        if not self.prompt_plan_data:
            self._stop_prompt_sequence()
            return
        if _clock() >= self.prompt_end_time:
            self.prompt_idx += 1
            if self.prompt_idx >= len(self.prompt_plan_data):
                self._append_log("[prompts] process exited")
                self._stop_prompt_sequence()
            else:
                self._emit_prompt()

    def _stop_prompt_sequence(self):
        if self.prompt_timer:
            self.prompt_timer.stop()
        self.prompt_timer = None
        self.prompt_plan_data = []
        self.prompt_idx = 0
        self.prompt_end_time = 0.0
        self.prompt_outlet = None
        if self.prompts_led:
            self.prompts_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        if self.prompts_start_btn and self.prompts_stop_btn:
            self.prompts_start_btn.setEnabled(True)
            self.prompts_stop_btn.setEnabled(False)
        if self.recording_active and self.record_start_btn and self.record_stop_btn:
            self._stop_process("record", self.record_start_btn, self.record_stop_btn)
        if self.task_start_btn and self.task_stop_btn:
            self.task_start_btn.setEnabled(True)
            self.task_stop_btn.setEnabled(False)

    def _start_task(self):
        if self.arm_button and not self.arm_button.isChecked():
            self.arm_button.setChecked(True)
            self._append_log("[task] Auto-armed recording.")

        ready = (
            self.monitor.emg_connected
            and ("Connected" in self.angle_status.text())
            and self.arm_button.isChecked()
        )
        if not ready:
            self._append_log(
                "[task] Not ready: need EMG LSL + Angles OK + Arm Recording."
            )
            return

        if self.prompts_start_btn and self.prompts_stop_btn:
            self._start_prompts(self.prompts_start_btn, self.prompts_stop_btn)
        if self.record_start_btn and self.record_stop_btn:
            self._start_record(self.record_start_btn, self.record_stop_btn)

        if self.task_start_btn and self.task_stop_btn:
            self.task_start_btn.setEnabled(False)
            self.task_stop_btn.setEnabled(True)

    def _stop_task(self):
        if self.proc_prompts and self.prompts_start_btn and self.prompts_stop_btn:
            self._stop_process("prompts", self.prompts_start_btn, self.prompts_stop_btn)
        elif self.prompt_timer:
            self._stop_prompt_sequence()
        if self.recording_active and self.record_start_btn and self.record_stop_btn:
            self._stop_process("record", self.record_start_btn, self.record_stop_btn)

        if self.task_start_btn and self.task_stop_btn:
            self.task_start_btn.setEnabled(True)
            self.task_stop_btn.setEnabled(False)

    def _start_record(self, btn_start, btn_stop):
        if self.recording_active:
            return
        if not (
            self.monitor.emg_connected
            and "Connected" in self.angle_status.text()
            and self.arm_button.isChecked()
        ):
            self._append_log(
                "[WARN] Not ready: need EMG LSL, Angles OK, and Arm Recording enabled."
            )
            return
        if get_target_keys:
            self.target_keys = get_target_keys(self.target_spec)
        else:
            self.target_keys = ANGLE_KEYS
        if self.record_filters is None:
            self._append_log(
                "[record] WARN: EMG filters unavailable; recording raw EMG."
            )
        out_path = self.record_out.text().strip()
        self.record_duration = int(self.record_duration_spin.value())
        self.record_window_ms = float(self.record_window_spin.value())
        self.record_overlap_ms = float(self.record_overlap_spin.value())
        self.record_window_len = self._ms_to_samples(self.record_window_ms)
        self.record_overlap = min(
            self._ms_to_samples(self.record_overlap_ms),
            max(0, self.record_window_len - 1),
        )
        self.angle_lag_ms = float(self.angle_lag_spin.value())
        self.angle_lag_s = self.angle_lag_ms / 1000.0
        self.record_max_age = max(
            0.2, (self.record_window_ms / 1000.0) + abs(self.angle_lag_s)
        )
        self.record_hand_idx = 0 if self.record_hand.currentIndex() == 0 else 1
        self.record_out_path = out_path
        self.record_angle_stream = self.angle_stream_name.text().strip()
        self.record_marker_stream = self.marker_stream_name.text().strip()
        self.record_plan_file = self.prompt_plan.text().strip()
        self.record_plan_json = ""
        if self.record_plan_file and os.path.exists(self.record_plan_file):
            try:
                self.record_plan_json = Path(self.record_plan_file).read_text(
                    encoding="utf-8"
                )
            except Exception:
                self.record_plan_json = ""
        self.record_subject = (
            self.exp_subject.text().strip() if hasattr(self, "exp_subject") else ""
        )
        self.record_session = (
            self.exp_session.text().strip() if hasattr(self, "exp_session") else ""
        )
        self.record_notes = (
            self.exp_notes.text().strip() if hasattr(self, "exp_notes") else ""
        )
        self.record_wrist = (
            self.exp_wrist.currentText() if hasattr(self, "exp_wrist") else ""
        )
        self.record_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.record_ts_buf = np.zeros((0,), dtype=np.float64)
        self.record_imu_buf = np.zeros((0, 9), dtype=np.float32)
        self.record_emg_windows = []
        self.record_angle_targets = []
        self.record_window_ts = []
        self.record_marker_labels = []
        self.record_imu_windows = []
        self.record_skip_no_angle = 0
        self.record_skip_nan_angle = 0
        self.record_skip_old_angle = 0
        self.record_fill_nan_angle = 0
        self.record_start_time = _clock()
        self.recording_active = True
        self.record_last_chunk_id = -1
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)
        if self.record_led:
            self.record_led.setStyleSheet("color: #44ff44; font-size: 16px;")
        if self.record_windows_label:
            self.record_windows_label.setText("Windows: 0")
        self._append_log(
            f"[record] Started (in-GUI): {out_path} (duration={self.record_duration}s, "
            f"window={self.record_window_ms:.0f}ms/{self.record_window_len} samples, "
            f"overlap={self.record_overlap_ms:.0f}ms/{self.record_overlap} samples, "
            f"lag={self.angle_lag_ms:.0f}ms, wrist={self.record_wrist or 'n/a'})"
        )

    def _start_train(self, btn_start, btn_stop):
        if self.proc_train:
            return
        train_out = self.train_out.text().strip()
        try:
            os.makedirs(train_out, exist_ok=True)
            cutoff_payload = {
                "mae_cutoff": float(self.train_mae_thresh.value()),
                "r2_cutoff": float(self.train_r2_thresh.value()),
            }
            cutoff_path = Path(train_out) / "cutoffs.json"
            cutoff_path.write_text(
                json.dumps(cutoff_payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        args = [
            "--data",
            self.train_data.text().strip(),
            "--feature_extractor",
            self.train_feat.text().strip(),
            "--out_dir",
            self.train_out.text().strip(),
            "--emg_transform",
            self.emg_transform,
            "--emg_features",
            self.emg_feature_mode,
            "--angle_scaler",
            self.angle_scaler_mode,
        ]
        if hasattr(self, "train_use_imu"):
            args += ["--use_imu", "on" if self.train_use_imu.isChecked() else "off"]
        self.proc_train = self._run_script("train_regressor.py", args, "train")
        self._append_log("[train] Stop criteria: max_iter or solver convergence.")
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)
        if hasattr(self, "train_pass_label"):
            self.train_pass_label.setText("Result: training...")
            self.train_pass_label.setStyleSheet("color: #9aa3ad;")

    def _on_feature_source_changed(self, text):
        if not hasattr(self, "train_feat"):
            return
        source = text.lower()
        use_transformer = "transformer" in source
        use_cnn = "cnn" in source
        self.train_feat.setEnabled(use_transformer)
        if use_cnn:
            self.train_feat.setText(self.feat_out_model.text().strip() or "none")
            self.train_feat.setEnabled(False)
            self.pipeline_preset_combo.setCurrentText(
                "Transformer + MLP (feature extractor)"
            )
            self.train_emg_transform_combo.setCurrentText("none")
            self.train_feature_combo.setCurrentText("raw_flat")
            self.train_emg_transform_combo.setEnabled(False)
            self.train_feature_combo.setEnabled(False)
        elif use_transformer:
            if self.train_feat.text().strip() in ("", "none"):
                self.train_feat.setText(
                    self._default_feature_extractor_path().as_posix()
                )
            self.pipeline_preset_combo.setCurrentText(
                "Transformer + MLP (feature extractor)"
            )
            self.train_emg_transform_combo.setCurrentText("none")
            self.train_feature_combo.setCurrentText("raw_flat")
            self.train_emg_transform_combo.setEnabled(False)
            self.train_feature_combo.setEnabled(False)
        else:
            self.train_feat.setText("none")
            self.train_emg_transform_combo.setEnabled(True)
            self.train_feature_combo.setEnabled(True)

    def _on_feature_extractor_output_changed(self, value):
        if (
            hasattr(self, "feature_source_combo")
            and "cnn" in self.feature_source_combo.currentText().lower()
        ):
            if self.train_feat:
                self.train_feat.setText(value.strip())

    def _update_train_result_banner(self):
        if not hasattr(self, "train_pass_label") or not self.train_out:
            return
        metrics_path = Path(self.train_out.text().strip()) / "metrics.json"
        if not metrics_path.exists():
            self.train_pass_label.setText("Result: N/A")
            self.train_pass_label.setStyleSheet("color: #9aa3ad;")
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mae = float(metrics.get("mae", 0.0))
            r2 = float(metrics.get("r2", 0.0))
            mae_cut = float(self.train_mae_thresh.value())
            r2_cut = float(self.train_r2_thresh.value())
            passed = (mae <= mae_cut) and (r2 >= r2_cut)
            label = (
                f"Result: {'PASS' if passed else 'FAIL'} (MAE {mae:.2f}, R² {r2:.2f})"
            )
            self.train_pass_label.setText(label)
            self.train_pass_label.setStyleSheet(
                "color: #44ff44;" if passed else "color: #ff6666;"
            )

            # Persist cutoffs into metrics.json so runs are self-describing
            metrics["mae_cutoff"] = mae_cut
            metrics["r2_cutoff"] = r2_cut
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except Exception:
            self.train_pass_label.setText("Result: N/A")
            self.train_pass_label.setStyleSheet("color: #9aa3ad;")

    def _pick_feature_datasets(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select datasets",
            str(Path(__file__).parent / "data"),
            "NPZ files (*.npz)",
        )
        if files:
            self.feat_data_paths = files
            self.feat_data_display.setText(f"{len(files)} files selected")
            self._validate_feature_datasets()

    def _clear_feature_datasets(self):
        self.feat_data_paths = []
        self.feat_data_display.setText("")
        if hasattr(self, "feat_data_summary"):
            self.feat_data_summary.setText("No datasets selected.")

    def _add_subject_sessions(self):
        subject = (
            self.exp_subject.text().strip() if hasattr(self, "exp_subject") else ""
        )
        if not subject:
            self._append_log("[feat] Enter Subject ID first.")
            return
        if not subject.startswith("sub-"):
            subject = f"sub-{subject.zfill(3) if subject.isdigit() else subject}"
        data_root = Path(__file__).parent / "data"
        subject_dir = data_root / subject
        if not subject_dir.exists():
            self._append_log(f"[feat] Subject folder not found: {subject_dir}")
            return
        self._add_sessions_from_root(subject_dir)

    def _add_all_sessions(self):
        data_root = Path(__file__).parent / "data"
        if not data_root.exists():
            self._append_log(f"[feat] Data root not found: {data_root}")
            return
        self._add_sessions_from_root(data_root)

    def _add_sessions_from_root(self, root):
        wrist_filter = (
            self.feat_wrist_filter.currentText()
            if hasattr(self, "feat_wrist_filter")
            else "Any wrist"
        )
        paths = sorted(root.glob("**/*.npz"))
        if not paths:
            self._append_log(f"[feat] No datasets found under {root}")
            return
        filtered = []
        for p in paths:
            if wrist_filter and wrist_filter != "Any wrist":
                try:
                    data = np.load(p, allow_pickle=True)
                    wrist = str(data.get("wrist_orientation", "")).strip().lower()
                except Exception:
                    wrist = ""
                if wrist != wrist_filter:
                    continue
            filtered.append(str(p))
        if not filtered:
            self._append_log(f"[feat] No datasets matched wrist filter: {wrist_filter}")
            return
        existing = set(self.feat_data_paths)
        added = [p for p in filtered if p not in existing]
        self.feat_data_paths = list(existing) + added
        self.feat_data_display.setText(f"{len(self.feat_data_paths)} files selected")
        self._validate_feature_datasets()

    def _validate_feature_datasets(self):
        if not self.feat_data_paths:
            if hasattr(self, "feat_data_summary"):
                self.feat_data_summary.setText("No datasets selected.")
            return
        total = 0
        win_len = None
        angle_dim = None
        errors = []
        for path in self.feat_data_paths:
            try:
                data = np.load(path, allow_pickle=True)
                if "emg" not in data or "angles" not in data:
                    raise KeyError("missing emg/angles")
                emg = np.asarray(data["emg"])
                angles = np.asarray(data["angles"])
                if emg.ndim == 4 and emg.shape[-1] == 1:
                    pass
                elif emg.ndim == 3 and emg.shape[1] == 8:
                    emg = emg[:, :, :, None]
                elif emg.ndim == 3 and emg.shape[2] == 8:
                    emg = emg.transpose(0, 2, 1)[:, :, :, None]
                else:
                    raise ValueError(f"emg shape {emg.shape}")
                if angles.ndim != 2:
                    raise ValueError(f"angles shape {angles.shape}")
                if emg.shape[0] != angles.shape[0]:
                    raise ValueError("sample count mismatch")
                if win_len is None:
                    win_len = emg.shape[2]
                if angle_dim is None:
                    angle_dim = angles.shape[1]
                if emg.shape[2] != win_len:
                    raise ValueError("window_len mismatch")
                if angles.shape[1] != angle_dim:
                    raise ValueError("angle_dim mismatch")
                total += emg.shape[0]
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if errors:
            msg = f"Check failed ({len(errors)} files)."
            details = "; ".join(errors[:3])
            if len(errors) > 3:
                details += " ..."
            self.feat_data_summary.setText(f"{msg} {details}")
            self.feat_data_summary.setStyleSheet("color: #d9b44a;")
        else:
            self.feat_data_summary.setText(
                f"{len(self.feat_data_paths)} files OK | {total} windows | window={win_len} | angles={angle_dim}"
            )
            self.feat_data_summary.setStyleSheet("color: #8f99a6;")

    def _start_feature_extractor(self, btn_start, btn_stop):
        if self.proc_feat:
            return
        if not self.feat_data_paths:
            self._append_log("[feat] No datasets selected.")
            return
        try:
            np_ver = np.__version__.split(".")[0]
            if np_ver.isdigit() and int(np_ver) >= 2:
                # TensorFlow wheels used here are built against NumPy <2.0
                self._append_log(
                    "[feat] NumPy 2.x detected. TensorFlow feature-extractor training requires numpy<2.\n"
                    "[feat] Please create a TF env with numpy<2 (e.g., 1.26.x) and rerun."
                )
                return
        except Exception:
            pass
        out_path = self.feat_out_model.text().strip()
        args = ["--out_model", out_path]
        args.extend(["--data"] + self.feat_data_paths)
        max_epochs = int(self.feat_epochs.value())
        args.extend(["--epochs", str(max_epochs)])
        args.extend(["--batch_size", str(int(self.feat_batch.value()))])
        args.extend(["--val_split", f"{float(self.feat_val_split.value()):.2f}"])
        args.extend(["--feature_dim", str(int(self.feat_dim.value()))])
        args.extend(["--lr", f"{float(self.feat_lr.value()):.5f}"])
        side = self.feat_side.currentText().strip()
        if side:
            args.extend(["--side", side])
        self.proc_feat = self._run_script("train_feature_extractor.py", args, "feat")
        self.feat_last_epoch = 0
        self.feat_max_epochs = max_epochs
        self.feat_early_stopped = False
        if hasattr(self, "feat_status_label"):
            self.feat_status_label.setText(
                f"Status: running (epochs={max_epochs}, early stop patience=7)"
            )
            self.feat_status_label.setStyleSheet("color: #9aa3ad;")
        self._append_log(
            f"[feat] Stop criteria: epochs={max_epochs} "
            f"or early stopping (patience=7, val_loss)."
        )
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)

    def _start_compare_gui(self, btn_start, btn_stop):
        if self.compare_active:
            return
        if not self.monitor.emg_connected:
            self._append_log("[compare] EMG LSL stream not available.")
            return
        if "Connected" not in self.angle_status.text():
            self._append_log("[compare] LSL Angles not available.")
            return
        if not ANGLE_KEYS:
            self._append_log("[compare] ANGLE_KEYS missing; cannot map joints.")
            return

        feat_path = self.compare_feat.text().strip()
        reg_path = self.compare_reg.text().strip()
        scaler_path = self.compare_scaler.text().strip()
        self.compare_emg_transform = self.emg_transform
        self.compare_feature_mode = self.emg_feature_mode

        if not os.path.exists(reg_path):
            self._append_log(f"[compare] Regressor not found: {reg_path}")
            return
        if not os.path.exists(scaler_path):
            self._append_log(f"[compare] Scaler not found: {scaler_path}")
            return

        try:
            with open(reg_path, "rb") as f:
                self.compare_regressor = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self.compare_scaler = pickle.load(f)
        except Exception as exc:
            self._append_log(f"[compare] Failed to load regressor/scaler: {exc}")
            return

        metrics_path = Path(reg_path).with_name("metrics.json")
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.compare_emg_transform = metrics.get(
                    "emg_transform", self.compare_emg_transform
                )
                self.compare_feature_mode = metrics.get(
                    "emg_feature_mode",
                    metrics.get("feature_mode", self.compare_feature_mode),
                )
                if metrics.get("target_spec"):
                    self.target_spec = metrics.get("target_spec")
                if metrics.get("angle_keys"):
                    self.target_keys = list(metrics.get("angle_keys"))
                self.compare_joints = (
                    list(self.target_keys) if self.target_keys else self.compare_joints
                )
            except Exception as exc:
                self._append_log(f"[compare] Failed to read metrics.json: {exc}")

        target_scaler_path = Path(reg_path).with_name("target_scaler.pkl")
        if not target_scaler_path.exists():
            alt_path = Path(reg_path).with_name("angle_scaler.pkl")
            target_scaler_path = alt_path if alt_path.exists() else target_scaler_path
        if target_scaler_path.exists():
            try:
                with open(target_scaler_path, "rb") as f:
                    self.compare_target_scaler = pickle.load(f)
            except Exception as exc:
                self._append_log(f"[compare] Failed to load target scaler: {exc}")
                self.compare_target_scaler = None

        try:
            self.compare_extractor = self._load_feature_extractor(feat_path)
        except Exception as exc:
            self._append_log(f"[compare] Failed to load feature extractor: {exc}")
            return

        self.angle_lag_ms = float(self.angle_lag_spin.value())
        self.angle_lag_s = self.angle_lag_ms / 1000.0
        self.compare_window_len = self._ms_to_samples(self.record_window_spin.value())
        self.compare_overlap = min(
            self._ms_to_samples(self.record_overlap_spin.value()),
            max(0, self.compare_window_len - 1),
        )
        self.compare_max_age = max(
            0.2, (self.record_window_spin.value() / 1000.0) + abs(self.angle_lag_s)
        )
        self.compare_smooth_ms = float(self.compare_smooth_spin.value())
        self.compare_pred_history = []
        if self.target_keys:
            self.compare_joints_idx = list(range(len(self.target_keys)))
        else:
            self.compare_joints_idx = [
                ANGLE_KEYS.index(j) for j in self.compare_joints if j in ANGLE_KEYS
            ]
            if not self.compare_joints_idx:
                self.compare_joints_idx = [0]

        self.compare_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.compare_ts_buf = np.zeros((0,), dtype=np.float64)
        self.compare_imu_buf = np.zeros((0, 9), dtype=np.float32)
        self.compare_last_chunk_id = -1
        self.compare_active = True
        self.compare_label.setText("Compare: running")
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)

    def _stop_compare_gui(self, btn_start, btn_stop):
        self.compare_active = False
        self.compare_extractor = None
        self.compare_regressor = None
        self.compare_scaler = None
        self.compare_target_scaler = None
        self.compare_pred_history = []
        self.compare_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.compare_ts_buf = np.zeros((0,), dtype=np.float64)
        self.compare_imu_buf = np.zeros((0, 9), dtype=np.float32)
        if self.compare_label:
            self.compare_label.setText("Compare: N/A")
        btn_start.setEnabled(True)
        btn_stop.setEnabled(False)

    def _ms_to_samples(self, ms):
        return max(1, int(round((float(ms) / 1000.0) * self.record_fs)))

    def _samples_to_ms(self, samples):
        return float(samples) * 1000.0 / self.record_fs if self.record_fs else 0.0

    def _update_window_labels(self):
        window_ms = float(self.record_window_spin.value())
        overlap_ms = float(self.record_overlap_spin.value())
        window_len = self._ms_to_samples(window_ms)
        requested_overlap = self._ms_to_samples(overlap_ms)
        overlap_len = min(requested_overlap, max(0, window_len - 1))
        stride_len = max(1, window_len - overlap_len)
        clipped = " (overlap clipped)" if requested_overlap >= window_len else ""
        self.record_window_info.setText(
            f"~{window_len} samples (stride {stride_len}, {self._samples_to_ms(stride_len):.0f} ms){clipped}"
        )

    def _on_flow_details_toggled(self, checked):
        if self.flow_canvas:
            self.flow_canvas.set_compact(not checked)
        if hasattr(self, "flow_toggle_compact"):
            self.flow_toggle_compact.blockSignals(True)
            self.flow_toggle_compact.setChecked(not checked)
            self.flow_toggle_compact.blockSignals(False)

    def _on_flow_lines_toggled(self, checked):
        if self.flow_canvas:
            self.flow_canvas.set_show_lines(checked)

    def _on_flow_compact_toggled(self, checked):
        if self.flow_canvas:
            self.flow_canvas.set_compact(checked)
        if hasattr(self, "flow_toggle_details"):
            self.flow_toggle_details.blockSignals(True)
            self.flow_toggle_details.setChecked(not checked)
            self.flow_toggle_details.blockSignals(False)

    def _on_snapshot_auto_toggled(self, checked):
        self.snapshot_on_open = bool(checked)
        if checked and not self._snapshot_done:
            self._queue_snapshot(self.snapshot_on_open_path)

    def _default_snapshot_path(self):
        out_dir = Path(__file__).parent / "figs"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return out_dir / f"gui_snapshot_{ts}.png"

    def _save_snapshot(self, path=None):
        out_path = Path(path) if path else self._default_snapshot_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix = QPixmap(self.size())
        self.render(pix)
        ok = pix.save(str(out_path))
        if ok:
            self._append_log(f"[snapshot] Saved: {out_path}")
        else:
            self._append_log(f"[snapshot] Failed to save: {out_path}")
        return ok

    def _queue_snapshot(self, path=None):
        QTimer.singleShot(self.snapshot_delay_ms, lambda: self._save_snapshot(path))

    def _on_pipeline_param_changed(self):
        if hasattr(self, "train_emg_transform_combo"):
            self.emg_transform = self.train_emg_transform_combo.currentText()
        if hasattr(self, "train_feature_combo"):
            self.emg_feature_mode = self.train_feature_combo.currentText()
        if hasattr(self, "train_angle_scaler_combo"):
            self.angle_scaler_mode = self.train_angle_scaler_combo.currentText()

    def _on_pipeline_preset_changed(self, preset):
        if preset.startswith("Bandpower"):
            self.train_emg_transform_combo.setCurrentText("log1p")
            self.train_feature_combo.setCurrentText("bandpower_stats")
            self.train_angle_scaler_combo.setCurrentText("minmax")
            if self.train_feat.text().strip().lower() in ("", "none", "raw", "flat"):
                self.train_feat.setText("none")
        elif preset.startswith("Raw flat"):
            self.train_emg_transform_combo.setCurrentText("none")
            self.train_feature_combo.setCurrentText("raw_flat")
            self.train_angle_scaler_combo.setCurrentText("none")
        else:
            # Transformer + MLP (feature extractor)
            self.train_emg_transform_combo.setCurrentText("none")
            self.train_feature_combo.setCurrentText("raw_flat")
            self.train_angle_scaler_combo.setCurrentText("minmax")
        self._on_pipeline_param_changed()
        if (
            hasattr(self, "flow_method_combo")
            and self.flow_method_combo.currentText() != preset
        ):
            self.flow_method_combo.blockSignals(True)
            self.flow_method_combo.setCurrentText(preset)
            self.flow_method_combo.blockSignals(False)
        if (
            hasattr(self, "pipeline_preset_combo")
            and self.pipeline_preset_combo.currentText() != preset
        ):
            self.pipeline_preset_combo.blockSignals(True)
            self.pipeline_preset_combo.setCurrentText(preset)
            self.pipeline_preset_combo.blockSignals(False)

    def _load_feature_extractor(self, path):
        if not path or str(path).lower() in ("none", "raw", "flat"):
            return None
        if not os.path.exists(path):
            self._append_log(
                f"[compare] Feature extractor not found: {path}. Using raw features."
            )
            return None
        global KerasFeatureExtractor, PyTorchFeatureExtractor
        if KerasFeatureExtractor is None and PyTorchFeatureExtractor is None:
            try:
                inference_path = (
                    Path(__file__).resolve().parents[2]
                    / "nml"
                    / "gesture_classifier"
                    / "inference.py"
                )
                if not inference_path.exists():
                    raise FileNotFoundError(
                        "inference.py not found in nml/gesture_classifier."
                    )
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "nml_gesture_inference", str(inference_path)
                )
                inference_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(inference_mod)
                KerasFeatureExtractor = getattr(
                    inference_mod, "KerasFeatureExtractor", None
                )
                PyTorchFeatureExtractor = getattr(
                    inference_mod, "PyTorchFeatureExtractor", None
                )
                if KerasFeatureExtractor is None and PyTorchFeatureExtractor is None:
                    raise RuntimeError(
                        "Feature extractor classes not found in inference.py."
                    )
            except Exception as exc:
                self._append_log(
                    "[compare] Feature extractor unavailable (import failed). "
                    "Check NumPy/TensorFlow compatibility."
                )
                self._append_log(f"[compare] Import error: {exc}")
                return None
        if path.endswith(".h5"):
            if KerasFeatureExtractor is None:
                raise RuntimeError(
                    "KerasFeatureExtractor unavailable in this environment."
                )
            return KerasFeatureExtractor(path)
        if path.endswith((".pt", ".pth")):
            if PyTorchFeatureExtractor is None:
                raise RuntimeError(
                    "PyTorchFeatureExtractor unavailable in this environment."
                )
            return PyTorchFeatureExtractor(path)
        if KerasFeatureExtractor is not None:
            return KerasFeatureExtractor(path)
        if PyTorchFeatureExtractor is not None:
            return PyTorchFeatureExtractor(path)
        raise RuntimeError("No feature extractor backends available.")

    def _apply_emg_transform(self, emg, mode):
        if mode == "none":
            return emg
        if mode == "log1p":
            return np.log1p(np.maximum(emg, 0.0))
        return emg

    def _bandpower_features(self, emg):
        # emg: (N, C, T)
        if emg.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        mean = emg.mean(axis=2)
        std = emg.std(axis=2)
        minv = emg.min(axis=2)
        maxv = emg.max(axis=2)
        if emg.shape[2] > 1:
            diff = np.diff(emg, axis=2)
            diff_energy = np.mean(diff**2, axis=2)
        else:
            diff_energy = np.zeros_like(mean)
        t = np.linspace(-0.5, 0.5, emg.shape[2], dtype=np.float32)
        denom = float(np.sum(t**2)) if emg.shape[2] > 1 else 1.0
        slope = np.tensordot(emg, t, axes=([2], [0])) / denom
        return np.concatenate([mean, std, minv, maxv, diff_energy, slope], axis=1)

    def _extract_features(self, emg_input, extractor):
        arr = emg_input
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = self._apply_emg_transform(arr, self.compare_emg_transform)
        if extractor is None:
            if self.compare_feature_mode == "bandpower_stats":
                return self._bandpower_features(arr)
            return arr.reshape(arr.shape[0], -1)
        return extractor.predict(arr[..., None], verbose=0)

    def _imu_features(self, window):
        if window is None:
            return None
        if window.ndim == 2 and window.shape[0] in (6, 9):
            imu = window
        elif window.ndim == 2 and window.shape[1] in (6, 9):
            imu = window.T
        else:
            return None
        mean = imu.mean(axis=1)
        std = imu.std(axis=1)
        return np.concatenate([mean, std], axis=0)[None, ...]

    def _nearest_angle_sample_at(self, timestamp, hand_idx, max_age=0.2):
        if not self.angle_buffer:
            return None
        best = min(self.angle_buffer, key=lambda x: abs(x[0] - timestamp))
        if abs(best[0] - timestamp) > max_age:
            return None
        sample = np.asarray(best[1], dtype=np.float32).reshape(-1)
        per_hand = len(ANGLE_KEYS) if ANGLE_KEYS else None
        if per_hand is None or per_hand <= 0:
            if sample.size % 2 == 0:
                per_hand = sample.size // 2
            else:
                return None
        start = hand_idx * per_hand
        end = start + per_hand
        if sample.size < end:
            return None
        sample = sample[start:end]
        if self.target_keys and ANGLE_KEYS:
            idxs = [ANGLE_KEYS.index(k) for k in self.target_keys if k in ANGLE_KEYS]
            if idxs:
                sample = sample[idxs]
        if not np.isfinite(sample).all():
            return None
        return sample

    def _stop_process(self, name, btn_start, btn_stop):
        if name == "record":
            self._finish_record(save=True)
        else:
            proc = getattr(self, f"proc_{name}", None)
            if name == "prompts" and proc is None and self.prompt_timer:
                self._stop_prompt_sequence()
                if btn_start and btn_stop:
                    btn_start.setEnabled(True)
                    btn_stop.setEnabled(False)
                return
            if proc:
                self._stop_qprocess(proc)
                if name == "feat":
                    self.proc_feat = None
                setattr(self, f"proc_{name}", None)
            if name == "prompts" and self.prompts_led:
                self.prompts_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        btn_start.setEnabled(True)
        btn_stop.setEnabled(False)

    def _stop_qprocess(self, proc):
        if proc is None:
            return
        try:
            proc.terminate()
            proc.waitForFinished(2000)
        except Exception:
            pass
        if proc.state() != QProcess.NotRunning:
            try:
                proc.kill()
                proc.waitForFinished(2000)
            except Exception:
                pass

    def _run_script(self, script_name, args, tag):
        script_path = str(Path(__file__).parent / script_name)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_proc_output(proc, tag))
        proc.finished.connect(lambda *_: self._on_proc_finished(tag))
        proc.start(sys.executable, ["-u", script_path] + args)
        return proc

    def _pick_file(self, target, title):
        path, _ = QFileDialog.getOpenFileName(self, title)
        if path:
            target.setText(path)
            if target is self.prompt_plan:
                self._load_prompt_preview()

    @staticmethod
    def _next_session_path():
        """Return the next available session_NNN.npz path."""
        data_dir = Path(__file__).parent / "data"
        n = 1
        while True:
            candidate = data_dir / f"session_{n:03d}.npz"
            if not candidate.exists():
                return candidate
            n += 1

    def _pick_save(self, target, title):
        path, _ = QFileDialog.getSaveFileName(self, title)
        if path:
            target.setText(path)

    def _pick_dir(self, target, title):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            target.setText(path)

    def _on_proc_output(self, proc, tag):
        data = proc.readAllStandardOutput().data().decode(errors="ignore")
        if data:
            for line in data.splitlines():
                self._append_log(f"[{tag}] {line.rstrip()}")
                if tag == "compare":
                    self._update_compare_label(line.rstrip())
                if tag == "feat":
                    self._update_feat_status_from_log(line.rstrip())

    def _update_feat_status_from_log(self, line):
        text = line.strip()
        if text.startswith("Epoch"):
            # Expected: "Epoch 3/50"
            try:
                core = text.split()[1]
                if "/" in core:
                    cur, total = core.split("/", 1)
                    self.feat_last_epoch = int(cur)
                    if total.isdigit():
                        self.feat_max_epochs = int(total)
                if self.feat_status_label:
                    self.feat_status_label.setText(
                        f"Status: epoch {self.feat_last_epoch}/{self.feat_max_epochs or '?'}"
                    )
                    self.feat_status_label.setStyleSheet("color: #9aa3ad;")
            except Exception:
                return
        if "EarlyStopping" in text or "Restoring model weights" in text:
            self.feat_early_stopped = True

    def _update_compare_label(self, line):
        if self.compare_label is None:
            return
        if "mean_mae=" not in line:
            return
        # Expected: [compare] t=... joint gt=.. pred=.. mean_mae=..
        try:
            core = line.split("] ", 1)[-1]
            parts = core.split()
            if len(parts) < 6:
                return
            joint = parts[1]
            gt = parts[2].split("=", 1)[-1]
            pred = parts[3].split("=", 1)[-1]
            mae = parts[-1].split("=", 1)[-1]
            self.compare_label.setText(
                f"Compare: {joint} gt={gt} pred={pred} mae={mae}"
            )
        except Exception:
            return

    def _on_proc_finished(self, tag):
        if self._closing:
            return
        self._append_log(f"[{tag}] process exited")
        if tag == "prompts" and self.prompts_led:
            self.prompts_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        if (
            tag == "prompts"
            and self.recording_active
            and self.record_start_btn
            and self.record_stop_btn
        ):
            self._stop_process("record", self.record_start_btn, self.record_stop_btn)
        if tag == "prompts" and self.task_start_btn and self.task_stop_btn:
            self.task_start_btn.setEnabled(True)
            self.task_stop_btn.setEnabled(False)
        if tag == "feat":
            if hasattr(self, "feat_start_btn") and self.feat_start_btn:
                self.feat_start_btn.setEnabled(True)
            if hasattr(self, "feat_stop_btn") and self.feat_stop_btn:
                self.feat_stop_btn.setEnabled(False)
            if hasattr(self, "feat_status_label") and self.feat_status_label:
                if self.feat_early_stopped:
                    msg = f"Status: early stopped at epoch {self.feat_last_epoch}/{self.feat_max_epochs or '?'}"
                else:
                    msg = f"Status: finished ({self.feat_last_epoch}/{self.feat_max_epochs or '?'})"
                self.feat_status_label.setText(msg)
                self.feat_status_label.setStyleSheet("color: #44ff44;")
        if tag == "train":
            self._update_train_result_banner()

    def _append_log(self, text):
        if self._closing or self.log_view is None:
            return
        self.log_buffer.append(text)
        if len(self.log_buffer) > 500:
            self.log_buffer = self.log_buffer[-500:]
        self.log_view.setText("\n".join(self.log_buffer))
        self.log_view.moveCursor(self.log_view.textCursor().End)

    def _record_tick(self):
        if not self.recording_active:
            return
        if not self.monitor.emg_connected:
            self._append_log("[record] EMG stream disconnected; stopping.")
            self._finish_record(save=True)
            return
        if self.monitor.last_emg_chunk is None or self.monitor.last_emg_ts is None:
            return
        if self.monitor.last_chunk_id == self.record_last_chunk_id:
            return
        chunk = self.monitor.last_emg_chunk
        ts_chunk = self.monitor.last_emg_ts
        imu_chunk = self.monitor.last_imu_chunk
        self.record_last_chunk_id = self.monitor.last_chunk_id

        if self.record_filters:
            chunk = self._apply_filters(chunk, self.record_filters)

        self.record_emg_buf = np.vstack([self.record_emg_buf, chunk])
        self.record_ts_buf = np.concatenate([self.record_ts_buf, ts_chunk])
        if imu_chunk is not None:
            # Pad to 9 channels if only accel+gyro
            if imu_chunk.shape[1] == 6:
                pad = np.zeros((imu_chunk.shape[0], 3), dtype=np.float32)
                imu_chunk = np.hstack([imu_chunk.astype(np.float32), pad])
            self.record_imu_buf = np.vstack(
                [self.record_imu_buf, imu_chunk.astype(np.float32)]
            )
        else:
            # Keep buffers aligned
            pad = np.zeros((chunk.shape[0], 9), dtype=np.float32)
            self.record_imu_buf = np.vstack([self.record_imu_buf, pad])
        if self.emg_buf_label:
            self.emg_buf_label.setText(f"EMG buf: {self.record_emg_buf.shape[0]}")

        if self.record_ts_buf.size:
            self._update_angle_match_status(
                self.record_ts_buf[-1],
                window_ready=self.record_emg_buf.shape[0] >= self.record_window_len,
            )

        stride = self.record_window_len - self.record_overlap
        if stride <= 0:
            self._append_log(
                "[record] Invalid window/overlap: overlap must be < window."
            )
            return
        while self.record_emg_buf.shape[0] >= self.record_window_len:
            window = self.record_emg_buf[: self.record_window_len]
            window_ts = self.record_ts_buf[: self.record_window_len]
            t_center = window_ts[self.record_window_len // 2]

            self._update_angle_match_status(t_center, window_ready=True)
            angle_sample = self._nearest_angle_sample(t_center)
            if angle_sample is not None:
                self.record_emg_windows.append(window.T[..., None])
                self.record_angle_targets.append(angle_sample)
                self.record_window_ts.append(t_center)
                self.record_marker_labels.append(self._nearest_marker_label(t_center))
                if self.record_imu_buf.shape[0] >= self.record_window_len:
                    imu_window = self.record_imu_buf[: self.record_window_len]
                    self.record_imu_windows.append(imu_window.T)
                if self.record_windows_label:
                    self.record_windows_label.setText(
                        f"Windows: {len(self.record_emg_windows)}"
                    )

            self.record_emg_buf = self.record_emg_buf[stride:]
            self.record_ts_buf = self.record_ts_buf[stride:]
            if self.record_imu_buf.shape[0] >= stride:
                self.record_imu_buf = self.record_imu_buf[stride:]

        if _clock() - self.record_start_time >= self.record_duration:
            self._finish_record(save=True)

    def _compare_tick(self):
        if not self.compare_active:
            return
        if not self.monitor.emg_connected:
            return
        if self.monitor.last_emg_chunk is None or self.monitor.last_emg_ts is None:
            return
        if self.monitor.last_chunk_id == self.compare_last_chunk_id:
            return
        if self.compare_regressor is None or self.compare_scaler is None:
            return

        chunk = self.monitor.last_emg_chunk
        ts_chunk = self.monitor.last_emg_ts
        imu_chunk = self.monitor.last_imu_chunk
        self.compare_last_chunk_id = self.monitor.last_chunk_id

        if self.record_filters:
            chunk = self._apply_filters(chunk, self.record_filters)

        self.compare_emg_buf = np.vstack([self.compare_emg_buf, chunk])
        self.compare_ts_buf = np.concatenate([self.compare_ts_buf, ts_chunk])
        if imu_chunk is not None:
            if imu_chunk.shape[1] == 6:
                pad = np.zeros((imu_chunk.shape[0], 3), dtype=np.float32)
                imu_chunk = np.hstack([imu_chunk.astype(np.float32), pad])
            self.compare_imu_buf = np.vstack(
                [self.compare_imu_buf, imu_chunk.astype(np.float32)]
            )
        else:
            pad = np.zeros((chunk.shape[0], 9), dtype=np.float32)
            self.compare_imu_buf = np.vstack([self.compare_imu_buf, pad])

        stride = self.compare_window_len - self.compare_overlap
        if stride <= 0:
            return

        while self.compare_emg_buf.shape[0] >= self.compare_window_len:
            window = self.compare_emg_buf[: self.compare_window_len]
            window_ts = self.compare_ts_buf[: self.compare_window_len]
            t_center = window_ts[self.compare_window_len // 2]

            angle_sample = self._nearest_angle_sample_at(
                t_center + self.angle_lag_s,
                self.record_hand_idx,
                max_age=self.compare_max_age,
            )
            if angle_sample is None:
                self.compare_emg_buf = self.compare_emg_buf[stride:]
                self.compare_ts_buf = self.compare_ts_buf[stride:]
                if self.compare_imu_buf.shape[0] >= stride:
                    self.compare_imu_buf = self.compare_imu_buf[stride:]
                continue

            emg_input = window.T[..., None][None, ...]
            feats = self._extract_features(emg_input, self.compare_extractor)
            if self.compare_imu_buf.shape[0] >= self.compare_window_len:
                imu_window = self.compare_imu_buf[: self.compare_window_len]
                imu_feat = self._imu_features(imu_window)
                if imu_feat is not None:
                    feats = np.concatenate([feats, imu_feat], axis=1)
            feats = self.compare_scaler.transform(feats)
            pred_scaled = self.compare_regressor.predict(feats)[0]
            if self.compare_target_scaler is not None:
                pred_angles = self.compare_target_scaler.inverse_transform(
                    pred_scaled.reshape(1, -1)
                )[0]
            else:
                pred_angles = pred_scaled
            pred_angles = self._smooth_predictions(pred_angles)

            idxs = self.compare_joints_idx or [0]
            gt = angle_sample[idxs]
            pr = pred_angles[idxs]
            diff = np.abs(gt - pr)
            rmse = float(np.sqrt(np.mean((gt - pr) ** 2))) if gt.size else 0.0
            mae = float(np.mean(diff)) if diff.size else 0.0
            if self.compare_label:
                self.compare_label.setText(f"Compare: RMSE={rmse:.2f} MAE={mae:.2f}")
            if self.compare_bar_label:
                self.compare_bar_label.setText(self._render_compare_bars(idxs, diff))
                self.compare_bar_label.setVisible(True)

            self.compare_emg_buf = self.compare_emg_buf[stride:]
            self.compare_ts_buf = self.compare_ts_buf[stride:]
            if self.compare_imu_buf.shape[0] >= stride:
                self.compare_imu_buf = self.compare_imu_buf[stride:]

    def _render_compare_bars(self, idxs, diff):
        if diff is None or diff.size == 0:
            return ""
        max_diff = 90.0
        width = 12
        lines = []
        for j_idx, d in zip(idxs, diff):
            bar_len = int(round((d / max_diff) * width))
            bar = "#" * bar_len
            if self.target_keys and j_idx < len(self.target_keys):
                label = _friendly_joint(self.target_keys[j_idx])
            else:
                label = _friendly_joint(ANGLE_KEYS[j_idx])
            lines.append(f"{label:<11} |{bar}")
        return "\n".join(lines)

    def _smooth_predictions(self, pred_angles):
        if pred_angles is None:
            return pred_angles
        if self.compare_smooth_ms <= 0:
            return pred_angles
        stride = self.compare_window_len - self.compare_overlap
        if stride <= 0:
            return pred_angles
        stride_ms = self._samples_to_ms(stride)
        if stride_ms <= 0:
            return pred_angles
        n = max(1, int(round(self.compare_smooth_ms / stride_ms)))
        self.compare_pred_history.append(pred_angles)
        if len(self.compare_pred_history) > n:
            self.compare_pred_history = self.compare_pred_history[-n:]
        return np.mean(self.compare_pred_history, axis=0)

    def _nearest_angle_sample(self, timestamp):
        timestamp = timestamp + self.angle_lag_s
        if not self.angle_buffer:
            self.record_skip_no_angle += 1
            return None
        best = min(self.angle_buffer, key=lambda x: abs(x[0] - timestamp))
        if abs(best[0] - timestamp) > self.record_max_age:
            self.record_skip_old_angle += 1
            return None
        sample = np.asarray(best[1], dtype=np.float32).reshape(-1)
        per_hand = len(ANGLE_KEYS) if ANGLE_KEYS else None
        if per_hand is None or per_hand <= 0:
            if sample.size % 2 == 0:
                per_hand = sample.size // 2
            else:
                self.record_skip_no_angle += 1
                return None
        start = self.record_hand_idx * per_hand
        end = start + per_hand
        if sample.size < end:
            self.record_skip_no_angle += 1
            return None
        sample = sample[start:end]
        # Map to target subset if requested
        if self.target_keys and ANGLE_KEYS:
            idxs = [ANGLE_KEYS.index(k) for k in self.target_keys if k in ANGLE_KEYS]
            if idxs:
                sample = sample[idxs]
        if not np.isfinite(sample).all():
            # Fill NaNs with last valid angles for this hand, if available
            last_valid = self.last_valid_angles[self.record_hand_idx]
            if last_valid is None or not np.isfinite(last_valid).all():
                self.record_skip_nan_angle += 1
                return None
            fill_mask = ~np.isfinite(sample)
            sample = sample.copy()
            sample[fill_mask] = last_valid[fill_mask]
            self.record_fill_nan_angle += 1
        # Update last valid angles
        if np.isfinite(sample).all():
            self.last_valid_angles[self.record_hand_idx] = sample.copy()
        return sample

    def _update_angle_match_status(self, timestamp, window_ready):
        if not self.angle_match_label:
            return
        if not self.angle_buffer:
            state = "waiting" if not window_ready else "no samples"
            self.angle_match_label.setText(f"Angle match: {state}")
            self.last_angle_match_ok = None
            return
        target_ts = timestamp + self.angle_lag_s
        best = min(self.angle_buffer, key=lambda x: abs(x[0] - target_ts))
        delta = abs(best[0] - target_ts)
        ok = delta <= self.record_max_age
        status = "OK" if ok else "MISS"
        suffix = "" if window_ready else " (buf<window)"
        self.angle_match_label.setText(
            f"Angle match: {status} ({delta * 1000:.0f} ms, lag={self.angle_lag_ms:.0f} ms){suffix}"
        )
        self.last_angle_match_ok = ok
        self.last_angle_match_delta_ms = delta * 1000.0

    def _nearest_marker_label(self, timestamp):
        if not self.marker_buffer:
            return ""
        best = min(self.marker_buffer, key=lambda x: abs(x[0] - timestamp))
        if abs(best[0] - timestamp) > 1.0:
            return ""
        return best[1]

    def _apply_filters(self, data, filters):
        out = data.copy()
        for i in range(out.shape[0]):
            for ch in range(out.shape[1]):
                for filt in filters:
                    out[i, ch] = filt.process(out[i, ch], ch)
        return out

    def _build_filters(self):
        if self._filter_class is None or self._filter_types is None:
            return None
        return [
            self._filter_class(
                8,
                self._filter_types.bq_type_highpass,
                self.filter_hp_freq / self.record_fs,
                self.filter_hp_q,
                0.0,
            ),
            self._filter_class(
                8,
                self._filter_types.bq_type_notch,
                self.filter_notch_freq / self.record_fs,
                self.filter_notch_q,
                0.0,
            ),
            self._filter_class(
                8,
                self._filter_types.bq_type_lowpass,
                self.filter_lp_freq / self.record_fs,
                self.filter_lp_q,
                0.0,
            ),
        ]

    def _on_filter_params_changed(self):
        self.filter_hp_freq = float(self.hp_freq.value())
        self.filter_hp_q = float(self.hp_q.value())
        self.filter_notch_freq = float(self.notch_freq.value())
        self.filter_notch_q = float(self.notch_q.value())
        self.filter_lp_freq = float(self.lp_freq.value())
        self.filter_lp_q = float(self.lp_q.value())
        self.record_filters = self._build_filters()

    def _finish_record(self, save=True):
        if not self.recording_active:
            return
        self.recording_active = False
        if self.record_led:
            self.record_led.setStyleSheet("color: #ff6666; font-size: 16px;")
        if self.record_windows_label:
            self.record_windows_label.setText(
                f"Windows: {len(self.record_emg_windows)}"
            )
        if save and self.record_out_path:
            os.makedirs(os.path.dirname(self.record_out_path), exist_ok=True)
            np.savez_compressed(
                self.record_out_path,
                emg=np.asarray(self.record_emg_windows, dtype=np.float32),
                angles=np.asarray(self.record_angle_targets, dtype=np.float32),
                timestamps=np.asarray(self.record_window_ts, dtype=np.float64),
                markers=np.asarray(self.record_marker_labels, dtype=object),
                imu=np.asarray(self.record_imu_windows, dtype=np.float32),
                imu_channels=np.asarray(
                    ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"], dtype=object
                ),
                angle_keys=np.asarray(self.target_keys or ANGLE_KEYS, dtype=object),
                target_spec=self.target_spec,
                fs=self.record_fs,
                window_len=self.record_window_len,
                overlap=self.record_overlap,
                window_ms=float(self.record_window_ms),
                overlap_ms=float(self.record_overlap_ms),
                angle_lag_ms=float(self.angle_lag_ms),
                emg_transform=self.emg_transform,
                emg_feature_mode=self.emg_feature_mode,
                angle_stream=self.record_angle_stream,
                marker_stream=self.record_marker_stream,
                plan_file=self.record_plan_file,
                plan_json=self.record_plan_json,
                session_start=self.record_start_time,
                subject_id=self.record_subject,
                session_id=self.record_session,
                notes=self.record_notes,
                wrist_orientation=self.record_wrist,
            )
            self._append_log(
                f"[record] Saved: {self.record_out_path} (windows={len(self.record_emg_windows)}, "
                f"no_angle={self.record_skip_no_angle}, old_angle={self.record_skip_old_angle}, "
                f"nan_angle={self.record_skip_nan_angle}, filled_nan={self.record_fill_nan_angle})"
            )
            # Auto-advance to next available session number
            next_path = self._next_session_path()
            self.record_out.setText(str(next_path))

    def _load_prompt_preview(self):
        path = self.prompt_plan.text().strip()
        if not path or not os.path.exists(path):
            self.prompt_preview.setText("Plan file not found.")
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            lines = []
            total = 0.0
            for i, item in enumerate(data, start=1):
                label = item.get("label", "unknown")
                dur = float(item.get("duration", 0))
                total += dur
                lines.append(f"{i:02d}. {label}  ({dur:.1f}s)")
            lines.append(f"\nTotal duration: {total:.1f}s")
            self.prompt_preview.setText("\n".join(lines))
            try:
                self.prompt_mtime = os.path.getmtime(path)
            except Exception:
                self.prompt_mtime = 0.0
        except Exception as exc:
            self.prompt_preview.setText(f"Failed to load plan: {exc}")

    def _watch_prompt_plan(self):
        path = self.prompt_plan.text().strip()
        if not path or not os.path.exists(path):
            return
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            return
        if mtime > self.prompt_mtime:
            self._load_prompt_preview()

    def _apply_styles(self):
        self.setStyleSheet("""
        QWidget {
            font-size: 11px;
            background: #262b33;
            color: #eef1f4;
        }
        #flow_block {
            border: 1px solid #3a414b;
            border-radius: 8px;
            background: #2d343d;
        }
        #flow_block[state="ok"] {
            border: 1px solid #58b985;
            background: #2b4333;
        }
        #flow_block[state="bad"] {
            border: 1px solid #d77878;
            background: #482f2f;
        }
        #flow_block[state="warn"] {
            border: 1px solid #e1c267;
            background: #4a402f;
        }
        #flow_block[state="active"] {
            border: 1px solid #e2897d;
            background: #482f2c;
        }
        QGroupBox {
            border: 1px solid #3a414b;
            border-radius: 10px;
            margin-top: 6px;
            padding: 8px;
            background: #303741;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
        }
        QToolButton {
            border: none;
            font-weight: bold;
            color: #dbe1e8;
        }
        QPushButton {
            background-color: #4a7fb6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 4px 8px;
        }
        QPushButton:checked {
            background-color: #3b6a9a;
        }
        QPushButton:hover {
            background-color: #5b91c7;
        }
        QLineEdit, QSpinBox, QComboBox, QTextEdit {
            border: 1px solid #454d58;
            border-radius: 6px;
            padding: 3px;
            background: #2a3039;
            color: #eef1f4;
            selection-background-color: #4a7fb6;
        }
        QLabel {
            color: #dbe1e8;
        }
        QScrollArea {
            border: none;
        }
        QCheckBox {
            color: #dbe1e8;
        }
        """)

    def closeEvent(self, event):
        self._closing = True
        if self.recording_active:
            self._finish_record(save=True)
        if self.compare_active:
            self.compare_active = False
        for name in ("prompts", "record", "train", "feat"):
            proc = getattr(self, f"proc_{name}", None)
            if proc:
                self._stop_qprocess(proc)
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        if self.snapshot_on_open and not self._snapshot_done:
            self._snapshot_done = True
            self._queue_snapshot(self.snapshot_on_open_path)


def main():
    parser = argparse.ArgumentParser(description="Joint Angle Regression GUI")
    parser.add_argument(
        "--snapshot", help="Save a PNG snapshot of the GUI to this path"
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Render snapshot without showing the GUI",
    )
    parser.add_argument(
        "--snapshot-delay", type=int, default=200, help="Delay (ms) before snapshot"
    )
    args, qt_args = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + qt_args)
    w = SessionConsole()
    w.snapshot_delay_ms = int(args.snapshot_delay)
    w.snapshot_on_open_path = args.snapshot
    if args.snapshot_only:
        w.setAttribute(Qt.WA_DontShowOnScreen, True)
        w.snapshot_on_open = True
        w.show()
        QTimer.singleShot(w.snapshot_delay_ms + 150, app.quit)
        return sys.exit(app.exec_())

    w.show()
    if args.snapshot:
        w._queue_snapshot(args.snapshot)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
