import argparse
import json
import os
import time
from pathlib import Path
from functools import lru_cache
import sys

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("QT_API", "pyqt5")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
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


_QT_MAIN_WINDOW_BASE = QMainWindow if HAS_QT else object

try:
    if HAS_QT:
        import pyqtgraph.opengl as gl
    else:
        gl = None
except Exception as exc:
    gl = None
    GL_IMPORT_ERROR = str(exc)
else:
    GL_IMPORT_ERROR = None

try:
    from examples.visualization.open_ephys_heatmap_viewer import OpenEphysHeatmapSource, _DARK_STYLE, NotReadyError
except Exception as exc:
    OpenEphysHeatmapSource = None
    _DARK_STYLE = ""
    NotReadyError = RuntimeError
    SOURCE_IMPORT_ERROR = str(exc)
else:
    SOURCE_IMPORT_ERROR = None


@lru_cache(maxsize=4)
def _load_obj_mesh(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = []
                for item in line.split()[1:]:
                    token = item.split("/")[0]
                    if token:
                        indices.append(int(token) - 1)
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) > 3:
                    for face_idx in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[face_idx], indices[face_idx + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no usable geometry: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _build_obj_meshdata(model_path: Path, arm_radius: float, arm_length: float):
    vertices, faces = _load_obj_mesh(str(model_path))
    verts = np.array(vertices, dtype=np.float32, copy=True)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = 0.5 * (mins + maxs)
    verts -= center

    extents = maxs - mins
    long_axis = int(np.argmax(extents))
    other_axes = [axis for axis in range(3) if axis != long_axis]
    verts = verts[:, [other_axes[0], other_axes[1], long_axis]]

    half_x = max(float(np.max(np.abs(verts[:, 0]))), 1e-6)
    half_y = max(float(np.max(np.abs(verts[:, 1]))), 1e-6)
    half_z = max(float(np.max(np.abs(verts[:, 2]))), 1e-6)
    verts[:, 0] *= float((arm_radius * 1.12) / half_x)
    verts[:, 1] *= float((arm_radius * 0.86) / half_y)
    verts[:, 2] *= float((arm_length * 0.49) / half_z)

    return gl.MeshData(vertexes=verts, faces=faces)


def _electrode_column_lengths(n_channels: int) -> list[int]:
    if n_channels == 128:
        return [13, 13, 13, 13, 12, 12, 13, 13, 13, 13]

    n_cols = max(1, int(np.ceil(np.sqrt(max(n_channels, 1)))))
    base = n_channels // n_cols
    extra = n_channels % n_cols
    return [base + (1 if idx < extra else 0) for idx in range(n_cols)]


@lru_cache(maxsize=1)
def _default_hdemg128_spec() -> dict:
    path = Path(__file__).resolve().parent / "electrode_layout_default_128.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Default electrode layout JSON must contain a top-level object")
    return data


def _hdemg128_z_positions(arm_length: float) -> list[np.ndarray]:
    spec = _default_hdemg128_spec()
    lengths = [int(value) for value in spec.get("column_lengths", [])]
    if not lengths:
        raise ValueError("Default electrode layout JSON is missing column_lengths")

    raw_offsets = spec.get("column_start_offsets_mm", [])
    if len(raw_offsets) != len(lengths):
        raise ValueError("column_start_offsets_mm must have the same length as column_lengths")

    pitch_m = float(spec.get("column_pitch_mm", 15.0)) / 1000.0
    start_offsets_m = np.asarray(raw_offsets, dtype=np.float32) / 1000.0

    top_edge = float(np.max(start_offsets_m))
    bottom_edge = float(
        np.min(np.array([start_offsets_m[col] - pitch_m * (length - 1) for col, length in enumerate(lengths)]))
    )
    layout_center = 0.5 * (top_edge + bottom_edge)

    usable_half_length = max(arm_length * 0.48, 1e-6)
    half_span = max(top_edge - layout_center, layout_center - bottom_edge, 1e-6)
    scale = min(1.0, usable_half_length / half_span)

    z_columns: list[np.ndarray] = []
    for col, length in enumerate(lengths):
        raw = start_offsets_m[col] - pitch_m * np.arange(length, dtype=np.float32)
        centered = (raw - layout_center) * scale
        z_columns.append(centered.astype(np.float32))
    return z_columns


def _build_electrode_positions(
    n_channels: int,
    arm_radius_x: float,
    arm_radius_y: float,
    arm_length: float,
    radial_offset: float,
) -> np.ndarray:
    lengths = _electrode_column_lengths(n_channels)
    n_cols = len(lengths)
    positions = np.zeros((n_channels, 3), dtype=np.float32)
    index = 0
    angle_offset = -0.5 * np.pi

    z_columns = None
    if n_channels == 128:
        z_columns = _hdemg128_z_positions(arm_length)

    for col, col_len in enumerate(lengths):
        angle = angle_offset + (2.0 * np.pi * col / max(n_cols, 1))
        x = (arm_radius_x + radial_offset) * np.cos(angle)
        y = (arm_radius_y + radial_offset) * np.sin(angle)
        if z_columns is not None:
            z_values = z_columns[col]
        elif col_len <= 1:
            z_values = np.array([0.0], dtype=np.float32)
        else:
            z_values = np.linspace(arm_length * 0.48, -arm_length * 0.48, col_len, dtype=np.float32)
        for z in z_values:
            if index >= n_channels:
                break
            positions[index] = (x, y, z)
            index += 1
    return positions


def _channel_column_indices(n_channels: int) -> np.ndarray:
    lengths = _electrode_column_lengths(n_channels)
    indices = np.empty(n_channels, dtype=np.int32)
    start = 0
    for col, length in enumerate(lengths):
        stop = min(start + length, n_channels)
        indices[start:stop] = col
        start = stop
    return indices


def _as_vector3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-element vector")
    return arr


def _resolve_optional_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def _load_layout_override(raw_path: str | None) -> tuple[dict | None, Path | None]:
    path = _resolve_optional_path(raw_path)
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(f"Electrode layout JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Electrode layout JSON must contain a top-level object")
    return data, path


def _reference_layout_summary(layout: str, layout_override: dict | None) -> str:
    if layout != "hdemg128_vertical_columns":
        return f"Array: layout={layout}"

    reference = None
    if layout_override is not None:
        candidate = layout_override.get("reference_hdemg_layout")
        if isinstance(candidate, dict):
            reference = candidate
    if reference is None:
        reference = _default_hdemg128_spec()

    channel_count = int(reference.get("channel_count", 128))
    pitch_mm = float(reference.get("column_pitch_mm", 15.0))
    stagger_mm = float(reference.get("stagger_offset_mm", 7.5))
    ranges = reference.get("column_channel_ranges", [])

    range_text = ""
    if isinstance(ranges, list) and ranges:
        formatted = []
        for pair in ranges:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                formatted.append(f"{int(pair[0])}-{int(pair[1])}")
        if formatted:
            range_text = " | cols=" + ", ".join(formatted)

    return f"Array: {channel_count}ch | pitch={pitch_mm:.1f} mm | stagger={stagger_mm:.1f} mm{range_text}"


def _preview_channel_count(layout: str, layout_override: dict | None, requested_channels: int) -> int:
    if requested_channels > 0:
        return requested_channels
    if layout == "hdemg128_vertical_columns":
        reference = None
        if layout_override is not None:
            candidate = layout_override.get("reference_hdemg_layout")
            if isinstance(candidate, dict):
                reference = candidate
        if reference is None:
            reference = _default_hdemg128_spec()
        return int(reference.get("channel_count", 128))
    return 32


def _series_override_mm(layout_override: dict, key: str, n_cols: int) -> np.ndarray:
    raw = layout_override.get(key)
    if raw is None:
        return np.zeros(n_cols, dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < n_cols:
        arr = np.pad(arr, (0, n_cols - arr.size))
    return arr[:n_cols]


def _apply_layout_override(base_positions: np.ndarray, layout_override: dict | None) -> np.ndarray:
    positions = np.array(base_positions, dtype=np.float32, copy=True)
    if not layout_override:
        return positions

    n_channels = positions.shape[0]
    lengths = _electrode_column_lengths(n_channels)
    n_cols = len(lengths)
    column_indices = _channel_column_indices(n_channels)

    absolute_positions = layout_override.get("positions_mm")
    if absolute_positions is not None:
        arr = np.asarray(absolute_positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < n_channels:
            raise ValueError("positions_mm must be an array with at least n_channels rows of [x, y, z]")
        positions = arr[:n_channels] / 1000.0

    global_offset = layout_override.get("global_offset_mm")
    if global_offset is not None:
        positions += _as_vector3(global_offset, "global_offset_mm") / 1000.0

    angle_offsets = np.deg2rad(_series_override_mm(layout_override, "column_angle_offsets_deg", n_cols))
    radial_offsets = _series_override_mm(layout_override, "column_radial_offsets_mm", n_cols) / 1000.0
    z_offsets = _series_override_mm(layout_override, "column_z_offsets_mm", n_cols) / 1000.0

    for idx in range(n_channels):
        col = int(column_indices[idx])
        angle = float(angle_offsets[col])
        if angle != 0.0:
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            x, y = float(positions[idx, 0]), float(positions[idx, 1])
            positions[idx, 0] = c * x - s * y
            positions[idx, 1] = s * x + c * y

        radial = float(radial_offsets[col])
        if radial != 0.0:
            norm = float(np.linalg.norm(positions[idx, :2]))
            if norm > 1e-6:
                positions[idx, :2] += (positions[idx, :2] / norm) * radial

        z_shift = float(z_offsets[col])
        if z_shift != 0.0:
            positions[idx, 2] += z_shift

    channel_offsets = layout_override.get("channel_offsets_mm", {})
    if channel_offsets:
        if not isinstance(channel_offsets, dict):
            raise ValueError("channel_offsets_mm must be a JSON object keyed by 1-based channel number")
        for key, value in channel_offsets.items():
            channel_idx = int(key) - 1
            if 0 <= channel_idx < n_channels:
                positions[channel_idx] += _as_vector3(value, f"channel_offsets_mm[{key}]") / 1000.0

    channel_positions = layout_override.get("channel_positions_mm", {})
    if channel_positions:
        if not isinstance(channel_positions, dict):
            raise ValueError("channel_positions_mm must be a JSON object keyed by 1-based channel number")
        for key, value in channel_positions.items():
            channel_idx = int(key) - 1
            if 0 <= channel_idx < n_channels:
                positions[channel_idx] = _as_vector3(value, f"channel_positions_mm[{key}]") / 1000.0

    return positions


def _build_inferno_colors(values: np.ndarray, levels: tuple[float, float]) -> np.ndarray:
    # Approximate inferno with a compact hand-tuned ramp to avoid extra runtime deps.
    anchors = np.array(
        [
            [0.00, 0.01, 0.07],
            [0.18, 0.03, 0.32],
            [0.42, 0.10, 0.54],
            [0.68, 0.23, 0.38],
            [0.88, 0.52, 0.14],
            [0.99, 0.88, 0.22],
        ],
        dtype=np.float32,
    )
    lo, hi = levels
    span = max(hi - lo, 1e-6)
    norm = np.clip(np.nan_to_num((values - lo) / span, nan=0.0), 0.0, 1.0)
    anchors_x = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    colors = np.empty((values.size, 4), dtype=np.float32)
    for idx in range(3):
        colors[:, idx] = np.interp(norm, anchors_x, anchors[:, idx])
    colors[:, 3] = 0.98
    return colors


class _OrientationView(gl.GLViewWidget if gl is not None else object):
    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def wheelEvent(self, event):
        event.ignore()


class _InteractiveView(gl.GLViewWidget if gl is not None else object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drag_selected_callback = None
        self.pick_channel_callback = None
        self._dragging_selected = False
        self._last_pos = None
        self._click_press_pos = None

    def mousePressEvent(self, event):
        if (
            self.drag_selected_callback is not None
            and event.button() == Qt.LeftButton
            and bool(event.modifiers() & Qt.ShiftModifier)
        ):
            self._dragging_selected = True
            self._last_pos = event.pos()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._click_press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_selected and self.drag_selected_callback is not None and self._last_pos is not None:
            pos = event.pos()
            dx = float(pos.x() - self._last_pos.x())
            dy = float(pos.y() - self._last_pos.y())
            self._last_pos = pos
            self.drag_selected_callback(dx, dy)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_selected and event.button() == Qt.LeftButton:
            self._dragging_selected = False
            self._last_pos = None
            event.accept()
            return
        if (
            event.button() == Qt.LeftButton
            and self.pick_channel_callback is not None
            and self._click_press_pos is not None
        ):
            delta = event.pos() - self._click_press_pos
            self._click_press_pos = None
            if abs(delta.x()) <= 4 and abs(delta.y()) <= 4:
                if self.pick_channel_callback(float(event.pos().x()), float(event.pos().y())):
                    event.accept()
                    return
        super().mouseReleaseEvent(event)


class Arm3DWindow(_QT_MAIN_WINDOW_BASE):
    def __init__(self, args):
        super().__init__()
        if not HAS_QT:
            raise RuntimeError("PyQt5 is required for the 3D viewer.")
        if gl is None:
            raise RuntimeError(
                "pyqtgraph OpenGL support is unavailable for the 3D viewer. Install PyOpenGL and PyOpenGL-accelerate."
                if GL_IMPORT_ERROR is None
                else f"pyqtgraph OpenGL import failed: {GL_IMPORT_ERROR}"
            )
        if OpenEphysHeatmapSource is None:
            raise RuntimeError(f"heatmap source import failed: {SOURCE_IMPORT_ERROR}")

        self.args = args
        self.source = None
        self.last_retry = 0.0
        self.color_levels = None
        self.electrode_positions = np.zeros((0, 3), dtype=np.float32)
        self.latest_rms = None
        self.scatter = None
        self.selected_channel_idx = None
        self.main_view = None
        self.orientation_view = None
        self.arm_mesh = None
        self.grid_item = None
        self.axis_item = None
        self.orientation_axis = None
        self.orientation_grid = None
        self.layout_override, self.layout_override_path = _load_layout_override(self.args.electrode_layout_json)
        if self.layout_override is None:
            self.layout_override = {}
        self._editor_updating = False
        self._column_editor_updating = False
        self._render_dirty = False

        self._init_ui()
        self.setStyleSheet(_DARK_STYLE)

        self.data_timer = QTimer(self)
        self.data_timer.timeout.connect(self._poll_data)
        self.data_timer.start(max(20, int(self.args.data_update_ms)))

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render_frame)
        self.render_timer.start(max(33, int(self.args.update_ms)))

    def _init_ui(self):
        self.setWindowTitle("Open Ephys 3D Arm Viewer")
        self.setMinimumSize(1280, 760)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setSpacing(8)

        controls = QWidget()
        controls.setMaximumWidth(440)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(6)

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

        view_group = QGroupBox("3D Model")
        vg = QGridLayout(view_group)
        vg.setSpacing(4)

        vg.addWidget(QLabel("RMS window (ms)"), 0, 0)
        self.window_edit = QSpinBox()
        self.window_edit.setRange(10, 5000)
        self.window_edit.setValue(self.args.window_ms)
        vg.addWidget(self.window_edit, 0, 1)

        vg.addWidget(QLabel("Data (ms)"), 0, 2)
        self.data_update_edit = QSpinBox()
        self.data_update_edit.setRange(20, 2000)
        self.data_update_edit.setValue(self.args.data_update_ms)
        self.data_update_edit.valueChanged.connect(self._on_data_interval_changed)
        vg.addWidget(self.data_update_edit, 0, 3)

        vg.addWidget(QLabel("Layout"), 1, 0)
        self.layout_edit = QComboBox()
        self.layout_edit.addItems(["hdemg128_vertical_columns", "auto"])
        self.layout_edit.setCurrentText(self.args.layout)
        vg.addWidget(self.layout_edit, 1, 1, 1, 3)

        vg.addWidget(QLabel("Arm radius"), 2, 0)
        self.arm_radius_edit = QSpinBox()
        self.arm_radius_edit.setRange(20, 200)
        self.arm_radius_edit.setValue(self.args.arm_radius_mm)
        self.arm_radius_edit.valueChanged.connect(self._refresh_scene)
        vg.addWidget(self.arm_radius_edit, 2, 1)

        vg.addWidget(QLabel("Arm length"), 2, 2)
        self.arm_length_edit = QSpinBox()
        self.arm_length_edit.setRange(80, 600)
        self.arm_length_edit.setValue(self.args.arm_length_mm)
        self.arm_length_edit.valueChanged.connect(self._refresh_scene)
        vg.addWidget(self.arm_length_edit, 2, 3)

        vg.addWidget(QLabel("Electrode lift"), 3, 0)
        self.offset_edit = QSpinBox()
        self.offset_edit.setRange(1, 40)
        self.offset_edit.setValue(self.args.electrode_offset_mm)
        self.offset_edit.valueChanged.connect(self._refresh_scene)
        vg.addWidget(self.offset_edit, 3, 1)

        vg.addWidget(QLabel("Marker size"), 3, 2)
        self.marker_size_edit = QSpinBox()
        self.marker_size_edit.setRange(2, 80)
        self.marker_size_edit.setValue(self.args.marker_size)
        self.marker_size_edit.valueChanged.connect(self._redraw_scatter)
        vg.addWidget(self.marker_size_edit, 3, 3)

        vg.addWidget(QLabel("Render (ms)"), 4, 0)
        self.render_update_edit = QSpinBox()
        self.render_update_edit.setRange(16, 2000)
        self.render_update_edit.setValue(self.args.update_ms)
        self.render_update_edit.valueChanged.connect(self._on_render_interval_changed)
        vg.addWidget(self.render_update_edit, 4, 1)

        vg.addWidget(QLabel("Mesh detail"), 4, 2)
        self.mesh_detail_edit = QSpinBox()
        self.mesh_detail_edit.setRange(12, 64)
        self.mesh_detail_edit.setValue(self.args.mesh_detail)
        self.mesh_detail_edit.valueChanged.connect(self._refresh_scene)
        vg.addWidget(self.mesh_detail_edit, 4, 3)

        vg.addWidget(QLabel("Arm model"), 5, 0)
        self.arm_model_edit = QComboBox()
        self.arm_model_edit.addItems(["right_arm.obj", "left_arm.obj", "cylinder"])
        self.arm_model_edit.setCurrentText(self.args.arm_model)
        self.arm_model_edit.currentTextChanged.connect(self._refresh_scene)
        vg.addWidget(self.arm_model_edit, 5, 1, 1, 3)

        self.auto_scale = QCheckBox("Auto color scale")
        self.auto_scale.setChecked(True)
        vg.addWidget(self.auto_scale, 6, 0, 1, 2)

        self.show_axis = QCheckBox("Show axis")
        self.show_axis.setChecked(self.args.show_axis)
        self.show_axis.stateChanged.connect(self._refresh_scene)
        vg.addWidget(self.show_axis, 6, 2, 1, 2)

        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(self.args.show_grid)
        self.show_grid.stateChanged.connect(self._refresh_scene)
        vg.addWidget(self.show_grid, 7, 0, 1, 2)

        vg.addWidget(QLabel("Inspect channel"), 7, 2)
        self.inspect_channel_edit = QSpinBox()
        self.inspect_channel_edit.setRange(0, 0)
        self.inspect_channel_edit.setSpecialValueText("None")
        self.inspect_channel_edit.valueChanged.connect(self._on_inspect_channel_changed)
        vg.addWidget(self.inspect_channel_edit, 7, 3)

        controls_layout.addWidget(view_group)

        edit_group = QGroupBox("Live Channel Edit")
        eg = QGridLayout(edit_group)
        eg.setSpacing(4)

        self.edit_hint = QLabel("Select a channel to adjust its local offset live.")
        self.edit_hint.setWordWrap(True)
        self.edit_hint.setStyleSheet("color: #9aa3ad;")
        eg.addWidget(self.edit_hint, 0, 0, 1, 4)

        eg.addWidget(QLabel("dX (mm)"), 1, 0)
        self.edit_dx = QDoubleSpinBox()
        self.edit_dx.setRange(-100.0, 100.0)
        self.edit_dx.setDecimals(2)
        self.edit_dx.setSingleStep(0.5)
        self.edit_dx.valueChanged.connect(self._on_channel_offset_changed)
        eg.addWidget(self.edit_dx, 1, 1)

        eg.addWidget(QLabel("dY (mm)"), 1, 2)
        self.edit_dy = QDoubleSpinBox()
        self.edit_dy.setRange(-100.0, 100.0)
        self.edit_dy.setDecimals(2)
        self.edit_dy.setSingleStep(0.5)
        self.edit_dy.valueChanged.connect(self._on_channel_offset_changed)
        eg.addWidget(self.edit_dy, 1, 3)

        eg.addWidget(QLabel("dZ (mm)"), 2, 0)
        self.edit_dz = QDoubleSpinBox()
        self.edit_dz.setRange(-100.0, 100.0)
        self.edit_dz.setDecimals(2)
        self.edit_dz.setSingleStep(0.5)
        self.edit_dz.valueChanged.connect(self._on_channel_offset_changed)
        eg.addWidget(self.edit_dz, 2, 1)

        self.btn_reset_channel = QPushButton("Reset Channel")
        self.btn_reset_channel.clicked.connect(self._on_reset_selected_channel)
        eg.addWidget(self.btn_reset_channel, 2, 2)

        self.btn_save_layout = QPushButton("Save Layout JSON")
        self.btn_save_layout.clicked.connect(self._on_save_layout_json)
        eg.addWidget(self.btn_save_layout, 2, 3)

        controls_layout.addWidget(edit_group)

        column_group = QGroupBox("Live Column Edit")
        cg2 = QGridLayout(column_group)
        cg2.setSpacing(4)

        self.column_hint = QLabel("Adjust the whole selected column. Selecting a channel also selects its column.")
        self.column_hint.setWordWrap(True)
        self.column_hint.setStyleSheet("color: #9aa3ad;")
        cg2.addWidget(self.column_hint, 0, 0, 1, 4)

        cg2.addWidget(QLabel("Column"), 1, 0)
        self.column_edit = QSpinBox()
        self.column_edit.setRange(1, 1)
        self.column_edit.valueChanged.connect(self._on_column_selection_changed)
        cg2.addWidget(self.column_edit, 1, 1)

        cg2.addWidget(QLabel("Angle (deg)"), 1, 2)
        self.column_angle_edit = QDoubleSpinBox()
        self.column_angle_edit.setRange(-180.0, 180.0)
        self.column_angle_edit.setDecimals(2)
        self.column_angle_edit.setSingleStep(0.5)
        self.column_angle_edit.valueChanged.connect(self._on_column_offset_changed)
        cg2.addWidget(self.column_angle_edit, 1, 3)

        cg2.addWidget(QLabel("Radial (mm)"), 2, 0)
        self.column_radial_edit = QDoubleSpinBox()
        self.column_radial_edit.setRange(-100.0, 100.0)
        self.column_radial_edit.setDecimals(2)
        self.column_radial_edit.setSingleStep(0.5)
        self.column_radial_edit.valueChanged.connect(self._on_column_offset_changed)
        cg2.addWidget(self.column_radial_edit, 2, 1)

        cg2.addWidget(QLabel("Z (mm)"), 2, 2)
        self.column_z_edit = QDoubleSpinBox()
        self.column_z_edit.setRange(-100.0, 100.0)
        self.column_z_edit.setDecimals(2)
        self.column_z_edit.setSingleStep(0.5)
        self.column_z_edit.valueChanged.connect(self._on_column_offset_changed)
        cg2.addWidget(self.column_z_edit, 2, 3)

        self.btn_reset_column = QPushButton("Reset Column")
        self.btn_reset_column.clicked.connect(self._on_reset_selected_column)
        cg2.addWidget(self.btn_reset_column, 3, 2, 1, 2)

        controls_layout.addWidget(column_group)

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

        status_group = QGroupBox("Status")
        sg = QVBoxLayout(status_group)
        sg.setSpacing(2)
        self.status = QLabel("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        self.ch_info = QLabel("Channels: EMG=0 ADC=0")
        self.fs_info = QLabel("Fs: ?")
        self.model_info = QLabel("Model: tapered forearm mesh")
        self.layout_info = QLabel(_reference_layout_summary(self.layout_edit.currentText(), self.layout_override))
        self.layout_info.setWordWrap(True)
        self.layout_info.setStyleSheet("color: #9aa3ad;")
        self.rms_stats = QLabel("RMS min/max/mean: N/A")
        self.scale_info = QLabel("Color scale: N/A")
        self.pick_info = QLabel("Selected: none")
        sg.addWidget(self.status)
        sg.addWidget(self.ch_info)
        sg.addWidget(self.fs_info)
        sg.addWidget(self.model_info)
        sg.addWidget(self.layout_info)
        sg.addWidget(self.rms_stats)
        sg.addWidget(self.scale_info)
        sg.addWidget(self.pick_info)
        controls_layout.addWidget(status_group)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("Connect")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.auto_retry = QCheckBox("Auto-retry (2 s)")
        self.auto_retry.setChecked(True)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.auto_retry)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        controls_layout.addLayout(btns)

        reminder = QLabel(
            "Click a marker to select it. Use Inspect Channel plus dX/dY/dZ or Shift+left-drag in the 3D view to reposition a selected electrode live. Column controls move whole sleeves."
        )
        reminder.setStyleSheet("color: #ffaa00; font-size: 11px;")
        reminder.setWordWrap(True)
        controls_layout.addWidget(reminder)
        controls_layout.addStretch()

        outer.addWidget(controls, stretch=0)

        view_panel = QWidget()
        view_layout = QHBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(8)

        self.main_view = _InteractiveView()
        self.main_view.setBackgroundColor("#1f2329")
        self.main_view.setCameraPosition(distance=3.8, elevation=14, azimuth=28)
        self.main_view.drag_selected_callback = self._on_view_drag_selected
        self.main_view.pick_channel_callback = self._on_view_pick_channel
        view_layout.addWidget(self.main_view, 1)

        orientation_panel = QWidget()
        orientation_layout = QVBoxLayout(orientation_panel)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.setSpacing(6)
        self.orientation_view = _OrientationView()
        self.orientation_view.setFixedSize(180, 180)
        self.orientation_view.setBackgroundColor("#1f2329")
        self.orientation_view.setCameraPosition(distance=2.8, elevation=14, azimuth=28)
        orientation_layout.addWidget(self.orientation_view)
        orientation_hint = QLabel("Inset axes: X red, Y green, Z blue\nDistal is toward -Z")
        orientation_hint.setStyleSheet("color: #9aa3ad; font-size: 11px;")
        orientation_hint.setWordWrap(True)
        orientation_layout.addWidget(orientation_hint)
        orientation_layout.addStretch()
        view_layout.addWidget(orientation_panel, 0)

        outer.addWidget(view_panel, stretch=1)

        self._refresh_scene()

    def _on_data_interval_changed(self, value):
        self.data_timer.setInterval(max(20, int(value)))

    def _on_render_interval_changed(self, value):
        self.render_timer.setInterval(max(16, int(value)))

    def _on_inspect_channel_changed(self, value):
        self._set_selected_channel(value - 1 if value > 0 else None)
        self._render_dirty = True
        self._render_frame()

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

    def _model_path(self) -> Path | None:
        model_name = self.arm_model_edit.currentText().strip()
        if model_name == "cylinder":
            return None
        return (Path(__file__).resolve().parent / model_name).resolve()

    def _view_center(self) -> np.ndarray:
        center = self.main_view.opts.get("center") if self.main_view is not None else None
        if center is None:
            return np.zeros(3, dtype=np.float32)
        if hasattr(center, "x") and hasattr(center, "y") and hasattr(center, "z"):
            return np.array([float(center.x()), float(center.y()), float(center.z())], dtype=np.float32)
        arr = np.asarray(center, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return arr[:3].astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center = self._view_center()
        opts = self.main_view.opts
        distance = float(opts.get("distance", 3.8))
        azimuth = np.deg2rad(float(opts.get("azimuth", 28.0)))
        elevation = np.deg2rad(float(opts.get("elevation", 14.0)))
        cam_offset = np.array(
            [
                distance * np.cos(elevation) * np.cos(azimuth),
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
            ],
            dtype=np.float32,
        )
        camera_pos = center + cam_offset
        forward = center - camera_pos
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        up = np.cross(right, forward)
        up = up / max(float(np.linalg.norm(up)), 1e-6)
        return camera_pos, forward, right, up

    def _project_points_to_screen(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.main_view is None or points.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)
        width = max(1, int(self.main_view.width()))
        height = max(1, int(self.main_view.height()))
        fov = np.deg2rad(float(self.main_view.opts.get("fov", 60.0)))
        focal = 0.5 * height / max(np.tan(0.5 * fov), 1e-6)
        camera_pos, forward, right, up = self._camera_basis()

        rel = points - camera_pos[None, :]
        z_cam = rel @ forward
        x_cam = rel @ right
        y_cam = rel @ up
        visible = z_cam > 1e-6
        screen = np.zeros((points.shape[0], 2), dtype=np.float32)
        if np.any(visible):
            screen[visible, 0] = width * 0.5 + (x_cam[visible] * focal / z_cam[visible])
            screen[visible, 1] = height * 0.5 - (y_cam[visible] * focal / z_cam[visible])
        return screen, visible

    def _arm_surface_axes(self, z_value: float) -> tuple[float, float]:
        arm_radius = float(self.arm_radius_edit.value()) / 100.0
        radial_offset = float(self.offset_edit.value()) / 100.0
        arm_length = float(self.arm_length_edit.value()) / 100.0
        half_len = max(arm_length * 0.49, 1e-6)
        frac = np.clip((z_value + half_len) / (2.0 * half_len), 0.0, 1.0)

        x_prox = arm_radius * 1.12 + radial_offset
        y_prox = arm_radius * 0.86 + radial_offset
        taper = 0.82 / 1.12
        x_dist = arm_radius * 0.82 + radial_offset
        y_dist = arm_radius * 0.86 * taper + radial_offset
        radius_x = float(x_dist + frac * (x_prox - x_dist))
        radius_y = float(y_dist + frac * (y_prox - y_dist))
        return max(radius_x, 1e-6), max(radius_y, 1e-6)

    def _constrain_to_arm_surface(self, position_m: np.ndarray) -> np.ndarray:
        result = np.array(position_m, dtype=np.float32, copy=True)
        arm_length = float(self.arm_length_edit.value()) / 100.0
        half_len = arm_length * 0.49
        result[2] = np.clip(result[2], -half_len, half_len)
        radius_x, radius_y = self._arm_surface_axes(float(result[2]))
        metric = (result[0] / radius_x) ** 2 + (result[1] / radius_y) ** 2
        if metric <= 1e-12:
            result[0] = radius_x
            result[1] = 0.0
        else:
            scale = 1.0 / np.sqrt(metric)
            result[0] *= float(scale)
            result[1] *= float(scale)
        return result

    def _reference_positions_without_channel_specific(self, n_channels: int) -> np.ndarray:
        arm_radius = float(self.arm_radius_edit.value()) / 100.0
        arm_length = float(self.arm_length_edit.value()) / 100.0
        radial_offset = float(self.offset_edit.value()) / 100.0
        base_positions = _build_electrode_positions(
            n_channels=n_channels,
            arm_radius_x=arm_radius * 1.12,
            arm_radius_y=arm_radius * 0.86,
            arm_length=arm_length,
            radial_offset=radial_offset,
        )
        override = dict(self.layout_override)
        override.pop("channel_offsets_mm", None)
        override.pop("channel_positions_mm", None)
        return _apply_layout_override(base_positions, override)

    def _preview_positions_if_idle(self):
        if self.source is not None and self.source.running:
            return
        preview_channels = _preview_channel_count(
            self.layout_edit.currentText(),
            self.layout_override,
            int(self.ch_edit.value()),
        )
        self.electrode_positions = self._electrode_positions_for_count(preview_channels)
        self.inspect_channel_edit.blockSignals(True)
        self.inspect_channel_edit.setRange(0, preview_channels)
        if self.selected_channel_idx is None or self.selected_channel_idx >= preview_channels:
            self.inspect_channel_edit.setValue(0)
            self.selected_channel_idx = None
        else:
            self.inspect_channel_edit.setValue(self.selected_channel_idx + 1)
        self.inspect_channel_edit.blockSignals(False)
        n_cols = len(_electrode_column_lengths(preview_channels))
        self.column_edit.blockSignals(True)
        self.column_edit.setRange(1, max(1, n_cols))
        if self.column_edit.value() < 1 or self.column_edit.value() > n_cols:
            self.column_edit.setValue(1)
        self.column_edit.blockSignals(False)
        self._sync_column_editor()

    def _selected_channel_offset_mm(self) -> np.ndarray:
        if self.selected_channel_idx is None or self.selected_channel_idx < 0:
            return np.zeros(3, dtype=np.float32)
        channel_offsets = self.layout_override.setdefault("channel_offsets_mm", {})
        raw = channel_offsets.get(str(self.selected_channel_idx + 1), [0.0, 0.0, 0.0])
        return _as_vector3(raw, f"channel_offsets_mm[{self.selected_channel_idx + 1}]")

    def _sync_editor_from_selection(self):
        self._editor_updating = True
        try:
            offset = self._selected_channel_offset_mm() if self.selected_channel_idx is not None else np.zeros(3, dtype=np.float32)
            self.edit_dx.setValue(float(offset[0]))
            self.edit_dy.setValue(float(offset[1]))
            self.edit_dz.setValue(float(offset[2]))
            enabled = self.selected_channel_idx is not None
            self.edit_dx.setEnabled(enabled)
            self.edit_dy.setEnabled(enabled)
            self.edit_dz.setEnabled(enabled)
            self.btn_reset_channel.setEnabled(enabled)
            if enabled:
                self.column_edit.blockSignals(True)
                self.column_edit.setValue(self._selected_column_index() + 1)
                self.column_edit.blockSignals(False)
            self._sync_column_editor()
        finally:
            self._editor_updating = False

    def _selected_column_index(self) -> int:
        active_channels = int(self.electrode_positions.shape[0]) if self.electrode_positions.size else _preview_channel_count(
            self.layout_edit.currentText(), self.layout_override, int(self.ch_edit.value())
        )
        indices = _channel_column_indices(active_channels)
        if self.selected_channel_idx is not None and 0 <= self.selected_channel_idx < indices.size:
            return int(indices[self.selected_channel_idx])
        return max(0, int(self.column_edit.value()) - 1)

    def _column_series_value(self, key: str, column_idx: int) -> float:
        n_cols = max(1, int(self.column_edit.maximum()))
        series = _series_override_mm(self.layout_override, key, n_cols)
        return float(series[column_idx])

    def _set_column_series_value(self, key: str, column_idx: int, value: float):
        n_cols = max(1, int(self.column_edit.maximum()))
        series = list(_series_override_mm(self.layout_override, key, n_cols))
        series[column_idx] = float(value)
        if all(abs(item) < 1e-9 for item in series):
            self.layout_override.pop(key, None)
        else:
            self.layout_override[key] = [float(item) for item in series]

    def _sync_column_editor(self):
        self._column_editor_updating = True
        try:
            column_idx = max(0, int(self.column_edit.value()) - 1)
            self.column_angle_edit.setValue(self._column_series_value("column_angle_offsets_deg", column_idx))
            self.column_radial_edit.setValue(self._column_series_value("column_radial_offsets_mm", column_idx))
            self.column_z_edit.setValue(self._column_series_value("column_z_offsets_mm", column_idx))
            enabled = self.column_edit.maximum() >= 1
            self.column_angle_edit.setEnabled(enabled)
            self.column_radial_edit.setEnabled(enabled)
            self.column_z_edit.setEnabled(enabled)
            self.btn_reset_column.setEnabled(enabled)
        finally:
            self._column_editor_updating = False

    def _apply_live_layout_update(self):
        self._preview_positions_if_idle()
        if self.source is not None and self.source.running and self.electrode_positions.shape[0] > 0:
            self.electrode_positions = self._electrode_positions_for_count(self.electrode_positions.shape[0])
        self._render_dirty = True
        self._redraw_scatter()

    def _on_view_pick_channel(self, x_px: float, y_px: float) -> bool:
        if self.electrode_positions.size == 0:
            return False
        screen, visible = self._project_points_to_screen(self.electrode_positions)
        if not np.any(visible):
            return False
        click = np.array([x_px, y_px], dtype=np.float32)
        distances = np.full(self.electrode_positions.shape[0], np.inf, dtype=np.float32)
        distances[visible] = np.linalg.norm(screen[visible] - click[None, :], axis=1)
        idx = int(np.argmin(distances))
        if not np.isfinite(distances[idx]) or float(distances[idx]) > 24.0:
            return False
        self._set_selected_channel(idx)
        self._render_dirty = True
        self._redraw_scatter()
        return True

    def _on_channel_offset_changed(self, *_args):
        if self._editor_updating or self.selected_channel_idx is None:
            return
        channel_offsets = self.layout_override.setdefault("channel_offsets_mm", {})
        key = str(self.selected_channel_idx + 1)
        values = [float(self.edit_dx.value()), float(self.edit_dy.value()), float(self.edit_dz.value())]
        if all(abs(value) < 1e-9 for value in values):
            channel_offsets.pop(key, None)
        else:
            channel_offsets[key] = values
        self._apply_live_layout_update()

    def _on_reset_selected_channel(self):
        if self.selected_channel_idx is None:
            return
        channel_offsets = self.layout_override.setdefault("channel_offsets_mm", {})
        channel_offsets.pop(str(self.selected_channel_idx + 1), None)
        self._sync_editor_from_selection()
        self._apply_live_layout_update()

    def _on_column_selection_changed(self, *_args):
        if self._column_editor_updating:
            return
        self._sync_column_editor()

    def _on_column_offset_changed(self, *_args):
        if self._column_editor_updating:
            return
        column_idx = max(0, int(self.column_edit.value()) - 1)
        self._set_column_series_value("column_angle_offsets_deg", column_idx, float(self.column_angle_edit.value()))
        self._set_column_series_value("column_radial_offsets_mm", column_idx, float(self.column_radial_edit.value()))
        self._set_column_series_value("column_z_offsets_mm", column_idx, float(self.column_z_edit.value()))
        self._apply_live_layout_update()

    def _on_reset_selected_column(self):
        column_idx = max(0, int(self.column_edit.value()) - 1)
        self._set_column_series_value("column_angle_offsets_deg", column_idx, 0.0)
        self._set_column_series_value("column_radial_offsets_mm", column_idx, 0.0)
        self._set_column_series_value("column_z_offsets_mm", column_idx, 0.0)
        self._sync_column_editor()
        self._apply_live_layout_update()

    def _on_view_drag_selected(self, dx_px: float, dy_px: float):
        if self.selected_channel_idx is None or self.main_view is None:
            return
        opts = self.main_view.opts
        distance = float(opts.get("distance", 3.8))
        fov = float(opts.get("fov", 60.0))
        height = max(1, int(self.main_view.height()))
        world_per_pixel = (2.0 * distance * np.tan(0.5 * np.deg2rad(fov))) / float(height)
        _camera_pos, _forward, right, up = self._camera_basis()

        delta_world = right * (dx_px * world_per_pixel) + up * (-dy_px * world_per_pixel)
        reference_positions = self._reference_positions_without_channel_specific(self.electrode_positions.shape[0])
        current_position = np.array(self.electrode_positions[self.selected_channel_idx], dtype=np.float32, copy=True)
        constrained_position = self._constrain_to_arm_surface(current_position + delta_world.astype(np.float32))

        channel_offsets = self.layout_override.setdefault("channel_offsets_mm", {})
        channel_positions = self.layout_override.setdefault("channel_positions_mm", {})
        key = str(self.selected_channel_idx + 1)
        channel_positions.pop(key, None)
        updated = (constrained_position - reference_positions[self.selected_channel_idx]) * 1000.0
        if np.all(np.abs(updated) < 1e-9):
            channel_offsets.pop(key, None)
        else:
            channel_offsets[key] = [float(updated[0]), float(updated[1]), float(updated[2])]
        self._sync_editor_from_selection()
        self._apply_live_layout_update()

    def _on_save_layout_json(self):
        if self.layout_override_path is None:
            self.status.setText("Save skipped: no --electrode-layout-json file loaded")
            self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
            return
        with self.layout_override_path.open("w", encoding="utf-8") as handle:
            json.dump(self.layout_override, handle, indent=2)
            handle.write("\n")
        self.status.setText(f"Saved layout: {self.layout_override_path.name}")
        self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")

    def _set_config_enabled(self, enabled):
        for widget in (
            self.host_edit,
            self.port_edit,
            self.ch_edit,
            self.fs_edit,
            self.window_edit,
            self.data_update_edit,
            self.render_update_edit,
            self.layout_edit,
            self.arm_radius_edit,
            self.arm_length_edit,
            self.offset_edit,
            self.marker_size_edit,
            self.mesh_detail_edit,
            self.show_axis,
            self.show_grid,
            self.arm_model_edit,
            self.bandpass_check,
            self.bp_low_edit,
            self.bp_high_edit,
            self.notch_check,
            self.notch_freq_edit,
        ):
            widget.setEnabled(enabled)

        self.inspect_channel_edit.setEnabled(True)
        self.edit_dx.setEnabled(self.selected_channel_idx is not None)
        self.edit_dy.setEnabled(self.selected_channel_idx is not None)
        self.edit_dz.setEnabled(self.selected_channel_idx is not None)
        self.btn_reset_channel.setEnabled(self.selected_channel_idx is not None)
        self.btn_save_layout.setEnabled(True)
        self.column_edit.setEnabled(True)
        self.column_angle_edit.setEnabled(True)
        self.column_radial_edit.setEnabled(True)
        self.column_z_edit.setEnabled(True)
        self.btn_reset_column.setEnabled(True)

    def _electrode_positions_for_count(self, n_channels: int):
        arm_radius = float(self.arm_radius_edit.value()) / 100.0
        arm_length = float(self.arm_length_edit.value()) / 100.0
        radial_offset = float(self.offset_edit.value()) / 100.0
        positions = _build_electrode_positions(
            n_channels=n_channels,
            arm_radius_x=arm_radius * 1.12,
            arm_radius_y=arm_radius * 0.86,
            arm_length=arm_length,
            radial_offset=radial_offset,
        )
        return _apply_layout_override(positions, self.layout_override)

    def _apply_axes_style(self):
        pass

    def _refresh_scene(self, *_args):
        if self.main_view is None or self.orientation_view is None:
            return

        for attr in ("grid_item", "axis_item", "arm_mesh", "scatter"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self.main_view.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)

        for attr in ("orientation_grid", "orientation_axis"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self.orientation_view.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)

        arm_radius = float(self.arm_radius_edit.value()) / 100.0
        arm_length = float(self.arm_length_edit.value()) / 100.0
        radius_prox = arm_radius * 1.12
        radius_dist = arm_radius * 0.82
        detail = int(self.mesh_detail_edit.value())

        if self.show_grid.isChecked():
            self.grid_item = gl.GLGridItem()
            self.grid_item.setSize(x=radius_prox * 3.4, y=radius_prox * 3.4)
            self.grid_item.setSpacing(x=radius_prox * 0.4, y=radius_prox * 0.4)
            self.grid_item.translate(0.0, 0.0, -arm_length * 0.62)
            self.grid_item.setDepthValue(-10)
            self.main_view.addItem(self.grid_item)

        if self.show_axis.isChecked():
            self.axis_item = gl.GLAxisItem()
            self.axis_item.setSize(x=radius_prox * 2.2, y=radius_prox * 2.2, z=arm_length * 0.4)
            self.main_view.addItem(self.axis_item)

        model_path = self._model_path()
        if model_path is not None and model_path.exists():
            meshdata = _build_obj_meshdata(model_path, arm_radius=arm_radius, arm_length=arm_length)
            model_label = model_path.name
        else:
            meshdata = gl.MeshData.cylinder(rows=max(12, detail), cols=max(18, detail * 2), radius=[radius_prox, radius_dist], length=arm_length)
            model_label = "tapered forearm mesh"
        self.arm_mesh = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            color=(0.78, 0.67, 0.58, 0.32),
            shader="shaded",
            glOptions="translucent",
        )
        self.main_view.addItem(self.arm_mesh)
        self.model_info.setText(
            f"Model: {model_label} | layout={self.layout_edit.currentText()}"
            + (f" | override={self.layout_override_path.name}" if self.layout_override_path is not None else "")
        )

        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32), color=np.zeros((0, 4), dtype=np.float32), size=10.0, pxMode=True)
        self.main_view.addItem(self.scatter)

        self.orientation_grid = gl.GLGridItem()
        self.orientation_grid.setSize(x=2.0, y=2.0)
        self.orientation_grid.setSpacing(x=0.5, y=0.5)
        self.orientation_grid.translate(0.0, 0.0, -1.0)
        self.orientation_view.addItem(self.orientation_grid)

        self.orientation_axis = gl.GLAxisItem()
        self.orientation_axis.setSize(x=1.1, y=1.1, z=1.1)
        self.orientation_view.addItem(self.orientation_axis)
        self.orientation_view.setCameraPosition(distance=3.0, elevation=self.main_view.opts["elevation"], azimuth=self.main_view.opts["azimuth"])

        self._preview_positions_if_idle()
        self._render_dirty = True
        self._redraw_scatter()

    def _colors_from_rms(self, rms_values: np.ndarray):
        finite = rms_values[np.isfinite(rms_values)]
        if finite.size == 0:
            return np.ones((rms_values.size, 4), dtype=np.float32)

        if self.auto_scale.isChecked() or self.color_levels is None:
            lo = float(np.nanpercentile(finite, 5))
            hi = float(np.nanpercentile(finite, 95))
            if hi <= lo:
                hi = lo + 1e-6
            self.color_levels = (lo, hi)

        return _build_inferno_colors(rms_values.astype(np.float32, copy=False), self.color_levels)

    def _channel_details(self, channel_idx):
        if channel_idx is None or channel_idx < 0:
            return None
        if self.source is None:
            if channel_idx >= self.electrode_positions.shape[0]:
                return None
            zmq_idx = channel_idx
            label = f"CH{channel_idx + 1}"
        else:
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

    def _set_selected_channel(self, channel_idx):
        details = self._channel_details(channel_idx)
        if details is None:
            self.selected_channel_idx = None
            self.pick_info.setText("Selected: none")
            self.inspect_channel_edit.blockSignals(True)
            self.inspect_channel_edit.setValue(0)
            self.inspect_channel_edit.blockSignals(False)
            return
        self.selected_channel_idx = int(channel_idx)
        self.inspect_channel_edit.blockSignals(True)
        self.inspect_channel_edit.setValue(int(channel_idx) + 1)
        self.inspect_channel_edit.blockSignals(False)
        self.pick_info.setText(
            f"Selected: ch {details['channel_number']} | label={details['label']} | zmq_idx={details['zmq_index']}"
        )
        self._sync_editor_from_selection()

    def _redraw_scatter(self, *_args):
        if self.main_view is None or self.scatter is None:
            return

        if self.electrode_positions.size == 0:
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=np.zeros((0, 4), dtype=np.float32), size=1.0, pxMode=True)
            self.main_view.update()
            return

        colors = np.tile(np.array([[0.85, 0.4, 0.2, 0.95]], dtype=np.float32), (self.electrode_positions.shape[0], 1))
        if self.latest_rms is not None and self.latest_rms.size == self.electrode_positions.shape[0]:
            colors = self._colors_from_rms(self.latest_rms)

        sizes = np.full(self.electrode_positions.shape[0], float(max(4, self.marker_size_edit.value())), dtype=np.float32)
        if self.selected_channel_idx is not None and 0 <= self.selected_channel_idx < self.electrode_positions.shape[0]:
            colors[self.selected_channel_idx] = np.array([0.97, 0.99, 1.0, 1.0], dtype=np.float32)
            sizes[self.selected_channel_idx] = float(max(6, int(self.marker_size_edit.value() * 1.8)))

        self.scatter.setData(pos=self.electrode_positions, color=colors, size=sizes, pxMode=True)
        self.main_view.update()

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
        self.electrode_positions = self._electrode_positions_for_count(n_emg)
        self.latest_rms = None
        self.color_levels = None
        self.selected_channel_idx = None
        self.inspect_channel_edit.blockSignals(True)
        self.inspect_channel_edit.setRange(0, n_emg)
        self.inspect_channel_edit.setValue(0)
        self.inspect_channel_edit.blockSignals(False)
        self._refresh_scene()

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
        self.model_info.setText(
            f"Model: {self.arm_model_edit.currentText()} | layout={self.layout_edit.currentText()} | electrodes={n_emg}"
            + (f" | override={self.layout_override_path.name}" if self.layout_override_path is not None else "")
        )
        self.layout_info.setText(_reference_layout_summary(self.layout_edit.currentText(), self.layout_override))
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_config_enabled(False)
        self._poll_data()
        self._render_frame()

    def _on_stop(self):
        if self.source is not None:
            self.source.stop()
        self.source = None
        self.status.setText("Disconnected")
        self.status.setStyleSheet("color: #ff6666; font-weight: bold; font-size: 14px;")
        preview_channels = _preview_channel_count(self.layout_edit.currentText(), self.layout_override, int(self.ch_edit.value()))
        self.ch_info.setText(f"Channels: preview={preview_channels} ADC=0")
        self.fs_info.setText("Fs: ?")
        self.layout_info.setText(_reference_layout_summary(self.layout_edit.currentText(), self.layout_override))
        self.rms_stats.setText("RMS min/max/mean: N/A")
        self.scale_info.setText("Color scale: N/A")
        self.pick_info.setText("Selected: none")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_config_enabled(True)
        self.color_levels = None
        self.latest_rms = None
        self.electrode_positions = np.zeros((0, 3), dtype=np.float32)
        self.selected_channel_idx = None
        self.inspect_channel_edit.blockSignals(True)
        self.inspect_channel_edit.setRange(0, preview_channels)
        self.inspect_channel_edit.setValue(0)
        self.inspect_channel_edit.blockSignals(False)
        self._sync_editor_from_selection()
        self._sync_column_editor()
        self._refresh_scene()

    def _poll_data(self):
        if self.source is None or not self.source.running:
            return

        try:
            rms = self.source.latest_rms(window_ms=self.window_edit.value())
            self.latest_rms = rms
            if self.electrode_positions.shape[0] != rms.size:
                self.electrode_positions = self._electrode_positions_for_count(rms.size)
                if self.selected_channel_idx is not None and self.selected_channel_idx >= rms.size:
                    self.selected_channel_idx = None
                    self.pick_info.setText("Selected: none")
            self.status.setText("Streaming")
            self.status.setStyleSheet("color: #44ff44; font-weight: bold; font-size: 14px;")
            self._render_dirty = True
        except NotReadyError:
            self.status.setText("Waiting for data stream...")
            self.status.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        except Exception as exc:
            self.status.setText(f"Error: {str(exc)[:140]}")
            self.status.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px;")
            self._on_stop()

    def _render_frame(self):
        if self.source is None or not self.source.running:
            if self.auto_retry.isChecked() and not self.btn_start.isEnabled():
                now = time.time()
                if now - self.last_retry > 2.0:
                    self.last_retry = now
                    self._on_start()
            return

        if not self._render_dirty or self.latest_rms is None:
            return

        try:
            rms = self.latest_rms
            self._redraw_scatter()
            self.orientation_view.setCameraPosition(
                distance=3.0,
                elevation=self.main_view.opts["elevation"],
                azimuth=self.main_view.opts["azimuth"],
            )

            finite = rms[np.isfinite(rms)]
            if finite.size > 0:
                self.rms_stats.setText(
                    f"RMS min/max/mean: {float(np.nanmin(finite)):.3f} / {float(np.nanmax(finite)):.3f} / {float(np.nanmean(finite)):.3f}"
                )
                self.scale_info.setText(
                    f"Color scale: {self.color_levels[0]:.3f} to {self.color_levels[1]:.3f} | Data={self.data_update_edit.value()} ms | Render={self.render_update_edit.value()} ms | Mesh={self.mesh_detail_edit.value()}"
                )
            self._render_dirty = False
        except Exception as exc:
            self.status.setText(f"Error: {str(exc)[:140]}")
            self.status.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px;")
            self._on_stop()

    def closeEvent(self, event):
        if self.source is not None:
            self.source.stop()
        event.accept()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Open Ephys realtime 3D arm viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys ZMQ host")
    parser.add_argument("--port", type=int, default=5556, help="Open Ephys ZMQ data port")
    parser.add_argument("--fs", type=float, default=0.0, help="Sampling rate in Hz (0 = auto-detect from stream)")
    parser.add_argument("--channels", type=int, default=0, help="EMG channel count to use (0 = auto-detect non-ADC channels)")
    parser.add_argument("--window-ms", type=int, default=250, help="RMS window in milliseconds")
    parser.add_argument("--data-update-ms", type=int, default=50, help="Signal polling interval in milliseconds")
    parser.add_argument("--update-ms", type=int, default=100, help="Viewer refresh period in milliseconds")
    parser.add_argument(
        "--layout",
        default="hdemg128_vertical_columns",
        choices=["hdemg128_vertical_columns", "auto"],
        help="Electrode arrangement mode around the arm",
    )
    parser.add_argument("--arm-radius-mm", type=int, default=42, help="Approximate forearm radius in mm")
    parser.add_argument("--arm-length-mm", type=int, default=260, help="Approximate forearm model length in mm")
    parser.add_argument("--electrode-offset-mm", type=int, default=7, help="Offset from the skin surface in mm")
    parser.add_argument("--marker-size", type=int, default=12, help="Electrode marker size in points")
    parser.add_argument("--mesh-detail", type=int, default=28, help="Arm mesh detail level")
    parser.add_argument("--arm-model", default="right_arm.obj", help="Arm model asset to render (right_arm.obj, left_arm.obj, or cylinder)")
    parser.add_argument(
        "--electrode-layout-json",
        default=None,
        help="Optional JSON file with manual electrode placement overrides (relative paths resolve next to this script)",
    )
    parser.add_argument("--show-axis", action="store_true", help="Show the 3D axes")
    parser.add_argument("--no-grid", dest="show_grid", action="store_false", help="Disable the floor grid")
    parser.add_argument("--bp-low", type=float, default=20.0, help="Bandpass low cutoff in Hz")
    parser.add_argument("--bp-high", type=float, default=500.0, help="Bandpass high cutoff in Hz")
    parser.add_argument("--no-bandpass", dest="bandpass", action="store_false", help="Disable bandpass filtering")
    parser.add_argument("--notch-freq", type=float, default=60.0, help="Notch frequency in Hz")
    parser.add_argument("--no-notch", dest="notch", action="store_false", help="Disable notch filtering")
    parser.set_defaults(bandpass=True, notch=True, show_axis=False, show_grid=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not HAS_QT:
        raise RuntimeError("PyQt5 is required for the 3D arm viewer.")

    app = QApplication([])
    win = Arm3DWindow(args)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()