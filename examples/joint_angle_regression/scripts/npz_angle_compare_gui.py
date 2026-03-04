import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]


def _cnn_custom_objects():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return {}

    class SinusoidalPositionalEncoding(keras.layers.Layer):
        def call(self, x):
            length = tf.shape(x)[1]
            dim = tf.shape(x)[2]
            pos = tf.cast(tf.range(length)[:, None], tf.float32)
            i = tf.cast(tf.range(dim)[None, :], tf.float32)
            angle_rates = 1.0 / tf.pow(
                10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(dim, tf.float32)
            )
            angle_rads = pos * angle_rates
            sin_terms = tf.sin(angle_rads[:, 0::2])
            cos_terms = tf.cos(angle_rads[:, 1::2])
            pe = tf.reshape(tf.stack([sin_terms, cos_terms], axis=-1), (length, -1))[
                :, :dim
            ]
            return x + pe[None, :, :]

    return {"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding}


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_regressor as tr

try:
    from lsl_utils import ANGLE_KEYS
except Exception:
    ANGLE_KEYS = [
        "thumb_cmc_mcp",
        "thumb_ip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
    ]


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_scalar_from_npz(data, key, default=""):
    if key not in data.files:
        return default
    v = data[key]
    try:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return default
            return v.item()
        return v
    except Exception:
        return default


def _angles_to_landmarks_3d(angles14: np.ndarray) -> np.ndarray:
    a = np.asarray(angles14, dtype=np.float32).reshape(-1)
    out = np.zeros(14, dtype=np.float32)
    out[: min(14, a.size)] = a[:14]
    a = out

    lmk = np.zeros((21, 3), dtype=np.float32)
    lmk[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    lmk[5] = np.array([-30.0, -5.0, 0.0], dtype=np.float32)
    lmk[9] = np.array([-10.0, -2.0, 0.0], dtype=np.float32)
    lmk[13] = np.array([12.0, -2.0, 0.0], dtype=np.float32)
    lmk[17] = np.array([30.0, -5.0, 0.0], dtype=np.float32)
    lmk[1] = np.array([-42.0, -18.0, -8.0], dtype=np.float32)

    finger_lengths = {
        "index": (33.0, 23.0, 16.0),
        "middle": (36.0, 26.0, 18.0),
        "ring": (34.0, 24.0, 17.0),
        "pinky": (29.0, 19.0, 14.0),
    }
    finger_splay = {"index": -0.15, "middle": -0.05, "ring": 0.08, "pinky": 0.18}

    def add_chain(base_idx, out_idxs, mcp_deg, pip_deg, dip_deg, lengths, splay):
        base = lmk[base_idx].copy()
        t1 = np.radians(float(mcp_deg))
        t2 = t1 + np.radians(float(pip_deg))
        t3 = t2 + np.radians(float(dip_deg))
        cur = base
        for j, (theta, seg_len) in enumerate(zip([t1, t2, t3], lengths)):
            dy = -seg_len * np.cos(theta)
            dz = seg_len * np.sin(theta)
            dx = splay * seg_len
            cur = cur + np.array([dx, dy, dz], dtype=np.float32)
            lmk[out_idxs[j]] = cur

    add_chain(
        5, [6, 7, 8], a[2], a[3], a[4], finger_lengths["index"], finger_splay["index"]
    )
    add_chain(
        9,
        [10, 11, 12],
        a[5],
        a[6],
        a[7],
        finger_lengths["middle"],
        finger_splay["middle"],
    )
    add_chain(
        13,
        [14, 15, 16],
        a[8],
        a[9],
        a[10],
        finger_lengths["ring"],
        finger_splay["ring"],
    )
    add_chain(
        17,
        [18, 19, 20],
        a[11],
        a[12],
        a[13],
        finger_lengths["pinky"],
        finger_splay["pinky"],
    )

    thumb_cmc = np.radians(float(a[0]))
    thumb_ip = np.radians(float(a[1]))
    t1 = 0.55 * thumb_cmc
    t2 = 0.45 * thumb_cmc
    t3 = thumb_ip
    thumb_lengths = (22.0, 19.0, 15.0)
    cur = lmk[1].copy()
    thumb_base_dir = np.array([-0.75, -0.65, -0.10], dtype=np.float32)
    thumb_base_dir = thumb_base_dir / (np.linalg.norm(thumb_base_dir) + 1e-8)
    for idx_out, theta, seg_len in zip(
        [2, 3, 4], [t1, t1 + t2, t1 + t2 + t3], thumb_lengths
    ):
        dy = -seg_len * np.cos(theta)
        dz = seg_len * np.sin(theta)
        step = np.array(
            [
                thumb_base_dir[0] * seg_len,
                0.8 * dy,
                0.7 * dz + thumb_base_dir[2] * seg_len,
            ],
            dtype=np.float32,
        )
        cur = cur + step
        lmk[idx_out] = cur

    return lmk


def _render_hand_iso(angle_vec_14, title=""):
    hand_landmarks = _angles_to_landmarks_3d(np.asarray(angle_vec_14, dtype=np.float32))
    w, h = 500, 500
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    grid_spacing = 50
    for i in range(0, w, grid_spacing):
        cv2.line(canvas, (i, 0), (i, h), (55, 55, 55), 1)
    for i in range(0, h, grid_spacing):
        cv2.line(canvas, (0, i), (w, i), (55, 55, 55), 1)

    if title:
        cv2.putText(
            canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1
        )

    valid_points = [p for p in hand_landmarks if np.linalg.norm(p) > 1e-3]
    if not valid_points:
        return canvas

    centroid = np.mean(valid_points, axis=0)
    max_extent = max(np.max(np.abs(p - centroid)) for p in valid_points)
    max_extent = max(max_extent, 1.0)
    scale = (w * 0.28) / max_extent

    iso_angle = np.radians(30)
    cos_a, sin_a = np.cos(iso_angle), np.sin(iso_angle)

    THUMB_COLOR = (0, 200, 255)
    INDEX_COLOR = (0, 255, 100)
    MIDDLE_COLOR = (255, 200, 0)
    RING_COLOR = (255, 0, 150)
    PINKY_COLOR = (100, 100, 255)
    PALM_COLOR = (180, 180, 180)

    connections = [
        (0, 1, THUMB_COLOR),
        (1, 2, THUMB_COLOR),
        (2, 3, THUMB_COLOR),
        (3, 4, THUMB_COLOR),
        (0, 5, INDEX_COLOR),
        (5, 6, INDEX_COLOR),
        (6, 7, INDEX_COLOR),
        (7, 8, INDEX_COLOR),
        (5, 9, PALM_COLOR),
        (9, 13, PALM_COLOR),
        (13, 17, PALM_COLOR),
        (0, 17, PALM_COLOR),
        (9, 10, MIDDLE_COLOR),
        (10, 11, MIDDLE_COLOR),
        (11, 12, MIDDLE_COLOR),
        (13, 14, RING_COLOR),
        (14, 15, RING_COLOR),
        (15, 16, RING_COLOR),
        (17, 18, PINKY_COLOR),
        (18, 19, PINKY_COLOR),
        (19, 20, PINKY_COLOR),
    ]

    def proj(p, cx, cy):
        x, y, z = -p[0], -p[1], p[2]
        u = int((x * cos_a - z * cos_a) * scale) + cx
        v = int((-y + x * sin_a + z * sin_a) * scale * 0.8) + cy
        return u, v

    cx, cy = w // 2, h // 2
    for s, e, col in connections:
        p1 = hand_landmarks[s]
        p2 = hand_landmarks[e]
        if np.linalg.norm(p1) < 1e-3 or np.linalg.norm(p2) < 1e-3:
            continue
        c1 = p1 - centroid
        c2 = p2 - centroid
        u1, v1 = proj(c1, cx, cy)
        u2, v2 = proj(c2, cx, cy)
        cv2.line(canvas, (u1, v1), (u2, v2), col, 2)
        cv2.circle(canvas, (u1, v1), 3, col, -1)

    return canvas


class NPZAngleCompareGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPZ Measured vs Predicted Angle Viewer")
        self.setMinimumSize(1300, 760)

        self.dataset_path = None
        self.model_dir = None
        self.feature_extractor_path = None

        self.data = None
        self.emg = None
        self.measured = None
        self.imu = None
        self.measured_keys = []
        self.timestamps = None

        self.extractor = None
        self.regressor = None
        self.scaler = None
        self.model_type = "mlp_regressor"
        self.target_scaler = None
        self.personalization = None
        self.predicted = None
        self.predicted_raw = None
        self.predicted_smoothed = None
        self.pred_keys = []
        self.common_keys = []
        self.model_metrics = {}
        self.extractor_load_error = ""
        self.runtime_feature_mode = None
        self.runtime_emg_transform = None
        self.runtime_use_imu = None
        self.runtime_extractor_used = False
        self.runtime_pipeline_note = ""

        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self._apply_session_theme()
        self._build_ui()

    def _apply_session_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1f232a;
                color: #d6dbe1;
            }
            QGroupBox {
                border: 1px solid #3a4048;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #cfd6dd;
            }
            QLineEdit, QComboBox, QPushButton {
                background-color: #2b313a;
                color: #e6ebf0;
                border: 1px solid #4b535f;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QPushButton:hover {
                background-color: #353d48;
            }
            QSlider::groove:horizontal {
                background: #3a4048;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #8fa7bb;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            """
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        view_row = QHBoxLayout()
        self.measured_label = QLabel()
        self.pred_label = QLabel()
        self.measured_label.setMinimumSize(500, 420)
        self.pred_label.setMinimumSize(500, 420)
        self.measured_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.measured_label.setStyleSheet("background:#111; border:1px solid #444;")
        self.pred_label.setStyleSheet("background:#111; border:1px solid #444;")
        view_row.addWidget(self.measured_label, stretch=1)
        view_row.addWidget(self.pred_label, stretch=1)
        root.addLayout(view_row)

        playback = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._render_current)
        self.frame_label = QLabel("Frame: 0")
        playback.addWidget(self.play_btn)
        playback.addWidget(self.slider, stretch=1)
        playback.addWidget(self.frame_label)
        root.addLayout(playback)

        self.error_label = QLabel("Error: N/A")
        root.addWidget(self.error_label)

        ctrl = QGroupBox("Controls (collapsed)")
        self.controls_group = ctrl
        ctrl.setCheckable(True)
        ctrl.setChecked(False)
        ctrl.toggled.connect(self._on_controls_toggled)
        ctrl_outer = QVBoxLayout(ctrl)
        self.controls_body = QWidget()
        ctrl_layout = QGridLayout(self.controls_body)

        self.dataset_edit = QLineEdit()
        self.dataset_edit.editingFinished.connect(self._load_dataset)
        btn_dataset = QPushButton("Dataset .npz")
        btn_dataset.clicked.connect(self._pick_dataset)

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.textChanged.connect(self._refresh_compatibility_indicator)
        btn_model_dir = QPushButton("Model Dir")
        btn_model_dir.clicked.connect(self._pick_model_dir)

        self.extractor_edit = QLineEdit(
            str(ROOT / "models" / "sub-001" / "ses-001to005_feature_extractor.h5")
        )
        self.extractor_edit.textChanged.connect(self._refresh_compatibility_indicator)
        btn_extractor = QPushButton("Extractor .h5")
        btn_extractor.clicked.connect(self._pick_extractor)

        btn_load_model = QPushButton("Load Model")
        btn_load_model.clicked.connect(self._load_model)
        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self._run_prediction)

        ctrl_layout.addWidget(QLabel("Dataset"), 0, 0)
        ctrl_layout.addWidget(self.dataset_edit, 0, 1)
        ctrl_layout.addWidget(btn_dataset, 0, 2)
        ctrl_layout.addWidget(QLabel("(auto-load on select)"), 0, 3)

        ctrl_layout.addWidget(QLabel("Regressor directory"), 1, 0)
        ctrl_layout.addWidget(self.model_dir_edit, 1, 1)
        ctrl_layout.addWidget(btn_model_dir, 1, 2)
        ctrl_layout.addWidget(btn_load_model, 1, 3)

        ctrl_layout.addWidget(QLabel("Feature extractor"), 2, 0)
        ctrl_layout.addWidget(self.extractor_edit, 2, 1)
        ctrl_layout.addWidget(btn_extractor, 2, 2)
        ctrl_layout.addWidget(btn_predict, 2, 3)

        self.status_label = QLabel("Status: select dataset and model")
        ctrl_layout.addWidget(self.status_label, 3, 0, 1, 4)

        self.compat_label = QLabel("Compatibility: unknown")
        ctrl_layout.addWidget(self.compat_label, 4, 0, 1, 4)

        self.profile_label = QLabel("Model profile: not loaded")
        self.profile_label.setWordWrap(True)
        ctrl_layout.addWidget(self.profile_label, 5, 0, 1, 4)

        self.session_map_label = QLabel(
            "Session map: compare_feature_mode=?, compare_emg_transform=?, compare_extractor=?"
        )
        self.session_map_label.setWordWrap(True)
        ctrl_layout.addWidget(self.session_map_label, 6, 0, 1, 4)

        ctrl_outer.addWidget(self.controls_body)
        root.addWidget(ctrl)

        self.filters_group = QGroupBox("Filters / Smoothing (collapsed)")
        self.filters_group.setCheckable(True)
        self.filters_group.setChecked(False)
        self.filters_group.toggled.connect(self._on_filters_toggled)
        filters_layout = QVBoxLayout(self.filters_group)
        self.filters_body = QWidget()
        smooth_row = QHBoxLayout(self.filters_body)
        smooth_row.addWidget(QLabel("Prediction smoothing"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItems(["None", "EMA", "Moving Avg", "Median"])
        self.smooth_method_combo.setCurrentText("EMA")
        self.smooth_method_combo.currentTextChanged.connect(
            self._on_smoothing_controls_changed
        )
        smooth_row.addWidget(self.smooth_method_combo)
        smooth_row.addWidget(QLabel("Strength"))
        self.smooth_strength_slider = QSlider(Qt.Horizontal)
        self.smooth_strength_slider.setMinimum(0)
        self.smooth_strength_slider.setMaximum(100)
        self.smooth_strength_slider.setValue(35)
        self.smooth_strength_slider.valueChanged.connect(
            self._on_smoothing_controls_changed
        )
        smooth_row.addWidget(self.smooth_strength_slider)
        self.smooth_strength_label = QLabel()
        smooth_row.addWidget(self.smooth_strength_label)
        self.show_raw_checkbox = QCheckBox("Show raw")
        self.show_raw_checkbox.setChecked(False)
        self.show_raw_checkbox.toggled.connect(self._on_smoothing_controls_changed)
        smooth_row.addWidget(self.show_raw_checkbox)
        filters_layout.addWidget(self.filters_body)
        root.addWidget(self.filters_group)
        self._update_smoothing_strength_label()
        self._on_controls_toggled(False)
        self._on_filters_toggled(False)
        self._refresh_compatibility_indicator()
        self._refresh_model_profile()

    def _on_filters_toggled(self, checked):
        self.filters_body.setVisible(bool(checked))
        if checked:
            self.filters_group.setTitle("Filters / Smoothing (expanded)")
        else:
            self.filters_group.setTitle("Filters / Smoothing (collapsed)")

    def _on_controls_toggled(self, checked):
        self.controls_body.setVisible(bool(checked))
        if checked:
            self.controls_group.setTitle("Controls (expanded)")
        else:
            self.controls_group.setTitle("Controls (collapsed)")

    def _update_smoothing_strength_label(self):
        self.smooth_strength_label.setText(
            f"{int(self.smooth_strength_slider.value())}%"
        )

    def _on_smoothing_controls_changed(self):
        self._update_smoothing_strength_label()
        if self.predicted_raw is None:
            return
        self.predicted_smoothed = self._smooth_predictions(self.predicted_raw)
        self.predicted = (
            self.predicted_raw
            if self.show_raw_checkbox.isChecked()
            else self.predicted_smoothed
        )
        self._render_current()

    def _smooth_predictions(self, pred):
        arr = np.asarray(pred, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            return arr.copy()

        method = self.smooth_method_combo.currentText().strip().lower()
        strength = int(self.smooth_strength_slider.value())
        if method == "none" or strength <= 0:
            return arr.copy()

        if method == "ema":
            alpha = max(0.02, 1.0 - (strength / 100.0) * 0.95)
            out = np.empty_like(arr)
            out[0] = arr[0]
            for i in range(1, arr.shape[0]):
                out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
            return out

        if method == "moving avg":
            win = 1 + int(round((strength / 100.0) * 24))
            if win <= 1:
                return arr.copy()
            half = win // 2
            pad = np.pad(arr, ((half, half), (0, 0)), mode="edge")
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i] = pad[i : i + win].mean(axis=0)
            return out

        if method == "median":
            win = 3 + 2 * int(round((strength / 100.0) * 7))
            half = win // 2
            pad = np.pad(arr, ((half, half), (0, 0)), mode="edge")
            out = np.empty_like(arr)
            for i in range(arr.shape[0]):
                out[i] = np.median(pad[i : i + win], axis=0)
            return out

        return arr.copy()

    def _set_compatibility_indicator(self, level, text):
        styles = {
            "ok": "color:#7bd88f;",
            "warn": "color:#ffd866;",
            "bad": "color:#ff6b6b;",
            "info": "color:#d0d0d0;",
        }
        self.compat_label.setStyleSheet(styles.get(level, styles["info"]))
        self.compat_label.setText(text)

    def _refresh_compatibility_indicator(self):
        md_text = self.model_dir_edit.text().strip()
        if not md_text:
            self._set_compatibility_indicator(
                "info", "Compatibility: select a model directory"
            )
            return

        md = Path(md_text)
        if not md.exists():
            self._set_compatibility_indicator(
                "bad", "Compatibility: ❌ model directory does not exist"
            )
            return

        reg_path = md / "mlp_regressor.pkl"
        scaler_path = md / "scaler.pkl"
        cnn_path = md / "cnn_attention_regressor.h5"
        is_cnn = cnn_path.exists()
        if not is_cnn and (not reg_path.exists() or not scaler_path.exists()):
            self._set_compatibility_indicator(
                "bad",
                "Compatibility: ❌ missing cnn_attention_regressor.h5 or mlp_regressor.pkl+scaler.pkl",
            )
            return

        metrics = (
            self.model_metrics
            if self.model_metrics
            else _load_json(md / "metrics.json")
        )
        feature_mode = metrics.get(
            "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
        )

        extractor_text = self.extractor_edit.text().strip()
        extractor_path = Path(extractor_text) if extractor_text else None
        if feature_mode == "extractor":
            if (
                extractor_path is None
                or not extractor_path.exists()
                or extractor_path.name.lower() in ("none", "raw", "flat")
            ):
                metrics_fx = metrics.get("feature_extractor")
                if metrics_fx:
                    candidate = Path(metrics_fx)
                    if not candidate.is_absolute():
                        candidate = ROOT / candidate
                    if candidate.exists():
                        extractor_path = candidate
                        self.extractor_edit.setText(str(candidate))

            if (
                extractor_path is None
                or not extractor_path.exists()
                or extractor_path.name.lower() in ("none", "raw", "flat")
            ):
                self._set_compatibility_indicator(
                    "bad",
                    "Compatibility: ❌ model requires extractor, but no valid .h5 is selected",
                )
                return

        model_dir_now = md
        extractor_now = extractor_text
        model_loaded = self.regressor is not None and (
            self.model_type == "cnn_attention_regressor" or self.scaler is not None
        )
        stale = (
            self.model_dir is None
            or Path(self.model_dir) != model_dir_now
            or (self.feature_extractor_path or "") != extractor_now
        )

        if model_loaded and not stale:
            if feature_mode == "extractor" and self.extractor is None:
                self._set_compatibility_indicator(
                    "bad", "Compatibility: ❌ extractor model required but not loaded"
                )
            else:
                self._set_compatibility_indicator(
                    "ok", "Compatibility: ✅ selected files are compatible and loaded"
                )
            return

        if stale and model_loaded:
            self._set_compatibility_indicator(
                "warn",
                "Compatibility: ⚠ selections changed; model will auto-reload on Predict",
            )
            return

        if feature_mode == "extractor":
            self._set_compatibility_indicator(
                "warn",
                "Compatibility: ⚠ files look valid; load model to validate extractor runtime",
            )
        else:
            self._set_compatibility_indicator(
                "warn", "Compatibility: ⚠ files look valid; load model to finalize"
            )

    def _refresh_model_profile(self):
        md_text = self.model_dir_edit.text().strip()
        if not md_text:
            self.profile_label.setText("Model profile: select a model directory")
            self.session_map_label.setText(
                "Session map: compare_feature_mode=?, compare_emg_transform=?, compare_extractor=?"
            )
            return

        md = Path(md_text)
        metrics = (
            self.model_metrics
            if self.model_metrics
            else _load_json(md / "metrics.json")
        )
        metrics_mode = metrics.get(
            "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
        )
        metrics_transform = metrics.get("emg_transform", "none")
        metrics_use_imu = bool(
            metrics.get("imu_included", metrics.get("use_imu", False))
        )

        extractor_text = self.extractor_edit.text().strip()
        extractor_path = Path(extractor_text) if extractor_text else None
        extractor_exists = bool(extractor_path is not None and extractor_path.exists())

        runtime_mode = (
            self.runtime_feature_mode if self.runtime_feature_mode else metrics_mode
        )
        runtime_transform = (
            self.runtime_emg_transform
            if self.runtime_emg_transform is not None
            else metrics_transform
        )
        runtime_use_imu = (
            metrics_use_imu
            if self.runtime_use_imu is None
            else bool(self.runtime_use_imu)
        )

        n_in = None
        if self.scaler is not None:
            n_in = getattr(self.scaler, "n_features_in_", None)

        if self.model_type == "cnn_attention_regressor":
            pipeline = "cnn_attention(.h5) -> target_scaler(optional)"
        elif runtime_mode == "extractor" and self.runtime_extractor_used:
            pipeline = "extractor(.h5) -> scaler -> regressor"
        elif runtime_mode == "extractor" and not self.runtime_extractor_used:
            pipeline = "extractor required, but not loaded"
        elif runtime_mode == "bandpower_stats":
            pipeline = "bandpower_stats -> scaler -> regressor"
        elif runtime_mode == "emd_stats":
            pipeline = f"emd_stats(imfs={metrics.get('emd_max_imfs', 3)}) -> scaler -> regressor"
        else:
            pipeline = "raw_flat -> scaler -> regressor"

        note = (
            f" | note: {self.runtime_pipeline_note}"
            if self.runtime_pipeline_note
            else ""
        )
        personal = "on" if self.personalization is not None else "off"
        self.profile_label.setText(
            "Model profile: "
            f"metrics_mode={metrics_mode}, runtime_mode={runtime_mode}, "
            f"emg_transform={runtime_transform}, imu={runtime_use_imu}, "
            f"extractor_selected={extractor_exists}, extractor_loaded={self.extractor is not None}, "
            f"personalization={personal}, scaler_in={n_in}, model_type={self.model_type}, pipeline={pipeline}{note}"
        )

        self.session_map_label.setText(
            "Session map: "
            f"compare_feature_mode={runtime_mode}, "
            f"compare_emg_transform={runtime_transform}, "
            f"compare_extractor={'loaded' if (self.extractor is not None and self.runtime_extractor_used) else 'none'}"
        )

    def _pick_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select NPZ dataset", str(ROOT / "data"), "NPZ (*.npz)"
        )
        if path:
            self.dataset_edit.setText(path)
            self._load_dataset()

    def _pick_model_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select model directory", str(ROOT / "models")
        )
        if path:
            self.model_dir_edit.setText(path)
            self._refresh_compatibility_indicator()
            self._refresh_model_profile()

    def _pick_extractor(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select feature extractor", str(ROOT / "models"), "Keras Model (*.h5)"
        )
        if path:
            self.extractor_edit.setText(path)
            self._refresh_compatibility_indicator()
            self._refresh_model_profile()

    def _load_dataset(self):
        p = Path(self.dataset_edit.text().strip())
        if not p.exists():
            QMessageBox.warning(self, "Missing", "Dataset file not found.")
            return
        d = np.load(p, allow_pickle=True)
        if "emg" not in d or "angles" not in d:
            QMessageBox.warning(
                self, "Invalid", "Dataset must contain 'emg' and 'angles'."
            )
            return

        self.data = d
        self.emg = tr._normalize_emg(np.asarray(d["emg"], dtype=np.float32))
        self.measured = np.asarray(d["angles"], dtype=np.float32)
        self.imu = None
        if "imu" in d.files:
            try:
                self.imu = tr._normalize_imu(np.asarray(d["imu"], dtype=np.float32))
            except Exception:
                self.imu = None

        if "angle_keys" in d.files:
            try:
                keys = d["angle_keys"]
                self.measured_keys = [str(x) for x in keys.tolist()]
            except Exception:
                self.measured_keys = list(ANGLE_KEYS[: self.measured.shape[1]])
        else:
            self.measured_keys = list(ANGLE_KEYS[: self.measured.shape[1]])

        self.timestamps = (
            np.asarray(d["timestamps"], dtype=np.float64)
            if "timestamps" in d.files
            else None
        )

        n = int(self.measured.shape[0])
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(0)
        self.status_label.setText(
            f"Status: dataset loaded ({n} windows, {self.measured.shape[1]} angles)"
        )
        self._refresh_compatibility_indicator()
        self._refresh_model_profile()
        self._render_current()

    def _load_model(self):
        md = Path(self.model_dir_edit.text().strip())
        if not md.exists():
            QMessageBox.warning(self, "Missing", "Model directory not found.")
            return

        reg_path = md / "mlp_regressor.pkl"
        scaler_path = md / "scaler.pkl"
        cnn_path = md / "cnn_attention_regressor.h5"
        if cnn_path.exists():
            try:
                from tensorflow.keras.models import load_model

                self.regressor = load_model(
                    str(cnn_path),
                    compile=False,
                    custom_objects=_cnn_custom_objects(),
                )
                self.scaler = None
                self.model_type = "cnn_attention_regressor"
            except Exception as exc:
                QMessageBox.warning(
                    self, "Load failed", f"Failed to load CNN model:\n{exc}"
                )
                return
        else:
            if not reg_path.exists() or not scaler_path.exists():
                QMessageBox.warning(
                    self,
                    "Missing",
                    "Model dir must contain cnn_attention_regressor.h5 or mlp_regressor.pkl + scaler.pkl.",
                )
                return
            with open(reg_path, "rb") as f:
                self.regressor = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.model_type = "mlp_regressor"

        target_scaler_path = md / "target_scaler.pkl"
        if target_scaler_path.exists():
            with open(target_scaler_path, "rb") as f:
                self.target_scaler = pickle.load(f)
        else:
            self.target_scaler = None

        if (md / "personalization_diagonal_identity.pkl").exists():
            self.personalization = tr.load_personalization(
                md / "personalization_diagonal_identity.pkl"
            )
        else:
            self.personalization = tr.load_personalization(md / "personalization.pkl")

        self.model_metrics = _load_json(md / "metrics.json")
        self.pred_keys = (
            [str(x) for x in self.model_metrics.get("angle_keys", [])]
            if self.model_metrics.get("angle_keys")
            else []
        )

        feature_mode = self.model_metrics.get(
            "emg_feature_mode", self.model_metrics.get("feature_mode", "raw_flat")
        )
        fx_text = self.extractor_edit.text().strip()
        fx = Path(fx_text) if fx_text else None

        if feature_mode == "extractor":
            if (
                fx is None
                or not fx.exists()
                or fx.name.lower() in ("none", "raw", "flat")
            ):
                metrics_fx = self.model_metrics.get("feature_extractor")
                if metrics_fx:
                    metrics_fx_path = Path(metrics_fx)
                    if not metrics_fx_path.is_absolute():
                        metrics_fx_path = ROOT / metrics_fx_path
                    if metrics_fx_path.exists():
                        fx = metrics_fx_path
                        self.extractor_edit.setText(str(metrics_fx_path))

        self.extractor = None
        self.extractor_load_error = ""
        if (
            fx is not None
            and fx.exists()
            and fx.name.lower() not in ("none", "raw", "flat")
        ):
            try:
                if fx.suffix.lower() == ".h5" and hasattr(
                    tr, "KerasFeatureExtractorLocal"
                ):
                    self.extractor = tr.KerasFeatureExtractorLocal(str(fx))
                else:
                    self.extractor = tr.load_feature_extractor(str(fx))
            except Exception as exc:
                self.extractor = None
                self.extractor_load_error = str(exc)

        self.model_dir = md
        self.feature_extractor_path = str(fx) if fx is not None else ""
        self.runtime_feature_mode = feature_mode
        self.runtime_emg_transform = self.model_metrics.get("emg_transform", "none")
        self.runtime_use_imu = bool(
            self.model_metrics.get(
                "imu_included", self.model_metrics.get("use_imu", False)
            )
        )
        self.runtime_extractor_used = (
            self.extractor is not None and feature_mode == "extractor"
        )
        self.runtime_pipeline_note = "model loaded"

        if feature_mode == "extractor" and self.extractor is None:
            detail = (
                f" ({self.extractor_load_error})" if self.extractor_load_error else ""
            )
            self.status_label.setText(
                f"Status: model loaded, but extractor unavailable{detail}"
            )
        else:
            self.status_label.setText(f"Status: model loaded ({md.name})")
        self._refresh_compatibility_indicator()
        self._refresh_model_profile()

    def _feature_mode_fallback(self):
        if self.emg is None or self.scaler is None:
            return None
        n_in = getattr(self.scaler, "n_features_in_", None)
        if n_in is None:
            return None
        imu_feat = None
        if self.imu is not None:
            try:
                imu_feat = tr._imu_features(self.imu[:1])
            except Exception:
                imu_feat = None
        for cand in ("raw_flat", "bandpower_stats", "emd_stats"):
            try:
                feats = tr.extract_features(
                    self.emg[:1],
                    None,
                    feature_mode=cand,
                    emg_transform="none",
                    emd_max_imfs=int(self.model_metrics.get("emd_max_imfs", 3))
                    if self.model_metrics
                    else 3,
                )
                if int(feats.shape[1]) == int(n_in):
                    return cand, False
                if imu_feat is not None:
                    f2 = np.concatenate([feats, imu_feat], axis=1)
                    if int(f2.shape[1]) == int(n_in):
                        return cand, True
            except Exception:
                continue
        return None

    def _run_prediction(self):
        model_dir_now = (
            Path(self.model_dir_edit.text().strip())
            if self.model_dir_edit.text().strip()
            else None
        )
        extractor_path_now = self.extractor_edit.text().strip()
        need_reload = (
            self.regressor is None
            or (self.model_type != "cnn_attention_regressor" and self.scaler is None)
            or self.model_dir is None
            or model_dir_now is None
            or Path(self.model_dir) != model_dir_now
            or (self.feature_extractor_path or "") != extractor_path_now
        )
        if need_reload:
            self._load_model()

        if self.emg is None or self.measured is None:
            QMessageBox.warning(self, "Missing", "Load dataset first.")
            return
        if self.regressor is None or (
            self.model_type != "cnn_attention_regressor" and self.scaler is None
        ):
            QMessageBox.warning(self, "Missing", "Load model first.")
            return

        md = Path(self.model_dir_edit.text().strip())
        metrics = (
            self.model_metrics
            if self.model_metrics
            else _load_json(md / "metrics.json")
        )
        feature_mode = metrics.get(
            "emg_feature_mode", metrics.get("feature_mode", "raw_flat")
        )
        emg_transform = metrics.get("emg_transform", "none")
        emd_max_imfs = int(metrics.get("emd_max_imfs", 3))

        try:
            if feature_mode == "extractor" and self.extractor is None:
                fallback = self._feature_mode_fallback()
                if fallback is None:
                    reason = (
                        f"\n\nExtractor load error: {self.extractor_load_error}"
                        if self.extractor_load_error
                        else ""
                    )
                    QMessageBox.warning(
                        self,
                        "Extractor required",
                        "This model requires a feature extractor, but it could not be loaded.\n"
                        "TensorFlow runtime is unavailable, and no compatible raw-feature fallback was found for this scaler.\n\n"
                        "Use a model trained with raw features or fix TensorFlow in this environment."
                        f"{reason}",
                    )
                    self.status_label.setText(
                        "Status: prediction blocked (extractor unavailable)"
                    )
                    self.runtime_feature_mode = "extractor"
                    self.runtime_emg_transform = emg_transform
                    self.runtime_use_imu = bool(metrics.get("imu_included", False))
                    self.runtime_extractor_used = False
                    self.runtime_pipeline_note = "blocked: extractor unavailable"
                    self._refresh_model_profile()
                    return
                feature_mode, fallback_use_imu = fallback
                self.status_label.setText(
                    f"Status: extractor unavailable, using fallback feature mode '{fallback}'"
                )
                self.runtime_pipeline_note = (
                    "extractor unavailable -> fallback feature mode"
                )
            else:
                fallback_use_imu = False
                self.runtime_pipeline_note = "using model-configured feature path"

            if self.model_type == "cnn_attention_regressor":
                emg_in = tr._apply_emg_transform(self.emg[..., 0], emg_transform)[
                    ..., None
                ]
                want_imu = bool(
                    metrics.get("use_imu", metrics.get("imu_included", False))
                )
                model_input = emg_in
                if want_imu:
                    if self.imu is None:
                        raise ValueError(
                            "Model expects IMU features but dataset has no IMU."
                        )
                    imu_feat_all = tr._imu_features(self.imu)
                    model_input = [emg_in, imu_feat_all]
                pred_scaled = self.regressor.predict(model_input, verbose=0)
            else:
                feats = tr.extract_features(
                    self.emg,
                    self.extractor,
                    feature_mode=feature_mode,
                    emg_transform=emg_transform,
                    emd_max_imfs=emd_max_imfs,
                )

                want_imu = bool(metrics.get("imu_included", False))
                if self.imu is not None and (want_imu or fallback_use_imu):
                    imu_feat_all = tr._imu_features(self.imu)
                    if imu_feat_all.shape[0] == feats.shape[0]:
                        feats = np.concatenate([feats, imu_feat_all], axis=1)

                n_in = getattr(self.scaler, "n_features_in_", None)
                if n_in is not None and int(feats.shape[1]) != int(n_in):
                    raise ValueError(
                        f"Feature size mismatch: built {feats.shape[1]} features but scaler expects {n_in}."
                    )

                feats = self.scaler.transform(feats)
                pred_scaled = self.regressor.predict(feats)
            if self.target_scaler is not None:
                self.predicted_raw = self.target_scaler.inverse_transform(
                    pred_scaled
                ).astype(np.float32)
            else:
                self.predicted_raw = pred_scaled.astype(np.float32)
            if self.personalization is not None:
                self.predicted_raw = tr.apply_personalization(
                    self.predicted_raw, self.personalization
                ).astype(np.float32)
            self.predicted_smoothed = self._smooth_predictions(self.predicted_raw)
            self.predicted = (
                self.predicted_raw
                if self.show_raw_checkbox.isChecked()
                else self.predicted_smoothed
            )
        except Exception as exc:
            QMessageBox.warning(self, "Prediction failed", f"Prediction failed:\n{exc}")
            self.status_label.setText("Status: prediction failed")
            self._refresh_compatibility_indicator()
            self.runtime_feature_mode = feature_mode
            self.runtime_emg_transform = emg_transform
            self.runtime_use_imu = bool(
                metrics.get("imu_included", metrics.get("use_imu", False))
            )
            self.runtime_extractor_used = (
                self.model_type != "cnn_attention_regressor"
                and self.extractor is not None
                and feature_mode == "extractor"
            )
            self.runtime_pipeline_note = f"prediction failed: {exc}"
            self._refresh_model_profile()
            return

        self.runtime_feature_mode = feature_mode
        self.runtime_emg_transform = emg_transform
        self.runtime_use_imu = bool(
            metrics.get("imu_included", metrics.get("use_imu", False))
            or fallback_use_imu
        )
        self.runtime_extractor_used = (
            self.model_type != "cnn_attention_regressor"
            and self.extractor is not None
            and feature_mode == "extractor"
        )

        if not self.pred_keys:
            if self.predicted.shape[1] == len(self.measured_keys):
                self.pred_keys = list(self.measured_keys)
            else:
                self.pred_keys = list(ANGLE_KEYS[: self.predicted.shape[1]])

        self.common_keys = [k for k in self.pred_keys if k in self.measured_keys]
        if not self.common_keys:
            self.common_keys = list(
                self.measured_keys[
                    : min(self.measured.shape[1], self.predicted.shape[1])
                ]
            )

        self.status_label.setText(
            f"Status: predicted {self.predicted.shape[0]} windows ({self.predicted.shape[1]} outputs), common joints={len(self.common_keys)}, smoothing={self.smooth_method_combo.currentText()}"
        )
        self._refresh_compatibility_indicator()
        self._refresh_model_profile()
        self._render_current()

    def _angles_row_to_full14(self, row, keys):
        out = np.zeros(len(ANGLE_KEYS), dtype=np.float32)
        key_to_idx = {k: i for i, k in enumerate(ANGLE_KEYS)}
        for i, k in enumerate(keys):
            if k in key_to_idx and i < len(row):
                out[key_to_idx[k]] = float(row[i])
        return out

    def _render_current(self):
        if self.measured is None:
            return
        idx = int(self.slider.value())
        idx = max(0, min(idx, self.measured.shape[0] - 1))

        meas_row = self.measured[idx]
        meas_full = self._angles_row_to_full14(meas_row, self.measured_keys)
        meas_img = _render_hand_iso(meas_full, title="Measured")

        if self.predicted is not None and idx < self.predicted.shape[0]:
            pred_row = self.predicted[idx]
            pred_title = (
                "Predicted (Raw)"
                if self.show_raw_checkbox.isChecked()
                else "Predicted (Smoothed)"
            )
            pred_full = self._angles_row_to_full14(pred_row, self.pred_keys)
            pred_img = _render_hand_iso(pred_full, title=pred_title)

            ck = self.common_keys
            if ck:
                m_idx = [
                    self.measured_keys.index(k) for k in ck if k in self.measured_keys
                ]
                p_idx = [self.pred_keys.index(k) for k in ck if k in self.pred_keys]
                if m_idx and p_idx:
                    d = np.abs(meas_row[m_idx] - pred_row[p_idx[: len(m_idx)]])
                    mae = float(np.mean(d))
                    self.error_label.setText(
                        f"Error: MAE={mae:.2f} over {len(m_idx)} joints ({'raw' if self.show_raw_checkbox.isChecked() else 'smoothed'})"
                    )
        else:
            pred_img = _render_hand_iso(
                np.zeros(14, dtype=np.float32), title="Predicted (run model)"
            )
            self.error_label.setText("Error: N/A")

        self._set_label_image(self.measured_label, meas_img)
        self._set_label_image(self.pred_label, pred_img)

        if self.timestamps is not None and idx < len(self.timestamps):
            self.frame_label.setText(f"Frame: {idx}  t={self.timestamps[idx]:.3f}")
        else:
            self.frame_label.setText(f"Frame: {idx}")

    def _set_label_image(self, label: QLabel, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current()

    def _toggle_play(self):
        if self.measured is None:
            return
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(50)
        else:
            self.timer.stop()

    def _advance_frame(self):
        if self.measured is None:
            self.timer.stop()
            return
        v = self.slider.value() + 1
        if v > self.slider.maximum():
            v = 0
        self.slider.setValue(v)


def main():
    app = QApplication(sys.argv)
    w = NPZAngleCompareGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
