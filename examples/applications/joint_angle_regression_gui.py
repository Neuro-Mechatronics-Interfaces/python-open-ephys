import os
import sys
import time
import json
import pickle
from pathlib import Path

# Ensure local repo src is preferred over any installed pyoephys
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QTextEdit,
    QToolButton,
    QFrame,
    QMessageBox,
)

from pyoephys.interface import ZMQClient, NotReadyError
from pyoephys.interface._lsl_client import OldLSLClient
from pyoephys.processing import RealtimeFilter, ChannelQC


# ---- target specs ----
ANGLE_KEYS = [
    "thumb_cmc_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]
TARGET_SPEC = "finger5"
TARGET_KEYS = ["thumb_cmc_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]


def _friendly_joint(name):
    mapping = {
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
    return mapping.get(name, name.replace("_", " "))


try:
    from pylsl import local_clock, StreamInfo, StreamOutlet
    HAS_LSL = True
except Exception:
    local_clock = None
    StreamInfo = None
    StreamOutlet = None
    HAS_LSL = False


def _clock():
    return local_clock() if local_clock is not None else time.time()


class CollapsibleSection(QFrame):
    def __init__(self, title, content_widget, parent=None, default_open=True, header_widget=None):
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


class JointAngleRegressionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open Ephys Joint Angle Regression")
        self.resize(1100, 820)

        # ZMQ streaming
        self.client = None
        self.connected = False
        self.selected_channels = []
        self.channel_names = []
        self.qc = None
        self.qc_excluded = set()
        self.filter = None
        self.emg_offset = None

        # LSL angles
        self.angle_client = None
        self.angle_buffer = []
        self.angle_count = 0
        self.last_angle_seen = 0.0

        # Prompts / markers
        self.prompt_items = []
        self.prompt_idx = 0
        self.prompt_end_time = 0.0
        self.prompt_active = False
        self.marker_outlet = None
        self.marker_buffer = []

        # Recording state
        self.recording_active = False
        self.record_start_time = 0.0
        self.record_duration = 300
        self.record_window_len = 100
        self.record_overlap = 50
        self.record_hand_idx = 0
        self.record_out_path = ""
        self.record_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.record_ts_buf = np.zeros((0,), dtype=np.float64)
        self.record_emg_windows = []
        self.record_angle_targets = []
        self.record_window_ts = []
        self.record_marker_labels = []
        self.record_skip_no_angle = 0
        self.record_skip_old_angle = 0
        self.record_skip_nan_angle = 0

        # Compare state
        self.compare_active = False
        self.compare_regressor = None
        self.compare_scaler = None
        self.compare_emg_buf = np.zeros((0, 8), dtype=np.float32)
        self.compare_ts_buf = np.zeros((0,), dtype=np.float64)
        self.compare_window_len = 100
        self.compare_overlap = 50
        self.compare_max_age = 0.2

        self._build_ui()

        self._apply_styles()

        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_all)
        self.timer.start(33)

        self.prompt_timer = QTimer()
        self.prompt_timer.timeout.connect(self._prompt_tick)

    # ---- UI ----
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top_row = QHBoxLayout()
        top_row.addWidget(self._build_source_panel(), 1)
        top_row.addWidget(self._build_lsl_panel(), 1)
        top_row.addWidget(self._build_live_panel(), 1)
        layout.addLayout(top_row)

        layout.addWidget(self._build_log_panel())

        mid_row = QHBoxLayout()
        mid_row.addWidget(self._build_prompt_panel(), 1)
        mid_row.addWidget(self._build_record_panel(), 1)
        layout.addLayout(mid_row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._build_train_panel(), 1)
        bottom_row.addWidget(self._build_compare_panel(), 1)
        layout.addLayout(bottom_row)

    def _apply_styles(self):
        self.setStyleSheet("""
        QWidget {
            font-size: 11px;
            background: #1c1f24;
            color: #e6e8eb;
        }
        QGroupBox {
            border: 1px solid #2a2f36;
            border-radius: 10px;
            margin-top: 6px;
            padding: 8px;
            background: #21262d;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
        }
        QToolButton {
            border: none;
            font-weight: bold;
            color: #cfd5dd;
        }
        QPushButton {
            background-color: #3a6ea5;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 4px 8px;
        }
        QPushButton:checked {
            background-color: #2b567f;
        }
        QPushButton:hover {
            background-color: #4b7fb3;
        }
        QLineEdit, QSpinBox, QComboBox, QTextEdit, QDoubleSpinBox {
            border: 1px solid #2f353d;
            border-radius: 6px;
            padding: 3px;
            background: #1a1e24;
            color: #e6e8eb;
            selection-background-color: #3a6ea5;
        }
        QLabel {
            color: #cfd5dd;
        }
        QScrollArea {
            border: none;
        }
        QCheckBox {
            color: #cfd5dd;
        }
        """)

    def _build_source_panel(self):
        group = QGroupBox("Open Ephys (ZMQ)")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.zmq_host = QLineEdit("127.0.0.1")
        self.zmq_port = QLineEdit("5556")
        row.addWidget(QLabel("Host"))
        row.addWidget(self.zmq_host)
        row.addWidget(QLabel("Port"))
        row.addWidget(self.zmq_port)
        layout.addLayout(row)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connection)
        layout.addWidget(self.btn_connect)

        self.status_label = QLabel("Status: Disconnected")
        layout.addWidget(self.status_label)

        self.fs_label = QLabel("Fs: N/A")
        self.ch_label = QLabel("Channels: 0")
        self.qc_label = QLabel("QC: N/A")
        layout.addWidget(self.fs_label)
        layout.addWidget(self.ch_label)
        layout.addWidget(self.qc_label)

        # Filter controls
        layout.addWidget(QLabel("Filters (RealtimeFilter)"))
        frow = QHBoxLayout()
        self.bp_low = QDoubleSpinBox()
        self.bp_low.setRange(1, 5000)
        self.bp_low.setValue(20.0)
        self.bp_high = QDoubleSpinBox()
        self.bp_high.setRange(1, 5000)
        self.bp_high.setValue(450.0)
        frow.addWidget(QLabel("BP low"))
        frow.addWidget(self.bp_low)
        frow.addWidget(QLabel("BP high"))
        frow.addWidget(self.bp_high)
        layout.addLayout(frow)

        nrow = QHBoxLayout()
        self.notch = QSpinBox()
        self.notch.setRange(0, 120)
        self.notch.setValue(60)
        nrow.addWidget(QLabel("Notch Hz (0=off)"))
        nrow.addWidget(self.notch)
        layout.addLayout(nrow)

        btn_apply = QPushButton("Apply Filters")
        btn_apply.clicked.connect(self._rebuild_filter)
        layout.addWidget(btn_apply)

        return group

    def _build_lsl_panel(self):
        group = QGroupBox("LSL Angles")
        layout = QVBoxLayout(group)
        self.angle_stream = QLineEdit("StereoHandTracker_Angles")
        layout.addWidget(QLabel("Angles stream"))
        layout.addWidget(self.angle_stream)
        btn = QPushButton("Connect LSL")
        btn.clicked.connect(self._connect_lsl)
        layout.addWidget(btn)
        self.angle_status = QLabel("Angles: N/A")
        self.angle_age = QLabel("Age: N/A")
        layout.addWidget(self.angle_status)
        layout.addWidget(self.angle_age)
        return group

    def _build_live_panel(self):
        group = QGroupBox("Live Metrics")
        layout = QVBoxLayout(group)
        self.emg_rms_label = QLabel("EMG RMS: N/A")
        self.angle_match_label = QLabel("Angle match: N/A")
        self.compare_label = QLabel("Compare: N/A")
        self.compare_bar_label = QLabel("")
        self.compare_bar_label.setStyleSheet("font-family: Consolas, monospace;")
        self.compare_bar_label.setWordWrap(True)
        layout.addWidget(self.emg_rms_label)
        layout.addWidget(self.angle_match_label)
        layout.addWidget(self.compare_label)
        layout.addWidget(self.compare_bar_label)
        return group

    def _build_record_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        default_out = Path(__file__).parent / "data" / "session_001.npz"
        self.record_out = QLineEdit(str(default_out))
        layout.addWidget(QLabel("Output"))
        layout.addWidget(self.record_out)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 3600)
        self.duration_spin.setValue(300)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(20, 500)
        self.window_spin.setValue(100)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 490)
        self.overlap_spin.setValue(50)
        layout.addWidget(QLabel("Duration (s)"))
        layout.addWidget(self.duration_spin)
        layout.addWidget(QLabel("Window (samples)"))
        layout.addWidget(self.window_spin)
        layout.addWidget(QLabel("Overlap (samples)"))
        layout.addWidget(self.overlap_spin)

        header_actions = QWidget()
        row = QHBoxLayout(header_actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_rec_start = QPushButton("Start")
        self.btn_rec_stop = QPushButton("Stop")
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_start.clicked.connect(self._start_record)
        self.btn_rec_stop.clicked.connect(self._stop_record)
        row.addWidget(self.btn_rec_start)
        row.addWidget(self.btn_rec_stop)
        section = CollapsibleSection("Recording", content, header_widget=header_actions)
        return section

    def _build_train_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.train_data = QLineEdit(str(Path(__file__).parent / "data" / "session_001.npz"))
        self.train_out = QLineEdit(str(Path(__file__).parent / "models" / "joint_regressor"))
        layout.addWidget(QLabel("Dataset (.npz)"))
        layout.addWidget(self.train_data)
        layout.addWidget(QLabel("Output dir"))
        layout.addWidget(self.train_out)
        header_actions = QWidget()
        row = QHBoxLayout(header_actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self._train_regressor)
        row.addWidget(self.btn_train)
        self.btn_train_settings = QPushButton("Settings")
        self.btn_train_settings.clicked.connect(self._show_train_settings)
        row.addWidget(self.btn_train_settings)
        section = CollapsibleSection("Training", content, default_open=False, header_widget=header_actions)
        return section

    def _build_compare_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.compare_reg = QLineEdit(str(Path(__file__).parent / "models" / "joint_regressor" / "mlp_regressor.pkl"))
        self.compare_scaler = QLineEdit(str(Path(__file__).parent / "models" / "joint_regressor" / "scaler.pkl"))
        layout.addWidget(QLabel("Regressor"))
        layout.addWidget(self.compare_reg)
        layout.addWidget(QLabel("Scaler"))
        layout.addWidget(self.compare_scaler)
        header_actions = QWidget()
        row = QHBoxLayout(header_actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_cmp_start = QPushButton("Start Compare")
        self.btn_cmp_stop = QPushButton("Stop Compare")
        self.btn_cmp_stop.setEnabled(False)
        self.btn_cmp_start.clicked.connect(self._start_compare)
        self.btn_cmp_stop.clicked.connect(self._stop_compare)
        row.addWidget(self.btn_cmp_start)
        row.addWidget(self.btn_cmp_stop)
        section = CollapsibleSection("Live Compare", content, default_open=False, header_widget=header_actions)
        return section

    def _build_log_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        section = CollapsibleSection("Console Log", content, default_open=True)
        return section

    def _build_prompt_panel(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.prompt_plan = QLineEdit(str(Path(__file__).parent / "prompts_default.json"))
        plan_row = QHBoxLayout()
        plan_row.addWidget(self.prompt_plan)
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._load_prompt_preview)
        plan_row.addWidget(btn_load)
        layout.addWidget(QLabel("Plan"))
        layout.addLayout(plan_row)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setPlaceholderText("Prompt plan preview...")
        layout.addWidget(QLabel("Preview"))
        layout.addWidget(self.prompt_preview)

        header_actions = QWidget()
        row = QHBoxLayout(header_actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.task_start_btn = QPushButton("Start Task")
        self.task_stop_btn = QPushButton("Stop Task")
        self.task_stop_btn.setEnabled(False)
        self.task_start_btn.clicked.connect(self._start_task)
        self.task_stop_btn.clicked.connect(self._stop_task)
        self.btn_prompt_start = QPushButton("Start Prompts")
        self.btn_prompt_stop = QPushButton("Stop Prompts")
        self.btn_prompt_stop.setEnabled(False)
        self.btn_prompt_start.clicked.connect(self._start_prompts)
        self.btn_prompt_stop.clicked.connect(self._stop_prompts)
        row.addWidget(self.task_start_btn)
        row.addWidget(self.task_stop_btn)
        row.addWidget(self.btn_prompt_start)
        row.addWidget(self.btn_prompt_stop)

        section = CollapsibleSection("Task Prompter", content, header_widget=header_actions)
        return section

    # ---- actions ----
    def _toggle_connection(self):
        if not self.connected:
            try:
                self.client = ZMQClient(
                    host_ip=self.zmq_host.text().strip(),
                    data_port=self.zmq_port.text().strip(),
                    auto_start=True,
                )
                self.connected = True
                self.btn_connect.setText("Disconnect")
                self.status_label.setText("Status: Connecting...")
                self._append_log("[oephys] ZMQ client started")
            except Exception as exc:
                self._append_log(f"[oephys] Connect error: {exc}")
        else:
            if self.client:
                self.client.stop()
            self.client = None
            self.connected = False
            self.btn_connect.setText("Connect")
            self.status_label.setText("Status: Disconnected")

    def _connect_lsl(self):
        try:
            if self.angle_client:
                self.angle_client.stop()
            self.angle_client = OldLSLClient(stream_name=self.angle_stream.text().strip(), auto_start=True)
            self._append_log("[lsl] Angles connected")
        except Exception as exc:
            self._append_log(f"[lsl] Error: {exc}")

    def _rebuild_filter(self):
        if not self.client:
            return
        if not self.selected_channels:
            return
        notch = float(self.notch.value())
        notch_freqs = (notch,) if notch > 0 else ()
        self.filter = RealtimeFilter(
            fs=self.client.fs,
            n_channels=len(self.selected_channels),
            bp_low=float(self.bp_low.value()),
            bp_high=float(self.bp_high.value()),
            notch_freqs=notch_freqs,
            enable_notch=notch > 0,
            enable_bandpass=True,
        )
        self._append_log("[filters] Rebuilt RealtimeFilter")

    def _start_record(self):
        if not self.connected or not self.client:
            self._append_log("[record] Connect to Open Ephys first.")
            return
        if not self.angle_client:
            self._append_log("[record] Connect LSL angles first.")
            return
        self.record_out_path = self.record_out.text().strip()
        self.record_duration = int(self.duration_spin.value())
        self.record_window_len = int(self.window_spin.value())
        self.record_overlap = int(self.overlap_spin.value())
        self.record_emg_buf = np.zeros((0, len(self.selected_channels)), dtype=np.float32)
        self.record_ts_buf = np.zeros((0,), dtype=np.float64)
        self.record_emg_windows = []
        self.record_angle_targets = []
        self.record_window_ts = []
        self.record_skip_no_angle = 0
        self.record_skip_old_angle = 0
        self.record_skip_nan_angle = 0
        self.record_start_time = _clock()
        self.recording_active = True
        self.btn_rec_start.setEnabled(False)
        self.btn_rec_stop.setEnabled(True)
        self._append_log(f"[record] Started: {self.record_out_path}")

    def _stop_record(self):
        self._finish_record(save=True)

    def _start_compare(self):
        if not self.connected:
            self._append_log("[compare] Connect to Open Ephys first.")
            return
        reg_path = self.compare_reg.text().strip()
        scaler_path = self.compare_scaler.text().strip()
        if not os.path.exists(reg_path) or not os.path.exists(scaler_path):
            self._append_log("[compare] Missing regressor or scaler.")
            return
        try:
            with open(reg_path, "rb") as f:
                self.compare_regressor = pickle.load(f)
            with open(scaler_path, "rb") as f:
                self.compare_scaler = pickle.load(f)
        except Exception as exc:
            self._append_log(f"[compare] Load error: {exc}")
            return
        self.compare_window_len = int(self.window_spin.value())
        self.compare_overlap = int(self.overlap_spin.value())
        self.compare_emg_buf = np.zeros((0, len(self.selected_channels)), dtype=np.float32)
        self.compare_ts_buf = np.zeros((0,), dtype=np.float64)
        self.compare_active = True
        self.btn_cmp_start.setEnabled(False)
        self.btn_cmp_stop.setEnabled(True)
        self._append_log("[compare] Started")

    def _stop_compare(self):
        self.compare_active = False
        self.compare_regressor = None
        self.compare_scaler = None
        self.compare_label.setText("Compare: N/A")
        self.compare_bar_label.setText("")
        self.btn_cmp_start.setEnabled(True)
        self.btn_cmp_stop.setEnabled(False)

    def _train_regressor(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPRegressor
            from sklearn.metrics import mean_absolute_error, r2_score
        except Exception as exc:
            self._append_log(f"[train] scikit-learn not available: {exc}")
            return

        path = self.train_data.text().strip()
        out_dir = self.train_out.text().strip()
        if not path or not os.path.exists(path):
            self._append_log("[train] Dataset not found.")
            return

        data = np.load(path, allow_pickle=True)
        emg = np.asarray(data["emg"], dtype=np.float32)  # (N, C, T, 1)
        angles = np.asarray(data["angles"], dtype=np.float32)
        angle_keys = data["angle_keys"].tolist() if "angle_keys" in data else None
        target_spec = data["target_spec"].item() if "target_spec" in data else None

        if emg.ndim == 4 and emg.shape[-1] == 1:
            emg = emg[..., 0]
        X = emg.reshape(emg.shape[0], -1)
        y = angles
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        X = X[mask]
        y = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred, multioutput="variance_weighted")

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "mlp_regressor.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        meta = {
            "mae": float(mae),
            "r2": float(r2),
            "data": path,
            "angle_keys": angle_keys,
            "target_spec": target_spec,
            "model": "MLP (256-128-64)",
        }
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self._append_log(f"[train] DONE MAE={mae:.4f} R2={r2:.4f}")

    def _show_train_settings(self):
        text = (
            "Advanced training options (planned):\n"
            "- Alternate model structures (MLP/Linear)\n"
            "- Multi-file dataset merge\n"
            "- Custom window/overlap for training\n"
            "- Feature extractor toggle\n\n"
            "Current defaults:\n"
            "- Model: MLP (256-128-64)\n"
            "- Scaler: StandardScaler\n"
            "- Outputs: mlp_regressor.pkl, scaler.pkl, metrics.json"
        )
        QMessageBox.information(self, "Training Settings", text)

    def _load_prompt_preview(self):
        path = self.prompt_plan.text().strip()
        if not path or not os.path.exists(path):
            self.prompt_preview.setPlainText("Plan file not found.")
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            lines = []
            for i, item in enumerate(data, 1):
                label = item.get("label", "unknown")
                dur = item.get("duration", 0)
                lines.append(f"{i:02d}. {label} ({dur}s)")
            self.prompt_items = data
            self.prompt_preview.setPlainText("\n".join(lines))
        except Exception as exc:
            self.prompt_preview.setPlainText(f"Failed to load: {exc}")

    def _start_prompts(self):
        if self.prompt_active:
            return
        if not self.prompt_items:
            self._load_prompt_preview()
        if not self.prompt_items:
            self._append_log("[prompts] No prompt plan loaded.")
            return
        if HAS_LSL and StreamInfo is not None:
            try:
                info = StreamInfo("NML_TaskMarkers", "Markers", 1, 0, "string", "nml_task_markers")
                self.marker_outlet = StreamOutlet(info)
            except Exception as exc:
                self._append_log(f"[prompts] Marker outlet error: {exc}")
        self.prompt_active = True
        self.prompt_idx = 0
        self._append_log("[prompts] Started")
        self._emit_prompt()
        self.prompt_timer.start(50)
        self.btn_prompt_start.setEnabled(False)
        self.btn_prompt_stop.setEnabled(True)

    def _stop_prompts(self):
        self.prompt_active = False
        self.prompt_timer.stop()
        self.btn_prompt_start.setEnabled(True)
        self.btn_prompt_stop.setEnabled(False)
        self._append_log("[prompts] Stopped")

    def _start_task(self):
        self._start_prompts()
        self._start_record()
        self.task_start_btn.setEnabled(False)
        self.task_stop_btn.setEnabled(True)

    def _stop_task(self):
        self._stop_prompts()
        if self.recording_active:
            self._stop_record()
        self.task_start_btn.setEnabled(True)
        self.task_stop_btn.setEnabled(False)

    def _emit_prompt(self):
        if not self.prompt_items or self.prompt_idx >= len(self.prompt_items):
            self._append_log("[prompts] Complete")
            self._stop_prompts()
            self.task_start_btn.setEnabled(True)
            self.task_stop_btn.setEnabled(False)
            if self.recording_active:
                self._stop_record()
            return
        item = self.prompt_items[self.prompt_idx]
        label = item.get("label", "unknown")
        dur = float(item.get("duration", 0))
        self.prompt_end_time = _clock() + dur
        self._append_log(f"[PROMPT] {label} ({dur:.1f}s)")
        if self.marker_outlet:
            try:
                self.marker_outlet.push_sample([label], _clock())
            except Exception:
                pass
        self.marker_buffer.append((_clock(), label))
        if len(self.marker_buffer) > 2000:
            self.marker_buffer = self.marker_buffer[-2000:]

    def _prompt_tick(self):
        if not self.prompt_active:
            return
        if _clock() >= self.prompt_end_time:
            self.prompt_idx += 1
            self._emit_prompt()

    # ---- polling ----
    def _poll_all(self):
        self._poll_emg()
        self._poll_angles()
        self._record_tick()
        self._compare_tick()
        self._update_match_status()

    def _poll_emg(self):
        if not self.client or not self.connected:
            return
        if not self.client.ready_event.is_set():
            self.status_label.setText("Status: Waiting for stream...")
            return

        self.status_label.setText("Status: Connected")
        self.fs_label.setText(f"Fs: {self.client.fs:.1f} Hz")

        if not self.client.channel_index:
            seen = sorted(self.client.seen_nums) if hasattr(self.client, "seen_nums") else []
            if not seen:
                return
            self.client.set_channel_index(seen)
            self.channel_names = self.client.channel_names
            self.selected_channels = list(seen)
            self.qc = ChannelQC(fs=int(self.client.fs), n_channels=len(seen))
            self._rebuild_filter()

        try:
            t_new, y_new = self.client.drain_new()
        except NotReadyError:
            return

        if t_new is None or y_new is None:
            return

        # QC update on all channels
        if self.qc:
            self.qc.update(y_new.T)
            qc_out = self.qc.evaluate()
            self.qc_excluded = qc_out["excluded"]
            selected = [i for i in range(y_new.shape[0]) if i not in self.qc_excluded]
            if selected:
                self.selected_channels = selected
            self.qc_label.setText(f"QC: used {len(self.selected_channels)} / excluded {len(self.qc_excluded)}")

        # select channels for processing
        sel = self.selected_channels or list(range(y_new.shape[0]))
        emg = y_new[sel, :]

        # update filter if channel count changes
        if self.filter is None or emg.shape[0] != getattr(self.filter, "C", None):
            self._rebuild_filter()

        if self.filter:
            emg = self.filter.process(emg)

        rms = float(np.sqrt(np.mean(emg ** 2))) if emg.size else 0.0
        self.emg_rms_label.setText(f"EMG RMS: {rms:.3f}")

        # time alignment to LSL clock
        t_now = _clock()
        offset = t_now - float(t_new[-1])
        if self.emg_offset is None:
            self.emg_offset = offset
        else:
            self.emg_offset = 0.98 * self.emg_offset + 0.02 * offset
        t_abs = t_new + self.emg_offset

        # stash for record/compare
        self._latest_emg = emg
        self._latest_ts = t_abs

    def _poll_angles(self):
        if not self.angle_client:
            return
        t, y = self.angle_client.drain_new()
        if t is None or y is None:
            return
        for ts, sample in zip(t, y):
            self.angle_buffer.append((float(ts), sample))
            self.angle_count += 1
        if len(self.angle_buffer) > 2000:
            self.angle_buffer = self.angle_buffer[-2000:]
        self.last_angle_seen = _clock()
        age = _clock() - self.last_angle_seen
        self.angle_status.setText(f"Angles: {self.angle_count}")
        self.angle_age.setText(f"Age: {age:.2f}s")

    def _record_tick(self):
        if not self.recording_active:
            return
        if not hasattr(self, "_latest_emg") or self._latest_emg is None:
            return
        emg = self._latest_emg
        ts = self._latest_ts
        if emg is None or ts is None:
            return

        # ensure consistent channel count
        if self.record_emg_buf.shape[1] != emg.shape[0]:
            self.record_emg_buf = np.zeros((0, emg.shape[0]), dtype=np.float32)
            self.record_ts_buf = np.zeros((0,), dtype=np.float64)

        self.record_emg_buf = np.vstack([self.record_emg_buf, emg.T])
        self.record_ts_buf = np.concatenate([self.record_ts_buf, ts])

        stride = self.record_window_len - self.record_overlap
        while self.record_emg_buf.shape[0] >= self.record_window_len:
            window = self.record_emg_buf[:self.record_window_len]
            window_ts = self.record_ts_buf[:self.record_window_len]
            t_center = window_ts[self.record_window_len // 2]
            angle_sample = self._nearest_angle_sample(t_center)
            if angle_sample is not None:
                self.record_emg_windows.append(window.T[..., None])
                self.record_angle_targets.append(angle_sample)
                self.record_window_ts.append(t_center)
                self.record_marker_labels.append(self._nearest_marker_label(t_center))
            self.record_emg_buf = self.record_emg_buf[stride:]
            self.record_ts_buf = self.record_ts_buf[stride:]

        if _clock() - self.record_start_time >= self.record_duration:
            self._finish_record(save=True)

    def _compare_tick(self):
        if not self.compare_active:
            return
        if self.compare_regressor is None or self.compare_scaler is None:
            return
        if not hasattr(self, "_latest_emg") or self._latest_emg is None:
            return
        emg = self._latest_emg
        ts = self._latest_ts
        if emg is None or ts is None:
            return

        if self.compare_emg_buf.shape[1] != emg.shape[0]:
            self.compare_emg_buf = np.zeros((0, emg.shape[0]), dtype=np.float32)
            self.compare_ts_buf = np.zeros((0,), dtype=np.float64)

        self.compare_emg_buf = np.vstack([self.compare_emg_buf, emg.T])
        self.compare_ts_buf = np.concatenate([self.compare_ts_buf, ts])

        stride = self.compare_window_len - self.compare_overlap
        while self.compare_emg_buf.shape[0] >= self.compare_window_len:
            window = self.compare_emg_buf[:self.compare_window_len]
            window_ts = self.compare_ts_buf[:self.compare_window_len]
            t_center = window_ts[self.compare_window_len // 2]
            angle_sample = self._nearest_angle_sample(t_center, max_age=self.compare_max_age)
            if angle_sample is None:
                self.compare_emg_buf = self.compare_emg_buf[stride:]
                self.compare_ts_buf = self.compare_ts_buf[stride:]
                continue
            feats = window.T.reshape(1, -1)
            feats = self.compare_scaler.transform(feats)
            pred = self.compare_regressor.predict(feats)[0]

            diff = np.abs(angle_sample - pred)
            rmse = float(np.sqrt(np.mean((angle_sample - pred) ** 2)))
            mae = float(np.mean(diff))
            self.compare_label.setText(f"Compare: RMSE={rmse:.2f} MAE={mae:.2f}")
            self.compare_bar_label.setText(self._render_bars(diff))

            self.compare_emg_buf = self.compare_emg_buf[stride:]
            self.compare_ts_buf = self.compare_ts_buf[stride:]

    # ---- helpers ----
    def _nearest_angle_sample(self, timestamp, max_age=0.5):
        if not self.angle_buffer:
            self.record_skip_no_angle += 1
            return None
        best = min(self.angle_buffer, key=lambda x: abs(x[0] - timestamp))
        if abs(best[0] - timestamp) > max_age:
            self.record_skip_old_angle += 1
            return None
        sample = np.asarray(best[1], dtype=np.float32).reshape(-1)
        per_hand = len(ANGLE_KEYS)
        start = self.record_hand_idx * per_hand
        end = start + per_hand
        if sample.size < end:
            self.record_skip_no_angle += 1
            return None
        sample = sample[start:end]
        idxs = [ANGLE_KEYS.index(k) for k in TARGET_KEYS if k in ANGLE_KEYS]
        sample = sample[idxs]
        if not np.isfinite(sample).all():
            self.record_skip_nan_angle += 1
            return None
        return sample

    def _update_match_status(self):
        if not hasattr(self, "_latest_ts") or self._latest_ts is None:
            return
        if not self.angle_buffer:
            self.angle_match_label.setText("Angle match: waiting")
            return
        t_ref = self._latest_ts[-1]
        best = min(self.angle_buffer, key=lambda x: abs(x[0] - t_ref))
        delta = abs(best[0] - t_ref)
        status = "OK" if delta <= 0.5 else "MISS"
        self.angle_match_label.setText(f"Angle match: {status} ({delta*1000:.0f} ms)")

    def _render_bars(self, diff):
        if diff is None or diff.size == 0:
            return ""
        max_diff = 90.0
        width = 12
        lines = []
        for key, d in zip(TARGET_KEYS, diff):
            bar_len = int(round((d / max_diff) * width))
            bar = "#" * bar_len
            label = _friendly_joint(key)
            lines.append(f"{label:<11} |{bar}")
        return "\n".join(lines)

    def _finish_record(self, save=True):
        if not self.recording_active:
            return
        self.recording_active = False
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)
        if save and self.record_out_path:
            os.makedirs(os.path.dirname(self.record_out_path), exist_ok=True)
            np.savez_compressed(
                self.record_out_path,
                emg=np.asarray(self.record_emg_windows, dtype=np.float32),
                angles=np.asarray(self.record_angle_targets, dtype=np.float32),
                timestamps=np.asarray(self.record_window_ts, dtype=np.float64),
                markers=np.asarray(self.record_marker_labels, dtype=object),
                angle_keys=np.asarray(TARGET_KEYS, dtype=object),
                target_spec=TARGET_SPEC,
                fs=float(self.client.fs) if self.client else None,
                window_len=self.record_window_len,
                channel_names=np.asarray(self.channel_names, dtype=object),
                selected_channels=np.asarray(self.selected_channels, dtype=np.int32),
            )
            self._append_log(
                f"[record] Saved: {self.record_out_path} (windows={len(self.record_emg_windows)}, "
                f"no_angle={self.record_skip_no_angle}, old_angle={self.record_skip_old_angle}, "
                f"nan_angle={self.record_skip_nan_angle})"
            )

    def _nearest_marker_label(self, timestamp):
        if not self.marker_buffer:
            return ""
        best = min(self.marker_buffer, key=lambda x: abs(x[0] - timestamp))
        if abs(best[0] - timestamp) > 1.0:
            return ""
        return best[1]

    def _append_log(self, text):
        self.log.append(text)

    def closeEvent(self, event):
        if self.recording_active:
            self._finish_record(save=True)
        if self.prompt_active:
            self._stop_prompts()
        if self.client:
            self.client.stop()
        if self.angle_client:
            self.angle_client.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = JointAngleRegressionGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
