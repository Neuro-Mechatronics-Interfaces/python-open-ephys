#!/usr/bin/env python3
"""
3cv2_predict_gui_zmq.py
-----------------------
PyQt5 GUI for EMG gesture prediction using ONLY the ZMQClient (Open Ephys GUI stream).

Why this script?
- No file playback and no offline OEBin reading.
- Connects to the Open Ephys GUI's ZeroMQ Data/Heartbeat ports.
- Locks preprocessing/window/step and TRAINING CHANNEL ORDER from training metadata.
- Uses ZMQClient's step pump to get aligned, training-ordered blocks in realtime.

UI bits
- Connection: ZMQ IP, Data port, Heartbeat port
- Model: root_dir + label (same convention as 2_train_model.py)
- Params: window_ms, step_ms (auto-filled from metadata)
- Live: current label, optional proba (if your ModelManager exposes it), and a rolling timeline
- Optional: CSV logging

Requires
- PyQt5, pyqtgraph, numpy, torch, and your `pyoephys` package in PYTHONPATH.
"""

import os
import sys
import json
import time
import argparse
import traceback
from typing import List, Optional, Tuple, Any

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# pyoephys imports
from pyoephys.io import (
    load_config_file, load_metadata_json, lock_params_to_meta
)
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier
from pyoephys.interface import ZMQClient


# --------------- Worker Thread (ZMQ streaming only) ---------------
class ZMQPredictWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)           # emitted samples seen
    prediction = QtCore.pyqtSignal(str, object) # label, proba(dict or None)
    finished_ok = QtCore.pyqtSignal(object)     # run info dict
    failed = QtCore.pyqtSignal(str)             # error str

    def __init__(self, *, root_dir: str, label: str,
                 zmq_ip: str, data_port: int, heartbeat_port: int,
                 window_ms: int | None, step_ms: int | None,
                 required_fraction: float = 1.0,
                 verbose: bool = False, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.label = label
        self.zmq_ip = zmq_ip
        self.data_port = int(data_port)
        self.heartbeat_port = int(heartbeat_port)
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.required_fraction = float(required_fraction)
        self.verbose = verbose
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        client = None
        try:
            # --- Metadata & params ---
            meta = load_metadata_json(self.root_dir, label=self.label)
            # lock params to training (also returns envelope cutoff)
            w_ms, s_ms, _, env_cut = lock_params_to_meta(meta, self.window_ms, self.step_ms, selected_channels=None)
            trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
            if not trained_names:
                raise RuntimeError("Training metadata missing 'channel_names' (training order).")

            # --- ZMQ client ---
            client = ZMQClient(
                zqm_ip=f"tcp://{self.zmq_ip}",
                http_ip="127.0.0.1",
                data_port=self.data_port,
                heartbeat_port=self.heartbeat_port,
                window_secs=max(1.0, w_ms / 1000.0 * 3),  # keep a few windows buffered
                channels=None,  # we'll set via resolve_training_order
                auto_start=True,
                verbose=self.verbose,
            )
            # Require the training channels (any order), block until ready
            client.set_required(channel_names=list(trained_names), enforce_complete=True)
            client.wait_until_ready(timeout_sec=30.0, poll_sec=0.05)

            fs = float(client.fs)
            W = int(round(w_ms / 1000.0 * fs))
            S = int(round(s_ms / 1000.0 * fs))

            # Start "step pump" in training order (gets aligned new blocks, reordered to model order)
            client.start_step_pump(
                step_samples=S,
                trained_channel_names=list(trained_names),
                required_fraction=self.required_fraction,
                timeout_sec=30.0,
                require_complete=True,
            )

            # Preprocessor (stateful) and ring buffer (preprocessed space)
            pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=self.verbose)
            C = len(trained_names)
            ring = np.zeros((C, W), dtype=np.float32)

            # Model
            manager = ModelManager(root_dir=self.root_dir, label=self.label, model_cls=EMGClassifier, config={"verbose": self.verbose})
            manager.load_model()
            # Determine feature dim expectation
            # NOTE: manager.scaler is used inside manager.predict; not queried here.
            label_classes = meta.get("label_classes", [])
            have_proba = hasattr(manager, "predict_proba")

            seen_samples = 0
            while not self._stop:
                item = client.get_step(timeout=0.5)  # (abs_idx_end, step_ord) or None
                if item is None:
                    continue
                abs_idx, step_ord = item
                if step_ord is None or step_ord.size == 0:
                    continue

                # Preprocess this step and update ring
                y_step = pre.preprocess(step_ord)
                ring = np.concatenate([ring[:, S:], y_step], axis=1)

                # One-window feature vector
                feats = pre.extract_emg_features(ring, window_ms=w_ms, step_ms=w_ms, return_windows=False)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)

                # Predict
                try:
                    pred = manager.predict(feats)[0]
                except Exception as e:
                    # Fallback: if scaler mismatch is the cause, surface full trace
                    raise

                # Try to unify to string label
                label_str = None
                if isinstance(pred, (str, bytes)):
                    label_str = pred.decode() if isinstance(pred, bytes) else pred
                elif isinstance(pred, (int, np.integer)) and isinstance(label_classes, (list, tuple)) and len(label_classes) > int(pred):
                    try:
                        label_str = str(label_classes[int(pred)])
                    except Exception:
                        label_str = str(pred)
                else:
                    label_str = str(pred)

                proba_dict = None
                if have_proba:
                    try:
                        p = manager.predict_proba(feats)[0]
                        classes = getattr(manager, "classes_", None)
                        if classes is None and hasattr(manager, "label_encoder"):
                            classes = manager.label_encoder.classes_
                        if classes is None and label_classes:
                            classes = label_classes
                        if classes is not None:
                            proba_dict = {str(c): float(p[j]) for j, c in enumerate(classes)}
                    except Exception:
                        proba_dict = None

                self.prediction.emit(label_str, proba_dict)

                seen_samples = int(abs_idx)
                self.progress.emit(seen_samples)

            # Cleanup & done
            self.finished_ok.emit({
                "fs": fs,
                "window_ms": w_ms,
                "step_ms": s_ms,
                "seen_samples": seen_samples,
            })

        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")
        finally:
            try:
                if client is not None:
                    client.stop_step_pump()
            except Exception:
                pass
            try:
                if client is not None:
                    client.stop()
            except Exception:
                pass


# --------------- GUI ---------------
class PredictGUI(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("pyoephys • ZMQ EMG Prediction (3cv2)")
        self.resize(1150, 720)

        self.args = args

        self.current_worker = None
        self.proba_keys: list[str] = []
        self.pred_history: list[int] = []
        self.max_history = 600

        self._build_ui()

        # Defaults
        if args.root_dir: self.le_root.setText(args.root_dir)
        if args.label:    self.le_label.setText(args.label)
        if args.zmq_ip:   self.le_ip.setText(args.zmq_ip)
        if args.data_port is not None: self.sb_data.setValue(int(args.data_port))
        if args.hb_port is not None:   self.sb_hb.setValue(int(args.hb_port))
        if args.window_ms is not None: self.sb_window.setValue(int(args.window_ms))
        if args.step_ms is not None:   self.sb_step.setValue(int(args.step_ms))

        # Try to pull metadata-derived params/classes
        self.btn_reload_meta.click()

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        # Left panel
        left = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(left)

        # Root/Label
        self.le_root = QtWidgets.QLineEdit(left)
        self.btn_browse = QtWidgets.QPushButton("Browse…", left)
        hb0 = QtWidgets.QHBoxLayout(); hb0.addWidget(self.le_root, 1); hb0.addWidget(self.btn_browse, 0)
        form.addRow("Root dir:", hb0)
        self.le_label = QtWidgets.QLineEdit(left)
        form.addRow("Label:", self.le_label)

        # ZMQ settings
        self.le_ip = QtWidgets.QLineEdit(left); self.le_ip.setText("127.0.0.1")
        self.sb_data = QtWidgets.QSpinBox(left); self.sb_data.setRange(1, 65535); self.sb_data.setValue(5556)
        self.sb_hb   = QtWidgets.QSpinBox(left); self.sb_hb.setRange(1, 65535); self.sb_hb.setValue(5557)
        hb1 = QtWidgets.QHBoxLayout(); hb1.addWidget(self.le_ip); hb1.addWidget(self.sb_data); hb1.addWidget(self.sb_hb)
        form.addRow("ZMQ ip | data | hb:", hb1)

        # Window/step
        self.sb_window = QtWidgets.QSpinBox(left); self.sb_window.setRange(20, 2000); self.sb_window.setSingleStep(10); self.sb_window.setValue(200)
        self.sb_step   = QtWidgets.QSpinBox(left); self.sb_step.setRange(10, 1000); self.sb_step.setSingleStep(10); self.sb_step.setValue(50)
        hb2 = QtWidgets.QHBoxLayout(); hb2.addWidget(self.sb_window); hb2.addWidget(self.sb_step)
        form.addRow("window_ms / step_ms:", hb2)

        # Options
        self.cb_log  = QtWidgets.QCheckBox("Log predictions to CSV")
        self.le_log  = QtWidgets.QLineEdit(left); self.le_log.setPlaceholderText("predictions.csv")
        hb3 = QtWidgets.QHBoxLayout(); hb3.addWidget(self.cb_log); hb3.addWidget(self.le_log)
        form.addRow("CSV logging:", hb3)

        # Buttons
        self.btn_reload_meta = QtWidgets.QPushButton("Reload metadata", left)
        self.btn_start = QtWidgets.QPushButton("Start", left)
        self.btn_stop  = QtWidgets.QPushButton("Stop", left)
        hb4 = QtWidgets.QHBoxLayout(); hb4.addWidget(self.btn_reload_meta); hb4.addStretch(1); hb4.addWidget(self.btn_start); hb4.addWidget(self.btn_stop)
        form.addRow("", hb4)

        # Status
        self.lbl_status = QtWidgets.QLabel("Ready.")
        form.addRow("Status:", self.lbl_status)

        # Right panel (live view)
        right = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(right)

        self.lbl_pred = QtWidgets.QLabel("—")
        f = self.lbl_pred.font(); f.setPointSize(28); f.setBold(True); self.lbl_pred.setFont(f)
        self.lbl_pred.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(self.lbl_pred)

        # Probability table
        self.tbl_proba = QtWidgets.QTableWidget(0, 2, right)
        self.tbl_proba.setHorizontalHeaderLabels(["Class", "p"])
        self.tbl_proba.horizontalHeader().setStretchLastSection(True)
        self.tbl_proba.verticalHeader().setVisible(False)
        self.tbl_proba.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_proba.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(self.tbl_proba, 1)

        # Timeline
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", "Prediction index")
        self.plot.setLabel("left", "Class ID")
        self.curve = self.plot.plot([], [], pen=None, symbol='o', symbolSize=6)
        vbox.addWidget(self.plot, 2)

        # Split
        splitter = QtWidgets.QSplitter(self)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)

        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(splitter)

        # Connect
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_reload_meta.clicked.connect(self._on_reload_meta)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

    # --- Slots ---
    def _on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select root directory", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def _on_reload_meta(self):
        root = self.le_root.text().strip()
        label = self.le_label.text().strip()
        if not root:
            QtWidgets.QMessageBox.warning(self, "Missing root_dir", "Please set Root dir.")
            return
        try:
            meta = load_metadata_json(root, label=label)
            w, s, _, _ = lock_params_to_meta(meta, None, None, selected_channels=None)
            self.sb_window.setValue(int(w)); self.sb_step.setValue(int(s))

            classes = meta.get("label_classes", [])
            if isinstance(classes, list) and len(classes) > 0:
                self.proba_keys = [str(c) for c in classes]
                self._setup_proba_table(self.proba_keys)

            self._set_status("Metadata loaded.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Metadata error", str(e))
            self._set_status("Failed to load metadata.")

    def _setup_proba_table(self, keys: List[str]):
        self.tbl_proba.setRowCount(len(keys))
        for i, k in enumerate(keys):
            self.tbl_proba.setItem(i, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.tbl_proba.setItem(i, 1, QtWidgets.QTableWidgetItem("—"))

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def _on_start(self):
        if self.current_worker is not None:
            QtWidgets.QMessageBox.information(self, "Already running", "A run is already in progress.")
            return

        root = self.le_root.text().strip()
        label = self.le_label.text().strip()
        ip = self.le_ip.text().strip() or "127.0.0.1"
        data_port = int(self.sb_data.value())
        hb_port = int(self.sb_hb.value())
        if not root:
            QtWidgets.QMessageBox.warning(self, "Missing root_dir", "Please set Root dir.")
            return

        w_ms = int(self.sb_window.value())
        s_ms = int(self.sb_step.value())

        # reset plot
        self.pred_history = []
        self._update_plot()

        w = ZMQPredictWorker(
            root_dir=root, label=label,
            zmq_ip=ip, data_port=data_port, heartbeat_port=hb_port,
            window_ms=w_ms, step_ms=s_ms,
            required_fraction=1.0, verbose=False
        )
        w.prediction.connect(self._on_prediction)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_finished_ok)
        w.failed.connect(self._on_failed)

        self.current_worker = w
        self._set_status("Connecting…")
        w.start()

    def _on_stop(self):
        if self.current_worker is not None:
            self.current_worker.stop()

    @QtCore.pyqtSlot(str, object)
    def _on_prediction(self, label: str, proba: Optional[dict]):
        self.lbl_pred.setText(str(label))

        # Timeline index mapping
        if self.proba_keys and label in self.proba_keys:
            val = self.proba_keys.index(label)
        else:
            val = abs(hash(label)) % 12
        self.pred_history.append(val)
        if len(self.pred_history) > self.max_history:
            self.pred_history = self.pred_history[-self.max_history:]
        self._update_plot()

        # Proba table (if provided)
        if isinstance(proba, dict) and len(proba) > 0:
            if not self.proba_keys:
                self.proba_keys = list(proba.keys())
                self._setup_proba_table(self.proba_keys)
            for i, k in enumerate(self.proba_keys):
                v = proba.get(k, None)
                self.tbl_proba.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{v:.3f}" if v is not None else "—"))

        # Optional CSV logging
        if self.cb_log.isChecked():
            path = self.le_log.text().strip() or "predictions.csv"
            try:
                with open(path, "a", encoding="utf-8") as f:
                    if proba is not None:
                        f.write(f"{time.time():.6f},{label},{json.dumps(proba)}\n")
                    else:
                        f.write(f"{time.time():.6f},{label},\n")
            except Exception as e:
                self._set_status(f"CSV write failed once: {e}")

    def _update_plot(self):
        x = np.arange(len(self.pred_history))
        y = np.array(self.pred_history, dtype=float)
        self.curve.setData(x, y)
        if len(y) > 0:
            self.plot.setYRange(np.nanmin(y) - 0.5, np.nanmax(y) + 0.5, padding=0)

    @QtCore.pyqtSlot(int)
    def _on_progress(self, seen: int):
        self._set_status(f"Seen samples: {seen}")

    @QtCore.pyqtSlot(object)
    def _on_finished_ok(self, info: dict):
        self._set_status("Finished.")
        self.current_worker = None

    @QtCore.pyqtSlot(str)
    def _on_failed(self, msg: str):
        self._set_status("Error.")
        QtWidgets.QMessageBox.critical(self, "Run failed", msg)
        self.current_worker = None


# --------------- CLI entry ---------------
def parse_args(argv=None):
    p = argparse.ArgumentParser("3cv2 (ZMQ-only): EMG prediction GUI from Open Ephys GUI stream")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--root_dir",    type=str, default="")
    p.add_argument("--label",       type=str, default="")

    p.add_argument("--zmq_ip",      type=str, default="127.0.0.1")
    p.add_argument("--data_port",   type=int, default=5556)
    p.add_argument("--hb_port",     type=int, default=5557)

    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Optional config file (CLI wins)
    if args.config_file:
        try:
            cfg = load_config_file(args.config_file) or {}
        except Exception:
            cfg = {}
        for k in ["root_dir", "label", "zmq_ip", "data_port", "hb_port", "window_ms", "step_ms"]:
            v = getattr(args, k, None) or cfg.get(k, None)
            setattr(args, k, v)

    app = QtWidgets.QApplication(sys.argv)
    gui = PredictGUI(args)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
