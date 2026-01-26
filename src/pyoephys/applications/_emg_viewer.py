"""
pyoephys.applications._emg_viewer

Graphical User Interface (GUI) for loading and visualizing EMG data.
Ported from python-intan for use with python-open-ephys.
"""

import os
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from scipy.signal import spectrogram, butter, filtfilt, iirnotch
except Exception:
    spectrogram = butter = filtfilt = iirnotch = None
try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:
    tk = None
    filedialog = None
    ttk = None

# IO Imports
try:
    from pyoephys.io import load_npz_file
except Exception:
    load_npz_file = None

# Processing Imports
try:
    from pyoephys.processing._filters import (
        bandpass_filter,
        lowpass_filter,
        notch_filter,
    )
    from pyoephys.processing._features import (
        rectify,
        window_rms,
        extract_features,
        common_average_reference,
        envelope_extraction,
    )
except Exception:
    bandpass_filter = lowpass_filter = notch_filter = rectify = None
    window_rms = extract_features = common_average_reference = envelope_extraction = None

# Visualization libraries (guarded)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except Exception:
    plt = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

# Try to provide a PyQt5-based viewer as modern alternative; fall back to Tkinter below
try:
    from PyQt5 import QtWidgets, QtCore
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQt
    from matplotlib.figure import Figure as MplFigure
    _HAS_PYQT = True
except Exception:
    _HAS_PYQT = False

# TO-DO: implement this into a separate intan module
try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    _HAS_SKLEARN = True
except Exception:
    PCA = None
    LabelEncoder = None
    train_test_split = None
    _HAS_SKLEARN = False

# TensorFlow/Keras is optional for model training; import lazily when available
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    _HAS_TF = True
except Exception:
    Sequential = None
    Dense = None
    Dropout = None
    Adam = None
    to_categorical = None
    _HAS_TF = False

def _downsample_for_plot(y, max_points=2000):
    import math
    n = len(y)
    if n <= max_points:
        return y
    binsize = int(math.ceil(n / float(max_points)))
    n_out = int(math.ceil(n / float(binsize)))
    mins = np.empty(n_out)
    maxs = np.empty(n_out)
    for i in range(n_out):
        s = i * binsize
        e = min(n, (i + 1) * binsize)
        seg = y[s:e]
        mins[i] = np.min(seg)
        maxs[i] = np.max(seg)
    y_ds = np.empty(n_out * 2)
    y_ds[0::2] = mins
    y_ds[1::2] = maxs
    return y_ds


# --- IO helper utilities usable without GUI ---
def save_model_obj(obj, path):
    """Save a model object to `path`. Tries joblib, then Keras `save`, then pickle."""
    try:
        if path.lower().endswith('.joblib') and joblib is not None:
            joblib.dump(obj, path); return
    except Exception:
        pass
    try:
        if _HAS_TF and hasattr(obj, 'save') and path:
            obj.save(path); return
    except Exception:
        pass
    # fallback
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_model_obj(path):
    """Load a model object from `path`. Tries joblib, then Keras load, then pickle."""
    try:
        if path.lower().endswith('.joblib') and joblib is not None:
            return joblib.load(path)
    except Exception:
        pass
    try:
        if _HAS_TF:
            from tensorflow.keras.models import load_model
            return load_model(path)
    except Exception:
        pass
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_trials_csv(trials, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start_sample', 'end_sample'])
        for s, e in trials:
            w.writerow([int(s), int(e)])


def load_trials_csv(path):
    import csv
    out = []
    with open(path, 'r', newline='') as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        for row in rdr:
            if not row: continue
            try:
                s = int(row[0]); e = int(row[1])
            except Exception:
                continue
            out.append((s, e))
    return out


def save_features_npz(features, path):
    np.savez_compressed(path, features=features)


def load_features_npz(path):
    d = np.load(path, allow_pickle=True)
    if 'features' in d:
        return d['features']
    # try common keys
    for k in d.files:
        return d[k]
    return None

if _HAS_PYQT:
    class EMGViewerQt(QtWidgets.QMainWindow):
        """Minimal PyQt5-based EMG viewer to mirror Tk viewer functionality.

        This provides a lightweight, modern GUI using matplotlib's Qt backend.
        """
        def __init__(self, parent=None, num_channels=None, sample_rate=None):
            super().__init__(parent)
            self.setWindowTitle('EMG Viewer (PyQt)')
            self.data = None
            self.fs = float(sample_rate) if sample_rate is not None else 2000.0
            self.current_pos = 0
            self.playing = False
            self.max_points = 2000

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            vlay = QtWidgets.QVBoxLayout(central)

            # Create tabbed control area so we can port Tk tabs into PyQt
            tabs = QtWidgets.QTabWidget()

            # --- Acquisition tab (simple open/play controls) ---
            acq_tab = QtWidgets.QWidget()
            acq_layout = QtWidgets.QHBoxLayout(acq_tab)
            btn_open = QtWidgets.QPushButton('Open')
            btn_open.clicked.connect(self.open_file)
            acq_layout.addWidget(btn_open)
            self.btn_play = QtWidgets.QPushButton('Play')
            self.btn_play.clicked.connect(self._toggle_play)
            acq_layout.addWidget(self.btn_play)
            tabs.addTab(acq_tab, 'Acquisition')

            # --- Filtering tab ---
            filt_tab = QtWidgets.QWidget()
            filt_layout = QtWidgets.QHBoxLayout(filt_tab)
            filt_layout.addWidget(QtWidgets.QLabel('HP (Hz)'))
            self.spin_hp = QtWidgets.QDoubleSpinBox(); self.spin_hp.setRange(0.0, 1000.0); self.spin_hp.setValue(10.0)
            filt_layout.addWidget(self.spin_hp)
            filt_layout.addWidget(QtWidgets.QLabel('LP (Hz)'))
            self.spin_lp = QtWidgets.QDoubleSpinBox(); self.spin_lp.setRange(0.0, 10000.0); self.spin_lp.setValue(500.0)
            filt_layout.addWidget(self.spin_lp)
            filt_layout.addWidget(QtWidgets.QLabel('Notch'))
            self.combo_notch = QtWidgets.QComboBox(); self.combo_notch.addItems(['None', '50', '60'])
            filt_layout.addWidget(self.combo_notch)
            self.btn_apply_filters = QtWidgets.QPushButton('Apply Filters')
            self.btn_apply_filters.clicked.connect(self.apply_filters)
            filt_layout.addWidget(self.btn_apply_filters)
            filt_layout.addWidget(QtWidgets.QLabel('Channel'))
            self.chan_selector = QtWidgets.QSpinBox(); self.chan_selector.setRange(1,256); self.chan_selector.setValue(1)
            self.chan_selector.valueChanged.connect(self.plot_waveforms)
            filt_layout.addWidget(self.chan_selector)
            filt_layout.addWidget(QtWidgets.QLabel('FS'))
            self.spin_fs = QtWidgets.QSpinBox(); self.spin_fs.setRange(1,100000); self.spin_fs.setValue(int(self.fs))
            self.spin_fs.valueChanged.connect(self._on_fs_changed)
            filt_layout.addWidget(self.spin_fs)
            filt_layout.addWidget(QtWidgets.QLabel('Window (s)'))
            self.window_sec = QtWidgets.QDoubleSpinBox(); self.window_sec.setRange(0.01,60.0); self.window_sec.setValue(0.2)
            self.window_sec.valueChanged.connect(self.plot_waveforms)
            filt_layout.addWidget(self.window_sec)
            filt_layout.addWidget(QtWidgets.QLabel('Downsample'))
            self.spin_max_points = QtWidgets.QSpinBox(); self.spin_max_points.setRange(100,200000); self.spin_max_points.setValue(self.max_points)
            self.spin_max_points.valueChanged.connect(self._on_max_points_changed)
            filt_layout.addWidget(self.spin_max_points)
            filt_layout.addWidget(QtWidgets.QLabel('RMS win (s)'))
            self.rms_window_sec = QtWidgets.QDoubleSpinBox(); self.rms_window_sec.setRange(0.01,10.0); self.rms_window_sec.setSingleStep(0.01); self.rms_window_sec.setValue(0.05)
            filt_layout.addWidget(self.rms_window_sec)
            self.heat_live_checkbox = QtWidgets.QCheckBox('Heat Live'); self.heat_live_checkbox.setChecked(True)
            filt_layout.addWidget(self.heat_live_checkbox)
            tabs.addTab(filt_tab, 'Filtering')

            # --- Trials tab (placeholder, will implement segmentation/visualization) ---
            trials_tab = QtWidgets.QWidget()
            trials_layout = QtWidgets.QVBoxLayout(trials_tab)
            h = QtWidgets.QHBoxLayout()
            btn_detect = QtWidgets.QPushButton('Detect Trials')
            btn_detect.clicked.connect(self.detect_trials)
            h.addWidget(btn_detect)
            btn_load_events = QtWidgets.QPushButton('Load Events')
            btn_load_events.clicked.connect(self.load_events)
            h.addWidget(btn_load_events)
            trials_layout.addLayout(h)
            tabs.addTab(trials_tab, 'Trials')

            # --- Feature dataset tab (placeholder) ---
            feat_tab = QtWidgets.QWidget()
            feat_layout = QtWidgets.QHBoxLayout(feat_tab)
            btn_extract = QtWidgets.QPushButton('Extract Features')
            btn_extract.clicked.connect(self.extract_features_action)
            feat_layout.addWidget(btn_extract)
            btn_rms = QtWidgets.QPushButton('Compute RMS')
            btn_rms.clicked.connect(self.compute_rms_action)
            feat_layout.addWidget(btn_rms)
            btn_spec = QtWidgets.QPushButton('Show Spectrogram')
            btn_spec.clicked.connect(self.show_spectrogram_action)
            feat_layout.addWidget(btn_spec)
            btn_save_feat = QtWidgets.QPushButton('Save Features')
            btn_save_feat.clicked.connect(self.save_features)
            feat_layout.addWidget(btn_save_feat)
            tabs.addTab(feat_tab, 'Feature Dataset')

            # --- Model training tab (placeholder) ---
            train_tab = QtWidgets.QWidget()
            train_layout = QtWidgets.QVBoxLayout(train_tab)

            # Dataset controls
            ds_h = QtWidgets.QHBoxLayout()
            btn_load_ds = QtWidgets.QPushButton('Load Training Dataset')
            btn_load_ds.clicked.connect(self.load_training_dataset_qt)
            ds_h.addWidget(btn_load_ds)
            self.dataset_shape_label = QtWidgets.QLabel('Dataset shape: N/A')
            ds_h.addWidget(self.dataset_shape_label)
            train_layout.addLayout(ds_h)

            # Model config
            cfg_frame = QtWidgets.QGroupBox('Model Configuration')
            cfg_layout = QtWidgets.QFormLayout(cfg_frame)
            self.layer_sizes_entry = QtWidgets.QLineEdit('512,256,128')
            cfg_layout.addRow('Layer sizes (comma):', self.layer_sizes_entry)
            self.dropout_entry = QtWidgets.QLineEdit('0.5,0.5,0.5')
            cfg_layout.addRow('Dropout rate (comma):', self.dropout_entry)
            self.lr_entry = QtWidgets.QLineEdit('0.001')
            cfg_layout.addRow('Learning rate:', self.lr_entry)
            self.batch_entry = QtWidgets.QLineEdit('32')
            cfg_layout.addRow('Batch size:', self.batch_entry)
            self.epochs_entry = QtWidgets.QLineEdit('30')
            cfg_layout.addRow('Epochs:', self.epochs_entry)
            self.train_test_split_entry = QtWidgets.QLineEdit('0.8')
            cfg_layout.addRow('Train/Test split:', self.train_test_split_entry)
            train_layout.addWidget(cfg_frame)

            # Train / Save buttons
            btns_h = QtWidgets.QHBoxLayout()
            btn_train = QtWidgets.QPushButton('Train Model')
            btn_train.clicked.connect(self.train_model_action)
            btns_h.addWidget(btn_train)
            btn_save_model = QtWidgets.QPushButton('Save Model')
            btn_save_model.clicked.connect(self.save_model)
            btns_h.addWidget(btn_save_model)
            train_layout.addLayout(btns_h)
            tabs.addTab(train_tab, 'Model Training')

            vlay.addWidget(tabs)

            # Make waveform view taller by increasing figure height and stretch
            # plot area: waveform and RMS tabs
            self.plot_tabs = QtWidgets.QTabWidget()

            # waveform tab
            wave_w = QtWidgets.QWidget()
            wave_layout = QtWidgets.QVBoxLayout(wave_w)
            self.fig = MplFigure(figsize=(10, 4))
            self.canvas = FigureCanvasQt(self.fig)
            wave_layout.addWidget(self.canvas, 1)
            self.plot_tabs.addTab(wave_w, 'Waveform')

            # RMS heatmap tab
            heat_w = QtWidgets.QWidget()
            heat_layout = QtWidgets.QVBoxLayout(heat_w)
            self.fig_rms = MplFigure(figsize=(10, 4))
            self.canvas_rms = FigureCanvasQt(self.fig_rms)
            heat_layout.addWidget(self.canvas_rms, 1)
            self.plot_tabs.addTab(heat_w, 'RMS Heatmap')

            vlay.addWidget(self.plot_tabs, 3)

            self.scrub_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.scrub_slider.setMinimum(0); self.scrub_slider.setMaximum(0)
            self.scrub_slider.valueChanged.connect(self._on_slider_moved)
            try:
                self.scrub_slider.sliderPressed.connect(self._on_scrub_pressed)
                self.scrub_slider.sliderReleased.connect(self._on_scrub_released)
            except Exception:
                pass
            vlay.addWidget(self.scrub_slider)

            self.play_timer = QtCore.QTimer(); self.play_timer.setInterval(50); self.play_timer.timeout.connect(self._on_timer_tick)
            self._was_playing_on_scrub = False

            self.resize(1000,600)

        def _on_fs_changed(self, val): self.fs = float(val)
        def _on_max_points_changed(self, val): self.max_points = int(val); self.plot_waveforms()
        def _on_slider_moved(self, val):
            if not self.playing:
                self.current_pos = int(val)
                self.plot_waveforms()

        def open_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open EMG file', '', 'EMG Files (*.npz *.npy *.csv *.dat *.rhd);;RHD Files (*.rhd);;All Files (*)')
            if not path: return
            try:
                data = np.load(path) if path.lower().endswith('.npy') else np.loadtxt(path, delimiter=',')
            except Exception:
                try:
                    data = np.loadtxt(path)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', str(e))
                    return
            data = np.asarray(data)
            if data.ndim == 1: data = data[:,None]
            self.data = data
            self.chan_selector.setMaximum(max(1, self.data.shape[1]))
            self.scrub_slider.setMaximum(max(0, self.data.shape[0]-1))
            self.current_pos = 0
            self.plot_waveforms()

        def plot_waveforms(self):
            # draw on waveform figure/canvas
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if self.data is None:
                ax.text(0.5,0.5,'Open a file to begin',ha='center',va='center')
                self.canvas.draw_idle(); return
            ch = int(self.chan_selector.value())-1
            fs = max(1.0, float(self.spin_fs.value()))
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            start = int(self.current_pos); end = min(self.data.shape[0], start + window_samples)
            t = np.arange(start, end) / fs
            y = self.data[start:end, ch]
            if len(y)==0:
                ax.text(0.5,0.5,'No data in window',ha='center',va='center'); self.canvas.draw_idle(); return
            # downsample by envelope
            ty = _downsample_for_plot(y, max_points=self.max_points)
            # If downsample returns an interleaved min/max envelope (length != t), build matching tx
            if ty is None:
                ax.plot(t, y, lw=0.8)
            elif len(ty) == len(t):
                ax.plot(t, ty, lw=0.8)
            else:
                # assume ty is interleaved min/max of length 2*n_out
                n_out = len(ty) // 2
                if n_out > 0:
                    binsize = int(np.ceil(len(y) / float(n_out)))
                    centers = np.empty(n_out)
                    for i in range(n_out):
                        s = i * binsize
                        e = min(len(y), (i+1)*binsize)
                        idx = s + (e - s) // 2
                        centers[i] = float((start + idx) / fs)
                    tx_ds = np.empty(n_out*2)
                    tx_ds[0::2] = centers
                    tx_ds[1::2] = centers
                    ax.plot(tx_ds, ty, lw=0.8)
                else:
                    ax.plot(t, y, lw=0.8)
            ax.set_xlabel('Time (s)'); ax.set_title(f'Channel {ch+1} [{start}:{end}]')
            self.fig.tight_layout(); self.canvas.draw_idle()

        def apply_filters(self):
            if self.data is None:
                QtWidgets.QMessageBox.warning(self, 'Filters', 'No data loaded')
                return
            hp = float(self.spin_hp.value())
            lp = float(self.spin_lp.value())
            notch_text = self.combo_notch.currentText()
            notch = None if notch_text == 'None' else float(notch_text)
            # Use scipy if available
            if butter is None or filtfilt is None:
                QtWidgets.QMessageBox.information(self, 'Filters', 'SciPy not available; skipping filter')
                return
            try:
                nyq = 0.5 * float(self.spin_fs.value())
                b, a = butter(4, [max(0.0001, hp/nyq), min(0.9999, lp/nyq)], btype='band')
                # apply along axis 0 (samples)
                self.data = filtfilt(b, a, self.data, axis=0)
                if notch is not None:
                    w0 = notch / nyq
                    bn, an = iirnotch(w0, 30)
                    self.data = filtfilt(bn, an, self.data, axis=0)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Filter error', str(e))
                return
            self.plot_waveforms()

        def plot_heatmap(self):
            # draw RMS heatmap into RMS figure/canvas
            if self.data is None:
                return
            rms_mean = np.sqrt(np.mean(self.data**2, axis=0))
            rows, cols = 8, 8
            grid = np.full((rows, cols), np.nan)
            nch = self.data.shape[1]
            for i in range(min(nch, rows*cols)):
                r = i // cols; c = i % cols; grid[r,c] = float(rms_mean[i])
            self.fig_rms.clear(); ax = self.fig_rms.add_subplot(111); im = ax.imshow(grid, cmap='inferno', aspect='auto')
            self.fig_rms.colorbar(im); self.canvas_rms.draw_idle()

        def _on_timer_tick(self):
            if self.data is None: return
            fs = float(max(1.0, float(self.spin_fs.value())))
            dt = self.play_timer.interval() / 1000.0
            step = max(1, int(fs * dt))
            self.current_pos = int(self.current_pos) + step
            window_s = float(self.window_sec.value())
            window_samples = max(1, int(window_s * fs))
            max_pos = max(0, self.data.shape[0] - window_samples)
            if self.current_pos >= max_pos:
                self.current_pos = max_pos
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')
            # update scrub slider without triggering extra updates
            try:
                self.scrub_slider.blockSignals(True)
                self.scrub_slider.setValue(int(self.current_pos))
            finally:
                try: self.scrub_slider.blockSignals(False)
                except Exception: pass
            self.plot_waveforms()

        def _toggle_play(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Play', 'No data loaded')
                return
            if self.playing:
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')
            else:
                # ensure slider maximum is set
                try:
                    self.scrub_slider.setMaximum(max(0, self.data.shape[0]-1))
                except Exception:
                    pass
                self.play_timer.start(); self.playing = True; self.btn_play.setText('Pause')

        def _on_scrub_pressed(self):
            # pause playback while user scrubs
            self._was_playing_on_scrub = self.playing
            if self.playing:
                self.play_timer.stop(); self.playing = False; self.btn_play.setText('Play')

        def _on_scrub_released(self):
            # resume if it was playing
            if getattr(self, '_was_playing_on_scrub', False):
                self.play_timer.start(); self.playing = True; self.btn_play.setText('Pause')

        # --- Placeholder actions for tabs ported from Tk viewer ---
        def detect_trials(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Detect Trials', 'No data loaded')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            if ch < 0 or ch >= self.data.shape[1]:
                QtWidgets.QMessageBox.information(self, 'Detect Trials', 'Invalid channel selected')
                return
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            win_s = float(getattr(self, 'rms_window_sec', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'rms_window_sec') else 0.05
            win = max(1, int(win_s * fs))
            # moving RMS
            try:
                kernel = np.ones(win) / float(win)
                rms = np.sqrt(np.convolve(sig**2, kernel, mode='same'))
            except Exception:
                rms = np.abs(sig)
            thresh = float(np.mean(rms) + 2.0 * np.std(rms))
            mask = rms > thresh
            # detect rising/falling edges
            d = np.diff(mask.astype(int))
            starts = list(np.where(d == 1)[0] + 1)
            ends = list(np.where(d == -1)[0] + 1)
            # handle edge cases
            if mask[0]:
                starts.insert(0, 0)
            if mask[-1]:
                ends.append(len(mask))
            trials = []
            for s, e in zip(starts, ends):
                trials.append((int(s), int(e)))
            self.trials = trials
            # show dialog with table of trials
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Detected Trials')
            dl = QtWidgets.QVBoxLayout(dlg)
            tbl = QtWidgets.QTableWidget()
            tbl.setColumnCount(3)
            tbl.setHorizontalHeaderLabels(['Index', 'Start (s)', 'End (s)'])
            tbl.setRowCount(len(trials))
            for i, (s, e) in enumerate(trials):
                tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
                tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s/float(fs):.3f}"))
                tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{e/float(fs):.3f}"))
            tbl.cellDoubleClicked.connect(lambda r, c: self._jump_to_trial_row(r, dlg))
            dl.addWidget(tbl)
            btns = QtWidgets.QHBoxLayout()
            save_btn = QtWidgets.QPushButton('Save CSV')
            def _save():
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Trials', '', 'CSV Files (*.csv);;All Files (*)')
                if not path:
                    return
                try:
                    import csv
                    with open(path, 'w', newline='') as f:
                        w = csv.writer(f)
                        w.writerow(['start_sample', 'end_sample'])
                        for s, e in trials:
                            w.writerow([s, e])
                    QtWidgets.QMessageBox.information(self, 'Save Trials', f'Saved {len(trials)} trials to {path}')
                except Exception as ex:
                    QtWidgets.QMessageBox.warning(self, 'Save Trials', str(ex))
            save_btn.clicked.connect(_save)
            btns.addWidget(save_btn)
            close_btn = QtWidgets.QPushButton('Close')
            close_btn.clicked.connect(dlg.accept)
            btns.addWidget(close_btn)
            dl.addLayout(btns)
            dlg.resize(400, 300)
            dlg.exec_()

        def load_events(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Events', '', 'CSV Files (*.csv);;All Files (*)')
            if not path:
                return
            try:
                import csv
                ev = []
                with open(path, 'r', newline='') as f:
                    rdr = csv.reader(f)
                    for row in rdr:
                        ev.append(row)
                self.events = ev
                QtWidgets.QMessageBox.information(self, 'Events', f'Loaded {len(ev)} event rows')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Events', str(e))

        def _jump_to_trial_row(self, row, dlg=None):
            if not hasattr(self, 'trials') or row < 0 or row >= len(self.trials):
                return
            s, e = self.trials[row]
            self.current_pos = int(s)
            try:
                self.scrub_slider.setValue(int(s))
            except Exception:
                pass
            self.plot_waveforms()
            if dlg is not None:
                dlg.accept()

        def load_trials(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Trials', '', 'CSV Files (*.csv);;All Files (*)')
            if not path:
                return
            try:
                import csv
                trials = []
                with open(path, 'r', newline='') as f:
                    rdr = csv.reader(f)
                    hdr = next(rdr, None)
                    for row in rdr:
                        if not row: continue
                        try:
                            s = int(row[0]); e = int(row[1])
                        except Exception:
                            continue
                        trials.append((s, e))
                self.trials = trials
                QtWidgets.QMessageBox.information(self, 'Load Trials', f'Loaded {len(trials)} trials')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Trials', str(e))

        def save_model(self):
            # Save the last trained model if present
            if not hasattr(self, 'trained_model') or self.trained_model is None:
                QtWidgets.QMessageBox.information(self, 'Save Model', 'No trained model to save')
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model', '', 'Joblib (*.joblib);;Keras (.keras);;All Files (*)')
            if not path:
                return
            try:
                save_model_obj(self.trained_model, path)
                QtWidgets.QMessageBox.information(self, 'Save Model', f'Model saved to {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Save Model', str(e))

        def load_model(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Model', '', 'Joblib (*.joblib);;Keras (.keras);;All Files (*)')
            if not path:
                return
            try:
                model = load_model_obj(path)
                self.trained_model = model
                QtWidgets.QMessageBox.information(self, 'Load Model', f'Loaded model from {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Model', str(e))

        def extract_features_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Features', 'No data loaded')
                return
            if extract_features is None:
                QtWidgets.QMessageBox.information(self, 'Features', 'Feature extraction function not available')
                return
            try:
                # expect data shape (samples, channels) -> extract_features handles shapes internally
                feats = extract_features(self.data.T) # extract_features typically expects channels first? pyoephys extract_features (features.py) says segment=(channels, samples)
                # But self.data is (samples, channels).
                # Checking pyoephys.processing._filters code from step 79:
                # def extract_features(segment, ...):
                #   for ch in segment: ...
                # So it expects (n_channels, n_samples).
                # self.data in viewer is (samples, channels) because of how it is loaded from loadtxt.
                # So we pass self.data.T
                feats = extract_features(self.data.T)
                self.last_features = feats
                QtWidgets.QMessageBox.information(self, 'Features', f'Extracted features shape: {getattr(feats, "shape", "unknown")}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Features', str(e))

        def save_features(self):
            if not hasattr(self, 'last_features'):
                QtWidgets.QMessageBox.information(self, 'Save Features', 'No features to save')
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Features', '', 'NumPy (.npz);;All Files (*)')
            if not path:
                return
            try:
                np.savez_compressed(path, features=self.last_features)
                QtWidgets.QMessageBox.information(self, 'Save Features', f'Saved features to {path}')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Save Features', str(e))

        def train_model_action(self):
            if not _HAS_TF:
                QtWidgets.QMessageBox.information(self, 'Train Model', 'TensorFlow/Keras not available in this environment')
                return
            QtWidgets.QMessageBox.information(self, 'Train Model', 'Training flow not implemented in placeholder')

        def compute_rms_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'RMS', 'No data loaded')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            if ch < 0 or ch >= self.data.shape[1]:
                QtWidgets.QMessageBox.information(self, 'RMS', 'Invalid channel selected')
                return
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            win_s = float(getattr(self, 'rms_window_sec', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'rms_window_sec') else 0.05
            win = max(1, int(win_s * fs))
            kernel = np.ones(win) / float(win)
            try:
                rms = np.sqrt(np.convolve(sig**2, kernel, mode='same'))
            except Exception:
                rms = np.abs(sig)
            self.last_rms = rms
            t = np.arange(len(rms)) / float(fs)
            self.fig.clear(); ax = self.fig.add_subplot(111)
            ax.plot(t, rms, lw=0.8)
            ax.set_title(f'RMS (ch {ch+1})')
            ax.set_xlabel('Time (s)')
            self.fig.tight_layout(); self.canvas.draw_idle()

        def show_spectrogram_action(self):
            if self.data is None:
                QtWidgets.QMessageBox.information(self, 'Spectrogram', 'No data loaded')
                return
            if spectrogram is None:
                QtWidgets.QMessageBox.information(self, 'Spectrogram', 'SciPy not available; cannot compute spectrogram')
                return
            ch = int(getattr(self, 'chan_selector', QtWidgets.QSpinBox()).value()) - 1
            sig = self.data[:, ch].astype(float)
            fs = float(getattr(self, 'spin_fs', QtWidgets.QSpinBox()).value()) if hasattr(self, 'spin_fs') else self.fs
            try:
                f, t, Sxx = spectrogram(sig, fs=fs, nperseg=256, noverlap=128)
                self.fig.clear(); ax = self.fig.add_subplot(111)
                im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='auto')
                ax.set_ylabel('Frequency [Hz]'); ax.set_xlabel('Time [s]')
                self.fig.colorbar(im)
                self.fig.tight_layout(); self.canvas.draw_idle()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Spectrogram', str(e))

        def load_training_dataset_qt(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Training Dataset', '', 'NumPy Compressed (*.npz);;All Files (*)')
            if not path:
                return
            try:
                data = np.load(path, allow_pickle=True)
                X = data.get('features') if 'features' in data else data.get('X') if 'X' in data else None
                y = data.get('labels') if 'labels' in data else data.get('y') if 'y' in data else None
                if X is None:
                    QtWidgets.QMessageBox.warning(self, 'Load Dataset', 'No features found in archive')
                    return
                self.X_train = X
                self.y_train = y
                self.dataset_shape_label.setText(f'Dataset shape: {getattr(X, "shape", "unknown")}')
                QtWidgets.QMessageBox.information(self, 'Load Dataset', 'Dataset loaded')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, 'Load Dataset', str(e))

        def update_network_diagram(self):
            # Placeholder: could visualize network architecture in future
            return

    # expose EMGViewer name to rest of package
    EMGViewer = EMGViewerQt

# =========================================================================================
# =========================================================================================
#   TKINTER FALLBACK (if PyQT is not available)
# =========================================================================================
class EMGViewerTk:
    """Fallback EMG viewer using Tkinter."""
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Viewer (Tkinter)")
        self.data = None
        label = tk.Label(root, text="PyQt5 not installed.\nThis is a minimal fallback viewer.")
        label.pack(padx=20, pady=20)

if not _HAS_PYQT:
    EMGViewer = EMGViewerTk
