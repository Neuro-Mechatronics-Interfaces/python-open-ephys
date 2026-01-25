
import math
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.QtGui import QDoubleValidator
from pyoephys.processing import RealtimeFilter


# --- additions/changes inside your existing module ---

class FilterControls(QtWidgets.QWidget):
    """Dockable panel to control RealtimeFilter + view settings."""
    applyRequested = QtCore.pyqtSignal(dict)     # filter kwargs
    resetStateRequested = QtCore.pyqtSignal()    # filter state reset
    bypassToggled = QtCore.pyqtSignal(bool)      # True => show raw

    # NEW signals for view controls
    viewApplyRequested = QtCore.pyqtSignal(float, float)  # (ymin, ymax)
    autoYChanged = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        ff = QtWidgets.QFormLayout(self)
        ff.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        # ---- Bypass
        self.chkBypass = QtWidgets.QCheckBox("Bypass (show raw)")
        ff.addRow(self.chkBypass)
        self.chkBypass.toggled.connect(self.bypassToggled.emit)

        # ---- Band-pass
        self.chkBP = QtWidgets.QCheckBox("Enable band-pass"); self.chkBP.setChecked(True)
        self.spBPlo = QtWidgets.QDoubleSpinBox(); self.spBPlo.setRange(0.1, 10_000); self.spBPlo.setValue(20.0)
        self.spBPhi = QtWidgets.QDoubleSpinBox(); self.spBPhi.setRange(0.1, 10_000); self.spBPhi.setValue(498.0)
        self.spBPord = QtWidgets.QSpinBox();      self.spBPord.setRange(1, 10); self.spBPord.setValue(4)
        ff.addRow(self.chkBP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Low (Hz):"));  row.addWidget(self.spBPlo)
        row.addWidget(QtWidgets.QLabel("High (Hz):")); row.addWidget(self.spBPhi)
        row.addWidget(QtWidgets.QLabel("Order:"));     row.addWidget(self.spBPord)
        ff.addRow(row)

        # ---- Notch
        self.chkNotch = QtWidgets.QCheckBox("Enable notch"); self.chkNotch.setChecked(True)
        self.leNotch = QtWidgets.QLineEdit("60, 120, 180")
        self.spQ = QtWidgets.QDoubleSpinBox(); self.spQ.setRange(1.0, 200.0); self.spQ.setValue(30.0)
        ff.addRow(self.chkNotch)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Freqs (Hz):")); row.addWidget(self.leNotch)
        row.addWidget(QtWidgets.QLabel("Q:"));          row.addWidget(self.spQ)
        ff.addRow(row)

        # ---- Low-pass
        self.chkLP = QtWidgets.QCheckBox("Enable low-pass (post)"); self.chkLP.setChecked(False)
        self.spLPcut = QtWidgets.QDoubleSpinBox(); self.spLPcut.setRange(0.1, 10_000); self.spLPcut.setValue(300.0)
        self.spLPord = QtWidgets.QSpinBox();      self.spLPord.setRange(1, 10); self.spLPord.setValue(4)
        ff.addRow(self.chkLP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cutoff (Hz):")); row.addWidget(self.spLPcut)
        row.addWidget(QtWidgets.QLabel("Order:"));       row.addWidget(self.spLPord)
        ff.addRow(row)

        # ---- Filter buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.btnApply = QtWidgets.QPushButton("Apply filter")
        self.btnReset = QtWidgets.QPushButton("Reset state")
        btnRow.addWidget(self.btnApply); btnRow.addStretch(1); btnRow.addWidget(self.btnReset)
        ff.addRow(btnRow)

        self.btnApply.clicked.connect(self._emit_apply)
        self.btnReset.clicked.connect(self.resetStateRequested.emit)

        # ===== NEW: View controls (Y range) =====
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine)
        ff.addRow(line)

        self.chkAutoY = QtWidgets.QCheckBox("Auto Y (robust)"); self.chkAutoY.setChecked(False)
        ff.addRow(self.chkAutoY)
        self.chkAutoY.toggled.connect(self.autoYChanged.emit)

        row = QtWidgets.QHBoxLayout()
        self.spYmin = QtWidgets.QDoubleSpinBox(); self.spYmin.setRange(-1e6, 1e6); self.spYmin.setDecimals(1); self.spYmin.setValue(-500.0)
        self.spYmax = QtWidgets.QDoubleSpinBox(); self.spYmax.setRange(-1e6, 1e6); self.spYmax.setDecimals(1); self.spYmax.setValue(500.0)
        row.addWidget(QtWidgets.QLabel("Y min (µV):")); row.addWidget(self.spYmin)
        row.addWidget(QtWidgets.QLabel("Y max (µV):")); row.addWidget(self.spYmax)
        ff.addRow(row)

        self.btnViewApply = QtWidgets.QPushButton("Apply view")
        ff.addRow(self.btnViewApply)
        self.btnViewApply.clicked.connect(lambda: self.viewApplyRequested.emit(self.spYmin.value(), self.spYmax.value()))

    def _emit_apply(self):
        txt = self.leNotch.text().replace(",", " ")
        freqs = [float(tok) for tok in txt.split() if tok.strip()]
        kwargs = dict(
            enable_bandpass=self.chkBP.isChecked(),
            bp_low=float(self.spBPlo.value()),
            bp_high=float(self.spBPhi.value()),
            bp_order=int(self.spBPord.value()),
            enable_notch=self.chkNotch.isChecked(),
            notch_freqs=tuple(freqs),
            notch_q=float(self.spQ.value()),
            enable_lowpass=self.chkLP.isChecked(),
            lp_cut=float(self.spLPcut.value()),
            lp_order=int(self.spLPord.value()),
        )
        self.applyRequested.emit(kwargs)


class StackedPlot(QtWidgets.QMainWindow):
    def __init__(
        self,
        client,
        window_secs=5.0,
        ui_hz=20,
        auto_ylim=False,
        y_limits=(-500.0, 500.0),
        robust_pct=(1, 99),
        min_span=1e-6,
        smooth_alpha=0.25,
        symmetric=False,
        max_points=2000,
        # filter defaults:
        enable_filter_ui=True,
    ):
        super().__init__()
        self.client = client
        self.window_secs = float(window_secs)
        self.auto_ylim = bool(auto_ylim)
        self.fixed_ylim = tuple(y_limits) if y_limits is not None else None
        self.robust_pct = robust_pct
        self.min_span = float(min_span)
        self.smooth_alpha = float(smooth_alpha)
        self.symmetric = bool(symmetric)
        self.max_points = int(max_points)
        self.N_channels = self.client.N_channels

        self.setWindowTitle(
            f"Stacked — {self.client.name}  type={self.client.type}  "
            f"nom fs={self.client.fs:.1f}Hz  chans={self.N_channels}"
        )
        self.resize(1200, 820)

        # ---- central scrollable area with vertical layout ----
        central = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(6, 6, 6, 6); vbox.setSpacing(4)

        self.scroll = QtWidgets.QScrollArea(); self.scroll.setWidgetResizable(True)
        self.inner = QtWidgets.QWidget()
        self.inner_layout = QtWidgets.QVBoxLayout(self.inner)
        self.inner_layout.setContentsMargins(0, 0, 0, 0); self.inner_layout.setSpacing(2)

        # Construct plots
        self.plots, self.curves = [], []
        self._ylims = [None] * self.N_channels
        for i in range(self.N_channels):
            pw = pg.PlotWidget()
            pw.showGrid(x=True, y=True)
            pw.setXRange(-self.window_secs, 0)
            if i < self.N_channels - 1:
                pw.hideAxis("bottom")
            pw.setLabel("left", f"Ch {self.client.channel_index[i]}")
            curve = pw.plot(pen=pg.mkPen(pg.intColor(i, hues=self.N_channels), width=1))
            if self.fixed_ylim is not None:
                pw.setYRange(self.fixed_ylim[0], self.fixed_ylim[1], padding=0)
            self.inner_layout.addWidget(pw)
            self.plots.append(pw); self.curves.append(curve)

        for i in range(1, self.N_channels):
            self.plots[i].setXLink(self.plots[0])

        self.scroll.setWidget(self.inner)
        vbox.addWidget(self.scroll)
        self.setCentralWidget(central)

        # ---- Realtime filter
        self._bypass = False
        self.filter = RealtimeFilter(
            fs=self.client.fs,
            n_channels=self.N_channels,
            # bp_low=20.0,
            # bp_high=498.0,
            # bp_order=4,
            # enable_bandpass=True,
            # notch_freqs=(60.0,),
            # notch_q=30.0,
            # enable_notch=True,
            # lp_cut=None,
            # lp_order=4,
            # enable_lowpass=False,
        )

        self._ft = np.zeros(self.client.N_samples, dtype=np.float64)  # filter state for each channel
        self._fy = np.zeros((self.N_channels, self.client.N_samples), dtype=np.float32)  # filtered output
        self._fwidx = 0
        self._fcount = 0

        # ---- Optional filter UI
        if enable_filter_ui:
            self._add_filter_dock()

        # timers
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / ui_hz))

        self.fs_timer = QtCore.QTimer()
        self.fs_timer.timeout.connect(self._update_title_fs)
        self.fs_timer.start(1000)

    # ---------- UI bits ----------
    def _add_filter_dock(self):
        dock = QtWidgets.QDockWidget("Filter", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.controls = FilterControls()
        dock.setWidget(self.controls)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # Filtering signals
        self.controls.applyRequested.connect(self._on_apply_filter)
        self.controls.resetStateRequested.connect(self.filter.reset)
        self.controls.bypassToggled.connect(self._on_bypass)

        # Viewing signals
        self.controls.viewApplyRequested.connect(self._on_view_apply)
        self.controls.autoYChanged.connect(self._on_auto_y)

    def _on_view_apply(self, ymin: float, ymax: float):
        if ymin >= ymax:
            QtWidgets.QMessageBox.warning(self, "View error", "Y min must be < Y max.")
            return
        self.fixed_ylim = (ymin, ymax)
        self.auto_ylim = False
        for pw in self.plots:
            pw.setYRange(ymin, ymax, padding=0)

    def _on_auto_y(self, enabled: bool):
        self.auto_ylim = bool(enabled)
        # when turning auto on, clear smoothed limits so it re-initializes
        if self.auto_ylim:
            self._ylims = [None] * self.N_channels
        else:
            # when turning it off, snap to current fixed range if set
            if self.fixed_ylim is not None:
                for pw in self.plots:
                    pw.setYRange(self.fixed_ylim[0], self.fixed_ylim[1], padding=0)

    def _on_apply_filter(self, kw):
        try:
            self.filter.reconfigure(**kw)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Filter config error", str(e))

    def _on_bypass(self, state: bool):
        self._bypass = bool(state)

    # ---------- window plumbing ----------
    def closeEvent(self, ev):
        self.client.stop()
        super().closeEvent(ev)

    def _update_title_fs(self):
        fs_hat = self.client.fs_estimate()
        if np.isfinite(fs_hat):
            self.setWindowTitle(
                f"LSL Stacked — {self.client.name}  type={self.client.type}  "
                f"nom fs={self.client.fs:.1f}Hz  est fs={fs_hat:.2f}Hz  chans={self.N_channels}"
            )

    def _robust_limits(self, y):
        lo, hi = np.nanpercentile(y, self.robust_pct)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if self.symmetric:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        span = max(self.min_span, hi - lo)
        pad = 0.10 * span
        return lo - pad, hi + pad

    def _smooth_set_ylim(self, idx, lo, hi):
        if self._ylims[idx] is None:
            self._ylims[idx] = [lo, hi]
        else:
            a = self.smooth_alpha
            self._ylims[idx][0] = (1 - a) * self._ylims[idx][0] + a * lo
            self._ylims[idx][1] = (1 - a) * self._ylims[idx][1] + a * hi
        self.plots[idx].setYRange(self._ylims[idx][0], self._ylims[idx][1], padding=0)

    def _maybe_decimate(self, t, y):
        n = t.size
        if n <= self.max_points:
            return t, y
        step = int(math.ceil(n / self.max_points))
        return t[::step], y[..., ::step]

    # ---------- main update ----------
    def update_plots(self):
        # get only new samples
        t_new, Y_new = self.client.drain_new()
        if t_new is not None and Y_new.shape[1] > 0:
            # filter just the new tail (or bypass)
            if not getattr(self, "_bypass", False):
                Y_new = self.filter.process(Y_new)

            # append to filtered ring
            n = Y_new.shape[1]
            N = int(self.client.N_samples)
            if n > N:
                # keep only the most recent N samples to avoid overflow
                Y_new = Y_new[:, -N:]
                t_new = t_new[-N:]
                n = N

            dst = self._fwidx % self.client.N_samples
            first = min(n, self.client.N_samples - dst)
            self._fy[:, dst:dst + first] = Y_new[:, :first]
            self._ft[dst:dst + first] = t_new[:first]
            rem = n - first
            if rem > 0:
                self._fy[:, :rem] = Y_new[:, first:]
                self._ft[:rem] = t_new[first:]
            self._fwidx = (self._fwidx + n) % self.client.N_samples
            self._fcount = min(self._fcount + n, self.client.N_samples)

        # nothing to draw yet
        if self._fcount == 0:
            return

        # build last window from filtered ring (exactly like client.latest, but on filtered data)
        end = self._fwidx
        if self._fcount < self.client.N_samples:
            t = self._ft[:self._fcount].copy()
            Y = self._fy[:, :self._fcount].copy()
        else:
            t = np.hstack((self._ft[end:], self._ft[:end])).copy()
            Y = np.hstack((self._fy[:, end:], self._fy[:, :end])).copy()

        t_last = t[-1]
        t_rel = t - t_last
        mask = t_rel >= -self.window_secs
        t_rel, Y = t_rel[mask], Y[:, mask]

        # (optional) decimate for draw speed
        t_rel, Y = self._maybe_decimate(t_rel, Y)

        for i in range(self.N_channels):
            yi = Y[i]
            self.curves[i].setData(t_rel, yi)
            if self.auto_ylim and yi.size:
                lims = self._robust_limits(yi)
                if lims is not None:
                    self._smooth_set_ylim(i, *lims)


    # def update_plots(self):
    #     t_rel, Y = self.client.latest()
    #     if t_rel is None:
    #         return
    #
    #     # filter (stateful) unless bypassed
    #     if not self._bypass:
    #         Y = self.filter.process(Y)
    #
    #     # decimate for draw speed
    #     t_rel, Y = self._maybe_decimate(t_rel, Y)
    #
    #     for i in range(self.N_channels):
    #         yi = Y[i]
    #         self.curves[i].setData(t_rel, yi)
    #         if self.auto_ylim and yi.size:
    #             lims = self._robust_limits(yi)
    #             if lims is not None:
    #                 self._smooth_set_ylim(i, *lims)


# Can be removed at a later time once the new StackedPlotter is functioning well
# class oldStackedPlotter(QtWidgets.QMainWindow):
#     def __init__(self, client, channels_to_plot="all", interval_ms=50, buffer_secs=5, downsample_factor=1, ylim=None,
#                  enable_car=False, enable_bandpass=False, enable_notch=False, lowcut=0.1, highcut=400.0):
#         super().__init__()
#         self.setWindowTitle("Real-time Stacked EMG Viewer")
#
#         self.client = client
#         self.sampling_rate = client.sampling_rate
#         self.N_channels_total = client.N_channels
#         self.interval_ms = interval_ms
#         self.buffer_secs = buffer_secs
#         self.downsample_factor = downsample_factor
#
#         self.channel_height = 150
#         self.broken_ch_std_threshold = 300  # std
#         self.last_sample_count = 0
#
#         # Channel setup
#         if channels_to_plot == "all":
#             self.channels_to_plot = list(range(self.N_channels_total))
#         else:
#             self.channels_to_plot = channels_to_plot or [0]
#
#         self.N_channels = len(self.channels_to_plot)
#
#         self.buffer_size = int(buffer_secs * self.sampling_rate // self.downsample_factor)
#         self.buffers = np.zeros((self.N_channels, self.buffer_size), dtype=np.float32)
#         self.x = np.linspace(-buffer_secs, 0, self.buffer_size)
#         self.ylim = ylim if ylim is not None else (-200, 200)
#         self.plot_ptr = np.zeros(self.N_channels, dtype=int)
#
#         # Initialize filter
#         self.emg_filter = RealtimeEMGFilter(
#             sampling_rate=self.sampling_rate,
#             N_channels=self.N_channels,
#             lowcut=lowcut,
#             highcut=highcut,
#             enable_car=enable_car,
#             enable_bandpass=enable_bandpass,
#             enable_notch=enable_notch,
#         )
#
#         self.init_ui()
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.elapsed_timer = QtCore.QElapsedTimer()
#
#     def init_ui(self):
#         self.main_widget = QtWidgets.QWidget()
#         main_layout = QtWidgets.QHBoxLayout(self.main_widget)
#         self.setCentralWidget(self.main_widget)
#
#         # Scrollable plot layout
#         scroll_area = QtWidgets.QScrollArea()
#         scroll_area.setWidgetResizable(True)
#         plot_container = QtWidgets.QWidget()
#         self.plot_layout = QtWidgets.QVBoxLayout(plot_container)
#         scroll_area.setWidget(plot_container)
#         main_layout.addWidget(scroll_area, stretch=5)
#
#         # Curves and plots
#         self.curves = []
#         self.plot_widgets = []
#         for i in range(self.N_channels):
#             pw = pg.PlotWidget()
#             pw.setMaximumHeight(self.channel_height)
#             pw.setYRange(self.ylim[0], self.ylim[1], padding=0)
#             pw.setMouseEnabled(x=False, y=False)
#             pw.setMenuEnabled(False)
#             pw.showGrid(x=True, y=True)
#             if i < self.N_channels - 1:
#                 pw.hideAxis('bottom')
#             pw.getPlotItem().getAxis('left').setWidth(50)  # Allocate label width
#             pw.setLabel('left', f'Ch {self.channels_to_plot[i]} (μV)', **{'color': '#000', 'font-size': '9pt'})
#             curve = pw.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
#             self.curves.append(curve)
#             self.plot_widgets.append(pw)
#             self.plot_layout.addWidget(pw)
#
#         # Control panel
#         self.ctrl_panel = QtWidgets.QGroupBox("Controls")
#         ctrl_layout = QtWidgets.QVBoxLayout()
#
#         self.car_checkbox = QtWidgets.QCheckBox("CAR Filter")
#         self.bandpass_checkbox = QtWidgets.QCheckBox("Bandpass Filter")
#         self.notch_checkbox = QtWidgets.QCheckBox("Notch Filter")
#
#         ctrl_layout.addWidget(self.car_checkbox)
#         ctrl_layout.addWidget(self.bandpass_checkbox)
#         ctrl_layout.addWidget(self.notch_checkbox)
#
#         self.low_cut_input = QtWidgets.QLineEdit("10")
#         self.high_cut_input = QtWidgets.QLineEdit("500")
#         for box in [self.low_cut_input, self.high_cut_input]:
#             box.setValidator(QDoubleValidator(0.0, self.sampling_rate / 2, 2))
#
#         ctrl_layout.addWidget(QtWidgets.QLabel("Bandpass Low Cut (Hz):"))
#         ctrl_layout.addWidget(self.low_cut_input)
#         ctrl_layout.addWidget(QtWidgets.QLabel("Bandpass High Cut (Hz):"))
#         ctrl_layout.addWidget(self.high_cut_input)
#
#         # Connect filter toggles
#         self.car_checkbox.stateChanged.connect(self.update_filters)
#         self.bandpass_checkbox.stateChanged.connect(self.update_filters)
#         self.notch_checkbox.stateChanged.connect(self.update_filters)
#         self.low_cut_input.textChanged.connect(self.update_filters)
#         self.high_cut_input.textChanged.connect(self.update_filters)
#
#         ctrl_layout.addStretch(1)
#         self.ctrl_panel.setLayout(ctrl_layout)
#         main_layout.addWidget(self.ctrl_panel, stretch=1)
#
#         self.resize(900, 600)
#
#     def update_filters(self):
#         try:
#             low = float(self.low_cut_input.text())
#             high = float(self.high_cut_input.text())
#             self.emg_filter.set_bandpass(low, high)
#         except ValueError:
#             pass
#
#         self.emg_filter.toggle_car(self.car_checkbox.isChecked())
#         self.emg_filter.toggle_notch(self.notch_checkbox.isChecked())
#         self.emg_filter.toggle_bandpass(self.bandpass_checkbox.isChecked())
#
#     def start(self):
#         self.client.start_streaming()
#         self.elapsed_timer.start()
#         self.timer.start(self.interval_ms)
#         self.show()
#
#     def update_plot(self):
#
#         raw = self.client.get_latest_window(window_ms=self.interval_ms)
#         if raw is None or raw.shape[1] == 0:
#             return
#
#         # Extract selected channels and filter
#         data = raw[self.channels_to_plot]
#         filtered = np.copy(data)
#         # filtered = self.emg_filter.update(data)
#
#         for i in range(self.N_channels):
#             new_data = filtered[i][::self.downsample_factor]
#             n_new = new_data.shape[0]
#             if n_new == 0:
#                 continue
#
#             #self.buffers[i] = np.roll(self.buffers[i], n_new)
#             #self.buffers[i][-len(new_data):] = new_data
#             #self.curves[i].setData(self.x, self.buffers[i])
#
#             buf = self.buffers[i]
#             ptr = self.plot_ptr[i]
#
#             if ptr + n_new < self.buffer_size:
#                 buf[ptr:ptr + n_new] = new_data
#                 self.plot_ptr[i] += n_new
#             else:
#                 # Roll only after buffer is filled
#                 extra = ptr + n_new - self.buffer_size
#                 buf[:-n_new] = buf[n_new:]
#                 buf[-n_new:] = new_data
#                 self.plot_ptr[i] = self.buffer_size  # Lock to full
#
#             self.curves[i].setData(self.x, buf)
#
#     def closeEvent(self, event):
#         self.client.stop_streaming()
#         event.accept()
#
#
# class StackedPlotter(QtWidgets.QMainWindow):
#     def __init__(self, client, interval_ms=100, window_secs=5.0, downsample=1, ylim=(-200, 200)):
#         super().__init__()
#         self.setWindowTitle("Real-time Stacked EMG Viewer")
#         self.client = client
#         self.fs = self.client.fs
#         self.interval_ms = interval_ms
#         self.buffer_len = int(window_secs * self.fs) // downsample
#         self.downsample = downsample
#
#         self.channels = client.channels
#         self.N_channels = len(self.channels)
#         self.ylim = ylim
#
#         self.x = np.linspace(-window_secs, 0, self.buffer_len)
#         self.buffers = np.zeros((self.N_channels, self.buffer_len), dtype=np.float32)
#
#         self._setup_ui()
#
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start(self.interval_ms)
#
#     def _setup_ui(self):
#         central = QtWidgets.QWidget()
#         self.setCentralWidget(central)
#         layout = QtWidgets.QHBoxLayout(central)
#
#         # --- Scrollable plots ---
#         scroll = QtWidgets.QScrollArea()
#         scroll.setWidgetResizable(True)
#         plot_container = QtWidgets.QWidget()
#         self.plot_layout = QtWidgets.QVBoxLayout(plot_container)
#         scroll.setWidget(plot_container)
#         layout.addWidget(scroll, stretch=5)
#
#         self.plots = []
#         self.curves = []
#
#         for i, ch in enumerate(self.channels):
#             pw = pg.PlotWidget()
#             pw.setMaximumHeight(150)
#             pw.setYRange(*self.ylim)
#             pw.setMouseEnabled(x=False, y=False)
#             pw.setMenuEnabled(False)
#             pw.showGrid(x=True, y=True)
#             pw.setLabel('left', f"CH {ch}", **{'color': '#000', 'font-size': '9pt'})
#             if i < len(self.channels) - 1:
#                 pw.hideAxis('bottom')
#             curve = pw.plot(pen=pg.mkPen(color=pg.intColor(i), width=1))
#             self.plot_layout.addWidget(pw)
#             self.plots.append(pw)
#             self.curves.append(curve)
#
#         # --- Filter controls ---
#         self.ctrl_panel = QtWidgets.QGroupBox("Filters")
#         ctrl_layout = QtWidgets.QVBoxLayout()
#
#         self.car_checkbox = QtWidgets.QCheckBox("CAR")
#         self.bp_checkbox = QtWidgets.QCheckBox("Bandpass")
#         self.notch_checkbox = QtWidgets.QCheckBox("Notch")
#         self.low_cut_input = QtWidgets.QLineEdit("10")
#         self.high_cut_input = QtWidgets.QLineEdit("500")
#         for box in [self.low_cut_input, self.high_cut_input]:
#             box.setValidator(QDoubleValidator(0.0, self.fs / 2, 2))
#
#         ctrl_layout.addWidget(self.car_checkbox)
#         ctrl_layout.addWidget(self.bp_checkbox)
#         ctrl_layout.addWidget(self.notch_checkbox)
#         ctrl_layout.addWidget(QtWidgets.QLabel("Low cut (Hz):"))
#         ctrl_layout.addWidget(self.low_cut_input)
#         ctrl_layout.addWidget(QtWidgets.QLabel("High cut (Hz):"))
#         ctrl_layout.addWidget(self.high_cut_input)
#         self.ctrl_panel.setLayout(ctrl_layout)
#         layout.addWidget(self.ctrl_panel, stretch=1)
#
#         # Connect controls
#         self.car_checkbox.stateChanged.connect(self.update_filters)
#         self.bp_checkbox.stateChanged.connect(self.update_filters)
#         self.notch_checkbox.stateChanged.connect(self.update_filters)
#         self.low_cut_input.textChanged.connect(self.update_filters)
#         self.high_cut_input.textChanged.connect(self.update_filters)
#
#     def update_filters(self):
#         try:
#             low = float(self.low_cut_input.text())
#             high = float(self.high_cut_input.text())
#             self.client.filter.set_bandpass(low, high)
#         except ValueError:
#             pass
#         self.client.filter.toggle_car(self.car_checkbox.isChecked())
#         self.client.filter.toggle_bandpass(self.bp_checkbox.isChecked())
#         self.client.filter.toggle_notch(self.notch_checkbox.isChecked())
#
#     def update_plot(self):
#         window = self.client.get_latest_window(window_ms=self.interval_ms)
#         if window is None or window.shape[1] == 0:
#             return
#
#         for i, ch in enumerate(self.channels):
#             new_data = window[i][::self.downsample]
#             n_new = new_data.shape[0]
#             if n_new == 0:
#                 continue
#             if n_new < self.buffer_len:
#                 self.buffers[i, :-n_new] = self.buffers[i, n_new:]
#                 self.buffers[i, -n_new:] = new_data
#             else:
#                 self.buffers[i, :] = new_data[-self.buffer_len:]
#
#             self.curves[i].setData(self.x, self.buffers[i])
#
#     def closeEvent(self, event):
#         self.client.stop_streaming()
#         event.accept()
#
#
# class RollingPlot(QtWidgets.QMainWindow):
#     def __init__(self, client, window_secs=5.0, interval_ms=50, ylim=(-1.2, 1.2)):
#         super().__init__()
#         self.client = client
#         self.fs = client.fs
#         self.window_len = int(self.fs * window_secs)
#         self.interval_ms = interval_ms
#         self.ylim = ylim
#
#         self.x = np.linspace(-window_secs, 0, self.window_len, dtype=np.float32)
#         self.buf = np.zeros((client.N_channels, self.window_len), dtype=np.float32)
#
#         w = QtWidgets.QWidget()
#         self.setCentralWidget(w)
#         layout = QtWidgets.QVBoxLayout(w)
#
#         self.plots = []
#         self.curves = []
#         for i, ch in enumerate(self.client.channel_index):
#             pw = pg.PlotWidget()
#             pw.setYRange(*ylim)
#             pw.showGrid(x=True, y=True)
#             if i < client.N_channels - 1:
#                 pw.hideAxis('bottom')
#             pw.setLabel('left', f'Ch {ch}')
#             layout.addWidget(pw)
#             self.plots.append(pw)
#             c = pw.plot(pen=pg.mkPen(pg.intColor(i), width=1))
#             self.curves.append(c)
#
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start(self.interval_ms)
#         self.setWindowTitle("LSL Cosine — Rolling Plot")
#
#     def update_plot(self):
#         t_rel, win = self.client.latest()
#         if win is None or win.size == 0:
#             return
#         for i in range(self.client.N_channels):
#             y = win[i]
#             n = y.shape[0]
#             if n == 0:
#                 continue
#             if n < self.window_len:
#                 self.buf[i, :-n] = self.buf[i, n:]
#                 self.buf[i, -n:] = y
#             else:
#                 self.buf[i] = y[-self.window_len:]
#             self.curves[i].setData(self.x, self.buf[i])


# class StackedPlot(QtWidgets.QMainWindow):
#     def __init__(
#         self,
#         client,
#         window_secs=5.0,
#         ui_hz=20,              # lower is gentler for 128ch
#         auto_ylim=True,
#         robust_pct=(1, 99),    # per-channel robust range
#         min_span=1e-6,
#         smooth_alpha=0.25,
#         symmetric=False,       # True => symmetric about 0 per-channel
#         max_points=2000,       # decimate draw to at most this many points
#     ):
#         super().__init__()
#         self.client = client
#         self.window_secs = float(window_secs)
#         self.auto_ylim = bool(auto_ylim)
#         self.robust_pct = robust_pct
#         self.min_span = float(min_span)
#         self.smooth_alpha = float(smooth_alpha)
#         self.symmetric = bool(symmetric)
#         self.max_points = int(max_points)
#         self.N_channels = self.client.N_channels
#
#         self.setWindowTitle(
#             f"LSL Stacked — {self.client.name}  type={self.client.type}  "
#             f"nom fs={self.client.fs:.1f}Hz  chans={self.N_channels}"
#         )
#         self.resize(1100, 800)
#
#         # ---- central scrollable area with vertical layout ----
#         central = QtWidgets.QWidget()
#         vbox = QtWidgets.QVBoxLayout(central)
#         vbox.setContentsMargins(6, 6, 6, 6)
#         vbox.setSpacing(4)
#
#         self.scroll = QtWidgets.QScrollArea()
#         self.scroll.setWidgetResizable(True)
#         self.inner = QtWidgets.QWidget()
#         self.inner_layout = QtWidgets.QVBoxLayout(self.inner)
#         self.inner_layout.setContentsMargins(0, 0, 0, 0)
#         self.inner_layout.setSpacing(2)
#
#         self.plots, self.curves = [], []
#         self._ylims = [None] * self.N_channels   # per-channel smoothed limits
#
#         self.filter = RealtimeFilter(
#             fs=self.client.fs,
#             n_channels=self.client.N_channels,
#             bp_low=20.0, bp_high=498.0, bp_order=4, enable_bandpass=True,
#             notch_freqs=(60.0,), notch_q=30.0, enable_notch=True,
#             lp_cut=None, lp_order=4, enable_lowpass=False,  # set True and a cutoff if you want
#         )
#
#         for i in range(self.N_channels):
#             pw = pg.PlotWidget()
#             pw.showGrid(x=True, y=True)
#             pw.setXRange(-self.window_secs, 0)
#             if i < self.N_channels - 1:
#                 pw.hideAxis("bottom")
#             pw.setLabel("left", f"Ch {self.client.channel_index[i]}")
#             curve = pw.plot(pen=pg.mkPen(pg.intColor(i, hues=self.N_channels), width=1))
#             self.inner_layout.addWidget(pw)
#             self.plots.append(pw)
#             self.curves.append(curve)
#
#         # link X-axes
#         for i in range(1, self.N_channels):
#             self.plots[i].setXLink(self.plots[0])
#
#         self.scroll.setWidget(self.inner)
#         vbox.addWidget(self.scroll)
#         self.setCentralWidget(central)
#
#         # timers
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_plots)
#         self.timer.start(int(1000 / ui_hz))
#
#         self.fs_timer = QtCore.QTimer()
#         self.fs_timer.timeout.connect(self._update_title_fs)
#         self.fs_timer.start(1000)
#
#     def closeEvent(self, ev):
#         self.client.stop()
#         super().closeEvent(ev)
#
#     def _update_title_fs(self):
#         fs_hat = self.client.fs_estimate()
#         if np.isfinite(fs_hat):
#             self.setWindowTitle(
#                 f"LSL Stacked — {self.client.name}  type={self.client.type}  "
#                 f"nom fs={self.client.fs:.1f}Hz  est fs={fs_hat:.2f}Hz  chans={self.N_channels}"
#             )
#
#     def _robust_limits(self, y):
#         lo, hi = np.nanpercentile(y, self.robust_pct)
#         if not np.isfinite(lo) or not np.isfinite(hi):
#             return None
#         if self.symmetric:
#             m = max(abs(lo), abs(hi))
#             lo, hi = -m, m
#         span = max(self.min_span, hi - lo)
#         pad = 0.10 * span
#         return lo - pad, hi + pad
#
#     def _smooth_set_ylim(self, idx, lo, hi):
#         if self._ylims[idx] is None:
#             self._ylims[idx] = [lo, hi]
#         else:
#             a = self.smooth_alpha
#             self._ylims[idx][0] = (1 - a) * self._ylims[idx][0] + a * lo
#             self._ylims[idx][1] = (1 - a) * self._ylims[idx][1] + a * hi
#         self.plots[idx].setYRange(self._ylims[idx][0], self._ylims[idx][1], padding=0)
#
#     def _maybe_decimate(self, t, y):
#         n = t.size
#         if n <= self.max_points:
#             return t, y
#         step = int(math.ceil(n / self.max_points))
#         return t[::step], y[..., ::step]
#
#     def update_plots(self):
#         t_rel, Y = self.client.latest()
#         if t_rel is None:
#             return
#
#         # decimate for draw speed
#         t_rel, Y = self._maybe_decimate(t_rel, Y)
#
#         Y_filt = self.filter.process(Y)
#
#         for i in range(self.N_channels):
#             yi = Y_filt[i]
#             self.curves[i].setData(t_rel, yi)
#             if self.auto_ylim and yi.size:
#                 lims = self._robust_limits(yi)
#                 if lims is not None:
#                     self._smooth_set_ylim(i, *lims)
