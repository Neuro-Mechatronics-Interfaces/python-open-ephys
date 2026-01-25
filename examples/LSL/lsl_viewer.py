#!/usr/bin/env python3
import sys
import argparse
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
import pyqtgraph as pg

from pyoephys.interface._lsl_client import LSLClient, NotReadyError
from pyoephys.io import parse_numeric_args
from pyoephys.logging import configure

def _parse_channels_arg(tokens):
    if tokens is None:
        return None
    if isinstance(tokens, str):
        s = tokens.strip().lower()
        if s == "all":
            return None
        s = s.replace(":", "-")
        return parse_numeric_args(s)
    if len(tokens) == 1:
        t0 = tokens[0].strip().lower()
        if t0 == "all":
            return None
        return parse_numeric_args(t0.replace(":", "-"))
    return parse_numeric_args([int(x) for x in tokens])

def _robust_span(y):
    """Median ± k*MAD for a robust amplitude estimate."""
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))  # ~std for Gaussian
    return max(1e-6, 4.0 * mad)  # 4*MAD ≈ ~97% span

class LSLViewer(QWidget):
    def __init__(
        self,
        stream_name=None,
        stream_type="EMG",
        window_s=5.0,
        downsample=1,
        ylim=None,
        channels=None,
        timeout_s=5.0,
        timer_ms=16,
        precise_timer=False,
        resample_hz=120.0,   # visual grid Hz
        clip_to_view=False,
        stacked=False,
        stack_spacing=None,  # float (units of signal). None = auto
        verbose=False,
    ):
        super().__init__()
        self.setWindowTitle("pyoephys LSL Viewer")
        self.client = LSLClient(
            stream_name=stream_name,
            stream_type=stream_type or "EMG",
            timeout_s=timeout_s,
            buffer_seconds=max(window_s * 2.5, 30.0),
            verbose=verbose,
        )
        self.window_s = float(window_s)
        self.downsample = max(1, int(downsample))
        self.fixed_ylim = ylim
        self.req_channels = channels
        self.idx = None
        self.resample_hz = float(resample_hz) if resample_hz else None
        self.clip_to_view = bool(clip_to_view)
        self.stacked = bool(stacked)
        self.stack_spacing = stack_spacing  # None => auto per frame

        # UI
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.addLegend(offset=(10, 10))
        if self.fixed_ylim is not None and not self.stacked:
            self.plot.setYRange(self.fixed_ylim[0], self.fixed_ylim[1])
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        self.curves = []

        # start client thread
        self.client.start()

        # timer
        self.timer = pg.QtCore.QTimer(self)
        if precise_timer:
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(int(timer_ms))

    def closeEvent(self, e):
        try:
            self.timer.stop()
        except Exception:
            pass
        self.client.stop()
        return super().closeEvent(e)

    def _ensure_curves(self):
        if not self.client.ready_event.is_set():
            return False
        if self.curves:
            return True

        C = self.client.n_channels or 0
        if C <= 0:
            return False

        # resolve channel indices
        if self.req_channels is None:
            self.idx = list(range(C))
        else:
            self.idx = [i for i in self.req_channels if 0 <= i < C] or [0]

        palette = [pg.intColor(i, hues=len(self.idx)) for i in range(len(self.idx))]
        for k, ch in enumerate(self.idx):
            item = self.plot.plot([], [], pen=pg.mkPen(palette[k], width=1.0),
                                  name=f"ch{ch}")
            if self.clip_to_view:
                item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
            self.curves.append(item)

        # In stacked mode, let Y auto-range; we’ll offset traces ourselves.
        if self.fixed_ylim is not None and not self.stacked:
            self.plot.setYRange(self.fixed_ylim[0], self.fixed_ylim[1])

        return True

    def _resample_for_display(self, t, y):
        if self.resample_hz is None or t.size < 4:
            return t, y
        t0 = max(t[0], t[-1] - self.window_s)
        n_pts = max(64, int(self.window_s * self.resample_hz))
        t_uniform = np.linspace(t0, t[-1], n_pts, dtype=np.float64)
        y_out = np.empty((y.shape[0], n_pts), dtype=y.dtype)
        for i in range(y.shape[0]):
            y_out[i] = np.interp(t_uniform, t, y[i])
        return t_uniform, y_out

    def _apply_stacking(self, y_disp):
        """Return y_disp with vertical offsets per channel."""
        n = y_disp.shape[0]
        if n <= 1:
            return y_disp, 0.0

        if self.stack_spacing is None:
            # auto spacing based on robust span of all channels in view
            span = max(_robust_span(y_disp[i]) for i in range(n))
            spacing = 1.25 * span  # small gap
        else:
            spacing = float(self.stack_spacing)

        y_off = np.empty_like(y_disp)
        for i in range(n):
            y_off[i] = y_disp[i] + (n - 1 - i) * spacing  # top-down stacking
        return y_off, spacing

    def _on_timer(self):
        if not self._ensure_curves():
            return
        try:
            y, t = self.client.get_window(self.window_s)
        except NotReadyError:
            return
        if y.size == 0:
            return

        # optional decimation
        if self.downsample > 1:
            y = y[:, :: self.downsample]
            t = t[:: self.downsample]

        # resample for smooth visual motion
        t_disp, y_disp = self._resample_for_display(t, y)

        # stacked vs overlay
        if self.stacked and y_disp.shape[0] > 1:
            y_disp, spacing = self._apply_stacking(y_disp)
            # annotate left axis with stack step
            self.plot.setLabel("left", f"stacked (step={spacing:.3g})")
        else:
            self.plot.setLabel("left", "")

        # update curves
        for k, ch in enumerate(self.idx):
            self.curves[k].setData(t_disp, y_disp[k])

        self.plot.setXRange(t_disp[0], t_disp[-1], padding=0.02)

def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-time LSL plot (overlay or stacked).")
    ap.add_argument("--stream_name", type=str, default=None)
    ap.add_argument("--stream_type", type=str, default="EMG")
    ap.add_argument("--channels", nargs="+", default=None,
                    help='Channels: "all" | "0,1,2" | "0-7" | "0:7" | multiple tokens.')
    ap.add_argument("--window_s", type=float, default=5.0)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="Ignored in stacked mode")
    ap.add_argument("--timeout_s", type=float, default=5.0)
    ap.add_argument("--timer_ms", type=int, default=16)
    ap.add_argument("--precise_timer", action="store_true")
    ap.add_argument("--resample_hz", type=float, default=120.0)
    ap.add_argument("--opengl", action="store_true")
    ap.add_argument("--clip_to_view", action="store_true")
    ap.add_argument("--stacked", action="store_true", help="Draw channels with vertical offsets")
    ap.add_argument("--stack_spacing", type=float, default=None, help="Fixed spacing (signal units) between stacked traces")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    configure("INFO" if args.verbose else "WARNING")
    if args.opengl:
        pg.setConfigOptions(useOpenGL=True)

    chans = _parse_channels_arg(args.channels)

    app = QApplication(sys.argv)
    w = LSLViewer(
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        window_s=args.window_s,
        downsample=args.downsample,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        channels=chans,
        timeout_s=args.timeout_s,
        timer_ms=args.timer_ms,
        precise_timer=args.precise_timer,
        resample_hz=args.resample_hz,
        clip_to_view=args.clip_to_view,
        stacked=args.stacked,
        stack_spacing=args.stack_spacing,
        verbose=args.verbose,
    )
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
