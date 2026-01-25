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
    """Accepts: 'all' | '0,1,2' | '0-7' | '0:7' | repeated tokens ['0','1','2']."""
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


class LSLSubplotViewer(QWidget):
    """
    Multi-axis (subplots) real-time LSL viewer.

    - One subplot per selected channel
    - All X-axes linked (time)
    - Optional visual resampling to smooth motion between chunk arrivals
    """

    def __init__(
        self,
        stream_name=None,
        stream_type="EMG",
        window_s=5.0,
        downsample=1,
        ylim=None,               # (lo, hi) applied to all subplots if provided
        channels=None,           # None = all
        timeout_s=5.0,
        timer_ms=16,             # ~60 fps
        precise_timer=False,
        resample_hz=120.0,       # visual display grid
        clip_to_view=False,
        verbose=False,
    ):
        super().__init__()
        self.setWindowTitle("pyoephys LSL Viewer (subplots)")
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

        # UI
        layout = QVBoxLayout(self)
        self.glw = pg.GraphicsLayoutWidget(show=False)
        self.glw.setBackground("w")
        layout.addWidget(self.glw)

        self.plots = []
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

    def _ensure_subplots(self):
        if not self.client.ready_event.is_set():
            return False
        if self.plots:
            return True

        C = self.client.n_channels or 0
        if C <= 0:
            return False

        # resolve channel indices
        if self.req_channels is None:
            self.idx = list(range(C))
        else:
            self.idx = [i for i in self.req_channels if 0 <= i < C] or [0]

        # build subplots (one per channel)
        self.plots = []
        self.curves = []
        master_plot = None
        palette = [pg.intColor(i, hues=len(self.idx)) for i in range(len(self.idx))]
        for k, ch in enumerate(self.idx):
            p = self.glw.addPlot(row=k, col=0)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", f"ch{ch}")
            if self.fixed_ylim is not None:
                p.setYRange(self.fixed_ylim[0], self.fixed_ylim[1])

            c = p.plot([], [], pen=pg.mkPen(palette[k], width=1.0), name=f"ch{ch}")
            if self.clip_to_view:
                c.setClipToView(True)
            c.setDownsampling(auto=True, method="peak")

            self.plots.append(p)
            self.curves.append(c)

            if master_plot is None:
                master_plot = p
            else:
                p.setXLink(master_plot)

        # nicer spacing
        self.glw.ci.layout.setRowStretchFactor(len(self.idx)-1, 1)
        return True

    def _resample_for_display(self, t, y):
        """Resample (for plotting only) to a uniform grid."""
        if self.resample_hz is None or t.size < 4:
            return t, y
        t0 = max(t[0], t[-1] - self.window_s)
        n_pts = max(64, int(self.window_s * self.resample_hz))
        t_uniform = np.linspace(t0, t[-1], n_pts, dtype=np.float64)
        y_out = np.empty((y.shape[0], n_pts), dtype=y.dtype)
        for i in range(y.shape[0]):
            y_out[i] = np.interp(t_uniform, t, y[i])
        return t_uniform, y_out

    def _on_timer(self):
        if not self._ensure_subplots():
            return
        try:
            y, t = self.client.get_window(self.window_s)
        except NotReadyError:
            return
        if y.size == 0:
            return

        if self.downsample > 1:
            y = y[:, :: self.downsample]
            t = t[:: self.downsample]

        t_disp, y_disp = self._resample_for_display(t, y)

        # update each subplot
        for k, ch in enumerate(self.idx):
            self.curves[k].setData(t_disp, y_disp[ch])

        # keep all X ranges aligned tightly to window
        x0, x1 = t_disp[0], t_disp[-1]
        for p in self.plots:
            p.setXRange(x0, x1, padding=0.02)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-time LSL subplots viewer.")
    ap.add_argument("--stream_name", type=str, default=None)
    ap.add_argument("--stream_type", type=str, default="EMG")
    ap.add_argument("--channels", nargs="+", default=None,
                    help='Channels: "all" | "0,1,2" | "0-7" | "0:7" | multiple tokens.')
    ap.add_argument("--window_s", type=float, default=5.0)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="Fixed y-limits (applied to all subplots)")
    ap.add_argument("--timeout_s", type=float, default=5.0)
    ap.add_argument("--timer_ms", type=int, default=16, help="UI refresh period (ms)")
    ap.add_argument("--precise_timer", action="store_true")
    ap.add_argument("--resample_hz", type=float, default=120.0,
                    help="Visual resample rate (Hz) for smooth motion")
    ap.add_argument("--opengl", action="store_true", help="Try OpenGL acceleration")
    ap.add_argument("--clip_to_view", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    configure("INFO" if args.verbose else "WARNING")
    if args.opengl:
        # Optional; may require PyOpenGL on some machines
        pg.setConfigOptions(useOpenGL=True)

    chans = _parse_channels_arg(args.channels)

    app = QApplication(sys.argv)
    w = LSLSubplotViewer(
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
        verbose=args.verbose,
    )
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
