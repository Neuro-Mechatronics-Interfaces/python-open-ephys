#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# csv_plotter_gui.py — Interactive CSV plotter for EMG + IMU

import sys, os
from typing import List
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters as exporters


def is_emg_col(name: str) -> bool:
    return name.lower().startswith("ch")


def is_imu_col(name: str) -> bool:
    ln = name.lower()
    return ("roll" in ln) or ("pitch" in ln) or ("yaw" in ln)


class CSVPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plotter — EMG + IMU")
        self.resize(1250, 750)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        left = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(left)

        right = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self.btnLoad = QtWidgets.QPushButton("Load CSV…")
        self.btnLoad.clicked.connect(self.on_load_csv)

        self.filterEdit = QtWidgets.QLineEdit()
        self.filterEdit.setPlaceholderText("Filter columns (e.g., ch12, roll)")
        self.filterEdit.textChanged.connect(self.apply_filter)

        self.listCols = QtWidgets.QListWidget()
        self.listCols.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listCols.itemSelectionChanged.connect(self.update_plot)

        row_buttons = QtWidgets.QHBoxLayout()
        self.btnAll = QtWidgets.QPushButton("Select All")
        self.btnNone = QtWidgets.QPushButton("Select None")
        self.btnEMG = QtWidgets.QPushButton("EMG Only")
        self.btnIMU = QtWidgets.QPushButton("IMU Only")
        for b in (self.btnAll, self.btnNone, self.btnEMG, self.btnIMU):
            row_buttons.addWidget(b)
        self.btnAll.clicked.connect(lambda: self.select_by(lambda _: True))
        self.btnNone.clicked.connect(lambda: self.select_by(lambda _: False))
        self.btnEMG.clicked.connect(lambda: self.select_by(is_emg_col))
        self.btnIMU.clicked.connect(lambda: self.select_by(is_imu_col))

        self.chkLegend = QtWidgets.QCheckBox("Show legend"); self.chkLegend.setChecked(True)
        self.chkLegend.stateChanged.connect(self.update_plot)
        self.chkNormalize = QtWidgets.QCheckBox("Normalize (z-score)"); self.chkNormalize.stateChanged.connect(self.update_plot)
        self.chkStack = QtWidgets.QCheckBox("Stack EMG"); self.chkStack.stateChanged.connect(self.update_plot)
        self.chkOpenGL = QtWidgets.QCheckBox("OpenGL"); self.chkOpenGL.setChecked(True); self.chkOpenGL.stateChanged.connect(self.toggle_opengl)

        h_stack = QtWidgets.QHBoxLayout()
        h_stack.addWidget(QtWidgets.QLabel("Stack offset:"))
        self.spinOffset = QtWidgets.QDoubleSpinBox()
        self.spinOffset.setRange(0.0, 1e9); self.spinOffset.setDecimals(3); self.spinOffset.setValue(100.0)
        self.spinOffset.valueChanged.connect(self.update_plot)
        h_stack.addWidget(self.spinOffset); h_stack.addStretch(1)

        self.btnExport = QtWidgets.QPushButton("Export PNG…")
        self.btnExport.clicked.connect(self.export_png)

        left_layout.addWidget(self.btnLoad)
        left_layout.addWidget(self.filterEdit)
        left_layout.addWidget(self.listCols, 1)
        left_layout.addLayout(row_buttons)
        left_layout.addSpacing(8)
        left_layout.addWidget(self.chkLegend)
        left_layout.addWidget(self.chkNormalize)
        left_layout.addWidget(self.chkStack)
        left_layout.addLayout(h_stack)
        left_layout.addWidget(self.chkOpenGL)
        left_layout.addStretch(1)
        left_layout.addWidget(self.btnExport)

        pg.setConfigOptions(antialias=True, useOpenGL=self.chkOpenGL.isChecked())
        self.plot = pg.PlotWidget(background="k")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "time", units="s")
        right_layout.addWidget(self.plot, 1)

        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setVisible(self.chkLegend.isChecked())

        self.df = pd.DataFrame()
        self.xname = "t_s"
        self.plot_items = []

        actOpen = QtWidgets.QAction("Open…", self, shortcut="Ctrl+O")
        actOpen.triggered.connect(self.on_load_csv)
        self.addAction(actOpen)

    def toggle_opengl(self):
        pg.setConfigOptions(useOpenGL=self.chkOpenGL.isChecked())
        self.update_plot()

    def select_by(self, fn):
        for i in range(self.listCols.count()):
            item = self.listCols.item(i)
            item.setSelected(bool(fn(item.text())))

    def apply_filter(self):
        text = self.filterEdit.text().strip().lower()
        for i in range(self.listCols.count()):
            item = self.listCols.item(i)
            item.setHidden(False if not text else (text not in item.text().lower()))

    def on_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV files (*.csv);;All files (*)")
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return
        cand = [c for c in df.columns if c.lower() in ("t", "time", "timestamp", "ts", "t_s")]
        self.xname = cand[0] if cand else df.columns[0]
        if self.xname not in df.columns:
            QtWidgets.QMessageBox.critical(self, "Error", "No timestamp column found."); return
        for c in df.columns:
            if c == self.xname: continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        self.df = df
        self.populate_columns()
        self.update_plot()
        self.setWindowTitle(f"CSV Plotter — {os.path.basename(path)}")

    def populate_columns(self):
        self.listCols.clear()
        if self.df.empty: return
        for c in self.df.columns:
            if c == self.xname: continue
            it = QtWidgets.QListWidgetItem(c)
            if is_imu_col(c): it.setSelected(True)
            self.listCols.addItem(it)

    def clear_plot(self):
        for it in self.plot_items:
            try: self.plot.removeItem(it)
            except Exception: pass
        self.plot_items = []
        try: self.plot.removeItem(self.legend)
        except Exception: pass
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setVisible(self.chkLegend.isChecked())

    def update_plot(self):
        self.clear_plot()
        if self.df.empty: return
        cols = [it.text() for it in self.listCols.selectedItems()]
        if not cols: return
        x = self.df[self.xname].to_numpy(dtype=float)
        normalize = self.chkNormalize.isChecked()
        stack = self.chkStack.isChecked()
        offset = float(self.spinOffset.value()) if stack else 0.0
        base = 0.0
        for idx, col in enumerate(cols):
            y = self.df[col].to_numpy(dtype=float)
            if normalize:
                mu = np.nanmean(y); sd = np.nanstd(y) or 1.0
                y = (y - mu) / sd
            yoff = 0.0
            if stack and is_emg_col(col):
                yoff = base; base += offset
            curve = self.plot.plot(x, y + yoff, pen=pg.intColor(idx, hues=len(cols)))
            # Downsample for speed
            try:
                curve.setDownsampling(method='peak')
            except Exception:
                pass
            self.plot_items.append(curve)
            if self.chkLegend.isChecked():
                self.legend.addItem(curve, col)
        self.plot.enableAutoRange()

    def export_png(self):
        if self.df.empty:
            QtWidgets.QMessageBox.information(self, "Export", "Load a CSV first."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PNG", "plot.png", "PNG (*.png)")
        if not path: return
        exp = exporters.ImageExporter(self.plot.plotItem)
        exp.params['width'] = 1920
        exp.export(path)
        QtWidgets.QMessageBox.information(self, "Export", f"Saved: {path}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CSVPlotter()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
