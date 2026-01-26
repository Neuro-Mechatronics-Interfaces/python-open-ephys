"""
Real-Time EMG Viewer for Open Ephys
-----------------------------------
Connects to Open Ephys GUI via ZMQ and displays live EMG signals.
Supports Real-Time Inference with PyTorch models.
"""

import sys
import argparse
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QCheckBox, QFileDialog, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt
import torch
from pyoephys.ml import EMGClassifierCNNLSTM

from pyoephys.interface import ZMQClient, NotReadyError
from pyoephys.processing import ChannelQC

class RealTimeEMGViewer(QMainWindow):
    def __init__(self, host="127.0.0.1", port=5556, channels=None):
        super().__init__()
        self.setWindowTitle("Open Ephys Real-Time EMG Viewer")
        self.resize(1200, 800)

        # Config
        self.host = host
        self.port = port
        self.selected_channels = channels or list(range(8)) # Default to first 8
        self.window_size_ms = 500
        
        # ZMQ Client
        self.client = ZMQClient(
            host_ip=self.host, 
            data_port=str(self.port), 
            auto_start=False
        )
        self.is_connected = False
        
        # Channel QC
        self.qc = ChannelQC(fs=2000) # fs will update on connect
        
        # ML State
        self.model = None
        self.ml_window_size = 200 # ms
        self.ml_buffer = [] 
        # For now, let's assume 4 classes, update when model loaded
        self.classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3'] 
        
        self.init_ui()
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.update_rate = 30 # Hz
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Controls
        controls = QHBoxLayout()
        
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_connection)
        controls.addWidget(self.btn_connect)
        
        self.status_label = QLabel("Status: Disconnected")
        controls.addWidget(self.status_label)
        
        # ML Controls
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        controls.addWidget(self.btn_load_model)
        
        self.lbl_prediction = QLabel("Prediction: N/A")
        self.lbl_prediction.setStyleSheet("font-weight: bold; font-size: 14px;")
        controls.addWidget(self.lbl_prediction)

        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Plot
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=3)
        
        self.plots = []
        self.curves = []
        
        # Create plots for selected channels
        for i, ch_idx in enumerate(self.selected_channels):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.setLabel('left', f"CH {ch_idx+1}")
            p.showGrid(y=True)
            if i < len(self.selected_channels) - 1:
                p.hideAxis('bottom')
            else:
                p.setLabel('bottom', "Time (s)")
            
            curve = p.plot(pen=pg.mkPen('y', width=1))
            self.plots.append(p)
            self.curves.append(curve)
            
            # Link X axes
            if i > 0:
                p.setXLink(self.plots[0])

        # ML Probability Bars
        self.prob_widget = pg.PlotWidget(title="Class Probabilities")
        self.prob_widget.setYRange(0, 1)
        # Use simple integer x-axis; we'll add labels via axis if possible, or just tooltip
        self.prob_widget.setLabel('bottom', "Class Index")
        self.bars = pg.BarGraphItem(x=range(4), height=[0]*4, width=0.6, brush='b')
        self.prob_widget.addItem(self.bars)
        self.prob_widget.hide() 
        layout.addWidget(self.prob_widget, stretch=1)

    def toggle_connection(self):
        if not self.is_connected:
            try:
                self.client.start()
                self.is_connected = True
                self.btn_connect.setText("Disconnect")
                self.status_label.setText("Status: Connecting...")
                self.timer.start(int(1000/self.update_rate))
                print(f"Connecting to {self.host}:{self.port}...")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        else:
            self.client.stop()
            self.is_connected = False
            self.btn_connect.setText("Connect")
            self.status_label.setText("Status: Disconnected")
            self.timer.stop()
            
    def load_model(self):
        """Load a PyTorch model from file."""
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth *.pt);;All Files (*)", options=options)
        if fileName:
            try:
                # We need to know model parameters to instantiate.
                # For this demo, we'll try to load standard parameters or infer them.
                # Ideally, metadata is saved with the model.
                # Assuming standard demo model for now: 8 channels, 4 classes.
                
                # Check for metadata sidecar or assume defaults
                n_channels = len(self.selected_channels)
                n_classes = 4 # Default
                
                self.model = EMGClassifierCNNLSTM(
                    num_classes=n_classes,
                    num_channels=n_channels,
                    input_window=200 # Default
                )
                self.model.load(fileName)
                self.status_label.setText(f"Status: Model Loaded ({fileName})")
                self.prob_widget.show()
                print(f"Loaded model: {fileName}")
            except Exception as e:
                self.status_label.setText(f"Error loading model: {e}")
                print(f"Error loading model: {e}")

    def update_plot(self):
        if not self.is_connected:
            return
            
        try:
            # Check if ready
            if not self.client.ready_event.is_set():
                self.status_label.setText("Status: Waiting for data stream...")
                return
            
            self.status_label.setText("Status: Connected (Streaming)")
            
            # Configure client channels if needed
            if not self.client.channel_index:
                 try:
                     self.client.set_channel_index(self.selected_channels)
                 except ValueError:
                     return

            # Get data for visualization
            data, t_vis = self.client.get_latest_window(window_ms=self.window_size_ms)
            
            if data is None or t_vis is None:
                return
                
            # Update plots
            for i in range(min(len(self.plots), data.shape[0])):
                self.curves[i].setData(t_vis, data[i])
                
            # Inference Logic
            if self.model:
                # Get data for ML (usually smaller window)
                # Ensure input window size matches model expectation
                # self.ml_window_size
                
                # We can reuse 'data' if window size matches, or fetch specifically
                # Fetching specifically is safer for correct window size
                
                ml_data, _ = self.client.get_latest_window(window_ms=self.ml_window_size)
                
                if ml_data is not None and ml_data.shape[1] >= (self.ml_window_size * self.client.fs / 1000 * 0.9):
                     # Reshape for model: (1, 1, Channels, Time)
                     # Check samples count
                     n_samples_req = int(self.ml_window_size * self.client.fs / 1000)
                     
                     if ml_data.shape[1] >= n_samples_req:
                         # Take exactly n_samples_req form the end
                         X = ml_data[:, -n_samples_req:]
                         
                         # (C, T) -> (1, 1, C, T)
                         X_tensor = torch.tensor(X[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
                         
                         # Inference
                         with torch.no_grad():
                             # Predict returns class indices, but we want probabilities for visualization
                             # The model class has .predict() which returns indices.
                             # Let's call forward directly for logits -> softmax
                             self.model.eval()
                             logits = self.model(X_tensor)
                             probs = torch.softmax(logits, dim=1).numpy()[0]
                             
                             # Update UI
                             pred_idx = np.argmax(probs)
                             confidence = probs[pred_idx]
                             self.lbl_prediction.setText(f"Prediction: Class {pred_idx} ({confidence:.1%})")
                             
                             # Update Bar Chart
                             self.bars.setOpts(height=probs)
                
        except NotReadyError:
            pass
        except Exception as e:
            print(f"Update error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real-Time EMG Viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys GUI IP")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ Data Port")
    parser.add_argument("--channels", type=int, nargs="+", default=[0,1,2,3,4,5,6,7], help="Channels to plot (0-indexed)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = RealTimeEMGViewer(host=args.host, port=args.port, channels=args.channels)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
