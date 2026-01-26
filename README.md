# Python OEphys

A comprehensive Python toolkit for working with Open Ephys devices, featuring signal processing, machine learning, and real-time visualization tools. This package seamlessly integrates with the Open Ephys GUI via ZMQ.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

### 🖥️ Applications (GUI)
-   **Real-Time EMG Viewer**: Live ZMQ streaming plot with signal quality checks (`channels` loaded dynamically).
-   **Offline EMG Viewer**: Advanced offline analysis of `.npz` recordings with filtering and spectrograms.
-   **Trial Selector**: Tool for manual segmentation and labeling of EMG trials.

### 🧠 Machine Learning
-   **EMGClassifierCNNLSTM**: Hybrid CNN-LSTM model for spatio-temporal gesture recognition.
-   **Model Manager**: Utilities to save, load, and manage PyTorch models.
-   **Evaluation**: Metrics and tools for evaluating model performance.

### 📡 Signal Processing
-   **Channel QC**: Automated signal quality assessment (noise, saturation).
-   **Synchronization**: Tools to align multi-modal data (e.g., EMG + Motion Capture/Video).
-   **Filtering**: Real-time and offline filters (Bandpass, Notch, Smoothing).
-   **Features**: Extract RMS, MAV, Zero Crossings, and IMU features.

### 💾 Data & IO
-   **Unified IO**: Robust support for Open Ephys Binary (`.oebin`) and `.npz` formats.
-   **Dataset Builder**: Advanced tools to discover, group, and merge multi-session datasets.
-   **Spatial Mapping**: Grid reorientation tools (`rot90`, `flipH`, etc.) for high-density arrays.

## Installation

### From Source
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git
    cd python-open-ephys
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -e .
    ```

### Dependencies
-   `numpy`, `scipy`, `matplotlib`, `pandas`
-   `torch` (for ML models)
-   `pyqt5`, `pyqtgraph` (for GUIs)
-   `pyzmq` (for streaming)
-   `open-ephys-python-tools`

## Usage

### 1. Real-Time Viewer
Launch the real-time plotter to visualize live data from Open Ephys:
```bash
python -m pyoephys.applications._realtime_viewer --host 127.0.0.1 --channels 0 1 2 3
```
*Note: Ensure the ZMQ Interface plugin is active in Open Ephys GUI.*

### 2. Machine Learning
Train a CNN-LSTM model for gesture recognition:
```python
from pyoephys.ml import EMGClassifierCNNLSTM
import torch

# Initialize model (4 classes, 8 channels, 200ms window)
model = EMGClassifierCNNLSTM(num_classes=4, num_channels=8, input_window=200)

# Train (see examples/ml/train_cnn_lstm.py for full loop)
# model.fit(X_train, y_train)
```

### 3. Signal Processing
Check signal quality of your recording:
```python
from pyoephys.processing import ChannelQC

qc = ChannelQC(fs=2000)
results = qc.compute_qc(data_chunk)
print(results) # Status (Good/Bad) per channel
```

### 4. Unified CLI Tools
Build a dataset from multiple sessions and train/predict with a single command workflow:
```bash
# 1. Build Dataset (Auto-discovery, Preprocessing, Merging)
python examples/gesture_classifier/1_build_dataset.py --root_dir ./data --multi_file --paper_style

# 2. Train Model
python examples/gesture_classifier/2_train_model.py --root_dir ./data --label my_model

# 3. Predict (Offline or Real-time Stream)
python examples/gesture_classifier/predict.py stream --root_dir ./data --label my_model
```

## Examples
Check the `examples/` directory for complete scripts:
-   `examples/gesture_classifier/2_train_model.py`: Train a gesture classifier.
-   `examples/synchronization/sync_multimodal_data.py`: Align EMG with 3D hand landmarks.
-   `examples/analysis/run_channel_qc.py`: Run quality control checks.
-   `examples/interface/zmq_client.py`: Real-time ZMQ client example.
-   `examples/read_files/example_load_oebin_file.py`: Load Open Ephys data.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Development & Release

### Versioning
This project uses **dynamic versioning** via `setuptools_scm`. The version is automatically derived from Git tags.

### How to Release
1.  **Commit** all changes.
2.  **Tag** the commit with the new version number:
    ```bash
    git tag -a v0.1.0 -m "Release v0.1.0"
    ```
3.  **Push** the tag to GitHub:
    ```bash
    git push origin v0.1.0
    ```
4.  **Create a Release** on GitHub:
    - Go to **Releases** > **Draft a new release**.
    - Select the tag `v0.1.0`.
    - Click **Publish release**.
5.  The `publish.yml` workflow will trigger, build the package, and upload to PyPI.
