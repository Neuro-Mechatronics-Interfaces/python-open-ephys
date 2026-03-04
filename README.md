![Logo](https://raw.githubusercontent.com/Neuro-Mechatronics-Interfaces/python-open-ephys/main/docs/figs/logo.jpg)

# Python OEphys

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://neuro-mechatronics-interfaces.github.io/python-open-ephys/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/python-oephys.svg)](https://badge.fury.io/py/python-oephys)
[![Downloads](https://pepy.tech/badge/python-oephys)](https://pepy.tech/project/python-oephys)

**python-oephys** is a comprehensive Python toolkit for working with Open Ephys devices and electrophysiology data. From file loading to real-time ZMQ streaming, signal processing to machine learning, and visualization tools—everything you need for high-density neural data analysis in one package.

---

## ✨ Key Features

- 📁 **File I/O**: Robust support for Open Ephys Binary (`.oebin`) and `.npz` formats
- 🔴 **Real-time Streaming**: Seamless integration with the Open Ephys GUI via ZMQ
- 🎛️ **Signal Processing**: Filtering (Bandpass, Notch), Channel QC, and synchronization
- 🤖 **Machine Learning**: Hybrid CNN-LSTM models for real-time gesture recognition
- 📊 **Visualization**: Real-time EMG viewer, offline analysis, and trial segmentation tools
- 🚀 **Performance**: Optimized for low-latency real-time applications

---

## 📦 Installation

### From TestPyPI (Current Development Release)

```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
```

### From Source

```bash
git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git
cd python-open-ephys
pip install -e .
```

### Optional Extras

- **GUI**: `pip install 'python-oephys[gui]'` (PyQt5, pyqtgraph)
- **ML**: `pip install 'python-oephys[ml]'` (PyTorch, scikit-learn)
- **Docs**: `pip install 'python-oephys[docs]'` (Sphinx)

---

## 🚀 Getting Started

### Load and Filter Data

```python
from pyoephys.io import load_open_ephys_session
from pyoephys.processing import filter_emg

# Load session
sess = load_open_ephys_session('path/to/recording.oebin')
data = sess['amplifier_data']
fs = sess['sample_rate']

# Apply filters
filtered = filter_emg(data, filter_type='bandpass', lowcut=10, highcut=500, fs=fs)
```

### File Conversion (Open Ephys/XDF -> MAT/NPZ)

You can convert recording sessions to `.mat` or `.npz` format using the interactive CLI tool:

```bash
# Run the interactive wizard
python examples/read_files/convert_open_ephys_session.py

# Or use command-line arguments
python examples/read_files/convert_open_ephys_session.py --input path/to/session.oebin --format mat
```

Or programmatically:

```python
from pyoephys.io import convert_session_api

convert_session_api(
    input_path='path/to/session.oebin',
    output_path='converted_data.mat',
    format='mat'
)
```

### Real-time ZMQ Streaming

```bash
# Launch the live viewer (ensure ZMQ Interface plugin is active in GUI)
python -m pyoephys.applications._realtime_viewer --host 127.0.0.1 --channels 0:8
```

---

## 🗂️ Package Structure

```text
pyoephys/
├── applications/     # GUI applications (Real-time & Offline viewers)
├── interface/        # ZMQ, LSL, and playback clients
├── io/               # Unified file loaders (.oebin, .npz)
├── ml/               # Gesture classification (CNN-LSTM)
├── plotting/         # Visualization utilities
└── processing/       # Signal filters, QC, and synchronization

examples/
├── benchmarks/       # Throughput and latency tests
├── interface/        # LSL, ZMQ, and hardware control
│   ├── hardware/     # Serial/UDP Pico integration
│   ├── imu/          # Sleeve IMU client & monitor
│   ├── lsl/          # LSL streaming & capture
│   └── zmq/          # ZMQ clients & plotters
├── applications/     # Standalone GUIs (EMG viewer, joint-angle regression)
└── machine_learning/ # Model training and evaluation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by the Neuromechatronics Lab
</p>
