# Joint Angle Regression from EMG

**Comprehensive GUI for collecting synchronized EMG and joint angle data, training regression models, and live comparison.**

Adapted from the MindRove-EMG `new_session_gui.py` to work with Open Ephys ZMQ streaming.

## Features

- 📊 **Visual Flow Diagram**: See your pipeline at a glance
- 🎯 **Guided Prompts**: Follow structured movement protocols
- 📝 **In-GUI Recording**: Synchronized EMG + angle capture with windowing
- 🧪 **EMG Filtering**: Optional highpass, notch, and lowpass filters
- 📡 **IMU Integration**: Optional Sleeve IMU for orientation tracking (RPY)
- 🤖 **Model Training**: Launch training scripts directly from GUI
- 🔴 **Live Comparison**: Real-time prediction vs. ground truth
- 📈 **Comprehensive Logging**: Track all operations

## Overview

This example demonstrates how to:
1. **Stream EMG data** from Open Ephys GUI via ZMQ
2. **Receive joint angles** from a hand tracking system via LSL
3. **Record synchronized data** for regression training
4. **Train regression models** to predict joint angles from EMG

## Prerequisites

### Hardware
- Open Ephys acquisition system with EMG amplifier
- Camera system or hand tracking device (outputting to LSL)

### Software
```bash
# Install python-open-ephys
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys

# Install required packages
pip install numpy PyQt5 pylsl
```

### System Setup
1. **Open Ephys GUI**: Launch with ZMQ Interface plugin enabled
   - Configure your EMG channels (e.g., 8 channels at 5000 Hz)
   - Note the ZMQ port (default: 5556)

2. **Hand Tracking**: Any system that broadcasts joint angles via LSL
   - Examples: MediaPipe hand tracking, finger goniometers, motion capture
   - Stream type: `JointAngles` or custom name
   - Typical output: 5 angles [MCP, PIP, DIP, Thumb_MCP, Thumb_IP]

## Quick Start

### Step 1: Launch Data Collection GUI

```bash
cd python-open-ephys/examples/joint_angle_regression
python new_session_gui.py
```

Or on Windows:
```batch
run_gui.bat
```

### Step 2: Connect to Data Sources

1. **EMG (Open Ephys)**:
   - Set ZMQ Host: `127.0.0.1` (or IP of Open Ephys computer)
   - Set ZMQ Port: `5556` (match Open Ephys ZMQ Interface settings)
   - Set EMG sampling rate: `5000` Hz (or your actual rate)
   - Set number of channels: `8` (or your actual count)
   - Click **"Connect"**
   - Verify status shows "Streaming" (green)

2. **Joint Angles (LSL)**:
   - Click **"Connect LSL"**
   - GUI will search for streams with type `JointAngles`
   - Verify status shows connected stream name (green)

3. **IMU (Optional - Sleeve IMU)**:
   - Check **"Enable Sleeve IMU"** checkbox
   - Set IMU Host: `192.168.4.1` (Sleeve IMU default IP)
   - Set IMU Port: `5555` (default)
   - Select Transport: `UDP` or `TCP`
   - Click **"Connect"** (IMU connects automatically with EMG)
   - Verify IMU status shows orientation data (e.g., "R10.5° P-5.2° Y45.3°")

### Step 3: Record Training Data

1. **Enter metadata**:
   - Subject ID: `P001`
   - Session ID: `S01`
   - Notes: `baseline, relaxed grip`

2. **Record data**:
   - Click **"Start Recording"**
   - Perform hand movements:
     - Open/close hand slowly (10 reps)
     - Individual finger flexion/extension (5 reps each)
     - Grip variations (power, pinch, precision)
     - Natural movements (reaching, grasping objects)
   - Recommended duration: **2-5 minutes**
   - Click **"Stop & Save"**

3. **Output**:
   ```
   data/sub-P001_ses-S01_emg-angles.npz
   ```

### Step 4: Train Regression Model

After collecting data, train a regression model to map EMG → joint angles.

**Using Hand-Landmark-Tracker Pipeline** (recommended):
```bash
# Navigate to Hand-Landmark-Tracker example
cd ../../Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG_OpenEphys

# The GUI saves data in compatible format - use it directly
# See that repository's README for full training pipeline
```

**Using Custom Training Script**:
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load recorded data
data = np.load('data/sub-P001_ses-S01_emg-angles.npz')
emg = data['emg']  # (samples, channels)
angles = data['angles']  # (samples, n_angles)
emg_timestamps = data['emg_timestamps']
angle_timestamps = data['angle_timestamps']

# Align timestamps (LSL synchronization)
# ... implement alignment logic ...

# Extract features from EMG
# ... implement preprocessing (notch, bandpass, envelope) ...

# Train model
model = RandomForestRegressor()
model.fit(features, angles)
```

## GUI Features

### Experiment Panel
- **Subject ID**: Participant identifier
- **Session ID**: Session/condition identifier
- **Notes**: Free-form metadata

### EMG Acquisition Panel
- **Connection Settings**:
  - ZMQ Host: IP address of Open Ephys computer
  - ZMQ Port: ZMQ Interface data port (default 5556)
  - EMG fs: Sampling frequency in Hz
  - Channels: Number of EMG channels to record
- **IMU Settings** (Optional):
  - Enable Sleeve IMU: Checkbox to enable/disable IMU
  - IMU Host: IP address of Sleeve IMU device (default 192.168.4.1)
  - IMU Port: UDP/TCP port (default 5555)
  - Transport: UDP (recommended) or TCP
- **Live Monitoring**:
  - Samples buffered: Current buffer size
  - EMG RMS: Root mean square of signal
  - EMG σ: Standard deviation
  - IMU: Roll/Pitch/Yaw orientation (when enabled)
  - Update rate: GUI refresh rate
- **Controls**:
  - Connect/Disconnect: Manage ZMQ connection (and IMU if enabled)
  - Auto-reconnect: Automatically reconnect if connection drops

### Joint Angle Input Panel
- **LSL Connection**: Searches for streams with type `JointAngles`
- **Status**: Shows connected stream name and rate
- **Compatibility**: Works with any LSL source (hand tracking, goniometers, etc.)

### Recording Panel
- **Output Path**: Auto-generated from subject/session or custom
- **Controls**: Start/stop recording
- **Status**: Shows recording progress and save confirmation

## Data Format

Saved NPZ files contain:

```python
{
    'emg': ndarray, shape (samples, channels)
        # EMG data in microvolts or raw ADC units
    
    'emg_timestamps': ndarray, shape (samples,)
        # LSL timestamps for each EMG sample
    
    'angles': ndarray, shape (samples, n_angles)
        # Joint angles in degrees or radians
    
    'angle_timestamps': ndarray, shape (samples,)
        # LSL timestamps for each angle sample
    
    'imu': ndarray, shape (samples, 9)
        # IMU data: [roll, pitch, yaw, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        # Note: Sleeve IMU only provides RPY, other channels are zeros
        # Synchronized to EMG timestamps
    
    'emg_fs': float
        # EMG sampling frequency (Hz)
    
    'emg_channels': int
        # Number of EMG channels
    
    'subject': str
        # Subject ID
    
    'session': str
        # Session ID
    
    'notes': str
        # Session notes
}
```

### Timestamp Synchronization

Both EMG and angles use **LSL timestamps** (`pylsl.local_clock()`), enabling precise synchronization even across different computers. This is critical for regression training.

## Typical Workflow

### 1. Calibration Session (5-10 minutes)
- Record baseline data with no EMG activity
- Record maximum voluntary contraction (MVC) for each muscle
- Record full range of motion for each joint

### 2. Training Data Collection (15-30 minutes)
- Multiple sessions with varied movements:
  - Session 1: Slow, controlled movements
  - Session 2: Fast, dynamic movements
  - Session 3: Object manipulation tasks
- Merge data from multiple sessions for robust training

### 3. Model Training
- Preprocess EMG (notch filter, bandpass, envelope)
- Extract features (RMS, MAV, waveform properties)
- Train regression model (MLP, Random Forest, or Transformer)
- Validate with held-out test set

### 4. Real-time Prediction
- Use trained model with live ZMQ stream
- See `python-open-ephys/examples/gesture_classifier/3_predict_realtime.py` for template

## Troubleshooting

### "python-oephys missing"
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
pip install numpy zmq
```

### "No LSL stream found"
- Verify hand tracking system is running
- Check that LSL broadcast is enabled
- Try alternative stream names in LSL search
- Use `pylsl` utilities to list available streams:
  ```python
  from pylsl import resolve_streams
  streams = resolve_streams(wait_time=2.0)
  for s in streams:
      print(f"{s.name()}: {s.type()}")
  ```

### EMG signal quality issues
- Check electrode impedance
- Verify channel mapping in Open Ephys
- Adjust gain settings if signal is clipping or too small
- Use EMG RMS/σ display to monitor signal quality

### Timestamp alignment errors
- Ensure both systems use LSL clock (`pylsl.local_clock()`)
- Check for clock drift over long recordings (>10 minutes)
- Verify sampling rates are accurate

## Integration with Hand-Landmark-Tracker

This example is designed to work seamlessly with the [Hand-Landmark-Tracker](https://github.com/Jshulgach/Hand-Landmark-Tracker) repository:

1. **Collect data** using this GUI (`session_gui.py`)
2. **Train models** using Hand-Landmark-Tracker's pipeline:
   ```bash
   cd Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG_OpenEphys
   python oephys_create_dataset.py  # Creates training dataset
   cd ../Joint_Kinematics_from_EMG
   python train_model.py  # Trains PyTorch EMGRegressor
   ```
3. **Real-time prediction** with trained model

See [Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG_OpenEphys/README.md](https://github.com/Jshulgach/Hand-Landmark-Tracker/tree/main/examples/Joint_Kinematics_from_EMG_OpenEphys) for full pipeline documentation.

## Example Use Cases

### 1. Prosthetic Control
- Train regression model to map EMG → desired joint angles
- Use for proportional control of robotic hand
- Real-time performance: <10 ms latency

### 2. Rehabilitation Assessment
- Track recovery of EMG-movement coupling after injury
- Compare affected vs. unaffected limb
- Longitudinal analysis of motor control

### 3. Biomechanics Research
- Study muscle synergies during complex tasks
- Validate musculoskeletal models
- EMG-driven joint angle estimation

## File Structure

```
joint_angle_regression/
├── session_gui.py          # Main data collection GUI
├── run_gui.bat             # Windows launcher
├── data/                   # Recorded datasets (not tracked)
├── models/                 # Trained models (not tracked)
└── README.md               # This file
```

## Citation

If you use this example in your research, please cite:

```bibtex
@software{python_oephys_2024,
  title = {python-open-ephys: Python interface for Open Ephys},
  author = {Neuro-Mechatronics Lab},
  year = {2024},
  url = {https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys}
}
```

## License

MIT License - see python-open-ephys repository root for details.

## Support

- **Issues**: https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/issues
- **Discussions**: https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/discussions
- **Email**: Contact NML team via repository

---

**Author**: Neuro-Mechatronics Lab (NML)  
**Created**: 2026-02-16  
**Updated**: 2026-02-16
