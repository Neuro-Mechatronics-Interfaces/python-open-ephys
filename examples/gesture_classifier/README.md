# EMG Gesture Classifier — Open Ephys Example

A complete, self-contained pipeline for training and running an EMG gesture
classifier from Open Ephys recordings.

---

## Quick Start

```bash
# 0. Install the package (from repo root)
pip install -e .

# 1. Build a windowed feature dataset from your recording
python 1_build_dataset.py

# 2. Train the CNN-LSTM gesture classifier
python 2_train_model.py

# 3a. Evaluate on the recorded data (offline)
python predict.py

# 3b. Run real-time prediction from a live Open Ephys stream
python 3_predict_realtime.py
```

All scripts default to `data/gestures/` — no arguments required if your recording is in that folder.

---

## Pipeline

```
  data/gestures/                ← Open Ephys binary recording
  data/gestures/emg.txt         ← gesture event labels (auto-discovered)

        │
        ▼
1_build_dataset.py
        │
        ▼
  data/gestures/training_dataset.npz  ← windowed features + labels

        │
        ▼
2_train_model.py
        │
        ▼
  data/model/                   ← trained model + scaler + metadata.json

        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
predict.py                                  3_predict_realtime.py
(offline evaluation on a file)              (live ZMQ stream from Open Ephys GUI)
```

---

## File Overview

| File | Purpose |
|------|---------|
| `1_build_dataset.py` | Segments recordings into windowed feature matrices |
| `2_train_model.py` | Trains a CNN-LSTM classifier and saves the model |
| `predict.py` | Offline prediction + accuracy evaluation on a recording |
| `3_predict_realtime.py` | Real-time prediction over a ZMQ stream from Open Ephys GUI |
| `.gesture_config.example` | Template config file (copy to `.gesture_config`) |
| `data/gestures/` | Open Ephys binary recording folder |
| `data/gestures/emg.txt` | Gesture event labels (transition-style, auto-discovered) |

---

## Data Formats

### Labels file (`emg.txt` / `labels.csv`)

Transition-style events — each row marks where a new gesture **begins**:

```
Sample Index,Label
0,rest
400,fist
800,rest
1200,open_hand
...
```

- `Sample Index` — zero-based sample number in the recording  
- `Label` — gesture name (case-sensitive)  

The labelling scheme is *transition-style*: a label applies from its sample
index until the next row's sample index.

---

## Parameters

All parameters can be set via CLI arguments or in a `.gesture_config` file
(copy `.gesture_config.example` → `.gesture_config` and edit):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `data/gestures/` | Path to EMG recording |
| `labels_path` | auto-detected | Path to labels/events CSV |
| `model_dir` | `./data/gesture_model` | Model save/load directory |
| `dataset_path` | `data/gestures/training_dataset.npz` | Feature dataset output |
| `window_ms` | `200` | Analysis window length (ms) |
| `step_ms` | `50` | Step between windows (ms) |
| `kfold` | `true` | K-fold cross-validation during training |
| `host` | `127.0.0.1` | Open Ephys host IP (real-time only) |
| `port` | `5556` | ZMQ data port (real-time only) |
| `smooth_k` | `5` | Prediction smoothing window length |
| `auto_select_channels` | `false` | Drop bad channels detected by QC automatically |
| `noise_threshold` | `30` | QC noise ceiling (µV RMS during rest); channels above this are flagged |
| `qc_quiet_sec` | auto | Explicit quiet window for QC, e.g. `0 10` for first 10 s |

---

## Channel Quality Control

`1_build_dataset.py` runs a per-channel QC check after loading data.  It
applies a 10–500 Hz bandpass filter + 60 Hz notch, then computes per-channel
RMS on the **quietest portion of the recording** (rest-like segments where
good channels should be near-zero).  Two thresholds are applied:

| Flag | Criterion | Default |
|------|-----------|--------|
| dead | filtered RMS < 0.5 µV | open-circuit electrode, no signal |
| noisy | filtered RMS > `noise_threshold` µV | artifact / high impedance during rest |

By default the quiet segment is **auto-detected** (bottom 20 % of 500 ms
windows ranked by mean-across-channels RMS).  You can pin it explicitly:

```bash
# Use the first 10 s as the known rest period
python 1_build_dataset.py --auto_select_channels --qc_quiet_sec 0 10

# Tune the noise ceiling (default 30 µV)
python 1_build_dataset.py --auto_select_channels --noise_threshold 20
```

Example output:
```
[INFO]   QC: using 58/293 quiet windows (≤20th pct global RMS 12.3 µV)
[INFO]   QC: 118 good, 10 bad  (median quiet-RMS 4.1 µV, thresholds: dead<0.5 µV, noisy>30 µV)
[WARNING]   Bad channels: [3, 27, 58, 76, 77, 78, 104, 115]
```

With `--auto_select_channels`, only the good channels are passed to feature
extraction and training.

---

## Using Your Own Open Ephys Data

### Option A — CSV export
1. Record EMG in Open Ephys GUI and export as CSV.
2. Make sure the file has `timestamp` and `ch0 … chN` columns  
   (rename if necessary).
3. Create a `labels.csv` alongside it with `Sample Index,Label` columns.
4. Run:
   ```bash
   python 1_build_dataset.py --data_path /path/to/your_recording.csv
   python 2_train_model.py
   python predict.py --data_path /path/to/your_recording.csv
   ```

### Option B — Open Ephys binary format
Pass a `.oebin` file or a recording folder directly:
```bash
python 1_build_dataset.py --data_path data/gestures/
```
The script auto-detects the format.

**Labels auto-discovery**: if a label file is placed alongside your recording
folder (or inside it), it will be found automatically — no `--labels_path` needed.
Accepted filenames (checked in order):

| Filename pattern | Example |
|---|---|
| `{recording}_emg.txt` | `session1_emg.txt` |
| `{recording}.txt` | `session1.txt` |
| `labels.csv` | `labels.csv` |
| `events.csv` | `events.csv` |
| `emg.txt` | `emg.txt` |
| `labels.txt` | `labels.txt` |

Both `emg` and `labels` variants are accepted.

For example, with this layout:
```
data/gestures/
├── Record Node 111/       ← Open Ephys binary recording
└── emg.txt                ← transition-style labels (auto-discovered)
```
Just run:
```bash
python 1_build_dataset.py --data_path data/gestures/
```

**Channel QC**: bad electrodes are detected using the quiet-segment method
(see [Channel Quality Control](#channel-quality-control) above).  Pass
`--auto_select_channels` to drop them automatically:
```bash
python 1_build_dataset.py --data_path data/gestures/ --auto_select_channels
# With an explicit rest window:
python 1_build_dataset.py --data_path data/gestures/ --auto_select_channels --qc_quiet_sec 0 10
```

### Option C — Real-time stream
1. Open Open Ephys GUI and add the **ZMQ Interface** plugin.
2. Train a model (steps 1–3 above).
3. Start recording / playback in the GUI.
4. Run:
   ```bash
   python 3_predict_realtime.py --host 127.0.0.1 --port 5556
   ```

---

## Project Layout

```
gesture_classifier/
├── data/
│   ├── gestures/                    # Open Ephys binary recording
│   │   ├── emg.txt                  # gesture event labels
│   │   └── training_dataset.npz    # built by 1_build_dataset.py
│   └── model/                      # built by 2_train_model.py
├── 1_build_dataset.py
├── 2_train_model.py
├── predict.py
├── 3_predict_realtime.py
├── .gesture_config.example
└── README.md
```

---

## Troubleshooting

**`ModuleNotFoundError: pyoephys`**  
→ Install from the repo root: `pip install -e .`

**`metadata.json not found`**  
→ Run `python 2_train_model.py` first to train a model.

**`training_dataset.npz not found`**  
→ Run `python 1_build_dataset.py` first.

**Real-time: "No data received … within 15s"**  
→ Ensure Open Ephys is running, the ZMQ Interface plugin is active,
and the GUI is in Record or Playback mode.

**Real-time: "ZMQ stream is sending N channel(s), but model was trained on 128"**  
→ Open the ZMQ Interface plugin in Open Ephys GUI and enable all channels.
The number of channels sent must match the recording used for training.
