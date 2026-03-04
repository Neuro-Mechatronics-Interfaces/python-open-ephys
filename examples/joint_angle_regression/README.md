# Joint Angle Regression (EMG → Joint Angles)

This folder supports:
- synchronized EMG + LSL angle recording,
- model training (feature extractor + regressor),
- live EMG inference with optional predicted-angle LSL output,
- analysis/figure generation.

## Canonical Layout

- `session_gui.py` → **single GUI** (record + train + live compare)
- `scripts/` → canonical training/analysis/utility scripts
- `figures/` → canonical location for generated plots/reports
- `data/` → recorded datasets (`.npz`)
- `models/` → trained artifacts (`.pkl`, `.h5`, metrics)

## Quick Start

From this directory:

```bash
cd python-open-ephys/examples/joint_angle_regression
```

### 1) Launch GUI (canonical)

```bash
python session_gui.py
```

Or on Windows:

```bat
run_gui.bat
```

### 2) Optional streamers (separate terminals)

```bash
python scripts/open_ephys_lsl_streamer.py --no-gui
python scripts/imu_lsl_streamer.py --imu-host 192.168.4.1 --imu-port 5555 --imu-transport UDP
```

### 3) Record data in GUI

Use the GUI to set subject/session metadata, connect EMG + angle LSL, and record.
Saved `.npz` files go under `data/` by default.

### 4) Train models (CLI option)

```bash
python scripts/train_feature_extractor.py \
  --data data/sub-P001_ses-S01_emg-angles.npz \
  --out_model models/feature_extractor.h5

python scripts/train_regressor.py \
  --data data/sub-P001_ses-S01_emg-angles.npz \
  --feature_extractor models/feature_extractor.h5 \
  --out_dir models/run_001 \
  --emg_transform none \
  --emg_features raw_flat \
  --angle_scaler minmax

# EMD-aware raw-feature training (requires: pip install EMD-signal)
python scripts/train_regressor.py \
  --data data/sub-P001_ses-S01_emg-angles.npz \
  --feature_extractor none \
  --out_dir models/run_emd \
  --emg_transform log1p \
  --emg_features emd_stats \
  --emd_max_imfs 3 \
  --angle_scaler minmax

# Fast personalization adaptor (few-minute calibration; no full retrain)
python scripts/personalize_regressor.py \
  --base_model_dir models/run_emd \
  --calib_data data/sub-P001_ses-S02_emg-angles.npz \
  --alpha 1.0
```

### 5) Generate figures (recommended outputs in `figures/`)

```bash
python scripts/plot_session6_model_comparison.py --out_dir figures/session6
python scripts/make_session6_qa_figures.py --out_dir figures/qa
python scripts/make_session_generalization_report.py --output_dir figures/generalization
```

## Live Compare: model selection + predicted-angle LSL

In `session_gui.py` → **Live Compare** panel:
- choose `Feature extractor`, `Regressor`, `Scaler`,
- click `Start Compare` to run EMG→angle inference in-GUI,
- optionally enable **Stream predicted angles over LSL**,
- set stream name (default `PredictedJointAngles`).

This publishes predicted angles as an LSL float stream for downstream visualizers.

## Quick NPZ Comparison GUI (measured vs predicted)

Use this when you want an offline side-by-side visual comparison from saved `.npz` sessions.

Launch:

```bash
python scripts/npz_angle_compare_gui.py
```

In the GUI:
- Select your dataset `.npz` (e.g., `data/sub-001/ses-006/sub-001_ses-006_task-jointangles_wrist-neutral.npz`). It loads automatically.
- Select model directory (contains `mlp_regressor.pkl` + `scaler.pkl`) and feature extractor `.h5`, then click **Load Model**.
- Click **Predict** to compute predicted angles for all windows.
- Use the slider/play controls to inspect side-by-side hand views:
  - left: measured angles,
  - right: predicted angles.

The GUI also reports per-frame MAE over common joints.

If `personalization.pkl` exists in the model directory, it is applied automatically after base prediction.

## Data Keys (`.npz`)

Typical saved fields include:
- `emg`, `angles`, `timestamps`, `markers`, `imu`,
- `angle_keys`, `target_spec`, `session_start`,
- `landmark_xyz`, `landmark_valid_windows`.

## Notes on Refactor

- Training/analysis scripts now live in `scripts/`.
- Use `scripts/*` paths in docs and automation.
- `figures/` is now the default destination for analysis outputs.
