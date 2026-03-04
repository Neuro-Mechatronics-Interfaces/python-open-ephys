# Joint Angle Regression: LabRecorder Setup Guide

To fix synchronization issues (jitter/lag) observed with Python-based recording, we recommend using **LabRecorder** (part of the LabStreamingLayer suite) for data acquisition.

## Prerequisites

1.  **Install LabRecorder**: Download the latest release from [sccn/fne-labrecorder](https://github.com/labstreaminglayer/App-LabRecorder/releases) or use the one provided in your lab's `LSL/Apps` folder.
2.  **Install pyxdf**: This Python library is needed to import the recorded files.
    ```bash
    pip install pyxdf
    ```

## Recording Procedure

1.  **Start Hardware Streams**:
    *   Launch your EMG streamer (e.g., `TMSiSAGA_MATLAB` logic or `open_ephys_lsl.py`). **Verify** it is streaming in LSL (use `LabRecorder` or `lsl_api_cfg` to check).
    *   Launch your Motion Capture streamer (e.g., `Markerless_Pose_Tracking` or `Unity`). **Verify** it is streaming.

2.  **Start the Session GUI**:
    *   Run `python session_gui.py`.
    *   **enable the "Use LabRecorder (External)" Checkbox** in the Recording panel.
    *   This disables the internal file writer and instead switches the GUI to "Broadcaster Mode".
    *   The GUI will now send LSL Markers (`SessionMarkers`) and cue information to the network.

3.  **Configure LabRecorder**:
    *   Open LabRecorder.
    *   Click **Update** to refresh streams.
    *   Select the following streams:
        *   **EMG Stream** (e.g., `TMSi`, `OpenEphys`)
        *   **Angle/Mocap Stream** (e.g., `HandAngles`)
        *   **SessionMarkers** (The stream created by `session_gui.py`)
    *   Enter a localized filename (e.g., `sub-00X_session-00Y_task-angles.xdf`).
    *   **Start Recording** in LabRecorder.

4.  **Run the Task**:
    *   In `session_gui.py`, click **Arm Recording** then **Start Task**.
    *   Follow the prompts on screen.
    *   The GUI will send markers for every cue (e.g., "thumb_flexion", "Rest") synchronized to the display time.
    *   When finished, click **Stop Task**.

5.  **Stop Recording**:
    *   Click **Stop** in LabRecorder.

## Data Import

Once you have the `.xdf` file, you can import it directly from the Session GUI or via the command line.

### Option 1: Using the GUI (Recommended)

1.  In `session_gui.py`, find the **"Import/Process Data"** section (bottom right).
2.  Click the **"Import XDF..."** button.
3.  Select your `.xdf` file.
4.  The conversion script will run automatically.
5.  Check the console output for the location of the saved `.npz` file (usually `processed_data/`).

### Option 2: Command Line

You can also run the import script manually:

```bash
python scripts/import_labrecorder_xdf.py path/to/your/recording.xdf --out_dir data/processed_xdf
```

This will generate `sub-00X_task-jointangles.npz`.

## Training

Train your model using the new data:

```bash
python scripts/train_cnn_attention_regressor.py \
  --train_data data/processed_xdf/sub-00X_task-jointangles.npz \
  --test_data data/processed_xdf/sub-00X_task-jointangles.npz \
  --out_dir models/sub-00X_labrecorder_test \
  --epochs 50
```

## Troubleshooting

*   **No "SessionMarkers" stream?**: Ensure you clicked "Arm Recording" or "Start Task" in the GUI *after* checking the LabRecorder box. The outlet is created when you start the task logic.
*   **Import fails?**: Check if `pyxdf` identifies the streams correctly. The import script looks for streams with types `EMG`, `Markers`, and `Mocap`/`Angles`. If your streams have different names/types, edit `data/import_labrecorder_xdf.py` and adjust the search logic.
