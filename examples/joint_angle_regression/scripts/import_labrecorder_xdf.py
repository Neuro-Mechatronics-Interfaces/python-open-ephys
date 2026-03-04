import argparse
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.signal

try:
    import pyxdf
except ImportError:
    print("Error: pyxdf not found. Please install it: pip install pyxdf")
    pyxdf = None

# Standard movements for 'full14' or general use
# Modify based on your prompt file labels
MOVEMENT_MAP = {
    "rest": 0,
    "Rest": 0,
    # Add your specific prompt labels here
    # If using dynamic mapping, the script builds it on the fly
}


def load_xdf(file_path):
    print(f"Loading {file_path}...")
    streams, header = pyxdf.load_xdf(str(file_path))

    emg_stream = None
    angle_stream = None
    marker_stream = None

    for s in streams:
        name = s["info"]["name"][0]
        stype = s["info"]["type"][0]

        # Heuristics
        if stype == "EMG" or "TMSi" in name or "OpenEphys" in name:
            if emg_stream is None:  # take first
                emg_stream = s
                print(f"  Found EMG Stream: {name} ({stype})")

        elif stype == "Markers" or "Markers" in name:
            marker_stream = s
            print(f"  Found Marker Stream: {name} ({stype})")

        elif stype == "Mocap" or "Angles" in name or "Tracker" in name:
            angle_stream = s
            print(f"  Found Angle Stream: {name} ({stype})")

    if emg_stream is None:
        raise ValueError("No EMG stream found")
    if angle_stream is None:
        raise ValueError("No Angle/Mocap stream found")

    return emg_stream, angle_stream, marker_stream


def resample_stream(source_time, source_data, target_time):
    # Linear interpolation for continuous signals
    n_ch = source_data.shape[1]
    n_target = len(target_time)
    out = np.zeros((n_target, n_ch), dtype=np.float32)

    for i in range(n_ch):
        # Fill with nearest or linear? Linear is better for smooth angles/EMG envelope
        # But raw EMG is high freq. If downsampling, use decimate ideally.
        # But for alignment, interp is okay if sampling rates are close or upsampling.
        # If EMG is 2kHz and Angles 60Hz, we upsample Angles.

        # Check kind
        f = scipy.interpolate.interp1d(
            source_time,
            source_data[:, i],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        out[:, i] = f(target_time)

    return out


def process_markers(marker_stream, target_time):
    # Create a stimulus vector aligned with target_time
    stimulus = np.zeros((len(target_time), 1), dtype=np.float32)
    repetition = np.zeros((len(target_time), 1), dtype=np.float32)

    if not marker_stream:
        print("  Warning: No marker stream. Stimulus will be 0.")
        return stimulus, repetition

    times = marker_stream["time_series"]  # List of strings
    stamps = marker_stream["time_stamps"]  # timestamps

    # Build dynamic map
    label_to_id = {"rest": 0, "Rest": 0}
    next_id = 1

    current_label = 0
    current_rep = 1  # increment on rest->move

    # Sort events by time just in case
    events = sorted(zip(stamps, times), key=lambda x: x[0])

    # Fill stimulus array
    # Iterate through time points
    # This is inefficient for long files. Better to iterate events.

    evt_idx = 0
    current_val = 0

    # Pre-compute intervals
    intervals = []

    for i in range(len(events) - 1):
        t_start = events[i][0]
        # label might be a list usually ['Label']
        lbl_raw = events[i][1]
        lbl = lbl_raw[0] if isinstance(lbl_raw, list) else str(lbl_raw)

        t_end = events[i + 1][0]

        # Parse Label
        if lbl.startswith("Session_Start") or lbl.startswith("Config"):
            continue

        # Update IDs
        if lbl.lower() in ["rest", "end_trial", "session_end"]:
            val = 0
            # Repetition stays 0 for rest? Or keeps previous? Usually 0 for rest.
            rep_val = 0
        else:
            if lbl not in label_to_id:
                label_to_id[lbl] = next_id
                next_id += 1
            val = label_to_id[lbl]
            current_rep += 1
            rep_val = current_rep

        intervals.append((t_start, t_end, val, rep_val))

    # Apply to grid
    # For each sample in target_time, find which interval it falls in
    # Or cleaner: fill slices

    for t_start, t_end, val, rep_val in intervals:
        mask = (target_time >= t_start) & (target_time < t_end)
        stimulus[mask] = val
        repetition[mask] = rep_val

    print(f"  Extracted {len(label_to_id)} unique labels: {label_to_id}")
    return stimulus, repetition


def main():
    parser = argparse.ArgumentParser(description="Convert LabRecorder XDF to Model NPZ")
    parser.add_argument("xdf_file", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--subject", type=str, default="sub-001")
    parser.add_argument("--window_ms", type=float, default=150.0)
    parser.add_argument("--overlap_ms", type=float, default=75.0)

    args = parser.parse_args()

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    if not pyxdf:
        return

    emg, angles, markers = load_xdf(args.xdf_file)

    # Base Timeline: EMG
    emg_ts = emg["time_stamps"]
    emg_data = emg["time_series"]

    print(f"  EMG: {len(emg_ts)} samples, {emg_data.shape[1]} channels")

    # Fix time offset if needed (start at 0 for convenience in arrays?)
    # No, keep absolute for alignment, then drop

    # Align Angles to EMG
    ang_ts = angles["time_stamps"]
    ang_data = angles["time_series"]

    print(f"  Angles: {len(ang_ts)} samples, {ang_data.shape[1]} channels")

    print("  Resampling Angles to EMG timeline...")
    aligned_angles = resample_stream(ang_ts, ang_data, emg_ts)

    # Markers
    print("  Processing Markers...")
    stimulus, repetition = process_markers(markers, emg_ts)

    # Save
    out_name = args.out_dir / f"{args.subject}_{args.task}.npz"

    # Segment logic?
    # train_cnn expects (N, C, Win) or raw?
    # import_ninapro saves raw? No, it segments.
    # But train_cnn can iterate over a long raw file if we change the loader,
    # or we adhere to the segment format.

    # Actually, import_ninapro_db2.py SAVES SEGMENTED DATA (N, Channels, Time).
    # We should do the same. Standardize to 150ms windows?

    # Segmenting
    fs = float(emg["info"]["nominal_srate"][0])
    win_len = int((args.window_ms / 1000.0) * fs)
    # step is what we advance by. overlap is what we save.
    # step = window - overlap
    step_samples = int(((args.window_ms - args.overlap_ms) / 1000.0) * fs)
    if step_samples < 1:
        step_samples = 1

    print(
        f"  Segmenting (Win={win_len} samples, Step={step_samples} samples, {args.window_ms}ms)..."
    )

    n_samples = emg_data.shape[0]
    n_wins = (n_samples - win_len) // step_samples

    if n_wins < 1:
        print("  Error: Not enough data for one window.")
        return

    w_emg = np.zeros((n_wins, emg_data.shape[1], win_len), dtype=np.float32)
    w_ang = np.zeros((n_wins, aligned_angles.shape[1]), dtype=np.float32)
    w_sti = np.zeros((n_wins, 1), dtype=np.float32)
    w_rep = np.zeros((n_wins, 1), dtype=np.float32)

    for i in range(n_wins):
        start = i * step_samples
        end = start + win_len

        # Copy data
        # EMG: (Channels, Time) -> Transpose to match (C, T) or (T, C)?
        # Ninapro loader: (N, C, T) -> emg (T, C) -> Transpose -> (C, T)
        # Here emg_data is (T, C).
        chunk = emg_data[start:end, :]
        w_emg[i, :, :] = chunk.T  # (C, T)

        # Angles: Mean of window (target) or last sample?
        # Usually instantaneous target -> last sample
        # But for smooth regression, mean or center is safer.
        w_ang[i, :] = aligned_angles[end - 1, :]

        # Stimulus
        w_sti[i] = stimulus[end - 1]
        w_rep[i] = repetition[end - 1]

    fs = float(emg["info"]["nominal_srate"][0])

    np.savez_compressed(
        out_name, emg=w_emg, angles=w_ang, stimulus=w_sti, repetition=w_rep, fs=fs
    )
    print(f"Saved {out_name}: {w_emg.shape}")


if __name__ == "__main__":
    main()
