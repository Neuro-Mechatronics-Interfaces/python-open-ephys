import argparse
import os

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def process_ninapro_file(mat_file_path, output_dir, subject_id, exercise_id):
    print(f"Processing {mat_file_path}...")
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return

    # Extract raw data
    # Ninapro DB2: 12 EMG channels, sampled at 2kHz
    if "emg" not in mat:
        print(f"  Skipping {mat_file_path}: No 'emg' data found.")
        return

    raw_emg = mat["emg"]  # (Time, 12)

    # Kinematics: 22 Glove sensors
    if "glove" not in mat:
        print(
            f"  Skipping {mat_file_path}: No 'glove' data found (likely force/grasp exercise)."
        )
        return

    raw_glove = mat["glove"]  # (Time_glove, 22)


def segment_data(emg, angles, stimulus, repetition, window_len, step_len):
    # Segment data into windows (N, Channels, Time)
    # Returns: emg, angles, stimulus, repetition
    n_samples = emg.shape[0]
    n_windows = (n_samples - window_len) // step_len

    if n_windows <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Pre-allocate
    # EMG: (N, C, T) - channels first for compatibility?
    # Checking train_regressor:
    #   if s1 < s2: (N, C, T) -> returns (N, C, T, 1)
    #   else: (N, T, C) -> returns (N, C, T, 1) via transpose
    # Let's use (N, C, T) standard.
    n_channels = emg.shape[1]

    EMG = np.zeros((n_windows, n_channels, window_len), dtype=np.float32)
    ANG = np.zeros((n_windows, angles.shape[1]), dtype=np.float32)
    STI = np.zeros((n_windows, 1), dtype=np.float32)
    REP = np.zeros((n_windows, 1), dtype=np.float32)

    for i in range(n_windows):
        start = i * step_len
        end = start + window_len

        # Taking EMG window
        # emg is (Time, Channels). We want (Channels, Time)
        window_emg = emg[start:end, :].T
        EMG[i] = window_emg

        # Taking last sample for Angles/Labels
        ANG[i] = angles[end - 1, :]
        STI[i] = stimulus[end - 1]
        REP[i] = repetition[end - 1]

    return EMG, ANG, STI, REP


def process_ninapro_file(mat_file_path, output_dir, subject_id, exercise_id):
    print(f"Processing {mat_file_path}...")
    try:
        mat = scipy.io.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading {mat_file_path}: {e}")
        return

    # Extract raw data
    # Ninapro DB2: 12 EMG channels, sampled at 2kHz
    if "emg" not in mat:
        print(f"  Skipping {mat_file_path}: No 'emg' data found.")
        return

    raw_emg = mat["emg"]  # (Time, 12)

    # Kinematics: 22 Glove sensors
    if "glove" not in mat:
        print(
            f"  Skipping {mat_file_path}: No 'glove' data found (likely force/grasp exercise)."
        )
        return

    raw_glove = mat["glove"]  # (Time_glove, 22)

    # Stimulus/Labels
    if "repetition" in mat:
        raw_repetition = mat["repetition"]
    else:
        # Some files might miss it? Usually DB2 has it.
        # If missing, we can't split by repetition easily.
        # Construct specific logic or warn.
        print(
            "  Warning: 'repetition' variable not found via standard key. Checking variants..."
        )
        if "rerepetition" in mat:
            raw_repetition = mat["rerepetition"]
        else:
            # Create dummy repetition if needed? No, better validation.
            # Default to all 0
            raw_repetition = np.zeros((raw_emg.shape[0], 1))

    # 'restimulus' contains the refined movement labels (0=rest, 1..N=movements)
    stimulus = mat["restimulus"]

    # Check lengths
    n_emg = raw_emg.shape[0]
    n_glove = raw_glove.shape[0]
    n_rep = raw_repetition.shape[0]

    print(f"  EMG: {n_emg}, Glove: {n_glove}, Reps: {n_rep}")

    # Upsample Glove
    if n_emg != n_glove:
        print(f"  Resampling glove data from {n_glove} to {n_emg} samples...")
        x_glove = np.linspace(0, 1, n_glove)
        x_emg = np.linspace(0, 1, n_emg)

        new_glove = np.zeros((n_emg, raw_glove.shape[1]))
        for i in range(raw_glove.shape[1]):
            # Use linear for continuous data
            new_glove[:, i] = np.interp(x_emg, x_glove, raw_glove[:, i])
        glove_data = new_glove
    else:
        glove_data = raw_glove

    # Upsample Repetition (Nearest Neighbor)
    if n_emg != n_rep:
        print(f"  Resampling repetition labels from {n_rep} to {n_emg}...")
        x_rep = np.linspace(0, 1, n_rep)
        x_emg = np.linspace(0, 1, n_emg)

        # Nearest neighbor interpolation for categorical data
        # np.interp is linear; use a different approach
        # indices = np.round((n_rep - 1) * x_emg).astype(int) ? No, simpler:
        # Map x_emg to indices in x_rep space
        idx = np.round(np.interp(x_emg, x_rep, np.arange(n_rep))).astype(int)
        idx = np.clip(idx, 0, n_rep - 1)
        repetition = raw_repetition[idx]

        # Also do stimulus if lengths differ (usually they match rep)
        if stimulus.shape[0] == n_rep:
            stimulus = stimulus[idx]
    else:
        repetition = raw_repetition

    # --- Filtering ---
    fs_emg = 2000.0

    print("  Filtering EMG (Bandpass 20-500Hz)...")
    b_emg, a_emg = butter_bandpass(20.0, 500.0, fs_emg, order=4)
    filtered_emg = filtfilt(b_emg, a_emg, raw_emg, axis=0)

    print("  Filtering Kinematics (Lowpass 5Hz)...")
    b_glove, a_glove = butter_lowpass(5.0, fs_emg, order=2)
    filtered_glove = filtfilt(b_glove, a_glove, glove_data, axis=0)

    # --- Splitting & Saving ---
    # Standard Ninapro Split:
    # Train: Repetitions 1, 3, 4, 6
    # Test: Repetitions 2, 5

    rep_flat = repetition.flatten()

    train_reps = [1, 3, 4, 6]
    test_reps = [2, 5]

    # Create masks
    mask_train = np.isin(rep_flat, train_reps)
    mask_test = np.isin(rep_flat, test_reps)

    # Filter Data FIRST, then window?
    # If we mask first, we lose continuity at boundaries.
    # Better: Window everything, then checking repetition of the window (based on label).

    # Window params: 200ms (400 samples), Step 50ms (100 samples)
    win_len = 400
    step_len = 100

    print(f"  Segmenting data (Window {win_len}, Step {step_len})...")
    w_emg, w_ang, w_sti, w_rep = segment_data(
        filtered_emg, filtered_glove, stimulus, repetition, win_len, step_len
    )

    # Now mask the WINDOWED data based on the Repetition label of the window
    w_rep_flat = w_rep.flatten()

    w_mask_train = np.isin(w_rep_flat, train_reps)
    w_mask_test = np.isin(w_rep_flat, test_reps)

    # Save Train
    if np.any(w_mask_train):
        out_train = os.path.join(
            output_dir, f"{subject_id}_task-exercise{exercise_id}_train.npz"
        )
        np.savez_compressed(
            out_train,
            emg=w_emg[w_mask_train],
            angles=w_ang[w_mask_train],
            fs=fs_emg,
            stimulus=w_sti[w_mask_train],
            repetition=w_rep[w_mask_train],
            exercise=exercise_id,
        )
        print(
            f"  Saved Train (Reps {train_reps}) to {out_train}: {w_emg[w_mask_train].shape}"
        )

    # Save Test
    if np.any(w_mask_test):
        out_test = os.path.join(
            output_dir, f"{subject_id}_task-exercise{exercise_id}_test.npz"
        )
        np.savez_compressed(
            out_test,
            emg=w_emg[w_mask_test],
            angles=w_ang[w_mask_test],
            fs=fs_emg,
            stimulus=w_sti[w_mask_test],
            repetition=w_rep[w_mask_test],
            exercise=exercise_id,
        )
        print(
            f"  Saved Test  (Reps {test_reps}) to {out_test}: {w_emg[w_mask_test].shape}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Import Ninapro DB2 .mat files to .npz"
    )
    parser.add_argument(
        "--db_dir", type=str, required=True, help="Directory containing DB2_sX folders"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for .npz files"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        help="List of subjects to process (e.g. 1 2). Default: all found.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Find subject folders
    if args.subjects:
        subjects = args.subjects
    else:
        # crude auto-discovery
        subjects = []
        for d in os.listdir(args.db_dir):
            if d.startswith("DB2_s"):
                try:
                    s_num = int(d.split("_s")[1])
                    subjects.append(str(s_num))
                except:
                    pass

    print(f"Found subjects: {subjects}")

    for sub in subjects:
        sub_dir_name = f"DB2_s{sub}"
        # Ninapro zip extraction often creates nested folders like DB2_s1/DB2_s1/S1_E1_A1.mat
        # OR just DB2_s1/S1_E1_A1.mat
        # Let's search recursively or check both depth levels.

        base_path = os.path.join(args.db_dir, sub_dir_name)
        if not os.path.exists(base_path):
            print(f"Directory for subject {sub} not found at {base_path}")
            continue

        # Look for mat files
        mat_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".mat") and file.startswith(f"S{sub}_"):
                    mat_files.append(os.path.join(root, file))

        if not mat_files:
            print(f"No .mat files found for subject {sub}")
            continue

        # Process each exercise
        for mat_f in mat_files:
            # Filename format: S1_E1_A1.mat
            fname = os.path.basename(mat_f)
            parts = fname.split("_")
            # parts[1] is E1, E2, etc.
            exercise_id = parts[1]

            # Create subject_id string
            subject_id_str = f"sub-{int(sub):03d}"

            process_ninapro_file(mat_f, args.output_dir, subject_id_str, exercise_id)


if __name__ == "__main__":
    main()
