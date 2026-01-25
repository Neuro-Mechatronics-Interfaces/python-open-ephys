import numpy as np
from tqdm import tqdm
from handtrack.processing import (
    notch_filter,
    bandpass_filter,
    lowpass_filter,
    rectify,
    extract_features
)


class EMGPreprocessor:
    def __init__(self, fs=5000, band=(20, 450), notch_freq=60, envelope_cutoff=5, verbose=False):
        self.fs = fs
        self.band = band
        self.notch_freq = notch_freq
        self.envelope_cutoff = envelope_cutoff
        self.verbose = verbose

    def preprocess(self, emg):
        """
        Applies filtering pipeline:
        1. Notch filter at 60Hz
        2. Bandpass filter
        3. Rectify
        4. Lowpass filter to get envelope
        """
        if self.verbose:
            print("\n================== EMG Preprocessing ================")
            print("|  Preprocessing EMG data...")
            print(f"|  |__  Sampling frequency: {self.fs} Hz")
            print(f"|  |__  Notch frequency: {self.notch_freq} Hz")
            print(f"|  |__  Bandpass filter: {self.band[0]}-{self.band[1]} Hz")
            print(f"|  |__  Envelope cutoff: {self.envelope_cutoff} Hz")

        emg = notch_filter(emg, self.fs, self.notch_freq)
        emg = bandpass_filter(emg, self.band[0], self.band[1], self.fs)
        emg = rectify(emg)
        return lowpass_filter(emg, self.envelope_cutoff, self.fs)

    def extract_emg_features(self, emg, start_index=0, end_index=None, window_ms=100, step_ms=20):
        """
        Applies rolling window feature extraction to the preprocessed EMG data.

        Args:
            emg (np.ndarray): Preprocessed EMG data of shape (num_channels, num_samples).
            window_ms (int): Size of the rolling window in milliseconds.
            step_ms (int): Step size for the rolling window in milliseconds.

        Returns:
            features (np.ndarray): Extracted features of shape (num_windows, num_features).

        """
        if self.verbose:
            print(f"\n=================== EMG Features ==================")
            print(f"|  Extracting features with window size {window_ms}ms and step size {step_ms}ms...")

        window_size = int(window_ms * self.fs / 1000)
        step_size = int(step_ms * self.fs / 1000)

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = emg.shape[1] - window_size + 1

        if start_index < 0 or start_index >= emg.shape[1]:
            raise ValueError(f"start_index {start_index} is out of bounds for EMG data with shape {emg.shape}")

        if window_size <= 0 or step_size <= 0:
            raise ValueError(f"Invalid window size {window_size} or step size {step_size}")
        if self.verbose:
            print(f"|  Window size: {window_size} samples, Step size: {step_size} samples")
            print(f"|  EMG data shape: {emg.shape}, starting from index {start_index}")

        if end_index is None:
            end_index = emg.shape[1] - window_size + 1
            print(f"|  No end index provided, using full EMG data up to {end_index} samples.")

        # if the end index exceeds the EMG data length, adjust it
        if end_index <= start_index:
            raise ValueError(f"End index {end_index} must be greater than start index {start_index} for window size {window_size}")
        #n_features = int(() / step_size) + 1
        #if self.verbose:
        #    print(f"|  Extracting {n_features} features from EMG data with shape {emg.shape}...")

        features = []
        for start in tqdm(range(start_index, end_index - window_size + 1, step_size), desc="|  Extracting features"):
            end = start + window_size
            features.append(extract_features(emg[:, start:end]))

        features = np.array(features)
        if self.verbose:
            print(f"|  Features collected, shape: {features.shape}")

        return features

    def extract_features_static(self, emg):
        """
        Extracts features from a static EMG window.
        Returns: (num_features)
        """
        if self.verbose:
            print(f"|  Extracting static features from EMG window of shape {emg.shape}...")

        features = extract_features(emg)

        if self.verbose:
            print(f"|  Static features extracted, shape: {features.shape}")

        return features
