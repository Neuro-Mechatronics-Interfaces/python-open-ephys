#!/usr/bin/env python3
"""
Build EMG Gesture Classification Dataset from Open Ephys Data.

Walks a directory structure, finds Open Ephys recordings (structure.oebin),
extracts EMG features (RMS/MAV), loads labels, and saves a consolidated .npz dataset.
"""

import os
import argparse
import numpy as np
import json
import scipy.signal

from pyoephys.io import load_open_ephys_session
from pyoephys.processing import normalize_emg, butter_bandpass_filter

def extract_windows(signal, fs, window_ms, step_ms):
    """Simple sliding window feature extraction (RMS)."""
    window_samples = int(fs * window_ms / 1000)
    step_samples = int(fs * step_ms / 1000)
    
    n_samples, n_channels = signal.shape
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    if n_windows <= 0:
        return np.empty((0, n_channels)), np.empty((0,))
        
    # Shape: (n_windows, n_channels)
    # Using RMS for simplicity
    features = np.zeros((n_windows, n_channels), dtype=np.float32)
    
    # Timestamps (center of window)
    window_centers = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        chunk = signal[start:end, :]
        
        # RMS
        rms = np.sqrt(np.mean(chunk**2, axis=0))
        features[i, :] = rms
        window_centers[i] = (start + end) / 2 / fs
        
    return features, window_centers

def load_labels(events_path, window_centers, duration_sec):
    """
    Load labels from an explicit events file or inference logic.
    For now, this is a placeholder or simple logic (e.g. filename based).
    """
    # TODO: Implement proper event loading from Open Ephys events or CSV
    # For now, return dummy class '0' or based on folder name
    return np.zeros(len(window_centers), dtype=int)


def build_dataset(root_dir, save_path, window_ms=200, step_ms=50):
    print(f"Scanning {root_dir} for Open Ephys recordings...")
    
    X_list = []
    y_list = []
    
    # Walk directory
    for root, dirs, files in os.walk(root_dir):
        if "structure.oebin" in files:
            print(f"Processing: {root}")
            try:
                # Load data
                session = load_open_ephys_session(root)
                # Assuming first continuous stream is EMG
                # In robust code, we'd select by name or metadata
                rec = session.recordnodes[0].recordings[0]
                data = rec.continuous[0].samples
                fs = rec.continuous[0].metadata['sample_rate']
                
                # Preprocess
                # Highpass to remove DC drift
                data_filt = butter_bandpass_filter(data.T, 20, fs/2 - 1, fs, order=4).T
                
                # Normalize (Z-score or similar) - maybe per file?
                # data_norm = normalize_emg(data_filt)
                
                # Extract Features
                feats, centers = extract_windows(data_filt, fs, window_ms, step_ms)
                
                if len(feats) > 0:
                    # Labeling logic
                    # Try to deduce label from folder name (e.g., 'fist', 'rest')
                    folder_name = os.path.basename(root).lower()
                    label = folder_name # string label for now
                    
                    labels = np.full(len(feats), label)
                    
                    X_list.append(feats)
                    y_list.append(labels)
                    print(f"  -> Added {len(feats)} windows, Label: {label}")
                    
            except Exception as e:
                print(f"  Failed: {e}")
                
    if not X_list:
        print("No data found!")
        return
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Convert string labels to int if needed, or keep as strings
    # The existing trainer expects somewhat formatted labels
    
    print(f"Saving {X.shape[0]} samples to {save_path}")
    np.savez(save_path, X=X, y=y, fs=fs, window_ms=window_ms, step_ms=step_ms)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--save_path", default="training_dataset.npz")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    args = p.parse_args()
    
    build_dataset(args.root_dir, args.save_path, args.window_ms, args.step_ms)
