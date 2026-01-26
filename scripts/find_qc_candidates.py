import numpy as np
import os
import sys

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyoephys.processing import ChannelQC

def find_candidate_channels(data_path):
    print(f"Analyzing {data_path} for QC candidates...")
    data = np.load(data_path, allow_pickle=True)
    raw_full = data['amplifier_data']
    fs = 2000
    
    # Analyze a representative 1-second segment
    start_idx = int(fs * 2.5) 
    end_idx = start_idx + int(fs * 1.0)
    seg = raw_full[:, start_idx:end_idx]
    
    qc = ChannelQC(fs=fs, n_channels=raw_full.shape[0])
    qc.update(seg.T)
    results = qc.evaluate()
    
    bad_indices = np.where(results['bad'])[0]
    good_indices = np.where(~results['bad'] & ~results['watch'])[0]
    
    print(f"Found {len(bad_indices)} bad channels: {bad_indices[:10]}...")
    print(f"Found {len(good_indices)} good channels: {good_indices[:10]}...")
    
    if len(bad_indices) > 0 and len(good_indices) >= 2:
        print(f"RECOMMENDED: Good=[{good_indices[0]}, {good_indices[1]}], Bad=[{bad_indices[0]}]")
    else:
        # If no bad ones, find the "worst" good one
        metrics = results['metrics']
        worst_idx = np.argmax(metrics['robust_z'])
        print(f"No naturally 'bad' channels found. Worst Z-score is channel {worst_idx}")

if __name__ == "__main__":
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz"
    find_candidate_channels(DATA_PATH)
