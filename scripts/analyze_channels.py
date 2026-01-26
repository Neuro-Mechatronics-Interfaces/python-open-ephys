import numpy as np
import os
import sys

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyoephys.processing import ChannelQC

def analyze_all_channels(data_path):
    data = np.load(data_path, allow_pickle=True)
    raw_full = data['amplifier_data']
    fs = 2000
    
    qc_start = 0
    qc_end = int(fs * 5.0)
    seg = raw_full[:, qc_start:qc_end]
    
    qc = ChannelQC(fs=fs, n_channels=raw_full.shape[0])
    qc.update(seg.T)
    results = qc.evaluate()
    
    metrics = results['metrics']
    robust_z = metrics['robust_z']
    is_bad = results['bad']
    is_watch = results['watch']
    
    print("Channel Analysis (0-5s):")
    # Indices of not bad
    not_bad = np.where(~is_bad)[0]
    print(f"Not Bad: {not_bad}")
    
    # Sort robust_z for not_bad
    if len(not_bad) > 0:
        sorted_good = not_bad[np.argsort(robust_z[not_bad])]
        print(f"Sorted Not-Bad (lowest Z first): {sorted_good}")
    
    # Sort robust_z for bad
    bad = np.where(is_bad)[0]
    if len(bad) > 0:
        sorted_bad = bad[np.argsort(robust_z[bad])]
        print(f"Sorted Bad (lowest Z first): {sorted_bad}")

if __name__ == "__main__":
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz"
    analyze_all_channels(DATA_PATH)
