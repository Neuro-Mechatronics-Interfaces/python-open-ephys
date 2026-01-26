import numpy as np
import os
import sys

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyoephys.processing import ChannelQC

def find_candidate_channels(data_path):
    print(f"Analyzing {data_path} for 5-channel QC candidates (0-5s window)...")
    data = np.load(data_path, allow_pickle=True)
    raw_full = data['amplifier_data']
    fs = 2000
    
    # Use first 5 seconds for QC as requested
    qc_start = 0
    qc_end = int(fs * 5.0)
    seg = raw_full[:, qc_start:qc_end]
    
    qc = ChannelQC(fs=fs, n_channels=raw_full.shape[0])
    qc.update(seg.T)
    results = qc.evaluate()
    
    bad_indices = np.where(results['bad'])[0]
    good_indices = np.where(~results['bad'] & ~results['watch'])[0]
    
    print(f"Found {len(bad_indices)} bad channels.")
    print(f"Found {len(good_indices)} good channels.")
    
    if len(bad_indices) >= 1 and len(good_indices) >= 4:
        # Proposed set: [Pass, Pass, Fail, Pass, Pass]
        final_set = [good_indices[0], good_indices[1], bad_indices[0], good_indices[2], good_indices[3]]
        print(f"RECOMMENDED SET (4 pass, 1 fail): {final_set}")
    else:
        print("Could not find enough candidates with strict criteria.")

if __name__ == "__main__":
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz"
    find_candidate_channels(DATA_PATH)
