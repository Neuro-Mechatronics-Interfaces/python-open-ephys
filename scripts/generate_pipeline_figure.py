import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyoephys.processing import bandpass_filter, notch_filter, calculate_rms, ChannelQC, common_average_reference, QCParams

def generate_pipeline_figure(data_path, save_path):
    """
    Refined JOSS figure:
    - 5 channels (4 pass, 1 fail)
    - 10-15s window for A & B
    - 0-5s window for C (QC analysis)
    - Side-by-side layout for C & D
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    raw_full = data['amplifier_data']
    t_full = data['t_amplifier']
    fs = 2000 
    
    # Cherry-picked indices (from 0-5s analysis):
    # Pass: 114, 113, 121, 51
    # Fail: 0
    viz_indices = [114, 113, 0, 121, 51] # Putting Fail in the middle for visual contrast
    
    # 1. QC Analysis (0-5s window)
    print("Running QC Analysis on 0-5s window...")
    qc_start_idx = 0
    qc_end_idx = int(fs * 5.0)
    qc_seg = raw_full[:, qc_start_idx:qc_end_idx]
    
    # Use slightly relaxed params for the figure's "Pass" labels to match user request
    params = QCParams(robust_z_bad=6.0, robust_z_warn=4.0) 
    qc = ChannelQC(fs=fs, n_channels=raw_full.shape[0], params=params)
    qc.update(qc_seg.T)
    qc_results = qc.evaluate()
    qc_status = [not qc_results['bad'][i] for i in viz_indices]
    
    # 2. Main Visualization Prep (10-20s window)
    print("Preparing visualization for 10-20s window...")
    viz_start_idx = int(fs * 10.0)
    viz_end_idx = int(fs * 20.0)
    
    raw_seg = raw_full[viz_indices, viz_start_idx:viz_end_idx]
    t_seg = t_full[viz_start_idx:viz_end_idx]
    t_plot = t_seg - t_seg[0] + 10.0 # Keep 10-20s x-axis label style

    # Processing (Full array for CAR, then subset)
    car_data = common_average_reference(raw_full[:, viz_start_idx:viz_end_idx])
    filt = notch_filter(car_data, fs=fs, f0=60)
    filt = bandpass_filter(filt, lowcut=20, highcut=500, fs=fs)
    processed_viz = filt[viz_indices, :]

    # 3. RMS Calculation (on the viz segment)
    rms_vals = calculate_rms(processed_viz, window_size=int(0.1 * fs)) 
    rms_avg = np.mean(rms_vals, axis=1)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.4, wspace=0.3)
    
    ax_raw = fig.add_subplot(gs[0, :])
    ax_filt = fig.add_subplot(gs[1, :])
    ax_qc = fig.add_subplot(gs[2, 0])
    ax_rms = fig.add_subplot(gs[2, 1])
    
    colors_main = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71', '#9b59b6']
    raw_offset = 600
    filt_offset = 250

    # A. Raw Waterfall
    for i in range(len(viz_indices)):
        sig = raw_seg[i] - np.mean(raw_seg[i])
        ax_raw.plot(t_plot, sig + i * raw_offset, color='black', alpha=0.7, linewidth=0.4)
    ax_raw.set_title("A) Raw High-Density EMG Signals (10–20s Window)", loc='left', fontsize=13, fontweight='bold')
    ax_raw.set_ylabel("Amplitude ($\mu$V)")
    ax_raw.set_ylim(-400, 4 * raw_offset + 400)
    ax_raw.set_yticks([i * raw_offset for i in range(len(viz_indices))])
    ax_raw.set_yticklabels([f"Ch {idx}" for idx in viz_indices])
    ax_raw.set_xlabel("Time (s)")
    ax_raw.grid(True, alpha=0.1)
    
    # B. Filtered Signal
    for i in range(len(viz_indices)):
        sig = processed_viz[i] - np.mean(processed_viz[i])
        ax_filt.plot(t_plot, sig + i * filt_offset, color=colors_main[i], linewidth=0.4)
    ax_filt.set_title("B) Preprocessed Waveforms (CAR + Bandpass + Notch)", loc='left', fontsize=13, fontweight='bold')
    ax_filt.set_ylabel("Amplitude ($\mu$V)")
    ax_filt.set_ylim(-150, 4 * filt_offset + 150)
    ax_filt.set_yticks([i * filt_offset for i in range(len(viz_indices))])
    ax_filt.set_yticklabels([f"Ch {idx}" for idx in viz_indices])
    ax_filt.set_xlabel("Time (s)")
    ax_filt.grid(True, alpha=0.1)
    
    # C. QC Status (using the 0-5s results)
    qc_colors = ['green' if s else 'red' for s in qc_status]
    qc_labels = ['Pass' if s else 'Fail' for s in qc_status]
    for i in range(len(viz_indices)):
        ax_qc.barh(i, 1, color=qc_colors[i], alpha=0.6, height=0.7)
        ax_qc.text(0.5, i, f"{qc_labels[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax_qc.set_title("C) Automated Channel QC (0-5s)", loc='left', fontsize=12, fontweight='bold')
    ax_qc.set_yticks(range(len(viz_indices)))
    ax_qc.set_yticklabels([f"Ch {idx}" for idx in viz_indices])
    ax_qc.set_xticks([])
    ax_qc.set_xlim(0, 1)

    # D. RMS Barplot
    ax_rms.bar(range(len(viz_indices)), rms_avg, color=qc_colors, alpha=0.8)
    ax_rms.set_title("D) RMS Features (10-20s)", loc='left', fontsize=12, fontweight='bold')
    ax_rms.set_xticks(range(len(viz_indices)))
    ax_rms.set_xticklabels([f"Ch {idx}" for idx in viz_indices], rotation=0)
    ax_rms.set_ylabel("RMS ($\mu$V)")
    ax_rms.grid(axis='y', alpha=0.2)
    ax_rms.set_xlabel("Channel Index")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated refined figure at {save_path}")

if __name__ == "__main__":
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz"
    SAVE_PATH = "docs/figs/pipeline.png"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    generate_pipeline_figure(DATA_PATH, SAVE_PATH)
