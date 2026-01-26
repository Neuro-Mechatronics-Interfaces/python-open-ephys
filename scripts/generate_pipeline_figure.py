import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyoephys.processing import bandpass_filter, notch_filter, calculate_rms, ChannelQC, common_average_reference

def generate_pipeline_figure(data_path, save_path):
    """
    Generates a professional 4-panel figure for the JOSS paper using REAL data:
    1. Raw signal (Waterfall)
    2. Filtered signal
    3. QC Status
    4. RMS Barplot
    """
    # Load Real Data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    raw_full = data['amplifier_data']
    t_full = data['t_amplifier']
    # fs = float(data['sample_rate'])
    fs = 2000 # Typical for this dataset, or extract if scalar
    # Cherry-picked indices: 51 (pass), 0 (fail), 113 (pass)
    viz_indices = [51, 0, 113]
    
    if raw_full.shape[1] > 10000:
        # Take a 1-second segment for visualization
        start_idx = int(fs * 2.5) # Pick a middle segment
        end_idx = start_idx + int(fs * 1.0)
        raw = raw_full[viz_indices, start_idx:end_idx]
        t = t_full[start_idx:end_idx]
        t = t - t[0] # Zero relative time
    else:
        raw = raw_full[viz_indices, :]
        t = t_full - t_full[0]

    # 1. Processing Pipeline
    # Apply CAR first for better quality visualization
    car_data = common_average_reference(raw_full[:, start_idx:end_idx])
    
    # Filter
    filtered_full_seg = notch_filter(car_data, fs=fs, f0=60)
    filtered_full_seg = bandpass_filter(filtered_full_seg, lowcut=20, highcut=500, fs=fs)
    
    processed_viz = filtered_full_seg[viz_indices, :]

    # 2. QC Status (Run on full segment or whole array)
    qc = ChannelQC(fs=fs, n_channels=raw_full.shape[0])
    # The current evaluate logic depends on buffering via update()
    # Or we can just use the compute_metrics logic if available
    # Actually ChannelQC.evaluate() works on the buffers. Let's update with a chunk.
    qc.update(raw_full[:, start_idx:end_idx].T) # Transpose to (samples, channels)
    qc_results = qc.evaluate()
    
    # Get actual results for the cherry-picked indices
    qc_status = [not qc_results['bad'][i] for i in viz_indices]
    
    # 3. RMS Calculation
    rms = calculate_rms(processed_viz, window_size=int(0.1 * fs)) # 100ms windows
    rms_avg = np.mean(rms, axis=1)

    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), gridspec_kw={'height_ratios': [2, 2, 1, 2]})
    plt.subplots_adjust(hspace=0.45)
    
    # colors for panels
    colors_main = ['#3498db', '#e67e22', '#2ecc71']
    
    # Scale parameters for visual consistency
    raw_offset = 500  # Offset between channels in raw plot
    raw_ylim = (-200, 1200) # Suitable range for 3 channels @ 500 offset
    
    filt_offset = 200 # Offset between channels in filtered plot
    filt_ylim = (-150, 550) # Suitable range for 3 channels @ 200 offset

    # A. Raw Waterfall
    for i in range(3):
        # Center signal around zero before adding offset
        sig = raw[i] - np.mean(raw[i])
        axes[0].plot(t, sig + i * raw_offset, color='black', alpha=0.7, linewidth=0.8)
    axes[0].set_title("A) Raw High-Density EMG Signals", loc='left', fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Amplitude ($\mu$V)")
    axes[0].set_ylim(raw_ylim)
    axes[0].set_yticks([0, raw_offset, 2*raw_offset])
    axes[0].set_yticklabels([f"Ch {viz_indices[0]}", f"Ch {viz_indices[1]}", f"Ch {viz_indices[2]}"])
    axes[0].grid(True, alpha=0.2)
    
    # B. Filtered Signal
    for i in range(3):
        sig = processed_viz[i] - np.mean(processed_viz[i])
        axes[1].plot(t, sig + i * filt_offset, color=colors_main[i], linewidth=1.0)
    axes[1].set_title("B) Preprocessed Waveforms (Bandpass, Notch, CAR)", loc='left', fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Amplitude ($\mu$V)")
    axes[1].set_ylim(filt_ylim)
    axes[1].set_yticks([0, filt_offset, 2*filt_offset])
    axes[1].set_yticklabels([f"Ch {viz_indices[0]}", f"Ch {viz_indices[1]}", f"Ch {viz_indices[2]}"])
    axes[1].grid(True, alpha=0.2)
    
    # C. QC Status
    qc_colors = ['green' if s else 'red' for s in qc_status]
    qc_labels = ['Pass' if s else 'Fail' for s in qc_status]
    for i in range(3):
        axes[2].barh(i, 1, color=qc_colors[i], alpha=0.6, height=0.6)
        axes[2].text(0.5, i, f"{qc_labels[i]} (Ch {viz_indices[i]})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    axes[2].set_title("C) Automated Channel Quality Monitoring", loc='left', fontsize=14, fontweight='bold')
    axes[2].set_yticks(range(3))
    axes[2].set_yticklabels([f"Ch {viz_indices[i]}" for i in range(3)])
    axes[2].set_xticks([])
    axes[2].set_xlim(0, 1)

    # D. RMS Barplot
    axes[3].bar(range(3), rms_avg, color=qc_colors, alpha=0.8)
    axes[3].set_title("D) Extracted RMS Activation Features", loc='left', fontsize=14, fontweight='bold')
    axes[3].set_xticks(range(3))
    axes[3].set_xticklabels([f"Ch {viz_indices[i]}" for i in range(3)])
    axes[3].set_ylabel("RMS Amplitude ($\mu$V)")
    axes[3].grid(axis='y', alpha=0.3)
    
    # Labels
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[3].set_xlabel("Channel Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated new figure at {save_path}")

if __name__ == "__main__":
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz"
    SAVE_PATH = "docs/figs/pipeline.png"
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    generate_pipeline_figure(DATA_PATH, SAVE_PATH)
