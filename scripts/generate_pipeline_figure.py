import numpy as np
import matplotlib.pyplot as plt
import os

def generate_pipeline_figure(save_path):
    """
    Generates a professional 4-panel figure for the JOSS paper:
    1. Raw signal (Waterfall)
    2. Filtered signal
    3. QC Status
    4. RMS Barplot
    """
    fs = 1000
    t = np.arange(fs) / fs
    n_channels = 3
    
    # Generate Synthetic Data
    # Channel 0: High signal, Channel 1: Noisy (failed QC), Channel 2: Normal
    np.random.seed(42)
    raw = np.zeros((n_channels, len(t)))
    raw[0] = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 0.5
    raw[1] = 2 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 10.0 # Noisy
    raw[2] = 3 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 0.5
    
    # Filtered (simulated bandpass)
    filtered = np.zeros_like(raw)
    filtered[0] = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 0.1
    filtered[1] = 2 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 0.1
    filtered[2] = 3 * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t)) * 0.1
    
    # QC Status
    qc_pass = [True, False, True] # Channel 1 fails
    
    # RMS
    rms = np.sqrt(np.mean(filtered**2, axis=1))
    
    # Create Figure
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 2, 1, 2]})
    plt.subplots_adjust(hspace=0.4)
    
    # 1. Raw Waterfall
    for i in range(n_channels):
        axes[0].plot(t, raw[i] + i * 25, color='black', alpha=0.7)
    axes[0].set_title("A) Raw High-Density Signals", loc='left', fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Amplitude (Offset)")
    axes[0].set_yticks([])
    axes[0].grid(True, alpha=0.3)
    
    # 2. Filtered
    for i in range(n_channels):
        axes[1].plot(t, filtered[i] + i * 10, label=f"Ch {i}")
    axes[1].set_title("B) Filtered Waveforms (Bandpass + Notch)", loc='left', fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Amplitude (Offset)")
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3)
    
    # 3. QC Status
    colors = ['green', 'red', 'green']
    labels = ['Pass', 'Fail', 'Pass']
    for i in range(n_channels):
        axes[2].barh(i, 1, color=colors[i], alpha=0.6, height=0.6)
        axes[2].text(0.5, i, labels[i], ha='center', va='center', color='white', fontweight='bold')
    axes[2].set_title("C) Automated Channel Quality Assessment (QC)", loc='left', fontsize=14, fontweight='bold')
    axes[2].set_yticks(range(n_channels))
    axes[2].set_yticklabels([f"Ch {i}" for i in range(n_channels)])
    axes[2].set_xticks([])
    axes[2].set_xlim(0, 1)
    
    # 4. RMS Barplot
    axes[3].bar(range(n_channels), rms, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    axes[3].set_title("D) Extracted RMS Features", loc='left', fontsize=14, fontweight='bold')
    axes[3].set_xticks(range(n_channels))
    axes[3].set_xticklabels([f"Ch {i}" for i in range(n_channels)])
    axes[3].set_ylabel("RMS Amplitude")
    axes[3].grid(axis='y', alpha=0.3)
    
    # Common X Axis
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[3].set_xlabel("Channel Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    output_dir = "docs/figs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generate_pipeline_figure(os.path.join(output_dir, "pipeline.png"))
