"""
Channel Quality Control Example
-------------------------------
Demonstrates how to use the ChannelQC class to identify bad channels 
based on line noise, impedance (if available), and signal artifacts.
"""

import numpy as np
from pyoephys.processing import ChannelQC

def generate_bad_data(n_channels=8, n_samples=10000):
    """
    Generates data where:
    - Ch 0 is good
    - Ch 1 is railed (saturated)
    - Ch 2 has high 60Hz noise
    - Ch 3 is disconnected (approx zero)
    """
    t = np.linspace(0, 5, n_samples)
    data = np.random.randn(n_channels, n_samples) * 10 # Good baseline (10 uV)
    
    # Ch 1: Railed
    data[1, :] = 5000.0 
    
    # Ch 2: 60Hz Noise
    data[2, :] += 500 * np.sin(2 * np.pi * 60 * t)
    
    # Ch 3: Dead/Disconnected
    data[3, :] = np.random.randn(n_samples) * 0.1
    
    return data

def main():
    fs = 2000.0
    print("Generating sample data with 8 channels...")
    data = generate_bad_data()
    
    print("Initializing Channel QC...")
    qc = ChannelQC(fs=fs)
    
    print("Running QC analysis...")
    # Analyze a window
    results = qc.compute_qc(data, window_idx=0)
    
    print("\n--- QC Results ---")
    print(f"Total Channels: {len(results)}")
    
    for ch, metrics in results.items():
        status = "BAD" if not metrics['status'] else "GOOD"
        print(f"Channel {ch}: {status}")
        if not metrics['status']:
            print(f"  Issues: {metrics.get('reasons', [])}")
            
    # Visualize if requested (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        print("\nPlotting results...")
        qc.plot_qc_summary(data)
        plt.show()
    except ImportError:
        print("\nMatplotlib not found, skipping plot.")

if __name__ == "__main__":
    main()
