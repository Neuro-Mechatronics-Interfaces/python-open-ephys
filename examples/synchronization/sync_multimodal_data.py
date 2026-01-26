"""
Multimodal Synchronization Example
----------------------------------
Aligns Open Ephys EMG data with Hand Tracking landmarks.

This script demonstrates:
1. Loading EMG data from an Open Ephys session (or mock data).
2. Loading Landmark data from a .npz file (captured via udp_landmark_logger.py).
3. Computing the movement signal from 3D hand landmarks.
4. Computing the EMG envelope.
5. Finding the temporal offset to align the two streams.

Usage:
    python sync_multimodal_data.py --emg <path_to_session> --landmarks landmarks.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyoephys.processing import (
    compute_landmark_movement_signal,
    compute_emg_envelope_signal,
    find_sync_offset
)
from pyoephys.io import load_open_ephys_session

def create_mock_data(duration=30.0, emg_fs=2000, landmark_fs=30.0, offset_sec=2.5):
    """Generate aligned mock data if no files provided."""
    print("Generating mock data...")
    t_emg = np.arange(0, duration, 1/emg_fs)
    t_lm = np.arange(0, duration, 1/landmark_fs)
    
    # Common activity profile (e.g., 3 bursts)
    activity = np.zeros_like(t_emg)
    burst_times = [5.0, 15.0, 25.0]
    for bt in burst_times:
        # gaussian burst
        activity += np.exp(-0.5 * ((t_emg - bt - offset_sec) / 0.5)**2)
    
    # EMG = noise modulated by activity
    emg_data = np.random.randn(8, len(t_emg)) * (1 + 10 * activity)
    
    # Landmarks = velocity matches activity
    # We integrate activity to get position, so derivative (velocity) matches activity
    # Here we just fake the landmarks array directly
    n_frames = len(t_lm)
    landmarks = np.zeros((n_frames, 21, 3))
    
    # Add movement at burst times (unshifted time for landmarks, shifted for EMG)
    # Effectively EMG is delayed by offset_sec relative to ground truth event, or vice versa
    # Let's say Event happens at t. Landmarks see it at t. EMG sees it at t + offset.
    
    # Re-do: 
    # Event times: 5, 15, 25
    # Landing on landmarks at: 5, 15, 25
    # Landing on EMG at: 5+offset, 15+offset, 25+offset
    
    # Landmark movement
    for i, t in enumerate(t_lm):
        dist = np.min(np.abs(t - np.array(burst_times)))
        if dist < 1.0:
            # Move index finger
            landmarks[i, 8, 1] = np.sin(20 * t) * 0.1
            
    return emg_data, t_emg, landmarks, t_lm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg", type=str, help="Path to Open Ephys Session folder")
    parser.add_argument("--landmarks", type=str, help="Path to landmarks.npz")
    args = parser.parse_args()

    if args.emg and args.landmarks:
        # Load Real Data
        print(f"Loading EMG from {args.emg}...")
        session = load_open_ephys_session(args.emg)
        emg_data = session['amplifier_data']
        emg_fs = session['sample_rate']
        t_emg = np.arange(emg_data.shape[1]) / emg_fs
        
        print(f"Loading Landmarks from {args.landmarks}...")
        lm_data = np.load(args.landmarks)
        # (T, Hands, 21, 3) -> take first hand
        landmarks = lm_data['landmarks'][:, 0, :, :] 
        t_lm = lm_data['timestamps']
        # zero-align timestamps for relative processing if needed, 
        # usually we use "system time" for both, so we keep absolute.
        
    else:
        # Use Mock Data
        emg_data, t_emg, landmarks, t_lm = create_mock_data()
        emg_fs = 1.0 / (t_emg[1] - t_emg[0])

    # 1. Process Landmarks -> Movement Signal
    print("Computing landmark movement signal...")
    # landmarks shape: (frames, 21, 3)
    lm_signal, t_lm_clean = compute_landmark_movement_signal(landmarks, t_lm)

    # 2. Process EMG -> Envelope
    print("Computing EMG envelope...")
    emg_env, t_emg_env = compute_emg_envelope_signal(emg_data, emg_fs)

    # 3. Find Sync
    print("Calculating synchronization offset...")
    # This finds lag such that: emg(t) ~ lm(t + offset)
    # Positive offset => Landmarks are DELAYED relative to EMG? 
    # Check docstring: "Positive offset means landmarks are DELAYED relative to EMG."
    sync_result = find_sync_offset(emg_env, t_emg_env, lm_signal, t_lm_clean)
    
    offset = sync_result['offset_sec']
    conf = sync_result['confidence']
    print(f"\nFound Offset: {offset:.4f} seconds")
    print(f"Confidence: {conf:.2f}")

    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Normalize for plotting
    def norm(x): return (x - np.mean(x)) / np.std(x)
    
    plt.subplot(2, 1, 1)
    plt.title("Original Signals")
    plt.plot(t_emg_env, norm(emg_env), label="EMG Envelope", alpha=0.7)
    plt.plot(t_lm_clean, norm(lm_signal), label="Landmark Movement", alpha=0.7)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title(f"Aligned (Landmarks shifted by {-offset:.2f}s)")
    plt.plot(t_emg_env, norm(emg_env), label="EMG Envelope", alpha=0.7)
    plt.plot(t_lm_clean - offset, norm(lm_signal), label="Aligned Landmarks", alpha=0.7)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
