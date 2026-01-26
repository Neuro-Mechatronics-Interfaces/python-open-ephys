"""
UDP Landmark Logger
-------------------
Listens for hand landmarks broadcasted by the Stereo Hand Tracker (port 5005)
and saves them to a structured .npz file for synchronization with EMG data.

Usage:
    python udp_landmark_logger.py --output landmarks.npz --duration 30
"""

import socket
import json
import time
import argparse
import numpy as np
import signal

def main():
    parser = argparse.ArgumentParser(description="Log UDP landmarks to NPZ")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default: 5005)")
    parser.add_argument("--output", type=str, default="landmarks.npz", help="Output filename")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening on port {args.port}...")
    print(f"Recording to {args.output} for {args.duration} seconds...")
    print("Press Ctrl+C to stop early.")

    timestamps = []
    frames = []
    # Store landmarks as list of (N_hands, 21, 3) arrays
    # Since N_hands can vary, we might just store flat lists and post-process
    landmark_history = [] 

    start_time = time.time()
    packet_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break
            
            try:
                data, addr = sock.recvfrom(65535)
                packet = json.loads(data.decode("utf-8"))
                
                # Packet format: {'timestamp': float, 'hands': [{'hand_index': i, 'landmarks': [[x,y,z],...]}, ...], ...}
                
                # Use local receive time for strict sync, or packet timestamp if reliable?
                # Usually best to trust packet timestamp if source clock is good, or mix.
                # Here we'll use packet timestamp if available, else local.
                ts = packet.get("timestamp", time.time())
                
                hands_list = packet.get("hands", [])
                
                # Sort hands by index to maximize consistency
                hands_list.sort(key=lambda h: h.get("hand_index", 0))
                
                # Extract landmarks: (N_hands, 21, 3)
                current_frame_landmarks = []
                for h in hands_list:
                    lm = np.array(h["landmarks"]) # (21, 3)
                    current_frame_landmarks.append(lm)
                
                if current_frame_landmarks:
                    # For simple storage, lets just store the first hand, or all if consistent?
                    # To be robust, let's keep it flexible: store list of objects
                    landmark_history.append(current_frame_landmarks)
                    timestamps.append(ts)
                    frames.append(packet.get("frame", packet_count))
                    packet_count += 1
                    
                if packet_count % 100 == 0:
                    print(f"\rCaptured {packet_count} frames ({elapsed:.1f}s)", end="")
                    
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")

    print(f"\nSaving {len(timestamps)} frames to {args.output}")
    
    # Post-process: Pad to max hands to make a dense array? 
    # Or just pickle the list. NPZ supports object arrays.
    # Let's try to make a dense array (T, MaxHands, 21, 3)
    if not landmark_history:
        print("No data captured.")
        return

    max_hands = max(len(h) for h in landmark_history)
    T = len(timestamps)
    dense_landmarks = np.zeros((T, max_hands, 21, 3), dtype=np.float32)
    
    for t, hands in enumerate(landmark_history):
        for h_i, hand_lm in enumerate(hands):
            if h_i < max_hands:
                dense_landmarks[t, h_i] = hand_lm

    np.savez_compressed(
        args.output,
        timestamps=np.array(timestamps),
        frames=np.array(frames),
        landmarks=dense_landmarks, # (T, Hands, 21, 3)
        fs_estimated=T/args.duration
    )
    print("Done.")

if __name__ == "__main__":
    main()
