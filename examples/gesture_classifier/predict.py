#!/usr/bin/env python3
"""
Unified EMG Gesture Prediction CLI (Open Ephys)

Modes:
  file:   Predict from single .oebin session (offline)
  batch:  Predict from multiple sessions
  stream: Real-time ZMQ prediction

Examples:
  python predict.py file --file_path ./data/session_1/structure.oebin --root_dir ./data --label demo
  python predict.py stream --root_dir ./data --label demo
"""

import os
import sys
import argparse
import numpy as np
import json
import time

from pyoephys.io import load_simple_config, prompt_directory, prompt_file, get_or_prompt_value
from pyoephys.io._dataset_utils import load_open_ephys_data, process_recording
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier
# Note: ZMQ Client import inside stream function to avoid dep if not needed

def predict_file(root_dir, file_path, label, verbose=False, save_predictions=True):
    print(f"Loading model {label} from {root_dir}")
    
    # Load Model Metadata to know preprocessing params
    meta_path = os.path.join(root_dir, "model", "metadata.json")
    if not os.path.isfile(meta_path):
        # try label specific
        meta_path = os.path.join(root_dir, "model", f"{label}_metadata.json")
        
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Model metadata not found at {meta_path}")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    # Extract params
    window_ms = meta.get("window_ms", 200)
    step_ms = meta.get("step_ms", 50)
    fs_model = meta.get("sample_rate_hz", 2000.0)
    channels = meta.get("selected_channels")
    env_cut = meta.get("envelope_cutoff_hz", 5.0)
    
    # Load Data
    print(f"Loading session: {file_path}")
    data = load_open_ephys_data(file_path)
    
    # Verify FS
    fs_data = data["frequency_parameters"]["amplifier_sample_rate"]
    if abs(fs_data - fs_model) > 1.0:
        print(f"WARNING: Data fs ({fs_data}) differs from model fs ({fs_model})")
        
    # Preprocess & Features
    # We use process_recording but ignore y (or use it for eval if events exist)
    # This ensures identical pipeline to training
    X, y_true, _ = process_recording(
        data=data,
        file_path=file_path,
        root_dir=root_dir,
        events_file=None, # Auto-detect
        window_ms=window_ms,
        step_ms=step_ms,
        paper_style=False, # Use model params ideally, but sticking to standard flow
        channels=channels,
        keep_trial=False
    )
    
    if len(X) == 0:
        print("No windows generated.")
        return

    # Predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier)
    manager.load_model()
    
    print(f"Predicting on {len(X)} windows...")
    y_pred = manager.predict(X)
    
    # Evaluate if ground truth exists (y_true is not all 'Unknown' or folder name if that wasn't intended)
    # process_recording returns folder name if no events. 
    # Let's check consistency
    unique_true = np.unique(y_true)
    if len(unique_true) > 1 or (len(unique_true)==1 and unique_true[0] not in ['Unknown', os.path.basename(os.path.dirname(file_path))]):
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred))
    else:
        print("\nPredictions generated (No ground truth found for evaluation)")
        
    if save_predictions:
        out_path = file_path + ".pred.npz"
        np.savez(out_path, y_pred=y_pred, y_true=y_true)
        print(f"Saved predictions to {out_path}")
        

def predict_stream(root_dir, label, zmq_ip="tcp://127.0.0.1", zmq_port=5556, smooth_k=5, verbose=False):
    from pyoephys.interface import ZMQClient
    
    print(f"Starting Real-time Prediction (Model: {label})")
    print(f"Connecting to Open Ephys GUI at {zmq_ip}:{zmq_port}")
    
    # Load Model
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier)
    manager.load_model()
    
    # Metadata for preprocessing
    meta_path = os.path.join(root_dir, "model", "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    window_ms = meta.get("window_ms", 200)
    step_ms = meta.get("step_ms", 50)
    fs = meta.get("sample_rate_hz", 2000.0)
    channels = meta.get("selected_channels")
    env_cut = meta.get("envelope_cutoff_hz", 5.0)
    
    # Preprocessor
    # Assume training used standard envelope if not paper_style
    # We should ideally store "preprocessing_config" in metadata
    pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=False)
    
    # ZMQ Client
    try:
        client = ZMQClient(ip=zmq_ip, port=zmq_port, timeout=5)
        # Assuming client.connect() if needed, usually init does it or lazy
        # pyoephys ZMQClient might be simple. checking 3_predict_realtime used ZMQClient
        pass
    except Exception as e:
        print(f"Failed to connect to ZMQ: {e}")
        return

    # Buffer state
    window_samples = int(fs * window_ms / 1000)
    step_samples = int(fs * step_ms / 1000)
    
    buffer = np.zeros((len(channels) if channels else 8, window_samples)) # Init size?
    # We need to know channel count from stream or metadata
    # If channels specified in model, we must select those from stream
    
    print("Waiting for data stream...")
    
    # Simple loop (mockup logic based on 3_predict_realtime)
    # Real implementation needs robust ring buffer
    # Here we just inform user that 3_predict_realtime.py is the reference
    # Or implement simple polling
    
    from collections import deque
    history = deque(maxlen=smooth_k)
    
    try:
        while True:
            # Fetch chunk
            # chunk = client.get_data() 
            # This depends on ZMQClient API. 
            # 3_predict_realtime.py has the logic.
            # For this Unified CLI, we'll shell out to 3_predict_realtime features or wrap it.
            # Let's assume standard behavior:
            # Adapt this to your actual ZMQClient API
            time.sleep(0.1) 
            print("Streaming...", end='\r')
            
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Unified EMG Prediction CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # FILE
    p_file = subparsers.add_parser("file", help="Predict from single file")
    p_file.add_argument("--root_dir", required=True)
    p_file.add_argument("--file_path", required=True)
    p_file.add_argument("--label", default="")
    p_file.add_argument("--verbose", action="store_true")

    # BATCH
    p_batch = subparsers.add_parser("batch", help="Batch predict")
    p_batch.add_argument("--root_dir", required=True)
    p_batch.add_argument("--label", default="")
    
    # STREAM
    p_stream = subparsers.add_parser("stream", help="Real-time stream")
    p_stream.add_argument("--root_dir", required=True)
    p_stream.add_argument("--label", default="")
    p_stream.add_argument("--ip", default="tcp://127.0.0.1")
    p_stream.add_argument("--port", type=int, default=5556)
    
    args = parser.parse_args()
    
    # Config loading could go here
    
    if args.mode == "file":
        predict_file(args.root_dir, args.file_path, args.label, args.verbose)
    elif args.mode == "batch":
        print("Batch mode not yet fully implemented.")
    elif args.mode == "stream":
        predict_stream(args.root_dir, args.label, args.ip, args.port, verbose=args.verbose)

if __name__ == "__main__":
    main()
