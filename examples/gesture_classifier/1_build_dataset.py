#!/usr/bin/env python3
"""
Unified Dataset Builder for EMG Gesture Classification (Open Ephys Version)

Optimized script leveraging pyoephys package utilities.
Supports Open Ephys (.oebin) and NPZ formats.

Examples:
    # Single session
    python 1_build_dataset.py --root_dir ./data --file_path ./data/session_1/structure.oebin --overwrite
    
    # Multi-session auto-discovery
    python 1_build_dataset.py --root_dir ./data --multi_file --overwrite
    
    # Paper-style preprocessing (120Hz highpass, RMS)
    python 1_build_dataset.py --root_dir ./data --multi_file --paper_style --overwrite
    
    # From config file
    python 1_build_dataset.py --config_file .gesture_config --overwrite
"""

import os
import sys
import argparse
import logging
from time import time
import numpy as np
from pathlib import Path

from pyoephys.io._config_utils import (
    load_simple_config,
    save_simple_config,
    prompt_directory,
    prompt_file,
    get_or_prompt_value,
    prompt_yes_no,
    prompt_text
)
from pyoephys.io._file_utils import (
    discover_and_group_files
)
from pyoephys.io._dataset_utils import (
    process_recording,
    save_dataset,
    select_channels,
    load_open_ephys_data
)
from pyoephys.io._grid_utils import (
    infer_grid_dimensions,
    apply_grid_permutation,
    parse_orientation_from_filename
)

def build_dataset(
    root_dir: str, 
    file_type: str = "oebin", 
    file_path: str = None,
    file_names: list = None, 
    multi_file: bool = False,
    events_file: str = None, 
    label: str = "", 
    save_path: str = None,
    window_ms: int = 200, 
    step_ms: int = 50, 
    paper_style: bool = False,
    channels: list = None, 
    channel_map: str = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False, 
    orientation: str = "auto",
    orientation_remap: str = "none", 
    ignore_labels: list = None, 
    ignore_case: bool = False,
    keep_trial_label: bool = False, 
    overwrite: bool = False,
    verbose: bool = False,
):
    start_time = time()
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    if not os.path.exists(root_dir):
        raise ValueError(f"Root dir does not exist: {root_dir}")
        
    save_path = save_path or os.path.join(
        root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz"
    )
    
    if os.path.exists(save_path) and not overwrite:
        print(f"[OK] Dataset exists: {save_path}. Use --overwrite to regenerate.")
        return

    # List of (X, y) tuples
    combined_results = []
    
    files_to_process = []
    
    if multi_file:
        # scan for structure.oebin
        logging.info("Scanning for recordings...")
        # Adapt discover to look for .oebin
        groups = discover_and_group_files(root_dir, file_type="oebin")
        # Flatten groups for now (each oebin is a session)
        for g in groups.values():
            files_to_process.extend(g)
    elif file_path:
        files_to_process = [file_path]
    else:
        raise ValueError("Must specify --file_path or --multi_file")
        
    logging.info(f"files to process: {len(files_to_process)}")
    
    meta_accum = {"fs": set()}
    
    for fp in files_to_process:
        logging.info(f"Processing: {fp}")
        try:
            # Load Data
            data = load_open_ephys_data(fp)
            
            # Select channels
            raw_names = data.get("channel_names")
            # Reuse channel selection logic for first file or consistency check?
            # For simplicity, re-select per file (assumes consistent naming)
            ch_indices, ch_names = select_channels(
                raw_names, channels, channel_map, channel_map_file, mapping_non_strict
            )
            
            # Orientation Remap
            if orientation_remap != "none":
                # Need n_rows, n_cols from map or inference
                # If using map
                # infer grid
                rows, cols = infer_grid_dimensions(ch_names)
                if rows and cols:
                     orient = parse_orientation_from_filename(fp) if orientation == "auto" else orientation
                     if orient:
                         ch_indices = apply_grid_permutation(ch_indices, rows, cols, orientation_remap)
                         logging.info(f"   Applied {orientation_remap} (found {orient})")
            
            # Process
            X, y, meta = process_recording(
                data, fp, root_dir, events_file, window_ms, step_ms,
                paper_style, ch_indices, 
                ignore_labels=ignore_labels,
                ignore_case=ignore_case,
                keep_trial=keep_trial_label
            )
            
            if len(X) > 0:
                combined_results.append((X, y))
                meta_accum["fs"].add(meta["fs"])
                logging.info(f"   -> {len(X)} windows")
            else:
                logging.warning("   -> No windows extracted")
                
        except Exception as e:
            logging.error(f"Failed to process {fp}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    if not combined_results:
        logging.error("No valid data collected.")
        return

    # Concatenate
    X_all = np.concatenate([r[0] for r in combined_results])
    y_all = np.concatenate([r[1] for r in combined_results])
    
    final_meta = {
        "fs": list(meta_accum["fs"])[0] if meta_accum["fs"] else 0,
        "selected_channels": ch_indices, # from last file
        "channel_names": ch_names
    }
    
    save_dataset(save_path, X_all, y_all, final_meta, window_ms, step_ms, channel_map, channel_map_file)
    logging.info(f"Completed in {time() - start_time:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir")
    p.add_argument("--file_path")
    p.add_argument("--multi_file", action="store_true")
    p.add_argument("--config_file")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--paper_style", action="store_true")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    
    # Add other args as needed
    args = p.parse_args()
    
    # Load config
    cfg = load_simple_config(Path(__file__).parent / ".gesture_config")
    if args.config_file:
        cfg.update(load_simple_config(args.config_file))
        
    root_dir = args.root_dir or cfg.get("root_dir")
    if not root_dir:
        root_dir = prompt_directory("Select Root Directory")
        
    if not args.file_path and not args.multi_file:
        # Logic to check config or prompt
        if cfg.get("multi_file"):
            args.multi_file = True
        else:
             # simple prompt
             pass

    build_dataset(
        root_dir=root_dir,
        file_path=args.file_path or cfg.get("file_path"),
        multi_file=args.multi_file,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        paper_style=args.paper_style,
        overwrite=args.overwrite
    )
