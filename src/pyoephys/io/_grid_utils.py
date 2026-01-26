"""
pyoephys.io._grid_utils

Utilities for working with high-density EMG electrode arrays and grids.
"""

import os
import re
from typing import List, Tuple, Optional
import numpy as np


def infer_grid_dimensions(channel_names) -> Tuple[Optional[int], Optional[int]]:
    if not channel_names:
        return None, None
    
    if isinstance(channel_names, str):
        match = re.search(r'(\d+)[-x](\d+)', channel_names)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    n = len(channel_names)
    sqrt_n = int(np.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return sqrt_n, sqrt_n
    
    common_grids = [
        (16, 8), (8, 16),
        (8, 4), (4, 8),
        (16, 4), (4, 16),
        (12, 8), (8, 12),
    ]
    
    for rows, cols in common_grids:
        if rows * cols == n:
            return rows, cols
    
    for rows in range(2, int(np.sqrt(n)) + 1):
        if n % rows == 0:
            cols = n // rows
            if cols >= 2:
                return rows, cols
    
    return None, None


def apply_grid_permutation(indices: List[int], n_rows: int, n_cols: int, mode: str) -> List[int]:
    mode = mode.lower().strip()
    
    if mode in ("none", "n", ""):
        return list(indices)
    
    n = len(indices)
    expected = n_rows * n_cols
    if n != expected:
        raise ValueError(f"Index count ({n}) doesn't match grid size ({n_rows}x{n_cols}={expected})")
    
    grid = np.array(indices).reshape(n_rows, n_cols)
    
    if mode in ("rot90", "r"):
        grid = np.rot90(grid, k=-1)
    elif mode == "rot180":
        grid = np.rot90(grid, k=2)
    elif mode in ("rot270", "ccw90"):
        grid = np.rot90(grid, k=1)
    elif mode in ("fliph", "h", "fliplr"):
        grid = np.fliplr(grid)
    elif mode in ("flipv", "v", "flipud"):
        grid = np.flipud(grid)
    elif mode in ("transpose", "t"):
        grid = grid.T
    else:
        raise ValueError(f"Unknown permutation mode: {mode}")
    
    return grid.flatten().tolist()


def parse_orientation_from_filename(path: str) -> Optional[str]:
    name = os.path.splitext(os.path.basename(str(path)))[0].upper()
    
    patterns = [
        (r'CCW\s*90', 'CCW90'),
        (r'CW\s*90', 'CW90'),
        (r'ROT\s*90', 'ROT90'),
        (r'ROT\s*180', 'ROT180'),
        (r'ROT\s*270', 'ROT270'),
        (r'FLIP\s*H', 'FLIPH'),
        (r'FLIP\s*V', 'FLIPV'),
        (r'FLIPPED', 'FLIPPED'),
    ]
    
    for pattern, tag in patterns:
        if re.search(pattern, name):
            return tag
    
    match = re.search(r'[_-]([RLNTV])(?:[_-]|$)', name)
    if match:
        return match.group(1)
    
    return None


def orientation_to_permutation_mode(orientation: str) -> str:
    if not orientation:
        return "none"
    
    orientation = orientation.upper().strip()
    
    mapping = {
        "CCW90": "rot270", "CW90": "rot90", "ROT90": "rot90",
        "ROT180": "rot180", "ROT270": "rot270",
        "FLIPH": "flipH", "FLIPV": "flipV", "FLIPPED": "rot180",
        "R": "rot90", "L": "rot270", "N": "none",
        "T": "transpose", "V": "flipV", "H": "flipH",
    }
    
    return mapping.get(orientation, "none")


def remap_grid_channels(data: np.ndarray, n_rows: int, n_cols: int, orientation: str) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, samples), got shape {data.shape}")
    
    n_channels = data.shape[0]
    expected = n_rows * n_cols
    
    if n_channels != expected:
        raise ValueError(f"Channel count ({n_channels}) doesn't match grid ({n_rows}x{n_cols})")
    
    mode = orientation_to_permutation_mode(orientation)
    
    if mode == "none":
        return data
    
    original_indices = list(range(n_channels))
    new_indices = apply_grid_permutation(original_indices, n_rows, n_cols, mode)
    
    return data[new_indices, :]
