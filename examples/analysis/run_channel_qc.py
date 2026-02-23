"""
Channel Quality Control Example
--------------------------------
Demonstrates two usage patterns of ChannelQC + QCParams:

  1. Realtime (streaming) — update() on each incoming chunk, evaluate() each window.
  2. Batch (offline)      — slide over a full recording; flag channels bad if
                            >50 % of evaluations mark them bad.  Flat / zero-crossing
                            checks are disabled because quiet rest-period windows are
                            healthy, not stuck.

API summary
-----------
    qc = ChannelQC(fs, n_channels, window_sec, params=QCParams(...))
    qc.update(chunk)   # chunk: (samples, n_channels), any number of rows
    out = qc.evaluate()
    # out['bad']      – np.bool_[n_channels]  single-eval verdict
    # out['excluded'] – set(int)              hysteresis-stabilised bad set
    # out['metrics']  – dict with 'rms', 'robust_z', 'pl_ratio', etc.
"""

import numpy as np
from pyoephys.processing import ChannelQC, QCParams


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_test_recording(n_channels: int = 16, duration_sec: float = 5.0, fs: float = 2000.0):
    """Return (samples, n_channels) array with a few planted bad channels."""
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)

    # Good EMG: band-limited noise, ~30 µV RMS
    data = rng.normal(0, 30, (n, n_channels)).astype(np.float32)

    # Ch 1 – railed / saturated
    data[:, 1] = 4500.0

    # Ch 2 – heavy 60 Hz powerline
    data[:, 2] += 800 * np.sin(2 * np.pi * 60 * t)

    # Ch 3 – dead electrode (~0.05 µV RMS)
    data[:, 3] = rng.normal(0, 0.05, n)

    return data, fs


# ---------------------------------------------------------------------------
# 1. Realtime pattern
# ---------------------------------------------------------------------------

def demo_realtime(data: np.ndarray, fs: float):
    """Process data chunk-by-chunk as it would arrive from hardware."""
    n_samples, n_ch = data.shape
    chunk_ms = 200
    chunk_size = int(fs * chunk_ms / 1000)

    params = QCParams(
        robust_z_bad=3.0,
        pl_ratio_thresh=0.30,
        flat_std_min=1.0,   # enabled in realtime — sustained flat = stuck ADC
        zc_min_hz=3.0,
    )
    qc = ChannelQC(fs=int(fs), n_channels=n_ch, window_sec=chunk_ms / 1000, params=params)

    print("\n=== Realtime pattern ===")
    for start in range(0, n_samples - chunk_size, chunk_size):
        chunk = data[start : start + chunk_size]
        qc.update(chunk)
        out = qc.evaluate()

    # Final stabilised verdict
    excluded = out["excluded"]
    m = out["metrics"]
    print(f"  Excluded channels (hysteresis): {sorted(excluded)}")
    print(f"  Median RMS: {m['median_rms']:.1f} µV")
    for ch in sorted(excluded):
        print(f"    ch {ch:3d}  rms={m['rms'][ch]:.1f}  z={m['robust_z'][ch]:.2f}  pl={m['pl_ratio'][ch]:.3f}")


# ---------------------------------------------------------------------------
# 2. Batch / offline pattern
# ---------------------------------------------------------------------------

def demo_batch(data: np.ndarray, fs: float):
    """Score every channel over the full recording; vote across windows."""
    n_samples, n_ch = data.shape
    chunk_size = int(fs * 0.5)  # 500 ms windows

    # Disable robust Z (HD-EMG has a wide biological RMS spread; active muscle
    # channels would be flagged as Z-score outliers at ±3 SD, which is wrong).
    # Disable flatline/ZC checks too — quiet rest-period channels are healthy.
    # Only powerline ratio + absolute dead-channel threshold are kept.
    dead_rms_uv = 0.5
    params = QCParams(
        robust_z_bad=999.0,     # effectively disabled
        robust_z_warn=999.0,
        pl_ratio_thresh=0.30,
        flat_std_min=0.0,       # disabled
        zc_min_hz=0.0,          # disabled
    )
    qc = ChannelQC(fs=int(fs), n_channels=n_ch, window_sec=0.5, params=params)

    bad_vote = np.zeros(n_ch, dtype=int)
    n_evals = 0
    for start in range(0, n_samples - chunk_size, chunk_size):
        qc.update(data[start : start + chunk_size])
        out = qc.evaluate()
        bad_vote += out["bad"].astype(int)
        n_evals += 1

    bad_channels = sorted(i for i in range(n_ch) if bad_vote[i] > n_evals * 0.5)

    # Also catch truly dead electrodes via absolute RMS threshold
    dead = [i for i in range(n_ch) if out["metrics"]["rms"][i] < dead_rms_uv]
    bad_channels = sorted(set(bad_channels) | set(dead))
    good_channels = sorted(set(range(n_ch)) - set(bad_channels))
    m = out["metrics"]

    print("\n=== Batch / offline pattern ===")
    print(f"  {len(good_channels)} good, {len(bad_channels)} bad channels")
    print(f"  Median RMS: {m['median_rms']:.1f} µV")
    print(f"  Bad channels: {bad_channels}")
    for ch in bad_channels:
        print(f"    ch {ch:3d}  rms={m['rms'][ch]:.1f}  z={m['robust_z'][ch]:.2f}  pl={m['pl_ratio'][ch]:.3f}")
    return good_channels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data, fs = make_test_recording()
    print(f"Data shape: {data.shape}  (samples × channels),  fs={fs:.0f} Hz")
    print("Planted bad channels: 1 (railed), 2 (60 Hz), 3 (dead)")

    demo_realtime(data, fs)
    good = demo_batch(data, fs)
    print(f"\nChannels safe for downstream use: {good}")

