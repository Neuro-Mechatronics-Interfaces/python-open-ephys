#!/usr/bin/env python3
"""
compare_zmq_oebin_waveforms.py
Grab a short window from Open Ephys GUI via ZMQ, align it to the same segment in the OEBin,
and compare waveforms per channel (training order): correlation, best lag, scale, offset.

Default compares RAW signals (no preprocessing). You can pass --preprocess to apply the same
EMGPreprocessor to both sides before comparison.

Usage:
  python compare_zmq_oebin_waveforms.py \
    --root_dir "G:\\Shared drives\\NML_shared\\DataShare\\HDEMG Human Healthy\\HD-EMG_Cuff\\Jonathan\\2025_07_31" \
    --label sleeve_15ch_ring \
    --collect-seconds 5 \
    --align-from oebin \
    --plot --plot-chan 0 --out-png compare_ch0.png
"""
import argparse, logging, os, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pyoephys.interface import ZMQClient
from pyoephys.io import load_oebin_file, load_metadata_json, lock_params_to_meta, normalize_name
from pyoephys.processing import EMGPreprocessor


def find_global_offset(ref_multi: np.ndarray, query_multi: np.ndarray, max_search_sec: float, fs: float) -> int:
    """
    Find sample offset 'o' so that ref[:, o:o+N] best matches query[:, :N].
    Uses FFT-based NCC across multiple channels (sum of per-channel NCC peaks).
    Returns offset in samples relative to ref start (can be negative if query starts earlier).
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    C, Nq = query_multi.shape
    max_lag = int(round(max_search_sec * fs))
    # Z-score each channel for robustness
    def z(x): return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)

    qz = z(query_multi)
    # Build a reference window a bit larger than query + search margins
    pad = max_lag + Nq
    # Accumulate cross-corr power across channels
    cc_sum = None
    for c in range(C):
        r = ref_multi[c]
        # pad both to next pow2 for speed
        nfft = 1
        while nfft < len(r) + Nq - 1:
            nfft <<= 1
        R = rfft((r - r.mean()))
        Q = rfft((qz[c] - qz[c].mean()), n=nfft)
        cc = irfft(R * np.conj(Q), nfft)  # circular corr
        # Unwrap to linear cross-corr range
        cc = np.concatenate([cc[-(Nq-1):], cc[:len(r)]])
        # keep only valid lags
        if cc_sum is None:
            cc_sum = cc[:len(r)]
        else:
            cc_sum = cc_sum[:len(r)] + cc[:len(r)]
    # Search offset within ±max_lag around rough guess (beginning)
    start = 0
    lo = max(0, start - max_lag)
    hi = min(len(cc_sum)-1, start + max_lag)
    best = int(np.argmax(cc_sum[lo:hi+1])) + lo
    return best  # ref index where query best aligns


def _select_training_channels_by_name(emg: np.ndarray, raw_names: List[str], trained_names: List[str]) -> Tuple[np.ndarray, List[int]]:
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"OEBin missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx


def _map_training_names_to_indices_ZMQ(client: ZMQClient, trained_names: List[str]) -> List[int]:
    name_by_idx: Dict[int, str] = getattr(client, "_name_by_index", {})
    norm_to_idx = {normalize_name(nm): i for i, nm in name_by_idx.items()}
    idx, missing = [], []
    for nm in trained_names:
        nrm = normalize_name(nm)
        if nrm in norm_to_idx:
            idx.append(norm_to_idx[nrm])
        else:
            missing.append(nm)
    if missing:
        logging.warning("ZMQ stream missing some training channels (yet?): %s", missing)
    if not idx:
        raise RuntimeError("No training channels found on the live stream.")
    return idx


def _samples_from_timestamps(t_new: np.ndarray, fs: float) -> np.ndarray:
    if t_new is None or t_new.size == 0:
        return np.array([], dtype=np.int64)
    t_new = np.asarray(t_new).ravel()
    frac = np.mean((t_new - np.floor(t_new)) != 0)
    if frac > 0.1 or np.nanmax(t_new) < 1e6:
        return np.asarray(np.round(t_new * fs), dtype=np.int64)
    return np.asarray(t_new, dtype=np.int64)


def _align_base_index(align_from: str, root_dir: str, fs: float, t_first: Optional[np.ndarray]) -> Tuple[int, int]:
    """Return (base_index_absolute, file_start_index_absolute)."""
    raw_dir = os.path.join(root_dir, "raw")
    data = load_oebin_file(raw_dir, verbose=False)
    t0_file = float(np.asarray(data["t_amplifier"])[0])
    fs_file = float(data["sample_rate"])
    start_index_abs = int(round(t0_file * fs_file))
    if abs(fs - fs_file) > 1e-3:
        logging.warning("Sample rate mismatch: ZMQ fs=%.3f vs OEBin fs=%.3f", fs, fs_file)

    if align_from == "oebin":
        base_idx_abs = start_index_abs
        logging.info("Aligning base to OEBin t0 (%.6f s) => base_index=%d", t0_file, base_idx_abs)
    elif align_from == "timestamps":
        s_idx = _samples_from_timestamps(np.asarray(t_first), fs)
        base_idx_abs = int(s_idx[0]) if s_idx.size else 0
        logging.info("Aligning base to ZMQ timestamps => base_index=%d", base_idx_abs)
    else:
        base_idx_abs = 0
        logging.info("Aligning base to zero => base_index=0")
    return base_idx_abs, start_index_abs


def _best_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 100) -> int:
    """Return lag (samples) maximizing correlation of y vs x (positive => y delayed)."""
    # normalize
    xz = (x - x.mean()) / (x.std() + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    lags = range(-max_lag, max_lag + 1)
    best_lag, best_val = 0, -np.inf
    for L in lags:
        if L < 0:
            v = np.dot(xz[-L:], yz[:len(yz)+L]) / (len(yz)+L)
        elif L > 0:
            v = np.dot(xz[:len(xz)-L], yz[L:]) / (len(yz)-L)
        else:
            v = np.dot(xz, yz) / len(yz)
        if v > best_val:
            best_val, best_lag = v, L
    return best_lag


def main():
    ap = argparse.ArgumentParser("Compare waveforms from ZMQ vs OEBin for the same training channels.")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--zmq", default="tcp://127.0.0.1")
    ap.add_argument("--data-port", type=int, default=5556)
    ap.add_argument("--heartbeat-port", type=int, default=5557)
    ap.add_argument("--collect-seconds", type=float, default=5.0, help="Amount of ZMQ data to collect.")
    ap.add_argument("--align-from", choices=["oebin", "timestamps", "zero"], default="oebin")
    ap.add_argument("--preprocess", action="store_true", help="Apply EMGPreprocessor to BOTH before comparing.")
    ap.add_argument("--envelope-cutoff-fallback", type=float, default=5.0, help="If metadata lacks cutoff.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-chan", type=int, default=0)
    ap.add_argument("--out-png", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Training metadata
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta, None, None, selected_channels=None)
    if env_cut is None:
        env_cut = args.envelope_cutoff_fallback

    # ZMQ client
    client = ZMQClient(
        zqm_ip=args.zmq,
        http_ip="127.0.0.1",
        data_port=args.data_port,
        heartbeat_port=args.heartbeat_port,
        window_secs=max(1.0, args.collect_seconds * 1.2),
        channels=None,
        auto_start=True,
        verbose=args.verbose,
        expected_channel_names=trained_names,
        expected_channel_count=None,
        require_complete=False,
        required_fraction=1.0,
        max_channels=512,
    )
    if not client.wait_for_channels(timeout_sec=10.0):
        logging.warning("Proceeding without waiting for all channels.")
    zmq_idx = _map_training_names_to_indices_ZMQ(client, trained_names)
    client.set_channel_index(zmq_idx)

    fs = float(client.fs)
    C = len(zmq_idx)
    needed = int(round(args.collect_seconds * fs))
    logging.info("ZMQ fs≈%.1f Hz; collecting %d samples (~%.2f s) for %d channels.", fs, needed, args.collect_seconds, C)

    # Collect ZMQ samples (timestamps + data)
    t0 = None
    buf = np.zeros((C, needed), dtype=np.float32)
    filled = 0
    while filled < needed:
        t_new, y_new = client.drain_new()
        if y_new is None or y_new.size == 0:
            time.sleep(0.002)
            continue
        if t0 is None:
            t0 = np.asarray(t_new).ravel()
        k = min(needed - filled, y_new.shape[1])
        buf[:, filled:filled+k] = y_new[:, :k]
        filled += k
    client.stop(); client.close()
    logging.info("Collected ZMQ buffer: %s", buf.shape)

    # Load OEBin and select training channels
    raw_dir = os.path.join(args.root_dir, "raw")
    d = load_oebin_file(raw_dir, verbose=False)
    fs_file = float(d["sample_rate"])
    emg = np.asarray(d["amplifier_data"])  # (C_all, N)
    raw_names = list(d.get("channel_names", []))
    emg_sel, oebin_idx = _select_training_channels_by_name(emg, raw_names, trained_names)
    start_index_abs = int(round(np.asarray(d["t_amplifier"])[0] * fs_file))
    if abs(fs_file - fs) > 1e-3:
        logging.warning("Sample rate mismatch: GUI fs=%.3f vs OEBin fs=%.3f", fs, fs_file)

    # Align indices
    base_idx_abs, file_start_abs = _align_base_index(args.align_from, args.root_dir, fs, t0)
    # absolute range of our ZMQ buffer
    abs_start = base_idx_abs
    abs_end   = base_idx_abs + buf.shape[1]
    # Convert abs -> OEBin array indices
    j0 = abs_start - file_start_abs
    j1 = abs_end   - file_start_abs
    j0c, j1c = max(0, j0), min(emg_sel.shape[1], j1)
    # Adjust ZMQ buffer segment to match trimmed OEBin slice if needed
    off0 = j0c - j0
    off1 = (j1 - j1c)
    if off0 > 0 or off1 > 0:
        logging.warning("Clipping to OEBin range (shift=%d, tail_clip=%d)", off0, off1)
        buf = buf[:, off0:buf.shape[1]-off1] if off1 > 0 else buf[:, off0:]
    ref = emg_sel[:, j0c:j1c]
    N = min(buf.shape[1], ref.shape[1])
    buf, ref = buf[:, :N], ref[:, :N]
    logging.info("Aligned compare length: %d samples", N)

    # Optional identical preprocessing on BOTH sides
    if args.preprocess:
        pre = EMGPreprocessor(fs=fs, envelope_cutoff=float(env_cut), verbose=args.verbose)
        buf = pre.preprocess(buf)
        ref = pre.preprocess(ref)

    # Compute metrics
    corrs, lags, slopes, offsets = [], [], [], []
    for ch in range(C):
        x = ref[ch].astype(np.float64)
        y = buf[ch].astype(np.float64)
        # scale/offset fit y ≈ a + b*x
        A = np.vstack([x, np.ones_like(x)]).T
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]  # slope, offset
        slopes.append(b); offsets.append(a)
        # correlation (zero-lag)
        xc = (x - x.mean()); yc = (y - y.mean())
        denom = (np.sqrt((xc**2).sum()) * np.sqrt((yc**2).sum()) + 1e-12)
        r0 = float((xc * yc).sum() / denom)
        corrs.append(r0)
        # best lag
        lag = _best_lag(x, y, max_lag=int(round(0.05 * fs)))  # search ±50 ms
        lags.append(int(lag))

    corrs = np.asarray(corrs); lags = np.asarray(lags)
    slopes = np.asarray(slopes); offsets = np.asarray(offsets)

    # Summary
    def q(a): return np.percentile(a, [5, 25, 50, 75, 95])
    logging.info("Corr r (zero-lag): min/med/max = %.3f / %.3f / %.3f", corrs.min(), np.median(corrs), corrs.max())
    logging.info("Lag* (best in ±50ms): unique modes (samples) -> %s", {int(v): int(np.sum(lags==v)) for v in np.unique(lags)})
    logging.info("Slope (y ≈ a + b*x): median=%.3f (5%%=%.3f, 95%%=%.3f)", float(np.median(slopes)), *q(slopes)[[0,4]])
    logging.info("Offset a: median=%.3f (5%%=%.3f, 95%%=%.3f)", float(np.median(offsets)), *q(offsets)[[0,4]])

    # Top suspects
    bad = np.where((np.abs(corrs) < 0.5) | (np.abs(slopes) < 0.5) | (np.abs(slopes) > 2.0))[0]
    if bad.size:
        logging.warning("Channels with poor match: %s", bad.tolist())
    else:
        logging.info("All channels look reasonably matched.")

    # Optional plot
    if args.plot:
        ch = int(np.clip(args.plot_chan, 0, C-1))
        t = np.arange(N)/fs
        plt.figure(figsize=(10,4))
        plt.plot(t, ref[ch], label=f"OEBin ch{ch}")
        plt.plot(t, buf[ch], alpha=0.7, label=f"ZMQ ch{ch}")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
        plt.title(f"Channel {ch} overlay | r0={corrs[ch]:.3f}, lag*={lags[ch]} samp, slope={slopes[ch]:.3f}, off={offsets[ch]:.3f}")
        plt.legend()
        plt.tight_layout()
        if args.out_png:
            plt.savefig(args.out_png, dpi=150)
            print(f"[saved] {args.out_png}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
